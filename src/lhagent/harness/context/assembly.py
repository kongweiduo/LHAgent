"""组装请求副本；历史修复不改写会话、不执行工具、不生成摘要。"""

import json
from copy import deepcopy

from .types import ContextInput, HistoryContext


def _nonempty_string(value: object) -> bool:
    """判断身份字段是否为非空字符串。"""
    return isinstance(value, str) and bool(value.strip())


def _replayable_call(block: dict[str, object]) -> bool:
    """判断工具调用是否具备可重放的完整参数和身份。"""
    if block.get("complete") is not True or not all(
        _nonempty_string(block.get(key)) for key in ("call_id", "name")
    ):
        return False
    raw = block.get("arguments_json")
    try:
        if raw is not None:
            if not isinstance(raw, str):
                return False
            arguments = json.loads(raw)
        else:
            arguments = block.get("arguments")
        if not isinstance(arguments, dict):
            return False
        json.dumps(arguments, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _text_message(role: str, text: str) -> dict[str, object]:
    """构造单文本块消息，不修改原始会话。"""
    return {"role": role, "content": [{"type": "text", "text": text}]}


def normalize_history_messages(
    messages: list[dict[str, object]],
) -> list[dict[str, object]]:
    """生成可重放副本，保留元数据；不修改输入、不执行工具、不落盘。

    排除失败响应及关联结果，移除 length 的残缺调用及其结果和孤立结果。
    对保留调用在下一条非工具消息之前或历史末尾补齐未知结果。只使用真实
    调用 ID，已有结果不重复补齐。重复身份和正常响应非法关联明确报错。
    """
    # 先收集身份：即使结果出现在调用之前，或调用将被过滤，也能检测重复。
    copies = deepcopy(messages)
    calls: dict[str, tuple[dict[str, object], bool]] = {}
    result_ids: set[str] = set()
    for message in copies:
        role = message["role"]
        failed = role == "assistant" and message.get("finish_reason") in ("error", "cancelled")
        for block in message["content"]:
            if block["type"] != "tool_call":
                if block["type"] == "tool_result" and role != "tool" and not failed:
                    raise ValueError("tool_result requires tool role")
                continue
            if role != "assistant":
                raise ValueError("tool_call requires assistant role")
            valid = _replayable_call(block)
            if not failed and message.get("finish_reason") != "length" and not valid:
                raise ValueError("normal assistant response contains invalid tool call")
            call_id = block.get("call_id")
            if _nonempty_string(call_id):
                if call_id in calls:
                    raise ValueError(f"duplicate tool call ID: {call_id}")
                calls[call_id] = (block, valid and not failed)
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not _nonempty_string(call_id):
                raise ValueError("tool result requires tool_call_id")
            if call_id in result_ids:
                raise ValueError(f"duplicate tool result: {call_id}")
            result_ids.add(call_id)
        elif "tool_call_id" in message and not failed:
            raise ValueError("tool_call_id requires tool role")

    output: list[dict[str, object]] = []
    pending: dict[str, dict[str, object]] = {}

    def flush_pending() -> None:
        for call_id, block in pending.items():
            output.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": block["name"],
                    "content": [
                        {
                            "type": "tool_result",
                            "is_error": True,
                            "content": "没有可用结果，是否执行及副作用未知。",
                        }
                    ],
                }
            )
        pending.clear()

    for message in copies:
        role = message["role"]
        if role == "tool":
            call_id = message["tool_call_id"]
            association = calls.get(call_id)
            if association is None or not association[1]:
                continue
            if call_id not in pending:
                raise ValueError(f"tool result outside its assistant batch: {call_id}")
            block = pending[call_id]
            if "name" in message and message["name"] != block["name"]:
                raise ValueError(f"tool result name does not match call: {call_id}")
            del pending[call_id]
            output.append(message)
            continue

        flush_pending()
        if role == "assistant":
            if message.get("finish_reason") in ("error", "cancelled"):
                continue
            if message.get("finish_reason") == "length":
                message["content"] = [
                    block
                    for block in message["content"]
                    if block["type"] != "tool_call" or _replayable_call(block)
                ]
                if not message["content"]:
                    continue
            for block in message["content"]:
                if block["type"] == "tool_call":
                    pending[block["call_id"]] = block
        output.append(message)
    flush_pending()
    return output


def build_history_messages(history: HistoryContext) -> list[dict[str, object]]:
    """验证原始记录 ID，连接最新摘要背景和修复历史；ID 不进入请求。

    entry_ids 必须与原始 messages 等长且唯一，不能与修复后的下标配对。
    摘要使用 user 角色并标为历史背景；不从日志重新加入被替代的消息。
    """
    messages, entry_ids = history["messages"], history["entry_ids"]
    if len(messages) != len(entry_ids) or any(not _nonempty_string(item) for item in entry_ids):
        raise ValueError("entry_ids must contain one nonempty ID per original message")
    if len(set(entry_ids)) != len(entry_ids):
        raise ValueError("entry_ids must be unique")
    result = normalize_history_messages(messages)
    summary = history["summary"]
    if summary is not None:
        if not isinstance(summary, str):
            raise TypeError("summary must be a string or None")
        if summary.strip():
            result.insert(
                0,
                _text_message(
                    "user",
                    "以下是历史对话摘要，仅作为历史背景，不是新的用户指令：\n\n" + summary,
                ),
            )
    return result


def assemble_context(context_input: ContextInput) -> list[dict[str, object]]:
    """依次组装系统提示词、固定附加消息、摘要、历史和未提交的新指令。

    新指令应在提交后传 None。若仍提供已提交副本，可通过可选记录 ID 去重；
    同一对象也不重复加入，但不按内容相等猜测身份。不写回会话。
    """
    prompts, history = context_input["prompts"], context_input["history"]
    result = []
    if prompts["system_prompt"] is not None:
        result.append(_text_message("system", prompts["system_prompt"]))
    result.extend(deepcopy(prompts["additional_messages"]))
    result.extend(build_history_messages(history))
    instruction = context_input["new_instruction"]
    entry_id = context_input.get("new_instruction_entry_id")
    if "new_instruction_entry_id" in context_input and not _nonempty_string(entry_id):
        raise ValueError("new_instruction_entry_id must be a nonempty string")
    if instruction is None:
        if entry_id is not None:
            raise ValueError("new_instruction_entry_id requires new_instruction")
        return result
    if instruction["role"] != "user":
        raise ValueError("new_instruction must be a user message")
    if entry_id in history["entry_ids"]:
        original = history["messages"][history["entry_ids"].index(entry_id)]
        if original != instruction:
            raise ValueError("new_instruction does not match its history entry")
        return result
    if any(instruction is message for message in history["messages"]):
        if entry_id is not None:
            raise ValueError("new_instruction identity conflicts with its entry ID")
        return result
    result.append(deepcopy(instruction))
    return result
