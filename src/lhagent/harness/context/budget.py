"""估算实际模型输入大小，并判断是否达到自动压缩阈值。

本地估算只用于预算决策，不冒充 client 返回的实际 token 用量或计费数据。
仅提供判断，不调用模型、不提交压缩记录，也不安排超限重试。
"""

import json

from .types import CompactionSettings, ContextBudget


def _tokens(value: str) -> int:
    """按 UTF-8 字节数近似估算 token，向上取整，不代表计费用量。"""
    return (len(value.encode("utf-8")) + 3) // 4


def _json(value: object) -> str:
    """生成用于预算估算的紧凑 JSON，拒绝非有限数字。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _integer(value: object, name: str, minimum: int) -> None:
    """校验预算整数及下限，拒绝布尔值。"""
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def validate_budget(budget: ContextBudget, settings: CompactionSettings) -> None:
    """校验窗口、额外输入开销和压缩参数。"""
    _integer(budget["context_window"], "context_window", 1)
    _integer(budget["extra_input_tokens"], "extra_input_tokens", 0)
    if type(settings["enabled"]) is not bool:
        raise TypeError("enabled must be a boolean")
    _integer(settings["reserve_tokens"], "reserve_tokens", 1)
    _integer(settings["keep_recent_tokens"], "keep_recent_tokens", 0)
    if settings["reserve_tokens"] >= budget["context_window"]:
        raise ValueError("reserve_tokens must be less than context_window")


def estimate_message_tokens(message: dict[str, object]) -> int:
    """估算一条可发送消息的输入 token，不包含日志元数据。"""
    role = message["role"]
    if role not in ("system", "user", "assistant", "tool"):
        raise ValueError("unsupported message role")
    # 消息和内容块的封装开销是本地估算，不代表协议实际计费用量。
    total = 4 + _tokens(role)
    if role == "tool":
        call_id = message["tool_call_id"]
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("tool message requires tool_call_id")
        total += _tokens(call_id) + 2
    if "name" in message:
        total += _tokens(message["name"]) + 1
    for block in message["content"]:
        kind = block["type"]
        total += 2
        if kind == "text" or (kind == "reasoning" and role == "assistant"):
            if not isinstance(block["text"], str):
                raise TypeError("content text must be a string")
            total += _tokens(block["text"])
        elif kind == "tool_call" and role == "assistant":
            if block.get("complete") is not True:
                raise ValueError("tool call must be complete before estimation")
            if any(not isinstance(block.get(k), str) or not block[k] for k in ("call_id", "name")):
                raise ValueError("tool call requires call_id and name")
            arguments = block.get("arguments_json")
            if arguments is None:
                arguments = _json(block["arguments"])
            if not isinstance(arguments, str) or not isinstance(json.loads(arguments), dict):
                raise ValueError("tool arguments must encode a JSON object")
            total += 4 + _tokens(block["call_id"]) + _tokens(block["name"]) + _tokens(arguments)
        elif kind == "tool_result" and role == "tool":
            value = block["content"]
            total += _tokens(value if isinstance(value, str) else _json(value))
        else:
            raise ValueError("content block is unsupported for message role")
    return total


def estimate_context_tokens(
    messages: list[dict[str, object]],
    budget: ContextBudget,
) -> int:
    """估算已组装并修复的完整请求及消息之外的输入开销。"""
    _integer(budget["context_window"], "context_window", 1)
    _integer(budget["extra_input_tokens"], "extra_input_tokens", 0)
    return (
        sum(estimate_message_tokens(message) for message in messages) + budget["extra_input_tokens"]
    )


def estimate_tool_tokens(definitions: list[dict[str, object]]) -> int:
    """估算 describe_tools 的模型可见定义，供 extra_input_tokens 使用。"""
    # 工具定义在消息之外，需另计每个函数的封装开销。
    return sum(4 + _tokens(_json(definition)) for definition in definitions)


def should_compact(
    context_tokens: int,
    budget: ContextBudget,
    settings: CompactionSettings,
) -> bool:
    """仅在启用且输入严格超过窗口减预留空间时触发。"""
    validate_budget(budget, settings)
    _integer(context_tokens, "context_tokens", 0)
    return (
        settings["enabled"]
        and context_tokens > budget["context_window"] - settings["reserve_tokens"]
    )
