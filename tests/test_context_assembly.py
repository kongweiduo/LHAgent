"""提示词读取、历史重放修复与请求副本组装，验证身份关联及原始历史不变。"""

import asyncio
import json
from copy import deepcopy

import pytest

from lhagent.client.transport import _message_body
from lhagent.harness.context.assembly import (
    assemble_context,
    build_history_messages,
    normalize_history_messages,
)
from lhagent.harness.context.sources import load_prompts, read_prompt_file
from lhagent.harness.session.jsonl import JsonlSessionStore
from lhagent.harness.session.session import Session
from tests.samples import mixed_result, user_message


def text(role="user", value="继续"):
    """构造指定角色的单文本消息。"""
    return {"role": role, "content": [{"type": "text", "text": value}]}


def call(call_id="t1", **updates):
    """构造完整工具调用，可覆盖字段模拟残缺协议。"""
    return {
        "type": "tool_call",
        "call_id": call_id,
        "name": "read",
        "complete": True,
        "arguments": {"path": "文件.txt"},
        **updates,
    }


def assistant(*blocks, reason="tool_call"):
    """将给定块封装成带终态的模型消息。"""
    return {
        "role": "assistant",
        "content": list(blocks),
        "finish_reason": reason,
        "call_id": "model-1",
    }


def tool(call_id="t1", **updates):
    """构造带调用关联的结构化工具结果消息。"""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": "read",
        "content": [{"type": "tool_result", "content": {"lines": ["原文"]}, "is_error": False}],
        **updates,
    }


def history(messages, summary=None):
    """为消息按原始位置分配独立条目 ID。"""
    return {
        "summary": summary,
        "messages": messages,
        "entry_ids": [f"entry-{i}" for i in range(len(messages))],
    }


def context(messages=(), instruction=None):
    """构造无固定提示词的请求素材与可选新指令。"""
    return {
        "prompts": {"system_prompt": None, "additional_messages": []},
        "history": history(list(messages)),
        "new_instruction": instruction,
    }


def assert_replay(messages):
    """验证修复不改源数据、可重复执行且结果符合发送协议。"""
    snapshot = deepcopy(messages)
    repaired = normalize_history_messages(messages)
    assert messages == snapshot
    assert normalize_history_messages(repaired) == repaired
    # 修复结果必须通过已有传输层的协议映射，无需网络。
    for message in repaired:
        _message_body(message)
    return repaired


def test_prompt_sources_utf8_order_roles_and_explicit_paths(tmp_path, monkeypatch):
    """提示词仅从显式 UTF-8 路径读取，保持顺序及角色。"""
    monkeypatch.chdir(tmp_path)
    for name, value in [("system.txt", "系统\n"), ("b.txt", "乙"), ("a.txt", "甲")]:
        (tmp_path / name).write_text(value, encoding="utf-8")
    # 不应发现或打开未指定文件。
    (tmp_path / "AGENTS.md").write_bytes(b"\xff")
    assert load_prompts(
        {"system_prompt_path": "system.txt", "additional_prompt_paths": ["b.txt", "a.txt"]}
    ) == {
        "system_prompt": "系统\n",
        "additional_messages": [text("system", "乙"), text("system", "甲")],
    }
    assert load_prompts({"system_prompt_path": None, "additional_prompt_paths": []}) == {
        "system_prompt": None,
        "additional_messages": [],
    }
    (tmp_path / "empty").write_text("", encoding="utf-8")
    assert read_prompt_file(str(tmp_path / "empty")) == ""
    with pytest.raises(FileNotFoundError):
        read_prompt_file("missing.txt")
    with pytest.raises(IsADirectoryError):
        read_prompt_file(str(tmp_path))
    with pytest.raises(UnicodeDecodeError):
        read_prompt_file("AGENTS.md")
    with pytest.raises(FileNotFoundError):
        load_prompts({"system_prompt_path": None, "additional_prompt_paths": ["a.txt", "missing"]})


@pytest.mark.parametrize("reason", ["error", "cancelled"])
def test_failed_response_and_associated_results_removed(reason):
    """失败模型响应及其关联工具结果不进入重放历史。"""
    messages = [text(), assistant(call(), reason=reason), tool(), text("assistant", "恢复")]
    assert assert_replay(messages) == [messages[0], messages[-1]]
    assert assert_replay([assistant(call(complete=False), reason=reason)]) == []


@pytest.mark.parametrize(
    "bad",
    [
        {"complete": False},
        {"call_id": ""},
        {"name": ""},
        {"arguments": None},
        {"arguments_json": '{"x":'},
        {"arguments_json": "[]"},
        {"arguments_json": 1},
        {"arguments": {"x": float("inf")}},
        {"arguments": {"x": object()}},
    ],
)
def test_length_drops_only_unreplayable_calls(bad):
    """截断响应只移除不可重放调用，保留可用内容。"""
    broken = call("broken")
    broken.update(bad)
    message = assistant(
        {"type": "text", "text": "进展"},
        {"type": "reasoning", "text": "分析"},
        broken,
        call("valid", arguments_json='{"path":"ok"}'),
        reason="length",
    )
    messages = [message]
    if broken["call_id"]:
        messages.append(tool(broken["call_id"]))
    messages.append(tool("valid"))
    # object/NaN 等非法输入不适用相等快照断言；依然检查内容块未被原地删除。
    repaired = normalize_history_messages(messages)
    assert len(message["content"]) == 4
    assert len(repaired[0]["content"]) == 3
    assert repaired[1] == tool("valid")
    assert repaired[0]["finish_reason"] == "length"
    assert_replay(repaired)
    with pytest.raises(ValueError, match="invalid tool call"):
        normalize_history_messages([assistant(broken)])


def test_truncated_only_response_and_orphan_results_disappear():
    """仅有残缺调用的响应及孤立结果一并过滤。"""
    assert (
        assert_replay([tool("orphan"), assistant(call(complete=False), reason="length"), tool()])
        == []
    )
    missing_id = call()
    del missing_id["call_id"]
    assert assert_replay([assistant(missing_id, reason="length")]) == []


@pytest.mark.parametrize("boundary", [None, text(), text("assistant", "下一步")])
def test_multiple_calls_partial_results_placeholder_order_and_ownership(boundary):
    """部分缺失结果按工具顺序补占位，且不改写源历史。"""
    messages = [assistant(call("a"), call("b"), call("c")), tool("b")]
    if boundary is not None:
        messages.append(boundary)
    repaired = assert_replay(messages)
    assert [m["tool_call_id"] for m in repaired[1:4]] == ["b", "a", "c"]
    for message in repaired[2:4]:
        assert message["content"][0]["is_error"] is True
        assert "是否执行及副作用未知" in message["content"][0]["content"]
    if boundary is not None:
        assert repaired[-1] == boundary
    repaired[0]["content"][0]["arguments"]["path"] = "changed"
    repaired[1]["content"][0]["content"]["lines"].clear()
    assert messages[0]["content"][0]["arguments"]["path"] == "文件.txt"
    assert messages[1]["content"][0]["content"]["lines"] == ["原文"]


def test_length_keeps_committed_not_executed_result_and_metadata():
    """截断时已提交的未执行结果及元数据仍可重放。"""
    result = tool()
    result["content"] = [{"type": "tool_result", "content": "截断响应，未执行", "is_error": True}]
    messages = [assistant(call(), reason="length"), result]
    assert assert_replay(messages) == messages


@pytest.mark.parametrize(
    "messages,match",
    [
        ([assistant(call(), call())], "duplicate tool call"),
        ([assistant(call()), tool(), assistant(call())], "duplicate tool call"),
        ([assistant(call(), reason="error"), assistant(call())], "duplicate tool call"),
        ([assistant(call()), tool(), tool()], "duplicate tool result"),
        ([tool("orphan"), tool("orphan")], "duplicate tool result"),
        ([assistant(call(), reason="cancelled"), tool(), tool()], "duplicate tool result"),
        ([tool(), assistant(call())], "outside its assistant batch"),
        ([assistant(call()), text(), tool()], "outside its assistant batch"),
        ([assistant(call()), tool(name="write")], "name does not match"),
        ([tool(tool_call_id="")], "requires tool_call_id"),
        ([{"role": "user", "content": [call()]}], "requires assistant role"),
        ([{"role": "assistant", "content": tool()["content"]}], "requires tool role"),
        ([{**text(), "tool_call_id": "t1"}], "requires tool role"),
    ],
)
def test_invalid_associations_raise(messages, match):
    """非法工具身份关联必须显式报错。"""
    with pytest.raises(ValueError, match=match):
        normalize_history_messages(messages)


def test_summary_and_fixed_prompts_order_and_deep_copies():
    """摘要和固定提示词按约定排序，返回内容与源对象隔离。"""
    data = context([text(value="历史")], text(value="新指令"))
    data["prompts"] = {
        "system_prompt": "系统",
        "additional_messages": [text("system", "固定"), text("user", "资料")],
    }
    data["history"]["summary"] = "之前的工作"
    snapshot = deepcopy(data)
    result = assemble_context(data)
    assert [m["role"] for m in result] == ["system", "system", "user", "user", "user", "user"]
    assert result[:3] == [text("system", "系统"), text("system", "固定"), text("user", "资料")]
    assert "历史对话摘要" in result[3]["content"][0]["text"]
    assert "之前的工作" in result[3]["content"][0]["text"]
    assert result[-2:] == [text(value="历史"), text(value="新指令")]
    assert "entry-0" not in json.dumps(result)
    for message in result:
        _message_body(message)
        message["content"].clear()
    assert data == snapshot


@pytest.mark.parametrize("ids", [[], ["a", "a"], ["a", ""], ["a", 1], ["a", "b", "c"]])
def test_history_entry_ids_validate_original_messages(ids):
    """条目 ID 必须对应原始消息序列。"""
    data = history([assistant(call(), reason="error"), tool()])
    data["entry_ids"] = ids
    with pytest.raises(ValueError, match="entry_ids"):
        build_history_messages(data)


def test_history_equal_content_distinct_ids_and_changed_repair_length():
    """相同内容仍可有不同身份，修复长度不改变原身份映射。"""
    data = history([text(), text(), assistant(call())])
    assert len(build_history_messages(data)) == 4
    assert data["entry_ids"] == ["entry-0", "entry-1", "entry-2"]
    assert build_history_messages(history([], "  ")) == []
    assert assemble_context(context()) == []


def test_new_instruction_deduplicates_identity_and_explicit_record_id_only():
    """新指令只按对象身份或显式记录 ID 去重，不按文本相等去重。"""
    first = text()
    data = context([first], first)
    assert assemble_context(data) == [first]
    data["new_instruction"] = deepcopy(first)
    data["new_instruction_entry_id"] = "entry-0"
    assert assemble_context(data) == [first]
    data["new_instruction_entry_id"] = "new-entry"
    assert assemble_context(data) == [first, first]
    del data["new_instruction_entry_id"]
    assert assemble_context(data) == [first, first]
    data["new_instruction"] = None
    assert assemble_context(data) == [first]


def test_new_instruction_invalid_identity_and_role():
    """非法新指令身份或角色不能进入组装结果。"""
    data = context([text()], text(value="不同"))
    data["new_instruction_entry_id"] = "entry-0"
    with pytest.raises(ValueError, match="does not match"):
        assemble_context(data)
    data["new_instruction"] = None
    with pytest.raises(ValueError, match="requires new_instruction"):
        assemble_context(data)
    data["new_instruction"] = data["history"]["messages"][0]
    data["new_instruction_entry_id"] = "wrong-id"
    with pytest.raises(ValueError, match="identity conflicts"):
        assemble_context(data)
    data["new_instruction_entry_id"] = ""
    with pytest.raises(ValueError, match="nonempty string"):
        assemble_context(data)
    del data["new_instruction_entry_id"]
    data["new_instruction"] = text("assistant")
    with pytest.raises(ValueError, match="user message"):
        assemble_context(data)


def test_repair_never_persists_placeholder_to_session(tmp_path):
    """重放占位只存在于请求副本，不写入会话。"""

    async def scenario():
        path = tmp_path / "session.jsonl"
        meta = {"id": "s1", "created_at": 123.5, "path": str(path)}
        session = Session(meta, await JsonlSessionStore.create(meta))
        try:
            await session.start_run("run-1")
            await session.append_user(user_message())
            await session.append_response(mixed_result())
            original = await session.get_history()
            before = path.read_bytes()
            repaired = build_history_messages(original)
            assert len(repaired) == len(original["messages"]) + 1
            assert repaired[-1]["role"] == "tool"
            assert await session.get_history() == original
            assert path.read_bytes() == before
        finally:
            await session.close()

    asyncio.run(scenario())
