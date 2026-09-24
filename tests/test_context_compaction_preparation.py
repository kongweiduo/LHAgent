"""压缩切点、工具批次原子性、原始条目身份和摘要素材截断。"""

import json
from copy import deepcopy

import pytest

from lhagent.harness.context.assembly import assemble_context
from lhagent.harness.context.budget import estimate_context_tokens, estimate_message_tokens
from lhagent.harness.context.compaction import find_cut_point, prepare_compaction, serialize_history


def text(role, value, **fields):
    """构造带可选元数据的文本消息。"""
    return {"role": role, "content": [{"type": "text", "text": value}], **fields}


def history(messages, summary=None):
    """为待压缩消息附加顺序身份和可选摘要。"""
    return {
        "messages": messages,
        "entry_ids": [f"entry-{i}" for i in range(len(messages))],
        "summary": summary,
    }


def prepare(messages, keep, summary=None):
    """准备压缩并验证输入不变、原预算按完整请求估算。"""
    data = {
        "prompts": {"system_prompt": "fixed", "additional_messages": []},
        "history": history(messages, summary),
        "new_instruction": text("user", "not committed"),
    }
    snapshot = deepcopy(data)
    budget = {"context_window": 100000, "extra_input_tokens": 17}
    result = prepare_compaction(
        data, budget, {"enabled": True, "reserve_tokens": 200, "keep_recent_tokens": keep}
    )
    assert data == snapshot
    if result:
        assert result["tokens_before"] == estimate_context_tokens(assemble_context(data), budget)
    return result


def batch():
    """构造两个工具调用及大型结果，用于验证批次不可拆分。"""
    assistant = {
        "role": "assistant",
        "finish_reason": "tool_call",
        "content": [
            {
                "type": "tool_call",
                "call_id": call_id,
                "name": "read",
                "complete": True,
                "arguments": {"path": call_id},
            }
            for call_id in ("a", "b")
        ],
    }
    results = [
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": "read",
            "content": [{"type": "tool_result", "content": "x" * 20000, "is_error": False}],
        }
        for call_id in ("a", "b")
    ]
    return [assistant, *results]


def test_zero_budget_and_empty_or_unsummarizable_history():
    """零保留预算及不可总结历史有明确切点和空结果行为。"""
    messages = [text("user", "first"), *batch()]
    assert find_cut_point(history(messages), 0) == len(messages)
    result = prepare(messages, 0, "old summary")
    assert result["retained_messages"] == []
    assert result["retained_entry_ids"] == []
    assert result["messages_to_summarize"] == messages
    assert result["previous_summary"] == "old summary"
    assert prepare([], 0, "old summary") is None
    assert prepare([text("assistant", "failed", finish_reason="error")], 0) is None
    assert prepare([text("user", "only")], 10000) is None
    with pytest.raises(ValueError):
        find_cut_point(history(messages), -1)
    with pytest.raises(TypeError):
        find_cut_point(history(messages), True)


def test_atomic_multi_result_batch_and_split_turn():
    """多工具结果批次不可拆开，轮次前缀单独提取。"""
    messages = [
        text("user", "earlier"),
        text("assistant", "done"),
        text("user", "current"),
        text("assistant", "progress"),
        *batch(),
    ]
    # Last result alone is much larger than this target; the entire batch stays.
    assert find_cut_point(history(messages), 1) == 4
    result = prepare(messages, 1, "prior")
    assert result["messages_to_summarize"] == messages[:2]
    assert result["turn_prefix_messages"] == messages[2:4]
    assert result["is_split_turn"] is True
    assert result["retained_messages"] == messages[4:]
    assert result["retained_entry_ids"] == ["entry-4", "entry-5", "entry-6"]
    assert result["retained_messages"] is not messages
    # A larger goal can reach the user boundary without splitting the turn.
    target = sum(estimate_message_tokens(m) for m in messages[3:]) + 1
    assert find_cut_point(history(messages), target) == 2
    whole = prepare(messages, target)
    assert whole["is_split_turn"] is False
    assert whole["turn_prefix_messages"] == []


def test_replay_repairs_and_original_tail_identity():
    """重放修复不改变保留尾部的原始条目身份。"""
    call = {
        "role": "assistant",
        "finish_reason": "tool_call",
        "content": [
            {
                "type": "tool_call",
                "call_id": "missing",
                "name": "bash",
                "complete": True,
                "arguments": {"command": "pwd"},
            }
        ],
    }
    messages = [
        text("user", "old"),
        text("assistant", "failed", finish_reason="cancelled"),
        text("user", "current"),
        call,
    ]
    result = prepare(messages, 1)
    assert result["retained_messages"] == [call]
    assert result["retained_entry_ids"] == ["entry-3"]
    assert result["turn_prefix_messages"] == messages[2:3]
    assert result["messages_to_summarize"] == messages[:2]
    assert "failed" not in serialize_history(result["messages_to_summarize"])
    material = [json.loads(line) for line in serialize_history([call]).splitlines()]
    assert [item["role"] for item in material] == ["assistant", "tool"]
    assert material[1]["tool_call_id"] == "missing"
    assert material[1]["content"][0]["result_unknown"] is True
    assert "未知" in material[1]["content"][0]["content"]
    assert call == messages[-1]


def test_summary_material_preserves_links_errors_and_large_output():
    """摘要素材保留调用关联、错误及大输出截断说明。"""
    messages = [text("user", "instruction"), *batch()]
    material = [json.loads(line) for line in serialize_history(messages).splitlines()]
    assert material[1]["content"][0]["call_id"] == "a"
    assert material[2]["tool_call_id"] == "a"
    assert material[2]["content"][0]["truncated_from_characters"] == 20000
    assert len(material[2]["content"][0]["content"]) == 16000
    assert "result_unknown" not in material[2]["content"][0]
    assert serialize_history([]) == ""
    with pytest.raises(ValueError):
        serialize_history([{"role": "user", "content": [{"type": "image", "data": "x"}]}])
    with pytest.raises(ValueError):
        find_cut_point(history([*batch()[1:], batch()[0]]), 1)


def test_multiple_batches_and_structured_error_result():
    """多个工具批次和结构化错误结果可正确序列化。"""
    first = batch()
    second = deepcopy(batch())
    for block in second[0]["content"]:
        block["call_id"] += "2"
    for result in second[1:]:
        result["tool_call_id"] += "2"
    second[1]["content"][0].update(content={"exit_code": 1, "stderr": "failed"}, is_error=True)
    messages = [text("user", "start"), *first, text("assistant", "next"), *second]
    assert find_cut_point(history(messages), 1) == 5
    material = [json.loads(line) for line in serialize_history(second).splitlines()]
    assert material[1]["content"][0]["is_error"] is True
    assert json.loads(material[1]["content"][0]["content"]) == {"exit_code": 1, "stderr": "failed"}
    assert "result_unknown" not in material[1]["content"][0]
