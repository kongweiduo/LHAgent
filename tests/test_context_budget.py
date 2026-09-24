"""完整请求预算估算和精确阈值，包含工具定义、修复消息及非法预算。"""

from copy import deepcopy

import pytest

from lhagent.harness.context.assembly import assemble_context
from lhagent.harness.context.budget import (
    estimate_context_tokens,
    estimate_message_tokens,
    estimate_tool_tokens,
    should_compact,
    validate_budget,
)


def text(role, value):
    """构造用于精确预算计算的单文本消息。"""
    return {"role": role, "content": [{"type": "text", "text": value}]}


def test_empty_context_and_exact_threshold():
    """空上下文和精确阈值遵守严格超过才压缩的规则。"""
    budget = {"context_window": 100, "extra_input_tokens": 0}
    settings = {"enabled": True, "reserve_tokens": 20, "keep_recent_tokens": 0}
    assert estimate_context_tokens([], budget) == 0
    assert should_compact(80, budget, settings) is False
    assert should_compact(81, budget, settings) is True
    assert should_compact(100, budget, {**settings, "enabled": False}) is False


def test_complete_view_counts_prompts_summary_instruction_and_repairs():
    """预算覆盖提示词、摘要、新指令及重放修复内容。"""
    call = {
        "type": "tool_call",
        "complete": True,
        "call_id": "c1",
        "name": "read",
        "arguments": {"path": "x"},
    }
    history = [
        text("user", "older"),
        {"role": "assistant", "content": [call], "finish_reason": "tool_call"},
    ]
    data = {
        "prompts": {"system_prompt": "system", "additional_messages": [text("system", "fixed")]},
        "history": {
            "summary": "summary",
            "messages": history,
            "entry_ids": ["record-1", "record-2"],
        },
        "new_instruction": text("user", "new"),
    }
    snapshot = deepcopy(data)
    messages = assemble_context(data)
    assert len(messages) == 7  # includes summary and missing tool result placeholder
    budget = {"context_window": 10000, "extra_input_tokens": 0}
    expected = sum(estimate_message_tokens(message) for message in messages)
    assert estimate_context_tokens(messages, budget) == expected
    assert estimate_context_tokens(messages, budget) == expected
    assert estimate_message_tokens(messages[-2]) > 0
    assert estimate_context_tokens(messages[:-2], budget) < expected
    assert data == snapshot
    assert "record-1" not in repr(messages)
    assert estimate_message_tokens(
        {**messages[0], "entry_id": "a" * 1000}
    ) == estimate_message_tokens(messages[0])


def test_tool_definitions_separate_and_structured_content():
    """工具定义单独计费估算，结构化内容也计入预算。"""
    definitions = [
        {
            "name": "read",
            "description": "read file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]
    overhead = estimate_tool_tokens(definitions)
    assert overhead > 0
    assert estimate_tool_tokens([]) == 0
    messages = [text("user", "go")]
    budget = {"context_window": 1000, "extra_input_tokens": overhead}
    assert (
        estimate_context_tokens(messages, budget) == estimate_message_tokens(messages[0]) + overhead
    )
    call = {
        "type": "tool_call",
        "complete": True,
        "call_id": "id",
        "name": "read",
        "arguments_json": '{"path":"x"}',
        "arguments": {"ignored": "y" * 1000},
    }
    assistant = {"role": "assistant", "content": [{"type": "reasoning", "text": "thinking"}, call]}
    assert estimate_message_tokens(assistant) == estimate_message_tokens(
        {
            "role": "assistant",
            "content": [
                assistant["content"][0],
                {k: v for k, v in call.items() if k != "arguments"},
            ],
        }
    )
    result = {
        "role": "tool",
        "tool_call_id": "id",
        "content": [{"type": "tool_result", "is_error": False, "content": {"lines": ["hello"]}}],
    }
    assert estimate_message_tokens(result) > 0
    assert estimate_message_tokens(
        {**result, "content": [{**result["content"][0], "content": {"lines": ["hello" * 100]}}]}
    ) > estimate_message_tokens(result)


@pytest.mark.parametrize(
    "field,value,exception",
    [
        ("context_window", 0, ValueError),
        ("context_window", True, TypeError),
        ("extra_input_tokens", -1, ValueError),
        ("extra_input_tokens", 1.2, TypeError),
    ],
)
def test_invalid_budget(field, value, exception):
    """非法上下文预算在计算前拒绝。"""
    with pytest.raises(exception):
        validate_budget(
            {"context_window": 100, "extra_input_tokens": 0, field: value},
            {"enabled": True, "reserve_tokens": 10, "keep_recent_tokens": 0},
        )


@pytest.mark.parametrize(
    "field,value,exception",
    [
        ("enabled", 1, TypeError),
        ("reserve_tokens", 0, ValueError),
        ("reserve_tokens", 100, ValueError),
        ("reserve_tokens", False, TypeError),
        ("keep_recent_tokens", -1, ValueError),
        ("keep_recent_tokens", 1.1, TypeError),
    ],
)
def test_invalid_settings(field, value, exception):
    """非法压缩设置不能进入阈值判断。"""
    with pytest.raises(exception):
        should_compact(
            0,
            {"context_window": 100, "extra_input_tokens": 0},
            {"enabled": True, "reserve_tokens": 10, "keep_recent_tokens": 0, field: value},
        )


def test_negative_context_and_invalid_structured_content():
    """负输入估算和非法结构化内容必须报错。"""
    with pytest.raises(ValueError):
        should_compact(
            -1,
            {"context_window": 100, "extra_input_tokens": 0},
            {"enabled": False, "reserve_tokens": 10, "keep_recent_tokens": 0},
        )
    with pytest.raises(ValueError):
        estimate_message_tokens({"role": "user", "content": [{"type": "image", "data": "x"}]})
    with pytest.raises(TypeError):
        estimate_tool_tokens([{"name": "x", "parameters": {"bad": object()}}])
