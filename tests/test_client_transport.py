"""协议分片解析，覆盖混合增量、工具索引、用量及畸形字段。"""

from copy import deepcopy

import pytest

from lhagent.client.config import load_config
from lhagent.client.errors import ProtocolError, classify_error
from lhagent.client.retry import should_retry
from lhagent.client.transport import decode_chunk


def chunk(delta=None, finish_reason=None, **metadata):
    """构造单 choice 的协议分片，可注入结束原因及额外元数据。"""
    return {
        "choices": [
            {"index": 0, "delta": {} if delta is None else delta, "finish_reason": finish_reason}
        ],
        **metadata,
    }


def test_mixed_chunk_preserves_all_deltas_and_metadata():
    """混合分片保留全部内容增量与元数据。"""
    raw = chunk(
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "thought",
            "tool_calls": [
                {
                    "index": 3,
                    "id": "call-a",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":'},
                },
                {"index": 0, "id": "call-b", "function": {"name": "ls", "arguments": "{}"}},
                {"index": 3, "function": {"arguments": '"a"}'}},
            ],
        },
        "tool_calls",
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
    )
    original = deepcopy(raw)
    result = decode_chunk(raw)
    assert result == {
        "deltas": [
            {"type": "text", "tool_index": None, "data": {"text": "answer"}},
            {"type": "reasoning", "tool_index": None, "data": {"text": "thought"}},
            {
                "type": "tool_call",
                "tool_index": 3,
                "data": {"call_id": "call-a", "name": "read", "arguments_json": '{"path":'},
            },
            {
                "type": "tool_call",
                "tool_index": 0,
                "data": {"call_id": "call-b", "name": "ls", "arguments_json": "{}"},
            },
            {"type": "tool_call", "tool_index": 3, "data": {"arguments_json": '"a"}'}},
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
            "cache_read_tokens": 2,
            "cache_write_tokens": None,
        },
        "finish_reason": "tool_calls",
    }
    assert raw == original
    result["deltas"][0]["data"]["text"] = "changed"
    assert raw == original


@pytest.mark.parametrize(
    "delta",
    [{}, {"role": "assistant"}, {"content": None, "reasoning_content": None, "tool_calls": []}],
)
def test_empty_delta_does_not_signal_completion(delta):
    """空增量不意味着响应已经结束。"""
    assert decode_chunk(chunk(delta)) == {"deltas": [], "usage": None, "finish_reason": None}


def test_empty_strings_and_index_only_tool_are_preserved():
    """空字符串与只有索引的工具分片不会被丢弃。"""
    assert decode_chunk(
        chunk(
            {
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {"index": 0},
                    {"index": 1, "function": {"arguments": ""}},
                ],
            }
        )
    )["deltas"] == [
        {"type": "text", "tool_index": None, "data": {"text": ""}},
        {"type": "reasoning", "tool_index": None, "data": {"text": ""}},
        {"type": "tool_call", "tool_index": 0, "data": {}},
        {"type": "tool_call", "tool_index": 1, "data": {"arguments_json": ""}},
    ]


def test_usage_only_and_no_cross_chunk_state():
    """纯用量分片有效，解析器不积累跨分片状态。"""
    first = decode_chunk(chunk({"content": "hello"}, "stop", usage={"prompt_tokens": 7}))
    last = decode_chunk({"choices": [], "usage": {"completion_tokens": 0}})
    assert first["finish_reason"] == "stop"
    assert last == {
        "deltas": [],
        "finish_reason": None,
        "usage": {
            "input_tokens": None,
            "output_tokens": 0,
            "total_tokens": None,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
        },
    }
    assert decode_chunk(chunk())["usage"] is None
    assert decode_chunk(chunk())["finish_reason"] is None


@pytest.mark.parametrize(
    "reason", ["stop", "length", "tool_calls", "content_filter", "future_reason"]
)
def test_finish_reason_is_preserved_without_public_mapping(reason):
    """传输保留原始结束原因，公共映射由客户端完成。"""
    assert decode_chunk(chunk(finish_reason=reason))["finish_reason"] == reason


@pytest.mark.parametrize(
    "usage",
    [
        {},
        {"prompt_tokens": None, "prompt_tokens_details": None},
        {"prompt_tokens_details": {"cached_tokens": None}},
    ],
)
def test_unknown_usage_is_not_estimated(usage):
    """缺失用量不使用本地估算补齐。"""
    assert set(decode_chunk({"choices": [], "usage": usage})["usage"].values()) == {None}


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        "[DONE]",
        {},
        {"choices": None},
        {"choices": {}},
        {"choices": [None]},
        {"choices": [{"delta": {}}]},
        {"choices": [{"index": 1, "delta": {}}]},
        {"choices": [{"index": True, "delta": {}}]},
        {"choices": [{"index": 0}]},
        {"choices": [{"index": 0, "delta": {}}, {"index": 1, "delta": {}}]},
        {"choices": [], "error": {"message": "secret"}},
        chunk("bad"),
        chunk({"content": []}),
        chunk({"reasoning_content": 1}),
        chunk({"role": "user"}),
        chunk({"function_call": {}}),
        chunk({"refusal": "refused"}),
        chunk({"audio": {}}),
        chunk(finish_reason=4),
        chunk(finish_reason=""),
        chunk({"tool_calls": {}}),
        chunk({"tool_calls": [None]}),
        *[
            chunk({"tool_calls": [tool]})
            for tool in [
                {"function": {"arguments": "{}"}},
                {"index": None},
                {"index": -1},
                {"index": True},
                {"index": 1.0},
                {"index": "0"},
                {"index": 0, "type": "custom"},
                {"index": 0, "id": 1},
                {"index": 0, "id": ""},
                {"index": 0, "function": []},
                {"index": 0, "function": {"name": ""}},
                {"index": 0, "function": {"name": 1}},
                {"index": 0, "function": {"arguments": {}}},
            ]
        ],
        *[
            {"choices": [], "usage": usage}
            for usage in [
                [],
                {"prompt_tokens": -1},
                {"completion_tokens": True},
                {"total_tokens": 1.0},
                {"total_tokens": "3"},
                {"prompt_tokens_details": []},
                {"prompt_tokens_details": {"cached_tokens": -1}},
            ]
        ],
    ],
)
def test_malformed_protocol_is_classified_and_not_retried(raw):
    """畸形协议统一分类为不可重试错误。"""
    with pytest.raises(ProtocolError) as caught:
        decode_chunk(raw)
    assert classify_error(caught.value) == "protocol"
    config = load_config({"base_url": "https://example.com/v1", "api_key": "key"})
    assert not should_retry(caught.value, 1, config)
    assert "secret" not in str(caught.value)


def test_protocol_error_classification_does_not_depend_on_message():
    """协议异常类型决定分类，不依赖诊断文本。"""
    assert classify_error(ProtocolError("rate limit: maximum context length")) == "protocol"


def test_unrelated_metadata_is_ignored():
    """无关元数据不影响受支持字段的解析。"""
    assert decode_chunk(
        chunk(
            id="response-id",
            model="model",
            created=123,
            object="chat.completion.chunk",
            extra="metadata",
        )
    ) == {
        "deltas": [],
        "usage": None,
        "finish_reason": None,
    }
