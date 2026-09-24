"""消息、模型结果、日志条目及事件的公共数据契约和身份隔离。"""

from lhagent.client.types import client_result_to_message
from lhagent.harness.loop.types import LoopEvent

from .samples import FakeStream, mixed_result, user_record


def test_mixed_result_preserves_content_kinds_and_call_id() -> None:
    """混合响应保留内容类别及模型/工具调用身份。"""
    result = mixed_result(complete_tool=False)

    assert [block["type"] for block in result["content"]] == [
        "text",
        "reasoning",
        "tool_call",
    ]
    tool_call = result["content"][2]
    assert tool_call["call_id"] == "tool-call-1"
    assert tool_call["complete"] is False


def test_client_result_projection_is_an_assistant_message() -> None:
    """结果投影生成 assistant 消息，剔除调用统计等非消息字段。"""
    result = mixed_result()

    message = client_result_to_message(result)

    assert message["role"] == "assistant"
    assert message["call_id"] == "model-call-1"
    assert message["finish_reason"] == "tool_call"
    assert message["content"] == result["content"]
    assert "stats" not in message
    assert "error" not in message


def test_jsonl_message_record_has_stable_entry_identity() -> None:
    """日志条目身份独立于运行身份且保持稳定。"""
    record = user_record()

    assert record["type"] == "user"
    assert record["id"] == "entry-1"
    assert record["message"]["role"] == "user"
    assert record["run_id"] != record["id"]


def test_fake_stream_is_async_and_can_be_closed() -> None:
    """假事件流支持异步消费与显式关闭。"""
    stream = FakeStream([{"type": "delta", "value": "ok"}])

    async def consume() -> list[dict[str, object]]:
        events = [event async for event in stream]
        await stream.aclose()
        return events

    import asyncio

    assert asyncio.run(consume()) == [{"type": "delta", "value": "ok"}]
    assert stream.closed is True


def test_loop_event_separates_run_call_tool_and_entry_ids() -> None:
    """循环事件中的运行身份和模型调用身份不可混用。"""
    event: LoopEvent = {
        "type": "response_update",
        "data": {
            "run_id": "run-1",
            "phase": "delta",
            "call_id": "model-call-1",
            "block_index": 0,
            "delta": {"type": "text", "text": "hello"},
        },
    }

    assert event["data"]["run_id"] == "run-1"
    assert event["data"]["call_id"] == "model-call-1"
