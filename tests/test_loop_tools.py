"""串行工具批次、不可执行响应、校验错误及未确认停止后的运行限制。"""

import asyncio
from copy import deepcopy

import pytest

from lhagent.harness.loop.loop import AgentLoop
from lhagent.harness.loop.queue import InputQueue
from lhagent.harness.session.jsonl import JsonlSessionStore
from lhagent.harness.session.session import Session
from lhagent.harness.tools.types import ToolStopError
from tests.samples import user_message
from tests.test_loop_basic import make_result


def call(call_id, *, complete=True, name="demo", arguments=None):
    """构造可控制名称、参数及完整性的工具调用块。"""
    return {
        "type": "tool_call",
        "call_id": call_id,
        "name": name,
        "complete": complete,
        "arguments": {} if arguments is None else arguments,
    }


class Client:
    """按脚本交付终结响应，按实际请求身份修正 call_id。"""

    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    async def stream(self, request, *, cancel_event):
        self.requests.append(deepcopy(request))
        result = deepcopy(self.responses.pop(0))
        result["call_id"] = request["call_id"]
        reason = result["finish_reason"]
        yield {
            "call_id": request["call_id"],
            "type": "error"
            if reason == "error"
            else "cancelled"
            if reason == "cancelled"
            else "done",
            "block_index": None,
            "data": {},
            "result": result,
        }


async def setup(tmp_path, responses, handler):
    """组装真实会话与单工具循环；调用者负责关闭会话。"""
    metadata = {"id": "tools", "created_at": 1.0, "path": str(tmp_path / "tools.jsonl")}
    session = Session(metadata, await JsonlSessionStore.create(metadata))
    client = Client(responses)
    events = []

    async def emit(event):
        events.append(event)

    tool = {
        "name": "demo",
        "description": "Demo tool",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
        "handler": handler,
    }
    config = {
        "model": "test",
        "parameters": {},
        "prompts": {"system_prompt": None, "additional_messages": []},
        "budget": {"context_window": 10000, "extra_input_tokens": 0},
        "compaction": {"enabled": False, "reserve_tokens": 10, "keep_recent_tokens": 0},
        "summary_request": None,
        "max_summary_output_tokens": 10,
        "tools": [tool],
        "new_call_id": iter(["request-1", "request-2"]).__next__,
        "new_run_id": lambda: "run-1",
    }
    context = {
        "cwd": str(tmp_path),
        "timeout_seconds": 5,
        "cancel_event": asyncio.Event(),
        "max_output_lines": 100,
        "max_output_bytes": 10000,
    }
    return session, client, events, AgentLoop(client, session, InputQueue(), config, context, emit)


def response(reason, *blocks):
    """用指定内容块替换基础模型结果，保留终态统计字段。"""
    result = make_result(reason)
    result["content"] = list(blocks)
    return result


def test_serial_tools_and_validation_error(tmp_path):
    """工具按顺序执行，校验错误记为结果并继续批次。"""

    async def scenario():
        invoked = []

        async def handler(args, context):
            invoked.append(args["x"])
            return {
                "content": [{"type": "text", "text": "ok"}],
                "details": {},
                "is_error": False,
                "truncated": False,
            }

        session, client, events, loop = await setup(
            tmp_path,
            [
                response(
                    "tool_call", call("a", arguments={"x": "wrong"}), call("b", arguments={"x": 2})
                ),
                response("stop", {"type": "text", "text": "done"}),
            ],
            handler,
        )
        try:
            result = await loop.run(user_message())
            assert result["status"] == "completed"
            assert invoked == [2]
            assert client.requests[0]["parameters"]["tools"] == [
                {
                    "name": "demo",
                    "description": "Demo tool",
                    "parameters": loop._config["tools"][0]["parameters"],
                }
            ]
            messages = client.requests[1]["messages"]
            assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool"]
            assert [m["tool_call_id"] for m in messages[2:]] == ["a", "b"]
            assert messages[2]["content"][0]["is_error"] is True
            assert [e["type"] for e in events].count("tool_start") == 2
            assert [e["type"] for e in events].count("tool_end") == 2
            event_types = [e["type"] for e in events]
            first = event_types.index("tool_start")
            assert event_types[first : first + 6] == [
                "tool_start",
                "message_committed",
                "tool_end",
                "tool_start",
                "message_committed",
                "tool_end",
            ]
            assert (await session.state())["state"] == "idle"
        finally:
            await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("reason", ["length", "error", "cancelled"])
def test_non_executable_calls(tmp_path, reason):
    """截断、失败或取消响应不触发不可执行工具。"""

    async def scenario():
        async def handler(args, context):
            pytest.fail("tool must not execute")

        terminal = response(
            reason,
            call("a", arguments={"x": 1}),
            call("b", complete=False, arguments={"x": 2}),
            {"type": "tool_call", "call_id": "", "name": "demo", "complete": False},
        )
        replies = [terminal, response("stop", {"type": "text", "text": "recovered"})]
        session, client, events, loop = await setup(tmp_path, replies, handler)
        try:
            outcome = await loop.run(user_message())
            assert outcome["status"] == ("completed" if reason == "length" else reason)
            assert len(client.requests) == (2 if reason == "length" else 1)
            assert not any(e["type"] == "tool_start" for e in events)
            ends = [e["data"] for e in events if e["type"] == "tool_end"]
            assert [e["tool_call_id"] for e in ends] == ["a", "b"]
            assert all(e["result"]["status"] == e["status"] for e in ends)
            kinds = [e["type"] for e in events]
            assert kinds[:8] == [
                "run_start",
                "message_committed",
                "response_update",
                "message_committed",
                "message_committed",
                "tool_end",
                "message_committed",
                "tool_end",
            ]
            history = await session.get_history()
            assert [m.get("tool_call_id") for m in history["messages"] if m["role"] == "tool"] == [
                "a",
                "b",
            ]
            if reason == "length":
                replay = client.requests[1]["messages"]
                assert [m["tool_call_id"] for m in replay if m["role"] == "tool"] == ["a"]
        finally:
            await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "reason,blocks",
    [
        ("stop", [call("a", arguments={"x": 1})]),
        ("tool_call", []),
        ("tool_call", [call("a", complete=False)]),
    ],
)
def test_contradictory_response(tmp_path, reason, blocks):
    """结束原因与工具内容矛盾时拒绝运行。"""

    async def scenario():
        async def handler(args, context):
            pytest.fail("tool must not execute")

        session, client, _, loop = await setup(tmp_path, [response(reason, *blocks)], handler)
        try:
            with pytest.raises(ValueError, match="tool call"):
                await loop.run(user_message())
            assert len(client.requests) == 1
            assert (await session.state())["state"] == "idle"
        finally:
            await session.close()

    asyncio.run(scenario())


def test_unconfirmed_stop_blocks_future_run(tmp_path):
    """停止未确认后阻止该循环实例再次运行。"""

    async def scenario():
        async def handler(args, context):
            raise ToolStopError(call("a"), "stop unconfirmed")

        session, client, events, loop = await setup(
            tmp_path,
            [response("tool_call", call("a", arguments={"x": 1}), call("b", arguments={"x": 2}))],
            handler,
        )
        try:
            with pytest.raises(ToolStopError):
                await loop.run(user_message())
            assert len(client.requests) == 1
            assert [m["role"] for m in (await session.get_history())["messages"]] == [
                "user",
                "assistant",
            ]
            assert events[-1]["data"]["status"] == "error"
            assert [e["type"] for e in events] == [
                "run_start",
                "message_committed",
                "response_update",
                "message_committed",
                "tool_start",
                "run_end",
            ]
            with pytest.raises(RuntimeError, match="unconfirmed"):
                await loop.run(user_message())
        finally:
            await session.close()

    asyncio.run(scenario())
