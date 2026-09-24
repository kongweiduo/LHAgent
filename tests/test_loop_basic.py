"""无工具模型循环的终态、预算、取消及会话收尾验收。"""

import asyncio
from copy import deepcopy

import pytest

from lhagent.harness.loop.loop import AgentLoop
from lhagent.harness.loop.queue import InputQueue
from lhagent.harness.session.jsonl import JsonlSessionStore
from lhagent.harness.session.session import Session
from tests.samples import user_message


def make_result(reason="stop"):
    """构造带完整统计字段的指定终态模型结果。"""
    return {
        "call_id": "call-1",
        "content": [{"type": "text", "text": "answer"}],
        "finish_reason": reason,
        "error": "failed" if reason == "error" else None,
        "error_kind": "other" if reason == "error" else None,
        "stats": {
            "usage": {
                key: None
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                )
            },
            "elapsed_seconds": 0.1,
            "first_content_seconds": 0.01,
            "attempts": 1,
        },
    }


class FakeClient:
    """可暂停的文本客户端替身，记录请求副本与流退出。"""

    def __init__(self, reason="stop"):
        self.reason = reason
        self.requests = []
        self.wait = None
        self.closed = False

    async def stream(self, request, *, cancel_event):
        self.requests.append(deepcopy(request))
        try:
            yield {
                "call_id": request["call_id"],
                "type": "delta",
                "block_index": 0,
                "data": {"type": "text", "text": "answer"},
                "result": None,
            }
            if self.wait is not None:
                await self.wait.wait()
            response = make_result(self.reason)
            yield {
                "call_id": request["call_id"],
                "type": "error"
                if self.reason == "error"
                else "cancelled"
                if self.reason == "cancelled"
                else "done",
                "block_index": None,
                "data": {},
                "result": response,
            }
        finally:
            self.closed = True


async def setup(tmp_path, *, reason="stop"):
    """组装真实临时会话和假客户端循环；调用者负责关闭返回会话。"""
    metadata = {"id": "s1", "created_at": 1.0, "path": str(tmp_path / "session.jsonl")}
    session = Session(metadata, await JsonlSessionStore.create(metadata))
    client = FakeClient(reason)
    events = []

    async def emit(event):
        events.append(deepcopy(event))

    config = {
        "model": "test",
        "parameters": {},
        "prompts": {"system_prompt": None, "additional_messages": []},
        "budget": {"context_window": 1000, "extra_input_tokens": 0},
        "compaction": {"enabled": False, "reserve_tokens": 10, "keep_recent_tokens": 0},
        "summary_request": None,
        "max_summary_output_tokens": 10,
        "tools": [],
        "new_call_id": lambda: "call-1",
        "new_run_id": lambda: "run-1",
    }
    queue = InputQueue()
    loop = AgentLoop(client, session, queue, config, {"cancel_event": asyncio.Event()}, emit)
    return session, client, events, config, queue, loop


@pytest.mark.parametrize(
    "reason,status",
    [("stop", "completed"), ("length", "length"), ("error", "error"), ("cancelled", "cancelled")],
)
def test_terminal_roundtrip(tmp_path, reason, status):
    """模型终态通过真实会话提交并可读取。"""

    async def scenario():
        session, client, events, _, _, loop = await setup(tmp_path, reason=reason)
        try:
            outcome = await loop.run(user_message())
            assert outcome["status"] == status
            assert outcome["last_response"] == make_result(reason)
            assert [event["type"] for event in events] == [
                "run_start",
                "message_committed",
                "response_update",
                "response_update",
                "message_committed",
                "run_end",
            ]
            assert events[2]["data"]["phase"] == "delta"
            assert events[2]["data"]["delta"] == {"type": "text", "text": "answer"}
            assert events[3]["data"]["phase"] == "end"
            assert events[3]["data"]["result"] == make_result(reason)
            assert events[-1]["data"]["status"] == status
            assert client.requests[0]["messages"] == [user_message()]
            history = await session.get_history()
            assert len(history["messages"]) == 2
            assert history["messages"][1]["content"] == make_result(reason)["content"]
            assert (await session.state())["state"] == "idle"
            assert client.closed
        finally:
            await session.close()

    asyncio.run(scenario())


def test_unsupported_and_budget_rejected(tmp_path):
    """不受支持参数及超预算请求在发起模型调用前拒绝。"""

    async def scenario():
        session, client, _, config, queue, loop = await setup(tmp_path)
        try:
            for change in (
                lambda: config["parameters"].update(tool_choice="auto"),
                lambda: config["parameters"].update(tools=[]),
            ):
                change()
                with pytest.raises(NotImplementedError):
                    await loop.run(user_message())
                config["tools"].clear()
                config["compaction"]["enabled"] = False
                queue.clear()
                config["parameters"].clear()
            config["budget"]["context_window"] = 11
            with pytest.raises(ValueError, match="budget"):
                await loop.run(user_message())
            assert not client.requests
            assert (await session.state())["state"] == "idle"
        finally:
            await session.close()

    asyncio.run(scenario())


def test_concurrency_and_external_cancel(tmp_path):
    """并发入口与外部取消遵守单次运行边界。"""

    async def scenario():
        session, client, events, _, _, loop = await setup(tmp_path)
        client.wait = asyncio.Event()
        try:
            task = asyncio.create_task(loop.run(user_message()))
            while not client.requests:
                await asyncio.sleep(0)
            with pytest.raises(RuntimeError, match="already running"):
                await loop.run(user_message())
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert client.closed
            assert [e["type"] for e in events][-1] == "run_end"
            assert events[-1]["data"]["status"] == "cancelled"
            assert len((await session.get_history())["messages"]) == 1
            assert (await session.state())["state"] == "idle"
        finally:
            await session.close()

    asyncio.run(scenario())


def test_cleanup_error_does_not_mask_original(tmp_path):
    """清理异常不掩盖原始运行失败。"""

    async def scenario():
        session, client, _, _, _, loop = await setup(tmp_path)

        async def fail_emit(event):
            raise ValueError("subscriber failed")

        async def fail_finish(result):
            raise RuntimeError("finish failed")

        original = session.finish_run
        loop._emit = fail_emit
        session.finish_run = fail_finish
        try:
            with pytest.raises(ValueError, match="subscriber failed") as exc:
                await loop.run(user_message())
            assert any("finish failed" in note for note in exc.value.__notes__)
            assert not client.requests
        finally:
            session.finish_run = original
            await session.close()

    asyncio.run(scenario())


def test_stream_interrupt_and_queued_input(tmp_path):
    """断流不丢失尚未消费的排队输入。"""

    async def scenario():
        session, client, events, _, queue, loop = await setup(tmp_path)
        original = client.stream

        async def interrupted(request, *, cancel_event):
            yield {
                "call_id": request["call_id"],
                "type": "delta",
                "block_index": 0,
                "data": {"type": "text", "text": "partial"},
                "result": None,
            }

        client.stream = interrupted
        try:
            with pytest.raises(RuntimeError, match="terminal result"):
                await loop.run(user_message())
            assert len((await session.get_history())["messages"]) == 1
            assert events[-1]["data"]["status"] == "error"
            client.stream = original
            client.wait = asyncio.Event()
            loop._config["new_run_id"] = lambda: "run-2"
            task = asyncio.create_task(loop.run(user_message()))
            while not client.requests:
                await asyncio.sleep(0)
            queue.follow_up(user_message())
            client.reason = "error"
            client.wait.set()
            assert (await task)["status"] == "error"
            assert queue.has_pending()
            assert events[-1]["data"]["status"] == "error"
            assert (await session.state())["state"] == "idle"
        finally:
            await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["start", "emit", "append", "finish"])
def test_failures(tmp_path, failure):
    """模型或会话失败走统一错误收尾。"""

    async def scenario():
        session, client, events, _, _, loop = await setup(tmp_path)
        original = (
            getattr(
                session,
                "start_run"
                if failure == "start"
                else "append_response"
                if failure == "append"
                else "finish_run",
            )
            if failure != "emit"
            else None
        )

        async def broken(*args):
            raise RuntimeError("broken " + failure)

        if failure == "emit":
            loop._emit = broken
        else:
            setattr(
                session,
                "start_run"
                if failure == "start"
                else "append_response"
                if failure == "append"
                else "finish_run",
                broken,
            )
        try:
            with pytest.raises(RuntimeError, match="broken " + failure):
                await loop.run(user_message())
            assert bool(client.requests) == (failure in ("append", "finish"))
            assert (not events or events[-1]["type"] != "run_end") == (
                failure in ("start", "emit", "finish")
            )
            if failure != "start":
                assert (await session.state())["state"] == (
                    "active" if failure == "finish" else "idle"
                )
        finally:
            if original is not None:
                setattr(
                    session,
                    "start_run"
                    if failure == "start"
                    else "append_response"
                    if failure == "append"
                    else "finish_run",
                    original,
                )
            await session.close()

    asyncio.run(scenario())
