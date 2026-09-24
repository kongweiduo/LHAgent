"""使用真实会话和可控假客户端验证 steering/follow-up 接入边界。"""

import asyncio
from itertools import count

import pytest

from lhagent.harness.loop.queue import InputQueue
from tests.samples import user_message
from tests.test_loop_basic import setup
from tests.test_loop_tools import call, response
from tests.test_loop_tools import setup as setup_tools


def message(text):
    """构造用于队列接入的用户文本消息。"""
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def users(messages):
    """按顺序提取用户文本，检查队列接入位置。"""
    return [m["content"][0]["text"] for m in messages if m["role"] == "user"]


def test_stream_priority_fifo_and_modes(tmp_path):
    """流式阶段排队输入遵守优先级、FIFO 和各自模式。"""

    async def scenario(mode):
        directory = tmp_path / mode
        directory.mkdir()
        session, client, _, _, _, loop = await setup(directory)
        queue = loop._queue = InputQueue(mode, mode)
        original = loop._emit

        async def emit(event):
            await original(event)
            if (
                event["type"] == "response_update"
                and event["data"]["phase"] == "delta"
                and len(client.requests) == 1
            ):
                queue.follow_up(message("f1"))
                queue.steer(message("s1"))
                queue.follow_up(message("f2"))
                queue.steer(message("s2"))
                assert users((await session.get_history())["messages"]) == ["start"]

        loop._emit = emit
        try:
            assert (await loop.run(message("start")))["status"] == "completed"
            snapshots = [users(r["messages"]) for r in client.requests]
            assert snapshots == (
                [
                    ["start"],
                    ["start", "s1"],
                    ["start", "s1", "s2"],
                    ["start", "s1", "s2", "f1"],
                    ["start", "s1", "s2", "f1", "f2"],
                ]
                if mode == "one_at_a_time"
                else [
                    ["start"],
                    ["start", "s1", "s2"],
                    ["start", "s1", "s2", "f1", "f2"],
                ]
            )
            assert users((await session.get_history())["messages"]) == snapshots[-1]
            assert not queue.has_pending()
        finally:
            await session.close()

    for mode in ("one_at_a_time", "all"):
        asyncio.run(scenario(mode))


@pytest.mark.parametrize("prequeued", [False, True])
def test_preparation_recheck_takes_only_one(tmp_path, prequeued):
    """准备阶段再次检查时最多接纳一个阶段批次。"""

    async def scenario():
        session, client, _, _, queue, loop = await setup(tmp_path)
        original = session.get_history
        reads = 0
        if prequeued:
            queue.steer(message("s1"))
        queue.follow_up(message("f1"))
        # Idle enqueue never starts the loop or persists a message.
        await asyncio.sleep(0)
        assert not client.requests
        assert not (await original())["messages"]

        async def history():
            nonlocal reads
            reads += 1
            snapshot = await original()
            if reads == 1:
                if not prequeued:
                    queue.steer(message("s1"))
                queue.steer(message("s2"))
            return snapshot

        session.get_history = history
        try:
            assert (await loop.run(message("start")))["status"] == "completed"
            assert [users(r["messages"]) for r in client.requests] == [
                ["start", "s1"],
                ["start", "s1", "s2"],
                ["start", "s1", "s2", "f1"],
            ]
        finally:
            await session.close()

    asyncio.run(scenario())


def test_tools_finish_entire_batch_before_steering(tmp_path):
    """steering 只能在整个工具批次结束后接入。"""

    async def scenario():
        invoked = []

        async def handler(args, context):
            invoked.append(args["x"])
            if args["x"] == 1:
                loop._queue.follow_up(message("follow"))
                loop._queue.steer(message("steer"))
            assert users((await session.get_history())["messages"]) == ["start"]
            return {
                "content": [{"type": "text", "text": "ok"}],
                "details": {},
                "is_error": False,
                "truncated": False,
            }

        session, client, _, loop = await setup_tools(
            tmp_path,
            [
                response("tool_call", call("a", arguments={"x": 1}), call("b", arguments={"x": 2})),
                response("stop"),
                response("stop"),
            ],
            handler,
        )
        ids = count()
        loop._config["new_call_id"] = lambda: f"request-{next(ids)}"
        try:
            assert (await loop.run(message("start")))["status"] == "completed"
            assert invoked == [1, 2]
            assert [m["role"] for m in client.requests[1]["messages"]] == [
                "user",
                "assistant",
                "tool",
                "tool",
                "user",
            ]
            assert users(client.requests[1]["messages"]) == ["start", "steer"]
            assert users(client.requests[2]["messages"]) == ["start", "steer", "follow"]
        finally:
            await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("reason", ["error", "cancelled", "length"])
def test_terminal_failure_keeps_inputs(tmp_path, reason):
    """失败终态不消费待处理输入。"""

    async def scenario():
        session, client, _, _, queue, loop = await setup(tmp_path, reason=reason)
        original = loop._emit

        async def emit(event):
            await original(event)
            if event["type"] == "response_update" and event["data"]["phase"] == "delta":
                queue.steer(message("steer"))
                queue.follow_up(message("follow"))

        loop._emit = emit
        try:
            assert (await loop.run(user_message()))["status"] == reason
            assert len(client.requests) == 1
            assert queue.clear() == {"steer": [message("steer")], "follow_up": [message("follow")]}
        finally:
            await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("during_preparation", [False, True])
def test_active_cancel_preserves_unconsumed_input(tmp_path, during_preparation):
    """活动运行取消保留尚未消费的队列消息。"""

    async def scenario():
        session, client, _, _, queue, loop = await setup(tmp_path)
        entered = asyncio.Event()
        release = asyncio.Event()
        original_history = session.get_history
        original_stream = client.stream

        async def history():
            snapshot = await original_history()
            entered.set()
            await release.wait()
            return snapshot

        async def stream(request, *, cancel_event):
            entered.set()
            await cancel_event.wait()
            async for event in original_stream(request, cancel_event=cancel_event):
                yield event

        if during_preparation:
            session.get_history = history
        else:
            client.stream = stream
        try:
            task = asyncio.create_task(loop.run(message("start")))
            await entered.wait()
            queue.steer(message("steer"))
            queue.follow_up(message("follow"))
            cancellation = asyncio.create_task(loop.cancel())
            await asyncio.sleep(0)
            release.set()
            assert (await task)["status"] == "cancelled"
            await cancellation
            await loop.wait_for_idle()
            assert len(client.requests) == (0 if during_preparation else 1)
            assert queue.clear() == {"steer": [message("steer")], "follow_up": [message("follow")]}
            assert users((await original_history())["messages"]) == ["start"]
        finally:
            await session.close()

    asyncio.run(scenario())
