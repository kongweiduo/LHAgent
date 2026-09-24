"""请求前自动压缩与输入边界，验证摘要提交及取消不重复消费输入。"""

import asyncio

import pytest

from lhagent.harness.context.budget import estimate_context_tokens
from tests.samples import user_message
from tests.test_context_compact import material, reply
from tests.test_loop_basic import setup
from tests.test_loop_tools import call, response
from tests.test_loop_tools import setup as setup_tools


def enable(config, callback):
    """缩小预算并启用摘要回调，便于触发请求前压缩。"""
    config["compaction"].update(enabled=True, keep_recent_tokens=0)
    config["budget"]["context_window"] = 100
    config["summary_request"] = callback


def compaction_events(events):
    """仅提取压缩事件，保留发生顺序。"""
    return [e for e in events if e["type"].startswith("compaction_")]


@pytest.mark.parametrize("over", [False, True])
def test_threshold_full_budget_and_summary_persistence(tmp_path, over):
    """达到阈值时按完整预算压缩并持久化摘要。"""

    async def scenario():
        session, client, events, config, _, loop = await setup(tmp_path)
        seen = []

        async def summary(messages, limit, signal):
            assert limit == 10 and signal is loop._tool_context["cancel_event"]
            seen.append(material(messages))
            return reply("brief")

        enable(config, summary)
        instruction = user_message("x" * 200)
        config["prompts"]["system_prompt"] = "fixed prompt"
        initial = [user_message("fixed prompt"), instruction]
        initial[0]["role"] = "system"
        config["budget"]["extra_input_tokens"] = 7
        tokens = estimate_context_tokens(initial, config["budget"])
        config["budget"]["context_window"] = tokens + 10 - int(over)
        try:
            assert (await loop.run(instruction))["status"] == "completed"
            assert len(seen) == int(over)
            assert len(compaction_events(events)) == 2 * int(over)
            records = await session._store.read_records()
            assert len([r for r in records if r["type"] == "assistant"]) == 1
            assert len([r for r in records if r["type"] == "compaction"]) == int(over)
            assert client.requests[0]["messages"][0] == initial[0]
            if over:
                assert compaction_events(events) == [
                    {"type": "compaction_start", "data": {"run_id": "run-1", "tokens": tokens}},
                    {
                        "type": "compaction_end",
                        "data": {"run_id": "run-1", "status": "success", "error": None},
                    },
                ]
                assert [e["type"] for e in events] == [
                    "run_start",
                    "message_committed",
                    "compaction_start",
                    "compaction_end",
                    "response_update",
                    "response_update",
                    "message_committed",
                    "run_end",
                ]
                assert seen == [(None, [instruction])]
                assert (await session.get_history())["summary"] == "brief"
                assert instruction not in client.requests[0]["messages"]
        finally:
            await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["disabled", "unchanged", "error", "still_large"])
def test_no_repeated_compaction_or_overbudget_request(tmp_path, mode):
    """压缩不足时不重复压缩或继续发送超预算请求。"""

    async def scenario():
        session, client, events, config, _, loop = await setup(tmp_path)
        calls = []

        async def summary(*args):
            calls.append(args)
            return reply(
                "x" * 1000 if mode == "still_large" else "brief",
                reason="length" if mode == "error" else "stop",
            )

        enable(config, summary)
        if mode == "disabled":
            config["compaction"]["enabled"] = False
        if mode == "unchanged":
            config["compaction"]["keep_recent_tokens"] = 10000
        try:
            with pytest.raises(ValueError):
                await loop.run(user_message("x" * 1000))
            assert not client.requests
            assert len(calls) == int(mode in ("error", "still_large"))
            assert events[-1]["data"]["status"] == "error"
            assert (await session.state())["state"] == "idle"
        finally:
            await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "when", ["summary", "before_commit", "commit", "task_summary", "task_commit"]
)
def test_cancel_summary_and_accepted_commit(tmp_path, when):
    """取消摘要与取消已接受提交分别遵守生成/持久化边界。"""

    async def scenario():
        session, client, events, config, queue, loop = await setup(tmp_path)
        entered, release = asyncio.Event(), asyncio.Event()
        cleaned = asyncio.Event()
        original_emit, original_commit = loop._emit, session.commit_compaction

        async def summary(*args):
            if when in ("summary", "task_summary"):
                entered.set()
                try:
                    await release.wait()
                finally:
                    cleaned.set()
            return reply("brief")

        async def emit(event):
            await original_emit(event)
            if event["type"] == "compaction_end" and when == "before_commit":
                entered.set()
                await release.wait()

        async def commit(result):
            entered.set()
            await release.wait()
            await original_commit(result)

        enable(config, summary)
        loop._emit = emit
        if when in ("commit", "task_commit"):
            session.commit_compaction = commit
        try:
            task = asyncio.create_task(loop.run(user_message("x" * 1000)))
            await asyncio.wait_for(entered.wait(), 2)
            queue.steer(user_message("pending"))
            if when.startswith("task_"):
                task.cancel()
                await asyncio.sleep(0)
                if when == "task_commit":
                    assert not task.done()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                cancellation = asyncio.create_task(loop.cancel())
                await asyncio.sleep(0)
                assert not cancellation.done()
                release.set()
                assert (await task)["status"] == "cancelled"
                await cancellation
            await loop.wait_for_idle()
            assert not client.requests
            assert queue.has_pending()
            assert bool((await session.get_history())["summary"]) == (
                when in ("commit", "task_commit")
            )
            if "summary" in when:
                assert cleaned.is_set()
            assert events[-1]["data"]["status"] == "cancelled"
            assert len(compaction_events(events)) == 2
        finally:
            await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("prequeued", [False, True])
def test_steering_during_summary_consumed_once(tmp_path, prequeued):
    """摘要期间到达的 steering 在边界只消费一次。"""

    async def scenario():
        session, client, events, config, queue, loop = await setup(tmp_path)
        seen = []

        async def summary(messages, limit, signal):
            seen.append(material(messages))
            queue.steer(user_message("later"))
            if not prequeued:
                queue.steer(user_message("last"))
            return reply("brief")

        enable(config, summary)
        if prequeued:
            queue.steer(user_message("earlier"))
        try:
            assert (await loop.run(user_message("x" * 1000)))["status"] == "completed"
            assert len(seen) == 1
            assert len(client.requests) == 2
            users = [
                [
                    m["content"][0]["text"]
                    for m in r["messages"]
                    if m["role"] == "user"
                    and m["content"][0]["text"] in ("earlier", "later", "last")
                ]
                for r in client.requests
            ]
            assert users == ([[], ["later"]] if prequeued else [["later"], ["later", "last"]])
            records = await session._store.read_records()
            texts = [r["message"]["content"][0]["text"] for r in records if r["type"] == "user"]
            assert texts.count("later") == 1
            assert not queue.has_pending()
        finally:
            await session.close()

    asyncio.run(scenario())


def test_tool_result_triggers_next_request_compaction(tmp_path):
    """工具结果增大上下文时在下一请求前压缩。"""

    async def scenario():
        async def handler(args, context):
            return {
                "content": [{"type": "text", "text": "x" * 2000}],
                "details": {},
                "is_error": False,
                "truncated": False,
            }

        session, client, events, loop = await setup_tools(
            tmp_path,
            [response("tool_call", call("a", arguments={"x": 1})), response("stop")],
            handler,
        )
        seen = []

        async def summary(messages, *args):
            seen.append(material(messages))
            return reply("brief")

        enable(loop._config, summary)
        try:
            assert (await loop.run(user_message()))["status"] == "completed"
            assert len(seen) == 1 and len(client.requests) == 2
            assert [m["role"] for m in seen[0][1]] == ["user", "assistant", "tool"]
            assert [e["type"] for e in events].index("tool_end") < [
                e["type"] for e in events
            ].index("compaction_start")
        finally:
            await session.close()

    asyncio.run(scenario())


def test_no_compaction_without_next_request(tmp_path):
    """没有下一模型请求时不额外压缩。"""

    async def scenario():
        session, client, events, config, _, loop = await setup(tmp_path)

        async def summary(*args):
            pytest.fail("no next request")

        enable(config, summary)
        original = client.stream

        async def stream(request, *, cancel_event):
            async for event in original(request, cancel_event=cancel_event):
                if event["result"] is not None:
                    event["result"]["content"] = [{"type": "text", "text": "x" * 2000}]
                yield event

        client.stream = stream
        try:
            assert (await loop.run(user_message()))["status"] == "completed"
            assert not compaction_events(events)
        finally:
            await session.close()

    asyncio.run(scenario())
