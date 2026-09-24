"""使用真实 JSONL 验证服务端明确超限的一次恢复、排除提交及取消边界。"""

import asyncio
from copy import deepcopy
from itertools import count

import pytest

from lhagent.harness.session.jsonl import JsonlSessionStore
from lhagent.harness.session.session import Session
from tests.samples import user_message
from tests.test_context_compact import material, reply
from tests.test_loop_basic import make_result, setup
from tests.test_loop_tools import Client, call, response
from tests.test_loop_tools import setup as setup_tools


def overflow(*blocks):
    """构造具有明确 context_overflow 分类的失败响应。"""
    result = response("error", *blocks)
    result.update(error="Model context limit exceeded.", error_kind="context_overflow")
    return result


async def configured(tmp_path, responses):
    """组装超限恢复场景，并在摘要回调中确认排除记录已提交。"""
    session, _, events, config, queue, loop = await setup(tmp_path)
    client = loop._client = Client(responses)
    ids = count()
    config["new_call_id"] = lambda: f"request-{next(ids)}"
    config["compaction"]["enabled"] = True
    summaries = []

    async def summary(messages, limit, signal):
        assert limit == 10 and signal is loop._tool_context["cancel_event"]
        summaries.append(material(messages))
        records = await session._store.read_records()
        assert records[-1]["type"] == "history_exclusion"
        return reply("brief")

    config["summary_request"] = summary
    return session, client, events, config, queue, loop, summaries


@pytest.mark.parametrize("again", [False, True])
def test_order_excludes_entire_failed_attempt_and_retries_once(tmp_path, again):
    """明确超限先排除完整失败尝试，再压缩并只重试一次。"""

    async def scenario():
        failed = overflow(call("a"), call("b", complete=False))
        session, client, events, config, _, loop, summaries = await configured(
            tmp_path, [failed, overflow() if again else make_result()]
        )
        settings = deepcopy(config["compaction"])
        original_stream = client.stream

        async def stream(request, *, cancel_event):
            if client.requests:
                assert (await session._store.read_records())[-1]["type"] == "compaction"
            async for event in original_stream(request, cancel_event=cancel_event):
                yield event

        client.stream = stream
        try:
            result = await loop.run(user_message())
            assert result["status"] == ("error" if again else "completed")
            assert len(client.requests) == 2 and len(summaries) == 1
            assert summaries == [(None, [user_message()])]
            assert config["compaction"] == settings
            assert client.requests[0]["call_id"] != client.requests[1]["call_id"]
            assert all(m["role"] != "tool" for m in client.requests[1]["messages"])
            records = await session._store.read_records()
            assert [r["type"] for r in records] == [
                "session",
                "run_start",
                "user",
                "assistant",
                "tool_result",
                "tool_result",
                "history_exclusion",
                "compaction",
                "assistant",
                "run_finish",
            ]
            assert records[6]["entry_ids"] == [r["id"] for r in records[3:6]]
            assert records[3]["response"]["content"] == failed["content"]
            assert not any(e["type"] == "tool_start" for e in events)
            expected = await session.get_history()
        finally:
            await session.close()
        reopened = Session(session.metadata, await JsonlSessionStore.open(session.metadata))
        try:
            assert await reopened.get_history() == expected
        finally:
            await reopened.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mode", ["error", "summary_overflow", "cancelled", "unchanged", "disabled"]
)
def test_incomplete_recovery_does_not_retry_or_undo_exclusion(tmp_path, mode):
    """恢复未完成时不重试，也不撤销已提交排除记录。"""

    async def scenario():
        session, client, events, config, queue, loop, summaries = await configured(
            tmp_path, [overflow()]
        )
        calls = []

        async def summary(*args):
            calls.append(args)
            return (
                overflow()
                if mode == "summary_overflow"
                else reply("", reason="cancelled" if mode == "cancelled" else "length")
            )

        config["summary_request"] = summary
        config["compaction"].update(
            enabled=mode != "disabled", keep_recent_tokens=10000 if mode == "unchanged" else 0
        )
        queue.follow_up(user_message("pending"))
        try:
            outcome = await loop.run(user_message())
            assert outcome["status"] == ("cancelled" if mode == "cancelled" else "error")
            assert len(client.requests) == 1 and queue.has_pending()
            assert len(calls) == int(mode not in ("unchanged", "disabled"))
            records = await session._store.read_records()
            assert not any(r["type"] == "compaction" for r in records)
            assert sum(r["type"] == "history_exclusion" for r in records) == int(mode != "disabled")
            expected = await session.get_history()
            assert len(expected["messages"]) == (2 if mode == "disabled" else 1)
            assert events[-1]["data"]["status"] == outcome["status"]
        finally:
            await session.close()
        reopened = Session(session.metadata, await JsonlSessionStore.open(session.metadata))
        try:
            assert await reopened.get_history() == expected
        finally:
            await reopened.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "reason,kind",
    [
        ("length", None),
        ("cancelled", None),
        ("error", "rate_limit"),
        ("error", "authentication"),
        ("error", "other"),
        ("error", "transport"),
        ("error", "invalid_request"),
        ("error", "protocol"),
    ],
)
def test_only_explicit_error_classification_triggers_recovery(tmp_path, reason, kind):
    """只有明确的错误分类触发超限恢复。"""

    async def scenario():
        result = make_result(reason)
        result.update(error_kind=kind, error="context window exceeded; prompt is too long")
        session, client, events, _, _, loop, summaries = await configured(tmp_path, [result])
        try:
            assert (await loop.run(user_message()))["status"] == reason
            assert len(client.requests) == 1 and not summaries
            assert not any(e["type"] == "compaction_start" for e in events)
            assert not any(
                r["type"] == "history_exclusion" for r in await session._store.read_records()
            )
        finally:
            await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["omit_failed_attempt", "commit_compaction"])
def test_failed_commit_stops_recovery(tmp_path, operation):
    """排除或压缩提交失败后立即停止恢复。"""

    async def scenario():
        session, client, events, _, _, loop, summaries = await configured(tmp_path, [overflow()])

        async def fail(*args):
            raise OSError("write failed")

        setattr(session, operation, fail)
        try:
            with pytest.raises(OSError, match="write failed"):
                await loop.run(user_message())
            assert len(client.requests) == 1
            assert len(summaries) == int(operation == "commit_compaction")
            records = await session._store.read_records()
            assert not any(r["type"] == "compaction" for r in records)
            assert sum(r["type"] == "history_exclusion" for r in records) == int(
                operation == "commit_compaction"
            )
            assert events[-1]["data"]["status"] == "error"
            assert (await session.state())["state"] == "idle"
        finally:
            await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("when", ["exclusion", "summary", "before_commit", "commit"])
@pytest.mark.parametrize("external", [False, True])
def test_cancellation_stops_retry_and_settles_accepted_writes(tmp_path, when, external):
    """取消阻止重试，但已接受写入仍等待落定。"""

    async def scenario():
        session, client, events, config, queue, loop, _ = await configured(tmp_path, [overflow()])
        entered, release = asyncio.Event(), asyncio.Event()
        original_emit = loop._emit

        async def summary(*args):
            if when == "summary":
                entered.set()
                await release.wait()
            return reply("brief")

        async def emit(event):
            await original_emit(event)
            if when == "before_commit" and event["type"] == "compaction_end":
                entered.set()
                await release.wait()

        if when in ("exclusion", "commit"):
            operation = "omit_failed_attempt" if when == "exclusion" else "commit_compaction"
            original = getattr(session, operation)

            async def delayed(value):
                entered.set()
                await release.wait()
                await original(value)

            setattr(session, operation, delayed)
        config["summary_request"] = summary
        loop._emit = emit
        try:
            task = asyncio.create_task(loop.run(user_message()))
            await asyncio.wait_for(entered.wait(), 2)
            queue.steer(user_message("pending"))
            if external:
                task.cancel()
            else:
                cancellation = asyncio.create_task(loop.cancel())
            await asyncio.sleep(0)
            release.set()
            if external:
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                assert (await task)["status"] == "cancelled"
                await cancellation
            await loop.wait_for_idle()
            assert len(client.requests) == 1 and queue.has_pending()
            records = await session._store.read_records()
            assert sum(r["type"] == "history_exclusion" for r in records) == 1
            assert sum(r["type"] == "compaction" for r in records) == int(when == "commit")
            assert events[-1]["data"]["status"] == "cancelled"
            assert (await session.state())["state"] == "idle"
        finally:
            release.set()
            await session.close()

    asyncio.run(scenario())


def test_nonzero_retention_budget_preserves_original_tail(tmp_path):
    """非零保留预算保持原始历史尾部身份。"""

    async def scenario():
        session, client, _, config, _, loop, summaries = await configured(
            tmp_path, [overflow(), make_result()]
        )
        config["compaction"]["keep_recent_tokens"] = 1
        try:
            await session.start_run("earlier")
            await session.append_user(user_message("old history"))
            await session.finish_run({"status": "completed", "last_response": None, "error": None})
            assert (await loop.run(user_message("current instruction")))["status"] == "completed"
            assert summaries == [(None, [user_message("old history")])]
            assert user_message("current instruction") in client.requests[1]["messages"]
            assert config["compaction"]["keep_recent_tokens"] == 1
            records = await session._store.read_records()
            current = next(r for r in records if r["type"] == "user" and r["run_id"] == "run-1")
            compressed = next(r for r in records if r["type"] == "compaction")
            assert compressed["retained_entry_ids"] == [current["id"]]
        finally:
            await session.close()

    asyncio.run(scenario())


def test_recovery_does_not_compact_again_when_summary_still_exceeds_budget(tmp_path):
    """摘要后仍超预算时不再次压缩。"""

    async def scenario():
        session, client, _, config, _, loop, _ = await configured(tmp_path, [overflow()])
        calls = []

        async def summary(*args):
            calls.append(args)
            return reply("x" * 10000)

        config["summary_request"] = summary
        try:
            with pytest.raises(ValueError, match="after compaction"):
                await loop.run(user_message())
            assert len(calls) == len(client.requests) == 1
            assert (await session.get_history())["summary"] == "x" * 10000
        finally:
            await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["steering", "success", "length", "queued_only", "new_run"])
def test_recovery_marker_resets_only_at_progress_boundaries(tmp_path, mode):
    """恢复次数标记仅在实际进展边界重置。"""

    async def scenario():
        async def handler(args, context):
            return {
                "content": [{"type": "text", "text": "ok"}],
                "details": {},
                "is_error": False,
                "truncated": False,
            }

        replies = [overflow()]
        if mode in ("success", "length"):
            replies.append(
                response(
                    "tool_call" if mode == "success" else "length", call("a", arguments={"x": 1})
                )
            )
        replies.extend([overflow(), make_result()])
        session, client, _, loop = await setup_tools(tmp_path, replies, handler)
        config = loop._config
        config["new_call_id"] = lambda: f"request-{next(ids)}"
        ids = count()
        config["compaction"]["enabled"] = True
        summaries = []

        async def summary(messages, *args):
            summaries.append(material(messages))
            if len(summaries) == 1:
                if mode == "steering":
                    loop._queue.steer(user_message("new direction"))
                elif mode == "queued_only":
                    loop._queue.follow_up(user_message("pending"))
            return reply("brief")

        config["summary_request"] = summary
        try:
            result = await loop.run(user_message())
            if mode == "new_run":
                assert result["status"] == "error" and len(summaries) == 1
                client.responses = [overflow(), make_result()]
                config["new_run_id"] = lambda: "run-2"
                result = await loop.run(user_message("new run"))
            assert result["status"] == (
                "error" if mode in ("queued_only", "length") else "completed"
            )
            assert len(summaries) == (1 if mode in ("queued_only", "length") else 2)
            assert (
                len(client.requests)
                == {"steering": 3, "success": 4, "length": 3, "queued_only": 2, "new_run": 4}[mode]
            )
            if mode == "queued_only":
                assert loop._queue.has_pending()
            if mode == "steering":
                assert summaries[1][1] == [user_message("new direction")]
            if mode == "success":
                assert [m["role"] for m in summaries[1][1]] == ["assistant", "tool"]
        finally:
            await session.close()

    asyncio.run(scenario())
