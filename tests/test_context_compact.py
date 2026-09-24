"""滚动压缩编排与真实会话集成，验证分段摘要、失败及提交后恢复。"""

import asyncio
import json
from copy import deepcopy

import pytest

from lhagent.harness.context.assembly import assemble_context
from lhagent.harness.context.budget import estimate_context_tokens
from lhagent.harness.context.compaction import compact
from lhagent.harness.session.jsonl import JsonlSessionStore
from lhagent.harness.session.session import Session
from tests.samples import mixed_result, user_message
from tests.test_session import loop_result, metadata

BUDGET = {"context_window": 70, "extra_input_tokens": 19}
SETTINGS = {"enabled": True, "reserve_tokens": 20, "keep_recent_tokens": 1}
PROMPTS = {"system_prompt": "fixed", "additional_messages": []}


def assistant(value):
    """构造正常结束的文本模型消息。"""
    return {
        "role": "assistant",
        "finish_reason": "stop",
        "content": [{"type": "text", "text": value}],
    }


def context(messages, summary=None, instruction=None):
    """将历史、摘要和新指令组合为压缩素材。"""
    return {
        "prompts": PROMPTS,
        "history": {
            "summary": summary,
            "messages": messages,
            "entry_ids": [f"id-{i}" for i in range(len(messages))],
        },
        "new_instruction": instruction,
    }


def reply(summary, *, reason="stop"):
    """构造指定摘要文本和终态的模型结果，填入可观测用量。"""
    result = mixed_result()
    result.update(content=[{"type": "text", "text": summary}], finish_reason=reason)
    result["stats"]["usage"]["input_tokens"] = len(summary)
    return result


def material(request):
    """从摘要请求解码旧摘要及 JSONL 历史，供回调断言使用。"""
    payload = json.loads(request[1]["content"][0]["text"])
    return payload["previous_summary"], [
        json.loads(line) for line in payload["history_jsonl"].splitlines()
    ]


def serialized(messages):
    """移除不进入摘要素材的结束元数据，生成期望消息列表。"""
    return [
        {key: value for key, value in message.items() if key != "finish_reason"}
        for message in messages
    ]


def test_ordinary_rolling_and_full_reestimate():
    """滚动摘要完成后按完整新请求重新估算预算。"""

    async def scenario():
        data = context(
            [user_message("old"), assistant("done"), user_message("recent")],
            "previous",
            user_message("new"),
        )
        snapshot = deepcopy(data)
        seen = []
        event = asyncio.Event()

        async def callback(request, limit, signal):
            assert signal is event and limit == 42
            seen.append(material(request))
            return reply("updated")

        result = await compact(data, BUDGET, SETTINGS, callback, 42, event)
        assert result["status"] == "success"
        assert result["history"] == {
            "summary": "updated",
            "messages": data["history"]["messages"][-1:],
            "entry_ids": ["id-2"],
        }
        assert seen == [("previous", serialized(data["history"]["messages"][:2]))]
        assert result["usage"] == [reply("updated")["stats"]["usage"]]
        assert result["tokens_before"] == estimate_context_tokens(assemble_context(data), BUDGET)
        assert result["estimated_tokens_after"] == estimate_context_tokens(
            assemble_context({**data, "history": result["history"]}), BUDGET
        )
        assert (
            result["estimated_tokens_after"] > BUDGET["context_window"] - SETTINGS["reserve_tokens"]
        )
        assert data == snapshot

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "older,prior,expected_calls",
    [
        (True, "prior", 2),
        (False, "prior", 1),
        (False, None, 1),
    ],
)
def test_split_turn_summaries(older, prior, expected_calls):
    """被切开的轮次使用独立前缀摘要。"""

    async def scenario():
        messages = ([user_message("earlier"), assistant("finished")] if older else []) + [
            user_message("current"),
            assistant("progress"),
            assistant("tail"),
        ]
        data = context(messages, prior)
        seen = []

        async def callback(request, limit, event):
            seen.append(material(request))
            return reply("history-updated" if len(seen) == 1 and older else "prefix-updated")

        result = await compact(data, BUDGET, SETTINGS, callback, 20, asyncio.Event())
        assert result["status"] == "success"
        assert len(seen) == expected_calls
        if older:
            assert seen[0] == (prior, serialized(messages[:2]))
            assert seen[1] == (None, serialized(messages[2:4]))
            assert "history-updated" in result["history"]["summary"]
        else:
            assert seen[0] == (None, serialized(messages[:2]))
            assert prior is None or prior in result["history"]["summary"]
        assert "prefix-updated" in result["history"]["summary"]
        assert result["history"]["entry_ids"] == [f"id-{len(messages) - 1}"]
        assert result["history"]["messages"] == messages[-1:]
        assert len(result["usage"]) == expected_calls

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["error", "cancelled", "signal", "exception", "task"])
def test_second_request_failure_drops_partial_summary(failure):
    """第二次摘要失败时不接纳部分压缩结果。"""

    async def scenario():
        data = context(
            [
                user_message("old"),
                assistant("done"),
                user_message("now"),
                assistant("progress"),
                assistant("tail"),
            ],
            "prior",
        )
        signal = asyncio.Event()
        calls = 0
        cleaned = asyncio.Event()

        async def callback(request, limit, event):
            nonlocal calls
            calls += 1
            if calls == 1:
                return reply("updated")
            if failure == "signal":
                signal.set()
            if failure == "exception":
                raise OSError("secret")
            if failure == "task":
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaned.set()
            return reply(
                "partial",
                reason="length"
                if failure == "error"
                else "cancelled"
                if failure == "cancelled"
                else "stop",
            )

        task = asyncio.create_task(compact(data, BUDGET, SETTINGS, callback, 20, signal))
        if failure == "task":
            while calls < 2:
                await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cleaned.is_set()
            return
        result = await task
        assert calls == 2
        assert result["status"] == ("error" if failure in ("error", "exception") else "cancelled")
        assert result["history"] is None and result["estimated_tokens_after"] is None
        assert result["error"] is None or "secret" not in result["error"]
        assert len(result["usage"]) == (1 if failure == "exception" else 2)

    asyncio.run(scenario())


def test_cancel_after_first_summary_skips_prefix():
    """首次摘要后取消不会再启动前缀请求。"""

    async def scenario():
        data = context(
            [
                user_message("old"),
                assistant("done"),
                user_message("now"),
                assistant("progress"),
                assistant("tail"),
            ]
        )
        signal = asyncio.Event()
        calls = 0

        async def callback(request, limit, event):
            nonlocal calls
            calls += 1
            signal.set()
            return reply("partial")

        result = await compact(data, BUDGET, SETTINGS, callback, 20, signal)
        assert calls == 1
        assert result["status"] == "cancelled" and result["history"] is None
        assert result["usage"] == [reply("partial")["stats"]["usage"]]

    asyncio.run(scenario())


def test_no_content_and_pre_request_cancellation():
    """无内容或预取消不会产生可提交的新摘要。"""

    async def callback(*args):
        pytest.fail("no request expected")

    async def scenario():
        empty = context([], "prior")
        result = await compact(empty, BUDGET, SETTINGS, callback, 20, asyncio.Event())
        assert result == {
            "status": "unchanged",
            "history": None,
            "error": None,
            "tokens_before": estimate_context_tokens(assemble_context(empty), BUDGET),
            "estimated_tokens_after": None,
            "usage": [],
        }
        signal = asyncio.Event()
        signal.set()
        result = await compact(
            context([user_message("old")]),
            BUDGET,
            {**SETTINGS, "keep_recent_tokens": 0},
            callback,
            20,
            signal,
        )
        assert result["status"] == "cancelled" and result["history"] is None

    asyncio.run(scenario())


def test_session_commit_reopen_and_compact_again(tmp_path):
    """压缩提交后可重新打开会话并继续滚动压缩。"""

    async def scenario():
        meta = metadata(tmp_path)
        session = Session(meta, await JsonlSessionStore.create(meta))
        await session.start_run("r1")
        for value in ("first", "second", "third"):
            await session.append_user(user_message(value))
        old_ids = (await session.get_history())["entry_ids"]
        seen = []

        async def callback(request, limit, signal):
            seen.append(material(request))
            return reply("summary-1" if len(seen) == 1 else "summary-2")

        def input_for(history):
            return {"prompts": PROMPTS, "history": history, "new_instruction": None}

        first = await compact(
            input_for(await session.get_history()), BUDGET, SETTINGS, callback, 20, asyncio.Event()
        )
        assert first["history"]["entry_ids"] == old_ids[-1:]
        await session.commit_compaction(first)
        await session.finish_run(loop_result())
        await session.close()
        session = Session(meta, await JsonlSessionStore.open(meta))
        try:
            assert await session.get_history() == first["history"]
            await session.start_run("r2")
            await session.append_user(user_message("fourth"))
            second = await compact(
                input_for(await session.get_history()),
                BUDGET,
                SETTINGS,
                callback,
                20,
                asyncio.Event(),
            )
            assert seen[1][0] == "summary-1"
            assert seen[1][1] == [user_message("third")]
            await session.commit_compaction(second)
            assert (await session.get_history())["summary"] == "summary-2"
            records = await session._store.read_records()
            assert all(
                "messages" not in record for record in records if record["type"] == "compaction"
            )
        finally:
            await session.close()

    asyncio.run(scenario())
