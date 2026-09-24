"""历史排除、压缩引用及恢复验收；不调用摘要模型。"""

import asyncio
from copy import deepcopy
from pathlib import Path

import pytest

from lhagent.harness.session.jsonl import JsonlSessionStore
from lhagent.harness.session.session import Session
from tests.samples import mixed_result, user_message
from tests.test_session import loop_result, metadata


def compact(history, keep=0, summary="summary"):
    """构造引用原历史尾部的成功压缩结果，不调用摘要模型。"""
    return {
        "status": "success",
        "history": {
            "summary": summary,
            "messages": deepcopy(history["messages"][-keep:]) if keep else [],
            "entry_ids": history["entry_ids"][-keep:] if keep else [],
        },
        "tokens_before": 100,
        "estimated_tokens_after": 20,
        "usage": [],
        "error": None,
    }


def test_multiple_compactions_exclusion_and_reopen(tmp_path):
    """多次压缩和排除在重新打开会话后保持有效。"""

    async def scenario():
        meta = metadata(tmp_path)
        session = Session(meta, await JsonlSessionStore.create(meta))
        await session.start_run("r1")
        first = await session.append_user(user_message())
        second = await session.append_user(user_message())
        third = await session.append_response(mixed_result())
        prefix = Path(meta["path"]).read_bytes()
        await session.commit_compaction(compact(await session.get_history(), 2, "first"))
        await session.omit_failed_attempt([second])
        fourth = await session.append_user(user_message())
        assert (await session.get_history())["entry_ids"] == [third, fourth]
        await session.commit_compaction(compact(await session.get_history(), 1, "second"))
        await session.finish_run(loop_result())
        expected = await session.get_history()
        expected["messages"][0]["content"].clear()
        assert (await session.get_history())["messages"][0] == user_message()
        await session.close()
        session = Session(meta, await JsonlSessionStore.open(meta))
        try:
            assert (await session.get_history())["entry_ids"] == [fourth]
            assert (await session.get_history())["summary"] == "second"
            await session.start_run("r2")
            await session.commit_compaction(compact(await session.get_history(), summary="all"))
            fifth = await session.append_user(user_message())
            await session.omit_failed_attempt([first, fourth])
            assert await session.get_history() == {
                "summary": "all",
                "messages": [user_message()],
                "entry_ids": [fifth],
            }
            records = await session._store.read_records()
            assert len([r for r in records if r["type"] in ("user", "assistant")]) == 5
            for record in records:
                if record["type"] == "compaction":
                    assert "messages" not in record and "history" not in record
            assert Path(meta["path"]).read_bytes().startswith(prefix)
        finally:
            await session.close()
        session = Session(meta, await JsonlSessionStore.open(meta))
        try:
            assert (await session.get_history())["entry_ids"] == [fifth]
            assert (await session.get_history())["summary"] == "all"
        finally:
            await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "invalid",
    [
        "duplicate",
        "missing",
        "reversed",
        "prefix",
        "identity",
        "rewritten",
        "placeholder",
        "length",
        "excluded",
        "summarized",
        "blank",
        "error",
        "cancelled",
        "unchanged",
        "tokens",
        "usage",
        "stale",
    ],
)
def test_invalid_compaction_never_appends(tmp_path, invalid):
    """非法压缩引用在追加前拒绝。"""

    async def scenario():
        meta = metadata(tmp_path)
        session = Session(meta, await JsonlSessionStore.create(meta))
        try:
            await session.start_run("r1")
            for _ in range(3):
                await session.append_user(user_message())
            history = await session.get_history()
            result = compact(history, 2)
            tail = result["history"]
            if invalid == "duplicate":
                tail["entry_ids"][0] = tail["entry_ids"][1]
            elif invalid == "missing":
                tail["entry_ids"][0] = "unknown"
            elif invalid == "reversed":
                tail["entry_ids"].reverse()
            elif invalid == "prefix":
                tail["entry_ids"] = history["entry_ids"][:2]
            elif invalid == "identity":
                tail["entry_ids"][0] = history["entry_ids"][0]
            elif invalid == "rewritten":
                tail["messages"][0]["content"].clear()
            elif invalid == "placeholder":
                tail["messages"][0] = {
                    "role": "tool",
                    "tool_call_id": "unknown",
                    "content": [
                        {"type": "tool_result", "content": "unavailable", "is_error": True}
                    ],
                }
            elif invalid == "length":
                tail["messages"].pop()
            elif invalid == "excluded":
                await session.omit_failed_attempt([tail["entry_ids"][0]])
            elif invalid == "summarized":
                await session.commit_compaction(compact(history, 1))
            elif invalid == "stale":
                await session.append_user(user_message())
            elif invalid == "blank":
                tail["summary"] = "  "
            elif invalid in ("error", "cancelled", "unchanged"):
                result["status"] = invalid
            elif invalid == "tokens":
                result["estimated_tokens_after"] = None
            elif invalid == "usage":
                result["usage"] = [None]
            before = Path(meta["path"]).read_bytes()
            with pytest.raises(ValueError):
                await session.commit_compaction(result)
            assert Path(meta["path"]).read_bytes() == before
        finally:
            await session.close()

    asyncio.run(scenario())


def test_exclusion_survives_failed_summary_and_write(tmp_path, monkeypatch):
    """摘要或后续写入失败不撤销已有排除记录。"""

    async def scenario():
        meta = metadata(tmp_path)
        store = await JsonlSessionStore.create(meta)
        session = Session(meta, store)
        try:
            with pytest.raises(RuntimeError, match="no active run"):
                await session.commit_compaction(compact(await session.get_history()))
            await session.start_run("r1")
            await session.append_user(user_message())
            response = mixed_result()
            response.update(finish_reason="error", error="overflow", error_kind="context_overflow")
            failed = await session.append_response(response)
            tool = await session.append_tool_result(
                {
                    "call_id": "tool-call-1",
                    "name": "read",
                    "status": "cancelled",
                    "output": None,
                    "error": "cancelled",
                }
            )
            await session.omit_failed_attempt([failed, tool])
            await session.omit_failed_attempt([failed])
            before = await session.get_history()
            for ids in (["missing"], [failed, failed], ["r1"], [meta["id"]]):
                with pytest.raises(ValueError):
                    await session.omit_failed_attempt(ids)
            result = compact(before)
            result["status"] = "error"
            with pytest.raises(ValueError):
                await session.commit_compaction(result)

            async def fail(records):
                raise OSError("disk unavailable")

            monkeypatch.setattr(store, "append", fail)
            with pytest.raises(OSError):
                await session.commit_compaction(compact(before))
            with pytest.raises(OSError):
                await session.omit_failed_attempt(before["entry_ids"])
            assert await session.get_history() == before
        finally:
            await session.close()
        reopened = Session(meta, await JsonlSessionStore.open(meta))
        try:
            assert await reopened.get_history() == before
            records = await reopened._store.read_records()
            assert any(r["id"] == failed for r in records)
            assert any(r["id"] == tool for r in records)
        finally:
            await reopened.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["compaction", "exclusion"])
def test_cancel_accepted_history_update_is_durable(tmp_path, monkeypatch, operation):
    """已接受历史更新在取消后仍完成提交。"""

    async def scenario():
        meta = metadata(tmp_path)
        store = await JsonlSessionStore.create(meta)
        session = Session(meta, store)
        await session.start_run("r1")
        entry_id = await session.append_user(user_message())
        result = compact(await session.get_history())
        entered, release = asyncio.Event(), asyncio.Event()
        original = store.append

        async def delayed(records):
            entered.set()
            await release.wait()
            await original(records)

        monkeypatch.setattr(store, "append", delayed)
        task = asyncio.create_task(
            session.commit_compaction(result)
            if operation == "compaction"
            else session.omit_failed_attempt([entry_id])
        )
        try:
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        expected = await session.get_history()
        assert expected["entry_ids"] == []
        assert expected["summary"] == ("summary" if operation == "compaction" else None)
        await session.close()
        reopened = Session(meta, await JsonlSessionStore.open(meta))
        try:
            assert await reopened.get_history() == expected
        finally:
            await reopened.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "invalid",
    [
        "duplicate",
        "missing",
        "reversed",
        "prefix",
        "excluded",
        "summarized",
        "future",
        "nonmessage",
        "exclusion",
    ],
)
def test_recovery_rejects_invalid_references(tmp_path, invalid):
    """恢复过程拒绝无效压缩及排除引用。"""

    async def scenario():
        meta = metadata(tmp_path)
        store = await JsonlSessionStore.create(meta)
        session = Session(meta, store)
        await session.start_run("r1")
        ids = [await session.append_user(user_message()) for _ in range(3)]
        retained = ids[1:]
        if invalid == "duplicate":
            retained = [ids[-1], ids[-1]]
        elif invalid == "missing":
            retained = ["missing"]
        elif invalid == "reversed":
            retained = ids[::-1]
        elif invalid == "prefix":
            retained = ids[:2]
        elif invalid == "excluded":
            await session.omit_failed_attempt([ids[1]])
        elif invalid == "summarized":
            await session.commit_compaction(compact(await session.get_history(), 1))
        elif invalid == "future":
            retained = ["future"]
        elif invalid == "nonmessage":
            retained = [(await store.read_records())[1]["id"]]
        record = {
            "type": "compaction",
            "id": "bad",
            "run_id": "r1",
            "timestamp": 125,
            "summary": "summary",
            "retained_entry_ids": retained,
            "tokens_before": 100,
            "estimated_tokens_after": 20,
            "usage": [],
        }
        if invalid == "exclusion":
            record = {
                "type": "history_exclusion",
                "id": "bad",
                "run_id": "r1",
                "timestamp": 125,
                "entry_ids": ["unknown"],
                "reason": "overflow_recovery",
            }
        await store.append([record])
        if invalid == "future":
            await store.append(
                [
                    {
                        "type": "user",
                        "id": "future",
                        "run_id": "r1",
                        "timestamp": 126,
                        "message": user_message(),
                    }
                ]
            )
        await session.close()
        before = Path(meta["path"]).read_bytes()
        reopened = Session(meta, await JsonlSessionStore.open(meta))
        try:
            with pytest.raises(ValueError):
                await reopened.state()
            assert Path(meta["path"]).read_bytes() == before
        finally:
            await reopened.close()

    asyncio.run(scenario())
