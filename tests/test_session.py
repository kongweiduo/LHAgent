"""会话历史、终态校验和运行生命周期，验证取消不能撤销已接受写入。"""

import asyncio
import json
from copy import deepcopy

import pytest

from lhagent.harness.session.jsonl import JsonlSessionStore
from lhagent.harness.session.session import Session
from tests.samples import mixed_result, user_message


def metadata(tmp_path):
    """生成指向临时文件的稳定会话身份。"""
    return {"id": "s1", "created_at": 123.5, "path": str(tmp_path / "s1.jsonl")}


def loop_result(status="completed"):
    """构造不含最后模型响应的循环终态样本。"""
    return {
        "status": status,
        "last_response": None,
        "error": "stopped" if status == "error" else None,
    }


def test_history_roundtrip_and_terminal_states(tmp_path):
    """历史往返保留各类终态及消息内容。"""

    async def scenario():
        meta = metadata(tmp_path)
        session = Session(meta, await JsonlSessionStore.create(meta))
        meta["id"] = "mutated"
        assert session.metadata["id"] == "s1"
        assert (await session.state())["state"] == "new"
        with pytest.raises(RuntimeError, match="no active run"):
            await session.finish_run(loop_result())
        with pytest.raises(RuntimeError, match="no active run"):
            await session.append_user(user_message())
        await session.start_run("run-1")
        with pytest.raises(RuntimeError, match="already active"):
            await session.start_run("run-2")
        first = await session.append_user(user_message())
        second = await session.append_user(user_message())
        assert first != second
        response = mixed_result(complete_tool=False)
        response["finish_reason"] = "error"
        response["error"] = "partial"
        response["error_kind"] = "context_overflow"
        response_id = await session.append_response(response)
        response["content"].clear()
        result = {
            "call_id": "tool-call-1",
            "name": "read",
            "status": "timeout",
            "output": None,
            "error": "timed out",
        }
        tool_id = await session.append_tool_result(result)
        result["error"] = "mutated"
        history = await session.get_history()
        assert history["entry_ids"] == [first, second, response_id, tool_id]
        assert history["messages"][0] == history["messages"][1]
        assert history["messages"][2]["content"][2]["complete"] is False
        assert history["messages"][2]["finish_reason"] == "error"
        assert history["messages"][3]["tool_call_id"] == "tool-call-1"
        history["messages"][2]["content"].clear()
        history["entry_ids"].clear()
        assert len((await session.get_history())["messages"][2]["content"]) == 3
        await session.finish_run(loop_result("error"))
        assert await session.state() == {
            "state": "idle",
            "active_run_id": None,
            "last_finished_run_id": "run-1",
            "interrupted": False,
        }
        assert await session.close() == await session.close()
        assert (await session.state())["state"] == "closed"
        with pytest.raises(RuntimeError, match="closed"):
            await session.start_run("run-2")
        reopened = Session(session.metadata, await JsonlSessionStore.open(session.metadata))
        try:
            assert (await reopened.get_history())["entry_ids"] == [
                first,
                second,
                response_id,
                tool_id,
            ]
            records = [
                json.loads(line) for line in open(session.metadata["path"], encoding="utf-8")
            ]
            assert records[4]["response"]["error_kind"] == "context_overflow"
            assert records[-1]["status"] == "error"
            assert records[-1]["run_id"] == "run-1"
            assert (await reopened.state())["state"] == "idle"
            with pytest.raises(ValueError, match="duplicate run_id"):
                await reopened.start_run("run-1")
        finally:
            await reopened.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("reason", ["stop", "tool_call", "length", "error", "cancelled"])
def test_terminal_validation(tmp_path, reason):
    """不符合终态契约的响应或工具结果不能写入。"""

    async def scenario():
        meta = metadata(tmp_path)
        session = Session(meta, await JsonlSessionStore.create(meta))
        try:
            await session.start_run("run-1")
            response = mixed_result()
            response["finish_reason"] = reason
            response["error_kind"] = "other" if reason == "error" else None
            await session.append_response(response)
            for invalid in ("pending", "delta"):
                bad = deepcopy(response)
                bad["finish_reason"] = invalid
                with pytest.raises(ValueError):
                    await session.append_response(bad)
            bad = deepcopy(response)
            bad["error_kind"] = None if reason == "error" else "other"
            with pytest.raises(ValueError):
                await session.append_response(bad)
            assert len((await session.get_history())["entry_ids"]) == 1
        finally:
            await session.close()

    asyncio.run(scenario())


def test_interrupted_run_survives_new_run(tmp_path):
    """恢复发现的中断标记不会被新运行悄悄抹掉。"""

    async def scenario():
        meta = metadata(tmp_path)
        first = Session(meta, await JsonlSessionStore.create(meta))
        await first.start_run("old")
        await first.append_user(user_message())
        await first.close()
        reopened = Session(meta, await JsonlSessionStore.open(meta))
        try:
            assert await reopened.state() == {
                "state": "interrupted",
                "active_run_id": None,
                "last_finished_run_id": None,
                "interrupted": True,
            }
            with pytest.raises(ValueError, match="duplicate run_id"):
                await reopened.start_run("old")
            await reopened.start_run("new")
            assert (await reopened.state())["state"] == "active"
            await reopened.finish_run(loop_result("cancelled"))
            assert await reopened.state() == {
                "state": "idle",
                "active_run_id": None,
                "last_finished_run_id": "new",
                "interrupted": True,
            }
            records = await reopened._store.read_records()
            assert [r["run_id"] for r in records if r["type"] == "run_finish"] == ["new"]
        finally:
            await reopened.close()
        again = Session(meta, await JsonlSessionStore.open(meta))
        try:
            assert await again.state() == {
                "state": "idle",
                "active_run_id": None,
                "last_finished_run_id": "new",
                "interrupted": True,
            }
        finally:
            await again.close()

    asyncio.run(scenario())


def test_start_finish_failure_does_not_claim_success(tmp_path, monkeypatch):
    """开始或结束写入失败时不虚报状态成功。"""

    async def scenario():
        meta = metadata(tmp_path)
        store = await JsonlSessionStore.create(meta)
        session = Session(meta, store)
        original = store.append

        async def fail(records):
            raise OSError("disk unavailable")

        try:
            monkeypatch.setattr(store, "append", fail)
            with pytest.raises(OSError):
                await session.start_run("r1")
            assert (await session.state())["state"] == "new"
            monkeypatch.setattr(store, "append", original)
            await session.start_run("r1")
            monkeypatch.setattr(store, "append", fail)
            with pytest.raises(OSError):
                await session.finish_run(loop_result())
            assert (await session.state())["active_run_id"] == "r1"
            monkeypatch.setattr(store, "append", original)
            await session.finish_run(loop_result())
            assert (await session.state())["state"] == "idle"
        finally:
            await session.close()

    asyncio.run(scenario())


def test_invalid_recovered_terminal_is_rejected(tmp_path):
    """恢复时拒绝非法终态记录。"""

    async def scenario():
        meta = metadata(tmp_path)
        store = await JsonlSessionStore.create(meta)
        await store.append(
            [
                {"type": "run_start", "id": "start", "run_id": "r1", "timestamp": 124.0},
                {
                    "type": "tool_result",
                    "id": "tool",
                    "run_id": "r1",
                    "timestamp": 125.0,
                    "result": {
                        "call_id": "c",
                        "name": "read",
                        "status": "pending",
                        "output": None,
                        "error": None,
                    },
                },
            ]
        )
        await store.close()
        session = Session(meta, await JsonlSessionStore.open(meta))
        try:
            with pytest.raises(ValueError, match="nonterminal tool result"):
                await session.get_history()
        finally:
            await session.close()

    asyncio.run(scenario())


def test_cancel_accepted_write_settles_before_return(tmp_path, monkeypatch):
    """已接受写入在调用取消返回前完成落定。"""

    async def scenario():
        meta = metadata(tmp_path)
        store = await JsonlSessionStore.create(meta)
        session = Session(meta, store)
        entered, release = asyncio.Event(), asyncio.Event()
        original = store.append

        async def delayed(records):
            entered.set()
            await release.wait()
            await original(records)

        monkeypatch.setattr(store, "append", delayed)
        task = asyncio.create_task(session.start_run("r1"))
        try:
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await session.state())["active_run_id"] == "r1"
        await session.close()

    asyncio.run(scenario())
