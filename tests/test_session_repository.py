"""会话发现、身份冲突、路径规则及仓库资源所有权。"""

import asyncio
import json
import threading
from pathlib import Path

import pytest

from lhagent.harness.session import JsonlSessionStore, Session, SessionRepository
from lhagent.harness.session import repository as module
from tests.samples import user_message


def test_roundtrip_and_ownership(tmp_path):
    """仓库创建/打开往返保持会话身份及资源归属。"""

    async def scenario():
        repo = SessionRepository({"directory": str(tmp_path / "sessions")})
        assert not (tmp_path / "sessions").exists()
        assert await repo.list() == []
        session = await repo.create()
        metadata = session.metadata
        assert len(metadata["id"]) == 32
        assert (await session.state())["state"] == "new"
        assert await repo.list() == [metadata]
        await session.start_run("interrupted-run")
        entry = await session.append_user(user_message())
        with pytest.raises(RuntimeError, match="already open"):
            await repo.open(metadata)
        with pytest.raises(RuntimeError, match="open session"):
            await repo.delete(metadata)
        await session.close()
        assert session.closed
        reopened = await repo.open(metadata)
        assert (await reopened.state())["state"] == "interrupted"
        assert (await reopened.get_history())["entry_ids"] == [entry]
        await reopened.close()
        await repo.delete(metadata)
        assert await repo.list() == []
        with pytest.raises(FileNotFoundError):
            await repo.delete(metadata)
        await repo.create({"id": metadata["id"]})
        await asyncio.gather(repo.close(), repo.close())
        await repo.close()
        for operation in (repo.create(), repo.open(metadata), repo.list(), repo.delete(metadata)):
            with pytest.raises(RuntimeError, match="closed"):
                await operation

    asyncio.run(scenario())


def test_paths_filters_sorting_and_copies(tmp_path, monkeypatch):
    """路径过滤、排序及返回副本遵守发现接口。"""
    monkeypatch.setattr(module.time, "time", lambda: 100.0)

    async def scenario():
        root = tmp_path / "sessions"
        repo = SessionRepository({"directory": str(root)})
        try:
            b = await repo.create({"id": "b"})
            a = await repo.create({"id": "a", "path": "custom.jsonl"})
            nested = await repo.create({"id": "nested", "path": "sub/n.jsonl"})
            external = await repo.create(
                {"id": "external", "path": str(tmp_path / "elsewhere/e.jsonl")}
            )
            monkeypatch.setattr(module.time, "time", lambda: 101.0)
            newest = await repo.create({"id": "new"})
            (root / "ignore.txt").write_text("not JSON")
            (root / "directory.jsonl").mkdir()
            (root / "alias.jsonl").symlink_to(Path(a.metadata["path"]))
            expected = [newest.metadata, a.metadata, b.metadata]
            assert await repo.list() == expected
            listed = await repo.list()
            listed[0]["id"] = "mutated"
            assert await repo.list() == expected
            assert await repo.list({"directory": str(root / "sub")}) == [nested.metadata]
            assert await repo.list({"directory": str(tmp_path / "elsewhere")}) == [
                external.metadata
            ]
            with pytest.raises(FileExistsError):
                await repo.create({"id": "external", "path": "another.jsonl"})
        finally:
            await repo.close()

    asyncio.run(scenario())


def test_default_directory_is_lazy_and_fixed(tmp_path, monkeypatch):
    """默认目录在构造时惰性解析，之后不随工作目录变化。"""
    monkeypatch.chdir(tmp_path)
    repo = SessionRepository({})
    assert repo._directory == tmp_path / ".lhagent/sessions"
    assert not repo._directory.exists()
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    assert repo._directory == tmp_path / ".lhagent/sessions"
    asyncio.run(repo.close())


@pytest.mark.parametrize(
    "options",
    [{"id": "../escape"}, {"id": ""}, {"id": "中文"}, {"path": ""}, {"path": "session.txt"}],
)
def test_invalid_create_options(tmp_path, options):
    """非法创建选项在产生文件前拒绝。"""

    async def scenario():
        repo = SessionRepository({"directory": str(tmp_path)})
        with pytest.raises(ValueError):
            await repo.create(options)
        assert list(tmp_path.iterdir()) == []
        await repo.close()

    asyncio.run(scenario())


def test_conflicts_survive_repository_restart(tmp_path):
    """仓库重建后仍能通过磁盘头记录检测身份冲突。"""

    async def scenario():
        repo = SessionRepository({"directory": str(tmp_path)})
        session = await repo.create({"id": "same", "path": "custom.jsonl"})
        metadata = session.metadata
        await repo.close()
        repo = SessionRepository({"directory": str(tmp_path)})
        try:
            with pytest.raises(FileExistsError):
                await repo.create({"id": "same", "path": "new.jsonl"})
            with pytest.raises(FileExistsError):
                await repo.create({"id": "different", "path": "custom.jsonl"})
            with pytest.raises(ValueError):
                await repo.open({**metadata, "created_at": metadata["created_at"] + 1})
            with pytest.raises(ValueError):
                await repo.delete({**metadata, "created_at": 0.0})
            assert Path(metadata["path"]).exists()
            opened = await asyncio.gather(
                repo.open(metadata), repo.open(metadata), return_exceptions=True
            )
            assert sum(isinstance(item, RuntimeError) for item in opened) == 1
        finally:
            await repo.close()

    asyncio.run(scenario())


def test_list_is_readonly_and_open_recovers_tail(tmp_path):
    """列举只读，尾部修复仅在打开时进行。"""

    async def scenario():
        repo = SessionRepository({"directory": str(tmp_path)})
        session = await repo.create()
        metadata = session.metadata
        await session.close()
        path = Path(metadata["path"])
        original = path.read_bytes()
        path.write_bytes(original + b'{"type":')
        assert await repo.list() == [metadata]
        assert path.read_bytes() == original + b'{"type":'
        recovered = await repo.open(metadata)
        assert path.read_bytes() == original
        await recovered.close()
        path.write_bytes(original + b"not JSON\n")
        assert await repo.list() == [metadata]
        with pytest.raises(ValueError, match="line 2"):
            await repo.open(metadata)
        # Deletion only needs the identity header, so corrupt bodies can be removed.
        await repo.delete(metadata)
        await repo.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("body", [b"", b"{}\n", b"\xff\n", b"[]\n", b'{"metadata":null}\n'])
def test_invalid_headers_fail_listing(tmp_path, body):
    """坏头使列举失败，不能静默隐藏损坏会话。"""
    (tmp_path / "bad.jsonl").write_bytes(body)

    async def scenario():
        repo = SessionRepository({"directory": str(tmp_path)})
        with pytest.raises(ValueError, match="line 1, byte 0"):
            await repo.list()
        await repo.close()

    asyncio.run(scenario())


def test_duplicate_ids_and_moved_files(tmp_path):
    """重复身份及移动后路径不匹配均被识别。"""

    async def scenario():
        repo = SessionRepository({"directory": str(tmp_path)})
        session = await repo.create({"id": "same"})
        metadata = session.metadata
        await session.close()
        other = {**metadata, "path": str(tmp_path / "other.jsonl")}
        store = await JsonlSessionStore.create(other)
        await store.close()
        with pytest.raises(ValueError, match="duplicate session id"):
            await repo.list()
        Path(other["path"]).unlink()
        Path(metadata["path"]).rename(other["path"])
        with pytest.raises(ValueError, match="path mismatch"):
            await repo.list()
        await repo.close()

    asyncio.run(scenario())


def test_failed_session_load_releases_store(tmp_path, monkeypatch):
    """会话投影加载失败会关闭刚打开的存储。"""

    async def scenario():
        repo = SessionRepository({"directory": str(tmp_path)})
        session = await repo.create()
        meta = session.metadata
        await session.close()
        with Path(meta["path"]).open("a") as file:
            file.write(
                json.dumps(
                    {
                        "type": "run_finish",
                        "id": "end",
                        "run_id": "unknown",
                        "timestamp": 1.0,
                        "status": "completed",
                        "error": None,
                    }
                )
                + "\n"
            )
        stores = []
        original = JsonlSessionStore.open.__func__

        async def capture(cls, metadata):
            store = await original(cls, metadata)
            stores.append(store)
            return store

        monkeypatch.setattr(JsonlSessionStore, "open", classmethod(capture))
        with pytest.raises(ValueError, match="invalid run finish"):
            await repo.open(meta)
        assert stores[0]._file is None
        await repo.close()

    asyncio.run(scenario())


def test_cancelled_creation_cleans_file_and_allows_retry(tmp_path, monkeypatch):
    """取消创建清理本次文件，之后可重试同一身份。"""

    async def scenario():
        repo = SessionRepository({"directory": str(tmp_path)})
        entered, release = threading.Event(), threading.Event()
        original = JsonlSessionStore._create_file

        def blocked(store, payload):
            original(store, payload)
            entered.set()
            assert release.wait(5)

        monkeypatch.setattr(JsonlSessionStore, "_create_file", blocked)
        task = asyncio.create_task(repo.create({"id": "cancelled"}))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not (tmp_path / "cancelled.jsonl").exists()
        assert not repo._sessions
        await repo.create({"id": "cancelled"})
        await repo.close()

    asyncio.run(scenario())


def test_cancelled_open_releases_handle(tmp_path, monkeypatch):
    """取消打开不遗留句柄。"""

    async def scenario():
        repo = SessionRepository({"directory": str(tmp_path)})
        session = await repo.create()
        metadata = session.metadata
        await session.close()
        entered = asyncio.Event()
        sessions = []
        original = Session.state

        async def blocked(self):
            sessions.append(self)
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(Session, "state", blocked)
        opening = asyncio.create_task(repo.open(metadata))
        await entered.wait()
        opening.cancel()
        with pytest.raises(asyncio.CancelledError):
            await opening
        assert sessions[0]._store._file is None
        monkeypatch.setattr(Session, "state", original)
        await repo.open(metadata)
        await repo.close()

    asyncio.run(scenario())


def test_concurrent_create_does_not_overwrite(tmp_path):
    """并发创建同一身份不会覆盖日志。"""

    async def scenario():
        repo = SessionRepository({"directory": str(tmp_path)})
        results = await asyncio.gather(
            repo.create({"id": "same"}), repo.create({"id": "same"}), return_exceptions=True
        )
        assert sum(isinstance(result, FileExistsError) for result in results) == 1
        assert len(await repo.list()) == 1
        await repo.close()

    asyncio.run(scenario())


def test_close_waits_for_create_and_closes_every_handle_on_error(tmp_path, monkeypatch):
    """关闭等待创建落定；部分关闭失败也尝试释放其余会话。"""

    async def scenario():
        repo = SessionRepository({"directory": str(tmp_path)})
        first = await repo.create({"id": "first"})
        entered, release = asyncio.Event(), asyncio.Event()
        original = JsonlSessionStore.create.__func__

        async def blocked(cls, metadata):
            entered.set()
            await release.wait()
            return await original(cls, metadata)

        monkeypatch.setattr(JsonlSessionStore, "create", classmethod(blocked))
        creating = asyncio.create_task(repo.create({"id": "second"}))
        await entered.wait()
        original_close = first._store.close

        async def failing_close():
            await original_close()
            raise OSError("close failed")

        monkeypatch.setattr(first._store, "close", failing_close)
        closing = asyncio.create_task(repo.close())
        await asyncio.sleep(0)
        closing.cancel()
        release.set()
        second = await creating
        with pytest.raises(asyncio.CancelledError) as error:
            await closing
        assert isinstance(error.value.__cause__, ExceptionGroup)
        assert first.closed and second.closed
        assert first._store._file is None and second._store._file is None
        with pytest.raises(ExceptionGroup):
            await repo.close()

    asyncio.run(scenario())
