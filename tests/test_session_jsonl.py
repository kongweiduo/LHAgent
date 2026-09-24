"""JSONL 独占创建、串行批次提交、I/O 失败及关闭竞态。"""

import asyncio
import json
import threading
from pathlib import Path

import pytest

from lhagent.harness.session.jsonl import JsonlSessionStore
from tests.samples import user_record


def metadata(tmp_path):
    """生成嵌套临时路径元数据，用于验证父目录创建。"""
    return {
        "id": "session-1",
        "created_at": 123.5,
        "path": str(tmp_path / "nested" / "session.jsonl"),
    }


def record(index):
    """为共享用户记录样本替换独立条目 ID。"""
    item = user_record()
    item["id"] = f"entry-{index}"
    return item


def disk_records(meta):
    """直接读取磁盘日志并断言完整换行，校验真实写入结果。"""
    data = Path(meta["path"]).read_bytes()
    assert data.endswith(b"\n")
    return [json.loads(line) for line in data.splitlines()]


class ControlledFile:
    """可控 flush 屏障及真实文件故障，不以定时 sleep 决定竞争顺序。"""

    def __init__(self, file, *, failure=None, block=False):
        self.file = file
        self.failure = failure
        self.loop = asyncio.get_running_loop()
        self.entered = asyncio.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()
        self.writes = 0
        self.closes = 0

    def write(self, data):
        self.writes += 1
        if self.failure == "write":
            self.file.write(data[:5])
            raise OSError("write failed")
        if self.failure == "short":
            return self.file.write(data[:5])
        return self.file.write(data)

    def flush(self):
        self.loop.call_soon_threadsafe(self.entered.set)
        if not self.release.wait(5):
            raise AssertionError("test did not release flush")
        if self.failure == "flush":
            raise OSError("flush failed")
        self.file.flush()

    def close(self):
        self.closes += 1
        self.file.close()
        if self.failure == "close":
            raise OSError("close failed")


def test_constructor_has_no_io_and_open_missing_file_does_not_create(tmp_path):
    """构造不做 I/O，打开不存在的会话不创建文件。"""

    async def scenario():
        meta = metadata(tmp_path)
        store = JsonlSessionStore(meta)
        meta["id"] = "changed"
        assert store.metadata["id"] == "session-1"
        assert not Path(meta["path"]).parent.exists()
        with pytest.raises(RuntimeError, match="not been created"):
            await store.append([record(1)])
        with pytest.raises(RuntimeError, match="not been created"):
            await store.read_records()
        with pytest.raises(FileNotFoundError):
            await JsonlSessionStore.open(meta)
        await store.close()
        await store.close()
        assert not Path(meta["path"]).parent.exists()

    asyncio.run(scenario())


def test_create_header_exclusive_conflict_and_deep_copies(tmp_path):
    """独占创建保留头记录并隔离输入副本。"""

    async def scenario():
        meta = metadata(tmp_path)
        original = dict(meta)
        store = await JsonlSessionStore.create(meta)
        try:
            expected = {
                "type": "session",
                "id": meta["id"],
                "run_id": "",
                "timestamp": meta["created_at"],
                "metadata": original,
            }
            assert disk_records(meta) == [expected]
            with pytest.raises(FileExistsError):
                await JsonlSessionStore.create(meta)
            assert disk_records(meta) == [expected]
            meta["id"] = "mutated"
            copy = await store.read_records()
            copy[0]["metadata"]["id"] = "mutated again"
            copy.clear()
            assert await store.read_records() == [expected]
        finally:
            await store.close()
        assert await store.read_records() == [expected]
        with pytest.raises(RuntimeError, match="closed"):
            await store.append([])

    asyncio.run(scenario())


def test_parent_failure_and_concurrent_create(tmp_path):
    """父路径失败及并发创建不覆盖其他调用方文件。"""

    async def scenario():
        meta = metadata(tmp_path)
        Path(meta["path"]).parent.write_text("not a directory")
        with pytest.raises(FileExistsError):
            await JsonlSessionStore.create(meta)
        Path(meta["path"]).parent.unlink()
        results = await asyncio.gather(
            *(JsonlSessionStore.create(meta) for _ in range(4)), return_exceptions=True
        )
        stores = [r for r in results if isinstance(r, JsonlSessionStore)]
        assert len(stores) == 1
        assert sum(isinstance(r, FileExistsError) for r in results) == 3
        await stores[0].close()
        assert len(disk_records(meta)) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["write", "short", "flush"])
def test_create_failure_removes_owned_file(tmp_path, monkeypatch, failure):
    """创建写入失败仅删除本次创建的文件。"""

    async def scenario():
        meta = metadata(tmp_path)
        original_open = Path.open
        handles = []
        # Path.open runs in a worker, so create the controller on the event loop.
        controller = ControlledFile(None, failure=failure)

        def faulty_open(path, *args, **kwargs):
            controller.file = original_open(path, *args, **kwargs)
            handles.append(controller.file)
            return controller

        with monkeypatch.context() as patch:
            patch.setattr(Path, "open", faulty_open)
            with pytest.raises(OSError):
                await JsonlSessionStore.create(meta)
        assert not Path(meta["path"]).exists()
        assert handles[0].closed
        store = await JsonlSessionStore.create(meta)
        await store.close()

    asyncio.run(scenario())


def test_cancel_create_waits_then_removes_file(tmp_path, monkeypatch):
    """创建被取消时先等 I/O 完成再清理文件。"""

    async def scenario():
        meta = metadata(tmp_path)
        original_open = Path.open
        controller = ControlledFile(None, block=True)

        def blocked_open(path, *args, **kwargs):
            controller.file = original_open(path, *args, **kwargs)
            return controller

        with monkeypatch.context() as patch:
            patch.setattr(Path, "open", blocked_open)
            task = asyncio.create_task(JsonlSessionStore.create(meta))
            try:
                await asyncio.wait_for(controller.entered.wait(), 2)
                for _ in range(3):
                    task.cancel()
                    await asyncio.sleep(0)
                    assert not task.done()
            finally:
                controller.release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert controller.file.closed
        assert not Path(meta["path"]).exists()

    asyncio.run(scenario())


def test_fifo_batches_snapshot_flush_and_close_race(tmp_path):
    """批次按接受顺序提交快照，flush 和关闭竞争不漏写。"""

    async def scenario():
        meta = metadata(tmp_path)
        store = await JsonlSessionStore.create(meta)
        controller = ControlledFile(store._file, block=True)
        store._file = controller
        first = record(0)
        writes = [asyncio.create_task(store.append([first]))]
        try:
            await asyncio.wait_for(controller.entered.wait(), 2)
            first["id"] = "caller changed input"
            assert not writes[0].done()
            for index in range(1, 10):
                writes.append(
                    asyncio.create_task(store.append([record(index * 2), record(index * 2 + 1)]))
                )
            reader = asyncio.create_task(store.read_records())
            closer = asyncio.create_task(store.close())
            other_closer = asyncio.create_task(store.close())
            await asyncio.sleep(0)
            assert not closer.done()
            assert not reader.done()
            with pytest.raises(RuntimeError, match="closing"):
                await store.append([record(99)])
            for _ in range(3):
                closer.cancel()
                writes[0].cancel()
                await asyncio.sleep(0)
                assert not closer.done()
                assert not writes[0].done()
        finally:
            controller.release.set()
        with pytest.raises(asyncio.CancelledError):
            await writes[0]
        await asyncio.gather(*writes[1:])
        with pytest.raises(asyncio.CancelledError):
            await closer
        await other_closer
        await store.close()
        assert controller.file.closed
        assert controller.closes == 1
        expected = ["entry-0"] + [f"entry-{i}" for i in range(2, 20)]
        records = disk_records(meta)
        assert [r["id"] for r in records[1:]] == expected
        assert await reader == records
        assert await store.read_records() == records
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["write", "short", "flush"])
@pytest.mark.parametrize("cancel", [False, True])
def test_io_failure_stops_queued_writes_and_close_releases_file(tmp_path, failure, cancel):
    """I/O 失败阻止后续排队写入，关闭仍释放文件。"""

    async def scenario():
        store = await JsonlSessionStore.create(metadata(tmp_path))
        controller = ControlledFile(store._file, failure=failure)
        store._file = controller
        first = asyncio.create_task(store.append([record(1)]))
        second = asyncio.create_task(store.append([record(2)]))
        await asyncio.sleep(0)  # Both batches accepted before cancellation.
        if cancel:
            first.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else OSError) as caught:
            await first
        if cancel:
            assert isinstance(caught.value.__cause__, OSError)
        with pytest.raises(OSError):
            await second
        with pytest.raises(OSError):
            await store.append([record(3)])
        with pytest.raises(OSError):
            await store.read_records()
        for _ in range(2):
            with pytest.raises(OSError):
                await store.close()
        assert controller.writes == 1
        assert controller.closes == 1
        assert controller.file.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid", [float("nan"), object(), "\ud800"])
def test_encoding_failure_does_not_write_batch_or_poison_store(tmp_path, invalid):
    """编码失败不写入半批次，也不破坏后续存储能力。"""

    async def scenario():
        meta = metadata(tmp_path)
        store = await JsonlSessionStore.create(meta)
        bad = record(2)
        bad["id"] = invalid
        try:
            with pytest.raises((TypeError, ValueError, UnicodeEncodeError)):
                await store.append([record(1), bad])
            assert len(disk_records(meta)) == 1
            good = record(3)
            good["id"] = "中文\n下一行"
            await store.append(iter([good]))
            await store.append([])
            assert disk_records(meta)[1:] == [good]
        finally:
            await store.close()

    asyncio.run(scenario())


def test_close_failure_is_shared_and_releases_handle(tmp_path):
    """重复关闭共享失败结果且文件句柄被释放。"""

    async def scenario():
        store = await JsonlSessionStore.create(metadata(tmp_path))
        controller = ControlledFile(store._file, failure="close")
        store._file = controller
        for _ in range(2):
            with pytest.raises(OSError, match="close failed"):
                await store.close()
        assert controller.file.closed
        assert controller.closes == 1
        with pytest.raises(RuntimeError, match="closed"):
            await store.append([record(1)])

    asyncio.run(scenario())
