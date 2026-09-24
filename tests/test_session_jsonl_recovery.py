"""JSONL 尾部恢复、损坏位置及打开生命周期，覆盖逐字节截断和修复失败。"""

import asyncio
import json
import threading
from pathlib import Path

import pytest

from lhagent.harness.session.jsonl import JsonlSessionStore
from tests.test_session_jsonl import disk_records, metadata, record


async def seed(meta):
    """创建并关闭含一条消息的日志，返回恢复场景的原始记录。"""
    store = await JsonlSessionStore.create(meta)
    await store.append([record(1)])
    await store.close()
    return await store.read_records()


@pytest.mark.parametrize(
    "tail",
    [
        b"",
        b"{",
        b'{"id":',
        b'{"id":"unfinished',
        b'{"id":"\\',
        b'{"id":"\\u12',
        b'{"n":tru',
        b'{"n":-',
        b'{"n":1.',
        b'{"n":1e+',
        b'{"a":[{},',
        '{"id":"中文'.encode()[:-1],
        '{"id":"中文'.encode()[:-2],
    ],
)
def test_recover_tail_append_and_reopen(tmp_path, tail):
    """不完整尾部修复后可追加并再次打开。"""

    async def scenario():
        meta = metadata(tmp_path)
        expected = await seed(meta)
        path = Path(meta["path"])
        intact = path.read_bytes()
        path.write_bytes(intact + tail)
        store = await JsonlSessionStore.open(meta)
        try:
            assert path.read_bytes() == intact
            assert await store.read_records() == expected
            snapshot = await store.read_records()
            snapshot[0]["metadata"]["id"] = "changed"
            assert await store.read_records() == expected
            await asyncio.gather(*(store.append([record(i)]) for i in range(2, 8)))
        finally:
            await store.close()
        expected += [record(i) for i in range(2, 8)]
        reopened = await JsonlSessionStore.open(meta)
        try:
            assert await reopened.read_records() == disk_records(meta) == expected
        finally:
            await reopened.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("header_only", [False, True])
def test_complete_last_line_without_newline_is_preserved(tmp_path, header_only):
    """完整末行缺少换行时保留记录。"""

    async def scenario():
        meta = metadata(tmp_path)
        expected = await seed(meta)
        if header_only:
            expected = expected[:1]
        path = Path(meta["path"])
        content = b"\n".join(json.dumps(r, ensure_ascii=False).encode() for r in expected)
        path.write_bytes(content)
        store = await JsonlSessionStore.open(meta)
        try:
            assert path.read_bytes() == content + b"\n"
            assert await store.read_records() == expected
            await store.append([record(9)])
        finally:
            await store.close()
        assert disk_records(meta) == expected + [record(9)]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "bad",
    [
        b'{"id":oops}',
        b'{"id":oops',
        b'{"id":"\\q',
        b'{"id":"\\uZ',
        b'{"n":01',
        b'{"n":trueX',
        b'{"n":1e+ ',
        b'{"a":[1,]}',
        b"{}",
        b"[]",
        b"null",
        b" ",
        b"\xff",
        b'{"id":"\xff',
        b'{"id":\xe4\xb8',
        b'{"id":"line\t',
        b'{"n":NaN}',
        b'{"n":Infinity}',
        b'{"id":"a"}garbage',
    ],
)
@pytest.mark.parametrize("position", ["tail", "middle"])
def test_corruption_is_reported_without_modifying_file(tmp_path, bad, position):
    """已有损坏仅报告位置，不擅自修改日志。"""

    async def scenario():
        meta = metadata(tmp_path)
        await seed(meta)
        path = Path(meta["path"])
        original = path.read_bytes()
        content = original + bad
        if position == "middle":
            content += b"\n" + json.dumps(record(2)).encode() + b"\n"
        path.write_bytes(content)
        with pytest.raises(ValueError, match=f"line 3, byte {len(original)}"):
            await JsonlSessionStore.open(meta)
        assert path.read_bytes() == content

    asyncio.run(scenario())


@pytest.mark.parametrize("bad", [b"{", b'{"id":"\xe4\xb8', b""])
def test_terminated_partial_line_is_corruption(tmp_path, bad):
    """有终止换行的残缺记录属于损坏，不能当作未完成尾行修复。"""

    async def scenario():
        meta = metadata(tmp_path)
        await seed(meta)
        path = Path(meta["path"])
        content = path.read_bytes() + bad + b"\n"
        path.write_bytes(content)
        with pytest.raises(ValueError, match="line 3"):
            await JsonlSessionStore.open(meta)
        assert path.read_bytes() == content

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change",
    [
        {"type": "user"},
        {"id": "other"},
        {"run_id": "run"},
        {"timestamp": 124},
        {"metadata": {}},
        {"metadata": None},
        {"timestamp": True},
    ],
)
def test_invalid_header(tmp_path, change):
    """非法头记录不能恢复为有效会话。"""

    async def scenario():
        meta = metadata(tmp_path)
        records = await seed(meta)
        records[0].update(change)
        path = Path(meta["path"])
        content = b"\n".join(json.dumps(r).encode() for r in records) + b'\n{"tail":'
        path.write_bytes(content)
        with pytest.raises(ValueError, match="line 1, byte 0"):
            await JsonlSessionStore.open(meta)
        assert path.read_bytes() == content

    asyncio.run(scenario())


@pytest.mark.parametrize("content", [b"", b"{", b"[]\n", b"\xef\xbb\xbf{}\n"])
def test_missing_or_incomplete_header_is_not_recovered(tmp_path, content):
    """缺失或残缺头记录不会被静默重建。"""

    async def scenario():
        meta = metadata(tmp_path)
        path = Path(meta["path"])
        path.parent.mkdir()
        path.write_bytes(content)
        with pytest.raises(ValueError, match="line 1, byte 0"):
            await JsonlSessionStore.open(meta)
        assert path.read_bytes() == content

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value", [("id", "other"), ("created_at", 125), ("path", "elsewhere.jsonl")]
)
def test_metadata_must_match_header(tmp_path, field, value):
    """调用方元数据必须与日志头一致。"""

    async def scenario():
        meta = metadata(tmp_path)
        records = await seed(meta)
        records[0]["metadata"][field] = value
        path = Path(meta["path"])
        content = b"\n".join(json.dumps(r).encode() for r in records)
        path.write_bytes(content)
        with pytest.raises(ValueError, match="metadata mismatch"):
            await JsonlSessionStore.open(meta)
        assert path.read_bytes() == content

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change",
    [
        {"type": "unknown"},
        {"type": "session"},
        {"id": ""},
        {"id": 1},
        {"run_id": None},
        {"timestamp": True},
        {"timestamp": float("inf")},
        {"message": []},
    ],
)
def test_invalid_record_structure(tmp_path, change):
    """非法记录结构在恢复时拒绝。"""

    async def scenario():
        meta = metadata(tmp_path)
        await seed(meta)
        bad = record(2)
        bad.update(change)
        path = Path(meta["path"])
        content = path.read_bytes() + json.dumps(bad).encode()
        path.write_bytes(content)
        with pytest.raises(ValueError, match="line 3"):
            await JsonlSessionStore.open(meta)
        assert path.read_bytes() == content

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [False, True])
def test_cancel_open_waits_and_closes_without_deleting(tmp_path, monkeypatch, failure):
    """打开被取消后关闭句柄但不删除原日志。"""

    async def scenario():
        meta = metadata(tmp_path)
        expected = await seed(meta)
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        original_open = Path.open
        handles = []

        class BlockedFile:
            def __init__(self, file):
                self.file = file

            def __getattr__(self, name):
                return getattr(self.file, name)

            def __iter__(self):
                loop.call_soon_threadsafe(entered.set)
                if not release.wait(5):
                    raise AssertionError("open was not released")
                if failure:
                    raise OSError("read failed")
                return iter(self.file)

        def blocked_open(path, *args, **kwargs):
            file = original_open(path, *args, **kwargs)
            handles.append(file)
            return BlockedFile(file)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "open", blocked_open)
            task = asyncio.create_task(JsonlSessionStore.open(meta))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                for _ in range(3):
                    task.cancel()
                    await asyncio.sleep(0)
                    assert not task.done()
            finally:
                release.set()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            if failure:
                assert isinstance(caught.value.__cause__, OSError)
        assert handles[0].closed
        assert disk_records(meta) == expected

    asyncio.run(scenario())


def test_every_byte_cut_of_valid_nested_record(tmp_path):
    """对嵌套有效记录逐字节截断验证尾部判定。"""

    async def scenario():
        meta = metadata(tmp_path)
        expected = await seed(meta)
        path = Path(meta["path"])
        intact = path.read_bytes()
        item = record(2)
        item["extra"] = {
            "中文": [True, False, None, -12.5e-12, {}, [], 'quote" slash\\ tab\t newline\n', "雪"]
        }
        payload = json.dumps(item, ensure_ascii=False).encode()
        for cut in range(1, len(payload)):
            path.write_bytes(intact + payload[:cut])
            store = await JsonlSessionStore.open(meta)
            try:
                assert await store.read_records() == expected, cut
                assert path.read_bytes() == intact, cut
            finally:
                await store.close()

    asyncio.run(scenario())


def test_repair_write_failure_closes_file(tmp_path, monkeypatch):
    """修复写入失败也释放文件句柄。"""

    async def scenario():
        meta = metadata(tmp_path)
        await seed(meta)
        path = Path(meta["path"])
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
        stores = []

        def fail_write(store, payload):
            stores.append((store, store._file))
            raise OSError("repair failed")

        monkeypatch.setattr(JsonlSessionStore, "_write", fail_write)
        with pytest.raises(OSError, match="repair failed"):
            await JsonlSessionStore.open(meta)
        assert stores[0][1].closed
        assert stores[0][0]._file is None
        assert path.exists()

    asyncio.run(scenario())
