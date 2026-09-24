"""会话 JSONL 的创建、恢复及串行写入边界。

单个事件循环内按接受顺序串行提交。成功表示 UTF-8 完整行写入并 flush，
不执行 fsync，不承诺掉电持久性。调用方取消不能撤销已经接受的 I/O。
"""

import asyncio
import json
import math
import re
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import BinaryIO, cast

from .types import JsonlRecord, SessionMetadata, SessionRecord


async def _settle[T](task: asyncio.Task[T]) -> T:
    """等待真实结果后传播取消；重复取消不能提前放弃资源操作。"""
    cancellation = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
        except Exception:
            break
    try:
        result = task.result()
    except Exception as exc:
        if cancellation is not None:
            raise cancellation from exc
        raise
    if cancellation is not None:
        raise cancellation
    return result


def _encode(records: Iterable[JsonlRecord]) -> tuple[bytes, list[JsonlRecord]]:
    """在接受批次之前完整编码，同时隔离调用方的可变对象。"""
    lines = []
    snapshot = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("JSONL records must be objects")
        line = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        lines.append((line + "\n").encode("utf-8"))
        snapshot.append(cast(JsonlRecord, json.loads(line)))
    return b"".join(lines), snapshot


def _reject_constant(value: str) -> None:
    """拒绝非标准 JSON 的非有限数字常量。"""
    raise ValueError(f"invalid JSON constant: {value}")


def _incomplete_object(text: str) -> bool:
    """仅接受合法对象的严格前缀；语法错误不能当作掉电残片丢弃。"""
    position = 0

    class Incomplete(Exception):
        """内部解析信号：输入在合法语法前缀处耗尽。"""

        pass

    def peek() -> str:
        """跳过 JSON 空白并读取下一字符，耗尽时标记尚未完成。"""
        nonlocal position
        while position < len(text) and text[position] in " \t\r\n":
            position += 1
        if position == len(text):
            raise Incomplete
        return text[position]

    def string() -> None:
        """校验字符串和转义；严格区分截断与已出现的语法错误。"""
        nonlocal position
        if peek() != '"':
            raise ValueError
        position += 1
        while position < len(text):
            char = text[position]
            position += 1
            if char == '"':
                return
            if ord(char) < 32:
                raise ValueError
            if char == "\\":
                if position == len(text):
                    raise Incomplete
                escape = text[position]
                position += 1
                if escape == "u":
                    for _ in range(4):
                        if position == len(text):
                            raise Incomplete
                        if text[position] not in "0123456789abcdefABCDEF":
                            raise ValueError
                        position += 1
                elif escape not in '"\\/bfnrt':
                    raise ValueError
        raise Incomplete

    def value() -> None:
        """递归消费 JSON 值，只有合法但不完整的前缀允许尾部修复。"""
        nonlocal position
        char = peek()
        if char == '"':
            string()
        elif char in "{[":
            position += 1
            closing = "}" if char == "{" else "]"
            if peek() == closing:
                position += 1
                return
            while True:
                if char == "{":
                    string()
                    if peek() != ":":
                        raise ValueError
                    position += 1
                value()
                separator = peek()
                position += 1
                if separator == closing:
                    return
                if separator != ",":
                    raise ValueError
        elif char in "tfn":
            literal = {"t": "true", "f": "false", "n": "null"}[char]
            for expected in literal:
                if position == len(text):
                    raise Incomplete
                if text[position] != expected:
                    raise ValueError
                position += 1
        elif char in "-0123456789":
            start = position
            while position < len(text) and text[position] not in " \t\r\n,]}":
                position += 1
            token = text[start:position]
            if re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", token):
                return
            if position == len(text) and (
                token == "-"
                or re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.|(?:\.[0-9]+)?[eE][+-]?)", token)
            ):
                raise Incomplete
            raise ValueError
        else:
            raise ValueError

    try:
        if peek() != "{":
            return False
        value()
    except Incomplete:
        return bool(text.strip())
    except (ValueError, RecursionError):
        return False
    return False


def _finite_number(value: object) -> bool:
    """判断时间戳等字段是否为有限数字，拒绝 bool。"""
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _validate_record(item: object, metadata: SessionMetadata, *, first: bool) -> None:
    """检查记录外壳及头部身份；消息语义和运行关系由上层处理。"""
    if not isinstance(item, dict):
        raise ValueError("record must be an object")
    for field in ("type", "id", "run_id"):
        if not isinstance(item.get(field), str):
            raise ValueError(f"record {field} must be a string")
    if not item["id"] or not _finite_number(item.get("timestamp")):
        raise ValueError("record requires a nonempty id and finite timestamp")
    kind = item["type"]
    if first:
        header = item.get("metadata")
        if (
            kind != "session"
            or item["run_id"] != ""
            or not isinstance(header, dict)
            or not isinstance(header.get("id"), str)
            or not isinstance(header.get("path"), str)
            or not _finite_number(header.get("created_at"))
        ):
            raise ValueError("invalid session header")
        if (
            header != metadata
            or item["id"] != header["id"]
            or item["timestamp"] != header["created_at"]
        ):
            raise ValueError("session header metadata mismatch")
        return
    payloads: dict[str, dict[str, object]] = {
        "run_start": {},
        "user": {"message": dict},
        "assistant": {"response": dict},
        "tool_result": {"result": dict},
        "history_exclusion": {"entry_ids": list, "reason": str},
        "compaction": {
            "summary": str,
            "retained_entry_ids": list,
            "tokens_before": int,
            "estimated_tokens_after": int,
            "usage": list,
        },
        "run_finish": {"status": str, "error": (str, type(None))},
    }
    if kind not in payloads:
        raise ValueError(f"unexpected record type: {kind!r}")
    for field, expected in payloads[kind].items():
        if field not in item or not isinstance(item[field], expected):  # type: ignore[arg-type]
            raise ValueError(f"invalid {kind}.{field}")


class JsonlSessionStore:
    """单个会话的追加式文件；业务语义由 Session 和 Repository 校验。"""

    def __init__(self, metadata: SessionMetadata) -> None:
        """复制元数据，不打开文件、不创建目录；使用 create/open 激活存储。"""
        self.metadata = deepcopy(metadata)
        self._path = Path(metadata["path"])
        self._file: BinaryIO | None = None
        self._records: list[JsonlRecord] = []
        self._tail: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._ready = False

    @classmethod
    async def create(cls, metadata: SessionMetadata) -> "JsonlSessionStore":
        """独占创建并 flush session 头；失败或取消时关闭并删除本次新文件。"""
        store = cls(metadata)
        header: SessionRecord = {
            "type": "session",
            "id": store.metadata["id"],
            "run_id": "",
            "timestamp": store.metadata["created_at"],
            "metadata": deepcopy(store.metadata),
        }
        payload, snapshot = _encode([header])
        task = asyncio.create_task(asyncio.to_thread(store._create_file, payload))
        try:
            await _settle(task)
        except asyncio.CancelledError:
            if not task.cancelled() and task.exception() is None:
                await _settle(asyncio.create_task(asyncio.to_thread(store._discard_file)))
            raise
        store._records = snapshot
        store._ready = True
        return store

    def _create_file(self, payload: bytes) -> None:
        """独占创建文件并提交头记录；失败只清理本次创建的文件。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # 独占打开失败时绝不能清理其他调用方已经拥有的文件。
        self._file = self._path.open("xb")
        try:
            self._write(payload)
        except BaseException:
            self._discard_file()
            raise

    def _discard_file(self) -> None:
        """关闭并删除本次创建的文件，不删除已存在的会话。"""
        try:
            if self._file is not None:
                self._file.close()
        finally:
            self._file = None
            self._path.unlink()

    @classmethod
    async def open(cls, metadata: SessionMetadata) -> "JsonlSessionStore":
        """校验已有记录后修复尾部；失败或取消均关闭文件，不删除原日志。"""
        store = cls(metadata)
        task = asyncio.create_task(asyncio.to_thread(store._open_file))
        try:
            await _settle(task)
        except asyncio.CancelledError:
            if not task.cancelled() and task.exception() is None:
                await store.close()
            raise
        store._ready = True
        return store

    def _open_file(self) -> None:
        """校验已有日志，仅修复可确认的未完成尾行；失败关闭句柄。"""
        self._file = self._path.open("r+b")
        try:
            records: list[JsonlRecord] = []
            offset = 0
            repair_at = None
            add_newline = False
            for line_number, raw in enumerate(self._file, 1):
                terminated = raw.endswith(b"\n")
                try:
                    try:
                        text = raw.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        if (
                            not terminated
                            and records
                            and exc.reason == "unexpected end of data"
                            and exc.end == len(raw)
                            and _incomplete_object(raw[: exc.start].decode("utf-8") + "中")
                        ):
                            repair_at = offset
                            break
                        raise ValueError("invalid UTF-8") from exc
                    try:
                        item = json.loads(text, parse_constant=_reject_constant)
                    except json.JSONDecodeError:
                        if not terminated and records and _incomplete_object(text):
                            repair_at = offset
                            break
                        raise
                    _validate_record(item, self.metadata, first=not records)
                except ValueError as exc:
                    raise ValueError(
                        f"{self._path}: line {line_number}, byte {offset}: {exc}"
                    ) from exc
                records.append(cast(JsonlRecord, item))
                offset += len(raw)
                add_newline = not terminated
            if not records:
                raise ValueError(f"{self._path}: line 1, byte 0: missing session header")
            # 先校验全部内容再修改，格式错误不会改写原文件。
            if repair_at is not None:
                self._file.seek(repair_at)
                self._file.truncate()
                self._file.flush()
            elif add_newline:
                self._write(b"\n")
            self._records = records
        except BaseException:
            try:
                self._file.close()
            finally:
                self._file = None
            raise

    def _require_ready(self) -> None:
        """拒绝未打开、正在关闭或已经关闭的存储操作。"""
        if not self._ready:
            raise RuntimeError("JSONL store has not been created")

    async def read_records(self) -> list[JsonlRecord]:
        """等待此前接受的追加，返回含 session 头的深副本；关闭后仍可读取。"""
        self._require_ready()
        tail = self._tail
        if tail is not None:
            await _settle(tail)
        return deepcopy(self._records)

    async def append(self, records: Iterable[JsonlRecord]) -> None:
        """整批预编码后入队，按接受顺序写入并 flush。

        编码失败不写入且不破坏存储。I/O 失败后后续批次传播该失败，不再写入；
        失败批次可能已部分落盘，不承诺批次原子性。取消等待真实写入结束再传播，
        写入同时失败时通过 CancelledError.__cause__ 交付错误。
        """
        self._require_ready()
        if self._close_task is not None:
            raise RuntimeError("JSONL store is closing or closed")
        payload, snapshot = _encode(records)
        previous = self._tail
        task = asyncio.create_task(self._append_after(previous, payload, snapshot))
        self._tail = task
        await _settle(task)

    async def _append_after(
        self,
        previous: asyncio.Task[None] | None,
        payload: bytes,
        snapshot: list[JsonlRecord],
    ) -> None:
        """等待前序提交后写入当前批次；写入失败使后续提交失败。"""
        if previous is not None:
            await previous
        if payload:
            await asyncio.to_thread(self._write, payload)
            self._records.extend(snapshot)

    def _write(self, payload: bytes) -> None:
        """写入完整编码批次并 flush；不执行 fsync。"""
        assert self._file is not None
        written = self._file.write(payload)
        if written != len(payload):
            raise OSError(f"Short JSONL write: {written}/{len(payload)} bytes")
        self._file.flush()

    async def close(self) -> None:
        """停止接受追加，等待已接受批次并关闭；并发/重复调用共享关闭结果。"""
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._finish_close())
        await _settle(self._close_task)

    async def _finish_close(self) -> None:
        """等待已接受的写入后释放句柄，并传播写入或关闭错误。"""
        try:
            if self._tail is not None:
                await self._tail
        finally:
            if self._file is not None:
                try:
                    await asyncio.to_thread(self._file.close)
                finally:
                    self._file = None
