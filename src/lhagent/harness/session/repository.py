"""多个线性 JSONL 会话的发现与资源所有权边界。"""

import asyncio
import json
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import cast
from uuid import uuid4

from .jsonl import JsonlSessionStore, _finite_number, _reject_constant, _settle, _validate_record
from .session import Session
from .types import (
    SessionCreateOptions,
    SessionListOptions,
    SessionMetadata,
    SessionRepositoryOptions,
)

# 只保存默认路径文本；展开用户目录仍在构造仓库时进行。
_DEFAULT_DIRECTORY = ".lhagent/sessions"


def _session_id(value: object) -> str:
    """校验会话标识可安全作为单个文件名使用。"""
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise ValueError("session id must contain only ASCII letters, digits, '_' or '-'")
    return value


def _path(value: str) -> Path:
    """解析仓库路径；相对路径基于调用时工作目录。"""
    if not isinstance(value, str) or not value:
        raise ValueError("path must be a nonempty string")
    return Path(value).expanduser().resolve()


def _metadata(value: SessionMetadata) -> SessionMetadata:
    """校验会话元数据的身份、路径和创建时间。"""
    result = deepcopy(value)
    if not isinstance(result, dict) or set(result) != {"id", "created_at", "path"}:
        raise ValueError("invalid session metadata")
    _session_id(result["id"])
    if not _finite_number(result["created_at"]):
        raise ValueError("created_at must be a finite timestamp")
    if str(_path(result["path"])) != result["path"]:
        raise ValueError("session metadata path must be canonical and absolute")
    return result


def _read_metadata(path: Path) -> SessionMetadata:
    """只读取并校验首行元数据，不恢复或改写日志。"""
    try:
        with path.open("rb") as file:
            text = file.readline().decode("utf-8")
        if text.startswith("\ufeff"):
            raise ValueError("unexpected BOM")
        header = json.loads(text, parse_constant=_reject_constant)
        if not isinstance(header, dict):
            raise ValueError("invalid session header")
        raw_metadata = header.get("metadata")
        if not isinstance(raw_metadata, dict):
            raise ValueError("invalid session metadata")
        metadata = _metadata(cast(SessionMetadata, raw_metadata))
        _validate_record(header, metadata, first=True)
        if metadata["path"] != str(path):
            raise ValueError("session header path mismatch")
        return metadata
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path}: line 1, byte 0: {exc}") from exc


def _scan(directory: Path) -> list[SessionMetadata]:
    """扫描目录中的会话头，检测身份冲突并按创建时间排序。"""
    if not directory.exists():
        return []
    results = []
    ids = set()
    for path in sorted(directory.iterdir()):
        if path.suffix != ".jsonl" or path.is_symlink() or not path.is_file():
            continue
        metadata = _read_metadata(path)
        if metadata["id"] in ids:
            raise ValueError(f"{directory}: duplicate session id: {metadata['id']}")
        ids.add(metadata["id"])
        results.append(metadata)
    return sorted(results, key=lambda item: (-item["created_at"], item["id"], item["path"]))


class SessionRepository:
    """单事件循环内串行管理会话；不提供跨仓库或跨进程锁。"""

    def __init__(self, options: SessionRepositoryOptions) -> None:
        """固定默认目录，不创建目录或打开文件。"""
        self._directory = _path(options.get("directory", _DEFAULT_DIRECTORY))
        self._lock = asyncio.Lock()
        self._sessions: dict[str, Session] = {}
        self._close_task: asyncio.Task[None] | None = None

    def _require_open(self) -> None:
        """拒绝关闭后继续使用仓库。"""
        if self._close_task is not None:
            raise RuntimeError("session repository is closing or closed")

    async def _check_conflicts(self, metadata: SessionMetadata, *, creating: bool) -> None:
        """检查身份和路径与仓库已知会话是否冲突。"""
        candidates = [session.metadata for session in self._sessions.values()]
        directories = {self._directory, Path(metadata["path"]).parent}
        for directory in sorted(directories):
            candidates.extend(
                await _settle(asyncio.create_task(asyncio.to_thread(_scan, directory)))
            )
        for candidate in candidates:
            if candidate["id"] == metadata["id"] and (
                creating or candidate["path"] != metadata["path"]
            ):
                raise FileExistsError(f"session id already exists: {metadata['id']}")
        for session in self._sessions.values():
            if not session.closed and (
                session.metadata["id"] == metadata["id"]
                or session.metadata["path"] == metadata["path"]
            ):
                raise RuntimeError("session is already open")

    async def create(self, options: SessionCreateOptions | None = None) -> Session:
        """独占创建文件及头部；显式相对路径以仓库目录为基准。"""
        options = deepcopy(options) if options is not None else {}
        session_id = _session_id(options.get("id", uuid4().hex))
        raw_path = options.get("path", f"{session_id}.jsonl")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("path must be a nonempty string")
        path = _path(str(self._directory / Path(raw_path).expanduser()))
        if path.suffix != ".jsonl":
            raise ValueError("session path must end in .jsonl")
        async with self._lock:
            self._require_open()
            metadata: SessionMetadata = {
                "id": session_id,
                "created_at": time.time(),
                "path": str(path),
            }
            await self._check_conflicts(metadata, creating=True)
            store = await JsonlSessionStore.create(metadata)
            session = Session(metadata, store)
            self._sessions[str(path)] = session
            return session

    async def open(self, metadata: SessionMetadata) -> Session:
        """校验完整日志及运行关系并恢复状态；失败和取消均释放新句柄。"""
        metadata = _metadata(metadata)
        async with self._lock:
            self._require_open()
            await self._check_conflicts(metadata, creating=False)
            store = await JsonlSessionStore.open(metadata)
            try:
                session = Session(metadata, store)
                await session.state()
            except BaseException:
                await _settle(asyncio.create_task(store.close()))
                raise
            self._sessions[metadata["path"]] = session
            return session

    async def list(self, options: SessionListOptions | None = None) -> list[SessionMetadata]:
        """只读扫描单层目录的普通 .jsonl 文件；坏头报错，不修复日志。"""
        options = deepcopy(options) if options is not None else {}
        directory = self._directory if "directory" not in options else _path(options["directory"])
        async with self._lock:
            self._require_open()
            return await _settle(asyncio.create_task(asyncio.to_thread(_scan, directory)))

    async def delete(self, metadata: SessionMetadata) -> None:
        """核对头部身份后删除；拒绝本仓库尚未关闭的句柄。"""
        metadata = _metadata(metadata)
        async with self._lock:
            self._require_open()
            for session in self._sessions.values():
                if not session.closed and (
                    session.metadata["id"] == metadata["id"]
                    or session.metadata["path"] == metadata["path"]
                ):
                    raise RuntimeError("cannot delete an open session")

            def remove() -> None:
                path = Path(metadata["path"])
                if _read_metadata(path) != metadata:
                    raise ValueError("session metadata mismatch")
                path.unlink()

            await _settle(asyncio.create_task(asyncio.to_thread(remove)))
            self._sessions.pop(metadata["path"], None)

    async def close(self) -> None:
        """停止接收操作并关闭所有拥有的会话；重复关闭共享结果。"""
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._finish_close())
        await _settle(self._close_task)

    async def _finish_close(self) -> None:
        """等待进行中的打开/创建操作并关闭所有自有会话，最后报告错误。"""
        async with self._lock:
            results = await asyncio.gather(
                *(session.close() for session in self._sessions.values()), return_exceptions=True
            )
            errors = [result for result in results if isinstance(result, BaseException)]
            if errors:
                raise BaseExceptionGroup("failed to close session repository", errors)
