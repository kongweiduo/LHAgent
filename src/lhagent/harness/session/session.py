"""单个会话的历史投影、压缩引用持久化与运行状态。"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from lhagent.client.types import ClientResult, Message, client_result_to_message
from lhagent.harness.context.types import CompactionResult, HistoryContext
from lhagent.harness.tools.types import ToolResult

from .jsonl import JsonlSessionStore, _settle
from .types import (
    DisplayEntry,
    DisplayHistory,
    JsonlRecord,
    SessionCloseResult,
    SessionMetadata,
    SessionStateInfo,
)

if TYPE_CHECKING:
    from lhagent.harness.loop.types import LoopResult


class Session:
    """一个可持久化、可恢复且暂不支持分支的线性会话。"""

    def __init__(self, metadata: SessionMetadata, store: JsonlSessionStore) -> None:
        """绑定元数据一致的存储并初始化投影；不立即读取日志。"""
        if metadata != store.metadata:
            raise ValueError("session metadata does not match store")
        self._metadata = deepcopy(metadata)
        self._store = store
        self._lock = asyncio.Lock()
        self._loaded = False
        self._records: list[JsonlRecord] = []
        self._run_ids: set[str] = set()
        self._active_run_id: str | None = None
        self._last_finished_run_id: str | None = None
        self._interrupted = False
        self._recovered_unfinished = False
        self._started_here = False
        self._closed = False
        self._close_task: asyncio.Task[SessionCloseResult] | None = None

    @property
    def closed(self) -> bool:
        """资源清理是否已经完成（即使关闭报告了写入错误）。"""
        return self._closed

    @property
    def metadata(self) -> SessionMetadata:
        """返回元数据副本，避免调用方改写会话身份。"""
        return deepcopy(self._metadata)

    async def _load(self) -> None:
        """按需重放并校验记录及引用，恢复未完成运行标记。"""
        if self._loaded:
            return
        records = await self._store.read_records()
        if not records or records[0]["type"] != "session":
            raise ValueError("missing session header")
        entry_ids = {records[0]["id"]}
        run_ids: set[str] = set()
        unfinished: set[str] = set()
        last_finished = None
        last_started = None
        for record in records[1:]:
            kind, entry_id, run_id = record["type"], record["id"], record["run_id"]
            if entry_id in entry_ids:
                raise ValueError(f"duplicate entry id: {entry_id}")
            entry_ids.add(entry_id)
            if kind == "run_start":
                if not run_id or run_id in run_ids:
                    raise ValueError(f"duplicate or empty run id: {run_id!r}")
                run_ids.add(run_id)
                unfinished.add(run_id)
                last_started = run_id
            elif kind == "run_finish":
                if (
                    run_id not in unfinished
                    or record["status"] not in ("completed", "length", "error", "cancelled")
                    or not isinstance(record["error"], (str, type(None)))
                ):
                    raise ValueError(f"invalid run finish: {run_id!r}")
                unfinished.remove(run_id)
                last_finished = run_id
            elif kind in ("user", "assistant", "tool_result"):
                if run_id not in unfinished:
                    raise ValueError(f"message outside running run: {run_id!r}")
                if kind == "assistant" and not self._terminal(record["response"]):
                    raise ValueError("nonterminal assistant record")
                if kind == "tool_result" and not self._finished_tool(record["result"]):
                    raise ValueError("nonterminal tool result record")
            elif kind in ("history_exclusion", "compaction"):
                if run_id not in unfinished:
                    raise ValueError(f"history update outside running run: {run_id!r}")
        self._project_history(records)
        self._records = records
        self._run_ids = run_ids
        self._interrupted = bool(unfinished)
        self._recovered_unfinished = last_started in unfinished
        self._last_finished_run_id = last_finished
        self._loaded = True

    @staticmethod
    def _terminal(response: object) -> bool:
        """判断响应是否具备允许持久化的终态形状。"""
        return (
            isinstance(response, dict)
            and response.get("finish_reason")
            in ("stop", "tool_call", "length", "error", "cancelled")
            and isinstance(response.get("call_id"), str)
            and isinstance(response.get("content"), list)
            and isinstance(response.get("stats"), dict)
            and isinstance(response.get("error"), (str, type(None)))
            and (
                response.get("error_kind")
                in (
                    "context_overflow",
                    "rate_limit",
                    "authentication",
                    "invalid_request",
                    "transport",
                    "protocol",
                    "other",
                )
                if response["finish_reason"] == "error"
                else response.get("error_kind") is None
            )
        )

    @staticmethod
    def _finished_tool(result: object) -> bool:
        """判断工具结果是否具备允许持久化的结束状态。"""
        return (
            isinstance(result, dict)
            and result.get("status")
            in ("success", "validation_error", "execution_error", "timeout", "cancelled")
            and isinstance(result.get("call_id"), str)
            and isinstance(result.get("name"), str)
            and isinstance(result.get("output"), (dict, type(None)))
            and isinstance(result.get("error"), (str, type(None)))
        )

    def _record(self, kind: str, run_id: str, **payload: object) -> JsonlRecord:
        """为当前运行创建带唯一条目标识及时间戳的记录。"""
        return cast(
            JsonlRecord,
            {
                "type": kind,
                "id": uuid4().hex,
                "run_id": run_id,
                "timestamp": time.time(),
                **payload,
            },
        )

    async def _append(self, record: JsonlRecord) -> str:
        """先完成存储提交，再更新内存记录，避免虚报持久化成功。"""
        await self._store.append([record])
        self._records.append(deepcopy(record))
        return record["id"]

    async def _run[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        """在锁内执行已接受操作；调用者取消仍等待提交落定。"""

        async def locked() -> T:
            async with self._lock:
                if self._close_task is not None:
                    raise RuntimeError("session is closing or closed")
                await self._load()
                return await operation()

        return await _settle(asyncio.create_task(locked()))

    @staticmethod
    def _ids(value: object) -> list[str]:
        """校验引用 ID 列表的类型、非空值和唯一性。"""
        if (
            not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)
            or len(set(value)) != len(value)
        ):
            raise ValueError("entry IDs must be unique nonempty strings")
        return value

    @classmethod
    def _tail(cls, value: object, current: list[str]) -> list[str]:
        """校验保留条目是有效历史的连续尾部。"""
        ids = cls._ids(value)
        if ids and (len(ids) > len(current) or current[-len(ids) :] != ids):
            raise ValueError("retained entry IDs must form the current effective history tail")
        return ids

    @staticmethod
    def _compaction_metadata(record: dict) -> None:
        """校验压缩摘要和 token 统计的持久化字段。"""
        if not isinstance(record.get("summary"), str) or not record["summary"].strip():
            raise ValueError("compaction requires a nonempty summary")
        for key in ("tokens_before", "estimated_tokens_after"):
            if type(record.get(key)) is not int or record[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if not isinstance(record.get("usage"), list) or any(
            not isinstance(item, dict) for item in record["usage"]
        ):
            raise ValueError("compaction usage must be a list of objects")

    @classmethod
    def _project_history(cls, records: list[JsonlRecord]) -> HistoryContext:
        """按顺序应用排除和压缩记录，验证引用并构造有效历史。"""
        summary = None
        originals: dict[str, Message] = {}
        entry_ids: list[str] = []
        for record in records:
            kind = record["type"]
            if kind == "user":
                message = record["message"]
            elif kind == "assistant":
                message = client_result_to_message(record["response"])
            elif kind == "tool_result":
                result = record["result"]
                output = result["output"]
                message = {
                    "role": "tool",
                    "tool_call_id": result["call_id"],
                    "name": result["name"],
                    "content": [
                        {
                            "type": "tool_result",
                            "content": output["content"] if output is not None else result["error"],
                            "is_error": result["status"] != "success",
                        }
                    ],
                }
            elif kind == "history_exclusion":
                excluded = cls._ids(record["entry_ids"])
                if record["reason"] not in ("overflow_recovery", "manual", "other") or any(
                    item not in originals for item in excluded
                ):
                    raise ValueError(f"invalid history exclusion reference: {record['id']}")
                excluded_set = set(excluded)
                entry_ids = [item for item in entry_ids if item not in excluded_set]
                continue
            elif kind == "compaction":
                cls._compaction_metadata(record)
                entry_ids = list(cls._tail(record["retained_entry_ids"], entry_ids))
                summary = record["summary"]
                continue
            else:
                continue
            originals[record["id"]] = cast(Message, message)
            entry_ids.append(record["id"])
        return deepcopy(
            {
                "summary": summary,
                "messages": [originals[item] for item in entry_ids],
                "entry_ids": entry_ids,
            }
        )

    async def get_history(self) -> HistoryContext:
        """返回最新摘要和未被摘要替代或排除的原始消息副本。"""

        async def operation() -> HistoryContext:
            return self._project_history(self._records)

        return await self._run(operation)

    async def get_display_history(self) -> DisplayHistory:
        """返回全部已提交消息的深副本，保留完整响应和工具结果。

        摘要或排除不删除展示项，仅将 in_context 标为 False；不补造中断工具
        的结果，也不包含存储头、运行记录或原始压缩/排除记录。与追加共用锁。
        """

        async def operation() -> DisplayHistory:
            history = self._project_history(self._records)
            effective_ids = set(history["entry_ids"])
            entries: list[DisplayEntry] = []
            for record in self._records:
                kind = record["type"]
                if kind not in ("user", "assistant", "tool_result"):
                    continue
                field = {"user": "message", "assistant": "response", "tool_result": "result"}[kind]
                entries.append(
                    cast(
                        DisplayEntry,
                        {
                            "type": kind,
                            "entry_id": record["id"],
                            "run_id": record["run_id"],
                            "timestamp": record["timestamp"],
                            "in_context": record["id"] in effective_ids,
                            field: deepcopy(record[field]),
                        },
                    )
                )
            return {"summary": history["summary"], "entries": entries}

        return await self._run(operation)

    async def append_user(self, message: Message) -> str:
        """为活动运行持久化用户消息，成功后返回条目 ID。"""
        if (
            not isinstance(message, dict)
            or message.get("role") != "user"
            or not isinstance(message.get("content"), list)
        ):
            raise ValueError("expected a user message")

        async def operation() -> str:
            if self._active_run_id is None:
                raise RuntimeError("no active run")
            return await self._append(self._record("user", self._active_run_id, message=message))

        return await self._run(operation)

    async def append_response(self, response: ClientResult) -> str:
        """仅持久化终结模型响应，保留其内容与错误信息。"""
        if not self._terminal(response):
            raise ValueError("assistant response must be terminal with valid error_kind")

        async def operation() -> str:
            if self._active_run_id is None:
                raise RuntimeError("no active run")
            return await self._append(
                self._record("assistant", self._active_run_id, response=response)
            )

        return await self._run(operation)

    async def append_tool_result(self, result: ToolResult) -> str:
        """仅持久化已结束工具结果，保留调用关联。"""
        if not self._finished_tool(result):
            raise ValueError("tool result must be finished")

        async def operation() -> str:
            if self._active_run_id is None:
                raise RuntimeError("no active run")
            return await self._append(
                self._record("tool_result", self._active_run_id, result=result)
            )

        return await self._run(operation)

    async def omit_failed_attempt(self, entry_ids: list[str]) -> None:
        """独立追加排除标记；允许重复排除已有原消息，不修改摘要。"""
        ids = deepcopy(self._ids(entry_ids))

        async def operation() -> None:
            if self._active_run_id is None:
                raise RuntimeError("no active run")
            record = self._record(
                "history_exclusion", self._active_run_id, entry_ids=ids, reason="overflow_recovery"
            )
            self._project_history([*self._records, record])
            if ids:
                await self._append(record)

        await self._run(operation)

    async def commit_compaction(self, result: CompactionResult) -> None:
        """校验成功结果的原始尾部身份和内容，仅保存摘要及引用。"""
        result = deepcopy(result)
        if (
            not isinstance(result, dict)
            or result.get("status") != "success"
            or result.get("error") is not None
            or not isinstance(result.get("history"), dict)
        ):
            raise ValueError("only successful compaction results can be committed")
        history = result["history"]

        async def operation() -> None:
            if self._active_run_id is None:
                raise RuntimeError("no active run")
            current = self._project_history(self._records)
            ids = self._tail(history.get("entry_ids"), current["entry_ids"])
            messages = history.get("messages")
            expected = current["messages"][-len(ids) :] if ids else []
            if not isinstance(messages, list) or len(messages) != len(ids) or messages != expected:
                raise ValueError("retained messages must match original history entries")
            record = self._record(
                "compaction",
                self._active_run_id,
                summary=history.get("summary"),
                retained_entry_ids=ids,
                tokens_before=result.get("tokens_before"),
                estimated_tokens_after=result.get("estimated_tokens_after"),
                usage=result.get("usage"),
            )
            self._compaction_metadata(record)
            await self._append(record)

        await self._run(operation)

    async def start_run(self, run_id: str) -> None:
        """持久化运行开始记录后更新活动状态，拒绝并发或重复身份。"""
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a nonempty string")

        async def operation() -> None:
            if self._active_run_id is not None:
                raise RuntimeError("run already active")
            if run_id in self._run_ids:
                raise ValueError("duplicate run_id")
            await self._append(self._record("run_start", run_id))
            self._run_ids.add(run_id)
            self._active_run_id = run_id
            self._started_here = True

        await self._run(operation)

    async def finish_run(self, result: "LoopResult") -> None:
        """校验并持久化当前运行终态，成功后清除活动运行。"""
        if (
            not isinstance(result, dict)
            or result.get("status") not in ("completed", "length", "error", "cancelled")
            or not isinstance(result.get("error"), (str, type(None)))
        ):
            raise ValueError("invalid run result")

        async def operation() -> None:
            if self._active_run_id is None:
                raise RuntimeError("no active run")
            run_id = self._active_run_id
            await self._append(
                self._record("run_finish", run_id, status=result["status"], error=result["error"])
            )
            self._last_finished_run_id = run_id
            self._active_run_id = None

        await self._run(operation)

    async def state(self) -> SessionStateInfo:
        """在串行边界读取状态；区分当前运行和恢复发现的中断。"""
        async with self._lock:
            await self._load()
            if self._closed:
                status = "closed"
            elif self._active_run_id is not None:
                status = "active"
            elif self._recovered_unfinished and not self._started_here:
                status = "interrupted"
            elif self._run_ids:
                status = "idle"
            else:
                status = "new"
            return {
                "state": status,
                "active_run_id": self._active_run_id,
                "last_finished_run_id": self._last_finished_run_id,
                "interrupted": self._interrupted,
            }

    async def close(self) -> SessionCloseResult:
        """共享同一关闭任务；等待已接受操作后释放存储。"""
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._finish_close())
        return await _settle(self._close_task)

    async def _finish_close(self) -> SessionCloseResult:
        """在会话锁内关闭存储，并在失败时也标记资源已关闭。"""
        async with self._lock:
            try:
                await self._store.close()
            finally:
                if self._active_run_id is not None:
                    self._interrupted = True
                    self._active_run_id = None
                self._closed = True
            return {"session_id": self._metadata["id"], "state": "closed"}
