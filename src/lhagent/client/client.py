"""编排客户端单次流式调用、重试、取消、内容累积和统计。"""

import asyncio
import json
from asyncio import Event
from collections.abc import AsyncIterator, Awaitable
from contextlib import aclosing, suppress
from dataclasses import dataclass

from .config import ClientConfig, validate_config
from .errors import ProtocolError, classify_error, error_detail
from .metrics import CallMetrics
from .retry import get_retry_delay, should_retry
from .transport import ResponseChunk, ResponseDelta, Transport
from .types import ClientRequest, ClientResult, ContentBlock, StreamEvent

_ERROR_MESSAGES = {
    "context_overflow": "Model context limit exceeded.",
    "rate_limit": "Model service rate limit exceeded.",
    "authentication": "Model service authentication failed.",
    "invalid_request": "Model service rejected the request.",
    "transport": "Model connection or stream failed.",
    "protocol": "Model response did not satisfy the supported protocol.",
    "other": "Model request failed.",
}
_FINISH_REASONS = {"stop": "stop", "length": "length", "tool_calls": "tool_call"}


class _CallCancelled(Exception):
    """内部控制流：调用已取消，且通信资源已经由调用方清理。"""


@dataclass
class _CallState:
    """单次活动调用的取消信号与清理完成信号，彼此不可替代。"""

    cancel_event: Event
    done: Event


def validate_request(request: ClientRequest) -> None:
    """校验一次调用的本地身份和基本请求形状，不回显请求内容。"""
    if not isinstance(request, dict):
        raise TypeError("request must be a mapping")
    call_id = request.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("call_id must be a nonempty string")
    if not isinstance(request.get("model"), str) or not request["model"]:
        raise ValueError("model must be a nonempty string")
    if not isinstance(request.get("messages"), list):
        raise TypeError("messages must be a list")
    if not isinstance(request.get("parameters"), dict):
        raise TypeError("parameters must be a mapping")


class Client:
    """无会话状态的客户端；每次调用独立累积内容、重试和统计。"""

    def __init__(self, config: ClientConfig) -> None:
        """校验配置并创建自有传输；调用状态按 call_id 隔离。"""
        validate_config(config)
        self._config = config
        self._transport = Transport(config)
        self._active_calls: dict[str, _CallState] = {}
        self._closed = False
        self._close_task: asyncio.Task | None = None

    async def stream(
        self, request: ClientRequest, *, cancel_event: Event | None = None
    ) -> AsyncIterator[StreamEvent]:
        """交付增量及唯一终结事件；提前结束消费时须 aclose 或使用 aclosing。"""
        if self._closed:
            raise RuntimeError("client is closed")
        validate_request(request)
        call_id = request["call_id"]
        if call_id in self._active_calls:
            raise ValueError("call_id is already active")
        state = _CallState(Event(), Event())
        self._active_calls[call_id] = state
        events = asyncio.Queue()
        acknowledged = Event()
        worker = asyncio.create_task(
            self._execute(request, state, cancel_event, events, acknowledged)
        )
        try:
            while True:
                event = await events.get()
                yield event
                if event["result"] is not None:
                    break
                acknowledged.set()
        finally:
            # 消费者暂停时后台任务只等待确认；仍可独立响应取消并释放通信资源。
            state.cancel_event.set()
            await _join(worker)

    async def _execute(self, request, state, cancel_event, events, acknowledged) -> None:
        """驱动流并交付唯一终结结果；增量等待消费确认，异常按固定文案脱敏。"""
        call_id = request["call_id"]
        metrics = CallMetrics()
        accumulator = _Accumulator()
        finish_reason = None
        error_kind = None
        error_message = None
        response = None
        try:
            try:
                if cancel_event is not None and cancel_event.is_set():
                    raise _CallCancelled
                response = await self._establish(request, state, cancel_event, metrics)
                async with aclosing(response):
                    response_iterator = response.__aiter__()
                    while True:
                        chunk = await self._read_chunk(response_iterator, state, cancel_event)
                        if chunk is None:
                            break
                        if chunk["usage"] is not None:
                            metrics.update_usage(chunk["usage"])
                        if finish_reason is not None and chunk["deltas"]:
                            raise ProtocolError("content after finish_reason")
                        for delta in chunk["deltas"]:
                            index = accumulator.append(delta)
                            if any(delta["data"].values()):
                                metrics.record_first_content()
                            acknowledged.clear()
                            events.put_nowait(
                                {
                                    "call_id": call_id,
                                    "type": "delta",
                                    "block_index": index,
                                    "data": {"type": delta["type"], **delta["data"]},
                                    "result": None,
                                }
                            )
                            await _await_cancellable(
                                acknowledged.wait(), state.cancel_event, cancel_event
                            )
                        reason = chunk["finish_reason"]
                        if reason is not None:
                            if finish_reason is not None or reason not in _FINISH_REASONS:
                                raise ProtocolError("invalid or repeated finish_reason")
                            finish_reason = _FINISH_REASONS[reason]
                if state.cancel_event.is_set() or (
                    cancel_event is not None and cancel_event.is_set()
                ):
                    raise _CallCancelled
                if finish_reason is None:
                    raise ProtocolError("stream ended without a valid finish_reason")
                accumulator.finish(finish_reason)
            except _CallCancelled:
                finish_reason = "cancelled"
                error_kind = error_message = None
            except Exception as error:
                error_kind = classify_error(error)
                error_message = _ERROR_MESSAGES[error_kind]
                detail = error_detail(error, self._config.api_key)
                if detail:
                    error_message += " " + detail
                finish_reason = "error"
            result: ClientResult = {
                "call_id": call_id,
                "content": accumulator.content,
                "finish_reason": finish_reason,
                "error": error_message,
                "error_kind": error_kind,
                "stats": metrics.finish(),
            }
        finally:
            self._active_calls.pop(call_id, None)
            state.done.set()
        events.put_nowait(
            {
                "call_id": call_id,
                "type": "cancelled"
                if finish_reason == "cancelled"
                else ("error" if error_kind else "done"),
                "block_index": None,
                "data": {},
                "result": result,
            }
        )

    async def complete(
        self, request: ClientRequest, *, cancel_event: Event | None = None
    ) -> ClientResult:
        """消费同一 stream，只返回终结快照，不重复拼接或发送请求。"""
        async with aclosing(self.stream(request, cancel_event=cancel_event)) as events:
            async for event in events:
                if event["result"] is not None:
                    return event["result"]
        raise RuntimeError("client stream ended without a result")

    async def cancel(self, call_id: str) -> None:
        """请求指定调用取消，并等待其通信资源清理和终结状态落定。"""
        state = self._active_calls.get(call_id)
        if state is None:
            return
        state.cancel_event.set()
        await state.done.wait()

    async def close(self) -> None:
        """取消并等待活动调用后关闭传输；重复调用幂等。"""
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await _join(self._close_task)

    async def _close(self) -> None:
        """先取消并等待所有活动调用清理，再关闭共享传输。"""
        states = tuple(self._active_calls.values())
        for state in states:
            state.cancel_event.set()
        await asyncio.gather(*(state.done.wait() for state in states))
        await self._transport.close()

    async def _establish(self, request, state, external, metrics):
        """只在取得响应前重试；取消与建立成功竞争时关闭未交付响应。"""
        attempts = 0
        while True:
            if state.cancel_event.is_set() or (external is not None and external.is_set()):
                raise _CallCancelled
            attempts += 1
            metrics.record_attempt()
            try:
                return await _await_cancellable(
                    self._transport.open_stream(request),
                    state.cancel_event,
                    external,
                    close_abandoned=True,
                )
            except _CallCancelled:
                raise
            except Exception as error:
                if not should_retry(error, attempts, self._config):
                    raise
                delay = get_retry_delay(error, attempts - 1, self._config)
                if delay is None:
                    raise
                await _wait_cancel(delay, state.cancel_event, external)

    async def _read_chunk(self, response_iterator, state, external) -> ResponseChunk | None:
        """读取下一分片并响应取消；流正常耗尽返回 None。"""
        try:
            return await _await_cancellable(anext(response_iterator), state.cancel_event, external)
        except StopAsyncIteration:
            return None


async def _wait_cancel(delay: float, internal: Event, external: Event | None) -> None:
    """可取消地等待退避间隔；零间隔仍检查信号并让出事件循环。"""
    if delay <= 0:
        if internal.is_set() or (external is not None and external.is_set()):
            raise _CallCancelled
        await asyncio.sleep(0)
        return
    await _await_cancellable(asyncio.sleep(delay), internal, external)


async def _join(task: asyncio.Task):
    """等待清理，即使等待者再次被取消也不让清理任务泄漏。"""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        # 取出潜在异常后遵循调用方的 Python 取消语义。
        with suppress(BaseException):
            task.result()
        raise asyncio.CancelledError
    return task.result()


async def _await_cancellable(
    awaitable: Awaitable, internal: Event, external: Event | None, *, close_abandoned=False
):
    """协调操作与取消信号；移交结果前回收监听任务，必要时关闭被放弃的响应。"""
    operation = asyncio.ensure_future(awaitable)
    signals = [asyncio.create_task(internal.wait())]
    if external is not None and external is not internal:
        signals.append(asyncio.create_task(external.wait()))
    try:
        if internal.is_set() or (external is not None and external.is_set()):
            raise _CallCancelled
        done, _ = await asyncio.wait((operation, *signals), return_when=asyncio.FIRST_COMPLETED)
        if (
            operation not in done
            or internal.is_set()
            or (external is not None and external.is_set())
        ):
            raise _CallCancelled
        for task in signals:
            task.cancel()
        await asyncio.gather(*signals, return_exceptions=True)
        # operation 的结果在取消监听任务全部退出后才移交。
        result = operation.result()
    except BaseException:
        operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)
        # 建立成功与取消同时发生时，未交付的响应也必须关闭。
        if close_abandoned and not operation.cancelled() and operation.exception() is None:
            await operation.result().aclose()
        raise
    finally:
        for task in signals:
            task.cancel()
        await asyncio.gather(*signals, return_exceptions=True)
    return result


class _Accumulator:
    """按首次出现顺序分配块；工具协议索引与公共块索引相互独立。"""

    def __init__(self) -> None:
        """初始化单次响应的内容块和协议索引映射。"""
        self.content: list[ContentBlock] = []
        self._indices: dict[tuple[str, int | None], int] = {}

    def append(self, delta: ResponseDelta) -> int:
        """按协议身份累加分片并返回公共块索引；同一工具索引不得改变身份。"""
        kind, data = delta["type"], delta["data"]
        key = (kind, delta["tool_index"])
        if key not in self._indices:
            self._indices[key] = len(self.content)
            if kind == "tool_call":
                self.content.append(
                    {
                        "type": "tool_call",
                        "call_id": "",
                        "name": "",
                        "arguments_json": "",
                        "complete": False,
                    }
                )
            else:
                self.content.append({"type": kind, "text": ""})
        index = self._indices[key]
        block = self.content[index]
        if kind != "tool_call":
            block["text"] += data["text"]
        else:
            for field in ("call_id", "name"):
                if field in data:
                    if block[field] and block[field] != data[field]:
                        raise ProtocolError("tool identity changed for the same index")
                    block[field] = data[field]
            if "arguments_json" in data:
                block["arguments_json"] += data["arguments_json"]
        return index

    def finish(self, reason: str) -> None:
        """校验终态与工具参数；全部工具解析成功后才统一标记完整。"""
        tools = [block for block in self.content if block["type"] == "tool_call"]
        if reason == "length":
            return
        if bool(tools) != (reason == "tool_call"):
            raise ProtocolError("tool content does not match finish_reason")
        parsed = []
        call_ids = set()
        for tool in tools:
            if not tool["call_id"] or not tool["name"] or tool["call_id"] in call_ids:
                raise ProtocolError("tool identity is missing or duplicated")
            call_ids.add(tool["call_id"])
            try:
                arguments = json.loads(tool["arguments_json"], parse_constant=_reject_constant)
            except ValueError:
                raise ProtocolError("tool arguments must encode a JSON object") from None
            if not isinstance(arguments, dict):
                raise ProtocolError("tool arguments must encode a JSON object")
            parsed.append(arguments)
        for tool, arguments in zip(tools, parsed, strict=True):
            tool["arguments"] = arguments
            tool["complete"] = True


def _reject_constant(value: str) -> None:
    """拒绝 JSON 中的 NaN、Infinity 等非有限常量。"""
    raise ValueError("non-finite JSON constant")
