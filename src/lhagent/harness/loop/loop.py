"""模型响应、串行工具批次、自动压缩、输入接入与运行取消。"""

import asyncio
import json
from contextlib import aclosing
from copy import deepcopy

from lhagent.client.client import Client
from lhagent.client.types import ClientRequest, ClientResult
from lhagent.harness.context.assembly import assemble_context
from lhagent.harness.context.budget import estimate_context_tokens, should_compact, validate_budget
from lhagent.harness.context.compaction import compact
from lhagent.harness.context.types import CompactionResult, ContextInput
from lhagent.harness.tools.execution import execute_tool_call
from lhagent.harness.tools.registry import describe_tools
from lhagent.harness.tools.types import ToolCall, ToolContext, ToolResult, ToolStopError

from .queue import InputQueue
from .types import EventSink, LoopConfig, LoopEvent, LoopResult, LoopSession


class _RunCancelled(Exception):
    """主动信号取消，在 run 边界转换为结果。"""


class AgentLoop:
    """同一实例同时只允许一次运行，不提供多 agent 调度或后台任务管理。

    client、会话和输入队列由外部提供；loop 不关闭共享客户端或整个会话。
    当前响应增量属于临时状态，完整或失败的终结响应才逐条交给会话层。
    """

    def __init__(
        self,
        client: Client,
        session: LoopSession,
        queue: InputQueue,
        config: LoopConfig,
        tool_context: ToolContext,
        emit: EventSink,
    ) -> None:
        """接收执行依赖，不启动请求、不加载历史、不创建持久化存储。

        tool_context 的取消信号须用于当前运行；每次运行创建新的 Event，
        同时传给 tools、Client.stream 和 compact；不复用或清除已取消信号。
        loop 负责协调信号，tools 负责终止执行资源并确认清理结果。
        """
        self._client = client
        self._session = session
        self._queue = queue
        self._config = config
        self._tool_context = tool_context
        self._emit = emit
        self._running: asyncio.Task | None = None
        self._run_id: str | None = None
        self._cancel_event: asyncio.Event | None = None
        self._tool_stop_unconfirmed = False
        self._idle: asyncio.Event | None = None
        self._external_cancel: asyncio.CancelledError | None = None
        self._overflow_recovery_attempted = False

    async def run(self, instruction: dict[str, object]) -> LoopResult:
        """接入新用户指令并驱动循环，运行中再次直接调用应明确报错。

        先通过 new_run_id 分配 ID 并等待 session.start_run 成功，再提交用户指令。
        开始提交失败时不启动模型或工具，也不提交无对应开始记录的 finish_run。
        已开始的运行由 loop 统一收尾；取消、准备失败也须尝试提交终态，提交
        失败明确上抛，不能用收尾异常掩盖原始失败或宣称已保存。
        指令逐条提交；每次请求前取得最新有效历史并检查上下文预算。
        消费响应流后提交终结响应，再执行完整工具调用并逐条提交结果。
        工具批次完成后优先接入 steering；本来可以结束时再接入 follow_up。
        压缩期间到达的 steering 在请求前再次检查，避免一次取入两条。
        正常、截断、错误和取消分别收尾，不因错误或取消自动消费后续请求。
        """
        if self._running is not None:
            raise RuntimeError("loop is already running")
        if self._tool_stop_unconfirmed:
            raise RuntimeError("tool stop is unconfirmed; this loop cannot run again")
        if "tools" in self._config["parameters"] or "tool_choice" in self._config["parameters"]:
            raise NotImplementedError("raw tool parameters are not supported yet")
        if (
            not isinstance(instruction, dict)
            or instruction.get("role") != "user"
            or not isinstance(instruction.get("content"), list)
        ):
            raise ValueError("instruction must be a user message")
        self._running = asyncio.current_task()
        self._idle = asyncio.Event()
        self._external_cancel = None
        self._cancel_event = asyncio.Event()
        self._tool_context["cancel_event"] = self._cancel_event
        started = False
        result: LoopResult = {"status": "error", "last_response": None, "error": None}
        failure: BaseException | None = None
        try:
            self._run_id = self._config["new_run_id"]()

            async def start() -> None:
                nonlocal started
                await self._session.start_run(self._run_id)
                started = True

            # 开始操作被取消时，会话可能已完成提交，因此仍需按已开始运行收尾。
            await self._accepted(start())
            self._raise_if_cancelled()
            await self._notify({"type": "run_start", "data": {"run_id": self._run_id}})
            self._raise_if_cancelled()
            entry_id = await self._accepted(self._session.append_user(deepcopy(instruction)))
            self._overflow_recovery_attempted = False
            await self._notify(
                {
                    "type": "message_committed",
                    "data": {"run_id": self._run_id, "entry_id": entry_id, "kind": "user"},
                }
            )
            self._raise_if_cancelled()
            input_consumed = False
            retry_compacted = False
            while True:
                self._raise_if_cancelled()
                if not input_consumed:
                    input_consumed = await self._consume_input()
                request = await self._prepare_request(compacted=retry_compacted)
                self._raise_if_cancelled()
                # 准备阶段读取历史或压缩时可能让出事件循环，期间可能收到新输入。
                # 仅当本阶段尚未接纳输入时再次检查，避免同一阶段重复消费。
                if not input_consumed and await self._consume_input():
                    request = await self._prepare_request(compacted=retry_compacted)
                    self._raise_if_cancelled()
                input_consumed = False
                response = await self._receive_response(request)
                retry_compacted = False
                result["last_response"] = response
                entry_id = await self._accepted(self._session.append_response(response))
                failed_entry_ids = [entry_id]
                await self._notify(
                    {
                        "type": "message_committed",
                        "data": {"run_id": self._run_id, "entry_id": entry_id, "kind": "assistant"},
                    }
                )
                reason = response["finish_reason"]
                blocks = [block for block in response["content"] if block["type"] == "tool_call"]
                if self._is_cancelled():
                    result["status"] = "cancelled"
                    result["error"] = None
                    await self._commit_unexecuted_calls(
                        blocks, "run was cancelled before execution"
                    )
                    break
                calls = self._get_tool_calls(response)
                if reason in ("stop", "tool_call"):
                    self._overflow_recovery_attempted = False
                if reason in ("length", "error", "cancelled"):
                    for block in blocks:
                        if (
                            isinstance(block.get("call_id"), str)
                            and block["call_id"]
                            and isinstance(block.get("name"), str)
                            and block["name"]
                        ):
                            failed_entry_ids.append(
                                await self._commit_tool_result(
                                    {
                                        "call_id": block["call_id"],
                                        "name": block["name"],
                                        "status": "validation_error",
                                        "output": None,
                                        "error": f"Tool call not executed: response ended with {reason}",
                                    }
                                )
                            )
                    if reason == "length" and calls:
                        continue
                elif calls:
                    await self._execute_tools(calls)
                    self._raise_if_cancelled()
                    continue
                self._raise_if_cancelled()
                if reason == "error" and await self._recover_overflow(response, failed_entry_ids):
                    retry_compacted = True
                    continue
                if reason == "stop":
                    input_consumed = await self._consume_input(allow_follow_up=True)
                    if input_consumed:
                        continue
                result["status"] = {
                    "stop": "completed",
                    "length": "length",
                    "error": "error",
                    "cancelled": "cancelled",
                }[reason]
                result["error"] = response["error"]
                break
        except BaseException as exc:
            if isinstance(exc, ToolStopError) or isinstance(exc.__cause__, ToolStopError):
                self._tool_stop_unconfirmed = True
            if isinstance(exc, asyncio.CancelledError):
                self._cancel_event.set()
            failure = None if isinstance(exc, _RunCancelled) else exc
            cancelled = isinstance(exc, (asyncio.CancelledError, _RunCancelled))
            result["status"] = (
                "error" if self._tool_stop_unconfirmed else "cancelled" if cancelled else "error"
            )
            result["error"] = (
                str(exc.__cause__ or exc) if self._tool_stop_unconfirmed or not cancelled else None
            )
        if started:
            try:
                await self._accepted(self._session.finish_run(result))
                await self._notify(
                    {
                        "type": "run_end",
                        "data": {
                            "run_id": self._run_id,
                            "status": result["status"],
                            "error": result["error"],
                        },
                    }
                )
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(f"run cleanup failed: {exc!r}")
        if self._external_cancel is not None:
            if failure is not None:
                self._external_cancel.__cause__ = failure
            failure = self._external_cancel
        self._running = None
        self._run_id = None
        self._cancel_event = None
        self._idle.set()
        if failure is not None:
            raise failure
        return result

    async def _consume_input(self, *, allow_follow_up: bool = False) -> bool:
        """在接入边界取得一批输入，逐条提交；steering 始终优先。"""
        self._raise_if_cancelled()
        messages = self._queue.drain_steering()
        if not messages and allow_follow_up:
            messages = self._queue.drain_follow_up()
        for message in messages:
            self._raise_if_cancelled()
            entry_id = await self._accepted(self._session.append_user(message))
            self._overflow_recovery_attempted = False
            await self._notify(
                {
                    "type": "message_committed",
                    "data": {"run_id": self._run_id, "entry_id": entry_id, "kind": "user"},
                }
            )
        self._raise_if_cancelled()
        return bool(messages)

    async def _prepare_request(self, *, compacted: bool = False) -> ClientRequest:
        """检查完整输入预算，必要时提交一次自动压缩后组装主请求。

        压缩只发生在确有主请求要发送时。compacted 表示已提交超限恢复摘要，
        此时只复核预算，不再次压缩。摘要请求使用同一取消信号，摘要期间
        到达的 steering 在压缩结果落定后接入并重新读取历史；每条消息只从队列
        取出一次。压缩失败或无可压缩内容不会启动主请求，避免在超预算上下文
        上继续发送；取消则由主循环统一收尾。
        """
        while True:
            self._raise_if_cancelled()
            history = await self._session.get_history()
            self._raise_if_cancelled()
            context_input = {
                "prompts": self._config["prompts"],
                "history": history,
                "new_instruction": None,
            }
            messages = assemble_context(context_input)
            budget = self._config["budget"]
            settings = self._config["compaction"]
            validate_budget(budget, settings)
            tokens = estimate_context_tokens(messages, budget)
            if not should_compact(tokens, budget, settings):
                if tokens > budget["context_window"] - settings["reserve_tokens"]:
                    raise ValueError(
                        "request exceeds configured context budget; automatic compaction is disabled"
                    )
                return self._build_request(messages)
            if compacted:
                raise ValueError("request remains over configured context budget after compaction")
            result = await self._compact_context(context_input, tokens)
            if result["status"] != "success":
                message = result["error"] or (
                    "context cannot be compacted because no history can be summarized"
                    if result["status"] == "unchanged"
                    else "automatic compaction failed"
                )
                raise ValueError(message)
            compacted = True
            # 重新读取已提交视图并检查完整预算，不能仅信任摘要生成时的估算。
            # 调用方沿用每阶段消费标记处理 steering，避免压缩后重复接纳。

    async def _compact_context(self, context_input: ContextInput, tokens: int) -> CompactionResult:
        """共享摘要生成、事件、取消和成功提交；失败结果留给调用路径处理。"""
        request_summary = self._config["summary_request"]
        if request_summary is None:
            raise ValueError("automatic compaction requires a summary request")
        await self._notify(
            {"type": "compaction_start", "data": {"run_id": self._run_id, "tokens": tokens}}
        )
        result = None
        try:
            self._raise_if_cancelled()
            result = await compact(
                context_input,
                self._config["budget"],
                self._config["compaction"],
                request_summary,
                self._config["max_summary_output_tokens"],
                self._cancel_event,
            )
        except asyncio.CancelledError:
            self._cancel_event.set()
            raise
        finally:
            # 成功仅指摘要生成完成，会话持久化仍在后续进行。
            await self._notify(
                {
                    "type": "compaction_end",
                    "data": {
                        "run_id": self._run_id,
                        "status": (
                            "cancelled"
                            if self._is_cancelled()
                            else result["status"]
                            if result is not None
                            else "error"
                        ),
                        "error": result["error"] if result is not None else None,
                    },
                }
            )
        self._raise_if_cancelled()
        if result["status"] == "cancelled":
            raise _RunCancelled
        if result["status"] == "success":
            # 已接受的持久化必须等待落定，即使期间收到取消。
            await self._accepted(self._session.commit_compaction(result))
            self._raise_if_cancelled()
        return result

    def _build_request(self, messages: list[dict[str, object]]) -> ClientRequest:
        """以已组装消息、独立参数副本及工具定义构造带新 call_id 的请求。"""
        parameters = deepcopy(self._config["parameters"])
        if self._config["tools"]:
            parameters["tools"] = describe_tools(self._config["tools"])
        return {
            "call_id": self._config["new_call_id"](),
            "model": self._config["model"],
            "messages": messages,
            "parameters": parameters,
        }

    async def _receive_response(self, request: ClientRequest) -> ClientResult:
        """消费 Client.stream 的统一事件，维护当前响应并向外发送增量。

        文本、推理及工具参数可在同一响应内共存；增量不逐个追加到历史。
        以唯一终结结果确定完整内容和结束原因，不将 done 一律视为任务完成。
        终结结果由主循环提交一次；流中断或取消不得伪装为正常结束。
        """
        async with aclosing(
            self._client.stream(request, cancel_event=self._cancel_event)
        ) as events:
            async for event in events:
                if event["call_id"] != request["call_id"]:
                    raise ValueError("client stream call_id mismatch")
                if event["type"] == "delta":
                    if not isinstance(event["block_index"], int):
                        raise ValueError("client delta requires block_index")
                    await self._notify(
                        {
                            "type": "response_update",
                            "data": {
                                "run_id": self._run_id,
                                "call_id": request["call_id"],
                                "phase": "delta",
                                "block_index": event["block_index"],
                                "delta": event["data"],
                            },
                        }
                    )
                    if self._external_cancel is not None:
                        raise self._external_cancel
                else:
                    result = event["result"]
                    if result is None or result["call_id"] != request["call_id"]:
                        raise ValueError("client stream missing terminal result")
                    expected = (
                        "error"
                        if result["finish_reason"] == "error"
                        else "cancelled"
                        if result["finish_reason"] == "cancelled"
                        else "done"
                    )
                    if event["type"] != expected:
                        raise ValueError("client stream terminal event mismatch")
                    await self._notify(
                        {
                            "type": "response_update",
                            "data": {
                                "run_id": self._run_id,
                                "call_id": request["call_id"],
                                "phase": "end",
                                "result": result,
                            },
                        }
                    )
                    return result
        raise RuntimeError("client stream ended without a terminal result")

    def _get_tool_calls(self, response: ClientResult) -> list[ToolCall]:
        """从完整可执行响应中取得工具调用，不解析命令文本或执行 schema 校验。

        length、error、cancelled 结果中的调用不可执行，即使参数碰巧能解析。
        截断调用的未执行结果由主循环协调补齐，保留能关联的调用 ID。
        """
        reason = response["finish_reason"]
        blocks = [block for block in response["content"] if block["type"] == "tool_call"]
        if (reason == "stop" and blocks) or (reason == "tool_call" and not blocks):
            raise ValueError("tool call content contradicts finish_reason")
        if reason in ("error", "cancelled"):
            return []
        calls = []
        seen: set[str] = set()
        for block in blocks:
            call_id, name = block.get("call_id"), block.get("name")
            if reason == "length" and (
                not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name
            ):
                continue
            if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
                raise ValueError("tool call requires a nonempty ID and name")
            if call_id in seen:
                raise ValueError(f"duplicate tool call ID: {call_id}")
            seen.add(call_id)
            arguments = block.get("arguments")
            if "arguments_json" in block:
                try:
                    arguments = json.loads(block["arguments_json"])
                except (TypeError, ValueError):
                    arguments = None
            valid = block.get("complete") is True and isinstance(arguments, dict)
            if valid:
                try:
                    json.dumps(arguments, allow_nan=False)
                except (TypeError, ValueError, OverflowError):
                    valid = False
            if not valid:
                if reason == "length":
                    continue
                raise ValueError("normal assistant response contains incomplete tool call")
            calls.append({"call_id": call_id, "name": name, "arguments": arguments})
        if reason == "tool_call" and not calls:
            raise ValueError("tool_call finish_reason requires complete tool calls")
        return calls

    async def _accepted(self, operation):
        """已接受的提交和通知落定后再处理外部取消，保留实际返回值。"""
        task = asyncio.create_task(operation)
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                self._external_cancel = self._external_cancel or exc
                self._cancel_event.set()
            except Exception:
                break
        return task.result()

    async def _notify(self, event: LoopEvent) -> None:
        """交付循环事件，让订阅方异常沿运行边界传播。"""
        await self._accepted(self._emit(deepcopy(event)))

    def _is_cancelled(self) -> bool:
        """读取当前运行的取消事件；外部任务取消由运行边界另行传播。"""
        return self._cancel_event is not None and self._cancel_event.is_set()

    def _raise_if_cancelled(self) -> None:
        """在运行边界将取消信号转换为内部控制流。"""
        if self._external_cancel is not None:
            raise self._external_cancel
        if self._is_cancelled():
            raise _RunCancelled

    async def _commit_unexecuted_calls(self, blocks: list[dict[str, object]], reason: str) -> None:
        """为具有调用身份的未执行工具提交取消结果，保持历史关联闭合。"""
        for block in blocks:
            call_id, name = block.get("call_id"), block.get("name")
            if isinstance(call_id, str) and call_id and isinstance(name, str) and name:
                await self._commit_tool_result(
                    {
                        "call_id": call_id,
                        "name": name,
                        "status": "cancelled",
                        "output": None,
                        "error": reason,
                    }
                )

    async def _commit_tool_result(self, result: ToolResult) -> str:
        """先持久化工具结果，再发出提交与工具结束事件。"""
        entry_id = await self._accepted(self._session.append_tool_result(result))
        await self._notify(
            {
                "type": "message_committed",
                "data": {"run_id": self._run_id, "entry_id": entry_id, "kind": "tool_result"},
            }
        )
        await self._notify(
            {
                "type": "tool_end",
                "data": {
                    "run_id": self._run_id,
                    "tool_call_id": result["call_id"],
                    "tool_name": result["name"],
                    "status": result["status"],
                    "result": result,
                },
            }
        )
        return entry_id

    async def _execute_tools(self, calls: list[ToolCall]) -> list[ToolResult]:
        """按模型顺序调用 tools.execute_tool_call，逐条等待并提交结果。

        普通校验和执行错误记为结果并继续批次；tools 不自动重试。
        steer 不打断批次；主动取消停止启动后续工具并协调未执行结果。
        捕获 ToolStopError 后停止批次，以 error 收尾；不把部分输出保存成已结束
        工具结果，不启动后续模型或工具。停止状态未确认期间拒绝本实例的新运行，
        不能仅结束当前 run 就允许下一次 run 与旧执行重叠。
        """
        results = []
        for index, call in enumerate(calls):
            if self._is_cancelled():
                for remaining in calls[index:]:
                    await self._commit_tool_result(
                        {
                            "call_id": remaining["call_id"],
                            "name": remaining["name"],
                            "status": "cancelled",
                            "output": None,
                            "error": "Tool execution cancelled before starting",
                        }
                    )
                break
            await self._notify(
                {
                    "type": "tool_start",
                    "data": {
                        "run_id": self._run_id,
                        "tool_call_id": call["call_id"],
                        "tool_name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
            )
            try:
                if self._is_cancelled():
                    result = {
                        "call_id": call["call_id"],
                        "name": call["name"],
                        "status": "cancelled",
                        "output": None,
                        "error": "Tool execution cancelled before starting",
                    }
                else:
                    result = await execute_tool_call(
                        call, self._config["tools"], self._tool_context
                    )
            except asyncio.CancelledError as exc:
                self._cancel_event.set()
                if isinstance(exc.__cause__, ToolStopError):
                    raise
                self._external_cancel = self._external_cancel or exc
                result = {
                    "call_id": call["call_id"],
                    "name": call["name"],
                    "status": "cancelled",
                    "output": None,
                    "error": "Tool execution cancelled",
                }
            await self._commit_tool_result(result)
            results.append(result)
        return results

    async def _recover_overflow(self, response: ClientResult, failed_entry_ids: list[str]) -> bool:
        """协调同一连续失败过程最多一次压缩恢复，返回是否可重试主请求。

        仅 finish_reason=error 且 error_kind=context_overflow、自动压缩启用时，
        先提交失败响应及关联结果的有效历史排除，再调用 context 压缩；
        仅成功提交完整摘要后允许重试，不从 error 展示文本重新推断分类。
        压缩失败、取消或 unchanged 不重试；再次超限明确终止恢复。
        不降低近期保留预算，不递归恢复摘要请求，不把普通 length 当成超限。
        新用户消息实际接入或主请求成功后重置本次恢复标记。
        """
        if (
            response["finish_reason"] != "error"
            or response["error_kind"] != "context_overflow"
            or not self._config["compaction"]["enabled"]
            or self._overflow_recovery_attempted
        ):
            return False
        self._raise_if_cancelled()
        self._overflow_recovery_attempted = True
        await self._accepted(self._session.omit_failed_attempt(failed_entry_ids))
        self._raise_if_cancelled()
        history = await self._session.get_history()
        self._raise_if_cancelled()
        context_input = {
            "prompts": self._config["prompts"],
            "history": history,
            "new_instruction": None,
        }
        tokens = estimate_context_tokens(assemble_context(context_input), self._config["budget"])
        result = await self._compact_context(context_input, tokens)
        return result["status"] == "success"

    async def cancel(self) -> None:
        """请求取消当前运行，并等待其所有收尾工作完成。"""
        task = self._running
        signal = self._cancel_event
        if task is None or signal is None:
            return
        signal.set()
        await self.wait_for_idle()

    async def wait_for_idle(self) -> None:
        """等待当前运行（包括会话和事件收尾）进入空闲。"""
        idle = self._idle
        if self._running is not None and idle is not None:
            await idle.wait()
