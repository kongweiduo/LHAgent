"""组装 CodingAgent 的配置、惰性依赖及运行入口。"""

import asyncio
from collections.abc import Callable
from copy import deepcopy
from typing import TypedDict
from uuid import uuid4

from lhagent.client.client import Client
from lhagent.client.config import load_config as load_client_config
from lhagent.client.types import ClientResult, Message
from lhagent.harness.configs.loader import load_coding_agent_config
from lhagent.harness.configs.types import CodingAgentConfig
from lhagent.harness.context.budget import estimate_tool_tokens
from lhagent.harness.context.sources import load_prompts
from lhagent.harness.loop.loop import AgentLoop
from lhagent.harness.loop.queue import InputQueue
from lhagent.harness.loop.types import EventSink, LoopConfig, LoopEvent, LoopResult
from lhagent.harness.session.repository import SessionRepository
from lhagent.harness.session.session import Session
from lhagent.harness.tools.builtin import create_builtin_tools
from lhagent.harness.tools.registry import ToolRegistry, describe_tools
from lhagent.harness.tools.types import ToolContext

# Agent 运行环境策略；通信超时由 client.config 独立管理。
_TOOL_TIMEOUT_SECONDS = 120.0
_TOOL_MAX_OUTPUT_LINES = 2000
_TOOL_MAX_OUTPUT_BYTES = 50 * 1024


class CodingAgentOptions(TypedDict, total=False):
    """构造 CodingAgent 的覆盖项与可选外部依赖。

    未提供 session 时由 agent 默认创建；未提供 client 时按配置创建。
    由调用方传入的对象视为外部所有资源，agent 不擅自关闭。
    """

    config: CodingAgentConfig
    client: Client
    session: Session


class CodingAgent:
    """组合单一 coding agent 所需模块，不实现底层循环或工具行为。"""

    def __init__(self, options: CodingAgentOptions | None = None) -> None:
        """保存配置及注入依赖，不执行异步资源创建；首次 run 时完成惰性初始化。

        模型与预算配置交给 loader 统一校验；会话通过异步仓库接口创建。
        初始化失败要清理已创建的自有资源，外部资源所有权保持不变。
        """
        options = options if options is not None else {}
        self._config = deepcopy(load_coding_agent_config(options.get("config")))
        registry = ToolRegistry()
        for tool in create_builtin_tools():
            registry.register(tool)
        self._tools = registry.select(self._config["tools"])
        self._client = options.get("client")
        self._session = options.get("session")
        self._owns_client = self._client is None
        self._owns_session = self._session is None
        self._repository: SessionRepository | None = None
        self._loop: AgentLoop | None = None
        self._queue = InputQueue()
        self._listeners: list[tuple[object, EventSink]] = []
        self._running = False
        self._cancel_requested = False
        self._initializing: asyncio.Task[None] | None = None
        self._completion: asyncio.Future[BaseException | None] | None = None
        self._close_task: asyncio.Task[None] | None = None

    def _ensure_open(self) -> None:
        """关闭开始后拒绝运行、入队和新订阅。"""
        if self._close_task is not None:
            raise RuntimeError("agent is closing or closed")

    async def _initialize(self) -> None:
        """只在全部依赖组装成功后发布实例；失败释放本次创建的资源。"""
        config = self._config
        prompts = load_prompts(
            {
                "system_prompt_path": config["system_prompt_path"],
                "additional_prompt_paths": config["additional_prompt_paths"],
            }
        )
        client = self._client
        session = self._session
        repository = None
        owned_client = None
        try:
            if client is None:
                client = owned_client = Client(load_client_config())
            if session is None:
                # 目录默认值由仓库统一定义：启动目录下的 .lhagent/sessions。
                repository = SessionRepository({})
                session = await repository.create()

            async def summary_request(
                messages: list[Message],
                max_output_tokens: int,
                cancel_event: asyncio.Event,
            ) -> ClientResult:
                parameters = deepcopy(config["generation_parameters"])
                parameters["max_output_tokens"] = max_output_tokens
                return await client.complete(
                    {
                        "call_id": f"call-{uuid4().hex}",
                        "model": config["model"],
                        "messages": messages,
                        "parameters": parameters,
                    },
                    cancel_event=cancel_event,
                )

            loop_config: LoopConfig = {
                "model": config["model"],
                "parameters": {
                    **deepcopy(config["generation_parameters"]),
                    "max_output_tokens": config["max_output_tokens"],
                },
                "prompts": prompts,
                "budget": {
                    "context_window": config["context_window"],
                    "extra_input_tokens": estimate_tool_tokens(describe_tools(self._tools)),
                },
                "compaction": deepcopy(config["compaction"]),
                "summary_request": summary_request,
                "max_summary_output_tokens": config["max_summary_output_tokens"],
                "tools": self._tools,
                "new_call_id": lambda: f"call-{uuid4().hex}",
                "new_run_id": lambda: f"run-{uuid4().hex}",
            }
            tool_context: ToolContext = {
                "cwd": config["cwd"],
                "timeout_seconds": _TOOL_TIMEOUT_SECONDS,
                "max_output_lines": _TOOL_MAX_OUTPUT_LINES,
                "max_output_bytes": _TOOL_MAX_OUTPUT_BYTES,
                "cancel_event": asyncio.Event(),
            }

            async def emit(event: LoopEvent) -> None:
                # 当前事件固定订阅者快照；分发期间的订阅变更
                # 从下一次事件才开始生效。
                for _, listener in tuple(self._listeners):
                    await listener(deepcopy(event))

            loop = AgentLoop(client, session, self._queue, loop_config, tool_context, emit)
        except BaseException as failure:
            # 屏蔽重复任务取消，等待每一个自有资源完成关闭，再传播原始失败。
            async def cleanup() -> list[BaseException | None]:
                """关闭自有资源并收集失败，供外层补充原始异常。"""
                resources = [
                    resource for resource in (repository, owned_client) if resource is not None
                ]
                return await asyncio.gather(
                    *(resource.close() for resource in resources), return_exceptions=True
                )

            task = asyncio.create_task(cleanup())
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            for result in task.result():
                if isinstance(result, BaseException):
                    failure.add_note(f"initialization cleanup failed: {result!r}")
            raise
        self._client = client
        self._session = session
        self._repository = repository
        self._loop = loop

    async def run(self, instruction: str) -> LoopResult:
        """完成初始化后委托 loop 运行，返回明确终态；并发 run 明确报错。

        工具停止未确认时保留 loop 的拒绝运行状态，不通过重新初始化绕过。
        """
        self._ensure_open()
        if self._running:
            raise RuntimeError("agent is already running")
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string")
        self._running = True
        self._cancel_requested = False
        completion = self._completion = asyncio.get_running_loop().create_future()
        failure = None
        try:
            if self._loop is None:
                self._initializing = asyncio.create_task(self._initialize())
                try:
                    await _settle(self._initializing, cancel_on_interrupt=True)
                finally:
                    self._initializing = None
            if self._cancel_requested:
                return {"status": "cancelled", "last_response": None, "error": None}
            return await self._loop.run(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": instruction}],
                }
            )
        except BaseException as exc:
            failure = exc
            raise
        finally:
            self._running = False
            completion.set_result(failure)

    def subscribe(self, listener: EventSink) -> Callable[[], None]:
        """订阅 loop 事件并返回幂等取消订阅函数，允许在首次运行前订阅。

        每个事件按订阅顺序交付并等待异步处理；订阅只做展示和通知，不保存
        会话。不在回调内等待 cancel/wait_for_idle/close，避免等待自身结束；
        控制操作由外部任务发起。订阅异常交付运行调用方，loop 仍完成必要收尾。
        """
        self._ensure_open()
        if not callable(listener):
            raise TypeError("listener must be callable")
        token = object()
        self._listeners.append((token, listener))
        subscribed = True

        def unsubscribe() -> None:
            nonlocal subscribed
            if not subscribed:
                return
            subscribed = False
            self._listeners[:] = [entry for entry in self._listeners if entry[0] is not token]

        return unsubscribe

    def steer(self, instruction: str) -> None:
        """将用户输入加入 steering 队列，在响应及工具批次结束后接入。

        空闲时允许入队，但不自动启动运行；取消不清空尚未消费的输入。
        """
        self._ensure_open()
        self._queue.steer(self._user_message(instruction))

    def follow_up(self, instruction: str) -> None:
        """将用户输入加入 follow-up 队列，仅在本来可以正常结束时接入。"""
        self._ensure_open()
        self._queue.follow_up(self._user_message(instruction))

    def clear_queue(self) -> dict[str, list[dict[str, object]]]:
        """清空并返回 steer、follow_up 两类尚未消费的结构化消息。"""
        return self._queue.clear()

    @staticmethod
    def _user_message(instruction: str) -> dict[str, object]:
        """校验公开文本输入并构造统一用户消息。"""
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string")
        return {"role": "user", "content": [{"type": "text", "text": instruction}]}

    async def wait_for_idle(self) -> None:
        """等待初始化、当前运行、会话提交和已接受的事件处理完成。"""
        completion = self._completion
        if self._running and completion is not None:
            await asyncio.shield(completion)

    async def cancel(self) -> None:
        """取消本 agent 的初始化或当前运行，等待清理后传播调用方取消。

        不取消共享 client 的其他调用，不清空尚未消费的输入队列。
        """
        completion = self._completion
        if not self._running or completion is None:
            return
        self._cancel_requested = True
        initializing = self._initializing
        if initializing is not None:
            if not initializing.done() and not initializing.cancelling():
                initializing.cancel()
        loop = self._loop if initializing is None else None

        async def finish() -> None:
            if loop is not None and not completion.done():
                await loop.cancel()
            failure = await asyncio.shield(completion)
            if failure is not None:
                if isinstance(failure, asyncio.CancelledError) and not (
                    failure.__cause__ or getattr(failure, "__notes__", None)
                ):
                    return
                raise failure

        await _settle(asyncio.create_task(finish()))

    async def close(self) -> None:
        """关闭前取消并等待本 agent 的初始化、运行和事件处理结束。

        关闭自有资源，不关闭外部共享 client/session；取消仅针对本 agent。
        重复关闭保持幂等；关闭后不接受 run、入队或新订阅。
        """
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._finish_close())
        await _settle(self._close_task)

    async def _finish_close(self) -> None:
        """先取消本 agent，再关闭自有资源；借用的 client/session 保持可用。"""
        failure = None
        try:
            await self.cancel()
        except BaseException as exc:
            failure = exc
        resources = []
        if self._owns_session and self._repository is not None:
            resources.append(self._repository)
        if self._owns_client and self._client is not None:
            resources.append(self._client)
        results = await asyncio.gather(
            *(resource.close() for resource in resources), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                if failure is None:
                    failure = result
                else:
                    failure.add_note(f"agent resource cleanup failed: {result!r}")
        if failure is not None:
            raise failure


async def _settle(task: asyncio.Task, *, cancel_on_interrupt: bool = False):
    """等待清理落定再传播调用方取消，并保留清理失败信息。"""
    interrupted = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if asyncio.current_task().cancelling():
                interrupted = interrupted or exc
                if cancel_on_interrupt and not task.done() and not task.cancelling():
                    task.cancel()
        except BaseException:
            break
    if interrupted is not None:
        try:
            task.result()
        except BaseException as exc:
            if exc is not interrupted:
                interrupted.__cause__ = exc
        raise interrupted
    return task.result()


def create_coding_agent(
    options: CodingAgentOptions | None = None,
) -> CodingAgent:
    """同步解析配置；异步资源在首次 run 时创建。"""
    return CodingAgent(options)
