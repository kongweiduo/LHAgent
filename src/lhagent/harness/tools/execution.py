"""执行单个工具调用，协调超时、取消与停止确认；不重试。"""

import asyncio
import math
from typing import cast

from .results import limit_output, result_from_output, validate_output
from .types import (
    PreparedToolCall,
    ToolCall,
    ToolContext,
    ToolDefinition,
    ToolResult,
    ToolStopError,
)
from .validation import prepare_tool_call


async def _join(task: asyncio.Task, cancellation: asyncio.CancelledError | None):
    """屏蔽后续取消直到子任务结束，并保存调用者的取消信息。"""
    owner = asyncio.current_task()
    if owner is None:
        raise RuntimeError("tool execution must run in an asyncio task")
    count = owner.cancelling()
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if owner.cancelling() > count:
                cancellation = cancellation or exc
                count = owner.cancelling()
        except Exception:
            # 由调用方读取 result，区分执行失败与停止未确认。
            pass
    return cancellation


async def execute_prepared_call(prepared: PreparedToolCall, context: ToolContext) -> ToolResult:
    """执行一次；取消后等待 handler 完成其清理，再报告停止。

    handler 必须协作取消：返回/抛出前停止其拥有的资源；无法确认停止时
    抛出 ToolStopError。清理没有额外截止时间，不强行遗弃仍在运行的任务。
    """
    call = prepared["call"]

    def result(status, error, output=None):
        return {
            "call_id": call["call_id"],
            "name": call["name"],
            "status": status,
            "output": output,
            "error": error,
        }

    timeout = context["timeout_seconds"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout < 0
    ):
        raise ValueError("timeout_seconds must be a finite non-negative number")
    if context["cancel_event"].is_set():
        return result("cancelled", "Tool execution cancelled before starting")
    if timeout == 0:
        return result("timeout", "Tool execution timed out before starting")

    async def invoke():
        return await prepared["tool"]["handler"](prepared["arguments"], context)

    execution = asyncio.create_task(invoke())
    watcher = asyncio.create_task(context["cancel_event"].wait())
    cancellation = None
    status = None
    try:
        done, _ = await asyncio.wait(
            {execution, watcher},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        # 同时就绪时，已结束的执行优先；Python 任务取消始终传播。
        if execution not in done:
            status = "cancelled" if watcher in done else "timeout"
    except asyncio.CancelledError as exc:
        cancellation = exc
        status = "cancelled"

    if status is not None and not execution.done():
        execution.cancel()
    cancellation = await _join(execution, cancellation)
    watcher.cancel()
    cancellation = await _join(watcher, cancellation)

    output = None
    failure: Exception | None = None
    try:
        raw_output = execution.result()
    except asyncio.CancelledError as exc:
        if status is None:
            cancellation = cancellation or exc
    except ToolStopError as exc:
        # 处理器没有调用身份，停止失败必须补回准备阶段的真实调用信息。
        exc.call = call
        failure = exc
    except Exception as exc:
        if status is not None:
            failure = ToolStopError(call, f"Tool cleanup failed: {type(exc).__name__}: {exc}")
            failure.__cause__ = exc
        else:
            failure = exc
    else:
        try:
            output = limit_output(validate_output(raw_output), context)
        except Exception as exc:
            failure = exc

    if cancellation is not None:
        if failure is not None:
            raise cancellation from failure
        raise cancellation
    if isinstance(failure, ToolStopError):
        raise failure
    if status is not None:
        message = "Tool execution timed out" if status == "timeout" else "Tool execution cancelled"
        if failure is not None:
            message += f"; invalid output: {type(failure).__name__}: {failure}"
        return result(status, message, output)
    if failure is not None:
        return result("execution_error", f"{type(failure).__name__}: {failure}")
    if output is None:
        return result("execution_error", "Tool produced no output")
    return result_from_output(call, output)


async def execute_tool_call(
    call: ToolCall, active_tools: list[ToolDefinition], context: ToolContext
) -> ToolResult:
    """校验可用性及参数后执行一次；loop 负责按序等待，不建立全局锁。"""
    prepared = prepare_tool_call(call, active_tools)
    if "status" in prepared:
        return cast(ToolResult, prepared)
    return await execute_prepared_call(cast(PreparedToolCall, prepared), context)
