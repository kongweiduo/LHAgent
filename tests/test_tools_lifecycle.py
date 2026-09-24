"""通过异步事件控制执行与清理边界，验证停止确认和重复取消。"""

import asyncio

import pytest

from lhagent.harness.tools.execution import execute_tool_call
from lhagent.harness.tools.types import ToolStopError
from tests.test_tools_execution import call, context, output, tool


@pytest.mark.parametrize("mode", ["cancel", "zero"])
def test_pre_stop_never_starts_handler(mode):
    """预先取消或超时不启动处理器。"""

    async def scenario():
        env = context()
        if mode == "cancel":
            env["cancel_event"].set()
        else:
            env["timeout_seconds"] = 0

        async def handler(arguments, env):
            pytest.fail("stopped call started")

        result = await execute_tool_call(call(), [tool(handler)], env)
        assert result["status"] == ("cancelled" if mode == "cancel" else "timeout")
        assert result["output"] is None

    asyncio.run(scenario())


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), True, "1"])
def test_invalid_timeout_does_not_start(timeout):
    """非法超时在启动前拒绝。"""

    async def handler(arguments, env):
        pytest.fail("invalid timeout started")

    env = context()
    env["timeout_seconds"] = timeout
    with pytest.raises(ValueError, match="timeout_seconds"):
        asyncio.run(execute_tool_call(call(), [tool(handler)], env))


@pytest.mark.parametrize("trigger", ["event", "timeout", "task"])
@pytest.mark.parametrize("ending", ["return", "reraise", "stop_error", "exception"])
def test_stop_waits_for_cleanup_and_preserves_output_or_failure(trigger, ending):
    """停止等待清理，保留部分输出或清理失败。"""

    async def scenario():
        started, cleaning, release, stopped = [asyncio.Event() for _ in range(4)]
        env = context(size=4)
        if trigger == "timeout":
            env["timeout_seconds"] = 0.01
        partial = output("partial")
        stop_error = ToolStopError(call(), "process still alive", partial)
        invocations = 0

        async def handler(arguments, env):
            nonlocal invocations
            invocations += 1
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as exc:
                cleaning.set()
                await release.wait()
                stopped.set()
                if ending == "return":
                    return partial
                if ending == "stop_error":
                    raise stop_error from exc
                if ending == "exception":
                    raise RuntimeError("cleanup broke") from exc
                raise

        task = asyncio.create_task(execute_tool_call(call(), [tool(handler)], env))
        await started.wait()
        if trigger == "event":
            env["cancel_event"].set()
        elif trigger == "task":
            task.cancel("outer cancellation")
        await asyncio.wait_for(cleaning.wait(), 1)
        assert not task.done()
        assert not stopped.is_set()
        release.set()
        if trigger == "task":
            with pytest.raises(asyncio.CancelledError, match="outer cancellation") as caught:
                await task
            if ending in ("stop_error", "exception"):
                assert isinstance(caught.value.__cause__, ToolStopError)
                if ending == "stop_error":
                    assert caught.value.__cause__ is stop_error
                    assert caught.value.__cause__.output is partial
        elif ending in ("stop_error", "exception"):
            with pytest.raises(ToolStopError) as caught:
                await task
            if ending == "stop_error":
                assert caught.value is stop_error
            else:
                assert isinstance(caught.value.__cause__, RuntimeError)
        else:
            result = await task
            assert result["status"] == ("timeout" if trigger == "timeout" else "cancelled")
            assert result["call_id"] == call()["call_id"]
            if ending == "return":
                assert result["output"]["content"][0]["text"] == "part"
                assert result["output"]["truncated"]
            else:
                assert result["output"] is None
        assert stopped.is_set()
        assert invocations == 1
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("trigger", ["event", "timeout", "task"])
def test_repeated_python_cancel_during_cleanup(trigger, failure):
    """清理期间重复 Python 取消不遗弃处理器。"""

    async def scenario():
        started, cleaning, release = [asyncio.Event() for _ in range(3)]
        env = context()
        if trigger == "timeout":
            env["timeout_seconds"] = 0.01
        error = ToolStopError(call(), "stop unknown", output("partial"))

        async def handler(arguments, env):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                if failure:
                    raise error

        task = asyncio.create_task(execute_tool_call(call(), [tool(handler)], env))
        await started.wait()
        if trigger == "event":
            env["cancel_event"].set()
        elif trigger == "task":
            task.cancel("first")
        await asyncio.wait_for(cleaning.wait(), 1)
        for _ in range(3):
            task.cancel("again")
            await asyncio.sleep(0)
            assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert str(caught.value) == ("first" if trigger == "task" else "again")
        if failure:
            assert caught.value.__cause__ is error
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    asyncio.run(scenario())


def test_completion_wins_event_race():
    """完成与取消事件同时就绪时采用已完成结果。"""

    async def scenario():
        env = context()

        async def handler(arguments, env):
            env["cancel_event"].set()
            return output()

        result = await execute_tool_call(call(), [tool(handler)], env)
        assert result["status"] == "success"
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    asyncio.run(scenario())
