"""命令输出预算与进程组清理；需要 Linux/macOS 的 Bash 和进程检查能力。"""

import asyncio
import itertools
import os
import shlex
import subprocess
import sys

import pytest

from lhagent.harness.tools.builtin import create_builtin_tools
from lhagent.harness.tools.builtin.bash import create_bash_tool
from lhagent.harness.tools.execution import execute_tool_call
from lhagent.harness.tools.registry import ToolRegistry, describe_tools
from lhagent.harness.tools.types import ToolStopError


def context(path, **overrides):
    """创建独立取消信号及有限输出预算的临时工具环境。"""
    return {
        "cwd": str(path),
        "timeout_seconds": 5,
        "cancel_event": asyncio.Event(),
        "max_output_lines": 20,
        "max_output_bytes": 1024,
        **overrides,
    }


async def invoke(command, ctx):
    """通过统一执行入口运行一次 Bash 工具调用。"""
    return await execute_tool_call(
        {"call_id": "bash-call", "name": "bash", "arguments": {"command": command}},
        [create_bash_tool()],
        ctx,
    )


def python_command(code):
    """以 shell 引用安全地生成当前 Python 解释器命令。"""
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def test_all_builtin_subsets():
    """内置工具集合的所有子集均可选择。"""
    tools = create_builtin_tools()
    names = [tool["name"] for tool in tools]
    assert names == ["read", "write", "edit", "bash", "ls", "find", "grep"]
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    for size in range(8):
        for subset in itertools.combinations(names, size):
            selected = registry.select(list(subset))
            assert [tool["name"] for tool in selected] == list(subset)
            assert all("handler" not in tool for tool in describe_tools(selected))
    assert create_builtin_tools()[0] is not tools[0]


@pytest.mark.skipif(sys.platform not in ("linux", "darwin"), reason="POSIX bash")
def test_success_failure_cwd_and_bad_arguments(tmp_path):
    """退出状态、工作目录及非法参数有明确结果。"""

    async def scenario():
        result = await invoke(
            "printf hello; printf error >&2; printf data > file; exit 7", context(tmp_path)
        )
        assert result["status"] == "execution_error"
        assert result["output"]["details"]["exit_code"] == 7
        assert result["output"]["content"][0]["text"] == "helloerror"
        assert (tmp_path / "file").read_text() == "data"
        assert (await invoke("[[ -f file ]]", context(tmp_path)))["status"] == "success"
        assert (await invoke("", context(tmp_path)))["status"] == "validation_error"
        assert (await invoke("true", context(tmp_path / "missing")))["status"] == "execution_error"

    asyncio.run(scenario())


@pytest.mark.parametrize("lines,byte_limit", [(2, 1024), (20, 5), (0, 100), (20, 0)])
def test_large_output_drains_both_streams(tmp_path, lines, byte_limit):
    """大输出持续排空管道，不因预算耗尽阻塞子进程。"""
    code = "import os; [(os.write(1, b'x'*8192), os.write(2, b'y\\n'*4096)) for _ in range(100)]"
    result = asyncio.run(
        invoke(
            python_command(code),
            context(tmp_path, max_output_lines=lines, max_output_bytes=byte_limit),
        )
    )
    assert result["status"] == "success"
    output = result["output"]
    assert output["truncated"]
    text = output["content"][0]["text"]
    assert len(text.encode()) <= byte_limit
    assert len(text.splitlines()) <= lines


def test_split_utf8_and_exact_budget(tmp_path):
    """分块 UTF-8 解码和精确输出预算不损坏文本。"""
    code = "import os,time; os.write(1,b'\\xe4'); time.sleep(.03); os.write(1,b'\\xbd\\xa0\\n')"
    result = asyncio.run(
        invoke(python_command(code), context(tmp_path, max_output_bytes=4, max_output_lines=1))
    )
    assert result["output"]["content"][0]["text"] == "你\n"
    assert not result["output"]["truncated"]


def stopped(pid):
    """通过 ps 判断进程不存在或已为僵尸，避免把停止和回收混为一谈。"""
    result = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True
    )
    return not result.stdout.strip() or result.stdout.strip().startswith("Z")


@pytest.mark.parametrize("mode", ["timeout", "event", "task", "normal"])
def test_child_processes_are_stopped(tmp_path, mode):
    # Child ignores TERM and does not hold the capture pipe, so EOF alone proves nothing.
    """取消或命令结束后确认子进程组已停止。"""
    command = "(trap '' TERM; while :; do sleep 1; done) >/dev/null 2>&1 & echo $! > child; echo $$ > shell; "
    command += "exit 0" if mode == "normal" else "wait"

    async def scenario():
        ctx = context(tmp_path, timeout_seconds=0.3 if mode == "timeout" else 5)
        task = asyncio.create_task(invoke(command, ctx))
        try:
            async with asyncio.timeout(3):
                while not (tmp_path / "shell").exists():
                    await asyncio.sleep(0.01)
            if mode == "event":
                ctx["cancel_event"].set()
            elif mode == "task":
                task.cancel()
            if mode == "task":
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                result = await task
                assert (
                    result["status"]
                    == {"normal": "success", "timeout": "timeout", "event": "cancelled"}[mode]
                )
            for filename in ("shell", "child"):
                assert stopped(int((tmp_path / filename).read_text()))
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            for filename in ("shell", "child"):
                path = tmp_path / filename
                if path.exists() and not stopped(int(path.read_text())):
                    os.kill(int(path.read_text()), 9)

    asyncio.run(scenario())


def test_stop_failure_retains_call_identity(tmp_path, monkeypatch):
    """停止无法确认时保留真实工具调用身份。"""
    from lhagent.harness.tools.builtin import bash

    async def failed_confirmation(pgid):
        raise RuntimeError("inspection unavailable")

    monkeypatch.setattr(bash, "_confirm_group_stopped", failed_confirmation)
    with pytest.raises(ToolStopError) as caught:
        asyncio.run(invoke("true", context(tmp_path)))
    assert caught.value.call["call_id"] == "bash-call"
    assert "inspection unavailable" in str(caught.value)


@pytest.mark.parametrize(
    "states,stopped_group", [(b"123 Z\n", True), (b"", True), (b"123 S\n", False)]
)
def test_permission_denied_probe_still_checks_process_states(monkeypatch, states, stopped_group):
    """进程组探测权限不足时仍检查成员状态，不把活进程误判为已停止。"""
    from unittest.mock import AsyncMock

    from lhagent.harness.tools.builtin import bash

    def denied(*args):
        raise PermissionError("probe denied")

    probe = AsyncMock()
    probe.returncode = 0
    probe.communicate.return_value = (states, b"")
    monkeypatch.setattr(bash.sys, "platform", "darwin")
    monkeypatch.setattr(bash.os, "killpg", denied)
    monkeypatch.setattr(bash.asyncio, "create_subprocess_exec", AsyncMock(return_value=probe))
    monkeypatch.setattr(bash, "_STOP_CHECK_ATTEMPTS", 1)
    monkeypatch.setattr(bash, "_STOP_CHECK_INTERVAL_SECONDS", 0)
    if stopped_group:
        asyncio.run(bash._confirm_group_stopped(123))
    else:
        with pytest.raises(RuntimeError, match="has not stopped"):
            asyncio.run(bash._confirm_group_stopped(123))
    probe.communicate.assert_awaited_once()


def test_linux_process_inspection_handles_names_and_zombies(tmp_path, monkeypatch):
    """无需 ps；名称中的右括号不能使 pgrp 错位，僵尸不阻塞回收。"""
    from lhagent.harness.tools.builtin import bash

    monkeypatch.setattr(bash, "_PROC_ROOT", tmp_path)
    for pid, state, group in [(101, "S", 123), (102, "Z", 456), (103, "X", 456)]:
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "stat").write_bytes(
            f"{pid} (name with ) parentheses".encode() + b"\xff) " + f"{state} 1 {group} 0".encode()
        )
    (tmp_path / "104").mkdir()  # 模拟 stat 读取前已退出。
    (tmp_path / "self").mkdir()
    assert bash._linux_group_running(123)
    assert not bash._linux_group_running(456)
    assert not bash._linux_group_running(789)


def test_linux_inspection_failure_is_not_reported_as_stopped(tmp_path, monkeypatch):
    from lhagent.harness.tools.builtin import bash

    monkeypatch.setattr(bash, "_PROC_ROOT", tmp_path / "missing-proc")
    with pytest.raises(FileNotFoundError):
        bash._linux_group_running(123)


def test_pre_cancel_and_zero_timeout_do_not_start(tmp_path):
    """预取消和零超时不启动有副作用命令。"""

    async def scenario():
        ctx = context(tmp_path)
        ctx["cancel_event"].set()
        assert (await invoke("touch forbidden", ctx))["status"] == "cancelled"
        assert (await invoke("touch forbidden", context(tmp_path, timeout_seconds=0)))[
            "status"
        ] == "timeout"
        assert not (tmp_path / "forbidden").exists()

    asyncio.run(scenario())
