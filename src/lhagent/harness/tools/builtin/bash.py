"""Linux/macOS 上受管理的非交互 Bash 命令。"""

import asyncio
import codecs
import os
import signal
import sys
from pathlib import Path

from ..results import limit_output
from ..types import ToolContext, ToolDefinition, ToolOutput, ToolStopError

# 进程清理与管道读取策略；命令执行时限仍由 ToolContext 提供。
_STOP_CHECK_ATTEMPTS = 100
_STOP_CHECK_INTERVAL_SECONDS = 0.02
_EXIT_POLL_INTERVAL_SECONDS = 0.01
_PIPE_CLOSE_TIMEOUT_SECONDS = 5
_READ_CHUNK_BYTES = 8192
_PROC_ROOT = Path("/proc")


def _linux_group_running(pgid: int) -> bool:
    """读取 Linux procfs；进程名可含空格和括号，Z/X 状态已停止。"""
    for entry in _PROC_ROOT.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            # comm 是括号包裹的任意名称；其后的字段为 state, ppid, pgrp。
            fields = (entry / "stat").read_bytes().rsplit(b")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue  # 枚举后进程已退出。
        if int(fields[2]) == pgid and fields[0] not in (b"Z", b"X"):
            return True
    return False


def create_bash_tool() -> ToolDefinition:
    """构造 Bash 工具定义；调用工厂不会启动进程。"""
    return {
        "name": "bash",
        "description": "Run a non-interactive Bash command in the working directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "minLength": 1},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        "handler": run_bash,
    }


async def _confirm_group_stopped(pgid: int) -> None:
    """确认进程组没有可运行成员；孤儿僵尸已停止，由系统负责回收。"""
    for _ in range(_STOP_CHECK_ATTEMPTS):
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            # macOS 对孤儿僵尸进程组可能拒绝探测，须检查成员状态后再判断清理失败。
            pass
        if sys.platform == "linux":
            if not await asyncio.to_thread(_linux_group_running, pgid):
                return
            await asyncio.sleep(_STOP_CHECK_INTERVAL_SECONDS)
            continue
        probe = await asyncio.create_subprocess_exec(
            "/bin/ps",
            "-axo",
            "pgid=,stat=",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await probe.communicate()
        finally:
            if probe.returncode is None:
                probe.kill()
            await probe.wait()
        if probe.returncode:
            raise RuntimeError(f"Cannot inspect process group: {stderr.decode(errors='replace')}")
        members = [line.split() for line in stdout.decode("ascii").splitlines()]
        if not any(int(group) == pgid and not state.startswith("Z") for group, state in members):
            return
        await asyncio.sleep(_STOP_CHECK_INTERVAL_SECONDS)
    raise RuntimeError(f"Process group {pgid} has not stopped")


async def run_bash(arguments: dict[str, object], context: ToolContext) -> ToolOutput:
    """执行一次，持续排空合并输出；返回前杀死残留进程组并确认停止。

    时限由统一执行入口执行；取消事件也可直接中断 handler。
    不支持主动脱离会话/进程组的守护进程或后台任务。
    """
    output: ToolOutput = {
        "content": [{"type": "text", "text": ""}],
        "details": {"exit_code": None},
        "is_error": False,
        "truncated": False,
    }
    # 启动有副作用的命令前先校验捕获预算。
    limit_output(output, context)
    if sys.platform not in ("linux", "darwin"):
        raise RuntimeError("bash requires Linux or macOS")
    if context["cancel_event"].is_set():
        raise asyncio.CancelledError
    spawn = asyncio.create_task(
        asyncio.create_subprocess_exec(
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            str(arguments["command"]),
            cwd=context["cwd"],
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
            # 非交互 shell 也可能通过 BASH_ENV 执行启动脚本，因此每次启动时过滤。
            env={key: value for key, value in os.environ.items() if key != "BASH_ENV"},
        )
    )
    process = None
    reader = watcher = exited = None

    async def drain():
        nonlocal output
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            chunk = await process.stdout.read(_READ_CHUNK_BYTES)
            text = decoder.decode(chunk, final=not chunk)
            if not output["truncated"]:
                output["content"][0]["text"] += text
                output = limit_output(output, context)
            if not chunk:
                return

    async def wait_exit():
        # shell 退出后继承的管道可能仍被子进程持有，Process.wait() 不一定立即返回。
        while process.returncode is None:
            await asyncio.sleep(_EXIT_POLL_INTERVAL_SECONDS)

    async def cleanup():
        nonlocal process, reader
        if process is None:
            process = await spawn
        if reader is None:
            reader = asyncio.create_task(drain())
        try:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await _confirm_group_stopped(process.pid)
            await asyncio.wait_for(
                asyncio.gather(process.wait(), reader), timeout=_PIPE_CLOSE_TIMEOUT_SECONDS
            )
        except Exception as exc:
            # 即使未托管进程还持有管道副本，也须关闭本端句柄。
            process._transport.close()
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            raise ToolStopError(
                {"call_id": "", "name": "bash", "arguments": arguments},
                f"Cannot confirm bash cleanup: {exc}",
                output,
            ) from exc
        finally:
            for task in (watcher, exited):
                if task is not None:
                    task.cancel()
            await asyncio.gather(
                *(task for task in (watcher, exited) if task is not None), return_exceptions=True
            )

    try:
        process = await asyncio.shield(spawn)
        reader = asyncio.create_task(drain())
        watcher = asyncio.create_task(context["cancel_event"].wait())
        exited = asyncio.create_task(wait_exit())
        done, _ = await asyncio.wait({exited, watcher, reader}, return_when=asyncio.FIRST_COMPLETED)
        if reader in done:
            reader.result()
            done, _ = await asyncio.wait({exited, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if exited not in done:
            raise asyncio.CancelledError
    finally:
        # 直接调用处理器时也屏蔽重复取消，防止清理尚未完成便返回。
        if (
            process is not None
            or not spawn.done()
            or (not spawn.cancelled() and spawn.exception() is None)
        ):
            cleaning = asyncio.create_task(cleanup())
            cancellation = None
            while not cleaning.done():
                try:
                    await asyncio.shield(cleaning)
                except asyncio.CancelledError as exc:
                    cancellation = exc
            cleaning.result()
            if cancellation is not None:
                raise cancellation
    if process is None or process.returncode is None:
        raise ToolStopError(
            {"call_id": "", "name": "bash", "arguments": arguments},
            "Bash process did not produce an exit code",
            output,
        )
    output["details"]["exit_code"] = process.returncode
    output["is_error"] = process.returncode != 0
    return output
