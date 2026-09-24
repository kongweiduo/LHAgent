"""单指令与交互终端入口，负责参数解析、信号转发和退出码。"""

import argparse
import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path

from lhagent.agents import CodingAgentOptions, create_coding_agent
from lhagent.diagnostics import diagnostic
from lhagent.harness.configs import load_coding_agent_config
from lhagent.harness.configs.types import CodingAgentConfig
from lhagent.harness.session import SessionRepository
from lhagent.tui import run_interactive


async def _run(config_path: str, instruction: str | None, session_path: str | None) -> int:
    """加载共享配置，再交给所选模式管理运行生命周期。"""
    if instruction is None and not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise RuntimeError(
            "interactive mode requires a TTY on stdin and stdout; "
            "use --instruction TEXT for pipes or automation"
        )
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    config = load_coding_agent_config(config_path=str(path))
    if instruction is not None:
        return await _run_once(config, instruction, session_path)
    return await _run_interactive(config, session_path)


async def _run_interactive(config: CodingAgentConfig, session_path: str | None) -> int:
    # 应用负责 agent、仓库/会话及输入终端资源的清理。
    """启动交互应用，由应用负责 agent、会话及终端资源清理。"""
    await run_interactive(config, session_path=session_path)
    return 0


async def _run_once(config: CodingAgentConfig, instruction: str, session_path: str | None) -> int:
    """执行单条指令并输出最终文本；无论成功与否均释放自有资源。"""
    repository = None
    agent = None
    try:
        options: CodingAgentOptions = {"config": config}
        if session_path is not None:
            path = Path(session_path).expanduser().resolve()
            repository = SessionRepository({"directory": str(path.parent)})
            matches = [item for item in await repository.list() if item["path"] == str(path)]
            if not matches:
                raise FileNotFoundError(f"session not found: {path}")
            options["session"] = await repository.open(matches[0])
        else:
            repository = SessionRepository({})
            options["session"] = await repository.create()
        print(f"Session: {options['session'].metadata['path']}", file=sys.stderr)
        agent = create_coding_agent(options)
        result = await agent.run(instruction)
        response = result["last_response"]
        if response is not None:
            text = "".join(
                block["text"] for block in response["content"] if block["type"] == "text"
            )
            if text:
                print(text)
        if result["status"] != "completed":
            print(
                diagnostic(f"run {result['status']}: {result['error'] or 'no final answer'}"),
                file=sys.stderr,
            )
            return 1
        return 0
    finally:
        try:
            if agent is not None:
                await agent.close()
        finally:
            if repository is not None:
                await repository.close()


async def _inspect_sessions(session_path: str | None) -> int:
    """列出已保存会话或展示历史对话，无需创建模型客户端。"""
    from prompt_toolkit.formatted_text import fragment_list_to_text

    from lhagent.tui.history import project_history
    from lhagent.tui.rendering import format_block

    path = Path(session_path).expanduser().resolve() if session_path else None
    repository = SessionRepository({"directory": str(path.parent)} if path else {})
    try:
        sessions = await repository.list()
        if path is None:
            for item in sessions:
                created = (
                    datetime.fromtimestamp(item["created_at"])
                    .astimezone()
                    .isoformat(timespec="seconds")
                )
                print(f"{created}\t{item['path']}")
            if not sessions:
                print("No saved sessions")
            return 0
        metadata = next((item for item in sessions if item["path"] == str(path)), None)
        if metadata is None:
            raise FileNotFoundError(f"session not found: {path}")
        session = await repository.open(metadata)
        state = await session.state()
        print(f"Session: {path}\nState: {state['state']}")
        display = project_history(await session.get_display_history(), state)
        for block in display.transcript:
            print(fragment_list_to_text(format_block(block)))
            print()
        return 0
    finally:
        await repository.close()


async def _run_with_signals(
    config_path: str, instruction: str | None, session_path: str | None
) -> int:
    """POSIX 终止信号只触发一次取消；清理结束前保留处理器。"""
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    previous = {}
    received: signal.Signals | None = None

    def stop(signum: signal.Signals) -> None:
        nonlocal received
        if received is None:
            received = signum
            assert task is not None
            task.cancel()

    try:
        for name in ("SIGTERM", "SIGHUP"):
            signum = getattr(signal, name, None)
            if signum is not None:
                try:
                    handler = signal.getsignal(signum)
                    loop.add_signal_handler(signum, stop, signum)
                    previous[signum] = handler
                except (NotImplementedError, RuntimeError, ValueError):
                    pass  # 非主线程嵌入或非 POSIX 循环可能不支持信号处理器。
        try:
            status = await _run(config_path, instruction, session_path)
            if received is not None:
                raise RuntimeError(f"terminated by {received.name}")
            return status
        except asyncio.CancelledError:
            if received is None:
                raise
            raise RuntimeError(f"terminated by {received.name}") from None
    finally:
        for signum, handler in previous.items():
            loop.remove_signal_handler(signum)
            signal.signal(signum, handler)


def main() -> None:
    """解析命令行并运行入口；参数、运行失败和正常结束使用既定退出码。"""
    parser = argparse.ArgumentParser(
        description="Run one LHAgent instruction or start an interactive terminal session",
        epilog="Exit codes: 0 normal completion/interactive quit; 1 configuration, session, "
        "run or terminal failure; 2 invalid command-line arguments.",
    )
    parser.add_argument(
        "--config", help="Required path to lhagent.toml (model and capacity must be configured)"
    )
    parser.add_argument(
        "--instruction",
        help="Run this instruction once and exit; omit for interactive TUI (stdin/stdout TTY required)",
    )
    parser.add_argument(
        "--session", help="Existing valid session JSONL path to resume in either mode"
    )
    inspection = parser.add_mutually_exclusive_group()
    inspection.add_argument(
        "--list-sessions", action="store_true", help="List saved sessions without calling the model"
    )
    inspection.add_argument(
        "--history",
        action="store_true",
        help="Show saved conversation from --session without calling the model",
    )
    args = parser.parse_args()
    if args.history and not args.session:
        parser.error("--history requires --session PATH")
    if (args.history or args.list_sessions) and args.instruction is not None:
        parser.error("session inspection cannot be combined with --instruction")
    if args.list_sessions and args.session:
        parser.error("--list-sessions cannot be combined with --session")
    if not (args.history or args.list_sessions) and not args.config:
        parser.error("--config is required when running the agent")
    try:
        if args.history or args.list_sessions:
            status = asyncio.run(_inspect_sessions(args.session))
        else:
            status = asyncio.run(_run_with_signals(args.config, args.instruction, args.session))
    except (Exception, KeyboardInterrupt) as exc:
        try:
            print(f"lhagent: {diagnostic(exc) or 'interrupted'}", file=sys.stderr)
        except OSError:
            pass  # 终端断开时 stderr 也可能不可写。
        status = 1
    raise SystemExit(status)
