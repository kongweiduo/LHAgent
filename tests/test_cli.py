"""命令行模式分派、退出状态和 POSIX 伪终端交互；使用假模型及临时会话。"""

import asyncio
import os
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lhagent import cli
from lhagent.harness.session import SessionRepository
from lhagent.tui.app import InteractiveApp
from lhagent.tui.input import InputEditor
from tests.test_tui_app import Agent, Repository, Session

ROOT = Path(__file__).resolve().parents[1]
# Execute the installed console script with only the network client replaced.
BOOTSTRAP = """
import runpy, shutil
from uuid import uuid4
from lhagent.agents import coding
from tests.test_coding_agent import Client
from tests.test_loop_tools import call, response
coding.load_client_config = lambda: object()
coding.Client = lambda _: Client([
    response("tool_call", call(uuid4().hex, name="read", arguments={"path": "source.txt"})),
    response("stop", {"type": "text", "text": "offline answer"}),
])
runpy.run_path(shutil.which("lhagent"), run_name="__main__")
"""


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """隔离工作目录与测试子进程环境，创建离线配置及工具输入。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.setenv("PROMPT_TOOLKIT_NO_CPR", "1")
    (tmp_path / "lhagent.toml").write_text(
        '[coding_agent]\nmodel = "test"\ncontext_window = 10000\n'
        'max_output_tokens = 1000\ntools = ["read"]\n',
        encoding="utf-8",
    )
    (tmp_path / "source.txt").write_text("offline tool content", encoding="utf-8")
    return tmp_path


def invoke(monkeypatch, *args):
    """临时设置 argv 调用 CLI，并返回 SystemExit 中的退出码。"""
    monkeypatch.setattr(sys, "argv", ["lhagent", *args])
    with pytest.raises(SystemExit) as result:
        cli.main()
    return result.value.code


def test_mode_dispatch_and_help(workspace, monkeypatch, capsys):
    """两种模式按参数分派，帮助入口无需运行资源。"""
    once, interactive = AsyncMock(return_value=0), AsyncMock(return_value=0)
    monkeypatch.setattr(cli, "_run_once", once)
    monkeypatch.setattr(cli, "_run_interactive", interactive)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert invoke(monkeypatch, "--config", "lhagent.toml", "--session", "saved.jsonl") == 0
    assert interactive.await_args.args[1] == "saved.jsonl"
    once.assert_not_awaited()
    assert invoke(monkeypatch, "--config", "lhagent.toml", "--instruction", "") == 0
    assert once.await_args.args[1:] == ("", None)
    assert invoke(monkeypatch, "--help") == 0
    help_text = capsys.readouterr().out
    assert "TTY" in help_text and "--instruction" in help_text and "Required path" in help_text
    assert invoke(monkeypatch) == 2


@pytest.mark.parametrize("stdin_tty,stdout_tty", [(False, False), (True, False), (False, True)])
def test_non_tty_interactive_fails_before_creating_resources(
    workspace, monkeypatch, capsys, stdin_tty, stdout_tty
):
    """非 TTY 的交互请求在创建资源前失败。"""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: stdin_tty)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: stdout_tty)
    assert invoke(monkeypatch, "--config", "lhagent.toml") == 1
    assert "use --instruction" in capsys.readouterr().err
    assert not (workspace / ".lhagent").exists()


@pytest.mark.parametrize("instruction", [None, "hello"])
@pytest.mark.parametrize("broken", ["missing", "header", "body"])
def test_session_failures_in_both_modes(workspace, monkeypatch, capsys, instruction, broken):
    """两种模式均报告会话打开失败。"""
    path = workspace / "sessions" / "saved.jsonl"
    if broken == "header":
        path.parent.mkdir()
        path.write_text("{}\n", encoding="utf-8")
    elif broken == "body":

        async def create():
            repo = SessionRepository({"directory": str(path.parent)})
            try:
                await repo.create({"id": "saved"})
            finally:
                await repo.close()

        asyncio.run(create())
        with path.open("a") as stream:
            stream.write("{}\n")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    # Use actual application/session lifecycle, substituting only terminal IO.
    async def interactive(config, session_path=None):
        with create_pipe_input() as pipe:
            await InteractiveApp(
                config,
                session_path=session_path,
                editor_factory=lambda footer: InputEditor(footer, input=pipe, output=DummyOutput()),
            ).run()

    monkeypatch.setattr(cli, "run_interactive", interactive)
    args = ["--config", "lhagent.toml", "--session", str(path)]
    if instruction is not None:
        args += ["--instruction", instruction]
    assert invoke(monkeypatch, *args) == 1
    assert "lhagent:" in capsys.readouterr().err
    assert not (workspace / ".lhagent" / "sessions").exists()


def test_terminal_initialization_failure_closes_resources(workspace, monkeypatch, capsys):
    """终端初始化失败也关闭已创建资源。"""
    agent = Agent()
    session = Session()
    session.close = AsyncMock()
    repo = Repository(session)
    repo.close = AsyncMock()

    class BrokenEditor(InputEditor):
        async def run(self):
            try:
                raise OSError("terminal initialization failed")
            finally:
                self.ready.set()

    async def interactive(config, session_path=None):
        with create_pipe_input() as pipe:
            await InteractiveApp(
                config,
                repository=repo,
                agent_factory=lambda _: agent,
                editor_factory=lambda footer: BrokenEditor(
                    footer, input=pipe, output=DummyOutput()
                ),
            ).run()

    monkeypatch.setattr(cli, "run_interactive", interactive)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert invoke(monkeypatch, "--config", "lhagent.toml") == 1
    assert "terminal initialization failed" in capsys.readouterr().err
    assert agent.calls[-2:] == ["close", "unsubscribe"]
    session.close.assert_awaited_once()
    repo.close.assert_awaited_once()


@pytest.mark.parametrize("instruction", [None, "hello"])
def test_fatal_diagnostics_redact_authorization(workspace, monkeypatch, capsys, instruction):
    """致命诊断不泄漏 Authorization 凭据。"""
    monkeypatch.setattr(
        cli,
        "_run",
        AsyncMock(
            side_effect=RuntimeError(
                "request failed: Authorization: Basic fake-credential; retry later"
            )
        ),
    )
    args = ["--config", "lhagent.toml"]
    if instruction is not None:
        args += ["--instruction", instruction]
    assert invoke(monkeypatch, *args) == 1
    error = capsys.readouterr().err
    assert "fake-credential" not in error
    assert "request failed:" in error and "retry later" in error


def test_console_pipes_and_resume(workspace):
    """真实 console 入口支持管道指令及会话恢复。"""
    command = [sys.executable, "-c", BOOTSTRAP, "--config", "lhagent.toml"]
    first = subprocess.run(
        [*command, "--instruction", "read"], capture_output=True, text=True, timeout=10
    )
    assert first.returncode == 0, first.stderr
    assert first.stdout.strip() == "offline answer"
    paths = list((workspace / ".lhagent" / "sessions").glob("*.jsonl"))
    assert len(paths) == 1
    second = subprocess.run(
        [*command, "--session", str(paths[0]), "--instruction", "again"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert second.returncode == 0, second.stderr
    assert len(list(paths[0].parent.glob("*.jsonl"))) == 1
    failed = subprocess.run(
        [shutil.which("lhagent"), "--config", "lhagent.toml"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert failed.returncode == 1 and "--instruction" in failed.stderr


def test_session_inspection_without_model_config(workspace, monkeypatch, capsys):
    """历史查询无需模型配置，并且保持已完成会话的文件内容不变。"""
    command = [sys.executable, "-c", BOOTSTRAP, "--config", "lhagent.toml", "--instruction", "read"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    path = next((workspace / ".lhagent" / "sessions").glob("*.jsonl"))
    assert str(path) in result.stderr
    before = path.read_bytes()
    (workspace / "lhagent.toml").unlink()
    assert invoke(monkeypatch, "--list-sessions") == 0
    assert str(path) in capsys.readouterr().out
    assert invoke(monkeypatch, "--history", "--session", str(path)) == 0
    output = capsys.readouterr().out
    assert "user:" in output and "offline answer" in output and "offline tool content" in output
    assert path.read_bytes() == before
    assert invoke(monkeypatch, "--history") == 2
    assert invoke(monkeypatch, "--list-sessions", "--instruction", "oops") == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX pseudo-terminal smoke test")
@pytest.mark.parametrize(
    "close_key,resume",
    [
        (b"/quit\r", False),
        (b"\x04", True),
        (signal.SIGTERM, False),
        (signal.SIGHUP, False),
    ],
)
def test_console_pty_text_tool_new_resume_and_exit(workspace, monkeypatch, close_key, resume):
    """伪终端覆盖文本、工具、新建、恢复及多种退出方式。"""
    import fcntl
    import pty
    import struct
    import termios

    monkeypatch.setenv("TERM", "dumb")

    args = ["--config", "lhagent.toml"]
    if resume:

        async def seed():
            repo = SessionRepository({})
            try:
                return (await repo.create()).metadata["path"]
            finally:
                await repo.close()

        args += ["--session", asyncio.run(seed())]
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 140, 0, 0))
    process = subprocess.Popen(
        [sys.executable, "-c", BOOTSTRAP, *args],
        stdin=slave,
        stdout=slave,
        stderr=slave,
    )
    os.close(slave)
    transcript = bytearray()

    def expect(text):
        expected = text.encode()
        start = len(transcript)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                transcript.extend(chunk)
                if expected in transcript[start:]:
                    return
            if process.poll() is not None:
                break
        pytest.fail(f"Did not see {text!r}: {transcript.decode(errors='replace')}")

    try:
        expect(">")
        deadline = time.monotonic() + 10
        while termios.tcgetattr(master)[3] & termios.ICANON:
            assert time.monotonic() < deadline, "input editor did not enter raw mode"
            time.sleep(0.01)
        os.write(master, b"read source\r")
        expect("run_end: completed")
        assert b"offline answer" in transcript and b"offline tool content" in transcript
        paths = list((workspace / ".lhagent" / "sessions").glob("*.jsonl"))
        assert len(paths) == 1
        original_id = paths[0].stem
        os.write(master, b"/new\r")
        expect("Session:")
        assert len(list(paths[0].parent.glob("*.jsonl"))) == 2
        os.write(master, b"/resume\r")
        expect("Resume session")
        os.write(master, b"2\r")  # Newest first; original session is second.
        expect(original_id)
        if isinstance(close_key, signal.Signals):
            process.send_signal(close_key)
        else:
            os.write(master, close_key)
        deadline = time.monotonic() + 10
        while process.poll() is None and time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    transcript.extend(os.read(master, 65536))
                except OSError:
                    break
        expected_status = 1 if isinstance(close_key, signal.Signals) else 0
        assert process.wait(timeout=1) == expected_status, transcript.decode(errors="replace")
        if expected_status:
            assert f"terminated by {close_key.name}".encode() in transcript
        assert b"\x1b[?1049h" not in transcript
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(master)
