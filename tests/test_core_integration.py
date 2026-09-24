"""公开入口和跨模块离线场景，使用真实文件工具、临时会话及假模型。"""

import asyncio
import sys

import pytest

from lhagent import cli
from lhagent.agents import coding, create_coding_agent
from lhagent.harness.session import SessionRepository
from tests.test_coding_agent import Client, config
from tests.test_loop_basic import make_result
from tests.test_loop_tools import call, response


def test_public_imports_do_not_load_config(monkeypatch):
    """公开包导入不加载运行配置或凭据。"""
    from lhagent.client import Client as PublicClient
    from lhagent.client import load_config
    from lhagent.harness.context import compact
    from lhagent.harness.loop import AgentLoop
    from lhagent.harness.tools import create_builtin_tools

    assert PublicClient and load_config and compact and AgentLoop and create_builtin_tools


def test_text_tool_and_reopen(tmp_path, monkeypatch):
    """真实文件工具可运行并从持久化会话恢复。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "source.txt").write_text("from disk", encoding="utf-8")

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path / "sessions")})
        session = await repository.create()
        metadata = session.metadata
        client = Client(
            [
                response("tool_call", call("a", name="read", arguments={"path": "source.txt"})),
                response("stop", {"type": "text", "text": "read done"}),
            ]
        )
        agent = create_coding_agent(
            {"config": config(tmp_path), "client": client, "session": session}
        )
        try:
            assert (await agent.run("read it"))["status"] == "completed"
            assert "from disk" in str(client.requests[1]["messages"])
        finally:
            await agent.close()
            await repository.close()
        assert session.closed
        reopened_repo = SessionRepository({"directory": str(tmp_path / "sessions")})
        try:
            assert metadata in await reopened_repo.list()
            reopened = await reopened_repo.open(metadata)
            next_client = Client([response("stop", {"type": "text", "text": "continued"})])
            next_agent = create_coding_agent(
                {"config": config(tmp_path), "client": next_client, "session": reopened}
            )
            try:
                result = await next_agent.run("continue")
                assert result["status"] == "completed"
                assert result["last_response"]["content"][0]["text"] == "continued"
                assert [m["role"] for m in next_client.requests[0]["messages"]] == [
                    "user",
                    "assistant",
                    "tool",
                    "assistant",
                    "user",
                ]
            finally:
                await next_agent.close()
        finally:
            await reopened_repo.close()

    asyncio.run(scenario())


def test_compaction_then_continue(tmp_path, monkeypatch):
    """压缩后的会话仍能继续完成请求。"""
    monkeypatch.chdir(tmp_path)

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path / "sessions")})
        session = await repository.create()
        client = Client([make_result(), make_result()])
        client.complete.return_value = response("stop", {"type": "text", "text": "brief summary"})
        options = config(
            tmp_path,
            context_window=300,
            max_output_tokens=40,
            compaction={"enabled": True, "reserve_tokens": 40, "keep_recent_tokens": 0},
            max_summary_output_tokens=30,
        )
        agent = create_coding_agent({"config": options, "client": client, "session": session})
        try:
            assert (await agent.run("x" * 1100))["status"] == "completed"
            assert (await session.get_history())["summary"] == "brief summary"
            assert (await agent.run("follow on"))["status"] == "completed"
            assert client.complete.await_count >= 1
            assert len(client.requests) == 2
            assert "brief summary" in str(client.requests[1]["messages"])
        finally:
            await agent.close()
            await repository.close()

    asyncio.run(scenario())


def test_cancel_and_error_close_resources(tmp_path, monkeypatch):
    """取消和错误路径都释放自有资源。"""
    monkeypatch.chdir(tmp_path)

    async def scenario():
        entered = asyncio.Event()

        class WaitingClient(Client):
            async def stream(self, request, *, cancel_event):
                entered.set()
                await cancel_event.wait()
                async for event in super().stream(request, cancel_event=cancel_event):
                    yield event

        repository = SessionRepository({"directory": str(tmp_path / "sessions")})
        session = await repository.create()
        client = WaitingClient([response("cancelled")])
        agent = create_coding_agent(
            {"config": config(tmp_path), "client": client, "session": session}
        )
        task = asyncio.create_task(agent.run("cancel me"))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            await agent.cancel()
            assert (await task)["status"] == "cancelled"
            assert (await session.state())["state"] == "idle"
        finally:
            await agent.close()
            await repository.close()
        assert session.closed

        second = SessionRepository({"directory": str(tmp_path / "other")})
        failing = await second.create()

        class BrokenClient(Client):
            async def stream(self, request, *, cancel_event):
                raise RuntimeError("model unavailable")
                yield

        other = create_coding_agent(
            {"config": config(tmp_path), "client": BrokenClient(), "session": failing}
        )
        try:
            with pytest.raises(RuntimeError, match="model unavailable"):
                await other.run("fail")
            assert (await failing.state())["state"] == "idle"
        finally:
            await other.close()
            await second.close()
        assert failing.closed

    asyncio.run(scenario())


def test_cli_exit_codes_and_session_resume(tmp_path, monkeypatch, capsys):
    """单次 CLI 使用既定退出码并支持恢复会话。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "lhagent.toml").write_text(
        '[coding_agent]\nmodel = "test"\ncontext_window = 10000\nmax_output_tokens = 1000\n',
        encoding="utf-8",
    )
    clients = []

    def factory(_):
        client = Client([response("stop", {"type": "text", "text": "answer"})])
        clients.append(client)
        return client

    monkeypatch.setattr(coding, "load_client_config", lambda: object())
    monkeypatch.setattr(coding, "Client", factory)
    assert asyncio.run(cli._run("lhagent.toml", "hello", None)) == 0
    assert capsys.readouterr().out.strip() == "answer"
    paths = list((tmp_path / ".lhagent" / "sessions").glob("*.jsonl"))
    assert len(paths) == 1
    assert asyncio.run(cli._run("lhagent.toml", "again", str(paths[0]))) == 0
    assert clients[0].close.await_count == clients[1].close.await_count == 1
    with pytest.raises(FileNotFoundError, match="session not found"):
        asyncio.run(cli._run("lhagent.toml", "again", str(tmp_path / "missing.jsonl")))
    monkeypatch.setattr(
        sys, "argv", ["lhagent", "--config", "missing.toml", "--instruction", "hello"]
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 1
    assert "config not found" in capsys.readouterr().err

    monkeypatch.setattr(coding, "Client", lambda _: Client([response("error")]))
    assert asyncio.run(cli._run("lhagent.toml", "error", None)) == 1
    assert "run error" in capsys.readouterr().err
