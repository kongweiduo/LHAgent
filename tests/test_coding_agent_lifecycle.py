"""Agent 关闭、初始化取消及共享资源所有权的异步竞态验收。"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from lhagent.agents import coding
from lhagent.harness.session.repository import SessionRepository
from tests.test_coding_agent import Client, config
from tests.test_loop_basic import make_result


@pytest.mark.parametrize(
    "owned_client,owned_session", [(False, False), (True, False), (False, True), (True, True)]
)
def test_close_ownership_and_closed_entrypoints(tmp_path, monkeypatch, owned_client, owned_session):
    """关闭遵守资源归属，并拒绝关闭后的入口操作。"""

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path)})
        session = await repository.create() if not owned_session else None
        client = Client([make_result()])
        monkeypatch.setattr(coding, "Client", lambda _: client)
        monkeypatch.setattr(coding, "load_client_config", lambda: object())
        monkeypatch.setattr(coding, "SessionRepository", lambda _: repository)
        options = {"config": config(tmp_path)}
        if not owned_client:
            options["client"] = client
        if not owned_session:
            options["session"] = session
        agent = coding.CodingAgent(options)
        await agent.run("hello")
        agent.follow_up("keep")
        await asyncio.gather(agent.close(), agent.close())
        await agent.close()
        assert client.close.await_count == int(owned_client)
        assert agent._session.closed == owned_session
        with pytest.raises(RuntimeError, match="closed"):
            await agent.run("again")
        for operation in (
            lambda: agent.steer("x"),
            lambda: agent.follow_up("x"),
            lambda: agent.subscribe(AsyncMock()),
        ):
            with pytest.raises(RuntimeError, match="closed"):
                operation()
        assert len(agent.clear_queue()["follow_up"]) == 1
        await agent.cancel()
        await agent.wait_for_idle()
        await repository.close()

    asyncio.run(scenario())


def test_close_before_initialization(tmp_path, monkeypatch):
    """尚未初始化也可关闭，之后不再初始化。"""

    async def scenario():
        agent = coding.CodingAgent({"config": config(tmp_path)})
        monkeypatch.setattr(coding, "Client", lambda _: pytest.fail("created client"))
        await agent.close()
        await agent.close()
        assert agent._loop is None

    asyncio.run(scenario())


@pytest.mark.parametrize("closing", [False, True])
def test_initialization_cancel_waits_for_cleanup(tmp_path, monkeypatch, closing):
    """初始化取消等待清理结束再传播。"""

    async def scenario():
        entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        repository = SessionRepository({"directory": str(tmp_path)})
        create, close = repository.create, repository.close

        async def blocked_create():
            await create()
            entered.set()
            await asyncio.Event().wait()

        async def blocked_close():
            cleaning.set()
            await release.wait()
            await close()

        monkeypatch.setattr(repository, "create", blocked_create)
        monkeypatch.setattr(repository, "close", blocked_close)
        monkeypatch.setattr(coding, "SessionRepository", lambda _: repository)
        client = Client()
        agent = coding.CodingAgent({"config": config(tmp_path), "client": client})
        agent.follow_up("keep")
        run = asyncio.create_task(agent.run("hello"))
        await entered.wait()
        control = asyncio.create_task(agent.close() if closing else agent.cancel())
        idle = asyncio.create_task(agent.wait_for_idle())
        await cleaning.wait()
        second = asyncio.create_task(agent.cancel())
        await asyncio.sleep(0)
        assert not control.done() and not idle.done()
        release.set()
        await asyncio.gather(control, second, idle)
        with pytest.raises(asyncio.CancelledError):
            await run
        assert all(session.closed for session in repository._sessions.values())
        assert len(agent.clear_queue()["follow_up"]) == 1
        client.close.assert_not_awaited()
        if not closing:
            monkeypatch.setattr(repository, "create", create)
            replacement = SessionRepository({"directory": str(tmp_path / "retry")})
            monkeypatch.setattr(coding, "SessionRepository", lambda _: replacement)
            client.responses.append(make_result())
            assert (await agent.run("retry"))["status"] == "completed"
            await agent.close()

    asyncio.run(scenario())


def test_close_waits_for_events_and_preserves_shared_client(tmp_path):
    """关闭等待事件处理结束，同时保留共享客户端。"""

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path)})
        first_session, second_session = await repository.create(), await repository.create()
        entered, ended, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class SharedClient(Client):
            async def stream(self, request, *, cancel_event):
                text = request["messages"][-1]["content"][0]["text"]
                if text == "blocked":
                    entered.set()
                    await cancel_event.wait()
                async for event in super().stream(request, cancel_event=cancel_event):
                    yield event

        client = SharedClient([make_result(), make_result(), make_result()])
        first = coding.CodingAgent(
            {"config": config(tmp_path), "client": client, "session": first_session}
        )
        second = coding.CodingAgent(
            {"config": config(tmp_path), "client": client, "session": second_session}
        )

        async def listener(event):
            if event["type"] == "run_end":
                ended.set()
                await release.wait()

        first.subscribe(listener)
        run = asyncio.create_task(first.run("blocked"))
        await entered.wait()
        first.follow_up("keep")
        closing = asyncio.create_task(first.close())
        await ended.wait()
        idle = asyncio.create_task(first.wait_for_idle())
        await asyncio.sleep(0)
        assert not closing.done() and not idle.done()
        assert (await second.run("other"))["status"] == "completed"
        release.set()
        await asyncio.gather(closing, idle)
        assert (await run)["status"] == "cancelled"
        assert (await first_session.state())["state"] == "idle"
        assert not first_session.closed
        assert (await second.run("still works"))["status"] == "completed"
        assert len(first.clear_queue()["follow_up"]) == 1
        client.close.assert_not_awaited()
        await second.close()
        await repository.close()

    asyncio.run(scenario())


def test_cancelled_close_finishes_resources_and_reports_failure(tmp_path, monkeypatch):
    """关闭调用者被取消时仍完成资源清理并保留失败信息。"""

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        repository = SessionRepository({"directory": str(tmp_path)})
        client = Client([make_result()])

        async def close_client():
            entered.set()
            await release.wait()
            raise OSError("client cleanup failed")

        client.close.side_effect = close_client
        monkeypatch.setattr(coding, "Client", lambda _: client)
        monkeypatch.setattr(coding, "load_client_config", lambda: object())
        monkeypatch.setattr(coding, "SessionRepository", lambda _: repository)
        agent = coding.CodingAgent({"config": config(tmp_path)})
        await agent.run("hello")
        task = asyncio.create_task(agent.close())
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert isinstance(caught.value.__cause__, OSError)
        assert agent._session.closed
        with pytest.raises(OSError, match="client cleanup failed"):
            await agent.close()
        client.close.assert_awaited_once()

    asyncio.run(scenario())


def test_cancel_after_initialization_before_loop_start(tmp_path, monkeypatch):
    """初始化完成到循环启动之间的取消不会启动新运行。"""

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path)})
        session = await repository.create()
        client = Client()
        agent = coding.CodingAgent(
            {"config": config(tmp_path), "client": client, "session": session}
        )
        initialize = agent._initialize
        controls = []

        async def initialize_then_cancel():
            await initialize()
            controls.append(asyncio.create_task(agent.cancel()))

        monkeypatch.setattr(agent, "_initialize", initialize_then_cancel)
        assert (await agent.run("hello"))["status"] == "cancelled"
        await asyncio.gather(*controls)
        assert not client.requests
        assert (await session.state())["state"] == "new"
        await agent.close()
        await repository.close()

    asyncio.run(scenario())


def test_cancel_caller_interrupted_still_waits_for_run_cleanup(tmp_path):
    """取消调用本身被打断也等待运行清理。"""

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path)})
        session = await repository.create()
        agent = coding.CodingAgent(
            {"config": config(tmp_path), "client": Client(), "session": session}
        )
        entered, release = asyncio.Event(), asyncio.Event()

        async def listener(event):
            if event["type"] == "run_start":
                entered.set()
                await release.wait()

        agent.subscribe(listener)
        run = asyncio.create_task(agent.run("hello"))
        await entered.wait()
        cancel = asyncio.create_task(agent.cancel())
        await asyncio.sleep(0)
        cancel.cancel()
        await asyncio.sleep(0)
        assert not cancel.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await cancel
        assert (await run)["status"] == "cancelled"
        assert (await session.state())["state"] == "idle"
        await agent.close()
        await repository.close()

    asyncio.run(scenario())


def test_initialization_failure_racing_close(tmp_path, monkeypatch):
    """初始化失败与关闭竞争时不重复泄漏资源。"""

    async def scenario():
        cleaning, release = asyncio.Event(), asyncio.Event()
        repository = SessionRepository({"directory": str(tmp_path)})
        original_close = repository.close

        async def close():
            cleaning.set()
            await release.wait()
            await original_close()
            raise OSError("cleanup failed")

        monkeypatch.setattr(repository, "close", close)
        monkeypatch.setattr(coding, "SessionRepository", lambda _: repository)
        monkeypatch.setattr(
            coding, "AgentLoop", lambda *args: (_ for _ in ()).throw(ValueError("assembly failed"))
        )
        agent = coding.CodingAgent({"config": config(tmp_path), "client": Client()})
        run = asyncio.create_task(agent.run("hello"))
        await cleaning.wait()
        closing = asyncio.create_task(agent.close())
        await asyncio.sleep(0)
        release.set()
        for task in (run, closing):
            with pytest.raises(ValueError, match="assembly failed") as caught:
                await task
            assert any("cleanup failed" in note for note in caught.value.__notes__)
        assert agent._loop is None
        assert all(session.closed for session in repository._sessions.values())

    asyncio.run(scenario())
