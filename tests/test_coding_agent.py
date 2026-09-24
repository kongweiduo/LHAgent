"""Agent 惰性组装、事件订阅、输入入口、摘要适配与初始化失败清理。"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from lhagent.agents import coding
from lhagent.harness.context.budget import estimate_tool_tokens
from lhagent.harness.session.repository import SessionRepository
from lhagent.harness.tools.registry import describe_tools
from lhagent.harness.tools.types import ToolStopError
from tests.test_loop_basic import make_result
from tests.test_loop_tools import Client as ScriptedClient
from tests.test_loop_tools import call, response


def config(tmp_path, **overrides):
    """构造临时工作目录配置，允许场景覆盖字段。"""
    return {
        "model": "test",
        "cwd": str(tmp_path),
        "context_window": 10000,
        "max_output_tokens": 1000,
        "tools": ["read", "read"],
        **overrides,
    }


class Client(ScriptedClient):
    """脚本响应客户端，使用 AsyncMock 观察关闭与摘要请求。"""

    def __init__(self, responses=()):
        super().__init__(list(responses))
        self.close = AsyncMock()
        self.complete = AsyncMock(return_value=make_result())


def test_event_subscriptions_are_ordered_and_use_dispatch_snapshots(tmp_path, monkeypatch):
    """订阅按顺序分发，分发中的成员变更从下一事件生效。"""
    monkeypatch.chdir(tmp_path)

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path / "sessions")})
        session = await repository.create()
        agent = coding.CodingAgent(
            {
                "config": config(tmp_path),
                "client": Client([make_result()]),
                "session": session,
            }
        )
        delivered = []
        unsubscribe_second = None

        async def third(event):
            delivered.append(("third", event["type"]))

        async def first(event):
            nonlocal unsubscribe_second
            delivered.append(("first", event["type"]))
            if event["type"] == "run_start":
                unsubscribe_second()
                unsubscribe_second()
                agent.subscribe(third)

        async def second(event):
            delivered.append(("second", event["type"]))

        agent.subscribe(first)
        unsubscribe_second = agent.subscribe(second)
        try:
            assert (await agent.run("hello"))["status"] == "completed"
            assert delivered[:2] == [("first", "run_start"), ("second", "run_start")]
            assert ("second", "message_committed") not in delivered
            assert ("third", "run_start") not in delivered
            assert delivered.index(("first", "message_committed")) < delivered.index(
                ("third", "message_committed")
            )
        finally:
            await repository.close()

    asyncio.run(scenario())


def test_listener_failure_reaches_run_after_loop_cleanup(tmp_path, monkeypatch):
    """监听器错误在循环清理后传播给运行调用者。"""
    monkeypatch.chdir(tmp_path)

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path / "sessions")})
        session = await repository.create()
        agent = coding.CodingAgent(
            {
                "config": config(tmp_path),
                "client": Client([make_result()]),
                "session": session,
            }
        )
        events = []

        failed = False

        async def listener(event):
            nonlocal failed
            events.append(event["type"])
            if event["type"] == "message_committed" and not failed:
                failed = True
                raise RuntimeError("listener failed")

        agent.subscribe(listener)
        try:
            with pytest.raises(RuntimeError, match="listener failed"):
                await agent.run("hello")
            assert events[-1] == "run_end"
            records = [
                json.loads(line) for line in Path(session.metadata["path"]).read_text().splitlines()
            ]
            assert [record["type"] for record in records].count("run_finish") == 1
            assert not agent._running
        finally:
            await repository.close()

    asyncio.run(scenario())


def test_input_entrypoints_share_queue_before_and_after_initialization(tmp_path, monkeypatch):
    """初始化前后输入入口共用同一队列。"""
    monkeypatch.chdir(tmp_path)
    agent = coding.CodingAgent({"config": config(tmp_path), "client": Client([make_result()])})
    with pytest.raises(TypeError, match="instruction"):
        agent.steer(None)
    with pytest.raises(TypeError, match="instruction"):
        agent.follow_up(1)
    agent.steer("before steer")
    agent.follow_up("before follow")
    assert agent.clear_queue() == {
        "steer": [{"role": "user", "content": [{"type": "text", "text": "before steer"}]}],
        "follow_up": [{"role": "user", "content": [{"type": "text", "text": "before follow"}]}],
    }
    assert agent.clear_queue() == {"steer": [], "follow_up": []}

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path / "sessions")})
        session = await repository.create()
        client = Client([make_result(), make_result()])
        initialized = coding.CodingAgent(
            {
                "config": config(tmp_path),
                "client": client,
                "session": session,
            }
        )
        initialized.steer("queued steer")
        initialized.follow_up("queued follow")
        try:
            assert (await initialized.run("start"))["status"] == "completed"
            history = await session.get_history()
            texts = [
                message["content"][0]["text"]
                for message in history["messages"]
                if message["role"] == "user"
            ]
            assert texts == ["start", "queued steer", "queued follow"]
            assert len(client.requests) == 2
            assert initialized.clear_queue() == {"steer": [], "follow_up": []}
        finally:
            await repository.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("bad_listener", [None, 1, "listener"])
def test_subscribe_rejects_non_callable(tmp_path, bad_listener):
    """非可调用订阅者在登记时拒绝。"""
    agent = coding.CodingAgent({"config": config(tmp_path), "client": Client()})
    with pytest.raises(TypeError, match="listener"):
        agent.subscribe(bad_listener)


def test_construction_is_synchronous_and_lazy(tmp_path, monkeypatch):
    """构造同步且惰性，不立即打开运行资源。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(coding, "Client", lambda *_: pytest.fail("client created"))
    monkeypatch.setattr(coding, "SessionRepository", lambda *_: pytest.fail("repository created"))
    agent = coding.create_coding_agent({"config": config(tmp_path)})
    assert agent._loop is None
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="model"):
        coding.create_coding_agent({"config": {}})
    with pytest.raises(KeyError, match="Unknown tool"):
        coding.create_coding_agent({"config": config(tmp_path, tools=["unknown"])})


def test_injected_tool_roundtrip_and_summary_adapter(tmp_path, monkeypatch):
    """注入工具可完成往返，摘要适配保留独立参数边界。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "source.txt").write_text("hello from file", encoding="utf-8")
    (tmp_path / "system.txt").write_text("system prompt", encoding="utf-8")
    (tmp_path / "extra.txt").write_text("extra prompt", encoding="utf-8")

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path / "sessions")})
        session = await repository.create()
        client = Client(
            [
                response(
                    "tool_call", call("tool-1", name="read", arguments={"path": "source.txt"})
                ),
                make_result(),
                make_result(),
            ]
        )
        options = config(
            tmp_path,
            system_prompt_path="system.txt",
            additional_prompt_paths=["extra.txt"],
            generation_parameters={"temperature": 0.2},
        )
        agent = coding.create_coding_agent(
            {"config": options, "client": client, "session": session}
        )
        options["generation_parameters"]["temperature"] = 0.9
        try:
            result = await agent.run("read source")
            assert result["status"] == "completed"
            assert len(client.requests) == 2
            request = client.requests[0]
            assert request["model"] == "test"
            assert request["parameters"]["max_output_tokens"] == 1000
            assert request["parameters"]["temperature"] == 0.2
            definitions = describe_tools(agent._tools)
            assert [tool["name"] for tool in definitions] == ["read"]
            assert request["parameters"]["tools"] == definitions
            assert agent._loop._config["budget"]["extra_input_tokens"] == estimate_tool_tokens(
                definitions
            )
            assert [m["content"][0]["text"] for m in request["messages"][:2]] == [
                "system prompt",
                "extra prompt",
            ]
            history = await session.get_history()
            assert [m["role"] for m in history["messages"]] == [
                "user",
                "assistant",
                "tool",
                "assistant",
            ]
            assert "hello from file" in json.dumps(history["messages"][2])
            assert agent._loop._tool_context["cwd"] == str(tmp_path)

            signal = asyncio.Event()
            signal.set()
            messages = [{"role": "user", "content": [{"type": "text", "text": "summarize"}]}]
            for limit in (42, 43):
                assert (
                    await agent._loop._config["summary_request"](messages, limit, signal)
                    == make_result()
                )
                args, kwargs = client.complete.call_args
                assert kwargs["cancel_event"] is signal
                assert args[0]["messages"] is messages
                assert args[0]["parameters"] == {"temperature": 0.2, "max_output_tokens": limit}
            loop = agent._loop
            assert (await agent.run("again"))["status"] == "completed"
            assert agent._loop is loop
            records = [
                json.loads(line) for line in Path(session.metadata["path"]).read_text().splitlines()
            ]
            starts = [r for r in records if r["type"] == "run_start"]
            ends = [r for r in records if r["type"] == "run_finish"]
            assert len(starts) == len(ends) == 2
            run_ids = {r["run_id"] for r in starts}
            call_ids = {r["call_id"] for r in client.requests}
            call_ids.update(args[0]["call_id"] for args, _ in client.complete.call_args_list)
            assert len(run_ids) == 2 and len(call_ids) == 5
            assert run_ids.isdisjoint(call_ids)
            client.close.assert_not_awaited()
            assert not session.closed
        finally:
            await repository.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "external_client,external_session", [(False, False), (True, False), (False, True), (True, True)]
)
def test_initialization_failure_closes_only_owned_resources(
    tmp_path, monkeypatch, external_client, external_session
):
    """初始化失败只关闭自有资源，不关闭借用对象。"""
    monkeypatch.chdir(tmp_path)

    async def scenario():
        client = Client([make_result()])
        repository = SessionRepository({"directory": str(tmp_path / "sessions")})
        session = await repository.create() if external_session else None
        original_loop = coding.AgentLoop
        monkeypatch.setattr(coding, "Client", lambda _: client)
        monkeypatch.setattr(coding, "load_client_config", lambda: object())
        monkeypatch.setattr(coding, "SessionRepository", lambda _: repository)
        monkeypatch.setattr(
            coding, "AgentLoop", lambda *args: (_ for _ in ()).throw(ValueError("assembly failed"))
        )
        options = {"config": config(tmp_path)}
        if external_client:
            options["client"] = client
        if external_session:
            options["session"] = session
        agent = coding.CodingAgent(options)
        try:
            with pytest.raises(ValueError, match="assembly failed"):
                await agent.run("first")
            assert agent._loop is None
            assert not agent._running
            assert client.close.await_count == (0 if external_client else 1)
            if external_session:
                assert not session.closed
            else:
                assert all(s.closed for s in repository._sessions.values())
            # A failed initialization can be retried with newly created owned resources.
            replacement = SessionRepository({"directory": str(tmp_path / "retry")})
            replacement_client = Client([make_result()])
            monkeypatch.setattr(coding, "Client", lambda _: replacement_client)
            monkeypatch.setattr(coding, "SessionRepository", lambda _: replacement)
            monkeypatch.setattr(coding, "AgentLoop", original_loop)
            try:
                assert (await agent.run("retry"))["status"] == "completed"
            finally:
                await replacement.close()
        finally:
            await repository.close()

    asyncio.run(scenario())


def test_concurrency_during_initialization_and_run(tmp_path, monkeypatch):
    """初始化和运行期间均拒绝不允许的并发运行。"""
    monkeypatch.chdir(tmp_path)

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        streaming = asyncio.Event()
        finish = asyncio.Event()
        repository = SessionRepository({"directory": str(tmp_path / "sessions")})
        create = repository.create

        async def delayed_create():
            entered.set()
            await release.wait()
            return await create()

        class BlockingClient(Client):
            async def stream(self, request, *, cancel_event):
                streaming.set()
                await finish.wait()
                async for event in super().stream(request, cancel_event=cancel_event):
                    yield event

        monkeypatch.setattr(repository, "create", delayed_create)
        monkeypatch.setattr(coding, "SessionRepository", lambda _: repository)
        client = BlockingClient([make_result()])
        agent = coding.CodingAgent({"config": config(tmp_path), "client": client})
        task = asyncio.create_task(agent.run("first"))
        try:
            await entered.wait()
            with pytest.raises(RuntimeError, match="already running"):
                await agent.run("second")
            release.set()
            await streaming.wait()
            with pytest.raises(RuntimeError, match="already running"):
                await agent.run("third")
            finish.set()
            assert (await task)["status"] == "completed"
            assert len(client.requests) == 1
        finally:
            release.set()
            finish.set()
            await task
            await repository.close()

    asyncio.run(scenario())


def test_cancelled_initialization_waits_for_cleanup(tmp_path, monkeypatch):
    """初始化被取消后先完成资源清理。"""
    monkeypatch.chdir(tmp_path)

    async def scenario():
        entered = asyncio.Event()
        closing = asyncio.Event()
        release = asyncio.Event()
        repository = SessionRepository({"directory": str(tmp_path / "sessions")})
        create = repository.create
        close = repository.close

        async def delayed_create():
            await create()
            entered.set()
            await asyncio.Event().wait()

        async def delayed_close():
            closing.set()
            await release.wait()
            await close()

        client = Client()
        monkeypatch.setattr(coding, "load_client_config", lambda: object())
        monkeypatch.setattr(coding, "Client", lambda _: client)
        monkeypatch.setattr(coding, "SessionRepository", lambda _: repository)
        monkeypatch.setattr(repository, "create", delayed_create)
        monkeypatch.setattr(repository, "close", delayed_close)
        agent = coding.CodingAgent({"config": config(tmp_path)})
        task = asyncio.create_task(agent.run("first"))
        await entered.wait()
        task.cancel()
        await closing.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert all(session.closed for session in repository._sessions.values())
        client.close.assert_awaited_once()
        assert agent._loop is None and not agent._running

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["prompt", "client", "session"])
def test_early_initialization_failure(tmp_path, monkeypatch, stage):
    """早期初始化失败不会留下半成品运行状态。"""
    monkeypatch.chdir(tmp_path)

    async def scenario():
        client = Client()
        repository = SessionRepository({"directory": str(tmp_path / "sessions")})
        create = AsyncMock(side_effect=OSError("session creation failed"))
        monkeypatch.setattr(repository, "create", create)
        monkeypatch.setattr(coding, "SessionRepository", lambda _: repository)
        factory = Mock(return_value=client)
        if stage == "client":
            factory.side_effect = ValueError("client creation failed")
        monkeypatch.setattr(coding, "load_client_config", lambda: object())
        monkeypatch.setattr(coding, "Client", factory)
        options = config(tmp_path)
        if stage == "prompt":
            options["system_prompt_path"] = "missing.txt"
        agent = coding.CodingAgent({"config": options})
        with pytest.raises((OSError, ValueError)):
            await agent.run("first")
        assert factory.called == (stage != "prompt")
        assert create.await_count == (1 if stage == "session" else 0)
        assert client.close.await_count == (1 if stage == "session" else 0)
        assert not agent._running and agent._loop is None
        assert not (tmp_path / "sessions").exists()
        await repository.close()

    asyncio.run(scenario())


def test_unconfirmed_tool_stop_does_not_rebuild_loop(tmp_path, monkeypatch):
    """工具停止未确认后不能靠重建循环绕过运行限制。"""
    monkeypatch.chdir(tmp_path)

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path / "sessions")})
        session = await repository.create()
        client = Client(
            [response("tool_call", call("tool-1", name="read", arguments={"path": "x"}))]
        )
        agent = coding.CodingAgent(
            {"config": config(tmp_path), "client": client, "session": session}
        )

        async def handler(args, context):
            raise ToolStopError(
                {"call_id": "tool-1", "name": "read", "arguments": args}, "unconfirmed"
            )

        agent._tools[0]["handler"] = handler
        try:
            with pytest.raises(ToolStopError):
                await agent.run("first")
            loop = agent._loop
            with pytest.raises(RuntimeError, match="unconfirmed"):
                await agent.run("second")
            assert agent._loop is loop
            assert len(client.requests) == 1
            client.close.assert_not_awaited()
            assert not session.closed
        finally:
            await repository.close()

    asyncio.run(scenario())
