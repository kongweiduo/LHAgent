"""假 agent 与真实编辑器的应用编排，验证单次运行及失败后的关闭。"""

import asyncio

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lhagent.tui.actions import InputAction
from lhagent.tui.app import InteractiveApp
from lhagent.tui.input import InputEditor


class Session:
    """无磁盘会话替身，记录展示读取次数并提供初始状态。"""

    metadata = {"id": "session-123", "path": "/tmp/session.jsonl", "created_at": 1.0}

    def __init__(self):
        self.reads = 0

    async def close(self):
        pass

    async def get_display_history(self):
        self.reads += 1
        return {"summary": None, "entries": []}

    async def state(self):
        return {
            "state": "new",
            "active_run_id": None,
            "last_finished_run_id": None,
            "interrupted": False,
        }


class Repository:
    """返回固定会话的仓库替身，关闭时释放该会话。"""

    def __init__(self, session):
        self.session = session

    async def create(self):
        return self.session

    async def close(self):
        await self.session.close()


class Agent:
    """可控制运行结束的 agent 替身，记录入口调用并交付事件。"""

    def __init__(self):
        self.calls = []
        self.listener = None
        self.finish = asyncio.Event()
        self.running = False

    def subscribe(self, listener):
        self.listener = listener
        self.calls.append("subscribe")

        def unsubscribe():
            self.listener = None
            self.calls.append("unsubscribe")

        return unsubscribe

    async def emit(self, type, **data):
        await self.listener({"type": type, "data": {"run_id": "run-1", **data}})

    async def run(self, text):
        assert not self.running
        self.running = True
        self.calls.append(("run", text))
        await self.emit("run_start")
        await self.emit("message_committed", kind="user", entry_id="u1")
        await self.finish.wait()
        await self.emit("run_end", status="completed", error=None)
        self.running = False
        return {"status": "completed", "last_response": None, "error": None}

    def steer(self, text):
        self.calls.append(("steer", text))

    def follow_up(self, text):
        self.calls.append(("follow_up", text))

    async def cancel(self):
        self.calls.append("cancel")
        self.finish.set()
        while self.running:
            await asyncio.sleep(0)

    async def close(self):
        self.calls.append("close")


async def until(predicate):
    """轮询异步状态至满足条件，超时使调度问题显式失败。"""

    async def wait():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), 2)


def test_run_failure_recovers_and_output_failure_closes():
    """运行失败可恢复，终端输出失败触发资源关闭。"""

    class FailingAgent(Agent):
        async def run(self, text):
            self.calls.append(("run", text))
            raise RuntimeError("model unavailable")

    async def scenario():
        with create_pipe_input() as pipe:
            agent = FailingAgent()
            app = InteractiveApp(
                {"cwd": "/tmp"},
                repository=Repository(Session()),
                agent_factory=lambda _: agent,
                editor_factory=lambda footer: InputEditor(footer, input=pipe, output=DummyOutput()),
            )
            task = asyncio.create_task(app.run())
            try:
                await app.ready.wait()
                app.editor._actions.put_nowait(InputAction("submit", "first"))
                await until(
                    lambda: any(
                        getattr(b, "text", "").startswith("Run error:")
                        for b in app.display.state.transcript
                    )
                )
                app.editor._actions.put_nowait(InputAction("submit", "second"))
                await until(lambda: ("run", "second") in agent.calls)
                await until(lambda: app.display.state.run_status == "idle")

                def broken(_):
                    raise OSError("terminal closed")

                app.scrollback.draw = broken
                app.editor._actions.put_nowait(InputAction("command", command="help", text="help"))
                with pytest.raises(RuntimeError, match="terminal output failed: terminal closed"):
                    await asyncio.wait_for(task, 2)
                assert app._render_failed and agent.calls[-2:] == ["close", "unsubscribe"]
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_single_run_events_commands_and_shutdown():
    """事件、命令及关闭由单运行应用统一协调。"""

    async def scenario():
        with create_pipe_input() as pipe:
            agent = Agent()
            session = Session()
            app = InteractiveApp(
                {"cwd": "/tmp"},
                repository=Repository(session),
                agent_factory=lambda _: agent,
                editor_factory=lambda footer: InputEditor(footer, input=pipe, output=DummyOutput()),
            )
            task = asyncio.create_task(app.run())
            try:
                await app.ready.wait()
                assert any(
                    session.metadata["path"] in getattr(block, "text", "")
                    for block in app.display.state.transcript
                )
                await app._handle(InputAction("command", command="history"))
                app.editor._actions.put_nowait(InputAction("submit", "hello"))
                await until(lambda: agent.running)
                buffer = app.editor.session.default_buffer
                buffer.text = "draft"
                await agent.emit(
                    "response_update",
                    phase="delta",
                    call_id="c1",
                    block_index=0,
                    delta={"type": "text", "text": "stream"},
                )
                await agent.emit(
                    "tool_start", tool_call_id="t1", tool_name="read", arguments={"path": "x"}
                )
                await agent.emit(
                    "tool_end",
                    tool_call_id="t1",
                    tool_name="read",
                    status="execution_error",
                    result={
                        "call_id": "t1",
                        "name": "read",
                        "status": "execution_error",
                        "output": None,
                        "error": "missing",
                    },
                )
                await agent.emit("compaction_start", tokens=100)
                await agent.emit("compaction_end", status="error", error="budget")
                assert buffer.text == "draft" and app.display.state.recent_error == "budget"
                app.editor._actions.put_nowait(InputAction("steer", "next"))
                app.editor._actions.put_nowait(InputAction("follow_up", "later"))
                await until(lambda: ("follow_up", "later") in agent.calls)
                assert ("steer", "next") in agent.calls
                assert len([x for x in agent.calls if isinstance(x, tuple) and x[0] == "run"]) == 1
                app.editor._actions.put_nowait(InputAction("command", command="session"))
                await until(
                    lambda: any(
                        getattr(b, "text", "").startswith("Session:")
                        for b in app.display.state.transcript
                    )
                )
                app.editor._actions.put_nowait(InputAction("command", command="clear"))
                await until(lambda: app.display.state.transcript == [])
                assert session.reads == 2 and agent.running
                app.editor._actions.put_nowait(InputAction("cancel"))
                await until(lambda: "cancel" in agent.calls)
                app.editor._actions.put_nowait(InputAction("close"))
                await asyncio.wait_for(task, 2)
                assert agent.calls[-2:] == ["close", "unsubscribe"]
                assert agent.listener is None
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
