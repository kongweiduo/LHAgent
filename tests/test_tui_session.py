"""会话绑定、切换和选择器，验证失败恢复及资源释放。"""

import asyncio
from pathlib import Path

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lhagent.agents import create_coding_agent
from lhagent.harness.session import SessionRepository
from lhagent.tui.actions import InputAction
from lhagent.tui.app import InteractiveApp
from lhagent.tui.input import InputEditor
from lhagent.tui.session import SessionCoordinator
from tests.samples import user_message
from tests.test_coding_agent import Client, config
from tests.test_loop_basic import make_result
from tests.test_tui_app import Agent, until


async def listener(event):
    """会话绑定使用的空异步订阅者，不产生额外副作用。"""
    pass


def test_default_new_resume_and_continue(tmp_path):
    """默认新建及显式恢复后均能继续运行。"""

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path)})
        agents = []

        def factory(options):
            agent = create_coding_agent({**options, "client": Client([make_result()])})
            agents.append(agent)
            return agent

        coordinator = SessionCoordinator(
            config(tmp_path), repository=repository, agent_factory=factory
        )
        try:
            assert (await coordinator.start(None, listener)).transcript == []
            original = coordinator.session
            metadata = original.metadata
            assert (await coordinator.agent.run("first"))["status"] == "completed"
            await coordinator.switch(None, listener)
            second = coordinator.session
            assert original.closed
            with pytest.raises(RuntimeError):
                await agents[0].run("closed")
            assert await second.get_display_history() == {"summary": None, "entries": []}
            state = await coordinator.switch(metadata, listener)
            assert second.closed
            assert [b.role for b in state.transcript] == ["user", "assistant"]
            restored = coordinator.session
            await coordinator.switch(metadata, listener)
            assert coordinator.session is restored and len(agents) == 3
            assert (await coordinator.agent.run("continue"))["status"] == "completed"
            history = await restored.get_display_history()
            assert [e["type"] for e in history["entries"]] == [
                "user",
                "assistant",
                "user",
                "assistant",
            ]
        finally:
            await coordinator.close()
        assert restored.closed
        with pytest.raises(RuntimeError):
            await repository.list()
        # Startup with an explicit path uses repository validation, too.
        other = SessionCoordinator(
            config(tmp_path),
            repository=SessionRepository({"directory": str(tmp_path)}),
            agent_factory=factory,
        )
        try:
            state = await other.start(metadata["path"], listener)
            assert len(state.transcript) == 4
        finally:
            await other.close()

    asyncio.run(scenario())


def test_corruption_and_binding_failure_release_handles(tmp_path):
    """损坏会话或绑定失败时释放已取得资源。"""

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path)})
        broken = await repository.create({"id": "broken"})
        metadata = broken.metadata
        await broken.close()
        with Path(metadata["path"]).open("a") as file:
            file.write('{"bad":true}\n')
        fail = False

        def factory(options):
            if fail:
                raise RuntimeError("agent setup failed")
            return Agent()

        coordinator = SessionCoordinator(
            {"cwd": str(tmp_path)}, repository=repository, agent_factory=factory
        )
        try:
            await coordinator.start(None, listener)
            old = coordinator.session
            with pytest.raises(ValueError):
                await coordinator.switch(metadata, listener)
            assert old.closed and coordinator.session is None and coordinator.agent is None
            fail = True
            with pytest.raises(RuntimeError, match="agent setup failed"):
                await coordinator.switch(None, listener)
            assert coordinator.session is None and coordinator.agent is None
            fail = False
            # The failed binding's handle was released and can be reopened.
            saved = [m for m in await repository.list() if m["id"] != "broken"]
            await coordinator.switch(saved[0], listener)
            assert coordinator.session.metadata == saved[0]
        finally:
            await coordinator.close()

    asyncio.run(scenario())


def test_interrupted_summary_and_unknown_tools(tmp_path):
    """恢复展示保留摘要、中断标记和未知工具。"""
    from tests.test_loop_tools import call, response

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path)})
        session = await repository.create()
        metadata = session.metadata
        await session.start_run("interrupted")
        await session.append_user(user_message("before"))
        await session.commit_compaction(
            {
                "status": "success",
                "error": None,
                "history": {"summary": "summary", "messages": [], "entry_ids": []},
                "tokens_before": 100,
                "estimated_tokens_after": 10,
                "usage": [],
            }
        )
        await session.append_response(response("tool_call", call("unfinished", arguments={"x": 1})))
        await session.close()
        coordinator = SessionCoordinator(
            config(tmp_path),
            repository=repository,
            agent_factory=lambda options: create_coding_agent(
                {**options, "client": Client([make_result()])}
            ),
        )
        try:
            state = await coordinator.start(metadata["path"], listener)
            assert "summary" in state.transcript[0].text
            assert "interrupted" in state.transcript[-1].text
            assert any(getattr(block, "status", None) == "unknown" for block in state.transcript)
            assert (await coordinator.agent.run("continue"))["status"] == "completed"
        finally:
            await coordinator.close()

    asyncio.run(scenario())


def test_app_switch_busy_failure_recovery_and_footer(tmp_path):
    """应用限制忙碌切换，失败后可恢复且页脚同步。"""

    async def scenario():
        with create_pipe_input() as pipe:
            agents = []

            def factory(options):
                agent = Agent()
                agents.append(agent)
                return agent

            repository = SessionRepository({"directory": str(tmp_path)})
            app = InteractiveApp(
                {"cwd": str(tmp_path)},
                repository=repository,
                agent_factory=factory,
                editor_factory=lambda footer: InputEditor(footer, input=pipe, output=DummyOutput()),
            )
            task = asyncio.create_task(app.run())
            try:
                await asyncio.wait_for(app.ready.wait(), 2)
                original = app.session
                metadata = original.metadata
                await app._handle(InputAction("submit", "busy"))
                await until(lambda: agents[0].running)
                await app._handle(InputAction("command", command="new"))
                await app._handle(InputAction("command", command="resume"))
                assert app.session is original and len(agents) == 1
                assert "cancel first" in app.display.state.transcript[-1].text
                await app._handle(InputAction("cancel"))
                await until(lambda: app._run_task is None and app._cancel_task is None)
                await app._handle(InputAction("command", command="new"))
                second = app.session
                assert original.closed and agents[0].listener is None
                assert "close" in agents[0].calls
                assert app.editor._footer().session_id == second.metadata["id"][:8]

                async def select(rows, current):
                    assert rows == sorted(
                        rows, key=lambda m: (-m["created_at"], m["id"], m["path"])
                    )
                    return metadata

                app.editor.select_session = select
                await app._handle(InputAction("command", command="resume"))
                assert second.closed and app.session.metadata == metadata
                current = app.session
                await app._handle(InputAction("command", command="resume"))
                assert app.session is current and len(agents) == 3
                broken = await repository.create()
                bad_metadata = broken.metadata
                await broken.close()
                with Path(bad_metadata["path"]).open("a") as file:
                    file.write("bad json\n")

                async def select_bad(rows, current):
                    return bad_metadata

                app.editor.select_session = select_bad
                await app._handle(InputAction("command", command="resume"))
                assert app.session is None and current.closed
                assert "No active session" in app.display.state.transcript[-1].text
                await app._handle(InputAction("submit", "do not run"))
                assert app._run_task is None
                await app._handle(InputAction("command", command="new"))
                final = app.session
                assert final is not None
                app.editor._actions.put_nowait(InputAction("close"))
                await asyncio.wait_for(task, 2)
                assert final.closed
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_app_selector_restores_editor_and_preserves_binding_on_list_error(tmp_path):
    """选择器结束后恢复编辑器，列举失败保持现有绑定。"""

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path)})
        saved = await repository.create()
        metadata = saved.metadata
        await saved.close()
        with create_pipe_input() as pipe:
            app = InteractiveApp(
                {"cwd": "/tmp"},
                repository=repository,
                agent_factory=lambda options: Agent(),
                editor_factory=lambda footer: InputEditor(footer, input=pipe, output=DummyOutput()),
            )
            task = asyncio.create_task(app.run())
            try:
                await asyncio.wait_for(app.ready.wait(), 2)
                original = app.session
                pipe.send_text("/resume\r")
                await asyncio.sleep(0.1)
                pipe.send_text("2\n")
                await until(lambda: app.session is not None and app.session.metadata == metadata)
                assert original.closed
                # The original prompt consumes input again after the selector exits.
                pipe.send_text("/session\r")
                await until(
                    lambda: any(
                        "Created:" in getattr(b, "text", "") for b in app.display.state.transcript
                    )
                )
                current = app.session

                async def broken_list(options=None):
                    raise OSError("cannot list")

                repository.list = broken_list
                pipe.send_text("/resume\r")
                await until(
                    lambda: any(
                        "cannot list" in getattr(b, "text", "")
                        for b in app.display.state.transcript
                    )
                )
                assert app.session is current and not current.closed

                async def empty_list(options=None):
                    return []

                repository.list = empty_list
                pipe.send_text("/resume\r")
                await until(
                    lambda: any(
                        "No saved sessions" in getattr(b, "text", "")
                        for b in app.display.state.transcript
                    )
                )
                assert app.session is current
                pipe.send_text("/quit\r")
                await asyncio.wait_for(task, 2)
                assert current.closed
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_startup_failure_closes_repository_and_owned_sessions(tmp_path):
    """启动失败关闭仓库及其自有会话。"""

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path)})
        owned = await repository.create()
        with create_pipe_input() as pipe:
            app = InteractiveApp(
                {"cwd": "/tmp"},
                repository=repository,
                session_path=str(tmp_path / "missing.jsonl"),
                editor_factory=lambda footer: InputEditor(footer, input=pipe, output=DummyOutput()),
            )
            with pytest.raises(FileNotFoundError):
                await app.run()
        assert owned.closed
        with pytest.raises(RuntimeError):
            await repository.list()

    asyncio.run(scenario())


def test_selector_empty_multiple_and_cancel(tmp_path):
    """选择器支持空列表、多选项及取消。"""

    async def scenario():
        with create_pipe_input() as pipe:
            editor = InputEditor(lambda: None, input=pipe, output=DummyOutput())
            assert await editor.select_session([], None) is None
            rows = [
                {"id": "same-prefix-1", "created_at": 2.0, "path": str(tmp_path / "a.jsonl")},
                {"id": "same-prefix-2", "created_at": 1.0, "path": str(tmp_path / "b.jsonl")},
            ]
            task = asyncio.create_task(editor.select_session(rows, rows[0]))
            await asyncio.sleep(0.05)
            pipe.send_text("2\n")
            assert await asyncio.wait_for(task, 2) is rows[1]
            task = asyncio.create_task(editor.select_session(rows, rows[0]))
            await asyncio.sleep(0.05)
            pipe.send_text("\n")
            assert await asyncio.wait_for(task, 2) is None
            task = asyncio.create_task(editor.select_session(rows, rows[0]))
            await asyncio.sleep(0.05)
            pipe.send_text("\x03")
            assert await asyncio.wait_for(task, 2) is None
            assert (await editor.next_action()).kind == "close"

    asyncio.run(scenario())
