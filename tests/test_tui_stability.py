"""离线终端稳定性回归，覆盖高频增量、输入草稿、有界尾部及任务回收。"""

import asyncio

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lhagent.diagnostics import diagnostic
from lhagent.tui.app import InteractiveApp, Scrollback
from lhagent.tui.events import DisplayReducer
from lhagent.tui.input import InputEditor
from lhagent.tui.models import NoticeBlock
from lhagent.tui.rendering import format_block
from tests.test_tui_app import Agent, Repository, Session, until


class RecordingOutput(DummyOutput):
    """记录文本、上移和清屏操作的虚拟终端，用于验证重绘范围。"""

    def __init__(self, columns=20):
        self.size = Size(rows=12, columns=columns)
        self.text = []
        self.moves = []
        self.clears = 0

    def get_size(self):
        return self.size

    def write(self, text):
        self.text.append(text)

    def cursor_up(self, amount):
        self.moves.append(amount)

    def erase_screen(self):
        self.clears += 1


@pytest.mark.parametrize("columns", [1, 2, 10, 80])
def test_long_live_tail_stays_on_screen_and_completion_is_delivered_once(columns):
    """长活动尾部限制在屏内，完整结果结束时仅交付一次。"""
    output = RecordingOutput(columns)
    scrollback = Scrollback(output)
    display = DisplayReducer()
    text = "中文\te\u0301🙂 long response\n" * 500
    event = {
        "type": "response_update",
        "data": {
            "run_id": "r",
            "call_id": "c",
            "phase": "delta",
            "block_index": 0,
            "delta": {"type": "text", "text": text},
        },
    }
    display.apply(event)
    for _ in range(10):
        scrollback.draw(display)
        assert 0 < scrollback._live_rows <= 4
    assert max(output.moves) <= 4
    output.size = Size(rows=6, columns=7)
    scrollback.draw(display)
    assert output.clears == 1 and scrollback._live_rows <= 2
    display.apply(
        {
            "type": "run_end",
            "data": {
                "run_id": "r",
                "status": "completed",
                "error": None,
            },
        }
    )
    output.text.clear()
    scrollback.draw(display)
    first = "".join(output.text)
    assert first.count("long response") == 500
    output.text.clear()
    scrollback.draw(display)
    assert "".join(output.text) == ""


def test_delta_pressure_preserves_pasted_draft_and_reclaims_tasks():
    """高频增量不破坏粘贴草稿，退出回收后台任务。"""

    async def scenario():
        baseline = set(asyncio.all_tasks())
        with create_pipe_input() as pipe:
            agent = Agent()
            app = InteractiveApp(
                {"cwd": "/tmp"},
                repository=Repository(Session()),
                agent_factory=lambda _: agent,
                editor_factory=lambda footer: InputEditor(footer, input=pipe, output=DummyOutput()),
            )
            task = asyncio.create_task(app.run())
            await asyncio.wait_for(app.ready.wait(), 2)
            pipe.send_text("start\r")
            await until(lambda: agent.running)
            draft = "中文草稿\ne\u0301🙂 /quit"
            pipe.send_text("\x1b[200~" + draft + "\x1b[201~")
            buffer = app.editor.session.default_buffer
            await until(lambda: buffer.text == draft)
            cursor = buffer.cursor_position
            draw = app.scrollback.draw
            calls = []

            def record(display):
                calls.append(1)
                draw(display)

            app.scrollback.draw = record
            for _ in range(3000):
                await agent.emit(
                    "response_update",
                    phase="delta",
                    call_id="c",
                    block_index=0,
                    delta={"type": "text", "text": "中"},
                )
            assert len(calls) == 0  # No terminal IO in the high-frequency callback.
            await until(lambda: bool(calls))
            assert len(calls) <= 2
            assert buffer.text == draft and buffer.cursor_position == cursor
            pipe.send_text("\x03")
            await asyncio.wait_for(task, 2)
            assert agent.calls.count("close") == 1 and agent.listener is None
            assert app._render_timer is None
        await asyncio.sleep(0)
        assert set(asyncio.all_tasks()) == baseline

    asyncio.run(scenario())


def test_input_reader_failure_propagates_and_closes():
    """输入读取任务失败会传播并触发关闭。"""

    class BrokenEditor(InputEditor):
        async def next_action(self):
            raise OSError("input terminal disappeared")

    async def scenario():
        with create_pipe_input() as pipe:
            agent = Agent()
            app = InteractiveApp(
                {"cwd": "/tmp"},
                repository=Repository(Session()),
                agent_factory=lambda _: agent,
                editor_factory=lambda footer: BrokenEditor(
                    footer, input=pipe, output=DummyOutput()
                ),
            )
            with pytest.raises(OSError, match="input terminal disappeared"):
                await asyncio.wait_for(app.run(), 2)
            assert agent.calls[-2:] == ["close", "unsubscribe"]

    asyncio.run(scenario())


def test_consecutive_tools_keep_completed_output_once_and_bound_preview():
    """连续工具的已完成输出只交付一次，活动预览保持有界。"""
    output = RecordingOutput(20)
    scrollback = Scrollback(output)
    display = DisplayReducer()
    for index in range(100):
        identity = {"run_id": "r", "tool_call_id": str(index), "tool_name": "read"}
        display.apply(
            {
                "type": "tool_start",
                "data": {
                    **identity,
                    "arguments": {"path": f"file-{index}"},
                },
            }
        )
        scrollback.draw(display)
        assert scrollback._live_rows <= 4
        display.apply(
            {
                "type": "tool_end",
                "data": {
                    **identity,
                    "status": "success",
                    "result": {
                        "call_id": str(index),
                        "name": "read",
                        "status": "success",
                        "error": None,
                        "output": {
                            "content": [{"type": "text", "text": "中文" * 10000}],
                            "details": {},
                            "truncated": False,
                        },
                    },
                },
            }
        )
        output.text.clear()
        scrollback.draw(display)
        assert "".join(output.text).count("[success]") == 1
        assert len("".join(output.text)) < 500
        output.text.clear()
        scrollback.draw(display)
        assert not output.text
    assert len(display.state.transcript) == 100


def test_diagnostics_hide_credentials_but_local_tool_parameters_remain(monkeypatch):
    """诊断遮盖凭据，同时保留工具本地业务参数。"""
    monkeypatch.setenv("LHAGENT_API_KEY", "private-test-token")
    text = (
        "failed private-test-token config={'api_key': 'other-secret'} Bearer third-secret; "
        "Authorization: Basic fourth-secret"
    )
    error = fragment_list_to_text(format_block(NoticeBlock(text, "error")))
    assert "private-test-token" not in error
    assert "other-secret" not in error and "third-secret" not in error
    assert "fourth-secret" not in error
    assert "failed" in diagnostic(text)
    from lhagent.tui.models import ToolBlock

    tool = ToolBlock("r", "t", "bash", {"command": "private-test-token"})
    assert "private-test-token" in fragment_list_to_text(format_block(tool))
