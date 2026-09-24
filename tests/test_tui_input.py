"""使用真实 prompt_toolkit 解析验证按键、命令、历史与刷新，不连接模型。"""

import asyncio
from contextlib import asynccontextmanager

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lhagent.tui.actions import InputAction
from lhagent.tui.commands import COMMANDS, help_text, parse_command, submission
from lhagent.tui.completion import CommandCompleter
from lhagent.tui.input import InputEditor
from lhagent.tui.models import Footer


def completions(text):
    """使用真实 Document 解析补全建议。"""
    return list(CommandCompleter().get_completions(Document(text), CompleteEvent()))


def test_registry_help_completion_and_handlers():
    """注册表、帮助、补全与处理器保持一致。"""
    assert [c.text for c in completions("/")] == [c.usage for c in COMMANDS]
    assert [c.text for c in completions("/re")] == ["/resume"]
    assert completions("/model") == completions("/help ") == completions("hi\n/") == []
    for command, completion in zip(COMMANDS, completions("/"), strict=True):
        assert command.usage in help_text()
        assert command.description in help_text()
        assert completion.display_meta_text == command.description
        action = parse_command(command.usage, "idle")
        assert action.kind == ("close" if command.name == "quit" else "command")
    assert parse_command("/help", "running").text == help_text()
    assert "Ctrl+O" in help_text()
    assert "Alt+Enter" in help_text()


@pytest.mark.parametrize(
    "text", ["/new arg", "/resume 'id'", "/help\n", "/quit\nhello", "/ session", "/help\r"]
)
def test_commands_reject_implicit_arguments_and_multiline(text):
    """命令不接受隐含参数或额外行。"""
    assert parse_command(text, "idle").kind == "error"


def test_unknown_commands_and_first_line_policy():
    """未知命令报错，只有首行首列斜杠触发命令解析。"""
    action = submission("/hlep", "idle")
    assert action.kind == "error" and "/help" in action.text
    assert submission("/model", "running").kind == "error"
    assert parse_command("/help \t", "idle").kind == "command"
    for text in (" /help", "hello\n/help", "\n/help", "!echo hello"):
        assert submission(text, "idle") == InputAction("submit", text)


@pytest.mark.parametrize("state", ["running", "compacting", "cancelling"])
def test_busy_command_policy(state):
    """忙碌状态只允许登记表允许的命令。"""
    for name in ("new", "resume"):
        assert parse_command(f"/{name}", state).kind == "error"
    for name in ("session", "clear", "help"):
        assert parse_command(f"/{name}", state).kind == "command"
    assert parse_command("/quit", state) == InputAction("close")


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
def test_empty_input(text):
    """纯空白提交不产生意图。"""
    assert submission(text, "idle") is None
    assert submission(text, "running", follow_up=True) is None


async def until(predicate):
    """在有限时间内等待编辑器状态，不永久阻塞测试。"""

    async def wait():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), 2)


@asynccontextmanager
async def editor_context():
    """使用管道输入启动编辑器，退出时取消并回收其任务。"""
    state = ["idle"]
    with create_pipe_input() as pipe:
        editor = InputEditor(
            lambda: Footer("/中文/work", "session", state[0], 2, 3),
            input=pipe,
            output=DummyOutput(),
        )
        task = asyncio.create_task(editor.run())
        try:
            await until(lambda: editor.session.app.is_running)
            yield editor, pipe, state, task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def action(editor):
    """限时取得下一输入意图，避免按键解析回归挂起。"""
    return await asyncio.wait_for(editor.next_action(), 2)


def test_real_key_dispatch_and_cancel_preserves_draft():
    """真实按键分派符合约定，取消保持编辑草稿。"""

    async def scenario():
        async with editor_context() as (editor, pipe, state, task):
            pipe.send_text("\r你好\r")  # Empty Enter must not produce an action.
            assert await action(editor) == InputAction("submit", "你好")
            state[0] = "running"
            pipe.send_text("调整\r")
            assert await action(editor) == InputAction("steer", "调整")
            for enter in ("\r", "\n"):
                pipe.send_text("稍后\x1b" + enter)
                assert await action(editor) == InputAction("follow_up", "稍后")
            pipe.send_text("草稿\x1b")
            assert await action(editor) == InputAction("cancel")
            assert editor.session.default_buffer.text == "草稿"
            state[0] = "idle"
            pipe.send_text("\x1b")
            # Let the bare Escape timeout finish, then submit the untouched draft.
            await asyncio.sleep(0.4)
            pipe.send_text("\r")
            assert await action(editor) == InputAction("submit", "草稿")
            pipe.send_text("\x04")
            assert await action(editor) == InputAction("close")
            await asyncio.wait_for(task, 2)

    asyncio.run(scenario())


def test_chinese_multiline_paste_emacs_and_history():
    """中文多行粘贴、编辑快捷键和历史回取可协同工作。"""

    async def scenario():
        async with editor_context() as (editor, pipe, state, task):
            pipe.send_text("第一行\x0f第二行\r")
            assert await action(editor) == InputAction("submit", "第一行\n第二行")
            pipe.send_text("\x1b[200~中文\r\n/help\x1b[201~")
            await until(lambda: editor.session.default_buffer.text == "中文\n/help")
            pipe.send_text("\r")
            assert await action(editor) == InputAction("submit", "中文\n/help")
            pipe.send_text("abc\x01\x06\x04Z\x05\r")
            assert await action(editor) == InputAction("submit", "aZc")
            pipe.send_text("/help\r")
            assert (await action(editor)).command == "help"
            pipe.send_text("\x1b[A")
            await until(lambda: editor.session.default_buffer.text == "aZc")
            pipe.send_text("\x1b[B")
            await until(lambda: editor.session.default_buffer.text == "")
            pipe.send_text("\x1b[A\r")
            assert await action(editor) == InputAction("submit", "aZc")
            assert editor.history.get_strings() == ["第一行\n第二行", "中文\n/help", "aZc"]
            pipe.send_text("/quit\r")
            assert await action(editor) == InputAction("close")
            await asyncio.wait_for(task, 2)

    asyncio.run(scenario())


def test_async_refresh_keeps_cursor_completion_and_current_state():
    """异步刷新保持光标及补全，同时读取最新状态。"""

    async def scenario():
        async with editor_context() as (editor, pipe, state, task):
            pipe.send_text("/re")
            buffer = editor.session.default_buffer
            await until(lambda: buffer.complete_state is not None)
            document, completion = buffer.document, buffer.complete_state
            state[0] = "running"
            rendered = asyncio.Event()
            editor.session.app.after_render += lambda _: rendered.set()
            editor.refresh()
            await asyncio.wait_for(rendered.wait(), 2)
            assert buffer.document == document
            assert buffer.complete_state is completion
            assert "running" in fragment_list_to_text(editor.toolbar())
            pipe.send_text("\t\r")
            error = await action(editor)
            assert error.kind == "error" and "空闲" in error.text
            assert buffer.text == "/resume"
            buffer.document = Document("中文草稿", cursor_position=2)
            rendered.clear()
            editor.refresh()
            await asyncio.wait_for(rendered.wait(), 2)
            assert buffer.document == Document("中文草稿", cursor_position=2)
            pipe.send_text("\x03")
            assert await action(editor) == InputAction("close")
            await asyncio.wait_for(task, 2)

    asyncio.run(scenario())


def test_input_stream_eof_closes():
    """输入流 EOF 生成关闭意图并结束编辑器。"""

    async def scenario():
        async with editor_context() as (editor, pipe, state, task):
            pipe.close()
            assert await action(editor) == InputAction("close")
            await asyncio.wait_for(task, 2)

    asyncio.run(scenario())
