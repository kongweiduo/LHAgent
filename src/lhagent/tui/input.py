"""非全屏输入编辑器，只交付输入意图，不拥有 agent 资源。"""

import asyncio
from collections.abc import Callable
from datetime import datetime

from prompt_toolkit import PromptSession
from prompt_toolkit.application import in_terminal
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.output import Output, create_output
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.validation import Validator

from lhagent.harness.session.types import SessionMetadata
from lhagent.tui.actions import InputAction
from lhagent.tui.commands import submission
from lhagent.tui.completion import CommandCompleter
from lhagent.tui.models import Footer
from lhagent.tui.rendering import format_footer
from lhagent.tui.styles import STYLE

# 终端按键解析等待与选择器路径展示预算，不读取环境或创建终端。
_TERMINAL_ESCAPE_TIMEOUT_SECONDS = 0.05
_KEY_SEQUENCE_TIMEOUT_SECONDS = 0.3
_SESSION_PATH_CHARACTERS = 72


class InputEditor:
    """由应用拥有的任务运行一次，通过 next_action 消费意图。

    页脚提供器是当前状态的唯一来源；refresh 不重置文本、光标、撤销或补全。
    关闭意图停止键盘读取，有序关闭由消费者负责；取消运行任务也释放终端资源。"""

    def __init__(
        self,
        footer: Callable[[], Footer],
        *,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        """借用输入输出并注册编辑快捷键；暂不开始读取终端。"""
        self.ready = asyncio.Event()
        self._footer = footer
        self._actions: asyncio.Queue[InputAction] = asyncio.Queue()
        self.history = InMemoryHistory()
        if output is None:
            # 显式指定 output，避免 TERM=dumb 时 prompt_toolkit 切换应用导致就绪等待挂起。
            output = create_output()
        bindings = KeyBindings()

        @bindings.add("c-m")
        def submit(event: KeyPressEvent) -> None:
            self._submit()

        @bindings.add("escape", "c-m")
        @bindings.add("escape", "c-j")
        def follow_up(event: KeyPressEvent) -> None:
            self._submit(follow_up=True)

        @bindings.add("c-o")
        def newline(event: KeyPressEvent) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add("escape")
        def cancel(event: KeyPressEvent) -> None:
            if self._footer().status != "idle":
                self._actions.put_nowait(InputAction("cancel"))

        @bindings.add("c-d")
        def eof(event: KeyPressEvent) -> None:
            if not event.current_buffer.text:
                self._close()
            else:
                event.current_buffer.delete()

        @bindings.add("c-c")
        def interrupt(event: KeyPressEvent) -> None:
            self._close()

        self.session: PromptSession[None] = PromptSession(
            "> ",
            multiline=True,
            key_bindings=bindings,
            history=self.history,
            completer=CommandCompleter(),
            complete_while_typing=True,
            bottom_toolbar=self.toolbar,
            style=STYLE,
            input=input,
            output=output,
            enable_open_in_editor=False,
        )
        # 延迟单独 Esc 的判定，让解析器能区分 ESC CR（Alt+Enter）组合。
        self.session.app.ttimeoutlen = _TERMINAL_ESCAPE_TIMEOUT_SECONDS
        self.session.app.timeoutlen = _KEY_SEQUENCE_TIMEOUT_SECONDS

    async def select_session(
        self,
        sessions: list[SessionMetadata],
        current: SessionMetadata | None,
    ) -> SessionMetadata | None:
        """暂停主编辑器，显示带编号的非全屏选择器。

        保持仓库顺序及原始元数据；未打开会话明确提示正文和状态尚未验证。"""
        if not sessions:
            return None
        bindings = KeyBindings()

        @bindings.add("escape")
        def cancel(event: KeyPressEvent) -> None:
            event.app.exit(result="")

        choices = {str(index): metadata for index, metadata in enumerate(sessions, 1)}
        selector: PromptSession[str] = PromptSession(
            input=self.session.app.input,
            output=self.session.app.output,
            key_bindings=bindings,
            validator=Validator.from_callable(
                lambda text: not text.strip() or text.strip() in choices,
                error_message="Choose a listed number, or Enter to cancel",
            ),
        )
        async with in_terminal():
            for number, metadata in choices.items():
                created = datetime.fromtimestamp(metadata["created_at"]).astimezone().isoformat()
                status = "current" if metadata == current else "saved; state checked on open"
                path = metadata["path"]
                if len(path) > _SESSION_PATH_CHARACTERS:
                    path = "…" + path[-(_SESSION_PATH_CHARACTERS - 1) :]
                print_formatted_text(
                    f"{number}. {created}  {metadata['id'][:12]}  [{status}]  {path}",
                    output=self.session.app.output,
                )
            try:
                result = await selector.prompt_async("Resume session (Enter/Esc cancels): ")
            except (EOFError, KeyboardInterrupt):
                self._actions.put_nowait(InputAction("close"))
                return None
        return choices.get(result.strip())

    def toolbar(self) -> FormattedText:
        """从当前页脚状态和终端宽度生成工具栏。"""
        return format_footer(
            self._footer(),
            width=max(1, self.session.app.output.get_size().columns),
        )

    def refresh(self) -> None:
        """只使绘制失效，保留编辑缓冲区与光标。"""
        self.session.app.invalidate()

    async def next_action(self) -> InputAction:
        """等待下一条输入意图，不执行其对应操作。"""
        return await self._actions.get()

    async def run(self) -> None:
        """读取终端直至结束；EOF/中断转为关闭意图，失败也释放就绪等待者。"""
        try:
            await self.session.prompt_async(pre_run=self.ready.set)
        except (EOFError, KeyboardInterrupt):
            self._actions.put_nowait(InputAction("close"))
        finally:
            self.ready.set()

    def _close(self) -> None:
        """交付关闭意图并结束编辑器，由应用继续清理运行资源。"""
        self._actions.put_nowait(InputAction("close"))
        self.session.app.exit()

    def _submit(self, *, follow_up: bool = False) -> None:
        """按状态解析缓冲区；保留无效输入，普通输入加入历史后清空。"""
        buffer = self.session.default_buffer
        action = submission(buffer.text, self._footer().status, follow_up=follow_up)
        if action is None:
            return
        if action.kind == "close":
            self._close()
            return
        self._actions.put_nowait(action)
        if action.kind == "error":
            return  # 保留无效输入，方便用户修正。
        if action.kind in ("submit", "steer", "follow_up"):
            buffer.append_to_history()
        buffer.reset()
        # 立即开始重载历史，避免等待受节流限制的重绘。
        buffer.load_history_if_not_yet_loaded()
