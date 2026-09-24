"""在终端主屏幕协调输入、事件展示、运行任务和有序关闭。"""

import asyncio
from collections import deque
from collections.abc import Callable
from datetime import datetime

from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.application.current import set_app
from prompt_toolkit.formatted_text import FormattedText, fragment_list_to_text
from prompt_toolkit.output import Output
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.utils import get_cwidth

from lhagent.agents.coding import CodingAgent, create_coding_agent
from lhagent.harness.configs.types import CodingAgentConfig
from lhagent.harness.loop.types import LoopEvent, LoopResult
from lhagent.harness.session import SessionRepository
from lhagent.harness.session.session import Session
from lhagent.tui.actions import InputAction
from lhagent.tui.events import DisplayReducer
from lhagent.tui.history import project_history
from lhagent.tui.input import InputEditor
from lhagent.tui.models import NoticeBlock
from lhagent.tui.rendering import format_block
from lhagent.tui.session import SessionCoordinator
from lhagent.tui.styles import STYLE

# 仅控制终端展示，不限制模型输出或持久化历史。
_RENDER_INTERVAL_SECONDS = 1 / 30
_LIVE_TAIL_MAX_ROWS = 6
_LIVE_TAIL_SCREEN_DIVISOR = 3


class Scrollback:
    """已完成块只输出一次；活动尾部有界重绘，避免重复写入滚动历史。"""

    def __init__(self, output: Output) -> None:
        """借用终端输出，记录活动尾部的行数和上次尺寸。"""
        self.output = output
        self._live_rows = 0
        self._size: tuple[int, int] | None = None

    def clear(self) -> None:
        """清屏并重置重绘位置，不修改会话历史。"""
        self.output.erase_screen()
        self.output.cursor_goto(0, 0)
        self.output.flush()
        self._live_rows = 0

    def draw(self, display: DisplayReducer) -> None:
        """提交完成内容并替换有界活动尾部，尺寸改变时重置光标定位。"""
        size = self.output.get_size()
        width = max(1, size.columns - 1)  # 预留一列，避免终端延迟换行导致光标位置歧义。
        completed = display.drain_completed()
        live = display.live_blocks()
        if self._live_rows:
            if self._size != (size.rows, size.columns):
                # 尺寸变化后终端重排，旧行数已不能准确定位光标。
                self.output.erase_screen()
                self.output.cursor_goto(0, 0)
            else:
                self.output.cursor_up(self._live_rows)
                self.output.write_raw("\r")
                self.output.erase_down()
        for block in completed:
            text = format_block(block, width=width)
            if fragment_list_to_text(text):
                print_formatted_text(text, style=STYLE, output=self.output)
        # 只重绘有界尾部；完整响应在结束时一次交付，
        # 即使其高度超过终端，也不重复输出到滚动历史。
        lines: deque[list[tuple[str, str]]] = deque(
            [[]], maxlen=max(1, min(_LIVE_TAIL_MAX_ROWS, size.rows // _LIVE_TAIL_SCREEN_DIVISOR))
        )
        column = 0
        for block in live:
            if lines[-1]:
                lines.append([])
                column = 0
            for style, value, *_ in format_block(block, width=width):
                for char in value:
                    if char == "\n":
                        lines.append([])
                        column = 0
                        continue
                    value = " " * (4 - column % 4) if char == "\t" else char
                    cells = sum(max(0, get_cwidth(c)) for c in value)
                    if cells > width:
                        value, cells = "?", 1
                    if column + cells > width:
                        lines.append([])
                        column = 0
                    lines[-1].append((style, value))
                    column += cells
        rows = 0
        for line in lines:
            if line or len(lines) > 1:
                print_formatted_text(FormattedText(line), style=STYLE, output=self.output)
                rows += 1
        self.output.flush()
        self._live_rows = rows
        self._size = (size.rows, size.columns)


class InteractiveApp:
    """管理会话协调器、当前运行和输入编辑器；退出时等待任务及资源清理。"""

    def __init__(
        self,
        config: CodingAgentConfig,
        *,
        session_path: str | None = None,
        repository: SessionRepository | None = None,
        agent_factory: Callable[..., CodingAgent] = create_coding_agent,
        editor_factory: Callable[..., InputEditor] = InputEditor,
    ) -> None:
        """组装界面与任务状态；会话在 run 时打开。"""
        self.config = config
        self.session_path = session_path
        self.sessions = SessionCoordinator(
            config, repository=repository, agent_factory=agent_factory
        )
        self.display = DisplayReducer()
        self.editor = editor_factory(
            lambda: self.display.state.footer(
                config["cwd"],
                self.session.metadata["id"] if self.session is not None else "no session",
            )
        )
        self.scrollback = Scrollback(self.editor.session.app.output)
        self._events: asyncio.Queue[tuple[str, InputAction | None]] = asyncio.Queue()
        self._run_task: asyncio.Task[LoopResult] | None = None
        self._cancel_task: asyncio.Task[None] | None = None
        self._render_timer: asyncio.TimerHandle | None = None
        self._render_lock = asyncio.Lock()
        self._closing = False
        self._render_failed = False
        self._terminal_error: Exception | None = None
        self.ready = asyncio.Event()
        self._steering: deque[str] = deque()
        self._follow_up: deque[str] = deque()
        self._initial_user: str | None = None
        self._initial_committed = False

    @property
    def session(self) -> Session | None:
        """返回当前绑定会话；切换失败后可能为空。"""
        return self.sessions.session

    @property
    def agent(self) -> CodingAgent:
        """返回当前 agent；没有活动会话时抛出 RuntimeError。"""
        if self.sessions.agent is None:
            raise RuntimeError("No active session; use /new or /resume")
        return self.sessions.agent

    async def _switch(self, command: str) -> None:
        """只在空闲时切换会话；选择取消保持原绑定，切换失败回到无会话状态。"""
        if self._run_task is not None or self._cancel_task is not None:
            await self._notice(
                "Cannot switch while running; cancel first and wait for completion", "warning"
            )
            return
        metadata = None
        if command == "resume":
            candidates = await self.sessions.repository.list()
            if not candidates:
                await self._notice("No saved sessions", "warning")
                return
            with set_app(self.editor.session.app):
                metadata = await self.editor.select_session(
                    candidates,
                    self.session.metadata if self.session is not None else None,
                )
            if metadata is None:
                return
            if self.session is not None and metadata == self.session.metadata:
                await self._notice("Already using this session")
                return
        try:
            state = await self.sessions.switch(metadata, self._on_event)
        except Exception as exc:
            self.display = DisplayReducer()
            await self._notice(
                f"Session switch failed: {exc}\nNo active session; use /new or /resume", "error"
            )
            return
        finally:
            self._steering.clear()
            self._follow_up.clear()
            self._initial_user = None
            self._initial_committed = False
        self.display = DisplayReducer(state)
        assert self.session is not None
        self.display.state.transcript.insert(
            0,
            NoticeBlock(
                f"Session: {self.session.metadata['id']}\nPath: {self.session.metadata['path']}",
                "status",
            ),
        )
        await self._render()

    async def _render(self) -> None:
        """串行执行重绘，避免多个事件同时操作终端。"""
        async with self._render_lock:
            await self._draw()

    async def _draw(self) -> None:
        """暂停编辑器后输出内容；终端失败记录错误并请求关闭。"""
        if self._render_failed:
            return
        try:
            with set_app(self.editor.session.app):
                await run_in_terminal(lambda: self.scrollback.draw(self.display))
            self.editor.refresh()
        except Exception as exc:
            self._terminal_error = exc
            self._render_failed = True
            self._events.put_nowait(("close", None))

    async def _notice(self, text: str, kind: str = "status") -> None:
        """追加本地状态提示并立即刷新。"""
        self.display.state.transcript.append(NoticeBlock(text, kind))
        await self._render()

    async def _on_event(self, event: LoopEvent) -> None:
        """归并事件及已提交输入；仅流式增量合并刷新，终态立即展示。"""
        if self._closing:
            return
        self.display.apply(event)
        if event["type"] == "run_start":
            self._initial_committed = False
            if self._initial_user is not None:
                self.display.add_user(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._initial_user},
                        ],
                    },
                    event["data"]["run_id"],
                )
                self._initial_user = None
        if event["type"] == "message_committed" and event["data"]["kind"] == "user":
            if not self._initial_committed:
                self._initial_committed = True
            elif self._steering or self._follow_up:
                text = self._steering.popleft() if self._steering else self._follow_up.popleft()
                self.display.add_user(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                        ],
                    },
                    event["data"]["run_id"],
                )
            self.display.set_queue_counts(
                steering=len(self._steering), follow_up=len(self._follow_up)
            )
        if event["type"] == "compaction_start":
            await self._notice("Compaction started")
        elif event["type"] in ("compaction_end", "run_end"):
            data = event["data"]
            await self._notice(
                f"{event['type']}: {data['status']}"
                + (f" ({data['error']})" if data["error"] else ""),
                "error" if data["error"] else "status",
            )
        elif event["type"] == "response_update" and event["data"]["phase"] == "delta":
            if self._render_timer is None:
                self._render_timer = asyncio.get_running_loop().call_later(
                    _RENDER_INTERVAL_SECONDS,
                    self._request_render,
                )
        else:
            await self._render()

    def _request_render(self) -> None:
        """将定时刷新转为主事件队列任务，不直接并发绘制。"""
        self._render_timer = None
        self._events.put_nowait(("render", None))

    async def _inputs(self) -> None:
        """转发编辑器意图到应用队列，关闭意图交付后退出。"""
        while True:
            action = await self.editor.next_action()
            self._events.put_nowait(("action", action))
            if action.kind == "close":
                return

    async def _finish_run(self) -> None:
        """消费运行任务结果并恢复空闲状态，错误以提示展示。"""
        task = self._run_task
        if task is None:
            return
        try:
            result = task.result()
            if self.display.state.run_status == "running":
                await self._notice(
                    f"Run: {result['status']}"
                    + (f" ({result['error']})" if result["error"] else ""),
                    "error" if result["error"] else "status",
                )
        except BaseException as exc:
            await self._notice(f"Run error: {exc}", "error")
        self.display.state.run_status = "idle"
        self._run_task = None
        self.editor.refresh()

    async def _handle(self, action: InputAction) -> None:
        # 完成回调可能排在下一次按键之后，处理输入前先归并已结束任务。
        """按当前任务状态执行输入意图，取消期间不接纳新输入。"""
        if self._run_task is not None and self._run_task.done():
            await self._finish_run()
        if action.kind == "error":
            await self._notice(action.text, "error")
        elif action.kind == "command":
            if action.command == "clear":
                try:
                    with set_app(self.editor.session.app):
                        await run_in_terminal(self.scrollback.clear)
                    state = self.display.state
                    state.transcript = []
                    self.display = DisplayReducer(state)
                    self.editor.refresh()
                except Exception as exc:
                    self._terminal_error = exc
                    self._render_failed = True
                    self._events.put_nowait(("close", None))
            elif action.command == "help":
                await self._notice(action.text)
            elif action.command in ("new", "resume"):
                await self._switch(action.command)
            elif action.command == "history":
                if self.session is None:
                    await self._notice("No active session; use /new or /resume", "warning")
                    return
                if self._run_task is not None or self._cancel_task is not None:
                    await self._notice("Cannot show saved history while running", "warning")
                    return
                self.display = DisplayReducer(
                    project_history(
                        await self.session.get_display_history(),
                        await self.session.state(),
                    )
                )
                await self._render()
            elif action.command == "session":
                if self.session is None:
                    await self._notice("No active session; use /new or /resume", "warning")
                    return
                metadata = self.session.metadata
                session_state = await self.session.state()
                await self._notice(
                    f"Session: {metadata['id']}\nPath: {metadata['path']}\n"
                    f"Created: {datetime.fromtimestamp(metadata['created_at']).astimezone().isoformat()}\n"
                    f"State: {session_state['state']}\nActive run: {session_state['active_run_id'] or '-'}\n"
                    f"Last finished run: {session_state['last_finished_run_id'] or '-'}\n"
                    f"Interrupted: {session_state['interrupted']}",
                )
            else:
                await self._notice(f"/{action.command} is not available in this version", "warning")
        elif action.kind == "cancel":
            if (
                self._run_task is not None
                and not self._run_task.done()
                and self._cancel_task is None
            ):
                self.display.request_cancel()
                self._cancel_task = asyncio.create_task(self.agent.cancel())
                self._cancel_task.add_done_callback(
                    lambda _: self._events.put_nowait(("cancel_done", None))
                )
                await self._render()
        elif action.kind in ("submit", "steer", "follow_up"):
            if self.session is None:
                await self._notice("No active session; use /new or /resume", "warning")
                return
            busy = self._run_task is not None
            if not busy and self._cancel_task is None:
                self._initial_user = action.text
                self._run_task = asyncio.create_task(self.agent.run(action.text))
                self._run_task.add_done_callback(
                    lambda _: self._events.put_nowait(("run_done", None))
                )
                self.display.state.run_status = "running"
                await self._render()
            elif busy and self._cancel_task is None:
                if action.kind == "follow_up":
                    self.agent.follow_up(action.text)
                    self._follow_up.append(action.text)
                    label = "follow-up"
                else:
                    self.agent.steer(action.text)
                    self._steering.append(action.text)
                    label = "steering"
                self.display.set_queue_counts(
                    steering=len(self._steering), follow_up=len(self._follow_up)
                )
                await self._notice(f"Queued {label}")
            else:
                await self._notice("Cancellation in progress; input was not queued", "warning")

    async def run(self) -> None:
        """驱动单消费者事件循环；退出时取消运行并回收编辑器、会话和任务。"""
        editor_task = input_task = None
        try:
            self.display = DisplayReducer(
                await self.sessions.start(self.session_path, self._on_event)
            )
            assert self.session is not None
            self.display.state.transcript.insert(
                0,
                NoticeBlock(
                    f"Session: {self.session.metadata['id']}\nPath: {self.session.metadata['path']}",
                    "status",
                ),
            )
            editor_task = asyncio.create_task(self.editor.run())
            editor_task.add_done_callback(lambda _: self._events.put_nowait(("close", None)))
            input_task = asyncio.create_task(self._inputs())
            input_task.add_done_callback(lambda _: self._events.put_nowait(("input_done", None)))
            await self.editor.ready.wait()
            if editor_task.done():
                editor_task.result()  # 传播终端初始化失败，不能将就绪信号当作初始化成功。
            await self._render()
            self.ready.set()
            while True:
                kind, action = await self._events.get()
                if kind == "close" or (action is not None and action.kind == "close"):
                    break
                if kind == "input_done":
                    input_task.result()
                    break
                if kind == "render":
                    await self._render()
                elif kind == "run_done":
                    if self._run_task is not None and self._run_task.done():
                        await self._finish_run()
                elif kind == "cancel_done":
                    assert self._cancel_task is not None
                    try:
                        self._cancel_task.result()
                    except BaseException as exc:
                        await self._notice(f"Cancel error: {exc}", "error")
                    self._cancel_task = None
                elif action is not None:
                    try:
                        await self._handle(action)
                    except Exception as exc:
                        await self._notice(f"Action error: {exc}", "error")
            if editor_task.done():
                editor_task.result()
            if input_task.done():
                input_task.result()
            if self._terminal_error is not None:
                raise RuntimeError(
                    f"terminal output failed: {self._terminal_error}"
                ) from self._terminal_error
        finally:
            self._closing = True
            if self._render_timer is not None:
                self._render_timer.cancel()
                self._render_timer = None
            if input_task is not None:
                input_task.cancel()
                await asyncio.gather(input_task, return_exceptions=True)
            try:
                if (
                    self._run_task is not None
                    and not self._run_task.done()
                    and self._cancel_task is None
                ):
                    try:
                        await self.agent.cancel()
                    except Exception:
                        pass
                if self._cancel_task is not None:
                    await asyncio.gather(self._cancel_task, return_exceptions=True)
                if self._run_task is not None:
                    await asyncio.gather(self._run_task, return_exceptions=True)
                await self.sessions.close()
            finally:
                for task in (input_task, editor_task):
                    if task is not None:
                        task.cancel()
                await asyncio.gather(
                    *(task for task in (input_task, editor_task) if task is not None),
                    return_exceptions=True,
                )


async def run_interactive(config: CodingAgentConfig, session_path: str | None = None) -> None:
    """启动交互应用；默认新建会话，也可恢复仓库校验过的 JSONL 路径。"""
    await InteractiveApp(config, session_path=session_path).run()
