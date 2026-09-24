"""统一维护斜杠命令的语法、状态限制、帮助和意图处理器。"""

from collections.abc import Callable
from dataclasses import dataclass
from difflib import get_close_matches

from lhagent.tui.actions import ActionKind, InputAction, InputState

ALL_STATES: frozenset[InputState] = frozenset({"idle", "running", "cancelling", "compacting"})
SHORTCUT_HELP = (
    "Enter: 空闲提交 / 忙碌时 steering\n"
    "Alt+Enter: 忙碌时 follow-up / 空闲提交（终端编码 ESC CR 或 ESC LF）\n"
    "Ctrl+O: 插入换行（Shift/Ctrl+Enter 无通用终端编码，不保证支持）\n"
    "Alt+Enter 被终端拦截时可依次按 Esc、Enter（0.3 秒内）\n"
    "Esc: 忙碌时取消；空闲无操作\n"
    "↑/↓: 编辑区边界处取回本进程普通输入历史（不含 slash command）\n"
    "Ctrl+D: 空缓冲区 EOF；非空时向前删除\n"
    "Ctrl+C / /quit: 请求有序关闭"
)


def _command(name: str) -> InputAction:
    """将注册表命令名包装为待执行意图。"""
    return InputAction("command", command=name)


def _help(name: str) -> InputAction:
    """附带当前注册表帮助文本，不执行终端输出。"""
    return InputAction("command", text=help_text(), command=name)


def _quit(name: str) -> InputAction:
    """生成关闭意图，由应用负责有序退出。"""
    return InputAction("close")


@dataclass(frozen=True)
class Command:
    """命令的唯一注册信息，包含使用形式、允许状态和意图处理器。"""

    name: str
    argument_hint: str
    description: str
    allowed_states: frozenset[InputState]
    handler: Callable[[str], InputAction]

    @property
    def usage(self) -> str:
        """返回命令的标准用法，供帮助、补全与解析共用。"""
        return f"/{self.name}" + (f" {self.argument_hint}" if self.argument_hint else "")


# 注册表实例依赖上方的 Command 和处理器，必须在其定义后构造。
COMMANDS = (
    Command("new", "", "创建并切换到新会话", frozenset({"idle"}), _command),
    Command("resume", "", "打开会话选择器", frozenset({"idle"}), _command),
    Command("session", "", "显示当前会话信息", ALL_STATES, _command),
    Command("history", "", "显示当前会话已保存的对话历史", frozenset({"idle"}), _command),
    Command("clear", "", "清理终端展示，保留会话历史", ALL_STATES, _command),
    Command("help", "", "显示命令和快捷键", ALL_STATES, _help),
    Command("quit", "", "有序退出", ALL_STATES, _quit),
)


def help_text() -> str:
    """由注册表生成命令帮助，附带统一快捷键说明。"""
    lines = ["命令（均不接受参数，仅允许单行；命令名区分大小写）："]
    for command in COMMANDS:
        policy = " [仅空闲]" if command.allowed_states != ALL_STATES else ""
        lines.append(f"{command.usage}: {command.description}{policy}")
    return "\n".join([*lines, "", SHORTCUT_HELP])


def parse_command(text: str, state: InputState) -> InputAction | None:
    """仅识别首行首列的斜杠命令，不执行 shell。

    命令均不接受参数；允许尾部水平空白，额外行（含空行）视为错误。"""
    if not text.startswith("/"):
        return None
    first_line = text.split("\n", 1)[0]
    name = first_line[1:].split(maxsplit=1)[0] if first_line[1:].strip() else ""
    command = next((item for item in COMMANDS if item.name == name), None)
    if command is None:
        matches = get_close_matches(name, [item.name for item in COMMANDS], n=3, cutoff=0.5)
        suggestion = (
            "；是否要输入 " + "、".join(f"/{item}" for item in matches)
            if matches
            else "；使用 /help 查看命令"
        )
        return InputAction("error", f"未知命令 /{name}{suggestion}")
    if text.rstrip(" \t") != command.usage:
        return InputAction("error", f"用法：{command.usage}（单行，不接受参数）")
    if state not in command.allowed_states:
        return InputAction(
            "error", f"/{name} 仅可在空闲时使用；当前状态：{state}。请先按 Esc 取消并等待结束"
        )
    return command.handler(name)


def submission(text: str, state: InputState, *, follow_up: bool = False) -> InputAction | None:
    """忽略空白输入，优先解析命令，再按空闲/忙碌状态生成提交或排队意图。"""
    if not text.strip():
        return None
    command = parse_command(text, state)
    if command is not None:
        return command
    kind: ActionKind = "submit" if state == "idle" else "follow_up" if follow_up else "steer"
    return InputAction(kind, text)
