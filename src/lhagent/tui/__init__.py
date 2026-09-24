"""交互终端的公开入口，汇集展示、输入和会话协调接口。"""

from lhagent.tui.actions import InputAction, InputState
from lhagent.tui.app import InteractiveApp, run_interactive
from lhagent.tui.commands import COMMANDS, help_text, parse_command, submission
from lhagent.tui.completion import CommandCompleter
from lhagent.tui.events import DisplayReducer
from lhagent.tui.history import project_history
from lhagent.tui.input import InputEditor
from lhagent.tui.models import DisplayState, Footer, MessageBlock, NoticeBlock, ToolBlock
from lhagent.tui.rendering import (
    Preview,
    PreviewLimits,
    format_block,
    format_footer,
    preview,
)
from lhagent.tui.styles import STYLE

__all__ = [
    "DisplayReducer",
    "DisplayState",
    "Footer",
    "MessageBlock",
    "NoticeBlock",
    "ToolBlock",
    "Preview",
    "PreviewLimits",
    "project_history",
    "format_block",
    "format_footer",
    "preview",
    "STYLE",
    "InputAction",
    "InputState",
    "InputEditor",
    "COMMANDS",
    "CommandCompleter",
    "help_text",
    "parse_command",
    "submission",
    "InteractiveApp",
    "run_interactive",
]
