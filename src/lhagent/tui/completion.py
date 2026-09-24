"""仅从命令注册表生成斜杠补全，不维护第二份命令列表。"""

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from lhagent.tui.commands import COMMANDS


class CommandCompleter(Completer):
    """从唯一注册表补全首行斜杠命令，不补全普通模型输入。"""

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        """仅在可识别的命令前缀位置提供名称及说明。"""
        prefix = document.text_before_cursor
        if (
            not prefix.startswith("/")
            or any(c.isspace() for c in prefix)
            or document.text_after_cursor
        ):
            return
        for command in COMMANDS:
            if command.usage.startswith(prefix):
                yield Completion(
                    command.usage,
                    start_position=-len(prefix),
                    display=command.usage,
                    display_meta=command.description,
                )
