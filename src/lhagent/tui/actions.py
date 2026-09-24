"""不持有资源的输入意图；实际操作由交互应用执行。"""

from dataclasses import dataclass
from typing import Literal

InputState = Literal["idle", "running", "cancelling", "compacting"]
ActionKind = Literal["submit", "steer", "follow_up", "cancel", "command", "close", "error"]


@dataclass(frozen=True)
class InputAction:
    """不可变输入意图，携带文本或命令名，不执行操作。"""

    kind: ActionKind
    text: str = ""
    command: str | None = None
