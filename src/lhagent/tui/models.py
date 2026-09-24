"""实时事件和历史投影共用的展示数据，不持有运行资源。"""

from dataclasses import dataclass, field
from typing import Literal

from lhagent.client.types import ContentBlock
from lhagent.harness.tools.types import ToolResult


@dataclass
class MessageBlock:
    """用户或模型消息的展示快照，完整性和上下文归属分别记录。"""

    role: Literal["user", "assistant"]
    run_id: str
    content: list[ContentBlock] = field(default_factory=list)
    call_id: str | None = None
    status: str | None = None
    error: str | None = None
    complete: bool = False
    in_context: bool = True


@dataclass
class ToolBlock:
    """单次工具调用展示，按运行与调用 ID 关联参数、状态及结果。"""

    run_id: str
    tool_call_id: str
    name: str
    arguments: object = None
    status: str = "pending"
    result: ToolResult | None = None
    complete: bool = False
    in_context: bool = True


@dataclass
class NoticeBlock:
    """已完成的本地提示块，不写入模型会话。"""

    text: str
    kind: str = "status"
    complete: bool = True


TranscriptBlock = MessageBlock | ToolBlock | NoticeBlock


@dataclass(frozen=True)
class Footer:
    """不可变页脚快照，包含当前状态和两类排队输入数。"""

    cwd: str
    session_id: str
    status: Literal["idle", "running", "cancelling", "compacting"]
    steering: int
    follow_up: int


@dataclass
class DisplayState:
    """界面状态及有序内容块；不保存 agent、存储或终端句柄。"""

    transcript: list[TranscriptBlock] = field(default_factory=list)
    run_id: str | None = None
    run_status: str = "idle"
    cancelling: bool = False
    compaction_status: str | None = None
    compaction_tokens: int | None = None
    steering: int = 0
    follow_up: int = 0
    recent_error: str | None = None
    notifications: list[str] = field(default_factory=list)

    def footer(self, cwd: str, session_id: str) -> Footer:
        """按取消、压缩、运行的优先级派生页脚，缩短会话标识用于展示。"""
        status: Literal["idle", "running", "cancelling", "compacting"] = "idle"
        if self.run_status == "running":
            status = "running"
        if self.compaction_status == "running":
            status = "compacting"
        if self.cancelling:
            status = "cancelling"
        return Footer(cwd, session_id[:8], status, self.steering, self.follow_up)
