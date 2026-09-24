"""声明会话元数据、JSONL 记录和恢复状态的数据契约。

记录格式采用追加式 JSONL。已终结的 assistant 响应和已结束的工具结果才进入
日志，终结响应允许包含残缺内容；流式中间结果和未完成工具结果不被伪造落盘。
"""

from typing import Literal, TypedDict

from lhagent.client.types import ClientResult, Message
from lhagent.harness.tools.types import ToolResult

RunStatus = Literal["completed", "length", "error", "cancelled"]
SessionState = Literal["new", "active", "idle", "interrupted", "closed"]


class SessionMetadata(TypedDict):
    """一个会话的稳定身份和创建信息。

    id 在会话仓库内唯一；created_at 使用 Unix 时间戳秒数。
    path 是实际 JSONL 文件路径，由仓库填充，不由 loop 修改。
    """

    id: str
    created_at: float
    path: str


class HistoryExclusion(TypedDict):
    """将已经保存但不适合重放的记录排除出有效历史。"""

    entry_ids: list[str]
    reason: Literal["overflow_recovery", "manual", "other"]


class SessionRecordBase(TypedDict):
    """所有 JSONL 记录共有的身份、顺序和时间字段；类型由各记录声明。"""

    id: str
    run_id: str
    timestamp: float


class SessionRecord(SessionRecordBase):
    """文件首行：id 为会话 ID，run_id 为空，timestamp 为创建时间。"""

    type: Literal["session"]
    metadata: SessionMetadata


class RunStartRecord(SessionRecordBase):
    """标识一次运行开始的持久化记录。"""

    type: Literal["run_start"]


class UserRecord(SessionRecordBase):
    """用户消息记录，保留所属运行与独立条目身份。"""

    type: Literal["user"]
    message: Message


class AssistantRecord(SessionRecordBase):
    """已终结的模型响应记录，允许保留失败或截断内容。"""

    type: Literal["assistant"]
    response: ClientResult


class ToolResultRecord(SessionRecordBase):
    """已结束的工具结果记录，不伪造进行中的执行结果。"""

    type: Literal["tool_result"]
    result: ToolResult


class HistoryExclusionRecord(SessionRecordBase):
    """按条目 ID 排除有效历史，不删除原始记录。"""

    type: Literal["history_exclusion"]
    entry_ids: list[str]
    reason: Literal["overflow_recovery", "manual", "other"]


class CompactionRecord(SessionRecordBase):
    """滚动摘要及保留尾部的引用记录，用于恢复有效历史。"""

    type: Literal["compaction"]
    summary: str
    retained_entry_ids: list[str]
    tokens_before: int
    estimated_tokens_after: int
    usage: list[dict[str, object]]


class RunFinishRecord(SessionRecordBase):
    """运行终态与错误记录，用于恢复运行生命周期。"""

    type: Literal["run_finish"]
    status: RunStatus
    error: str | None


JsonlRecord = (
    SessionRecord
    | RunStartRecord
    | UserRecord
    | AssistantRecord
    | ToolResultRecord
    | HistoryExclusionRecord
    | CompactionRecord
    | RunFinishRecord
)


class SessionStateInfo(TypedDict):
    """打开会话后可供上层查看的生命周期状态。

    state、active_run_id 描述当前会话；打开时未收尾的旧运行报告 interrupted，
    但不充当仍在执行的运行锁。interrupted 标志保留旧中断事实，不因启动新
    运行而伪造旧运行已完成。new 表示尚未运行，active 表示正在运行，idle
    表示当前运行已收尾；恢复中断且未启动新运行时为 interrupted，关闭后为
    closed。active_run_id 只标识当前进程正在执行的运行，否则为 None。
    """

    state: SessionState
    active_run_id: str | None
    last_finished_run_id: str | None
    interrupted: bool


class DisplayEntryBase(TypedDict):
    """展示身份与时间；in_context 表示仍在有效历史中，不推测执行状态。"""

    entry_id: str
    run_id: str
    timestamp: float
    in_context: bool


class DisplayUserEntry(DisplayEntryBase):
    """带上下文归属的用户消息展示投影。"""

    type: Literal["user"]
    message: Message


class DisplayAssistantEntry(DisplayEntryBase):
    """保留完整终结响应的模型展示投影。"""

    type: Literal["assistant"]
    response: ClientResult


class DisplayToolEntry(DisplayEntryBase):
    """保留结果详情的工具展示投影。"""

    type: Literal["tool_result"]
    result: ToolResult


DisplayEntry = DisplayUserEntry | DisplayAssistantEntry | DisplayToolEntry


class DisplayHistory(TypedDict):
    """按提交顺序保留所有消息及完整结果的独立展示投影，无存储记录。"""

    summary: str | None
    entries: list[DisplayEntry]


class SessionCreateOptions(TypedDict, total=False):
    """可选 ID 与 .jsonl 路径；相对路径以仓库默认目录为基准。"""

    id: str
    path: str


class SessionListOptions(TypedDict, total=False):
    """替换单层扫描目录；相对目录以调用时工作目录为基准。"""

    directory: str


class SessionRepositoryOptions(TypedDict, total=False):
    """默认目录配置；未提供时使用启动目录下的 .lhagent/sessions，构造时固定路径。"""

    directory: str


class SessionCloseResult(TypedDict):
    """会话关闭结果，表示关闭请求已经完成。"""

    session_id: str
    state: Literal["closed"]
