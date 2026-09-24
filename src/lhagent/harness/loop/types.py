"""声明循环配置、运行结果、严格事件联合和会话协作接口。"""

from collections.abc import Awaitable, Callable
from typing import Literal, Protocol, TypedDict

from lhagent.client.types import ClientResult, Message
from lhagent.harness.context.types import (
    CompactionResult,
    CompactionSettings,
    ContextBudget,
    HistoryContext,
    PromptBundle,
    SummaryRequest,
)
from lhagent.harness.tools.types import ToolDefinition, ToolResult


class LoopConfig(TypedDict):
    """外部确定的一次运行配置，不包含总轮数、总 token 或总耗时限制。

    tools 是本次运行固定的可用集合，发送给模型的描述必须与之一致。
    parameters 提供生成参数，不得覆盖 loop 生成的工具定义和消息。
    summary_request 适配 Client.complete，摘要不进入普通工具循环；回调须将
    loop 传入的 cancel_event 原样传给 client，不自行创建独立取消域。
    max_summary_output_tokens 是 agent 根据显式配置及模型输出能力确定的上限。
    新 call_id 须在共享客户端的进行中调用之间唯一，包括摘要调用。
    new_run_id 生成会话内唯一的运行 ID，与模型调用 ID 分开管理。
    """

    model: str
    parameters: dict[str, object]
    prompts: PromptBundle
    budget: ContextBudget
    compaction: CompactionSettings
    summary_request: SummaryRequest
    max_summary_output_tokens: int
    tools: list[ToolDefinition]
    new_call_id: Callable[[], str]
    new_run_id: Callable[[], str]


class LoopResult(TypedDict):
    """本次运行的终态，不把停止生成解释为任务质量验证通过。

    length 表示未成功恢复的输出截断；错误及取消保留已提交记录。
    最后一次主请求结果可为空，例如在首个模型请求前就被取消。
    """

    status: Literal["completed", "length", "error", "cancelled"]
    last_response: ClientResult | None
    error: str | None


class RunEventData(TypedDict):
    """运行事件共用的运行身份。"""

    run_id: str


class ResponseDeltaData(RunEventData):
    """带公共块索引的模型增量事件载荷。"""

    phase: Literal["delta"]
    call_id: str
    block_index: int
    delta: dict[str, object]


class ResponseEndData(RunEventData):
    """模型调用终结快照，供订阅者校准增量内容。"""

    phase: Literal["end"]
    call_id: str
    result: ClientResult


class MessageCommittedData(RunEventData):
    """仅通知持久化完成；不作为响应或工具展示的数据源。"""

    entry_id: str
    kind: Literal["user", "assistant", "tool_result"]


class ToolStartData(RunEventData):
    """工具执行开始载荷，包含调用身份和模型参数；schema 校验由执行层完成。"""

    tool_call_id: str
    tool_name: str
    arguments: object


class ToolEndData(RunEventData):
    """工具执行结束载荷，包含状态及完整结果。"""

    tool_call_id: str
    tool_name: str
    status: Literal["success", "validation_error", "execution_error", "timeout", "cancelled"]
    result: ToolResult


class CompactionStartData(RunEventData):
    """压缩开始时的输入 token 估算。"""

    tokens: int


class CompactionEndData(RunEventData):
    """摘要生成终态；success 不表示已经提交会话。"""

    status: Literal["success", "unchanged", "error", "cancelled"]
    error: str | None


class RunEndData(RunEventData):
    """循环终态及错误，关联当前运行。"""

    status: Literal["completed", "length", "error", "cancelled"]
    error: str | None


class RunStartEvent(TypedDict):
    """带 run_start 判别字段的运行开始事件。"""

    type: Literal["run_start"]
    data: RunEventData


class ResponseUpdateEvent(TypedDict):
    """带 response_update 判别字段的增量或终结事件。"""

    type: Literal["response_update"]
    data: ResponseDeltaData | ResponseEndData


class MessageCommittedEvent(TypedDict):
    """持久化提交事件，不代表新增一份展示内容。"""

    type: Literal["message_committed"]
    data: MessageCommittedData


class ToolStartEvent(TypedDict):
    """带 tool_start 判别字段的工具开始事件。"""

    type: Literal["tool_start"]
    data: ToolStartData


class ToolEndEvent(TypedDict):
    """带 tool_end 判别字段的工具结束事件。"""

    type: Literal["tool_end"]
    data: ToolEndData


class CompactionStartEvent(TypedDict):
    """带 compaction_start 判别字段的压缩开始事件。"""

    type: Literal["compaction_start"]
    data: CompactionStartData


class CompactionEndEvent(TypedDict):
    """带 compaction_end 判别字段的摘要生成结果事件。"""

    type: Literal["compaction_end"]
    data: CompactionEndData


class RunEndEvent(TypedDict):
    """带 run_end 判别字段的唯一运行终结事件。"""

    type: Literal["run_end"]
    data: RunEndData


# 按 type 收窄；response_update 再按 data.phase 区分增量与完整终结结果。
# 每个载荷的所有声明字段均必填，通知使用独立深副本，不是持久化日志。
LoopEvent = (
    RunStartEvent
    | ResponseUpdateEvent
    | MessageCommittedEvent
    | ToolStartEvent
    | ToolEndEvent
    | CompactionStartEvent
    | CompactionEndEvent
    | RunEndEvent
)


# 事件接收方可异步处理；循环结束须等待已经接受的事件处理完成。
EventSink = Callable[[LoopEvent], Awaitable[None]]


class LoopSession(Protocol):
    """会话层向 loop 提供的协作边界，不规定文件格式、分支或恢复实现。

    返回的历史是有效视图，原始记录由会话层持有。追加接口完成即表示该条
    记录已按会话层约定提交；失败时 loop 不继续启动新的模型或工具调用。
    """

    async def start_run(self, run_id: str) -> None:
        """提交唯一运行 ID 的开始记录，成功后才允许提交用户消息。

        会话关联后续记录和 finish_run；已有活跃运行时拒绝重入。
        恢复出的中断运行保留原状，新运行不得自动重放旧工具。
        """
        ...

    async def get_history(self) -> HistoryContext:
        """取得摘要和排除投影后的历史及对应 entry_ids；重放修复由 context 完成。"""
        ...

    async def append_user(self, message: Message) -> str:
        """提交实际接入的用户消息并返回记录标识，避免新指令重复加入上下文。"""
        ...

    async def append_response(self, response: ClientResult) -> str:
        """提交任一终结状态的响应，允许部分内容；不接受 pending 或 delta。"""
        ...

    async def append_tool_result(self, result: ToolResult) -> str:
        """提交单个工具结果并保留调用 ID 关联，完成后才能推进下一条工具。"""
        ...

    async def omit_failed_attempt(self, entry_ids: list[str]) -> None:
        """将失败模型响应及关联结果排除出有效历史，保留原始记录。

        此操作独立提交；后续压缩失败不撤销已提交的排除操作。
        """
        ...

    async def commit_compaction(self, result: CompactionResult) -> None:
        """仅提交完整成功的压缩结果，不保存部分摘要或删除原始记录。

        按 history.entry_ids 校验保留尾部及消息身份，持久化原始记录引用，
        不复制消息或保存修复占位；空尾部表示此前消息全部由摘要替代。
        """
        ...

    async def finish_run(self, result: LoopResult) -> None:
        """关联 start_run 的当前运行提交终态，成功后解除运行，不关闭会话。"""
        ...
