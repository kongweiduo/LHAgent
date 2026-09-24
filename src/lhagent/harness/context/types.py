"""声明上下文素材、有效历史、预算和压缩结果的数据契约。

消息用结构化字典表示，保留角色、内容块和工具调用关联信息；
内容遵循 client 的消息契约，不在此拼接纯文本会话。
"""

from asyncio import Event
from collections.abc import Awaitable, Callable
from typing import Literal, NotRequired, TypedDict

from lhagent.client.types import ClientResult, Message, TokenUsage


class PromptSources(TypedDict):
    """调用方指定的提示词文件来源，不包含自动发现或检索策略。

    系统提示词文件可缺省；附加提示词文件按传入顺序读取为 system 消息。
    所有文件使用 UTF-8，相对路径以调用时工作目录为基准。
    直接提供的提示词文本可由调用方构造 PromptBundle。
    """

    system_prompt_path: str | None
    additional_prompt_paths: list[str]


class PromptBundle(TypedDict):
    """独立于可压缩历史的提示词素材，附加消息保留各自的角色。

    系统提示词和这些固定附加提示词不参与 history 压缩。
    任务过程中产生的普通消息应进入 history，不应放进此结构逃避压缩。
    """

    system_prompt: str | None
    additional_messages: list[Message]


class HistoryContext(TypedDict):
    """当前有效历史，由最新摘要和尚未被摘要替代的消息组成。

    messages 按时间排序，包含用户、模型和工具消息，不包含 summary 的副本。
    此结构是原始会话的上下文视图，不是持久化日志；外部负责保存和恢复。
    消息保留 finish_reason 和工具关联，尚未进行模型重放修复；context 的
    组装、预算和摘要资料均使用统一修复规则，原始消息对象不被改写。
    entry_ids 与 messages 等长且逐项对应，来自原始 user/assistant/tool_result
    记录的唯一 id，不是模型 call_id。摘要没有对应列表项；临时修复占位没有
    记录 ID。该列表只用于关联存储，不发送给模型、不计入 token 或摘要资料。
    外部构造历史也必须提供稳定且不重复的 ID，不能通过消息内容反推身份。
    """

    summary: str | None
    messages: list[Message]
    entry_ids: list[str]


class ContextInput(TypedDict):
    """本次组装需要的提示词、有效历史与可选新指令。

    new_instruction 仅用于尚未加入 history 的用户消息，避免重复发送。
    指令进入会话后按普通历史处理，本轮尚未结束也允许被压缩。
    已提交指令应传 None；若仍提供消息副本，必须通过 new_instruction_entry_id
    指定记录身份以防重复。相同内容不能作为同一记录的证据。
    """

    prompts: PromptBundle
    history: HistoryContext
    new_instruction: Message | None
    new_instruction_entry_id: NotRequired[str]


class CompactionSettings(TypedDict):
    """压缩阈值与近期保留预算，使用 token 数而非消息条数。

    enabled 仅控制自动压缩判断；reserve_tokens 为窗口预留空间，
    keep_recent_tokens 为近似保留目标，不是可以拆开工具交互的硬上限。
    发生超限恢复时不自动降低 keep_recent_tokens。
    """

    enabled: bool
    reserve_tokens: int
    keep_recent_tokens: int


class ContextBudget(TypedDict):
    """调用方提供的模型窗口与实际请求中消息之外的输入开销。

    extra_input_tokens 包括未编码在 messages 内的工具定义等内容，避免漏算
    或重复计数。模型选择、窗口元数据和生成参数归调用方所有。
    """

    context_window: int
    extra_input_tokens: int


class CompactionPreparation(TypedDict):
    """一次压缩的输入快照，切分时保留工具调用与对应结果的完整性。

    messages_to_summarize 是较早历史；turn_prefix_messages 是被切开的
    当前轮前缀，包含该轮用户请求；retained_messages 是保留原文的尾部。
    previous_summary 单独传给摘要更新流程，不重复混进待总结消息。
    三组消息均保留原视图内容；重放修复产生的占位仅用于请求和预算，不能
    混进 retained_messages 并作为真实工具结果提交。
    retained_entry_ids 按同一原历史切点截取，与 retained_messages 逐项对应。
    """

    previous_summary: str | None
    messages_to_summarize: list[Message]
    turn_prefix_messages: list[Message]
    retained_messages: list[Message]
    retained_entry_ids: list[str]
    is_split_turn: bool
    tokens_before: int


class SummaryResult(TypedDict):
    """单次摘要生成结果；失败或取消不能作为有效摘要提交。

    usage 为客户端返回的实际用量，缺失值保留为 None。
    error 不包含凭据；不完整、空白或带工具调用的摘要不可视为成功。
    """

    status: Literal["success", "error", "cancelled"]
    summary: str | None
    usage: TokenUsage | None
    error: str | None


class CompactionResult(TypedDict):
    """待会话层提交的压缩结果，不在 context 内改写原始记录。

    success 时 history 为新摘要和保留消息；其他状态下为 None，调用方
    保留原有效历史。usage 收集本次摘要调用的实际用量，不推算缺失计费。
    unchanged 表示没有可压缩内容，不能当作成功恢复后重试的依据。
    成功 history 的 entry_ids 必须保留原消息身份，形成压缩前有效历史的尾部；
    不保留任何消息时 messages 和 entry_ids 同为空。session 按 ID 校验并保存
    引用，不用内容匹配定位边界，不把 history.messages 复制为新的原始消息。
    """

    status: Literal["success", "unchanged", "error", "cancelled"]
    history: HistoryContext | None
    tokens_before: int
    estimated_tokens_after: int | None
    usage: list[TokenUsage]
    error: str | None


# 摘要调用边界：接收准备好的消息、输出 token 上限和当前运行的取消信号。
# 调用方负责选择模型、生成唯一 call_id，并将信号传给 Client.complete(cancel_event=...)。
# context/适配器只观察信号，不清除或替换；信号取消返回 cancelled，任务取消继续传播。
# 此回调仅发送一次摘要请求，不进入普通 agent 循环或执行工具。
SummaryRequest = Callable[[list[Message], int, Event], Awaitable[ClientResult]]
