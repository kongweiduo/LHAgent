"""定义与中转站通信的数据契约。

仅声明请求、响应和统计结构，不组装业务消息、不发送请求，也不保存会话。
协议字段转换由 transport 模块负责。
"""

from typing import Literal, TypedDict

MessageRole = Literal["system", "user", "assistant", "tool"]
ContentBlockType = Literal["text", "reasoning", "tool_call", "tool_result"]
FinishReason = Literal["stop", "length", "tool_call", "error", "cancelled"]


class TextContent(TypedDict):
    """文本内容块。"""

    type: Literal["text"]
    text: str


class ReasoningContent(TypedDict):
    """推理内容块，与面向用户的文本保持区分。"""

    type: Literal["reasoning"]
    text: str


class ToolCallContentBase(TypedDict):
    """工具调用块的必需字段；完整性标志决定参数是否可执行。"""

    type: Literal["tool_call"]
    call_id: str
    name: str
    complete: bool


class ToolCallContent(ToolCallContentBase, total=False):
    """工具调用内容块；不完整参数不得交给工具执行。"""

    arguments: dict[str, object]
    arguments_json: str


class ToolResultContent(TypedDict):
    """工具结果消息中的结构化内容块。"""

    type: Literal["tool_result"]
    content: object
    is_error: bool


ContentBlock = TextContent | ReasoningContent | ToolCallContent | ToolResultContent


class MessageBase(TypedDict):
    """消息的必需角色和内容字段；关联身份由具体消息补充。"""

    role: MessageRole
    content: list[ContentBlock]


class Message(MessageBase, total=False):
    """跨 client、context、session 共享的结构化消息。"""

    tool_call_id: str
    name: str
    finish_reason: Literal["stop", "length", "tool_call", "error", "cancelled"]
    call_id: str


ClientErrorKind = Literal[
    "context_overflow",
    "rate_limit",
    "authentication",
    "invalid_request",
    "transport",
    "protocol",
    "other",
]


class ClientRequest(TypedDict):
    """外部准备好的单次调用输入。

    call_id 在当前客户端进行中的调用之间唯一；model、messages 和 parameters
    均由上层决定。messages 包含已准备且完成重放修复的上下文；parameters
    可携带工具定义和生成参数，不用于覆盖凭据、目标地址或内部流式模式。
    tools 使用内部工具描述（name、description、parameters），由 Transport 包装。
    内部统一使用 parameters.max_output_tokens 表达单次输出上限，由 Transport
    映射到所选兼容端点的字段；不同时接受其他输出上限别名。
    """

    call_id: str
    model: str
    messages: list[Message]
    parameters: dict[str, object]


class TokenUsage(TypedDict):
    """中转站实际返回的 token 用量；缺失项为 None，不进行本地估算。

    输入、输出与缓存字段按协议语义映射，不能未经确认直接相加。
    该结构不承诺包含未返回 usage 的失败尝试所产生的计费用量。
    """

    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None


class CallStats(TypedDict):
    """单次调用的最终统计，不负责跨调用汇总。

    总耗时和首个内容耗时均以秒计，从调用开始计时，包含此前的重试等待。
    未收到内容时首个内容耗时为 None；attempts 统计实际发起的网络尝试，
    包含首次请求，发送前主动信号取消时为零。
    """

    usage: TokenUsage
    elapsed_seconds: float
    first_content_seconds: float | None
    attempts: int


class ClientResult(TypedDict):
    """本次调用的最终或失败结果，不自动写入会话历史。

    content 保存解析后的内容块，包括可获得的文本、推理或工具调用信息。
    失败和主动信号取消时保留部分内容，但不保证工具参数完整、可执行。任一结束原因
    均可作为终结记录交给 session，客户端不裁剪历史或修复缺失工具结果。
    error 为可交付上层的错误说明，必须排除凭据等敏感信息。
    error_kind 由 client.errors 统一分类，仅 finish_reason=error 时非空；
    未识别错误使用 other。正常、length 和 cancelled 的 error_kind 为 None。
    loop 只依据 context_overflow 分类触发超限恢复，不解析展示用错误文本。
    """

    call_id: str
    content: list[ContentBlock]
    finish_reason: Literal["stop", "length", "tool_call", "error", "cancelled"]
    error: str | None
    error_kind: ClientErrorKind | None
    stats: CallStats


def client_result_to_message(result: ClientResult) -> Message:
    """将终结 client 结果投影为 assistant 历史消息。

    保留残缺内容和结束原因，不把错误文本伪装成工具结果；call_id 仍与
    session 的原始 entry id 分离。
    """

    return {
        "role": "assistant",
        "content": list(result["content"]),
        "call_id": result["call_id"],
        "finish_reason": result["finish_reason"],
    }


class StreamEvent(TypedDict):
    """统一的增量或终结事件，不承载界面渲染和工具执行行为。

    delta 事件通过 block_index 关联内容块，data 表示该块的协议解析增量，
    result 为 None。done、error、cancelled 事件的 result 为本次最终结果，
    block_index 为 None。正常消费至结束时仅交付一个终结事件，之后不再产生
    增量；消费者提前关闭或外部 Python 任务取消时不保证交付终结事件。
    """

    call_id: str
    type: Literal["delta", "done", "error", "cancelled"]
    block_index: int | None
    data: dict[str, object]
    result: ClientResult | None
