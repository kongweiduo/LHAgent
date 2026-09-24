"""声明工具定义、运行环境、调用与结果的数据契约。

仅声明接口；schema 校验由 validation 按 Draft 2020-12 执行，消息协议映射归 client。
"""

from asyncio import Event
from collections.abc import Awaitable, Callable
from typing import Literal, TypedDict

from lhagent.client.types import ContentBlock


class ToolContext(TypedDict):
    """外部提供的单次执行环境，不包含模型调用或会话写入能力。

    cwd 是相对路径解析基准；timeout_seconds 为本次执行时限。
    cancel_event 由 loop 设置，执行层取消 handler 并等待其停止和清理。
    timeout_seconds 必须为有限非负数，零表示不启动；清理不受执行时限截断。
    输出限制按文本行数和字节数约束，截断必须在结果中明确标识。
    """

    cwd: str
    timeout_seconds: float
    cancel_event: Event
    max_output_lines: int
    max_output_bytes: int


class ToolOutput(TypedDict):
    """工具实现返回的统一输出，可显式报告未抛异常的执行失败。

    content 保存结构化内容块；details 保存退出码等执行信息。
    is_error 只描述工具执行，不判断是否满足用户要求。
    truncated 表示输出经过截断，不将截断本身视为执行失败。
    """

    content: list[ContentBlock]
    details: dict[str, object]
    is_error: bool
    truncated: bool


# 工具处理函数接收已校验参数和环境，只执行一次，不调用 LLM。
# 必须让出事件循环，响应 CancelledError，并在返回或抛出前停止所有自有资源。
# 可在取消清理后返回部分 ToolOutput，或重新抛出 CancelledError；停止无法确认
# 必须抛出 ToolStopError（可带部分输出）。不可遗弃后台任务、线程或进程。
# 执行层只发送一次取消，屏蔽后续调用者取消以保护异步清理，不替工具回滚副作用。
ToolHandler = Callable[[dict[str, object], ToolContext], Awaitable[ToolOutput]]


class ToolDefinition(TypedDict):
    """已登记工具的名称、描述、JSON Schema 和执行入口。

    parameters 描述参数对象，遵循 validation.validate_schema 的方言与支持范围；
    handler 仅在本地使用，不发送给模型，也不写入 agent 的持久化配置。
    """

    name: str
    description: str
    parameters: dict[str, object]
    handler: ToolHandler


class ToolCall(TypedDict):
    """client 解析完成后交给执行层的结构化调用。

    call_id 用于关联模型调用与结果；arguments 允许携带非法类型，
    由校验层报告错误。被截断或流中断的调用不得交给工具执行。
    """

    call_id: str
    name: str
    arguments: object


class ToolStopError(Exception):
    """执行停止未获确认，必须向 loop 传播，不能转成普通 ToolResult。

    call 标识无法确认停止的调用；output 只保存可获得的部分输出，不代表结束。
    loop 停止批次并以 error 收尾，不伪造该调用已结束的工具结果；原 assistant
    调用仍留在会话中，重放时由 context 补充结果未知的临时占位。
    """

    call: ToolCall
    output: ToolOutput | None

    def __init__(self, call: ToolCall, error: str, output: ToolOutput | None = None) -> None:
        """保存调用、错误和可获得的部分输出。"""
        super().__init__(error)
        self.call = call
        self.output = output


class ToolResult(TypedDict):
    """交给 loop 的单次调用结果，不自行写入会话或触发下一轮模型请求。

    validation_error 表示尚未执行；execution_error 表示执行失败。
    timeout 或 cancelled 时不能假设外部副作用未发生；已启动的执行必须确认
    停止后才可返回这些结果，无法确认时抛出 ToolStopError。
    error 为可反馈模型的错误说明，成功时为 None。
    """

    call_id: str
    name: str
    status: Literal["success", "validation_error", "execution_error", "timeout", "cancelled"]
    output: ToolOutput | None
    error: str | None


class PreparedToolCall(TypedDict):
    """工具已解析且参数通过 schema 校验的调用，仅供执行层使用。"""

    call: ToolCall
    tool: ToolDefinition
    arguments: dict[str, object]
