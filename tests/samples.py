"""构造离线契约样本和轻量异步替身，不访问网络或真实会话文件。"""

from asyncio import Event

from lhagent.client.types import (
    CallStats,
    ClientResult,
    Message,
    TokenUsage,
)
from lhagent.harness.session.types import UserRecord


def usage() -> TokenUsage:
    """构造所有计数未知的用量样本。"""
    return {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
    }


def stats() -> CallStats:
    """构造未发起请求的统计样本。"""
    return {
        "usage": usage(),
        "elapsed_seconds": 0.0,
        "first_content_seconds": None,
        "attempts": 0,
    }


def mixed_result(*, complete_tool: bool = True) -> ClientResult:
    """构造同时包含文本、推理和工具调用的响应，可切换调用完整性。"""
    return {
        "call_id": "model-call-1",
        "content": [
            {"type": "text", "text": "先检查文件。"},
            {"type": "reasoning", "text": "需要读取配置。"},
            {
                "type": "tool_call",
                "call_id": "tool-call-1",
                "name": "read",
                "arguments": {"path": "pyproject.toml"},
                "complete": complete_tool,
            },
        ],
        "finish_reason": "tool_call",
        "error": None,
        "error_kind": None,
        "stats": stats(),
    }


def user_message(text: str = "检查项目") -> Message:
    """构造单文本用户消息，每次返回独立字典。"""
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}],
    }


def user_record() -> UserRecord:
    """构造身份与运行 ID 分离的用户日志条目。"""
    return {
        "type": "user",
        "id": "entry-1",
        "run_id": "run-1",
        "timestamp": 0.0,
        "message": user_message(),
    }


class FakeStream:
    """无需网络的异步事件流替身，供后续 client/loop 测试复用。"""

    def __init__(self, events: list[dict[str, object]]) -> None:
        self.events = events
        self.closed = False

    def __aiter__(self) -> "FakeStream":
        return self

    async def __anext__(self) -> dict[str, object]:
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)

    async def aclose(self) -> None:
        self.closed = True


class FakeSession:
    """只记录调用顺序，不写文件、不依赖工作区。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.cancel_event = Event()

    async def start_run(self, run_id: str) -> None:
        self.calls.append(("start_run", run_id))

    async def append_user(self, message: Message) -> str:
        self.calls.append(("append_user", message))
        return "entry-user"

    async def append_response(self, response: ClientResult) -> str:
        self.calls.append(("append_response", response))
        return "entry-assistant"
