"""按行读取 UTF-8 文本，不加载整个文件。"""

import asyncio
from pathlib import Path

from ..types import ToolContext, ToolDefinition, ToolOutput

# read 的分页与单行内存预算，不替代调用方更小的输出限制。
_PAGE_LINES = 200
_MAX_LINE_BYTES = 65536


def create_read_tool() -> ToolDefinition:
    """定义 read 工具；offset 从 1 开始，limit 为最大完整行数。"""
    return {
        "name": "read",
        "description": "Read a UTF-8 text file by line number.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "offset": {"type": "integer", "minimum": 1},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "handler": read_file,
    }


async def read_file(arguments: dict[str, object], context: ToolContext) -> ToolOutput:
    """返回完整行和可继续读取的行号；资源在取消传播前关闭。"""
    path = Path(context["cwd"]) / str(arguments["path"])
    raw_offset = arguments.get("offset", 1)
    raw_limit = arguments.get("limit", _PAGE_LINES)
    if not isinstance(raw_offset, int) or isinstance(raw_offset, bool):
        raise ValueError("offset must be an integer")
    if not isinstance(raw_limit, int) or isinstance(raw_limit, bool):
        raise ValueError("limit must be an integer")
    offset = raw_offset
    limit = min(raw_limit, _PAGE_LINES)
    lines: list[str] = []
    line_number = 1
    remaining = context["max_output_bytes"]
    max_lines = context["max_output_lines"]
    truncated = False
    try:
        with path.open("rb") as stream:
            # 按块跳过长行，使远距离行偏移也只使用有界内存。
            while line_number < offset:
                if context["cancel_event"].is_set():
                    raise asyncio.CancelledError
                chunk = stream.readline(_MAX_LINE_BYTES)
                if not chunk:
                    break
                if chunk.endswith(b"\n") or (len(chunk) < _MAX_LINE_BYTES or not stream.peek(1)):
                    line_number += 1
                await asyncio.sleep(0)
            while line_number >= offset:
                if context["cancel_event"].is_set():
                    raise asyncio.CancelledError
                if len(lines) >= min(limit, max_lines) or remaining == 0:
                    truncated = bool(stream.peek(1))
                    break
                raw = stream.readline(min(remaining, _MAX_LINE_BYTES) + 1)
                if not raw:
                    break
                if len(raw) > _MAX_LINE_BYTES or (
                    remaining >= _MAX_LINE_BYTES and not raw.endswith(b"\n") and stream.peek(1)
                ):
                    if not lines:
                        raise ValueError(
                            f"Line exceeds read budget (max {_MAX_LINE_BYTES} bytes); cannot advance by line offset"
                        )
                    truncated = True
                    break
                if len(raw) > remaining:
                    if not lines:
                        raise ValueError(
                            "Line exceeds max_output_bytes; cannot advance by line offset"
                        )
                    truncated = True
                    break
                lines.append(raw.decode("utf-8"))
                remaining -= len(raw)
                line_number += 1
                await asyncio.sleep(0)
    except (OSError, UnicodeError, ValueError) as exc:
        return {
            "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
            "details": {"path": str(path), "next_offset": line_number},
            "is_error": True,
            "truncated": False,
        }
    return {
        "content": [{"type": "text", "text": "".join(lines)}],
        "details": {"path": str(path), "next_offset": line_number if truncated else None},
        "is_error": False,
        "truncated": truncated,
    }
