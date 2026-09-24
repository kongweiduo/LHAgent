"""创建或完整覆盖 UTF-8 文本文件。"""

import asyncio
from pathlib import Path

from ..types import ToolContext, ToolDefinition, ToolOutput


def create_write_tool() -> ToolDefinition:
    """定义 write 工具及参数 schema：path 和 content。"""
    return {
        "name": "write",
        "description": "Create or overwrite a UTF-8 text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "handler": write_file,
    }


async def write_file(arguments: dict[str, object], context: ToolContext) -> ToolOutput:
    """写入完整文本，文件不存在时创建，并按需创建父目录。"""
    path = Path(context["cwd"]) / str(arguments["path"])
    try:
        data = str(arguments["content"]).encode("utf-8")
        await asyncio.sleep(0)
        if context["cancel_event"].is_set():
            raise asyncio.CancelledError
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except (OSError, UnicodeError, ValueError) as exc:
        return {
            "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
            "details": {"path": str(path)},
            "is_error": True,
            "truncated": False,
        }
    return {
        "content": [{"type": "text", "text": f"Wrote {len(data)} bytes to {path}"}],
        "details": {"path": str(path), "bytes_written": len(data)},
        "is_error": False,
        "truncated": False,
    }
