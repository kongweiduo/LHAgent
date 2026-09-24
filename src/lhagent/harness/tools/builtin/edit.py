"""在 UTF-8 文件中执行一次唯一文本替换。"""

import asyncio
from pathlib import Path

from ..types import ToolContext, ToolDefinition, ToolOutput


def create_edit_tool() -> ToolDefinition:
    """定义 edit 工具及参数 schema：path、old_text 和 new_text。"""
    return {
        "name": "edit",
        "description": "Replace one unique exact text occurrence in a UTF-8 file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "old_text": {"type": "string", "minLength": 1},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
        "handler": edit_file,
    }


async def edit_file(arguments: dict[str, object], context: ToolContext) -> ToolOutput:
    """精确匹配旧文本后执行局部替换，返回修改摘要。"""
    path = Path(context["cwd"]) / str(arguments["path"])
    try:
        await asyncio.sleep(0)
        if context["cancel_event"].is_set():
            raise asyncio.CancelledError
        text = path.read_bytes().decode("utf-8")
        old = str(arguments["old_text"])
        first = text.find(old)
        if first < 0:
            raise ValueError("old_text must match exactly once (found 0)")
        if text.find(old, first + 1) >= 0:
            raise ValueError("old_text must match exactly once (found multiple)")
        data = text.replace(old, str(arguments["new_text"]), 1).encode("utf-8")
        # 最终取消检查与同步写入之间不能让出控制权，避免已取消后才开始写入。
        if context["cancel_event"].is_set():
            raise asyncio.CancelledError
        path.write_bytes(data)
    except (OSError, UnicodeError, ValueError) as exc:
        return {
            "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
            "details": {"path": str(path)},
            "is_error": True,
            "truncated": False,
        }
    return {
        "content": [{"type": "text", "text": f"Replaced one occurrence in {path}"}],
        "details": {"path": str(path), "replacements": 1},
        "is_error": False,
        "truncated": False,
    }
