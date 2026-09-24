"""列出目录的直接子项。"""

import asyncio
import json
import os
from bisect import insort
from pathlib import Path

from ..types import ToolContext, ToolDefinition, ToolOutput

# 本工具的结果行数上限；调用方可通过输出预算进一步收紧。
_MAX_RESULT_LINES = 200


def create_ls_tool() -> ToolDefinition:
    """定义可选 path 的目录查看工具。"""
    return {
        "name": "ls",
        "description": "List immediate directory entries.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "minLength": 1}},
            "additionalProperties": False,
        },
        "handler": list_directory,
    }


async def list_directory(arguments: dict[str, object], context: ToolContext) -> ToolOutput:
    """按名称排序；扫描时只保留输出行数以内的最小名称。"""
    path = Path(context["cwd"]) / str(arguments.get("path", "."))
    max_lines = min(context["max_output_lines"], _MAX_RESULT_LINES)
    entries: list[tuple[str, str]] = []
    total = 0
    try:
        with os.scandir(path) as iterator:
            for entry in iterator:
                if context["cancel_event"].is_set():
                    raise asyncio.CancelledError
                kind = (
                    "directory"
                    if entry.is_dir(follow_symlinks=False)
                    else ("file" if entry.is_file(follow_symlinks=False) else "other")
                )
                total += 1
                if max_lines:
                    insort(entries, (entry.name, kind))
                    if len(entries) > max_lines:
                        entries.pop()
                await asyncio.sleep(0)
    except OSError as exc:
        return {
            "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
            "details": {"path": str(path)},
            "is_error": True,
            "truncated": False,
        }
    lines: list[str] = []
    remaining = context["max_output_bytes"]
    for name, kind in entries:
        display = name.encode("utf-8", "backslashreplace").decode("utf-8")
        line = f"{kind}\t{json.dumps(display, ensure_ascii=False)}\n"
        size = len(line.encode("utf-8"))
        if size > remaining:
            break
        lines.append(line)
        remaining -= size
    return {
        "content": [{"type": "text", "text": "".join(lines)}],
        "details": {"path": str(path), "total_entries": total},
        "is_error": False,
        "truncated": len(lines) < total,
    }
