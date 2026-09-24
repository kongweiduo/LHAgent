"""按相对于搜索根目录的 glob 路径模式查找普通文件。"""

from .._search import run_search
from ..types import ToolContext, ToolDefinition, ToolOutput


def create_find_tool() -> ToolDefinition:
    """定义必填 pattern 和可选搜索目录 path。"""
    return {
        "name": "find",
        "description": "Find files by a relative glob; * matches one segment, ** spans directories. Includes hidden files and ignores no files.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "minLength": 1},
                "path": {"type": "string", "minLength": 1},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
        "handler": find_files,
    }


async def find_files(arguments: dict[str, object], context: ToolContext) -> ToolOutput:
    """返回 JSON 字符串路径行；无匹配为空正文，搜索失败显式标记。"""
    return await run_search("find", arguments, context)
