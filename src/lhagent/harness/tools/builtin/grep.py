"""使用 Python 正则或字面文本逐行搜索 UTF-8 文件。"""

from .._search import run_search
from ..types import ToolContext, ToolDefinition, ToolOutput


def create_grep_tool() -> ToolDefinition:
    """定义必填 pattern、可选文件/目录 path 和布尔 literal。"""
    return {
        "name": "grep",
        "description": "Search UTF-8 files by Python regex or literal text. Returns JSON lines with path, line number and text; includes hidden files, skips binary samples.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "minLength": 1},
                "literal": {"type": "boolean"},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
        "handler": search_content,
    }


async def search_content(arguments: dict[str, object], context: ToolContext) -> ToolOutput:
    """每个匹配行返回一次；无匹配成功，正则、解码及文件错误报告失败。"""
    return await run_search("grep", arguments, context)
