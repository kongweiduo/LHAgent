"""管理内存中的工具目录及按 agent 配置选择的工具集合。"""

from copy import deepcopy

from .types import ToolDefinition
from .validation import validate_schema


class ToolRegistry:
    """保存已登记工具，按名称为 agent 构造可用集合。"""

    def __init__(self) -> None:
        """创建空的内存工具目录；定义校验在 register 时执行。"""
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """校验定义后登记；重复名称或无效 schema 不会改变目录。"""
        if tool["name"] in self._tools:
            raise ValueError(f"Tool already registered: {tool['name']}")
        validate_schema(tool["parameters"])
        self._tools[tool["name"]] = tool

    def get(self, name: str) -> ToolDefinition:
        """按名称取得已登记定义。"""
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Unknown tool: {name}") from None

    def list_tools(self) -> list[ToolDefinition]:
        """按登记顺序列出定义。"""
        return list(self._tools.values())

    def select(self, names: list[str]) -> list[ToolDefinition]:
        """按配置顺序选择可用工具，重复名称去重。"""
        return [self.get(name) for name in dict.fromkeys(names)]


def describe_tools(tools: list[ToolDefinition]) -> list[dict[str, object]]:
    """导出模型可见描述，不暴露执行函数。"""
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": deepcopy(tool["parameters"]),
        }
        for tool in tools
    ]
