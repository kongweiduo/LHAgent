"""工具登记、调用校验、执行和内置工具。

本包负责工具注册、参数校验和执行，不持久化配置或驱动模型循环。
"""

from .builtin import create_builtin_tools
from .execution import execute_tool_call
from .registry import ToolRegistry, describe_tools

__all__ = ["ToolRegistry", "create_builtin_tools", "describe_tools", "execute_tool_call"]
