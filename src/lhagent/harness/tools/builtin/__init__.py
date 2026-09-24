"""提供七个基础工具的定义工厂，不在导入时自动注册或执行工具。"""

from ..registry import ToolRegistry
from ..types import ToolDefinition
from .bash import create_bash_tool
from .edit import create_edit_tool
from .find import create_find_tool
from .grep import create_grep_tool
from .ls import create_ls_tool
from .read import create_read_tool
from .write import create_write_tool

_BUILTIN_FACTORIES = (
    create_read_tool,
    create_write_tool,
    create_edit_tool,
    create_bash_tool,
    create_ls_tool,
    create_find_tool,
    create_grep_tool,
)


def create_builtin_tools() -> list[ToolDefinition]:
    """构造 read、write、edit、bash、ls、find、grep 的工具定义。"""
    registry = ToolRegistry()
    for factory in _BUILTIN_FACTORIES:
        registry.register(factory())
    return registry.list_tools()
