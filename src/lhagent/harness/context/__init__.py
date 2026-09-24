"""定义上下文获取、组装、预算判断与历史压缩的职责边界。

接收外部维护的会话数据，生成模型可见消息及待提交的压缩结果。
不保存原始会话、不执行工具、不驱动主循环，也不实现客户端通信。
已实现提示词读取、历史重放修复、预算估算与压缩。
"""

from .assembly import assemble_context
from .budget import estimate_context_tokens, estimate_tool_tokens
from .compaction import compact
from .sources import load_prompts

__all__ = [
    "assemble_context",
    "compact",
    "estimate_context_tokens",
    "estimate_tool_tokens",
    "load_prompts",
]
