"""LHAgent 与统一 OpenAI 兼容中转站之间的客户端通信边界。

接收外部准备好的请求，负责通信配置、流式接收、完整结果交付、重试、取消、关闭和单次统计；不组织消息、不执行工具、不保存会话、不驱动 agent 循环。
"""

from .client import Client
from .config import ClientConfig, load_config, validate_config
from .types import ClientRequest, ClientResult, Message

__all__ = [
    "Client",
    "ClientConfig",
    "ClientRequest",
    "ClientResult",
    "Message",
    "load_config",
    "validate_config",
]
