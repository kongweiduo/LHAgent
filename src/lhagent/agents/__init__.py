"""说明 coding agent 包的职责与边界。

agents 将 harness 的通用能力组装成面向具体任务的 agent。首期提供单一
CodingAgent：加载所需配置，选择模型与工具，默认创建会话，并组合 client、
context、loop 和 tools。通用循环、会话存储、工具实现和协议通信仍归各自模块。
"""

from .coding import CodingAgent, CodingAgentOptions, create_coding_agent

__all__ = ["CodingAgent", "CodingAgentOptions", "create_coding_agent"]
