"""声明 agent 循环的职责边界。

协调上下文、模型通信、串行工具批次、输入队列、取消和超限恢复；
会话逐条提交，运行结束时收尾。
"""

from .loop import AgentLoop
from .queue import InputQueue
from .types import LoopConfig, LoopEvent, LoopResult

__all__ = ["AgentLoop", "InputQueue", "LoopConfig", "LoopEvent", "LoopResult"]
