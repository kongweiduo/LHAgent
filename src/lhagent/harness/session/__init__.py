"""定义会话创建、恢复、追加和 JSONL 持久化的职责边界。

会话层持有原始记录，并向 harness loop 提供有效历史视图。它负责记录的
持久化顺序、运行收尾和中断识别；不执行模型调用、工具调用或上下文压缩算法。
Session 的历史、运行状态、压缩引用和会话仓库已实现。
"""

from .jsonl import JsonlSessionStore
from .repository import SessionRepository
from .session import Session
from .types import (
    DisplayEntry,
    DisplayHistory,
    HistoryExclusion,
    JsonlRecord,
    RunStatus,
    SessionMetadata,
    SessionState,
    SessionStateInfo,
)

__all__ = [
    "DisplayEntry",
    "DisplayHistory",
    "SessionStateInfo",
    "HistoryExclusion",
    "JsonlRecord",
    "JsonlSessionStore",
    "RunStatus",
    "Session",
    "SessionMetadata",
    "SessionRepository",
    "SessionState",
]
