"""维护单次客户端调用的 token 用量、耗时及尝试次数。

只记录本次可观测的数据，不估算缺失 token，不保存会话，不做跨调用汇总。
时间间隔使用单调时钟。
"""

from time import monotonic

from .types import CallStats, TokenUsage


class CallMetrics:
    """一次逻辑调用的统计记录器，覆盖其所有尝试和重试等待。"""

    def __init__(self) -> None:
        """建立调用起点及未知用量状态，不启动请求或全局统计任务。"""
        self._started_at = monotonic()
        self._finished_at: float | None = None
        self._first_content_at: float | None = None
        self._attempts = 0
        self._usage: TokenUsage = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
        }

    def record_attempt(self) -> None:
        """在实际开始一次网络尝试时记录次数，首次请求也计入，不触发请求。"""
        if self._finished_at is None:
            self._attempts += 1

    def record_first_content(self) -> None:
        """首次收到文本、推理或工具调用内容时记录耗时，重复调用不覆盖。

        仅收到响应头或不含内容的分片不能调用本方法。
        """
        if self._finished_at is None and self._first_content_at is None:
            self._first_content_at = monotonic()

    def update_usage(self, usage: TokenUsage) -> None:
        """逐字段保留最近一次非 None 的服务端值，不累加重复的累计 usage。

        未知项保持未知，不根据字符数推算，不补算未收到用量的失败尝试。
        """
        if self._finished_at is not None:
            return
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
        ):
            value = usage[key]
            if value is not None:
                self._usage[key] = value

    def finish(self) -> CallStats:
        """在成功、失败或主动取消时形成统计快照，重复调用保持结束时间不变。

        总耗时包含所有尝试和等待；没有内容时首个内容耗时为 None。
        不打印、不持久化、不汇总到其他调用。
        """
        if self._finished_at is None:
            self._finished_at = monotonic()
        return {
            "usage": self._usage.copy(),
            "elapsed_seconds": self._finished_at - self._started_at,
            "first_content_seconds": (
                None
                if self._first_content_at is None
                else self._first_content_at - self._started_at
            ),
            "attempts": self._attempts,
        }
