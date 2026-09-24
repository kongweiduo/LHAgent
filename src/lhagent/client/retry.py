"""判断取得响应流之前的失败是否允许重试及等待多久。

调用时机由 Client 决定；本模块不发送请求、不执行等待。"""

import math
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx
import openai

from .config import ClientConfig
from .errors import classify_error, error_status

# 本地退避指数的数值上限，实际等待仍受 ClientConfig 约束。
_MAX_BACKOFF_EXPONENT = 60


def should_retry(error: Exception, attempts: int, config: ClientConfig) -> bool:
    """判断是否允许再次尝试；attempts 包含首次请求，协议及鉴权错误不重试。"""
    if type(attempts) is not int or attempts < 0:
        raise ValueError("attempts must be a nonnegative integer")
    if attempts == 0 or attempts > config.max_retries:
        return False
    if classify_error(error) in (
        "authentication",
        "invalid_request",
        "context_overflow",
        "protocol",
    ):
        return False
    status = error_status(error)
    if status is not None:
        return status in (408, 409, 425, 429) or 500 <= status <= 599
    return isinstance(error, (httpx.TransportError, openai.APIConnectionError))


def get_retry_delay(error: Exception, retry_index: int, config: ClientConfig) -> float | None:
    """返回重试秒数；索引从零开始，服务端等待提示过长时返回 None 拒绝重试。"""
    if type(retry_index) is not int or retry_index < 0:
        raise ValueError("retry_index must be a nonnegative integer")

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    hint = headers.get("retry-after") if headers is not None else None
    if hint is not None:
        try:
            seconds = float(hint)
        except (TypeError, ValueError):
            try:
                date = parsedate_to_datetime(hint)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=UTC)
                seconds = max(0.0, (date - datetime.now(UTC)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                seconds = float("nan")
        if math.isfinite(seconds) and seconds >= 0:
            return seconds if seconds <= config.max_retry_delay_seconds else None

    return min(2.0 ** min(retry_index, _MAX_BACKOFF_EXPONENT), config.max_retry_delay_seconds)
