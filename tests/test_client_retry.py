"""错误分类、有限重试、指数退避与 Retry-After 提示的离线验收。"""

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import httpx2
import openai
import pytest

from lhagent.client.config import ClientConfig, load_config
from lhagent.client.errors import classify_error
from lhagent.client.retry import get_retry_delay, should_retry

CONFIG = ClientConfig("https://example.com/v1", "test-key", 60, 2, 30)
REQUEST = httpx.Request("POST", "https://example.com/v1/chat/completions")


def status_error(status, body=None, message="failure", headers=None):
    """构造带状态、响应头和 JSON 错误体的 HTTP 异常，不发送请求。"""
    response = httpx.Response(status, request=REQUEST, json=body or {}, headers=headers)
    error = httpx.HTTPStatusError(message, request=REQUEST, response=response)
    return error


@pytest.mark.parametrize(
    "error,kind",
    [
        (status_error(400), "invalid_request"),
        (status_error(413), "invalid_request"),
        (status_error(400, {"error": {"code": "context_length_exceeded"}}), "context_overflow"),
        (
            status_error(
                400, {"error": {"code": "invalid_request_error"}}, "maximum context length"
            ),
            "invalid_request",
        ),
        (status_error(429, message="maximum context length"), "rate_limit"),
        (status_error(401, message="maximum context length"), "authentication"),
        (status_error(503, message="maximum context length"), "other"),
        (Exception("rate limit: maximum context length"), "rate_limit"),
        (Exception("service unavailable: prompt too long"), "other"),
        (Exception("prompt is too long"), "context_overflow"),
        (Exception("finish_reason=length, usage missing, zero output tokens"), "other"),
        (Exception("token limit"), "other"),
        (httpx.ConnectError("connection failed", request=REQUEST), "transport"),
        (httpx.DecodingError("bad encoding", request=REQUEST), "protocol"),
    ],
)
def test_classification(error, kind):
    """结构化错误和有限文本规则映射到预期分类。"""
    assert classify_error(error) == kind


@pytest.mark.parametrize(
    "status,expected",
    [
        (400, False),
        (401, False),
        (403, False),
        (413, False),
        (408, True),
        (409, True),
        (425, True),
        (429, True),
        (500, True),
        (503, True),
        (504, True),
        (501, True),
    ],
)
def test_status_retry(status, expected):
    """HTTP 状态只对允许的失败启用重试。"""
    assert should_retry(status_error(status), 1, CONFIG) is expected


def test_retry_attempts_and_transport():
    """首次请求计入尝试数，网络错误遵守重试上限。"""
    error = httpx.ConnectTimeout("timeout", request=REQUEST)
    assert not should_retry(error, 0, CONFIG)
    assert should_retry(error, 1, CONFIG)
    assert should_retry(error, 2, CONFIG)
    assert not should_retry(error, 3, CONFIG)
    assert not should_retry(
        error, 1, load_config({"base_url": CONFIG.base_url, "api_key": "key", "max_retries": 0})
    )
    assert not should_retry(Exception("service unavailable"), 1, CONFIG)
    assert not should_retry(
        status_error(400, {"error": {"code": "context_length_exceeded"}}), 1, CONFIG
    )
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError):
            should_retry(error, value, CONFIG)


def test_backoff_and_server_hint():
    """本地退避受上限约束，过长服务端提示拒绝重试。"""
    error = status_error(429)
    assert [get_retry_delay(error, i, CONFIG) for i in range(7)] == [1, 2, 4, 8, 16, 30, 30]
    assert get_retry_delay(error, 100000, CONFIG) == 30
    for value, expected in [
        ("0", 0),
        ("12.5", 12.5),
        ("31", None),
        ("broken", 1),
        ("-3", 1),
        ("nan", 1),
    ]:
        assert (
            get_retry_delay(status_error(429, headers={"Retry-After": value}), 0, CONFIG)
            == expected
        )
    zero = load_config(
        {"base_url": CONFIG.base_url, "api_key": "key", "max_retry_delay_seconds": 0}
    )
    assert get_retry_delay(error, 0, zero) == 0
    assert get_retry_delay(status_error(429, headers={"Retry-After": "1"}), 0, zero) is None
    for value in (-1, False):
        with pytest.raises(ValueError):
            get_retry_delay(error, value, CONFIG)


def test_openai_sdk_error_fields():
    """SDK 异常字段可供统一错误分类读取。"""
    request = httpx2.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx2.Response(400, request=request, headers={"Retry-After": "3"})
    error = openai.BadRequestError(
        "bad request", response=response, body={"error": {"code": "context_length_exceeded"}}
    )
    assert classify_error(error) == "context_overflow"
    assert not should_retry(error, 1, CONFIG)
    response = httpx2.Response(429, request=request, headers={"Retry-After": "3"})
    error = openai.RateLimitError("rate limit", response=response, body=None)
    assert classify_error(error) == "rate_limit"
    assert should_retry(error, 1, CONFIG)
    assert get_retry_delay(error, 0, CONFIG) == 3
    connection = openai.APIConnectionError(request=request)
    assert classify_error(connection) == "transport"
    assert should_retry(connection, 1, CONFIG)


def test_retry_after_http_date():
    """HTTP 日期形式的 Retry-After 转为有限等待时间。"""
    future = format_datetime(datetime.now(UTC) + timedelta(seconds=60), usegmt=True)
    past = format_datetime(datetime.now(UTC) - timedelta(seconds=60), usegmt=True)
    assert get_retry_delay(status_error(503, headers={"Retry-After": future}), 0, CONFIG) is None
    assert get_retry_delay(status_error(503, headers={"Retry-After": past}), 0, CONFIG) == 0
