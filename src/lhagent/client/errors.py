"""集中分类通信及协议错误，不决定重试或历史压缩策略。

参考 pi 的 utils/overflow.ts 将超限识别集中在一处；本项目面向统一兼容接口，
优先使用结构化错误码，不移植多 provider 的全部启发式规则。
"""

import asyncio
import re
from collections.abc import Mapping

import httpx
import openai

from lhagent.diagnostics import diagnostic

from .types import ClientErrorKind

# 固定兼容协议错误码与保守文本识别规则，不决定重试策略。
_CODES: dict[str, ClientErrorKind] = {
    "context_length_exceeded": "context_overflow",
    "context_window_exceeded": "context_overflow",
    "rate_limit_exceeded": "rate_limit",
    "rate_limit_error": "rate_limit",
    "insufficient_quota": "rate_limit",
    "invalid_api_key": "authentication",
    "authentication_error": "authentication",
    "invalid_request_error": "invalid_request",
    "invalid_request": "invalid_request",
    "server_error": "other",
    "service_unavailable": "other",
}
_OVERFLOW = re.compile(
    r"\b(?:context (?:length|window|limit) (?:is )?(?:exceeded|too (?:long|large))|"
    r"maximum context length|too many tokens in (?:the )?(?:context|prompt)|"
    r"prompt (?:is )?too long|input (?:is )?too long for (?:the )?context)\b",
    re.IGNORECASE,
)
_RATE_LIMIT = re.compile(r"\b(?:rate limit|too many requests|quota exceeded)\b", re.I)
_AUTH = re.compile(r"\b(?:unauthorized|invalid api key|authentication failed|forbidden)\b", re.I)
_SERVICE = re.compile(
    r"\b(?:service unavailable|internal server error|bad gateway|gateway timeout)\b", re.I
)


class ProtocolError(ValueError):
    """兼容协议字段格式错误；消息只描述字段约束，不包含原始响应。"""


def error_detail(error: Exception, api_key: str) -> str:
    """提取有长度限制的脱敏服务诊断，不输出完整请求或响应头。"""
    status = error_status(error)
    parts = [f"HTTP {status}"] if status is not None else []
    body = getattr(error, "body", None)
    if isinstance(body, Mapping):
        detail = body.get("error", body)
        if isinstance(detail, Mapping):
            for field in ("code", "param", "message"):
                value = detail.get(field)
                if isinstance(value, str) and value:
                    parts.append(f"{field}: {value}")
    if isinstance(error, ProtocolError):
        parts.append(str(error))
    text = diagnostic("; ".join(parts).replace(api_key, "[redacted]"))
    return " ".join("".join(c for c in text if c.isprintable() or c.isspace()).split())[:1000]


def error_status(error: Exception) -> int | None:
    """读取 HTTP 状态；单凭状态码不能认定上下文超限。"""
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status if type(status) is int else None


def error_code(error: Exception) -> str | None:
    """从 SDK 错误体或 HTTP JSON 响应提取 OpenAI 兼容错误码。"""
    body = getattr(error, "body", None)
    if not isinstance(body, Mapping):
        response = getattr(error, "response", None)
        if response is not None:
            try:
                body = response.json()
            except (ValueError, RuntimeError):
                pass
    if isinstance(body, Mapping):
        detail = body.get("error", body)
        if isinstance(detail, Mapping):
            code = detail.get("code")
            if isinstance(code, str):
                return code.lower().strip()
    code = getattr(error, "code", None)
    return code.lower().strip() if isinstance(code, str) else None


def classify_error(error: Exception) -> ClientErrorKind:
    """在底层错误信息尚未丢失时分类，供 ClientResult.error_kind 使用。

    识别明确的鉴权、限流和服务故障信息，不能因文本包含 token 就当作超限。
    上下文超限优先依据兼容接口的结构化错误码，例如 context_length_exceeded；
    没有明确分类的错误码时，才使用文档化的上下文超限文本匹配。HTTP 400/413
    本身不足以判定超限；明确的其他错误分类优先于模糊文本匹配。
    未识别错误归为 other；普通 length、未知 usage 或零输出不是超限证据。
    返回稳定分类，不返回底层异常、原始响应体或凭据；展示文本另行脱敏。
    Python CancelledError 不进入普通错误分类，须在清理后传播。
    """
    if isinstance(error, asyncio.CancelledError):
        raise error
    if isinstance(error, ProtocolError):
        return "protocol"
    code = error_code(error)
    if code in _CODES:
        return _CODES[code]

    status = error_status(error)
    if status == 429 or isinstance(error, openai.RateLimitError):
        return "rate_limit"
    if status in (401, 403) or isinstance(error, openai.AuthenticationError):
        return "authentication"
    if status is not None and status >= 500:
        return "other"

    # 不扫描原始响应体中的模糊超限措辞，避免把无关正文误判为可恢复错误。
    message = str(error)
    if _RATE_LIMIT.search(message):
        return "rate_limit"
    if _AUTH.search(message):
        return "authentication"
    if _SERVICE.search(message):
        return "other"
    if _OVERFLOW.search(message):
        return "context_overflow"
    if status in (408, 409, 425):
        return "other"
    if status is not None and 400 <= status < 500:
        return "invalid_request"
    if isinstance(error, (httpx.DecodingError, openai.APIResponseValidationError)):
        return "protocol"
    if isinstance(error, (httpx.TransportError, openai.APIConnectionError)):
        return "transport"
    return "other"
