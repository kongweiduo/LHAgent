"""终端诊断脱敏；不输出整个配置或异常 repr，凭据在调用时读取。"""

import os
import re

API_KEY_ENV_VAR = "LHAGENT_API_KEY"
REDACTION_MARKER = "[redacted]"
CREDENTIAL_PATTERN = (
    r"(?i)(?<![\w-])([\"']?(?:api_key|api-key|authorization|access_token|"
    r"auth_token|token|password|secret)[\"']?\s*[:=]\s*)"
    r"(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|"
    r"(?:Basic|Bearer)\s+[^\s,;'\"}]+|[^\s,;}]+)"
)
BEARER_PATTERN = r"(?i)(bearer\s+)[^\s,;'\"}]+"


def diagnostic(value: object) -> str:
    """保留可用错误信息并遮盖凭据；每次调用读取当前环境中的密钥。"""
    text = str(value)
    key = os.environ.get(API_KEY_ENV_VAR)
    if key:
        text = text.replace(key, REDACTION_MARKER)
    # 先匹配完整字段值，避免转义引号或鉴权方案后的空格留下凭据尾部。
    text = re.sub(
        CREDENTIAL_PATTERN,
        lambda match: match[1] + REDACTION_MARKER,
        text,
    )
    return re.sub(BEARER_PATTERN, lambda match: match[1] + REDACTION_MARKER, text)


__all__ = ["diagnostic"]
