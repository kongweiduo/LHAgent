"""加载并校验客户端通信配置。

只处理地址、凭据、超时和重试参数，不管理全项目配置、消息内容或模型选择。
配置按显式覆盖、LHAGENT_* 环境变量、当前工作目录 .env、默认值的优先级加载。
导入不读取配置，加载和校验不发起网络请求。
"""

import ipaddress
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from dotenv import load_dotenv

_ENV_FIELDS = {
    "base_url": "LHAGENT_BASE_URL",
    "api_key": "LHAGENT_API_KEY",
    "timeout_seconds": "LHAGENT_TIMEOUT_SECONDS",
    "max_retries": "LHAGENT_MAX_RETRIES",
    "max_retry_delay_seconds": "LHAGENT_MAX_RETRY_DELAY_SECONDS",
}
_DEFAULTS: dict[str, object] = {
    "timeout_seconds": 60.0,
    "max_retries": 2,
    "max_retry_delay_seconds": 30.0,
}


@dataclass(frozen=True, slots=True)
class ClientConfig:
    """通信参数契约；不包含会话状态和业务生成策略。

    timeout_seconds 表示一次网络尝试的超时配置，不代表整次调用的时间上限。
    max_retries 不包含首次尝试；max_retry_delay_seconds 限制单次重试等待。
    api_key 仅用于鉴权，排除在配置对象的 repr/str 之外，不得写入日志或结果。
    使用属性访问配置字段；repr=False 不阻止 asdict 或手动导出，禁止将整个
    配置对象转成字典用于日志或 agent 持久化配置。
    """

    base_url: str
    api_key: str = field(repr=False)
    timeout_seconds: float
    max_retries: int
    max_retry_delay_seconds: float


def load_config(overrides: Mapping[str, object] | None = None) -> ClientConfig:
    """加载通信配置并应用显式覆盖，返回通过校验的配置。

    仅读取客户端需要的配置，不发起网络请求、不加载会话、不选择模型。
    overrides > LHAGENT_* 环境变量 > 当前工作目录 .env > 默认值。
    按需加载 .env 到进程环境，不覆盖已有变量，不向父目录搜索或展开变量引用。
    仅环境变量中的数字字符串会转换；显式覆盖必须使用正确的 Python 类型。
    返回不可变 ClientConfig，原始映射及 api_key 值不得用于诊断输出。
    """
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise TypeError("overrides must be a mapping")
    if any(key not in _ENV_FIELDS for key in overrides):
        raise ValueError("overrides contains unsupported fields")

    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False, interpolate=False)

    values: dict[str, object] = {}
    for name, env_name in _ENV_FIELDS.items():
        if name in overrides:
            values[name] = overrides[name]
        elif env_name in os.environ:
            raw = os.environ[env_name]
            if name in _DEFAULTS:
                try:
                    values[name] = int(raw) if name == "max_retries" else float(raw)
                except (ValueError, OverflowError):
                    raise ValueError(f"{name} environment value must be numeric") from None
            else:
                values[name] = raw
        else:
            values[name] = _DEFAULTS.get(name)

    config = ClientConfig(
        base_url=cast(str, values["base_url"]),
        api_key=cast(str, values["api_key"]),
        timeout_seconds=cast(float, values["timeout_seconds"]),
        max_retries=cast(int, values["max_retries"]),
        max_retry_delay_seconds=cast(float, values["max_retry_delay_seconds"]),
    )
    validate_config(config)
    return config


def validate_config(config: ClientConfig) -> None:
    """校验通信参数的必填项、格式和取值范围，不探测网络或验证远端凭据。

    类型错误抛出 TypeError，缺失或越界值抛出 ValueError；错误不包含输入值。
    """
    if not isinstance(config, ClientConfig):
        raise TypeError("config must be a ClientConfig")
    for name in ("base_url", "api_key"):
        value = getattr(config, name)
        if value is None:
            raise ValueError(f"{name} is required")
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        if not value or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError(f"{name} must be nonempty and contain no whitespace or controls")

    if not _valid_base_url(config.base_url):
        raise ValueError(
            "base_url must be an absolute HTTP(S) URL with a valid host and port, without credentials, query or fragment"
        )

    for name in ("timeout_seconds", "max_retry_delay_seconds"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or value < 0 or (name == "timeout_seconds" and value == 0):
            raise ValueError(
                f"{name} must be finite and {'positive' if name == 'timeout_seconds' else 'nonnegative'}"
            )

    if isinstance(config.max_retries, bool) or not isinstance(config.max_retries, int):
        raise TypeError("max_retries must be an integer")
    if config.max_retries < 0:
        raise ValueError("max_retries must be nonnegative")


def _valid_base_url(value: str) -> bool:
    """只验证本地 URL 语法，不做 DNS 查询；解析异常不透出输入值。"""
    try:
        url = urlsplit(value)
        host = url.hostname
        if (
            url.scheme not in ("http", "https")
            or not host
            or url.username is not None
            or url.password is not None
            or "?" in value
            or "#" in value
            or "\\" in value
            or url.netloc.endswith(":")
            or (url.port is not None and not 1 <= url.port <= 65535)
        ):
            return False
        if ":" in host:
            ipaddress.IPv6Address(host)
        else:
            host = host.encode("idna").decode("ascii").rstrip(".")
            if len(host) > 253 or not all(
                re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?", label)
                for label in host.split(".")
            ):
                return False
            if re.fullmatch(r"[0-9.]+", host):
                ipaddress.IPv4Address(host)
        return True
    except (ValueError, UnicodeError):
        return False
