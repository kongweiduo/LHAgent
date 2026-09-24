"""按需读取 TOML 配置，合并覆盖并校验声明式设置。"""

import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from .types import CodingAgentConfig, HarnessConfig

# 声明式默认值；路径解析和容器创建必须保留在加载时。
_DEFAULT_CONFIGS_DIR = "configs"
_DEFAULT_AGENT_NAME = "coding"
_DEFAULT_TOOLS = ("read", "ls", "find", "grep", "write", "edit", "bash")
_DEFAULT_COMPACTION_ENABLED = True
_DEFAULT_KEEP_RECENT_TOKENS = 20000


_FILE = "lhagent.toml"
_HARNESS_FIELDS = {"configs_dir"}
_AGENT_FIELDS = set(CodingAgentConfig.__annotations__)
_COMPACTION_FIELDS = {"enabled", "reserve_tokens", "keep_recent_tokens"}
_FORBIDDEN_PARAMETERS = {
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "messages",
    "tools",
    "model",
    "stream",
    "api_key",
    "api-key",
    "authorization",
    "access_token",
    "auth_token",
    "token",
    "password",
    "secret",
    "base_url",
    "tool_choice",
    "stream_options",
    "n",
    "call_id",
    "url",
    "headers",
    "default_headers",
    "extra_headers",
    "extra_query",
    "extra_body",
    "query",
    "body",
    "options",
    "timeout",
    "http_client",
    "max_retries",
    "organization",
    "project",
    "webhook_secret",
}


def _mapping(value: object, name: str, allowed: set[str]) -> Mapping[str, object]:
    """校验声明式映射及允许字段，错误不回显配置值。"""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) or key not in allowed for key in value):
        raise ValueError(f"{name} contains unsupported fields")
    return value


def _file(path: Path) -> Mapping[str, object]:
    """读取可选 TOML 文件；缺失返回空映射，损坏文件报安全配置错误。"""
    if not path.is_file():
        if path.exists():
            raise ValueError("config_path must be a file")
        return {}
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        raise ValueError("config file must be readable TOML") from None
    root = _mapping(data, "config file", {"harness", "coding_agent"})
    _mapping(root.get("harness", {}), "harness", _HARNESS_FIELDS)
    agent = _mapping(root.get("coding_agent", {}), "coding_agent", _AGENT_FIELDS)
    _mapping(agent.get("compaction", {}), "compaction", _COMPACTION_FIELDS)
    return root


def _path(value: object, base: Path, name: str) -> str:
    """以指定基目录解析非空路径，不创建目录或文件。"""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return str((base / value).resolve())


def _integer(value: object, name: str, minimum: int) -> None:
    """校验整数下限，显式拒绝 bool。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def load_harness_config(
    overrides: Mapping[str, object] | None = None,
) -> HarnessConfig:
    """读取当前目录的可选 lhagent.toml；覆盖项优先于文件及默认值。"""
    override = _mapping(overrides if overrides is not None else {}, "overrides", _HARNESS_FIELDS)
    base = Path.cwd()
    data = _file(base / _FILE)
    section = _mapping(data.get("harness", {}), "harness", _HARNESS_FIELDS)
    return {
        "configs_dir": _path(
            {**section, **override}.get("configs_dir", _DEFAULT_CONFIGS_DIR), base, "configs_dir"
        )
    }


def load_coding_agent_config(
    overrides: Mapping[str, object] | None = None,
    *,
    config_path: str | None = None,
) -> CodingAgentConfig:
    """加载静态配置；不探测模型或创建运行时资源。"""
    override = _mapping(overrides if overrides is not None else {}, "overrides", _AGENT_FIELDS)
    if config_path is not None and (not isinstance(config_path, str) or not config_path.strip()):
        raise ValueError("config_path must be a nonempty string")
    path = Path(config_path).resolve() if config_path is not None else Path.cwd() / _FILE
    base = path.parent
    data = _file(path)
    section = _mapping(data.get("coding_agent", {}), "coding_agent", _AGENT_FIELDS)
    values = {**section, **override}
    compaction = dict(_mapping(section.get("compaction", {}), "compaction", _COMPACTION_FIELDS))
    compaction.update(_mapping(override.get("compaction", {}), "compaction", _COMPACTION_FIELDS))
    values["compaction"] = compaction

    for field in ("cwd", "system_prompt_path"):
        if field in values and values[field] is not None:
            values[field] = _path(values[field], base, field)
    if "additional_prompt_paths" in values:
        paths = values["additional_prompt_paths"]
        if not isinstance(paths, list):
            raise TypeError("additional_prompt_paths must be a list")
        values["additional_prompt_paths"] = [
            _path(item, base, "additional_prompt_paths") for item in paths
        ]
    values.setdefault("name", _DEFAULT_AGENT_NAME)
    # cwd 取当前配置文件目录；可变默认容器每次加载独立创建，不能共享。
    values.setdefault("cwd", str(base))
    values.setdefault("system_prompt_path", None)
    values.setdefault("additional_prompt_paths", [])
    values.setdefault("tools", list(_DEFAULT_TOOLS))
    values.setdefault("generation_parameters", {})
    compaction.setdefault("enabled", _DEFAULT_COMPACTION_ENABLED)
    compaction.setdefault("keep_recent_tokens", _DEFAULT_KEEP_RECENT_TOKENS)
    if "max_output_tokens" in values:
        compaction.setdefault("reserve_tokens", values["max_output_tokens"])
    if "max_summary_output_tokens" not in values:
        reserve = compaction.get("reserve_tokens")
        maximum = values.get("max_output_tokens")
        if type(reserve) is int and type(maximum) is int:
            values["max_summary_output_tokens"] = max(1, min(maximum, (4 * reserve) // 5))
    config = cast(CodingAgentConfig, values)
    validate_coding_agent_config(config)
    return config


def validate_coding_agent_config(config: CodingAgentConfig) -> None:
    """检查类型、预算及生成参数；诊断不包含传入值。"""
    values = _mapping(config, "config", _AGENT_FIELDS)
    for name in ("name", "model", "cwd"):
        value = values.get(name)
        if value is None:
            raise ValueError(f"{name} is required")
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        if not value.strip():
            raise ValueError(f"{name} must be nonempty")
    prompt = values.get("system_prompt_path")
    if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
        raise TypeError("system_prompt_path must be a nonempty string or None")
    for name in ("additional_prompt_paths", "tools"):
        items = values.get(name)
        if not isinstance(items, list) or any(
            not isinstance(x, str) or not x.strip() for x in items
        ):
            raise TypeError(f"{name} must be a list of nonempty strings")
    for name in ("context_window", "max_output_tokens"):
        if name not in values:
            raise ValueError(f"{name} is required")
        _integer(values[name], name, 1)
    window = cast(int, values["context_window"])
    maximum = cast(int, values["max_output_tokens"])
    if maximum >= window:
        raise ValueError("max_output_tokens must be less than context_window")
    compaction = _mapping(values.get("compaction"), "compaction", _COMPACTION_FIELDS)
    if not isinstance(compaction.get("enabled"), bool):
        raise TypeError("compaction.enabled must be a boolean")
    for name, minimum in (("reserve_tokens", 1), ("keep_recent_tokens", 0)):
        _integer(compaction.get(name), f"compaction.{name}", minimum)
    reserve = cast(int, compaction["reserve_tokens"])
    if reserve < maximum or reserve >= window:
        raise ValueError(
            "compaction.reserve_tokens must cover max_output_tokens and be less than context_window"
        )
    _integer(values.get("max_summary_output_tokens"), "max_summary_output_tokens", 1)
    summary = cast(int, values["max_summary_output_tokens"])
    if summary > maximum or summary > reserve:
        raise ValueError("max_summary_output_tokens exceeds output or reserve limit")
    params = values.get("generation_parameters")
    if not isinstance(params, dict):
        raise TypeError("generation_parameters must be a dictionary")
    if any(not isinstance(key, str) or key.lower() in _FORBIDDEN_PARAMETERS for key in params):
        raise ValueError("generation_parameters contains unsupported fields")
    try:
        json.dumps(params, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("generation_parameters must be JSON serializable") from None
