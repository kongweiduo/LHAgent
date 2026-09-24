"""客户端配置优先级、地址语法、数值约束及凭据诊断边界。"""

import importlib
import socket
import traceback
from dataclasses import FrozenInstanceError, replace

import pytest

import lhagent.client.config as module
from lhagent.client.config import ClientConfig, load_config, validate_config

SECRET = "test-secret-do-not-disclose"
VALID = {"base_url": "https://example.com/v1", "api_key": SECRET}
ENV = {
    "base_url": "LHAGENT_BASE_URL",
    "api_key": "LHAGENT_API_KEY",
    "timeout_seconds": "LHAGENT_TIMEOUT_SECONDS",
    "max_retries": "LHAGENT_MAX_RETRIES",
    "max_retry_delay_seconds": "LHAGENT_MAX_RETRY_DELAY_SECONDS",
}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    """每个测试清除客户端环境字段，防止宿主配置影响默认值断言。"""
    monkeypatch.chdir(tmp_path)
    for name in ENV.values():
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name, raising=False)


def test_defaults_immutability_and_hidden_key():
    """默认配置不可变，凭据不出现在对象展示中。"""
    config = load_config(VALID)
    assert (config.timeout_seconds, config.max_retries, config.max_retry_delay_seconds) == (
        60,
        2,
        30,
    )
    assert SECRET not in repr(config)
    assert SECRET not in str(config)
    with pytest.raises(FrozenInstanceError):
        config.api_key = "replacement"


def test_environment_and_explicit_priority(monkeypatch):
    """显式覆盖优先于环境变量，环境优先于默认值。"""
    values = {**VALID, "timeout_seconds": 12.5, "max_retries": 4, "max_retry_delay_seconds": 8}
    for field, value in values.items():
        monkeypatch.setenv(ENV[field], str(value))
    assert load_config() == ClientConfig(**values)
    for name in ENV.values():
        monkeypatch.setenv(name, SECRET)
    overrides = {**VALID, "timeout_seconds": 10, "max_retries": 0, "max_retry_delay_seconds": 0}
    before = overrides.copy()
    assert load_config(overrides) == ClientConfig(**overrides)
    assert overrides == before
    with pytest.raises(ValueError, match="api_key"):
        load_config({**overrides, "api_key": None})


@pytest.mark.parametrize("field", ["base_url", "api_key"])
def test_missing_required(field):
    """缺少必填连接字段时拒绝加载。"""
    with pytest.raises(ValueError, match=field):
        load_config({key: value for key, value in VALID.items() if key != field})


@pytest.mark.parametrize(
    "url",
    [
        "",
        "example.com",
        "/v1",
        "ftp://example.com",
        "https://",
        "https://bad host/v1",
        "https://example.com:0",
        "https://example.com:65536",
        "https://example.com:abc",
        "https://example.com:",
        "https://[broken",
        "https://-bad.com",
        "https://a..com",
        "https://999.0.0.1",
        "https://example.com?",
        "https://example.com#",
        f"https://user:{SECRET}@example.com",
        f"https://example.com?key={SECRET}",
        "https://example.com\n",
        "https://example.com\\bad",
    ],
)
def test_invalid_addresses(url):
    """非法地址在本地校验阶段拒绝。"""
    with pytest.raises(ValueError, match="base_url") as error:
        load_config({**VALID, "base_url": url})
    assert SECRET not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/v1",
        "https://example.com/",
        "https://127.0.0.1",
        "http://[::1]:8080/v1",
        "https://例子.测试/v1",
    ],
)
def test_valid_addresses(url):
    """合法地址无需网络探测即可通过。"""
    assert load_config({**VALID, "base_url": url}).base_url == url


@pytest.mark.parametrize(
    "field, value, error",
    [
        ("api_key", "", ValueError),
        ("api_key", " \t", ValueError),
        ("api_key", "key\n", ValueError),
        ("api_key", 123, TypeError),
        ("base_url", 123, TypeError),
        *[
            (field, value, TypeError)
            for field in ("timeout_seconds", "max_retries", "max_retry_delay_seconds")
            for value in (True, False, "1", None)
        ],
        *[
            (field, value, ValueError)
            for field in ("timeout_seconds", "max_retry_delay_seconds")
            for value in (-1, float("nan"), float("inf"), float("-inf"), 10**400)
        ],
        ("timeout_seconds", 0, ValueError),
        ("max_retries", -1, ValueError),
        ("max_retries", 1.5, TypeError),
    ],
)
def test_invalid_fields_in_loader_and_direct_validation(field, value, error):
    """加载入口与直接校验遵守相同的字段约束。"""
    config = load_config(VALID)
    with pytest.raises(error, match=field):
        validate_config(replace(config, **{field: value}))
    with pytest.raises(error, match=field):
        load_config({**VALID, field: value})


@pytest.mark.parametrize("field", ["timeout_seconds", "max_retries", "max_retry_delay_seconds"])
def test_invalid_environment_diagnostics_are_redacted(monkeypatch, field):
    """环境数值错误不回显原始值或密钥。"""
    monkeypatch.setenv(ENV[field], SECRET)
    with pytest.raises(ValueError, match=field) as error:
        load_config(VALID)
    assert SECRET not in "".join(traceback.format_exception(error.value))


def test_invalid_containers_and_unknown_fields():
    """非法容器和未知字段不能绕过配置契约。"""
    with pytest.raises(TypeError):
        load_config([])
    with pytest.raises(TypeError):
        validate_config({})
    with pytest.raises(ValueError) as error:
        load_config({**VALID, SECRET: SECRET})
    assert SECRET not in str(error.value)


def test_import_is_lazy_and_load_reads_local_dotenv(monkeypatch, tmp_path):
    """导入不读取 dotenv，加载时读取当前目录文件且不访问网络。"""

    def forbidden(*args, **kwargs):
        pytest.fail("network access during configuration")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        f"LHAGENT_BASE_URL=https://example.com\nLHAGENT_API_KEY={SECRET}\n"
    )
    # Reload in a separate module namespace to preserve the dataclass identity.
    spec = importlib.util.spec_from_file_location("config_import_probe", module.__file__)
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(__import__("sys").modules, spec.name, probe)
    spec.loader.exec_module(probe)
    assert all(name not in module.os.environ for name in ENV.values())
    assert probe.load_config().api_key == SECRET
    assert probe.load_config().base_url == "https://example.com"
    assert probe.load_config(VALID).api_key == SECRET
    from lhagent.diagnostics import diagnostic

    assert SECRET not in diagnostic(SECRET)


def test_dotenv_priority_numeric_values_and_literal_key(monkeypatch, tmp_path):
    """文件提供缺省值，环境与显式配置优先，密钥不做变量插值。"""
    (tmp_path / ".env").write_text(
        "LHAGENT_BASE_URL=https://file.example/v1\n"
        'LHAGENT_API_KEY="literal-${UNSET_TEST_VALUE}"\n'
        "LHAGENT_MAX_RETRIES=4\nLHAGENT_TIMEOUT_SECONDS=12.5\n"
    )
    monkeypatch.setenv("LHAGENT_BASE_URL", "https://env.example/v1")
    config = load_config()
    assert config.base_url == "https://env.example/v1"
    assert config.api_key == "literal-${UNSET_TEST_VALUE}"
    assert config.max_retries == 4
    assert config.timeout_seconds == 12.5
    assert load_config(VALID).api_key == SECRET
    monkeypatch.setenv("LHAGENT_API_KEY", "")
    with pytest.raises(ValueError, match="api_key"):
        load_config()


def test_dotenv_does_not_search_parent_directories(monkeypatch, tmp_path):
    """启动目录没有文件时不意外加载父目录的凭据。"""
    (tmp_path / ".env").write_text(
        "LHAGENT_BASE_URL=https://example.com\nLHAGENT_API_KEY=parent-key\n"
    )
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    with pytest.raises(ValueError, match="base_url"):
        load_config()
