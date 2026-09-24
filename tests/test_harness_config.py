"""TOML 配置合并、路径解析、预算默认值及禁止字段校验，不访问模型。"""

import json
import socket
import traceback

import pytest

from lhagent.harness.configs.loader import (
    load_coding_agent_config,
    load_harness_config,
    validate_coding_agent_config,
)

BUDGET = {"model": "test-model", "context_window": 1000, "max_output_tokens": 100}
SECRET = "secret-not-for-errors"


@pytest.mark.parametrize(
    "field", ["tool_choice", "stream_options", "extra_body", "extra_headers", "n", "timeout"]
)
def test_unsupported_wire_options_rejected_before_run(field):
    """不支持的协议参数在加载配置时拒绝，避免运行后才报错。"""
    with pytest.raises(ValueError, match="generation_parameters"):
        load_coding_agent_config({**BUDGET, "generation_parameters": {field: "unsupported"}})


def test_defaults_and_harness_config(monkeypatch, tmp_path):
    """默认值及 harness 配置按约定加载。"""
    monkeypatch.chdir(tmp_path)
    assert load_harness_config() == {"configs_dir": str(tmp_path / "configs")}
    config = load_coding_agent_config(BUDGET)
    assert config == {
        **BUDGET,
        "name": "coding",
        "cwd": str(tmp_path),
        "system_prompt_path": None,
        "additional_prompt_paths": [],
        "tools": ["read", "ls", "find", "grep", "write", "edit", "bash"],
        "generation_parameters": {},
        "compaction": {"enabled": True, "reserve_tokens": 100, "keep_recent_tokens": 20000},
        "max_summary_output_tokens": 80,
    }
    assert not (tmp_path / "lhagent.toml").exists()


def test_file_priority_partial_compaction_and_paths(monkeypatch, tmp_path):
    """文件和覆盖项按优先级合并，压缩子字段与相对路径正确解析。"""
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "lhagent.toml").write_text("""
[harness]
configs_dir = "stored"
[coding_agent]
model = "file-model"
context_window = 2000
max_output_tokens = 200
cwd = "work"
system_prompt_path = "prompts/system.md"
additional_prompt_paths = ["prompts/extra.md"]
tools = ["read", "ls"]
[coding_agent.compaction]
reserve_tokens = 300
keep_recent_tokens = 15
""")
    config = load_coding_agent_config(
        {"model": "override-model", "compaction": {"enabled": False}, "tools": ["read"]},
        config_path=str(folder / "lhagent.toml"),
    )
    assert config["model"] == "override-model"
    assert config["compaction"] == {
        "enabled": False,
        "reserve_tokens": 300,
        "keep_recent_tokens": 15,
    }
    assert config["max_summary_output_tokens"] == 200
    assert config["cwd"] == str(folder / "work")
    assert config["system_prompt_path"] == str(folder / "prompts/system.md")
    assert config["additional_prompt_paths"] == [str(folder / "prompts/extra.md")]
    assert config["tools"] == ["read"]
    monkeypatch.chdir(folder)
    assert load_harness_config()["configs_dir"] == str(folder / "stored")
    assert load_harness_config({"configs_dir": "custom"})["configs_dir"] == str(folder / "custom")


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"model": None}, ValueError),
        ({"context_window": None}, TypeError),
        ({"max_output_tokens": None}, TypeError),
        ({"context_window": True}, TypeError),
        ({"max_output_tokens": False}, TypeError),
        ({"max_output_tokens": 1000}, ValueError),
        ({"compaction": {"enabled": 0}}, TypeError),
        ({"compaction": {"enabled": False, "reserve_tokens": True}}, TypeError),
        ({"compaction": {"enabled": False, "reserve_tokens": 99}}, ValueError),
        ({"compaction": {"reserve_tokens": 1000}}, ValueError),
        ({"compaction": {"keep_recent_tokens": -1}}, ValueError),
        ({"max_summary_output_tokens": True}, TypeError),
        ({"max_summary_output_tokens": 0}, ValueError),
        ({"max_summary_output_tokens": 101}, ValueError),
        ({"compaction": {"reserve_tokens": 120}, "max_summary_output_tokens": 121}, ValueError),
        ({"tools": ["read", 1]}, TypeError),
        ({"additional_prompt_paths": [None]}, TypeError),
        ({"generation_parameters": {"temperature": float("nan")}}, ValueError),
        ({"generation_parameters": {"temperature": object()}}, ValueError),
    ],
)
def test_invalid_values(monkeypatch, tmp_path, changes, error):
    """非法声明式字段值不能通过校验。"""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(error):
        load_coding_agent_config({**BUDGET, **changes})


def test_missing_budget_and_direct_validation(monkeypatch, tmp_path):
    """预算必填要求在加载与直接校验中一致。"""
    monkeypatch.chdir(tmp_path)
    for field in ("model", "context_window", "max_output_tokens"):
        with pytest.raises(ValueError, match=field):
            load_coding_agent_config({key: value for key, value in BUDGET.items() if key != field})
    config = load_coding_agent_config(BUDGET)
    validate_coding_agent_config(config)
    config["compaction"]["enabled"] = False
    config["compaction"]["reserve_tokens"] = 0
    with pytest.raises(ValueError, match="reserve_tokens"):
        validate_coding_agent_config(config)


@pytest.mark.parametrize(
    "key",
    [
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "messages",
        "tools",
        "model",
        "stream",
        "api_key",
        "Authorization",
        "access_token",
        "base_url",
    ],
)
def test_forbidden_generation_parameter_redacted(monkeypatch, tmp_path, key):
    """禁止的生成参数报错时不回显其值。"""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError) as error:
        load_coding_agent_config({**BUDGET, "generation_parameters": {key: SECRET}})
    assert SECRET not in "".join(traceback.format_exception(error.value))


def test_unknown_fields_malformed_file_and_no_side_effects(monkeypatch, tmp_path):
    """未知字段和畸形文件报错且不创建运行资源。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: pytest.fail("network access"))
    with pytest.raises(ValueError) as error:
        load_coding_agent_config({**BUDGET, SECRET: SECRET})
    assert SECRET not in str(error.value)
    with pytest.raises(ValueError):
        load_coding_agent_config({**BUDGET, "compaction": {SECRET: SECRET}})
    with pytest.raises(TypeError):
        load_harness_config([])
    (tmp_path / "lhagent.toml").write_text("[harness]\nunknown = 1\n")
    with pytest.raises(ValueError):
        load_coding_agent_config(BUDGET)
    (tmp_path / "lhagent.toml").write_text("[coding_agent.compaction]\nunknown = 1\n")
    with pytest.raises(ValueError):
        load_harness_config()
    (tmp_path / "lhagent.toml").write_text(f"[coding_agent\n{SECRET}")
    with pytest.raises(ValueError) as error:
        load_coding_agent_config(BUDGET)
    assert SECRET not in "".join(traceback.format_exception(error.value))
    assert not (tmp_path / "configs").exists()


@pytest.mark.parametrize("loader", [load_harness_config, load_coding_agent_config])
def test_non_utf8_file_has_safe_configuration_error(monkeypatch, tmp_path, loader):
    """非 UTF-8 文件转为安全配置错误。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lhagent.toml").write_bytes(b"[coding_agent]\nmodel = '\xff'\n")
    with pytest.raises(ValueError, match="^config file must be readable TOML$") as error:
        loader()
    assert "UnicodeDecodeError" not in "".join(traceback.format_exception(error.value))


def test_serializable_tools_and_summary_floor(monkeypatch, tmp_path):
    """工具配置可序列化，摘要默认输出至少为一个 token。"""
    monkeypatch.chdir(tmp_path)
    config = load_coding_agent_config(
        {"model": "m", "context_window": 3, "max_output_tokens": 1, "tools": ["future-tool"]}
    )
    assert config["max_summary_output_tokens"] == 1
    assert json.loads(json.dumps(config))["tools"] == ["future-tool"]


def test_default_tools_match_builtins_and_explicit_selection(monkeypatch, tmp_path):
    """省略工具启用全部，显式子集和空列表保持原样，各次加载互不污染。"""
    from lhagent.harness.tools.builtin import create_builtin_tools

    monkeypatch.chdir(tmp_path)
    config = load_coding_agent_config(BUDGET)
    assert set(config["tools"]) == {tool["name"] for tool in create_builtin_tools()}
    config["tools"].clear()
    assert len(load_coding_agent_config(BUDGET)["tools"]) == 7
    for selected in ([], ["read", "ls"]):
        assert load_coding_agent_config({**BUDGET, "tools": selected})["tools"] == selected
    (tmp_path / "lhagent.toml").write_text("[coding_agent]\ntools = []\n")
    assert load_coding_agent_config(BUDGET)["tools"] == []
