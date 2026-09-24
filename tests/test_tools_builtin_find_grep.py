"""搜索语义、输出预算、文件错误及搜索子进程取消回收。"""

import asyncio
import json
import os

import pytest

from lhagent.harness.tools import _search
from lhagent.harness.tools.builtin.find import create_find_tool
from lhagent.harness.tools.builtin.grep import create_grep_tool
from lhagent.harness.tools.execution import execute_tool_call
from lhagent.harness.tools.validation import validate_schema
from tests.test_tools_builtin_read_ls import invoke


def records(result):
    """断言工具成功并解码逐行 JSON 搜索结果。"""
    assert result["status"] == "success", result
    return [json.loads(line) for line in result["output"]["content"][0]["text"].splitlines()]


def test_find_globs_unicode_hidden_and_ignored(tmp_path):
    """glob 搜索覆盖 Unicode、隐藏文件和忽略规则边界。"""
    for name in ["root.py", "root.txt", "中文/a.py", "中文/deep/b.py", ".hidden/c.py"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / ".gitignore").write_text("*.py\n", encoding="utf-8")
    tool = create_find_tool()
    validate_schema(tool["parameters"])
    assert set(records(invoke(tool, tmp_path, {"pattern": "**/*.py"}))) == {
        "root.py",
        "中文/a.py",
        "中文/deep/b.py",
        ".hidden/c.py",
    }
    assert records(invoke(tool, tmp_path, {"pattern": "*.py"})) == ["root.py"]
    assert records(invoke(tool, tmp_path, {"pattern": "中文/[ab].?y"})) == ["中文/a.py"]
    assert records(invoke(tool, tmp_path, {"pattern": "中文/**/b.py"})) == ["中文/deep/b.py"]
    assert records(
        invoke(tool, tmp_path / "absent", {"pattern": "*.py", "path": str(tmp_path)})
    ) == ["root.py"]


@pytest.mark.parametrize("pattern", ["[bad", "a/**b", "/absolute", "../*", "a//b", "a\\b", "a\x00"])
def test_invalid_glob(tmp_path, pattern):
    """非法 glob 在搜索前拒绝。"""
    assert invoke(create_find_tool(), tmp_path, {"pattern": pattern})["status"] == "execution_error"


def test_grep_regex_literal_unicode_and_lines(tmp_path):
    """正则/字面搜索保留 Unicode 和原始行号。"""
    (tmp_path / "中文\n.txt").write_bytes("零\r\na.b\r\naxb\n终点".encode())
    tool = create_grep_tool()
    validate_schema(tool["parameters"])
    matches = records(invoke(tool, tmp_path, {"pattern": "^a.b$"}))
    assert matches == [
        {"path": "中文\n.txt", "line": 2, "text": "a.b"},
        {"path": "中文\n.txt", "line": 3, "text": "axb"},
    ]
    assert records(invoke(tool, tmp_path, {"pattern": "a.b", "literal": True})) == matches[:1]
    assert records(invoke(tool, tmp_path, {"pattern": "终", "path": "中文\n.txt"}))[0]["line"] == 4
    assert len(records(invoke(tool, tmp_path, {"pattern": ""}))) == 4
    assert invoke(tool, tmp_path, {"pattern": "["})["status"] == "execution_error"
    assert records(invoke(tool, tmp_path, {"pattern": "[", "literal": True})) == []
    assert (
        invoke(tool, tmp_path, {"pattern": "x", "literal": "true"})["status"] == "validation_error"
    )


@pytest.mark.parametrize("factory", [create_find_tool, create_grep_tool])
def test_empty_missing_and_symlinks(tmp_path, factory):
    """空目录、缺失路径与符号链接按约定处理。"""
    tool = factory()
    assert records(invoke(tool, tmp_path, {"pattern": "missing"})) == []
    assert (
        invoke(tool, tmp_path, {"pattern": "*", "path": "missing"})["status"] == "execution_error"
    )
    folder = tmp_path / "dir"
    folder.mkdir()
    (folder / "target").write_text("needle", encoding="utf-8")
    (tmp_path / "alias").symlink_to(folder, target_is_directory=True)
    (folder / "loop").symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "file-link").symlink_to(folder / "target")
    (tmp_path / "broken").symlink_to(tmp_path / "missing")
    pattern = "**" if tool["name"] == "find" else "needle"
    found = records(invoke(tool, tmp_path, {"pattern": pattern}))
    assert len(found) == 1
    assert len(records(invoke(tool, tmp_path, {"pattern": pattern, "path": "alias"}))) == 1
    if tool["name"] == "grep":
        assert (
            records(invoke(tool, tmp_path, {"pattern": pattern, "path": "file-link"}))[0]["path"]
            == "file-link"
        )
    else:
        assert (
            invoke(tool, tmp_path, {"pattern": pattern, "path": "file-link"})["status"]
            == "execution_error"
        )


@pytest.mark.parametrize("factory", [create_find_tool, create_grep_tool])
def test_budgets_and_large_results(tmp_path, factory):
    """大结果受字节和行数预算共同限制。"""
    tool = factory()
    if tool["name"] == "find":
        for index in range(250):
            (tmp_path / f"文件{index}").touch()
        pattern = "*"
    else:
        (tmp_path / "many").write_text("匹配\n" * 100000, encoding="utf-8")
        pattern = "匹配"
    result = invoke(tool, tmp_path, {"pattern": pattern}, lines=10000, size=100000)
    assert len(records(result)) == 200
    assert result["output"]["truncated"]
    for lines, size in [(0, 1000), (100, 0), (1, 1000), (100, 100)]:
        result = invoke(tool, tmp_path, {"pattern": pattern}, lines=lines, size=size)
        text = result["output"]["content"][0]["text"]
        assert result["output"]["truncated"]
        assert len(text.encode("utf-8")) <= size
        assert len(records(result)) <= lines
    assert not invoke(tool, tmp_path, {"pattern": "absent"}, size=0)["output"]["truncated"]


def test_exact_budget_is_not_truncated(tmp_path):
    """恰好达到预算且无剩余结果时不标记截断。"""
    (tmp_path / "a").touch()
    result = invoke(create_find_tool(), tmp_path, {"pattern": "*"}, lines=1, size=4)
    assert records(result) == ["a"]
    assert not result["output"]["truncated"]


def test_binary_encoding_and_long_lines(tmp_path):
    """二进制、编码错误和超长行有明确处理。"""
    tool = create_grep_tool()
    (tmp_path / "binary").write_bytes(b"needle\x00\n")
    result = invoke(tool, tmp_path, {"pattern": "needle"})
    assert records(result) == []
    assert result["output"]["details"]["skipped_binary_files"] == 1
    for name, content in [
        ("invalid", b"\xff"),
        ("long", b"x" * 65537),
        ("late-nul", b"x\n" * 4096 + b"\x00"),
    ]:
        (tmp_path / name).write_bytes(content)
        result = invoke(tool, tmp_path, {"pattern": "absent", "path": name})
        assert result["status"] == "execution_error"
    (tmp_path / "valid").write_bytes(b"x" * 65536)
    assert records(invoke(tool, tmp_path, {"pattern": "absent", "path": "valid"})) == []


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason="requires Unix non-root permissions"
)
@pytest.mark.parametrize("factory", [create_find_tool, create_grep_tool])
def test_permission_error(tmp_path, factory):
    """权限错误作为工具错误返回。"""
    folder = tmp_path / "locked"
    folder.mkdir()
    folder.chmod(0)
    try:
        assert invoke(factory(), tmp_path, {"pattern": "x"})["status"] == "execution_error"
    finally:
        folder.chmod(0o700)


@pytest.mark.parametrize("factory", [create_find_tool, create_grep_tool])
def test_hidden_content_and_special_files(tmp_path, factory):
    """隐藏内容与特殊文件不会造成无限读取。"""
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "ignored.txt").write_text("needle", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(".hidden/\n", encoding="utf-8")
    if hasattr(os, "mkfifo"):
        os.mkfifo(tmp_path / "fifo")
    pattern = "**/*.txt" if factory == create_find_tool else "needle"
    assert len(records(invoke(factory(), tmp_path, {"pattern": pattern}))) == 1
    event = asyncio.Event()
    event.set()
    assert invoke(factory(), tmp_path, {"pattern": pattern}, cancel=event)["status"] == "cancelled"
    if hasattr(os, "mkfifo"):
        assert (
            invoke(factory(), tmp_path, {"pattern": pattern, "path": "fifo"})["status"]
            == "execution_error"
        )


def test_spawn_failure_is_execution_error(tmp_path, monkeypatch):
    """子进程启动失败转换为执行错误。"""

    async def fail(*args, **kwargs):
        raise OSError("cannot spawn worker")

    monkeypatch.setattr(_search.asyncio, "create_subprocess_exec", fail)
    result = invoke(create_find_tool(), tmp_path, {"pattern": "*"})
    assert result["status"] == "execution_error"
    assert "cannot spawn worker" in result["error"]


def test_worker_ignores_task_python_environment(tmp_path, monkeypatch):
    """任务的 Python 配置不应改变 agent 私有搜索解释器的初始化。"""
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "other-python"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "other-site-packages"))
    (tmp_path / "sample.txt").write_text("needle\n", encoding="utf-8")
    assert records(invoke(create_find_tool(), tmp_path, {"pattern": "*.txt"})) == ["sample.txt"]
    assert records(invoke(create_grep_tool(), tmp_path, {"pattern": "needle"})) == [
        {"path": "sample.txt", "line": 1, "text": "needle"}
    ]


@pytest.mark.parametrize("mode", ["event", "task", "timeout", "spawn"])
def test_cancellation_reaps_worker(tmp_path, monkeypatch, mode):
    """取消搜索后回收工作进程。"""
    (tmp_path / "expensive").write_text("a" * 1000 + "!", encoding="utf-8")
    original = asyncio.create_subprocess_exec
    processes = []

    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        async def spawn(*args, **kwargs):
            process = await original(*args, **kwargs)
            processes.append(process)
            started.set()
            if mode == "spawn":
                await release.wait()
            return process

        monkeypatch.setattr(_search.asyncio, "create_subprocess_exec", spawn)
        context = {
            "cwd": str(tmp_path),
            "timeout_seconds": 0.3 if mode == "timeout" else 5,
            "cancel_event": asyncio.Event(),
            "max_output_lines": 200,
            "max_output_bytes": 4096,
        }
        task = asyncio.create_task(
            execute_tool_call(
                {"call_id": "c", "name": "grep", "arguments": {"pattern": "(a+)+$"}},
                [create_grep_tool()],
                context,
            )
        )
        await started.wait()
        if mode in ("task", "spawn"):
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif mode == "event":
            context["cancel_event"].set()
            assert (await task)["status"] == "cancelled"
        else:
            assert (await task)["status"] == "timeout"
        assert processes and all(process.returncode is not None for process in processes)

    asyncio.run(scenario())
