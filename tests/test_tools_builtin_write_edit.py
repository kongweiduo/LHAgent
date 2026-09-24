"""临时文件写入与唯一替换，验证参数错误和预取消不修改文件。"""

import asyncio
import os

import pytest

from lhagent.harness.tools.builtin.edit import create_edit_tool
from lhagent.harness.tools.builtin.write import create_write_tool
from lhagent.harness.tools.execution import execute_tool_call
from lhagent.harness.tools.validation import validate_schema


def invoke(tool, cwd, arguments, *, cancel=None):
    """在临时工作目录通过统一工具入口执行写入或替换。"""
    context = {
        "cwd": str(cwd),
        "timeout_seconds": 5,
        "cancel_event": cancel or asyncio.Event(),
        "max_output_lines": 200,
        "max_output_bytes": 4096,
    }
    return asyncio.run(
        execute_tool_call(
            {"call_id": "c1", "name": tool["name"], "arguments": arguments},
            [tool],
            context,
        )
    )


def test_write_creates_overwrites_and_preserves_newlines(tmp_path):
    """写入支持创建与覆盖并保留原始换行。"""
    tool = create_write_tool()
    validate_schema(tool["parameters"])
    path = tmp_path / "nested" / "中文.txt"
    first = invoke(tool, tmp_path, {"path": "nested/中文.txt", "content": "你好\r\n世界\n"})
    assert first["status"] == "success"
    assert path.read_bytes() == "你好\r\n世界\n".encode()
    assert first["output"]["details"]["bytes_written"] == len(path.read_bytes())
    assert (
        invoke(tool, tmp_path / "other", {"path": str(path), "content": ""})["status"] == "success"
    )
    assert path.read_bytes() == b""
    assert (
        invoke(tool, tmp_path, {"path": "nested/中文.txt", "content": "new"})["status"] == "success"
    )
    assert path.read_bytes() == b"new"


def test_write_invalid_arguments_encoding_and_path(tmp_path):
    """非法参数、编码或路径错误不被误报为成功。"""
    tool = create_write_tool()
    assert invoke(tool, tmp_path, {"path": "new"})["status"] == "validation_error"
    assert invoke(tool, tmp_path, {"path": "", "content": "a"})["status"] == "validation_error"
    assert (
        invoke(tool, tmp_path, {"path": "new", "content": "\ud800"})["status"] == "execution_error"
    )
    assert not (tmp_path / "new").exists()
    assert invoke(tool, tmp_path, {"path": ".", "content": "x"})["status"] == "execution_error"


def test_edit_unique_unicode_and_exact_newlines(tmp_path):
    """唯一替换正确处理 Unicode 和精确换行。"""
    tool = create_edit_tool()
    validate_schema(tool["parameters"])
    path = tmp_path / "sample.txt"
    path.write_bytes("一\r\n二\n三\r\n".encode())
    result = invoke(
        tool,
        tmp_path / "else",
        {
            "path": str(path),
            "old_text": "二\n",
            "new_text": "世界\n",
        },
    )
    assert result["status"] == "success"
    assert result["output"]["details"]["replacements"] == 1
    assert path.read_bytes() == "一\r\n世界\n三\r\n".encode()
    assert (
        invoke(
            tool,
            tmp_path,
            {
                "path": "sample.txt",
                "old_text": "世界\n",
                "new_text": "",
            },
        )["status"]
        == "success"
    )
    assert path.read_bytes() == "一\r\n三\r\n".encode()


def test_edit_rejects_non_unique_and_invalid_text_without_changes(tmp_path):
    """非唯一匹配及非法文本不修改文件。"""
    tool = create_edit_tool()
    path = tmp_path / "sample.txt"
    original = b"same\r\nsame\r\n"
    path.write_bytes(original)
    for old, status in [
        ("", "validation_error"),
        ("missing", "execution_error"),
        ("same", "execution_error"),
        ("same\n", "execution_error"),
    ]:
        result = invoke(tool, tmp_path, {"path": "sample.txt", "old_text": old, "new_text": "new"})
        assert result["status"] == status
        assert path.read_bytes() == original
    assert (
        "found multiple"
        in invoke(
            tool,
            tmp_path,
            {
                "path": "sample.txt",
                "old_text": "same",
                "new_text": "new",
            },
        )["output"]["content"][0]["text"]
    )
    path.write_bytes(b"aaa")
    assert (
        invoke(
            tool,
            tmp_path,
            {
                "path": "sample.txt",
                "old_text": "aa",
                "new_text": "x",
            },
        )["status"]
        == "execution_error"
    )
    assert path.read_bytes() == b"aaa"
    path.write_bytes(b"unique\r\n")
    result = invoke(
        tool, tmp_path, {"path": "sample.txt", "old_text": "unique", "new_text": "\ud800"}
    )
    assert result["status"] == "execution_error"
    assert path.read_bytes() == b"unique\r\n"
    path.write_bytes(b"\xffsame")
    assert (
        invoke(tool, tmp_path, {"path": "sample.txt", "old_text": "same", "new_text": "x"})[
            "status"
        ]
        == "execution_error"
    )
    assert path.read_bytes() == b"\xffsame"
    assert (
        invoke(tool, tmp_path, {"path": "missing", "old_text": "x", "new_text": "y"})["status"]
        == "execution_error"
    )


def test_pre_cancelled_write_and_edit_do_not_modify(tmp_path):
    """预取消的写入和替换不得产生文件副作用。"""
    path = tmp_path / "sample.txt"
    path.write_text("old", encoding="utf-8")
    cancel = asyncio.Event()
    cancel.set()
    assert (
        invoke(
            create_write_tool(), tmp_path, {"path": "sample.txt", "content": "new"}, cancel=cancel
        )["status"]
        == "cancelled"
    )
    assert (
        invoke(
            create_edit_tool(),
            tmp_path,
            {"path": "sample.txt", "old_text": "old", "new_text": "new"},
            cancel=cancel,
        )["status"]
        == "cancelled"
    )
    assert path.read_bytes() == b"old"
    assert not (tmp_path / "nested").exists()
    assert (
        invoke(
            create_write_tool(), tmp_path, {"path": "nested/file", "content": "x"}, cancel=cancel
        )["status"]
        == "cancelled"
    )
    assert not (tmp_path / "nested").exists()


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason="requires Unix non-root permissions"
)
def test_permission_errors(tmp_path):
    """文件权限错误作为执行失败返回。"""
    locked = tmp_path / "locked"
    locked.mkdir()
    path = locked / "sample.txt"
    path.write_text("old", encoding="utf-8")
    locked.chmod(0)
    try:
        for tool, args in [
            (create_write_tool(), {"path": "locked/new.txt", "content": "x"}),
            (
                create_edit_tool(),
                {"path": "locked/sample.txt", "old_text": "old", "new_text": "new"},
            ),
        ]:
            result = invoke(tool, tmp_path, args)
            assert result["status"] == "execution_error"
            assert "PermissionError" in result["output"]["content"][0]["text"]
    finally:
        locked.chmod(0o700)
    assert path.read_text(encoding="utf-8") == "old"
    assert not (locked / "new.txt").exists()
