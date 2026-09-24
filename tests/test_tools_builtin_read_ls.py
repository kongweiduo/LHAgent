"""临时目录中的 UTF-8 读取、目录列举、分页、预算及取消行为。"""

import asyncio
import os

import pytest

from lhagent.harness.tools.builtin.ls import create_ls_tool
from lhagent.harness.tools.builtin.read import create_read_tool
from lhagent.harness.tools.execution import execute_tool_call
from lhagent.harness.tools.validation import validate_schema


def invoke(tool, cwd, arguments, *, lines=200, size=4096, cancel=None):
    """以可调输出预算和取消信号执行单个文件查看工具。"""
    context = {
        "cwd": str(cwd),
        "timeout_seconds": 5,
        "cancel_event": cancel or asyncio.Event(),
        "max_output_lines": lines,
        "max_output_bytes": size,
    }
    return asyncio.run(
        execute_tool_call(
            {"call_id": "c1", "name": tool["name"], "arguments": arguments},
            [tool],
            context,
        )
    )


def test_read_pages_unicode_and_absolute_path(tmp_path):
    """读取分页保持 Unicode，并支持绝对路径。"""
    path = tmp_path / "中文.txt"
    path.write_text("一\n二\n三", encoding="utf-8")
    tool = create_read_tool()
    validate_schema(tool["parameters"])
    first = invoke(tool, tmp_path, {"path": path.name, "limit": 2})
    assert first["status"] == "success"
    assert first["output"]["content"][0]["text"] == "一\n二\n"
    assert first["output"]["truncated"]
    assert first["output"]["details"]["next_offset"] == 3
    last = invoke(tool, tmp_path / "other", {"path": str(path), "offset": 3})
    assert last["output"]["content"][0]["text"] == "三"
    assert last["output"]["details"]["next_offset"] is None
    assert not last["output"]["truncated"]


def test_read_limits_and_large_file(tmp_path):
    """大文件读取受行数、单行字节和输出预算限制。"""
    path = tmp_path / "large"
    path.write_text("a\n" * 100000, encoding="utf-8")
    tool = create_read_tool()
    first = invoke(tool, tmp_path, {"path": "large", "offset": 99999}, lines=1)
    assert first["output"]["content"][0]["text"] == "a\n"
    assert first["output"]["details"]["next_offset"] == 100000
    second = invoke(tool, tmp_path, {"path": "large", "offset": 100000})
    assert second["output"]["content"][0]["text"] == "a\n"
    assert not second["output"]["truncated"]
    small = invoke(tool, tmp_path, {"path": "large"}, size=3)
    assert small["output"]["content"][0]["text"] == "a\n"
    assert small["output"]["details"]["next_offset"] == 2
    assert invoke(tool, tmp_path, {"path": "large"}, size=1)["status"] == "execution_error"
    (tmp_path / "long").write_text("x" * 70000, encoding="utf-8")
    assert invoke(tool, tmp_path, {"path": "long"}, size=100000)["status"] == "execution_error"
    assert invoke(tool, tmp_path, {"path": "large", "offset": 0})["status"] == "validation_error"


def test_read_errors_and_cancel(tmp_path):
    """读取错误及取消不泄漏文件资源。"""
    tool = create_read_tool()
    assert invoke(tool, tmp_path, {"path": "missing"})["status"] == "execution_error"
    (tmp_path / "bad").write_bytes(b"\xff\n")
    assert invoke(tool, tmp_path, {"path": "bad"})["status"] == "execution_error"
    (tmp_path / "dir").mkdir()
    assert invoke(tool, tmp_path, {"path": "dir"})["status"] == "execution_error"
    event = asyncio.Event()
    event.set()
    assert invoke(tool, tmp_path, {"path": "bad"}, cancel=event)["status"] == "cancelled"


def test_ls_sorted_types_empty_and_truncation(tmp_path):
    """列举按名称排序并保留类型、空目录及截断语义。"""
    tool = create_ls_tool()
    validate_schema(tool["parameters"])
    assert invoke(tool, tmp_path, {})["output"]["content"][0]["text"] == ""
    (tmp_path / "z").mkdir()
    (tmp_path / "a").write_text("", encoding="utf-8")
    (tmp_path / "中文\nname").touch()
    (tmp_path / ".hidden").touch()
    (tmp_path / "link").symlink_to(tmp_path / "z", target_is_directory=True)
    listing = invoke(tool, tmp_path / "else", {"path": str(tmp_path)})
    assert listing["output"]["content"][0]["text"] == (
        'file\t".hidden"\nfile\t"a"\nother\t"link"\ndirectory\t"z"\nfile\t"中文\\nname"\n'
    )
    first = invoke(tool, tmp_path, {}, lines=1)
    assert first["output"]["content"][0]["text"] == 'file\t".hidden"\n'
    assert first["output"]["truncated"]
    assert first["output"]["details"]["total_entries"] == 5
    assert invoke(tool, tmp_path, {}, size=0)["output"]["truncated"]
    assert invoke(tool, tmp_path, {"path": "missing"})["status"] == "execution_error"
    assert invoke(tool, tmp_path, {"path": "a"})["status"] == "execution_error"


def test_ls_large_directory_and_cancel(tmp_path):
    """大目录扫描支持取消且内存受结果数限制。"""
    for n in range(250):
        (tmp_path / f"{n:03}").touch()
    result = invoke(create_ls_tool(), tmp_path, {}, lines=100000)
    assert result["output"]["truncated"]
    assert len(result["output"]["content"][0]["text"].splitlines()) == 200
    event = asyncio.Event()
    event.set()
    assert invoke(create_ls_tool(), tmp_path, {}, cancel=event)["status"] == "cancelled"


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason="requires Unix non-root permissions"
)
def test_permission_failure(tmp_path):
    """权限不足返回明确工具错误。"""
    path = tmp_path / "locked"
    path.mkdir()
    (path / "file").write_text("secret", encoding="utf-8")
    path.chmod(0)
    try:
        assert invoke(create_ls_tool(), tmp_path, {"path": "locked"})["status"] == "execution_error"
        assert (
            invoke(create_read_tool(), tmp_path, {"path": "locked/file"})["status"]
            == "execution_error"
        )
    finally:
        path.chmod(0o700)
