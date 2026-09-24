"""搜索子进程入口，仅依赖标准库；输入一个请求，输出一个有界 ToolOutput。"""

import fnmatch
import json
import os
import re
import stat
import sys
from pathlib import Path

MAX_LINE_BYTES = 65536
BINARY_SAMPLE_BYTES = 8192
DEFAULT_SEARCH_PATH = "."


def compile_glob(pattern):
    """以 / 分段，** 匹配零个或多个目录；拒绝含糊或不完整的模式。"""
    if not pattern or "\x00" in pattern or "\\" in pattern or pattern.startswith("/"):
        raise ValueError("Glob must be a nonempty relative path using / separators")
    parts = pattern.split("/")
    for part in parts:
        if part in ("", ".", "..") or ("**" in part and part != "**"):
            raise ValueError("Glob requires nonempty path segments; ** must be a whole segment")
        if part == "**":
            continue
        # fnmatch 会把未闭合字符类当作字面量左括号，因此需提前拒绝。
        cursor = 0
        while cursor < len(part):
            if part[cursor] == "[":
                end = cursor + 1
                if end < len(part) and part[end] == "!":
                    end += 1
                if end < len(part) and part[end] == "]":
                    end += 1
                end = part.find("]", end)
                if end == -1:
                    raise ValueError("Unclosed glob character class")
                cursor = end
            cursor += 1
    # 按路径段匹配，避免 fnmatch 通配符跨越路径分隔符。
    compiled = [None if part == "**" else re.compile(fnmatch.translate(part)) for part in parts]

    def matches(path):
        """推进路径段状态集合；双星号可跨段或空匹配，普通通配符不可跨段。"""
        states = {0}
        for segment in path.split("/"):
            for index in range(len(parts)):
                if index in states and parts[index] == "**":
                    states.add(index + 1)
            following = set()
            for index in states:
                if index < len(parts):
                    if parts[index] == "**":
                        following.add(index)
                    elif compiled[index].fullmatch(segment):
                        following.add(index + 1)
            states = following
        for index in range(len(parts)):
            if index in states and parts[index] == "**":
                states.add(index + 1)
        return len(parts) in states

    return matches


def walk_files(root):
    """增量深度优先枚举，内存和打开的迭代器仅随目录深度增长。"""
    stack = [os.scandir(root)]
    try:
        while stack:
            entry = next(stack[-1], None)
            if entry is None:
                stack.pop().close()
            elif entry.is_dir(follow_symlinks=False):
                stack.append(os.scandir(entry.path))
            elif entry.is_file(follow_symlinks=False):
                yield Path(entry.path)
    finally:
        for iterator in stack:
            iterator.close()


def search(request):
    """执行一次 find/grep 请求，按输出预算截断并报告文件读取问题。"""
    args = request["arguments"]
    root = Path(request["cwd"]) / args.get("path", DEFAULT_SEARCH_PATH)
    lines = []
    remaining = request["bytes"]
    skipped = 0
    truncated = False

    def output(error=None):
        """组合搜索结果及错误、截断和二进制文件统计。"""
        return {
            "content": [{"type": "text", "text": error if error else "".join(lines)}],
            "details": {
                "path": str(root),
                "returned_matches": 0 if error else len(lines),
                "skipped_binary_files": skipped,
            },
            "is_error": error is not None,
            "truncated": truncated,
        }

    def append(value):
        """在行数和字节预算内追加一条 JSON 结果。"""
        nonlocal remaining, truncated
        # ASCII JSON 转义保留特殊文件名，并保证每条结果只占一个物理行。
        line = json.dumps(value, ensure_ascii=True) + "\n"
        size = len(line.encode("utf-8"))
        if len(lines) >= request["lines"] or size > remaining:
            truncated = True
            return False
        lines.append(line)
        remaining -= size
        return True

    try:
        pattern = args["pattern"]
        if request["kind"] == "find":
            match = compile_glob(pattern)
        else:
            regex = None if args.get("literal", False) else re.compile(pattern)
        mode = root.stat().st_mode
        if request["kind"] == "find" and not stat.S_ISDIR(mode):
            raise NotADirectoryError(str(root))
        directory = stat.S_ISDIR(mode)
        if not directory and not stat.S_ISREG(mode):
            raise ValueError("Search path must be a regular file or directory")
        files = walk_files(root) if directory else iter([root])
        try:
            for path in files:
                relative = path.relative_to(root).as_posix() if directory else path.name
                if request["kind"] == "find":
                    if match(relative) and not append(relative):
                        break
                    continue
                with path.open("rb") as stream:
                    if b"\x00" in stream.read(BINARY_SAMPLE_BYTES):
                        skipped += 1
                        continue
                    stream.seek(0)
                    line_number = 0
                    while raw := stream.readline(MAX_LINE_BYTES + 1):
                        line_number += 1
                        if len(raw) > MAX_LINE_BYTES:
                            raise ValueError(
                                f"{relative}:{line_number}: line exceeds {MAX_LINE_BYTES} bytes"
                            )
                        if b"\x00" in raw:
                            raise ValueError(
                                f"{relative}:{line_number}: NUL outside binary detection sample"
                            )
                        text = raw.decode("utf-8").removesuffix("\n").removesuffix("\r")
                        matched = (
                            pattern in text if regex is None else regex.search(text) is not None
                        )
                        if matched:
                            if not append({"path": relative, "line": line_number, "text": text}):
                                break
                if truncated:
                    break
        finally:
            if directory:
                files.close()
    except (OSError, ValueError, UnicodeError, re.error) as exc:
        return output(f"{type(exc).__name__}: {exc}")
    return output()


if __name__ == "__main__":
    json.dump(search(json.load(sys.stdin)), sys.stdout, ensure_ascii=True)
