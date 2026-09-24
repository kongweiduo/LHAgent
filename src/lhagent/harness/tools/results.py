"""检查工具输出契约并整理可反馈模型的结果，不判断任务完成质量。"""

from copy import deepcopy
from typing import Any, cast

from .types import ToolCall, ToolContext, ToolOutput, ToolResult
from .validation import _check_json


def validate_output(output: object) -> ToolOutput:
    """检查必要字段、内容块及 JSON 可保存性；非法输出抛出 ValueError。"""
    if not isinstance(output, dict):
        raise ValueError("Tool output must be an object")
    for field, expected in (
        ("content", list),
        ("details", dict),
        ("is_error", bool),
        ("truncated", bool),
    ):
        if not isinstance(output.get(field), expected):
            raise ValueError(f"Tool output.{field} must be {expected.__name__}")
    _check_json(output)
    for index, block in enumerate(output["content"]):
        prefix = f"Tool output.content[{index}]"
        if not isinstance(block, dict):
            raise ValueError(f"{prefix} must be a content block")
        kind = block.get("type")
        fields: dict[str, object]
        if kind in ("text", "reasoning"):
            fields = {"text": str}
        elif kind == "tool_call":
            fields = {"call_id": str, "name": str, "complete": bool}
            if "arguments" in block:
                fields["arguments"] = dict  # type: ignore[assignment]
            if "arguments_json" in block:
                fields["arguments_json"] = str
        elif kind == "tool_result":
            fields = {"is_error": bool}  # type: ignore[assignment]
            if "content" not in block:
                raise ValueError(f"{prefix}.content is required")
        else:
            raise ValueError(f"{prefix} has unsupported content type: {kind!r}")
        for field, expected_type in cast(dict[str, object], fields).items():
            expected = cast(Any, expected_type)
            if not isinstance(block.get(field), expected):
                raise ValueError(f"{prefix}.{field} must be {expected.__name__}")
        if kind in ("text", "reasoning"):
            try:
                block["text"].encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(f"{prefix}.text must be valid UTF-8 text") from exc
    return cast(ToolOutput, output)


def limit_output(output: ToolOutput, context: ToolContext) -> ToolOutput:
    """限制文本/推理块的累计行数和 UTF-8 字节数，返回独立副本。

    每块以 splitlines(keepends=True) 计行；保留原换行，字节截断不拆字符。
    结构化块和 details 完整保留、不计文本预算；truncated 是截断标记，
    不追加占用预算的提示文本。已截断的输出不会被重置为未截断。
    """
    for field in ("max_output_lines", "max_output_bytes"):
        if type(context[field]) is not int or context[field] < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    limited = deepcopy(output)
    lines_left = context["max_output_lines"]
    bytes_left = context["max_output_bytes"]
    exhausted = False
    for block in limited["content"]:
        if block["type"] not in ("text", "reasoning"):
            continue
        text = block["text"]
        lines = text.splitlines(keepends=True)
        candidate = "" if exhausted else "".join(lines[:lines_left])
        encoded = candidate.encode("utf-8")
        kept = encoded[:bytes_left].decode("utf-8", errors="ignore")
        if kept != text:
            limited["truncated"] = True
            exhausted = True
        block["text"] = kept
        lines_left -= len(kept.splitlines(keepends=True))
        bytes_left -= len(kept.encode("utf-8"))
    return limited


def result_from_output(call: ToolCall, output: ToolOutput) -> ToolResult:
    """根据显式失败标记生成结果；截断本身不表示失败。"""
    return {
        "call_id": call["call_id"],
        "name": call["name"],
        "status": "execution_error" if output["is_error"] else "success",
        "output": output,
        "error": "Tool reported execution failure" if output["is_error"] else None,
    }
