"""纯终端格式化与有界预览，按显示列宽处理 Unicode 和 ANSI 样式。"""

import json
from dataclasses import dataclass

from prompt_toolkit.formatted_text import ANSI, FormattedText, to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.utils import get_cwidth

from lhagent.client.types import ContentBlock
from lhagent.diagnostics import diagnostic
from lhagent.tui.models import Footer, MessageBlock, NoticeBlock, TranscriptBlock

# 界面预览预算；工具执行输出上限由 ToolContext 独立控制。
_PREVIEW_LINES = 6
_PREVIEW_BYTES = 2048
_DETAIL_PREVIEW_LINES = 2
_DETAIL_PREVIEW_BYTES = 256


@dataclass(frozen=True)
class PreviewLimits:
    """预览行数和 UTF-8 字节预算；列宽由每次渲染传入。"""

    max_lines: int = _PREVIEW_LINES
    max_bytes: int = _PREVIEW_BYTES

    def __post_init__(self) -> None:
        """拒绝非正行数和负字节预算。"""
        if self.max_lines < 1 or self.max_bytes < 0:
            raise ValueError("preview requires positive lines and non-negative bytes")


@dataclass(frozen=True)
class Preview:
    """预览样式片段及省略的可见源文本字节数。"""

    text: FormattedText
    omitted_bytes: int


def formatted(text: str, style: str = "assistant") -> FormattedText:
    """将 ANSI 解析为样式并过滤控制序列；错误样式文本先脱敏。"""
    if style == "error":
        text = diagnostic(text)
    fragments = to_formatted_text(ANSI(text))
    return FormattedText(
        [
            (
                f"class:{style} {ansi_style}".strip(),
                "".join(ch for ch in value if ch in "\n\t" or (ord(ch) >= 32 and ord(ch) != 127)),
            )
            for ansi_style, value, *_ in fragments
            if "[ZeroWidthEscape]" not in ansi_style
        ]
    )


def _json(value: object) -> str:
    """稳定序列化结构化展示内容，保留 Unicode。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(", ", ": "))


_DEFAULT_PREVIEW_LIMITS = PreviewLimits()


def preview(
    text: str,
    *,
    width: int = 80,
    limits: PreviewLimits = _DEFAULT_PREVIEW_LIMITS,
    style: str = "tool",
) -> Preview:
    """截取同时满足行数、UTF-8 字节和终端列宽限制的前缀。

    制表符展开到四列边界；宽字符连空行也放不下时停止。省略量计可见源文本
    字节（不含 ANSI 样式），不是列数；提示标记不占内容预算并按给定宽度换行。"""
    if width < 1:
        raise ValueError("width must be positive")
    source = formatted(text, style)
    total = len(fragment_list_to_text(source).encode("utf-8"))
    output: list[tuple[str, str]] = []
    used = column = 0
    line = 1
    stopped = False
    for token_style, value, *_ in source:
        for char in value:
            size = len(char.encode("utf-8"))
            if used + size > limits.max_bytes:
                stopped = True
                break
            if char == "\n":
                if line >= limits.max_lines:
                    stopped = True
                    break
                output.append((token_style, char))
                line, column = line + 1, 0
            else:
                rendered = " " * (4 - column % 4) if char == "\t" else char
                cells = sum(get_cwidth(c) for c in rendered)
                if cells > width:
                    stopped = True
                    break
                if column + cells > width:
                    if line >= limits.max_lines:
                        stopped = True
                        break
                    output.append((token_style, "\n"))
                    line, column = line + 1, 0
                output.append((token_style, rendered))
                column += cells
            used += size
        if stopped:
            break
    omitted = total - used
    if omitted:
        marker = f"... (+{omitted} UTF-8 bytes)"
        wrapped = "\n".join(marker[i : i + width] for i in range(0, len(marker), width))
        output.append(("class:muted", ("\n" if output else "") + wrapped))
    return Preview(FormattedText(output), omitted)


def content_text(content: list[ContentBlock]) -> str:
    """保留内容块类别及结构化载荷，不把所有输出假定为纯文本。"""
    pieces = []
    for part in content:
        if part["type"] == "text":
            pieces.append(part["text"])
        elif part["type"] == "reasoning":
            pieces.append("[reasoning]\n" + part["text"])
        elif part["type"] == "tool_result":
            payload = part["content"]
            pieces.append(
                ("[tool result: error] " if part["is_error"] else "[tool result] ")
                + (payload if isinstance(payload, str) else _json(payload))
            )
        else:
            pieces.append("[tool call] " + _json(part))
    return "\n".join(pieces)


def format_block(block: TranscriptBlock, *, width: int = 80) -> FormattedText:
    """按块类型生成展示片段；工具参数、结果与详情分别受预览预算约束。"""
    if width < 1:
        raise ValueError("width must be positive")
    if isinstance(block, NoticeBlock):
        return formatted(block.text, block.kind)
    output = FormattedText([])
    if not block.in_context:
        output.append(("class:muted", "[outside active context]\n"))
    if isinstance(block, MessageBlock):
        visible = [p for p in block.content if p["type"] != "tool_call"]
        if visible:
            output.append((f"class:{block.role}", f"{block.role}:\n"))
        for index, part in enumerate(visible):
            if index:
                output.append(("", "\n"))
            style: str = "reasoning" if part["type"] == "reasoning" else block.role
            output.extend(formatted(content_text([part]), style))
        if block.status in ("length", "error", "cancelled", "unknown"):
            style = "error" if block.status == "error" else "warning"
            output.extend(
                formatted(f"\n[{block.status}]" + (f" {block.error}" if block.error else ""), style)
            )
        return FormattedText(output)
    output.extend(
        preview(
            f"{block.name} {_json(block.arguments)}",
            width=width,
            limits=PreviewLimits(max_lines=_DETAIL_PREVIEW_LINES, max_bytes=_DETAIL_PREVIEW_BYTES),
        ).text
    )
    status = block.status.replace("_", " ")
    style = (
        "success"
        if block.status == "success"
        else "error"
        if block.status in ("validation_error", "execution_error", "timeout")
        else "warning"
        if block.status in ("cancelled", "unknown")
        else "status"
    )
    output.append((f"class:{style}", f"\n[{status}]"))
    if block.result:
        result = block.result
        if result["error"]:
            output.append(("", "\n"))
            output.extend(preview(result["error"], width=width, style="error").text)
        tool_output = result["output"]
        if tool_output is not None:
            text = content_text(tool_output["content"])
            output.append(("", "\n"))
            output.extend(preview(text or "(empty output)", width=width).text)
            if tool_output["details"]:
                output.append(("", "\n"))
                output.extend(
                    preview(
                        "details: " + _json(tool_output["details"]),
                        width=width,
                        limits=PreviewLimits(_DETAIL_PREVIEW_LINES, _DETAIL_PREVIEW_BYTES),
                        style="muted",
                    ).text
                )
            if tool_output["truncated"]:
                output.append(
                    ("class:warning", "\n[tool output truncated; original remainder unknown]")
                )
        elif not result["error"]:
            output.append(("class:muted", "\n(no output)"))
    return FormattedText(output)


def format_footer(footer: Footer, *, width: int = 80) -> FormattedText:
    """将页脚限制在单行列宽内，超宽时保留前缀及省略号。"""
    text = (
        f"{footer.cwd} | {footer.session_id} | {footer.status}"
        f" | steering {footer.steering} / follow-up {footer.follow_up}"
    )
    if width < 1:
        raise ValueError("width must be positive")
    source = formatted(text.replace("\n", " ").replace("\t", " "), "status")
    if sum(get_cwidth(c) for fragment in source for c in fragment[1]) <= width:
        return source
    output: list[tuple[str, str]] = []
    columns = 0
    for style, value, *_ in source:
        for char in value:
            cells = get_cwidth(char)
            if columns + cells > width - 1:
                return FormattedText([*output, ("class:muted", "…")])
            output.append((style, char))
            columns += cells
    return FormattedText(output)
