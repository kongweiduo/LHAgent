"""集中定义语义样式，渲染代码通过样式名引用颜色。"""

from prompt_toolkit.styles import Style

SEMANTIC_STYLES = {
    "user": "bold ansicyan",
    "assistant": "",
    "reasoning": "italic ansibrightblack",
    "tool": "ansiblue",
    "success": "ansigreen",
    "warning": "ansiyellow",
    "error": "ansired",
    "muted": "ansibrightblack",
    "status": "bold",
}
STYLE = Style.from_dict(SEMANTIC_STYLES)
