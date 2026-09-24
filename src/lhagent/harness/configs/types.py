"""声明 harness 与 coding agent 的配置数据契约。

配置仅包含可序列化的声明式选项，不包含运行时对象。
文件格式、来源及默认值见本目录 readme；预算字段见相应章节。
"""

from typing import TypedDict


class HarnessConfig(TypedDict, total=False):
    """通用 harness 配置；加载后 configs_dir 为绝对路径。"""

    configs_dir: str


class CompactionConfig(TypedDict, total=False):
    """声明式压缩覆盖项；loader 按 readme 逐字段补齐为 CompactionSettings。"""

    enabled: bool
    reserve_tokens: int
    keep_recent_tokens: int


class CodingAgentConfig(TypedDict, total=False):
    """启动单个 CodingAgent 所需的声明式选项。

    模型、工作目录、提示词来源、启用工具和生成参数由调用方配置；运行中不
    切换模型。字段采用可选形式，实际必填项及默认值由加载与校验逻辑确定。
    client 的通信参数由 client.config 独立管理。
    加载完成后 model、context_window、max_output_tokens 必须明确；不从模型
    名称猜测窗口或输出能力。可选项的默认值由 loader 统一补齐。
    """

    name: str
    model: str
    cwd: str
    system_prompt_path: str
    additional_prompt_paths: list[str]
    tools: list[str]
    generation_parameters: dict[str, object]
    context_window: int
    max_output_tokens: int
    compaction: CompactionConfig
    max_summary_output_tokens: int
