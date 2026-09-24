"""获取调用方指定的提示词素材，不主动检索资料或接收终端输入。

用户消息、模型消息和工具结果由外部传入；原始会话的读取与持久化归会话层。
本文件只读取显式指定的 UTF-8 文件；不进行自动发现。
"""

from pathlib import Path

from .types import PromptBundle, PromptSources


def read_prompt_file(path: str) -> str:
    """读取指定提示词文件并返回文本，读取失败时报告明确的本地异常。

    不遍历项目、不隐式寻找其他配置文件，也不将读取失败伪装成空提示词。
    """
    return Path(path).read_text(encoding="utf-8")


def load_prompts(sources: PromptSources) -> PromptBundle:
    """按指定来源和顺序加载系统提示词及固定附加提示词。

    文件素材统一映射为 system 消息；其他角色的附加消息由外部直接提供。
    不加载会话、不发现技能、不执行工具，不替调用方选择 agent。
    """
    system_path = sources["system_prompt_path"]
    return {
        "system_prompt": read_prompt_file(system_path) if system_path is not None else None,
        "additional_messages": [
            {"role": "system", "content": [{"type": "text", "text": read_prompt_file(path)}]}
            for path in sources["additional_prompt_paths"]
        ],
    }
