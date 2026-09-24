"""受管理的标准库搜索进程；搜索与正则运算不会阻塞主事件循环。"""

import asyncio
import json
import sys
from pathlib import Path

from .types import ToolContext, ToolOutput

# 搜索结果行数上限；调用方可通过输出预算进一步收紧。
_MAX_RESULT_LINES = 200


async def run_search(kind: str, arguments: dict[str, object], context: ToolContext) -> ToolOutput:
    """等待搜索完成；取消时杀死并回收进程及管道后才传播取消。"""
    if context["cancel_event"].is_set():
        raise asyncio.CancelledError
    request = {
        "kind": kind,
        "arguments": arguments,
        "cwd": context["cwd"],
        "lines": min(context["max_output_lines"], _MAX_RESULT_LINES),
        "bytes": context["max_output_bytes"],
    }
    # 保护创建操作，避免取消时丢失刚启动的子进程句柄。
    spawn = asyncio.create_task(
        asyncio.create_subprocess_exec(
            sys.executable,
            "-I",  # 私有 worker 不读取任务环境的 PYTHONHOME/PYTHONPATH。
            str(Path(__file__).with_name("_search_worker.py")),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    )
    process = None
    communication = None
    watcher = None
    try:
        process = await asyncio.shield(spawn)
        communication = asyncio.create_task(
            process.communicate(json.dumps(request).encode("utf-8"))
        )
        watcher = asyncio.create_task(context["cancel_event"].wait())
        done, _ = await asyncio.wait({communication, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if communication not in done:
            raise asyncio.CancelledError
        stdout, _ = communication.result()
        if process.returncode != 0:
            raise RuntimeError(f"Search worker exited with code {process.returncode}")
        return json.loads(stdout)
    finally:
        # 执行层屏蔽调用者的重复取消，使进程清理能够完成。
        if process is None:
            process = await spawn
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        if communication is None:
            communication = asyncio.create_task(process.communicate())
        await communication
        await process.wait()
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
