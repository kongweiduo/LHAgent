"""从会话公开展示历史生成界面投影，不直接读取存储记录。"""

from lhagent.harness.session.types import DisplayHistory, SessionStateInfo
from lhagent.tui.events import DisplayReducer
from lhagent.tui.models import DisplayState, MessageBlock, NoticeBlock, ToolBlock


def project_history(
    history: DisplayHistory,
    session_state: SessionStateInfo | None = None,
) -> DisplayState:
    """将公开历史和会话状态投影为展示，明确标注压缩外内容及中断工具。"""
    reducer = DisplayReducer()
    if history["summary"] is not None:
        reducer.state.transcript.append(
            NoticeBlock(
                "Compacted history summary:\n" + history["summary"],
                "muted",
            )
        )
    for entry in history["entries"]:
        run_id = entry["run_id"]
        if entry["type"] == "user":
            reducer.add_user(entry["message"], run_id)
            block = reducer.state.transcript[-1]
            assert isinstance(block, MessageBlock)
            block.in_context = entry["in_context"]
        elif entry["type"] == "assistant":
            assistant = reducer.finish_response(run_id, entry["response"])
            assistant.in_context = entry["in_context"]
            for part in assistant.content:
                if part["type"] == "tool_call":
                    reducer.tools[(run_id, part["call_id"])].in_context = entry["in_context"]
        else:
            result = entry["result"]
            reducer.apply(
                {
                    "type": "tool_end",
                    "data": {
                        "run_id": run_id,
                        "tool_call_id": result["call_id"],
                        "tool_name": result["name"],
                        "status": result["status"],
                        "result": result,
                    },
                }
            )
            reducer.tools[(run_id, result["call_id"])].in_context = entry["in_context"]
    for block in reducer.state.transcript:
        if isinstance(block, ToolBlock) and not block.complete:
            block.status, block.complete = "unknown", True
    if session_state and session_state["interrupted"]:
        reducer.state.transcript.append(
            NoticeBlock(
                "Session contains an interrupted run; unfinished tool results and tool side effects are unknown.",
                "warning",
            )
        )
    return reducer.state
