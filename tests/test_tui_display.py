"""无需终端或模型的展示验收，覆盖事件归并、历史投影和有界预览。"""

import asyncio
from copy import deepcopy

import pytest
from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.utils import get_cwidth

from lhagent.harness.session import SessionRepository
from lhagent.tui import (
    DisplayReducer,
    Footer,
    PreviewLimits,
    ToolBlock,
    format_block,
    format_footer,
    preview,
    project_history,
)
from tests.samples import user_message
from tests.test_loop_tools import call, response, setup


def event(event_type, **data):
    """构造默认运行身份下的指定类型事件。"""
    return {"type": event_type, "data": {"run_id": "run-1", **data}}


def finish(reducer, result):
    """向归并器交付模型终结快照。"""
    reducer.apply(event("response_update", phase="end", call_id=result["call_id"], result=result))


def plain(block):
    """去除样式获取块的可见文本，用于内容断言。"""
    return fragment_list_to_text(format_block(block))


def tool_result(status="success", content=None):
    """构造可选状态及内容的完整工具结果。"""
    return {
        "call_id": "a",
        "name": "demo",
        "status": status,
        "error": None if status == "success" else "failure details",
        "output": {
            "content": content or [],
            "details": {},
            "is_error": status != "success",
            "truncated": False,
        },
    }


def test_stream_multiple_blocks_final_reconciliation_and_scrollback():
    """多个流块最终校准后只向滚动历史交付一次。"""
    reducer = DisplayReducer()
    reducer.apply(event("run_start"))
    reducer.add_user(user_message("你好"), "run-1")
    assert len(reducer.drain_completed()) == 1
    result = response(
        "stop",
        {"type": "reasoning", "text": "think"},
        {"type": "text", "text": "你好"},
        {"type": "text", "text": "```py\nx = 1\n```"},
    )
    for index, kind, text in [
        (0, "reasoning", "thi"),
        (0, "reasoning", "nk"),
        (1, "text", "你"),
        (1, "text", "好"),
        (2, "text", "```py\nx = 1\n```"),
    ]:
        reducer.apply(
            event(
                "response_update",
                phase="delta",
                call_id=result["call_id"],
                block_index=index,
                delta={"type": kind, "text": text},
            )
        )
    live = reducer.live_blocks()
    assert live[0].content == result["content"]
    assert reducer.drain_completed() == []
    live[0].content.clear()
    assert reducer.live_blocks()[0].content
    finish(reducer, result)
    reducer.apply(event("message_committed", entry_id="e", kind="assistant"))
    drained = reducer.drain_completed()
    assert len(drained) == 1
    assert plain(drained[0]).count("你好") == 1
    assert "```py\nx = 1\n```" in plain(drained[0])
    finish(reducer, result)
    reducer.apply(event("run_end", status="completed", error=None))
    assert reducer.drain_completed() == []
    assert reducer.state.footer("/tmp", "123456789").status == "idle"
    result["content"].clear()
    assert drained[0].content


@pytest.mark.parametrize(
    "status", ["success", "validation_error", "execution_error", "timeout", "cancelled"]
)
def test_tool_identity_serial_order_status_and_isolation(status):
    """工具身份、串行顺序、状态及结果副本互相隔离。"""
    reducer = DisplayReducer()
    finish(
        reducer, response("tool_call", call("a", arguments={"x": 1}), call("b", arguments={"x": 2}))
    )
    assert len(reducer.drain_completed()) == 1
    for tool_id in ("a", "b"):
        start = event("tool_start", tool_call_id=tool_id, tool_name="demo", arguments={"x": [1]})
        reducer.apply(start)
        start["data"]["arguments"]["x"].append(2)
        tool = reducer.tools[("run-1", tool_id)]
        assert tool.arguments == {"x": [1]}
        assert tool.status == "running"
        assert reducer.drain_completed() == []
        result = tool_result(status)
        result["call_id"] = tool_id
        reducer.apply(
            event("tool_end", tool_call_id=tool_id, tool_name="demo", status=status, result=result)
        )
        assert reducer.tools[("run-1", tool_id)] is tool
        assert [b.tool_call_id for b in reducer.drain_completed()] == [tool_id]
        assert status.replace("_", " ") in plain(tool)
        result["output"]["content"].clear()
    assert [b.tool_call_id for b in reducer.state.transcript if isinstance(b, ToolBlock)] == [
        "a",
        "b",
    ]
    assert reducer.drain_completed() == []


def test_compaction_cancel_queue_footer_and_unknown_tool():
    """压缩、取消、队列页脚与未知工具状态明确区分。"""
    reducer = DisplayReducer()
    reducer.set_queue_counts(steering=2, follow_up=1)
    reducer.apply(event("run_start"))
    reducer.apply(event("compaction_start", tokens=100))
    assert reducer.state.footer("/tmp", "abcdefghijk") == Footer(
        "/tmp", "abcdefgh", "compacting", 2, 1
    )
    reducer.apply(event("compaction_end", status="success", error=None))
    assert reducer.state.footer("/tmp", "id").status == "running"
    assert reducer.state.notifications == ["Compaction: success"]
    finish(reducer, response("tool_call", call("a")))
    reducer.request_cancel()
    assert reducer.state.footer("/tmp", "id").status == "cancelling"
    reducer.apply(event("run_end", status="cancelled", error=None))
    assert reducer.tools[("run-1", "a")].status == "unknown"
    assert reducer.tools[("run-1", "a")].result is None
    assert reducer.state.footer("/tmp", "id").status == "idle"
    assert reducer.state.steering == 2  # No queue consumption can be inferred.
    assert len(reducer.drain_completed()) == 2
    with pytest.raises(ValueError):
        reducer.set_queue_counts(steering=-1, follow_up=0)


@pytest.mark.parametrize("status", ["unchanged", "error", "cancelled"])
def test_compaction_terminal_status(status):
    """压缩终态不被误认为持久化确认。"""
    reducer = DisplayReducer()
    reducer.apply(event("compaction_start", tokens=20))
    reducer.apply(
        event("compaction_end", status=status, error="failed" if status == "error" else None)
    )
    assert reducer.state.compaction_status == status
    assert reducer.state.recent_error == ("failed" if status == "error" else None)


@pytest.mark.parametrize("reason", ["stop", "length", "error", "cancelled"])
def test_real_session_history_matches_live_transcript(tmp_path, reason):
    """真实会话历史与实时展示内容一致。"""

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path)})
        session = await repository.create()
        reducer = DisplayReducer()
        reducer.apply(event("run_start"))
        await session.start_run("run-1")
        message = user_message("task")
        reducer.add_user(message, "run-1")
        await session.append_user(message)
        first = response("tool_call", call("a", arguments={"x": 42}))
        first["call_id"] = "first"
        finish(reducer, first)
        await session.append_response(first)
        result = tool_result("timeout", [{"type": "text", "text": "partial"}])
        reducer.apply(event("tool_start", tool_call_id="a", tool_name="demo", arguments={"x": 42}))
        reducer.apply(
            event("tool_end", tool_call_id="a", tool_name="demo", status="timeout", result=result)
        )
        await session.append_tool_result(result)
        last = response(reason, {"type": "text", "text": "answer"})
        last["call_id"] = "last"
        if reason == "error":
            last["error"], last["error_kind"] = "request failed", "other"
        finish(reducer, last)
        await session.append_response(last)
        status = "completed" if reason == "stop" else reason
        reducer.apply(event("run_end", status=status, error=last["error"]))
        await session.finish_run({"status": status, "error": last["error"], "last_response": last})
        metadata = session.metadata
        await session.close()
        session = await repository.open(metadata)
        try:
            history = await session.get_display_history()
            before = deepcopy(history)
            projected = project_history(history, await session.state())
            assert projected.transcript == reducer.state.transcript
            assert [plain(b) for b in projected.transcript] == [
                plain(b) for b in reducer.state.transcript
            ]
            assert history == before
            if reason != "stop":
                assert f"[{reason}]" in plain(projected.transcript[-1])
        finally:
            await repository.close()

    asyncio.run(scenario())


def test_actual_loop_events_match_projection(tmp_path):
    """真实循环事件能形成预期展示投影。"""

    async def scenario():
        async def handler(args, context):
            return {
                "content": [
                    {"type": "tool_result", "content": {"items": [1, 2]}, "is_error": False}
                ],
                "details": {"exit_code": 0},
                "is_error": False,
                "truncated": False,
            }

        first, last = response("tool_call", call("a"), call("b")), response("stop")
        first["call_id"], last["call_id"] = "first", "last"
        session, client, events, loop = await setup(tmp_path, [first, last], handler)
        reducer = DisplayReducer()
        reducer.add_user(user_message(), "run-1")
        try:
            assert (await loop.run(user_message()))["status"] == "completed"
            for item in events:
                reducer.apply(item)
            assert (
                project_history(await session.get_display_history()).transcript
                == reducer.state.transcript
            )
        finally:
            await session.close()

    asyncio.run(scenario())


def test_compacted_history_and_interrupted_tools_are_explicit():
    """压缩外历史和中断工具有明确标记。"""
    history = {
        "summary": "earlier work",
        "entries": [
            {
                "type": "assistant",
                "entry_id": "e",
                "run_id": "old",
                "timestamp": 0,
                "in_context": False,
                "response": response("tool_call", call("a")),
            }
        ],
    }
    state = project_history(
        history,
        {
            "state": "interrupted",
            "interrupted": True,
            "active_run_id": None,
            "last_finished_run_id": None,
        },
    )
    assert "Compacted history summary:\nearlier work" in plain(state.transcript[0])
    tool = next(b for b in state.transcript if isinstance(b, ToolBlock))
    assert tool.status == "unknown" and tool.result is None
    assert "outside active context" in plain(tool)
    assert "interrupted" in plain(state.transcript[-1])
    history["entries"][0]["response"]["content"].clear()
    assert tool.arguments is not None


@pytest.mark.parametrize("width", [1, 2, 4, 12, 80])
def test_unicode_ansi_and_narrow_preview_columns(width):
    """Unicode、ANSI 和窄列宽预览不产生非法终端输出。"""
    result = preview(
        "\x1b[31m你好🙂e\u0301\x1b[0m\nnext\tline", width=width, limits=PreviewLimits(2, 20)
    )
    text = fragment_list_to_text(result.text)
    assert "\x1b" not in text
    assert all(sum(get_cwidth(c) for c in line) <= width for line in text.splitlines())
    assert result.omitted_bytes > 0
    assert str(result.omitted_bytes) in text.replace("\n", "")
    footer = fragment_list_to_text(
        format_footer(Footer("/项目", "abcdefgh", "idle", 0, 0), width=width)
    )
    assert "\n" not in footer
    assert sum(get_cwidth(c) for c in footer) <= width


def test_preview_exact_byte_line_limits_and_empty_output():
    """精确字节/行预算及空输出保留正确省略统计。"""
    result = preview("你好abc", width=80, limits=PreviewLimits(6, 4))
    assert result.omitted_bytes == 6
    assert fragment_list_to_text(result.text).startswith("你\n... (+6 UTF-8 bytes)")
    assert preview("a\nb\nc", limits=PreviewLimits(2, 99)).omitted_bytes == 2
    assert preview("", limits=PreviewLimits(1, 0)).omitted_bytes == 0
    assert preview("\x1b[31mred\x1b[0m").text[0][0].endswith("ansired")
    with pytest.raises(ValueError):
        preview("x", width=0)
    with pytest.raises(ValueError):
        PreviewLimits(0, 0)
    tool = ToolBlock("r", "a", "demo", {}, "success", tool_result(), True)
    assert "(empty output)" in plain(tool)
    tool.result["output"] = None
    assert "(no output)" in plain(tool)


def test_long_arguments_structured_outputs_details_and_truncation():
    """长参数、结构化输出和详情分别遵守预览预算。"""
    result = tool_result(
        content=[{"type": "tool_result", "content": {"items": ["中"] * 1000}, "is_error": False}]
    )
    result["output"]["details"] = {"exit_code": 0}
    result["output"]["truncated"] = True
    tool = ToolBlock("r", "a", "demo", {"path": "x" * 1000}, "success", result, True)
    before = deepcopy(tool)
    text = plain(tool)
    assert text.count("UTF-8 bytes") == 2
    assert "[tool result]" in text
    assert '"exit_code": 0' in text
    assert "original remainder unknown" in text
    assert tool == before
    assert len(text) < 2000


def test_call_ids_are_scoped_to_runs_and_empty_response():
    """调用 ID 按运行隔离，空响应不重复生成内容。"""
    reducer = DisplayReducer()
    result = response("stop")
    result["content"] = []
    finish(reducer, result)
    reducer.apply(
        {
            "type": "response_update",
            "data": {
                "run_id": "run-2",
                "call_id": result["call_id"],
                "phase": "end",
                "result": result,
            },
        }
    )
    assert len(reducer.drain_completed()) == 2
    assert all(plain(b) == "" for b in reducer.state.transcript)
