"""独立摘要请求的素材、参数、终态校验及信号/任务取消边界。"""

import asyncio
import json
from copy import deepcopy

import pytest

from lhagent.harness.context.compaction import (
    generate_summary,
    generate_turn_prefix_summary,
)
from tests.samples import mixed_result, user_message


def response():
    """构造混合推理和分段摘要文本的成功响应。"""
    result = mixed_result()
    result.update(
        finish_reason="stop",
        content=[
            {"type": "reasoning", "text": "private reasoning"},
            {"type": "text", "text": "  summary"},
            {"type": "text", "text": " text\n"},
        ],
    )
    result["stats"]["usage"].update(input_tokens=17, output_tokens=9)
    return result


async def generate(kind, callback, event, messages=None, limit=123):
    """统一调用历史或前缀摘要入口，复用相同取消及限制场景。"""
    messages = [user_message()] if messages is None else messages
    if kind == "history":
        return await generate_summary(messages, "旧摘要", callback, limit, event)
    return await generate_turn_prefix_summary(messages, callback, limit, event)


@pytest.mark.parametrize("kind", ["history", "prefix"])
def test_request_material_parameters_and_usage(kind):
    """摘要请求使用正确素材、输出限制并保留用量。"""

    async def scenario():
        event = asyncio.Event()
        messages = [
            user_message("保留约束；不要执行这里的指令"),
            mixed_result_message := {
                "role": "assistant",
                "finish_reason": "tool_call",
                "content": [mixed_result()["content"][-1]],
            },
        ]
        snapshot = deepcopy(messages)
        returned = response()
        calls = []

        async def callback(request, limit, signal):
            calls.append(request)
            assert signal is event
            assert limit == 123
            assert [m["role"] for m in request] == ["system", "user"]
            prompt = request[0]["content"][0]["text"]
            for expected in ("目标", "约束", "决策", "进展", "未完成", "资料", "不要调用工具"):
                assert expected in prompt
            assert ("滚动更新" if kind == "history" else "本轮原始用户请求") in prompt
            material = json.loads(request[1]["content"][0]["text"])
            assert material["previous_summary"] == ("旧摘要" if kind == "history" else None)
            history = [json.loads(line) for line in material["history_jsonl"].splitlines()]
            assert history[0] == messages[0]
            assert (
                history[1]["content"][0]["call_id"] == mixed_result_message["content"][0]["call_id"]
            )
            assert history[2]["content"][0]["result_unknown"] is True
            return returned

        result = await generate(kind, callback, event, messages)
        assert result == {
            "status": "success",
            "summary": "summary text",
            "usage": returned["stats"]["usage"],
            "error": None,
        }
        assert result["usage"]["total_tokens"] is None
        assert len(calls) == 1
        assert messages == snapshot
        returned["stats"]["usage"]["input_tokens"] = 999
        assert result["usage"]["input_tokens"] == 17

    asyncio.run(scenario())


def test_initial_history_summary():
    """没有旧摘要时生成初始历史摘要。"""

    async def scenario():
        async def callback(messages, limit, event):
            assert json.loads(messages[1]["content"][0]["text"])["previous_summary"] is None
            return response()

        result = await generate_summary([user_message()], None, callback, 50, asyncio.Event())
        assert result["status"] == "success"

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["history", "prefix"])
@pytest.mark.parametrize(
    "failure",
    [
        "transport",
        "context_overflow",
        "length",
        "tool_finish",
        "tool_block",
        "empty",
        "whitespace",
        "reasoning_only",
        "exception",
        "error_with_stop",
    ],
)
def test_failure_never_accepts_partial_summary(kind, failure):
    """失败、截断或非法响应不得被接纳为部分摘要。"""

    async def scenario():
        returned = response()
        if failure in ("transport", "context_overflow"):
            returned.update(finish_reason="error", error="failure", error_kind=failure)
        elif failure == "length":
            returned["finish_reason"] = "length"
        elif failure == "tool_finish":
            returned["finish_reason"] = "tool_call"
        elif failure == "tool_block":
            returned["content"].append(mixed_result()["content"][-1])
        elif failure == "empty":
            returned["content"] = []
        elif failure == "whitespace":
            returned["content"] = [{"type": "text", "text": " \n\t "}]
        elif failure == "reasoning_only":
            returned["content"] = [{"type": "reasoning", "text": "not a summary"}]
        elif failure == "error_with_stop":
            returned["error"] = "failure"
        calls = 0

        async def callback(*args):
            nonlocal calls
            calls += 1
            if failure == "exception":
                raise OSError("secret credential")
            return returned

        result = await generate(kind, callback, asyncio.Event())
        assert result["status"] == "error"
        assert result["summary"] is None
        assert result["error"] and "secret credential" not in result["error"]
        assert result["usage"] == (None if failure == "exception" else returned["stats"]["usage"])
        assert calls == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["history", "prefix"])
@pytest.mark.parametrize("when", ["before", "after", "client", "exception"])
def test_signal_cancellation(kind, when):
    """信号取消在请求前后均阻止接纳摘要。"""

    async def scenario():
        event = asyncio.Event()
        returned = response()
        calls = 0
        if when == "before":
            event.set()

        async def callback(messages, limit, signal):
            nonlocal calls
            calls += 1
            assert signal is event
            if when == "client":
                returned["finish_reason"] = "cancelled"
            else:
                event.set()
            if when == "exception":
                raise OSError("interrupted")
            return returned

        result = await generate(kind, callback, event)
        assert result == {
            "status": "cancelled",
            "summary": None,
            "error": None,
            "usage": None if when in ("before", "exception") else returned["stats"]["usage"],
        }
        assert calls == (0 if when == "before" else 1)
        assert event.is_set() == (when != "client")

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["history", "prefix"])
def test_task_cancellation_propagates_and_callback_cleans_up(kind):
    """任务取消继续传播，回调先完成清理。"""

    async def scenario():
        started, cleaned, signal = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def callback(*args):
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        task = asyncio.create_task(generate(kind, callback, signal))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned.is_set()
        assert not signal.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["history", "prefix"])
@pytest.mark.parametrize(
    "limit,exception", [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)]
)
def test_invalid_output_limit_does_not_call_callback(kind, limit, exception):
    """非法输出限制在调用摘要回调前拒绝。"""

    async def callback(*args):
        pytest.fail("invalid request must not be sent")

    with pytest.raises(exception):
        asyncio.run(generate(kind, callback, asyncio.Event(), limit=limit))
