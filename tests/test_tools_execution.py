"""单次工具执行、输出契约及控制异常传播，不依赖真实进程或网络。"""

import asyncio
import math
from copy import deepcopy

import pytest

from lhagent.harness.tools.execution import execute_prepared_call, execute_tool_call
from lhagent.harness.tools.results import limit_output, result_from_output, validate_output
from lhagent.harness.tools.types import ToolStopError


def output(text="ok", **overrides):
    """构造可覆盖字段的合法文本工具输出。"""
    return {
        "content": [{"type": "text", "text": text}],
        "details": {"exit_code": 0},
        "is_error": False,
        "truncated": False,
        **overrides,
    }


def context(lines=100, size=1000):
    """构造独立取消信号及可调行/字节预算的工具环境。"""
    return {
        "cwd": ".",
        "timeout_seconds": 10,
        "cancel_event": asyncio.Event(),
        "max_output_lines": lines,
        "max_output_bytes": size,
    }


def call(arguments=None, name="fake"):
    """构造待执行调用，可替换参数或名称。"""
    return {"call_id": "tool-12", "name": name, "arguments": {} if arguments is None else arguments}


def tool(handler):
    """为给定处理器定义只接受空对象的假工具。"""
    return {
        "name": "fake",
        "description": "Test tool",
        "parameters": {"type": "object", "additionalProperties": False},
        "handler": handler,
    }


@pytest.mark.parametrize(
    "value",
    [
        None,
        "ok",
        {},
        output(content={}),
        output(details=[]),
        output(is_error=1),
        output(truncated="false"),
        output(content=["text"]),
        output(content=[{"type": "unknown"}]),
        output(content=[{"type": "text", "text": 1}]),
        output(content=[{"type": "reasoning"}]),
        output(content=[{"type": "tool_call", "call_id": "id", "name": "x", "complete": 1}]),
        output(
            content=[
                {
                    "type": "tool_call",
                    "call_id": "id",
                    "name": "x",
                    "complete": True,
                    "arguments": [],
                }
            ]
        ),
        output(content=[{"type": "tool_result", "is_error": False}]),
        output(content=[{"type": "tool_result", "is_error": "false", "content": None}]),
        output(details={"number": math.nan}),
        output(details={"number": math.inf}),
        output(details={1: "value"}),
        output(details={"value": object()}),
        output("\ud800"),
    ],
)
def test_invalid_outputs(value):
    """非法工具输出不满足返回契约。"""
    with pytest.raises(ValueError):
        validate_output(value)


def test_required_fields_and_circular_values():
    """缺失字段和循环引用不能进入序列化结果。"""
    for field in output():
        value = output()
        del value[field]
        with pytest.raises(ValueError):
            validate_output(value)
    value = output()
    value["details"]["self"] = value
    with pytest.raises(ValueError, match="circular"):
        validate_output(value)


def test_all_content_kinds_and_structured_retention():
    """所有内容类别及结构化载荷都保留。"""
    value = output(
        content=[
            {"type": "text", "text": "你好"},
            {"type": "reasoning", "text": "thought"},
            {
                "type": "tool_call",
                "call_id": "nested",
                "name": "fake",
                "complete": False,
                "arguments": {},
                "arguments_json": "{",
            },
            {"type": "tool_result", "content": {"nested": [1, None, True]}, "is_error": False},
        ]
    )
    original = deepcopy(value)
    assert validate_output(value) is value
    limited = limit_output(value, context(lines=0, size=0))
    assert limited["content"][:2] == [
        {"type": "text", "text": ""},
        {"type": "reasoning", "text": ""},
    ]
    assert limited["content"][2:] == original["content"][2:]
    assert limited["details"] == original["details"]
    assert limited["truncated"] is True
    limited["content"][3]["content"]["nested"].append(2)
    assert value == original


@pytest.mark.parametrize(
    "text,lines,size,expected,truncated",
    [
        ("a\nb\nc", 2, 100, "a\nb\n", True),
        ("a\r\nb\r\n", 1, 100, "a\r\n", True),
        ("你好🙂x", 10, 8, "你好", True),
        ("你好🙂x", 10, 10, "你好🙂", True),
        ("你好🙂x", 10, 11, "你好🙂x", False),
        ("你好", 10, 1, "", True),
        ("a\nb", 1, 1, "a", True),
        ("", 0, 0, "", False),
        ("x", 0, 100, "", True),
        ("x", 10, 0, "", True),
        ("a\n", 1, 2, "a\n", False),
    ],
)
def test_output_boundaries(text, lines, size, expected, truncated):
    """精确字节/行数边界使用正确截断标记。"""
    value = output(text)
    limited = limit_output(value, context(lines, size))
    assert limited["content"][0]["text"] == expected
    assert limited["truncated"] is truncated
    assert limited["is_error"] is False
    assert value == output(text)
    assert limit_output(limited, context(lines, size)) == limited


def test_cumulative_limits_and_prefix_after_unicode_cut():
    """多个内容块共用累计预算，Unicode 截断保留合法前缀。"""
    value = output(
        content=[
            {"type": "text", "text": "a\n"},
            {"type": "reasoning", "text": "你好"},
            {"type": "text", "text": "z"},
        ]
    )
    limited = limit_output(value, context(lines=3, size=6))
    assert [block["text"] for block in limited["content"]] == ["a\n", "你", ""]
    limited = limit_output(value, context(lines=1))
    assert [block["text"] for block in limited["content"]] == ["a\n", "", ""]
    assert limit_output(output(truncated=True), context())["truncated"] is True


@pytest.mark.parametrize("field", ["max_output_lines", "max_output_bytes"])
@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_limits(field, value):
    """非法输出预算在执行前拒绝。"""
    env = context()
    env[field] = value
    with pytest.raises(ValueError, match=field):
        limit_output(output(), env)


@pytest.mark.parametrize("prepared_entry", [False, True])
@pytest.mark.parametrize("mode", ["success", "failure", "exception", "invalid"])
def test_execution_once_and_error_classification(prepared_entry, mode):
    """处理器只执行一次，错误按统一结果分类。"""
    invocations = []
    value = output("long text", is_error=mode == "failure", details={"exit_code": 1})

    async def handler(arguments, env):
        invocations.append((arguments, env))
        if mode == "exception":
            raise RuntimeError("broken")
        return {} if mode == "invalid" else value

    request, definition, env = call(), tool(handler), context(size=4)
    if prepared_entry:
        coro = execute_prepared_call({"call": request, "tool": definition, "arguments": {}}, env)
    else:
        coro = execute_tool_call(request, [definition], env)
    result = asyncio.run(coro)
    assert invocations == [({}, env)]
    assert result["call_id"] == request["call_id"]
    assert result["name"] == request["name"]
    assert result["status"] == ("success" if mode == "success" else "execution_error")
    if mode in ("success", "failure"):
        assert result["output"]["truncated"] is True
        assert result["output"]["content"][0]["text"] == "long"
        assert result["output"]["details"] == {"exit_code": 1}
    else:
        assert result["output"] is None
    assert (result["error"] is None) == (mode == "success")


@pytest.mark.parametrize("tool_call", [call(name="missing"), call({"extra": 1}), call("{}")])
def test_validation_failure_never_executes(tool_call):
    """校验失败不得调用处理器。"""

    async def handler(arguments, env):
        pytest.fail("invalid call executed")

    result = asyncio.run(execute_tool_call(tool_call, [tool(handler)], context()))
    assert result["status"] == "validation_error"
    assert result["call_id"] == tool_call["call_id"]
    assert result["name"] == tool_call["name"]
    assert result["output"] is None
    assert result["error"]


@pytest.mark.parametrize("kind", ["stop", "cancel"])
def test_control_exceptions_propagate_unchanged(kind):
    """控制异常保持原有传播语义。"""
    request, partial = call(), output("partial")
    error = (
        ToolStopError(request, "stop unconfirmed", partial)
        if kind == "stop"
        else asyncio.CancelledError()
    )

    async def handler(arguments, env):
        raise error

    with pytest.raises(type(error)) as caught:
        asyncio.run(execute_tool_call(request, [tool(handler)], context()))
    assert caught.value is error
    if kind == "stop":
        assert error.call is request
        assert error.output is partial
        assert str(error) == "stop unconfirmed"
        assert ToolStopError(request, "no output").output is None


def test_external_task_cancellation_propagates_and_runs_handler_finally():
    """外部任务取消等待处理器 finally 执行后传播。"""

    async def scenario():
        started, cleaned = asyncio.Event(), asyncio.Event()

        async def handler(arguments, env):
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        task = asyncio.create_task(execute_tool_call(call(), [tool(handler)], context()))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned.is_set()

    asyncio.run(scenario())


def test_explicit_error_without_text_has_error_description():
    """没有文本的显式错误仍生成可读错误说明。"""
    result = result_from_output(call(), output(content=[], is_error=True))
    assert result["status"] == "execution_error"
    assert result["error"]
