"""通过真实 SDK 和模拟 HTTP 验证单次传输及响应释放，无需网络或密钥。"""

import asyncio
import json
from copy import deepcopy

import httpx2
import openai
import pytest

from lhagent.client.config import load_config
from lhagent.client.errors import ProtocolError, classify_error, error_code, error_status
from lhagent.client.transport import Transport


class Body(httpx2.AsyncByteStream):
    """模拟 HTTP 字节流，可阻塞或抛错并记录关闭状态。"""

    def __init__(self, frames=(), failure=None, wait=False):
        self.frames = frames
        self.failure = failure
        self.wait = wait
        self.reading = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        for frame in self.frames:
            yield frame
        self.reading.set()
        if self.wait:
            await asyncio.Event().wait()
        if self.failure:
            raise self.failure

    async def aclose(self):
        self.closed = True


def frame(data):
    """将 JSON 数据编码成 SSE 帧。"""
    return ("data: " + json.dumps(data) + "\n\n").encode()


def request():
    """构造包含文本与输出限制的协议映射样本。"""
    return {
        "call_id": "local",
        "model": "test",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "parameters": {"max_output_tokens": 10, "temperature": 0.5},
    }


def setup(monkeypatch, body, status=200):
    """为真实 SDK 注入模拟 HTTP，返回传输及捕获请求列表。"""
    calls = []

    async def handler(req):
        calls.append(req)
        return httpx2.Response(
            status,
            stream=body,
            headers={
                "content-type": "text/event-stream" if status == 200 else "application/json",
                "retry-after": "3",
            },
        )

    sdk_class = openai.AsyncOpenAI

    def factory(**kwargs):
        assert kwargs["max_retries"] == 0
        assert kwargs["timeout"] == 60
        return sdk_class(
            **kwargs, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
        )

    monkeypatch.setattr("lhagent.client.transport.AsyncOpenAI", factory)
    transport = Transport(load_config({"base_url": "https://example.test/v1/", "api_key": "fake"}))
    return transport, calls


def test_wrapped_tool_definition_fails_before_network(monkeypatch):
    """工具定义只接受内部格式，拒绝预先包装的协议格式。"""

    async def run():
        transport, calls = setup(monkeypatch, Body())
        req = request()
        req["parameters"]["tools"] = [
            {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}
        ]
        try:
            with pytest.raises(ValueError, match="tool function requires a name"):
                await transport.open_stream(req)
            assert not calls
        finally:
            await transport.close()

    asyncio.run(run())


def test_mapping_and_complete_stream(monkeypatch):
    """请求映射和完整响应流通过真实 SDK 与模拟 HTTP 验证。"""

    async def run():
        body = Body(
            [
                frame(
                    {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]}
                ),
                frame({"choices": [], "usage": {"completion_tokens": 2}}),
                b"data: [DONE]\n\n",
            ]
        )
        transport, calls = setup(monkeypatch, body)
        req = request()
        req["messages"] += [
            {
                "role": "assistant",
                "call_id": "private",
                "finish_reason": "tool_call",
                "content": [
                    {"type": "reasoning", "text": "think"},
                    {
                        "type": "tool_call",
                        "call_id": "a",
                        "name": "read",
                        "complete": True,
                        "arguments": {"path": "中文"},
                    },
                    {
                        "type": "tool_call",
                        "call_id": "b",
                        "name": "ls",
                        "complete": True,
                        "arguments_json": "{}",
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "a",
                "content": [{"type": "tool_result", "content": {"ok": True}, "is_error": False}],
            },
        ]
        req["parameters"]["tools"] = [{"name": "read", "parameters": {"type": "object"}}]
        original = deepcopy(req)
        stream = await transport.open_stream(req)
        assert len(calls) == 1 and not body.reading.is_set()
        chunks = [chunk async for chunk in stream]
        assert chunks[0]["deltas"][0]["data"] == {"text": "hi"}
        assert chunks[1]["usage"]["output_tokens"] == 2
        assert body.closed
        sent = json.loads(calls[0].content)
        assert str(calls[0].url) == "https://example.test/v1/chat/completions"
        assert calls[0].headers["authorization"] == "Bearer fake"
        assert sent["stream"] is True and sent["n"] == 1
        assert sent["stream_options"] == {"include_usage": True}
        assert sent["max_completion_tokens"] == 10 and "max_output_tokens" not in sent
        assert sent["tools"] == [{"type": "function", "function": req["parameters"]["tools"][0]}]
        assert sent["messages"][1] == {
            "role": "assistant",
            "content": "",
            "reasoning_content": "think",
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"中文"}'},
                },
                {"id": "b", "type": "function", "function": {"name": "ls", "arguments": "{}"}},
            ],
        }
        assert sent["messages"][2] == {
            "role": "tool",
            "tool_call_id": "a",
            "content": '{"ok":true}',
        }
        assert req == original and "call_id" not in sent
        await transport.close()
        await transport.close()
        with pytest.raises(RuntimeError):
            await transport.open_stream(req)

    asyncio.run(run())


@pytest.mark.parametrize(
    "status,code,kind",
    [
        (429, "rate_limit_exceeded", "rate_limit"),
        (400, "context_length_exceeded", "context_overflow"),
        (500, "server_error", "other"),
    ],
)
def test_establishment_error_preserves_details_without_retry(monkeypatch, status, code, kind):
    """建立失败保留错误详情且传输层不自行重试。"""

    async def run():
        body = Body([json.dumps({"error": {"code": code, "message": "failed"}}).encode()])
        transport, calls = setup(monkeypatch, body, status)
        with pytest.raises(openai.APIStatusError) as caught:
            await transport.open_stream(request())
        assert len(calls) == 1 and body.closed
        assert error_code(caught.value) == code
        assert error_status(caught.value) == status
        assert caught.value.response.headers["retry-after"] == "3"
        assert classify_error(caught.value) == kind
        await transport.close()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["network", "json", "protocol", "service"])
def test_iteration_errors_release_response(monkeypatch, mode):
    """迭代异常关闭已取得的响应。"""

    async def run():
        body = Body(
            {
                "json": [b"data: {broken\n\n"],
                "protocol": [frame({})],
                "service": [
                    frame({"error": {"code": "context_length_exceeded", "message": "failed"}})
                ],
            }.get(mode, []),
            failure=httpx2.ReadError("broken") if mode == "network" else None,
        )
        transport, calls = setup(monkeypatch, body)
        stream = await transport.open_stream(request())
        with pytest.raises((openai.APIError, ProtocolError)) as caught:
            await anext(stream)
        assert len(calls) == 1 and body.closed
        assert (
            classify_error(caught.value)
            == {
                "network": "transport",
                "json": "protocol",
                "protocol": "protocol",
                "service": "context_overflow",
            }[mode]
        )
        if mode == "service":
            assert caught.value.response.headers["retry-after"] == "3"
        await transport.close()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["unread", "partial", "cancel", "transport"])
def test_early_release(monkeypatch, mode):
    """提前停止消费也可显式释放响应。"""

    async def run():
        body = Body([frame({"choices": []})], wait=True)
        transport, _ = setup(monkeypatch, body)
        stream = await transport.open_stream(request())
        if mode in ("partial", "cancel"):
            await anext(stream)
        if mode == "cancel":
            task = asyncio.create_task(anext(stream))
            await body.reading.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif mode == "transport":
            await transport.close()
        else:
            await stream.aclose()
        assert body.closed
        await stream.aclose()
        await transport.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "base_url",
        "extra_body",
        "extra_headers",
        "extra_query",
        "timeout",
        "stream",
        "stream_options",
        "n",
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
    ],
)
def test_reserved_parameters_fail_before_network(monkeypatch, key):
    """保留参数在发送网络请求前被拒绝。"""

    async def run():
        transport, calls = setup(monkeypatch, Body())
        req = request()
        req["parameters"][key] = "secret"
        with pytest.raises(ValueError) as caught:
            await transport.open_stream(req)
        assert "secret" not in str(caught.value) and not calls
        await transport.close()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["connect", "cancel", "close"])
def test_pending_establishment(monkeypatch, mode):
    """建立尚未完成时的关闭/取消不遗失响应句柄。"""

    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        body = Body()
        calls = []

        async def handler(req):
            calls.append(req)
            started.set()
            if mode == "connect":
                raise httpx2.ConnectError("unavailable", request=req)
            await release.wait()
            return httpx2.Response(200, stream=body)

        sdk_class = openai.AsyncOpenAI
        monkeypatch.setattr(
            "lhagent.client.transport.AsyncOpenAI",
            lambda **kwargs: sdk_class(
                **kwargs, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
            ),
        )
        transport = Transport(
            load_config({"base_url": "https://example.test/v1", "api_key": "fake"})
        )
        task = asyncio.create_task(transport.open_stream(request()))
        await started.wait()
        if mode == "connect":
            with pytest.raises(openai.APIConnectionError):
                await task
        elif mode == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await transport.close()
            release.set()
            with pytest.raises(RuntimeError, match="closed"):
                await task
            assert body.closed
        assert len(calls) == 1
        await transport.close()
        assert transport._client.is_closed()

    asyncio.run(run())
