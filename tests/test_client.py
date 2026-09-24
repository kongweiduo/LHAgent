"""客户端流累积、终结结果、工具参数和错误分类的离线回归。"""

import asyncio

from lhagent.client.client import Client
from lhagent.client.config import load_config


def request():
    """构造不含真实凭据的最小请求。"""
    return {
        "call_id": "call-1",
        "model": "test",
        "messages": [],
        "parameters": {},
    }


class FakeTransport:
    """交付混合工具/文本分片及尾部用量，记录打开和关闭行为。"""

    def __init__(self, config):
        self.closed = False
        self.opened = 0

    async def open_stream(self, request):
        self.opened += 1

        async def chunks():
            yield {
                "deltas": [
                    {
                        "type": "tool_call",
                        "tool_index": 2,
                        "data": {"call_id": "tool-2", "name": "read", "arguments_json": '{"path":'},
                    },
                    {"type": "text", "tool_index": None, "data": {"text": "hi"}},
                ],
                "usage": None,
                "finish_reason": None,
            }
            yield {
                "deltas": [
                    {"type": "tool_call", "tool_index": 2, "data": {"arguments_json": '"a.txt"}'}},
                ],
                "usage": None,
                "finish_reason": "tool_calls",
            }
            yield {
                "deltas": [],
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 4,
                    "total_tokens": 7,
                    "cache_read_tokens": None,
                    "cache_write_tokens": None,
                },
                "finish_reason": None,
            }

        return _Closable(chunks())

    async def close(self):
        self.closed = True


class _Closable:
    """包装异步迭代器并记录显式关闭，供流所有权断言使用。"""

    def __init__(self, source):
        self.source = source
        self.closed = False

    def __aiter__(self):
        return self.source.__aiter__()

    async def aclose(self):
        self.closed = True


def test_stream_accumulates_mixed_content_and_tail_usage(monkeypatch):
    """混合内容按分片累积，尾部用量仍保留在最终结果。"""
    monkeypatch.setattr("lhagent.client.client.Transport", FakeTransport)

    async def run():
        client = Client(
            load_config({"base_url": "https://example.test", "api_key": "x", "max_retries": 0})
        )
        events = [event async for event in client.stream(request())]
        assert [event["block_index"] for event in events[:-1]] == [0, 1, 0]
        result = events[-1]["result"]
        assert result["finish_reason"] == "tool_call"
        assert result["content"][0]["arguments"] == {"path": "a.txt"}
        assert result["content"][0]["complete"] is True
        assert result["stats"]["usage"]["total_tokens"] == 7
        await client.close()

    asyncio.run(run())


def test_complete_reports_interrupted_stream_as_error(monkeypatch):
    """未收到完整终态的断流必须报告错误。"""

    class BrokenTransport(FakeTransport):
        async def open_stream(self, request):
            async def chunks():
                yield {
                    "deltas": [{"type": "text", "tool_index": None, "data": {"text": "partial"}}],
                    "usage": None,
                    "finish_reason": None,
                }

            return _Closable(chunks())

    monkeypatch.setattr("lhagent.client.client.Transport", BrokenTransport)

    async def run():
        client = Client(
            load_config({"base_url": "https://example.test", "api_key": "x", "max_retries": 0})
        )
        result = await client.complete(request())
        assert result["finish_reason"] == "error"
        assert result["error_kind"] == "protocol"
        assert result["content"] == [{"type": "text", "text": "partial"}]
        await client.close()

    asyncio.run(run())
