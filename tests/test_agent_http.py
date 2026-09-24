"""联合验证代理、SDK 请求协议、工具执行与历史持久化。"""

import asyncio
import json

import httpx2
import openai
import pytest

from lhagent.agents import create_coding_agent
from lhagent.client import Client, load_config
from lhagent.harness.session import SessionRepository
from tests.test_client_transport_stream import frame
from tests.test_coding_agent import config


@pytest.mark.parametrize("reject", [False, True])
def test_agent_http_tools_and_persisted_result(tmp_path, monkeypatch, reject):
    """通过真实 SDK 和模拟 HTTP 验证工具调用与成功、失败记录的恢复。"""
    secret = "fake-explicit-api-key"
    requests = []
    (tmp_path / "source.txt").write_text("from disk", encoding="utf-8")

    async def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["tools"][0]["type"] == "function"
        assert body["tools"][0]["function"]["name"] == "read"
        assert body["tools"][0]["function"]["parameters"]["required"] == ["path"]
        if reject:
            return httpx2.Response(
                400,
                json={
                    "error": {
                        "code": "InvalidParameter",
                        "param": "tools[0].type",
                        "message": f"type must be function; {secret}; Authorization: Bearer other-secret",
                    }
                },
            )
        if len(requests) == 1:
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "read-1",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path":"source.txt"}'},
                    }
                ]
            }
            reason = "tool_calls"
        else:
            assert body["messages"][-1]["role"] == "tool"
            assert body["messages"][-1]["tool_call_id"] == "read-1"
            assert "from disk" in body["messages"][-1]["content"]
            delta, reason = {"content": "read done"}, "stop"
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=frame({"choices": [{"index": 0, "delta": delta, "finish_reason": reason}]})
            + b"data: [DONE]\n\n",
        )

    sdk = openai.AsyncOpenAI
    monkeypatch.setattr(
        "lhagent.client.transport.AsyncOpenAI",
        lambda **kwargs: sdk(
            **kwargs, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
        ),
    )

    async def scenario():
        client = Client(load_config({"base_url": "https://example.test/v1", "api_key": secret}))
        repo = SessionRepository({"directory": str(tmp_path / "sessions")})
        session = await repo.create()
        metadata = session.metadata
        agent = create_coding_agent(
            {"config": config(tmp_path), "client": client, "session": session}
        )
        try:
            result = await agent.run("read source.txt")
            assert result["status"] == ("error" if reject else "completed")
            if reject:
                assert "HTTP 400" in result["error"]
                assert "tools[0].type" in result["error"]
                assert "type must be function" in result["error"]
                assert secret not in result["error"] and "other-secret" not in result["error"]
        finally:
            await agent.close()
            await repo.close()
            await client.close()
        repo = SessionRepository({"directory": str(tmp_path / "sessions")})
        try:
            restored = await repo.open(metadata)
            history = await restored.get_display_history()
            assert [entry["type"] for entry in history["entries"]] == (
                ["user", "assistant"]
                if reject
                else ["user", "assistant", "tool_result", "assistant"]
            )
            assert history["entries"][-1]["response"]["finish_reason"] == (
                "error" if reject else "stop"
            )
            assert (await restored.state())["state"] == "idle"
        finally:
            await repo.close()
        raw = (tmp_path / "sessions" / f"{metadata['id']}.jsonl").read_text()
        assert secret not in raw and "other-secret" not in raw
        assert json.loads(raw.splitlines()[-1])["status"] == ("error" if reject else "completed")

    asyncio.run(scenario())
