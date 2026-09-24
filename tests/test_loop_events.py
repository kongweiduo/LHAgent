"""循环事件执行顺序、独立副本及订阅者隔离。"""

import asyncio
from copy import deepcopy

import pytest

from lhagent.agents.coding import CodingAgent
from lhagent.harness.session import SessionRepository
from tests.samples import user_message
from tests.test_coding_agent import Client, config
from tests.test_loop_basic import make_result
from tests.test_loop_tools import call, response, setup


@pytest.mark.parametrize(
    "status", ["success", "validation_error", "execution_error", "timeout", "cancelled"]
)
def test_tool_events_are_complete_ordered_and_isolated(tmp_path, status):
    """工具事件完整、有序且副本互相隔离。"""

    async def scenario():
        invoked = []

        async def handler(args, context):
            invoked.append(deepcopy(args))
            if status == "execution_error":
                raise ValueError("handler failed")
            if status == "cancelled":
                context["cancel_event"].set()
                await asyncio.Event().wait()
            return {
                "content": [{"type": "text", "text": "original"}],
                "details": {"nested": [1]},
                "is_error": False,
                "truncated": False,
            }

        arguments = {
            "x": "invalid" if status == "validation_error" else 7,
            "nested": {"items": [1]},
        }
        session, client, events, loop = await setup(
            tmp_path,
            [
                response("tool_call", call("a", arguments=arguments)),
                response("stop"),
            ],
            handler,
        )
        if status == "timeout":
            loop._tool_context["timeout_seconds"] = 0
        original = loop._emit

        async def mutate(event):
            await original(deepcopy(event))
            if event["type"] == "tool_start":
                event["data"]["arguments"]["x"] = 999
                event["data"]["arguments"]["nested"]["items"].append(2)
            elif event["type"] == "tool_end":
                event["data"]["result"]["name"] = "corrupted"
                output = event["data"]["result"]["output"]
                if output:
                    output["content"].clear()
                    output["details"].clear()
            elif event["type"] == "response_update":
                event["data"]["result"]["content"].clear()

        loop._emit = mutate
        try:
            outcome = await loop.run(user_message())
            assert outcome["status"] == ("cancelled" if status == "cancelled" else "completed")
            starts = [e["data"] for e in events if e["type"] == "tool_start"]
            ends = [e["data"] for e in events if e["type"] == "tool_end"]
            assert starts == [
                {
                    "run_id": "run-1",
                    "tool_call_id": "a",
                    "tool_name": "demo",
                    "arguments": arguments,
                }
            ]
            assert len(ends) == 1 and ends[0]["status"] == status
            assert ends[0]["tool_name"] == "demo"
            assert ends[0]["result"]["call_id"] == "a"
            assert invoked == ([] if status in ("validation_error", "timeout") else [arguments])
            entries = (await session.get_display_history())["entries"]
            saved = next(e["result"] for e in entries if e["type"] == "tool_result")
            assert saved == ends[0]["result"]
            assert entries[1]["response"]["content"][0]["arguments"] == arguments
            kinds = [e["type"] for e in events]
            index = kinds.index("tool_start")
            assert kinds[index : index + 3] == ["tool_start", "message_committed", "tool_end"]
            assert kinds[-1] == "run_end"
        finally:
            await session.close()

    asyncio.run(scenario())


def test_agent_listeners_receive_independent_events(tmp_path):
    """每个 Agent 监听器收到独立事件副本。"""

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path)})
        session = await repository.create()
        agent = CodingAgent(
            {"config": config(tmp_path), "client": Client([make_result()]), "session": session}
        )
        received = []

        async def corrupt(event):
            event["type"] = "corrupted"
            event["data"].clear()

        async def observe(event):
            received.append(event)

        agent.subscribe(corrupt)
        agent.subscribe(observe)
        try:
            assert (await agent.run("hello"))["status"] == "completed"
            assert [e["type"] for e in received] == [
                "run_start",
                "message_committed",
                "response_update",
                "message_committed",
                "run_end",
            ]
            assert (
                received[2]["data"]["result"]
                == (await session.get_display_history())["entries"][1]["response"]
            )
            assert all(e["data"]["run_id"] for e in received)
        finally:
            await agent.close()
            await repository.close()

    asyncio.run(scenario())
