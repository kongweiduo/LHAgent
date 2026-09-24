"""展示投影保留结果详情及上下文归属，不暴露可修改的持久化记录。"""

import asyncio

from lhagent.harness.session import SessionRepository
from tests.samples import user_message
from tests.test_loop_tools import call, response


def test_display_history_new_compacted_excluded_and_interrupted(tmp_path):
    """新建、压缩、排除和中断历史均有完整且独立的展示投影。"""

    async def scenario():
        repository = SessionRepository({"directory": str(tmp_path)})
        session = await repository.create()
        metadata = session.metadata
        try:
            assert (await session.state())["state"] == "new"
            assert await session.get_display_history() == {"summary": None, "entries": []}
            assert await session.get_history() == {"summary": None, "messages": [], "entry_ids": []}
            session.metadata["id"] = "changed"
            assert session.metadata == metadata
            await session.start_run("old-run")
            user_id = await session.append_user(user_message("old instruction"))
            response_id = await session.append_response(
                response("tool_call", call("a", arguments={"x": 1}))
            )
            result = {
                "call_id": "a",
                "name": "demo",
                "status": "timeout",
                "output": {
                    "content": [{"type": "text", "text": "partial"}],
                    "details": {"exit_code": 9},
                    "truncated": True,
                    "is_error": True,
                },
                "error": "timed out",
            }
            tool_id = await session.append_tool_result(result)
            await session.omit_failed_attempt([response_id, tool_id])
            await session.commit_compaction(
                {
                    "status": "success",
                    "error": None,
                    "history": {"summary": "old summary", "messages": [], "entry_ids": []},
                    "tokens_before": 100,
                    "estimated_tokens_after": 10,
                    "usage": [],
                }
            )
            last = response("tool_call", call("unfinished", arguments={"x": 2}))
            last_id = await session.append_response(last)
            expected = await session.get_display_history()
            assert expected["summary"] == "old summary"
            assert [e["entry_id"] for e in expected["entries"]] == [
                user_id,
                response_id,
                tool_id,
                last_id,
            ]
            assert [e["in_context"] for e in expected["entries"]] == [False, False, False, True]
            assert expected["entries"][2]["result"] == result
            assert all(e["run_id"] == "old-run" for e in expected["entries"])
            assert all("id" not in e and "metadata" not in e for e in expected["entries"])
            returned = await session.get_display_history()
            returned["entries"][2]["result"]["output"]["details"].clear()
            returned["entries"][0]["message"]["content"].clear()
            returned["entries"].pop()
            state = await session.state()
            state["state"] = "closed"
            assert (await session.state())["state"] == "active"
            assert await session.get_display_history() == expected
            await session.close()
            session = await repository.open(metadata)
            assert (await session.state())["state"] == "interrupted"
            assert (await session.state())["active_run_id"] is None
            assert await session.get_display_history() == expected
            assert (await session.get_history())["entry_ids"] == [last_id]
            assert expected["entries"][-1]["response"] == last
            # 中断工具仍然只有调用，没有被伪造的结果。
            assert len([e for e in expected["entries"] if e["type"] == "tool_result"]) == 1
        finally:
            await repository.close()

    asyncio.run(scenario())
