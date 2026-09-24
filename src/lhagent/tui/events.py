"""将公开循环事件归并为展示状态，不持有或调用 agent 资源。"""

from copy import deepcopy

from lhagent.client.types import ClientResult, ContentBlock, Message
from lhagent.harness.loop.types import LoopEvent
from lhagent.tui.models import (
    DisplayState,
    MessageBlock,
    ToolBlock,
    TranscriptBlock,
)


class DisplayReducer:
    """单消费者展示状态及按顺序推进的滚动输出游标。

    界面原位刷新 live_blocks，将 drain_completed 写入滚动历史；两者均返回副本。
    最终响应替换累计快照，不再追加第二条 assistant 消息。"""

    def __init__(self, state: DisplayState | None = None) -> None:
        """复制初始展示状态并创建按运行身份隔离的响应、工具索引。"""
        self.state = deepcopy(state) if state is not None else DisplayState()
        self._cursor = 0
        self._responses: dict[tuple[str, str], MessageBlock] = {}
        self._parts: dict[tuple[str, str], dict[int, ContentBlock]] = {}
        self.tools: dict[tuple[str, str], ToolBlock] = {}

    def add_user(self, message: Message, run_id: str) -> None:
        """由输入所有者在消息实际被接纳时调用一次。"""
        self.state.transcript.append(
            MessageBlock(
                "user",
                run_id,
                deepcopy(message["content"]),
                complete=True,
            )
        )

    def set_queue_counts(self, *, steering: int, follow_up: int) -> None:
        """使用本地显式计数；提交事件本身不能识别排队输入。"""
        if steering < 0 or follow_up < 0:
            raise ValueError("queue counts must be non-negative")
        self.state.steering, self.state.follow_up = steering, follow_up

    def request_cancel(self) -> None:
        """仅标记正在运行的展示为取消中，不调用 agent。"""
        if self.state.run_status == "running":
            self.state.cancelling = True

    def drain_completed(self) -> list[TranscriptBlock]:
        """取出游标后连续完成的块并推进游标，返回独立副本。"""
        start = self._cursor
        while (
            self._cursor < len(self.state.transcript)
            and self.state.transcript[self._cursor].complete
        ):
            self._cursor += 1
        return deepcopy(self.state.transcript[start : self._cursor])

    def live_blocks(self) -> list[TranscriptBlock]:
        """返回待替换尾部副本，包括被活动块挡住的已完成块。"""
        return deepcopy(self.state.transcript[self._cursor :])

    def _response(self, run_id: str, call_id: str) -> MessageBlock:
        """按 run_id 和 call_id 获取或创建响应展示块。"""
        key = (run_id, call_id)
        if key not in self._responses:
            block = MessageBlock("assistant", run_id, call_id=call_id)
            self._responses[key] = block
            self._parts[key] = {}
            self.state.transcript.append(block)
        return self._responses[key]

    def _tool(self, run_id: str, tool_id: str, name: str) -> ToolBlock:
        """按运行和工具调用身份获取或创建展示块。"""
        key = (run_id, tool_id)
        if key not in self.tools:
            self.tools[key] = ToolBlock(run_id, tool_id, name)
            self.state.transcript.append(self.tools[key])
        return self.tools[key]

    def finish_response(self, run_id: str, result: ClientResult) -> MessageBlock:
        """以最终结果校准响应和工具参数，供实时事件与历史投影共用。"""
        block = self._response(run_id, result["call_id"])
        if block.complete:
            return block
        block.content = deepcopy(result["content"])
        block.status, block.error = result["finish_reason"], result["error"]
        block.complete = True
        self._parts.pop((run_id, result["call_id"]), None)
        if block.error:
            self.state.recent_error = block.error
        for part in block.content:
            if part["type"] == "tool_call":
                tool = self._tool(run_id, part["call_id"], part["name"])
                tool.arguments = deepcopy(part.get("arguments", part.get("arguments_json")))
        return block

    def apply(self, event: LoopEvent) -> None:
        # 仅深拷贝需保留的数据，不修改订阅事件载荷。
        """归并一个公开循环事件；不修改载荷，不以提交事件重复添加消息。"""
        run_id = event["data"]["run_id"]
        if event["type"] == "run_start":
            self.state.run_id, self.state.run_status = run_id, "running"
            self.state.cancelling = False
            self.state.compaction_status = None
            self.state.compaction_tokens = None
            self.state.recent_error = None
        elif event["type"] == "response_update":
            response = event["data"]
            if response["phase"] == "end":
                self.finish_response(run_id, response["result"])
                return
            key = (run_id, response["call_id"])
            block = self._response(*key)
            if block.complete:
                return
            delta = response["delta"]
            index = response["block_index"]
            kind = delta.get("type")
            if kind not in ("text", "reasoning"):
                # 工具参数以最终响应或工具开始事件为准，不能执行或展示残缺参数片段。
                return
            parts = self._parts[key]
            if index not in parts:
                if kind == "text":
                    parts[index] = {"type": "text", "text": ""}
                else:
                    parts[index] = {"type": "reasoning", "text": ""}
            part = parts[index]
            if part["type"] in ("text", "reasoning"):
                text = delta.get("text", "")
                if isinstance(text, str):
                    part["text"] += text
            block.content = [parts[i] for i in sorted(parts)]
        elif event["type"] == "tool_start":
            start = event["data"]
            tool = self._tool(run_id, start["tool_call_id"], start["tool_name"])
            if not tool.complete:
                tool.arguments = deepcopy(start["arguments"])
                tool.status = "running"
        elif event["type"] == "tool_end":
            end = event["data"]
            tool = self._tool(run_id, end["tool_call_id"], end["tool_name"])
            if not tool.complete:
                tool.status = end["status"]
                tool.result = deepcopy(end["result"])
                tool.complete = True
                if tool.result["error"]:
                    self.state.recent_error = tool.result["error"]
        elif event["type"] == "compaction_start":
            self.state.compaction_status = "running"
            self.state.compaction_tokens = event["data"]["tokens"]
        elif event["type"] == "compaction_end":
            compact = event["data"]
            self.state.compaction_status = compact["status"]
            # 摘要生成成功不代表会话已经持久化，展示不得混淆两者。
            self.state.notifications.append(f"Compaction: {compact['status']}")
            if compact["error"]:
                self.state.recent_error = compact["error"]
        elif event["type"] == "run_end":
            end_run = event["data"]
            self.state.run_status = end_run["status"]
            self.state.cancelling = False
            if self.state.compaction_status == "running":
                self.state.compaction_status = "unknown"
            if end_run["error"]:
                self.state.recent_error = end_run["error"]
            self.state.notifications.append(f"Run: {end_run['status']}")
            for pending in self.state.transcript:
                if (
                    isinstance(pending, (MessageBlock, ToolBlock))
                    and pending.run_id == run_id
                    and not pending.complete
                ):
                    # 没有 tool_end 就没有已确认结果，取消也不能伪造执行终态。
                    pending.status = "unknown"
                    pending.complete = True
        # message_committed 只确认持久化，不能重复追加展示内容。
