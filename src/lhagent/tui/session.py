"""交互应用的会话绑定；仓库拥有会话，agent 借用会话。"""

from collections.abc import Callable
from pathlib import Path

from lhagent.agents.coding import CodingAgent, create_coding_agent
from lhagent.harness.configs.types import CodingAgentConfig
from lhagent.harness.loop.types import EventSink
from lhagent.harness.session import SessionRepository
from lhagent.harness.session.session import Session
from lhagent.harness.session.types import SessionMetadata
from lhagent.tui.history import project_history
from lhagent.tui.models import DisplayState


class SessionCoordinator:
    """拥有仓库和 agent；agent 仅借用会话，由协调器负责释放绑定。"""

    def __init__(
        self,
        config: CodingAgentConfig,
        *,
        repository: SessionRepository | None = None,
        agent_factory: Callable[..., CodingAgent] = create_coding_agent,
    ) -> None:
        """建立资源归属；注入仓库同样随协调器关闭。"""
        self.config = config
        self.repository = repository if repository is not None else SessionRepository({})
        self.agent_factory = agent_factory
        self.session: Session | None = None
        self.agent: CodingAgent | None = None
        self._unsubscribe: Callable[[], None] | None = None

    async def start(self, path: str | None, listener: EventSink) -> DisplayState:
        """默认新建会话；给定路径须先在仓库扫描结果中匹配。"""
        metadata = None
        if path is not None:
            target = Path(path).expanduser().resolve()
            candidates = await self.repository.list({"directory": str(target.parent)})
            metadata = next((item for item in candidates if item["path"] == str(target)), None)
            if metadata is None:
                raise FileNotFoundError(f"session not found: {target}")
        return await self.switch(metadata, listener)

    async def release(self) -> None:
        """先关闭 agent 再退订和关闭会话；异常也清空绑定。"""
        try:
            if self.agent is not None:
                await self.agent.close()
        finally:
            self.agent = None
            if self._unsubscribe is not None:
                self._unsubscribe()
                self._unsubscribe = None
            try:
                if self.session is not None:
                    await self.session.close()
            finally:
                self.session = None

    async def switch(self, metadata: SessionMetadata | None, listener: EventSink) -> DisplayState:
        """先释放旧绑定再打开新会话；失败释放部分资源，同一会话直接返回投影。"""
        # 选中当前会话时直接复用，不能重新打开仍活动的仓库句柄。
        if self.session is not None and metadata == self.session.metadata:
            return project_history(
                await self.session.get_display_history(), await self.session.state()
            )
        await self.release()
        try:
            self.session = (
                await self.repository.create()
                if metadata is None
                else await self.repository.open(metadata)
            )
            state = project_history(
                await self.session.get_display_history(), await self.session.state()
            )
            self.agent = self.agent_factory({"config": self.config, "session": self.session})
            self._unsubscribe = self.agent.subscribe(listener)
            return state
        except BaseException:
            await self.release()
            raise

    async def close(self) -> None:
        """释放绑定后关闭仓库，即使释放阶段抛出异常。"""
        try:
            await self.release()
        finally:
            await self.repository.close()
