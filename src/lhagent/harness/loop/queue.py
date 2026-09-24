"""声明运行期间的两类输入队列，不读取终端、不持久化用户配置。

steer 在当前响应及工具批次结束后的请求边界接入；follow_up 在本来可以
结束运行时接入。队列不主动取消生成或工具，消费时机由 loop 控制。
"""

from collections import deque
from copy import deepcopy
from typing import Literal


class InputQueue:
    """保存尚未接入历史的用户消息，先入先出，默认每次消费一条。

    入队不等于模型已看到消息；消费后由 loop 交给会话层逐条提交。
    仅供单事件循环使用，不承诺线程安全或跨进程恢复。入队和出队均
    深拷贝消息，调用方与队列互不共享可变的消息内容。
    """

    def __init__(
        self,
        steering_mode: Literal["one_at_a_time", "all"] = "one_at_a_time",
        follow_up_mode: Literal["one_at_a_time", "all"] = "one_at_a_time",
    ) -> None:
        """分别配置两种队列的消费方式，不建立后台调度任务。"""
        for name, mode in (("steering_mode", steering_mode), ("follow_up_mode", follow_up_mode)):
            if mode not in ("one_at_a_time", "all"):
                raise ValueError(f"{name} must be 'one_at_a_time' or 'all'")
        self._steering_mode = steering_mode
        self._follow_up_mode = follow_up_mode
        self._steering: deque[dict[str, object]] = deque()
        self._follow_up: deque[dict[str, object]] = deque()

    def steer(self, message: dict[str, object]) -> None:
        """追加用于调整当前任务的消息，不中断当前响应或当前工具批次。"""
        self._steering.append(deepcopy(message))

    def follow_up(self, message: dict[str, object]) -> None:
        """追加后续请求，等待没有待执行工具和 steering 输入时接入。"""
        self._follow_up.append(deepcopy(message))

    def drain_steering(self) -> list[dict[str, object]]:
        """按配置取出一条或全部 steering 消息，由 loop 在请求边界调用。"""
        return self._drain(self._steering, self._steering_mode)

    def drain_follow_up(self) -> list[dict[str, object]]:
        """按配置取出后续请求，仅在循环本来可以正常结束时调用。"""
        return self._drain(self._follow_up, self._follow_up_mode)

    @staticmethod
    def _drain(
        pending: deque[dict[str, object]],
        mode: Literal["one_at_a_time", "all"],
    ) -> list[dict[str, object]]:
        """按配置取出一条或全部 FIFO 消息，并返回独立副本。"""
        count = min(1, len(pending)) if mode == "one_at_a_time" else len(pending)
        return [deepcopy(pending.popleft()) for _ in range(count)]

    def has_pending(self) -> bool:
        """判断是否仍有未消费输入，不消费或修改队列。"""
        return bool(self._steering or self._follow_up)

    def clear(self) -> dict[str, list[dict[str, object]]]:
        """显式清空并返回两种队列的未消费消息，键为 steer 和 follow_up。

        主动取消运行不隐式调用此接口；未消费输入仍可由调用方查看和处理。
        """
        return {
            "steer": self._drain(self._steering, "all"),
            "follow_up": self._drain(self._follow_up, "all"),
        }
