"""两类输入队列的独立模式、FIFO 消费和副本所有权。"""

import pytest

from lhagent.harness.loop.queue import InputQueue


def message(text: str) -> dict[str, object]:
    """构造可观察深拷贝隔离性的用户消息。"""
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def test_default_modes_are_independent_fifo_and_clear_preserves_order() -> None:
    """两类默认队列独立，FIFO 取出及清空保持顺序。"""
    queue = InputQueue()
    assert not queue.has_pending()
    assert queue.drain_steering() == []
    assert queue.drain_follow_up() == []
    assert queue.clear() == {"steer": [], "follow_up": []}

    queue.steer(message("s1"))
    queue.follow_up(message("f1"))
    queue.steer(message("s2"))
    queue.follow_up(message("f2"))

    assert queue.has_pending()
    assert queue.has_pending()
    assert queue.drain_steering() == [message("s1")]
    assert queue.drain_follow_up() == [message("f1")]
    assert queue.has_pending()
    assert queue.clear() == {"steer": [message("s2")], "follow_up": [message("f2")]}
    assert not queue.has_pending()
    assert queue.clear() == {"steer": [], "follow_up": []}


@pytest.mark.parametrize(
    "steering_mode,follow_up_mode",
    [
        ("all", "one_at_a_time"),
        ("one_at_a_time", "all"),
        ("all", "all"),
    ],
)
def test_modes_are_configured_separately(steering_mode: str, follow_up_mode: str) -> None:
    """steering 与 follow-up 可以分别配置消费模式。"""
    queue = InputQueue(steering_mode=steering_mode, follow_up_mode=follow_up_mode)
    for index in range(3):
        queue.steer(message(f"s{index}"))
        queue.follow_up(message(f"f{index}"))

    s_count = 3 if steering_mode == "all" else 1
    f_count = 3 if follow_up_mode == "all" else 1
    assert queue.drain_steering() == [message(f"s{i}") for i in range(s_count)]
    assert queue.drain_follow_up() == [message(f"f{i}") for i in range(f_count)]
    assert queue.clear() == {
        "steer": [message(f"s{i}") for i in range(s_count, 3)],
        "follow_up": [message(f"f{i}") for i in range(f_count, 3)],
    }


@pytest.mark.parametrize("parameter", ["steering_mode", "follow_up_mode"])
def test_invalid_modes_rejected(parameter: str) -> None:
    """未知消费模式在创建队列时拒绝。"""
    with pytest.raises(ValueError, match=parameter):
        InputQueue(**{parameter: "unknown"})


def test_messages_are_copied_on_enqueue_and_dequeue() -> None:
    """入队和出队均复制消息，防止外部改写队列。"""
    queue = InputQueue(steering_mode="all")
    original = message("original")
    queue.steer(original)
    queue.follow_up(original)
    original["content"][0]["text"] = "changed"

    assert queue.drain_steering() == [message("original")]
    assert queue.clear() == {"steer": [], "follow_up": [message("original")]}

    queue.steer(message("next"))
    queue.steer(message("last"))
    first = queue.drain_steering()
    first[0]["content"][0]["text"] = "mutated"
    assert first[1] == message("last")
    assert not queue.has_pending()

    queue.follow_up(message("pending"))
    returned = queue.clear()
    returned["follow_up"][0]["content"][0]["text"] = "mutated"
    queue.follow_up(message("fresh"))
    assert queue.drain_follow_up() == [message("fresh")]
