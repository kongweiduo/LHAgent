"""单次调用的用量与时序统计；未知计数保持空值，累计用量不重复相加。"""

from unittest.mock import patch

from lhagent.client.metrics import CallMetrics
from tests.samples import usage


def test_no_attempts_or_content_on_early_finish():
    """提前结束不会虚构请求尝试或首内容时间。"""
    with patch("lhagent.client.metrics.monotonic", side_effect=[10.0, 12.5]) as clock:
        metrics = CallMetrics()
        result = metrics.finish()
        assert result == {
            "usage": usage(),
            "elapsed_seconds": 2.5,
            "first_content_seconds": None,
            "attempts": 0,
        }
        assert metrics.finish() == result
        assert clock.call_count == 2


def test_multiple_attempts_include_wait_and_first_content_only_once():
    """总耗时包含重试等待，首内容时间只记录一次。"""
    with patch("lhagent.client.metrics.monotonic", side_effect=[100.0, 105.0, 111.0]):
        metrics = CallMetrics()
        metrics.record_attempt()
        metrics.record_attempt()
        metrics.record_first_content()
        metrics.record_first_content()
        assert metrics.finish() == {
            "usage": usage(),
            "elapsed_seconds": 11.0,
            "first_content_seconds": 5.0,
            "attempts": 2,
        }


def test_failed_call_without_content_keeps_known_usage_and_unknown_fields():
    """无内容的失败仍保留已知用量，未知字段保持空值。"""
    with patch("lhagent.client.metrics.monotonic", side_effect=[1.0, 4.0]):
        metrics = CallMetrics()
        metrics.record_attempt()
        metrics.record_attempt()
        metrics.update_usage({**usage(), "input_tokens": 7})
        assert metrics.finish() == {
            "usage": {**usage(), "input_tokens": 7},
            "elapsed_seconds": 3.0,
            "first_content_seconds": None,
            "attempts": 2,
        }


def test_usage_updates_replace_cumulative_values_without_summing():
    """用量分片是累计快照，更新时替换而非累加。"""
    with patch("lhagent.client.metrics.monotonic", side_effect=[0.0, 1.0]):
        metrics = CallMetrics()
        first = {**usage(), "input_tokens": 10, "output_tokens": 2}
        metrics.update_usage(first)
        first["input_tokens"] = 999
        metrics.update_usage({**usage(), "input_tokens": 10, "output_tokens": 2})
        metrics.update_usage(
            {**usage(), "output_tokens": 5, "total_tokens": 15, "cache_read_tokens": 0}
        )
        result = metrics.finish()
        assert result["usage"] == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "cache_read_tokens": 0,
            "cache_write_tokens": None,
        }
        result["usage"]["input_tokens"] = 100
        metrics.record_attempt()
        metrics.record_first_content()
        metrics.update_usage({**usage(), "input_tokens": 100})
        assert metrics.finish()["usage"]["input_tokens"] == 10
        assert metrics.finish()["first_content_seconds"] is None
        assert metrics.finish()["attempts"] == 0
