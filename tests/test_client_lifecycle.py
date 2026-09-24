"""响应建立、迭代及关闭中的取消竞态，验证通信资源最终释放。"""

import asyncio
from contextlib import aclosing

import httpx
import pytest

from lhagent.client.client import Client
from lhagent.client.config import load_config
from tests.test_client_retry import status_error


def request(call_id="one"):
    """构造指定调用身份的离线请求。"""
    return {"call_id": call_id, "model": "test", "messages": [], "parameters": {}}


def chunk(text=None, finish=None):
    """构造可选文本和终态的内部响应分片。"""
    return {
        "deltas": []
        if text is None
        else [{"type": "text", "tool_index": None, "data": {"text": text}}],
        "usage": None,
        "finish_reason": finish,
    }


class Response:
    """可阻塞或故障注入的响应，记录读取开始及关闭完成。"""

    def __init__(self, blocked=False, failure=None):
        self.index = 0
        self.reading = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False
        self.blocked = blocked
        self.failure = failure

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.index += 1
        if self.index == 1:
            return chunk("partial")
        if self.index == 2:
            self.reading.set()
            if self.blocked:
                await self.release.wait()
            if self.failure:
                raise self.failure
            return chunk(finish="stop")
        raise StopAsyncIteration

    async def aclose(self):
        await asyncio.sleep(0)
        self.closed = True


class Transport:
    """可控制建立阶段与逐次失败的传输替身，按调用身份保留响应。"""

    def __init__(self, config):
        self.opened = 0
        self.closes = 0
        self.started = asyncio.Event()
        self.establish_cleaned = asyncio.Event()
        self.block_open = False
        self.failures = []
        self.responses = {}

    async def open_stream(self, req):
        self.opened += 1
        self.started.set()
        if self.block_open:
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                self.establish_cleaned.set()
        if self.failures:
            raise self.failures.pop(0)
        return self.responses.setdefault(req["call_id"], Response())

    async def close(self):
        assert all(r.closed for r in self.responses.values())
        self.closes += 1


def client(monkeypatch, **config):
    """注入假传输并设置零等待重试配置，不发起网络通信。"""
    monkeypatch.setattr("lhagent.client.client.Transport", Transport)
    return Client(
        load_config(
            {
                "base_url": "https://example.test",
                "api_key": "fake",
                "max_retry_delay_seconds": 0,
                **config,
            }
        )
    )


async def no_tasks():
    """让出一次调度后断言只剩当前任务，检测清理泄漏。"""
    await asyncio.sleep(0)
    assert asyncio.all_tasks() == {asyncio.current_task()}


def run(coro):
    """在有限超时内执行异步场景，避免生命周期回归永久挂起。"""

    async def bounded():
        async with asyncio.timeout(3):
            await coro

    asyncio.run(bounded())


def test_precancel_and_unknown_id(monkeypatch):
    """预取消不建立请求，取消未知调用保持幂等。"""

    async def scenario():
        c = client(monkeypatch)
        signal = asyncio.Event()
        signal.set()
        events = [e async for e in c.stream(request(), cancel_event=signal)]
        assert len(events) == 1 and events[0]["type"] == "cancelled"
        result = events[0]["result"]
        assert result["stats"]["attempts"] == 0 and result["error_kind"] is None
        assert signal.is_set() and c._transport.opened == 0
        await c.cancel("unknown")
        await c.close()
        await no_tasks()

    run(scenario())


@pytest.mark.parametrize(
    "errors,attempts,reason",
    [
        ([httpx.ConnectError("secret")], 2, "stop"),
        ([status_error(503)] * 3, 3, "error"),
        ([status_error(401)], 1, "error"),
        ([status_error(429, headers={"Retry-After": "31"})], 1, "error"),
    ],
)
def test_retry_policy_integration(monkeypatch, errors, attempts, reason):
    """客户端只按既定分类和次数重试建立失败。"""

    async def scenario():
        c = client(monkeypatch)
        c._transport.failures = errors.copy()
        result = await c.complete(request())
        assert result["stats"]["attempts"] == attempts
        assert c._transport.opened == attempts and result["finish_reason"] == reason
        assert "secret" not in str(result)
        await c.close()
        await no_tasks()

    run(scenario())


@pytest.mark.parametrize("phase", ["establish", "backoff", "read"])
@pytest.mark.parametrize("mode", ["signal", "cancel", "task", "close"])
def test_cancel_all_wait_phases(monkeypatch, phase, mode):
    """建立、读取和退避等待均可被取消。"""

    async def scenario():
        c = client(monkeypatch, max_retry_delay_seconds=30)
        t = c._transport
        signal = asyncio.Event()
        waiting = asyncio.Event()
        if phase == "establish":
            t.block_open = True
            waiting = t.started
        elif phase == "backoff":
            t.failures = [status_error(503)]

            def delay(*args):
                waiting.set()
                return 30

            monkeypatch.setattr("lhagent.client.client.get_retry_delay", delay)
        else:
            response = Response(blocked=True)
            t.responses["one"] = response
            waiting = response.reading
        task = asyncio.create_task(c.complete(request(), cancel_event=signal))
        await waiting.wait()
        if mode == "signal":
            signal.set()
        elif mode == "cancel":
            await c.cancel("one")
        elif mode == "task":
            task.cancel()
        else:
            await c.close()
        if mode == "task":
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert result["finish_reason"] == "cancelled"
            assert result["error"] is None and result["error_kind"] is None
            assert result["stats"]["attempts"] == 1
            if phase == "read":
                assert result["content"] == [{"type": "text", "text": "partial"}]
        if phase == "establish":
            assert t.establish_cleaned.is_set()
        if phase == "read":
            assert response.closed
        assert t.opened == 1 and signal.is_set() == (mode == "signal")
        await c.close()
        await c.close()
        assert t.closes == 1
        with pytest.raises(RuntimeError, match="closed"):
            await c.complete(request())
        await no_tasks()

    run(scenario())


@pytest.mark.parametrize("mode", ["cancel", "close", "early", "signal"])
def test_paused_consumer_cleanup(monkeypatch, mode):
    """消费者暂停时取消仍能释放通信资源。"""

    async def scenario():
        c = client(monkeypatch)
        signal = asyncio.Event()
        async with aclosing(c.stream(request(), cancel_event=signal)) as stream:
            assert (await anext(stream))["type"] == "delta"
            response = c._transport.responses["one"]
            if mode == "cancel":
                await c.cancel("one")
            elif mode == "close":
                await c.close()
            elif mode == "signal":
                signal.set()
            if mode != "early":
                terminal = await anext(stream)
                assert terminal["type"] == "cancelled" and response.closed
        assert response.closed
        await c.close()
        await no_tasks()

    run(scenario())


def test_parallel_calls_and_duplicate_id(monkeypatch):
    """并行调用相互隔离，同一活动身份不得重复使用。"""

    async def scenario():
        c = client(monkeypatch)
        first, second = Response(blocked=True), Response(blocked=True)
        c._transport.responses = {"one": first, "two": second}
        a = asyncio.create_task(c.complete(request("one")))
        b = asyncio.create_task(c.complete(request("two")))
        await first.reading.wait()
        await second.reading.wait()
        with pytest.raises(ValueError, match="already active"):
            await c.complete(request("one"))
        await c.cancel("unknown")
        await c.cancel("one")
        assert (await a)["finish_reason"] == "cancelled"
        assert not b.done() and not second.closed
        second.release.set()
        assert (await b)["finish_reason"] == "stop"
        # 终结后身份可以复用。
        c._transport.responses["one"] = Response()
        assert (await c.complete(request()))["finish_reason"] == "stop"
        await asyncio.gather(c.close(), c.close())
        assert c._transport.closes == 1
        await no_tasks()

    run(scenario())


@pytest.mark.parametrize("mode", ["cancel", "task", "close"])
def test_cleanup_is_awaited_even_if_caller_is_cancelled(monkeypatch, mode):
    """调用者取消后仍等待响应清理落定。"""

    async def scenario():
        c = client(monkeypatch)
        cleaning, release = asyncio.Event(), asyncio.Event()

        class SlowResponse(Response):
            async def aclose(self):
                cleaning.set()
                await release.wait()
                self.closed = True

        response = SlowResponse(blocked=True)
        c._transport.responses["one"] = response
        call = asyncio.create_task(c.complete(request()))
        await response.reading.wait()
        if mode == "task":
            call.cancel()
            stopper = call
        else:
            stopper = asyncio.create_task(c.close() if mode == "close" else c.cancel("one"))
        await cleaning.wait()
        assert not stopper.done() and not call.done() and c._transport.closes == 0
        if mode in ("close", "task"):
            stopper.cancel()
            await asyncio.sleep(0)
            assert not stopper.done()
        release.set()
        if mode in ("close", "task"):
            with pytest.raises(asyncio.CancelledError):
                await stopper
        else:
            await stopper
        if mode != "task":
            assert (await call)["finish_reason"] == "cancelled"
        assert response.closed
        await c.close()
        await no_tasks()

    run(scenario())


def test_cancel_racing_successful_establishment_releases_response(monkeypatch):
    """建立成功与取消同时发生时关闭未交付响应。"""

    async def scenario():
        c = client(monkeypatch)
        signal = asyncio.Event()
        response = Response()

        async def open_stream(req):
            signal.set()
            return response

        c._transport.open_stream = open_stream
        result = await c.complete(request(), cancel_event=signal)
        assert result["finish_reason"] == "cancelled"
        assert response.closed and response.index == 0
        await c.close()
        await no_tasks()

    run(scenario())


def test_real_sdk_response_is_closed_on_cancellation(monkeypatch):
    """真实 SDK 响应在取消后关闭底层传输。"""
    from tests.test_client_transport_stream import Body, frame, setup

    async def scenario():
        body = Body([frame({"choices": [{"index": 0, "delta": {"content": "hi"}}]})], wait=True)
        transport, calls = setup(monkeypatch, body)
        monkeypatch.setattr("lhagent.client.client.Transport", lambda config: transport)
        c = Client(load_config({"base_url": "https://example.test", "api_key": "fake"}))
        task = asyncio.create_task(c.complete(request()))
        await body.reading.wait()
        await c.cancel("one")
        result = await task
        assert result["finish_reason"] == "cancelled"
        assert result["content"] == [{"type": "text", "text": "hi"}]
        assert body.closed and len(calls) == 1
        await c.close()
        assert transport._client.is_closed()
        await no_tasks()

    run(scenario())


def test_no_retry_after_acquiring_response(monkeypatch):
    """取得响应后发生错误不得重新发送请求。"""

    async def scenario():
        c = client(monkeypatch)
        response = Response(failure=httpx.ReadError("secret"))
        c._transport.responses["one"] = response
        result = await c.complete(request())
        assert result["finish_reason"] == "error" and result["error_kind"] == "transport"
        assert result["content"] == [{"type": "text", "text": "partial"}]
        assert response.closed and c._transport.opened == 1
        await c.close()
        await no_tasks()

    run(scenario())
