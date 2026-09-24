"""POSIX 子进程信号与终端失联回归，验证有序清理及非成功退出。"""

import asyncio
import signal

import pytest

from lhagent import cli


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_signal_during_work_cancels_once_and_waits_for_cleanup(monkeypatch, signum):
    """工作期间的终止信号只取消一次，并等待清理完成再退出。"""

    async def scenario():
        loop = asyncio.get_running_loop()
        callbacks = {}
        removed = []
        restored = []
        ready, cleaning, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
        previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
        monkeypatch.setattr(
            loop, "add_signal_handler", lambda s, f, *args: callbacks.update({s: (f, args)})
        )
        monkeypatch.setattr(loop, "remove_signal_handler", lambda s: removed.append(s))
        monkeypatch.setattr(signal, "signal", lambda s, h: restored.append((s, h)))

        async def run(*args):
            try:
                ready.set()
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await finish.wait()

        monkeypatch.setattr(cli, "_run", run)
        task = asyncio.create_task(cli._run_with_signals("config", None, None))
        await ready.wait()
        callback, args = callbacks[signum]
        callback(*args)
        await cleaning.wait()
        callback(*args)  # Repeated termination cannot interrupt cleanup.
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(RuntimeError, match=signum.name):
            await task
        assert set(removed) == set(previous)
        assert dict(restored) == previous

    asyncio.run(scenario())
