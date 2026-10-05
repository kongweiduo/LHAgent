"""Install the existing Linux bundle into a Harbor task environment."""

from __future__ import annotations

import asyncio
import os
import shlex
import tarfile
import tempfile
import tomllib
from pathlib import Path

import toml
from harbor.agents.base import BaseAgent
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.trial.errors import AgentTimeoutError


def discover_bundles(path: Path) -> dict[str, Path]:
    paths = sorted(path.glob("*.tar.gz")) if path.is_dir() else [path]
    bundles = {}
    for archive in paths:
        with tarfile.open(archive, "r:gz") as tar:
            marker = tar.extractfile("lhagent/TARGET")
            if marker is None:
                raise ValueError(f"missing TARGET in {archive}")
            target = marker.read().decode().strip()
        if target not in {"linux/amd64", "linux/arm64"}:
            raise ValueError(f"unsupported bundle target: {target}")
        if target in bundles:
            raise ValueError(f"multiple bundles for {target}")
        bundles[target] = archive.resolve()
    if not bundles:
        raise ValueError(f"no .tar.gz bundles in {path}")
    return bundles


class LHAgent(BaseAgent):
    def __init__(self, *, bundles, config, trace_dir, timeout=1800, instruction=None, **kwargs):
        super().__init__(**kwargs)
        self.bundles = bundles
        self.config = Path(config)
        self.trace_dir = Path(trace_dir)
        self.timeout = timeout
        self.instruction = instruction
        self._trace_ready = False

    @staticmethod
    def name() -> str:
        return "lhagent"

    def version(self) -> str:
        return "linux-bundle"

    async def _exec(self, environment, command, **kwargs):
        result = await environment.exec(command, **kwargs)
        if result.return_code:
            raise RuntimeError(f"Agent setup failed: {result.stderr or result.stdout}")
        return (result.stdout or "").strip()

    async def setup(self, environment):
        arch = await self._exec(environment, "uname -m")
        platform = "linux/" + {"x86_64": "amd64", "aarch64": "arm64"}.get(arch, arch)
        if platform not in self.bundles:
            raise ValueError(f"no LHAgent bundle for task platform {platform}")
        await environment.upload_file(self.bundles[platform], "/tmp/lhagent.tar.gz")
        await self._exec(
            environment,
            "mkdir -p /opt && tar -xzf /tmp/lhagent.tar.gz -C /opt && chmod -R a+rX /opt/lhagent",
            user="root",
        )
        await self._exec(environment, "/opt/lhagent/lhagent --bundle-check")
        with self.config.open("rb") as stream:
            config = tomllib.load(stream)
        # An omitted cwd follows the official environment's working directory.
        config["coding_agent"].setdefault("cwd", await self._exec(environment, "pwd"))
        with tempfile.TemporaryDirectory(prefix="lhagent-tb-config-") as directory:
            path = Path(directory) / "lhagent.toml"
            path.write_text(toml.dumps(config))
            await environment.upload_file(path, "/tmp/lhagent.toml")
        logs = shlex.quote(str(self.environment_logs_dir))
        await self._exec(
            environment,
            f"mkdir -p {logs}/native/sessions /tmp/lhagent-run && "
            f"ln -sfn {logs}/native /tmp/lhagent-run/.lhagent && "
            f"chmod -R a+rwX {logs} /tmp/lhagent-run && chmod a+r /tmp/lhagent.toml && command -v timeout",
            user="root",
        )
        self._trace_ready = True

    async def run(self, instruction, environment, context):
        logs = shlex.quote(str(self.environment_logs_dir))
        prompt = self.instruction if self.instruction is not None else instruction
        command = shlex.join(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=10",
                str(self.timeout if self.timeout is not None else 0),
                "/opt/lhagent/lhagent",
                "--config",
                "/tmp/lhagent.toml",
                "--instruction",
                prompt,
            ]
        )
        try:
            result = await environment.exec(
                f"{command} > {logs}/stdout.log 2> {logs}/stderr.log",
                cwd="/tmp/lhagent-run",
                env={
                    key: os.environ[key]
                    for key in ("LHAGENT_API_KEY", "LHAGENT_BASE_URL")
                    if key in os.environ
                },
                timeout_sec=self.timeout + 30 if self.timeout is not None else None,
            )
            if result.return_code in {124, 137}:
                raise AgentTimeoutError(f"LHAgent timed out after {self.timeout} seconds")
            if result.return_code:
                raise NonZeroAgentExitCodeError(
                    f"LHAgent exited {result.return_code}; see agent/stderr.log"
                )
            context.metadata = {"native_trajectory": str(self.trace_dir / "sessions")}
        except (asyncio.CancelledError, TimeoutError):
            await environment.exec("pkill -KILL -f '^/opt/lhagent/runtime/bin/python3.12' || true")
            raise
        finally:
            await self.save_traces(environment)

    async def save_traces(self, environment):
        if self._trace_ready:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            await environment.download_dir(
                str(self.environment_logs_dir / "native"), self.trace_dir
            )
