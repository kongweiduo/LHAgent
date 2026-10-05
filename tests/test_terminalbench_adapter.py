import asyncio
import hashlib
import io
import json
import tarfile
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("harbor")

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.models.agent.context import AgentContext
from harbor.models.task.task import Task
from harbor.models.trial.config import TaskConfig
from harbor.models.verifier.result import VerifierResult
from harbor.trial.artifact_handler import ArtifactHandler
from harbor.trial.errors import AgentTimeoutError
from harbor.verifier.verifier import Verifier

from lhagent.evals.benchmarks.terminalbench import adapter, agent, environment


def run(awaitable):
    return asyncio.run(awaitable)


def test_bundle_target_is_read_from_archive(tmp_path):
    archive = tmp_path / "arbitrary-name.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        data = b"linux/arm64\n"
        info = tarfile.TarInfo("lhagent/TARGET")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    assert agent.discover_bundles(tmp_path) == {"linux/arm64": archive}
    with tarfile.open(tmp_path / "duplicate.tar.gz", "w:gz") as tar:
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match="multiple"):
        agent.discover_bundles(tmp_path)


def test_defaults_and_selection():
    args = adapter.build_parser().parse_args(["--bundle", "bundles", "--config", "config.toml"])
    assert (args.dataset, args.dataset_version) == ("terminal-bench/terminal-bench", "4.0.0")
    assert args.timeout == args.test_timeout == 1800
    tasks = [TaskConfig(path=Path(name)) for name in ["c", "a", "b"]]
    assert adapter.select_tasks(tasks, count=2, seed=42) == adapter.select_tasks(
        tasks[::-1], count=2, seed=42
    )
    assert [adapter.task_id(task) for task in adapter.select_tasks(tasks, task_ids=["b", "a"])] == [
        "b",
        "a",
    ]
    for kwargs in [
        {"count": 0},
        {"count": 4},
        {"task_ids": ["missing"]},
        {"task_ids": ["a", "a"]},
        {"count": 1, "task_ids": ["a"]},
    ]:
        with pytest.raises(ValueError):
            adapter.select_tasks(tasks, **kwargs)


@pytest.mark.parametrize(
    "platforms,bundles,native,expected",
    [
        (
            {"linux/amd64", "linux/arm64"},
            {"linux/amd64", "linux/arm64"},
            "linux/arm64",
            "linux/arm64",
        ),
        ({"linux/amd64"}, {"linux/amd64", "linux/arm64"}, "linux/arm64", "linux/amd64"),
        ({"linux/arm64"}, {"linux/amd64"}, "linux/amd64", None),
    ],
)
def test_platform_matching(platforms, bundles, native, expected):
    if expected is None:
        with pytest.raises(ValueError, match="no matching"):
            environment.select_platform(platforms, bundles, native)
    else:
        assert environment.select_platform(platforms, bundles, native) == expected


def test_setup_follows_task_workdir_without_changing_user_config(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[coding_agent]\nmodel = "test-model"\n')
    instance = agent.LHAgent(
        bundles={"linux/arm64": "bundle"},
        config=config,
        trace_dir=tmp_path / "traces",
        logs_dir=tmp_path / "logs",
    )
    uploaded = []

    async def upload(source, target):
        if target.endswith(".toml"):
            uploaded.append(tomllib.loads(Path(source).read_text()))

    async def execute(command, **kwargs):
        value = "aarch64" if command == "uname -m" else "/workspace" if command == "pwd" else None
        return SimpleNamespace(return_code=0, stdout=value, stderr="")

    env = SimpleNamespace(exec=execute, upload_file=upload)
    run(instance.setup(env))
    assert uploaded[0]["coding_agent"]["cwd"] == "/workspace"
    assert "cwd" not in tomllib.loads(config.read_text())["coding_agent"]


@pytest.mark.parametrize(
    "status,exception", [(0, None), (1, NonZeroAgentExitCodeError), (124, AgentTimeoutError)]
)
def test_agent_saves_trajectory_on_success_and_failure(tmp_path, monkeypatch, status, exception):
    instance = agent.LHAgent(
        bundles={}, config="config.toml", trace_dir=tmp_path / "traces", logs_dir=tmp_path / "logs"
    )
    instance._trace_ready = True
    monkeypatch.setenv("LHAGENT_API_KEY", "test-secret")
    env = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=status)),
        download_dir=AsyncMock(),
    )
    if exception:
        with pytest.raises(exception):
            run(instance.run("instruction with 'quotes'", env, AgentContext()))
    else:
        run(instance.run("instruction", env, AgentContext()))
    env.download_dir.assert_awaited_once_with("/logs/agent/native", tmp_path / "traces")
    call = env.exec.call_args_list[0]
    assert call.kwargs["env"]["LHAGENT_API_KEY"] == "test-secret"
    assert "test-secret" not in call.args[0]
    assert "--kill-after=10" in call.args[0]


def make_task(root, *, agent_image="official:agent", verifier_image="official:verifier"):
    root.mkdir()
    (root / "environment").mkdir()
    (root / "tests").mkdir()
    (root / "instruction.md").write_text("Create the requested output.")
    (root / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
    (root / "task.toml").write_text(
        'schema_version = "1.0"\n[agent]\ntimeout_sec = 28800\n[environment]\n'
        + (f'docker_image = "{agent_image}"\n' if agent_image else "")
        + '[verifier]\nenvironment_mode = "separate"\n[verifier.environment]\n'
        + (f'docker_image = "{verifier_image}"\n' if verifier_image else "")
    )
    return Task(root)


def test_requires_prebuilt_images_for_agent_and_verifier(tmp_path):
    adapter.validate_prebuilt_task(make_task(tmp_path / "valid"))
    with pytest.raises(ValueError, match="agent image"):
        adapter.validate_prebuilt_task(make_task(tmp_path / "bad", agent_image=None))
    task = make_task(tmp_path / "bad-verifier", verifier_image=None)
    (task.paths.tests_dir / "Dockerfile").write_text("FROM scratch\n")
    with pytest.raises(ValueError, match="verifier image"):
        adapter.validate_prebuilt_task(task)


@pytest.mark.parametrize("existing,cleanup_failure", [(True, False), (False, False), (False, True)])
def test_cleanup_preserves_initial_images_and_records_failures(
    tmp_path, monkeypatch, existing, cleanup_failure
):
    instance = object.__new__(environment.PrebuiltDockerEnvironment)
    instance.initial_images = {"sha256:old"} if existing else set()
    instance.task_images = {"official:agent"}
    instance.cleanup_errors_path = tmp_path / "cleanup-errors.json"
    instance.task_images_path = tmp_path / "images.json"
    instance.prepare_logs_for_host = AsyncMock()
    instance._run_docker_compose_command = AsyncMock()
    for suffix in ("mounts", "resources", "env", "egress_control_services"):
        setattr(instance, f"_cleanup_{suffix}_compose_file", lambda: None)
    calls = []

    async def docker(*args):
        calls.append(args)
        if args[:2] == ("image", "inspect"):
            return "sha256:old\n"
        if cleanup_failure:
            raise RuntimeError("image removal failed")
        return ""

    monkeypatch.setattr(environment, "docker", docker)
    if cleanup_failure:
        run(instance.stop())
        run(environment.cleanup_images(tmp_path, instance.initial_images))
        assert "image removal failed" in instance.cleanup_errors_path.read_text()
    else:
        run(instance.stop())
        run(environment.cleanup_images(tmp_path, instance.initial_images))
    instance._run_docker_compose_command.assert_awaited_once_with(
        ["down", "--volumes", "--remove-orphans"]
    )
    assert (("image", "rm", "official:agent") in calls) == (not existing)


def grade(name, reward=1.0, error=None):
    return SimpleNamespace(
        exception_info=SimpleNamespace(exception_message=error) if error else None,
        verifier_result=SimpleNamespace(rewards={"reward": reward}),
        model_dump_json=lambda **kw: json.dumps({"task_name": name, "reward": reward}),
    )


@pytest.mark.parametrize("failure", [None, "solve", "grade", "cleanup"])
def test_serial_run_directories_summary_and_failures(tmp_path, monkeypatch, failure):
    config = tmp_path / "config.toml"
    config.write_text('[coding_agent]\nmodel = "test-model"\n')
    args = adapter.build_parser().parse_args(
        [
            "--bundle",
            str(tmp_path),
            "--config",
            str(config),
            "--run-id",
            "test",
            "--output-dir",
            str(tmp_path / "terminalbench"),
        ]
    )
    tasks = [TaskConfig(path=tmp_path / name) for name in ["first", "second"]]
    monkeypatch.setattr(adapter, "load_tasks", AsyncMock(return_value=(tasks, {})))
    monkeypatch.setattr(adapter, "docker", AsyncMock(return_value="aarch64"))
    events = []

    async def trial(task, **kwargs):
        name = adapter.task_id(task)
        events.extend([(name, "solve"), (name, "grade"), (name, "cleanup")])
        run_dir = kwargs["run_dir"]
        assert kwargs["args"].timeout == 1800
        if name == "first":
            if failure == "solve":
                raise RuntimeError("solve failed")
            if failure == "grade":
                return grade(name, float("nan"))
            if failure == "cleanup":
                path = run_dir / "logs/attempts/first/1/cleanup-errors.json"
                path.write_text('["cleanup failed"]')
        return grade(name, 1 if name == "first" else 0)

    monkeypatch.setattr(adapter, "run_trial", trial)
    assert run(adapter.evaluate(args, {})) == int(bool(failure))
    run_dir = tmp_path / "terminalbench/test"
    summary = json.loads((run_dir / "logs/run_evaluation/test/results.json").read_text())
    assert summary["total_instances"] == 2
    if failure == "cleanup":
        assert events == [("first", "solve"), ("first", "grade"), ("first", "cleanup")]
        assert summary["incomplete_ids"] == ["second"]
    elif failure:
        assert summary["incomplete_ids"] == ["first"]
        assert summary["unresolved_ids"] == ["second"]
    else:
        assert summary["resolved_ids"] == ["first"]
        assert summary["unresolved_ids"] == ["second"]
    assert (run_dir / "logs/test.failures.json").is_file()
    with pytest.raises(SystemExit, match="already exists"):
        adapter.main(
            [
                "--bundle",
                str(tmp_path),
                "--config",
                str(config),
                "--run-id",
                "test",
                "--output-dir",
                str(tmp_path / "terminalbench"),
            ]
        )


@pytest.mark.parametrize("reward", [None, True, float("nan"), float("inf"), "1"])
def test_invalid_grades_are_incomplete(reward):
    with pytest.raises(RuntimeError, match="finite reward"):
        adapter.graded_reward(grade("task", reward))


@pytest.mark.parametrize("official_timeouts", [False, True])
def test_real_harbor_trial_saves_outputs_before_image_cleanup(
    tmp_path, monkeypatch, official_timeouts
):
    task = make_task(tmp_path / "task")
    config = tmp_path / "config.toml"
    config.write_text('[coding_agent]\nmodel = "test-model"\n')
    args = adapter.build_parser().parse_args(["--bundle", str(tmp_path), "--config", str(config)])
    args.run_id = "test"
    args.official_timeouts = official_timeouts
    events = []

    async def start(self, force_build=False):
        assert force_build is False
        events.append((self.task_env_config.docker_image, "start"))

    async def stop(self, delete=True):
        events.append((self.task_env_config.docker_image, "stop"))

    async def execute(self, *args, **kwargs):
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def solve(self, instruction, environment, context):
        assert self.timeout == (task.config.agent.timeout_sec if official_timeouts else 1800)
        events.append((environment.task_env_config.docker_image, "solve"))
        native = self.logs_dir / "native/sessions"
        native.mkdir(parents=True)
        (native / "test.jsonl").write_text('{"event":"test"}\n')
        (self.logs_dir / "stdout.log").write_text("agent stdout")

    async def collect(self, env, artifacts_dir, **kwargs):
        events.append((env.task_env_config.docker_image, "collect"))
        (artifacts_dir / "output.txt").write_text("answer")

    async def upload(self, env, *, artifacts_dir, **kwargs):
        assert (artifacts_dir / "output.txt").read_text() == "answer"
        events.append((env.task_env_config.docker_image, "upload"))

    async def verify(self):
        events.append(("official:verifier", "grade"))
        return VerifierResult(rewards={"reward": 1.0})

    async def cleanup(attempt_dir, initial_images):
        assert (tmp_path / "run/logs/run_evaluation/test/lhagent/task/report.json").is_file()
        assert (tmp_path / "run/.lhagent/task/1/sessions/test.jsonl").is_file()
        assert (attempt_dir / "task.stdout.log").read_text() == "agent stdout"
        events.append(("images", "cleanup"))

    provider = environment.PrebuiltDockerEnvironment
    monkeypatch.setattr(provider, "_egress_control_kernel_support", classmethod(lambda cls: False))
    for method in ("empty_dirs", "prepare_logs_for_host", "run_healthcheck", "upload_dir"):
        monkeypatch.setattr(provider, method, AsyncMock())
    monkeypatch.setattr(provider, "start", start)
    monkeypatch.setattr(provider, "stop", stop)
    monkeypatch.setattr(provider, "exec", execute)
    monkeypatch.setattr(agent.LHAgent, "setup", AsyncMock())
    monkeypatch.setattr(agent.LHAgent, "run", solve)
    monkeypatch.setattr(ArtifactHandler, "download_artifacts", collect)
    monkeypatch.setattr(ArtifactHandler, "upload_artifacts", upload)
    monkeypatch.setattr(Verifier, "verify", verify)
    monkeypatch.setattr(adapter, "cleanup_images", cleanup)
    result = run(
        adapter.run_trial(
            TaskConfig(path=task.paths.task_dir),
            args=args,
            run_dir=tmp_path / "run",
            bundles={"linux/amd64": tmp_path / "bundle"},
            initial_images=set(),
            native="linux/amd64",
        )
    )
    assert result.exception_info is None
    assert adapter.graded_reward(result) == 1.0
    assert events == [
        ("official:agent", "start"),
        ("official:agent", "solve"),
        ("official:agent", "collect"),
        ("official:agent", "stop"),
        ("official:verifier", "start"),
        ("official:verifier", "upload"),
        ("official:verifier", "grade"),
        ("official:verifier", "stop"),
        ("images", "cleanup"),
    ]


@pytest.mark.parametrize("corrupt", [False, True])
def test_release_download_checksum_and_complete_manifest(tmp_path, monkeypatch, corrupt):
    source = tmp_path / "release.tar.gz"
    with tarfile.open(source, "w:gz") as tar:
        for name, data in {
            "tasks/dataset.toml": b'[[tasks]]\nname = "terminal-bench/example"\n',
            "tasks/example/task.toml": b'schema_version = "1.0"\n',
            "tasks/example/instruction.md": b"Solve the task",
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(adapter, "RELEASE_SHA256", "bad" if corrupt else checksum)

    async def download(*args):
        Path(args[args.index("--output") + 1]).write_bytes(source.read_bytes())
        return SimpleNamespace(wait=AsyncMock(return_value=0))

    downloader = AsyncMock(side_effect=download)
    monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", downloader)
    cache = tmp_path / "cache"
    if corrupt:
        with pytest.raises(ValueError, match="checksum"):
            run(adapter.download_release(cache))
        assert not (cache / "release.sha256").exists()
        return
    assert run(adapter.download_release(cache)) == [cache / "tasks/example"]
    run(adapter.download_release(cache))
    assert downloader.await_count == 1
    (cache / "tasks/example/instruction.md").unlink()
    run(adapter.download_release(cache))
    assert downloader.await_count == 2


def test_compose_requires_prebuilt_images_for_every_service(monkeypatch):
    instance = object.__new__(environment.PrebuiltDockerEnvironment)
    instance.task_images = set()
    parent = AsyncMock(
        return_value=SimpleNamespace(
            stdout=json.dumps(
                {
                    "services": {
                        "main": {"image": "official:agent"},
                        "db": {"image": "official:db"},
                    }
                }
            )
        )
    )
    monkeypatch.setattr(environment.DockerEnvironment, "_run_docker_compose_command", parent)
    run(instance._run_docker_compose_command(["up", "-d", "db"]))
    assert instance.task_images == {"official:agent", "official:db"}
    assert parent.call_args.args[0] == ["up", "--no-build", "-d", "db"]
    parent.return_value.stdout = json.dumps({"services": {"db": {"build": "."}}})
    with pytest.raises(ValueError, match="prebuilt"):
        run(instance._run_docker_compose_command(["up", "-d"]))
    with pytest.raises(RuntimeError, match="builds are disabled"):
        run(instance._run_docker_compose_command(["build"]))
