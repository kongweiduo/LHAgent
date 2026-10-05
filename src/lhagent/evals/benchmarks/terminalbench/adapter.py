"""Run Terminal-Bench 4.0 through Harbor, one task and one attempt at a time."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import re
import shutil
import tarfile
import tempfile
import tomllib
import uuid
from pathlib import Path

from harbor.models.task.config import TaskOS
from harbor.models.task.verifier_mode import resolve_verifier_environment_definition
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.registry.client.package import PackageDatasetClient
from harbor.trial.trial import Trial

from .agent import discover_bundles
from .environment import cleanup_images, docker

DATASET = "terminal-bench/terminal-bench"
DATASET_VERSION = "4.0.0"
RELEASE_SHA256 = "6d2c57cbcb1a75b5cdc0b0f989747fa68cdc65df8ff0a6893045a70ced7e668e"
RELEASE_URL = (
    "https://github.com/harbor-framework/terminal-bench/releases/download/v4.0.0/"
    "terminal-bench-prebuilt-v4.0.0.tar.gz"
)
RELEASE_DOWNLOAD_API = (
    "https://api.github.com/repos/harbor-framework/terminal-bench/"
    "releases/assets/530865343?download=1"
)


def release_task_paths(root: Path) -> list[Path]:
    """Use the release manifest rather than discovering unrelated nested TOML files."""
    dataset_root = root / "tasks"
    with (dataset_root / "dataset.toml").open("rb") as stream:
        manifest = tomllib.load(stream)
    names = [item["name"].split("/")[-1] for item in manifest["tasks"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("invalid official release task manifest")
    if any(re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None or name in {".", ".."} for name in names):
        raise ValueError("unsafe release task name")
    paths = [dataset_root / name for name in sorted(names)]
    if any(
        not (path / "task.toml").is_file() or not (path / "instruction.md").is_file()
        for path in paths
    ):
        raise ValueError("official release task cache is incomplete")
    return paths


async def download_release(cache_dir: Path):
    """Cache the pinned official prebuilt release, checking its published digest."""
    marker = cache_dir / "release.sha256"
    if marker.is_file() and marker.read_text().strip() == RELEASE_SHA256:
        try:
            return release_task_paths(cache_dir)
        except (OSError, ValueError, KeyError):
            pass
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tb4-download-", dir=cache_dir.parent) as directory:
        archive = Path(directory) / "release.tar.gz"
        process = await asyncio.create_subprocess_exec(
            "curl",
            "--fail",
            "--location",
            "--retry",
            "2",
            "--connect-timeout",
            "30",
            "--max-time",
            "1800",
            "--output",
            str(archive),
            "--header",
            "Accept: application/octet-stream",
            RELEASE_DOWNLOAD_API,
        )
        try:
            status = await process.wait()
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        if status:
            raise RuntimeError("Failed to download the official Terminal-Bench 4.0 release")
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != RELEASE_SHA256:
            raise ValueError("Official Terminal-Bench release checksum mismatch")
        extracted = Path(directory) / "tasks"
        extracted.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extracted, filter="data")
        paths = release_task_paths(extracted)
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(extracted, cache_dir, dirs_exist_ok=True)
        marker.write_text(RELEASE_SHA256 + "\n")
    return [cache_dir / path.relative_to(extracted) for path in paths]


def task_id(task: TaskConfig) -> str:
    return task.get_task_id().get_name().split("/")[-1]


def select_tasks(tasks, *, count=None, task_ids=None, seed=0):
    items = sorted(tasks, key=task_id)
    names = [task_id(task) for task in items]
    if len(names) != len(set(names)):
        raise ValueError("duplicate task IDs")
    if any(re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None or name in {".", ".."} for name in names):
        raise ValueError("unsafe task ID")
    if count is not None and task_ids:
        raise ValueError("count and task_ids cannot be combined")
    if task_ids:
        by_id = dict(zip(names, items, strict=True))
        missing = set(task_ids) - by_id.keys()
        if missing:
            raise ValueError(f"task IDs not found: {' '.join(sorted(missing))}")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("duplicate requested task IDs")
        return [by_id[name] for name in task_ids]
    if count is not None:
        if not 1 <= count <= len(items):
            raise ValueError(f"count must be between 1 and {len(items)}")
        random.Random(seed).shuffle(items)
        return items[:count]
    return items


async def load_tasks(args):
    if args.dataset_path:
        root = args.dataset_path.resolve()
        paths = [root] if (root / "task.toml").is_file() else sorted(root.iterdir())
        return [TaskConfig(path=path) for path in paths if (path / "task.toml").is_file()], None
    if (args.dataset, args.dataset_version) == (DATASET, DATASET_VERSION):
        paths = await download_release(Path.home() / ".cache/lhagent/terminalbench/4.0.0")
        return [TaskConfig(path=path) for path in paths], {
            "source": RELEASE_URL,
            "sha256": RELEASE_SHA256,
        }
    metadata = await PackageDatasetClient().get_dataset_metadata(
        f"{args.dataset}@{args.dataset_version}"
    )
    tasks = [TaskConfig(name=f"{item.org}/{item.name}", ref=item.ref) for item in metadata.task_ids]
    return tasks, json.loads(metadata.model_dump_json())


def validate_prebuilt_task(task):
    if task.config.environment.os != TaskOS.LINUX:
        raise ValueError("LHAgent bundles currently support Linux tasks only")
    if not task.config.environment.docker_image:
        raise ValueError("task has no official prebuilt agent image; local builds disabled")
    steps = task.config.steps or [None]
    for step in steps:
        definition = resolve_verifier_environment_definition(task.config, task.paths, step)
        if definition is not None and not definition.config.docker_image:
            raise ValueError("task has no official prebuilt verifier image; local builds disabled")


def graded_reward(result) -> float:
    if result.exception_info is not None:
        raise RuntimeError(result.exception_info.exception_message)
    rewards = result.verifier_result.rewards if result.verifier_result else None
    value = rewards.get("reward") if rewards else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError("Official grading did not produce a finite reward")
    return float(value)


async def run_trial(task, *, args, run_dir, bundles, initial_images, native):
    name = task_id(task)
    attempt_dir = run_dir / "logs/attempts" / name / "1"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = run_dir / ".lhagent" / name / "1"
    agent_config = AgentConfig(
        import_path="lhagent.evals.benchmarks.terminalbench.agent:LHAgent",
        model_name=tomllib.loads(args.config.read_text())["coding_agent"].get("model"),
        override_timeout_sec=None if args.official_timeouts else args.timeout + 60,
        kwargs={
            "bundles": {key: str(path) for key, path in bundles.items()},
            "config": str(args.config.resolve()),
            "trace_dir": str(trace_dir),
            "timeout": args.timeout,
            "instruction": args.instruction,
        },
    )
    config = TrialConfig(
        task=task,
        trial_name="harbor",
        trials_dir=attempt_dir,
        agent=agent_config,
        environment=EnvironmentConfig(
            import_path="lhagent.evals.benchmarks.terminalbench.environment:PrebuiltDockerEnvironment",
            force_build=False,
            delete=True,
            kwargs={
                "initial_images": sorted(initial_images),
                "cleanup_errors_path": str(attempt_dir / "cleanup-errors.json"),
                "task_images_path": str(attempt_dir / "images.json"),
                "platforms_path": str(run_dir / "logs" / f"{args.run_id}.platforms.json"),
                "bundle_platforms": sorted(bundles),
                "native_platform": native,
            },
        ),
        verifier=VerifierConfig(
            override_timeout_sec=None if args.official_timeouts else args.test_timeout
        ),
    )
    trial = await Trial.create(config)
    try:
        validate_prebuilt_task(trial.task)
        if args.official_timeouts:
            trial.agent.timeout = trial.agent_timeout_sec
        result = await trial.run()
        report_dir = run_dir / "logs/run_evaluation" / args.run_id / "lhagent" / name
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "report.json").write_text(result.model_dump_json(indent=2))
        return result
    finally:
        try:
            native_logs = attempt_dir / "harbor/agent/native"
            if native_logs.exists():
                shutil.copytree(native_logs, trace_dir, dirs_exist_ok=True)
            for filename in ("stdout.log", "stderr.log"):
                source = attempt_dir / "harbor/agent" / filename
                if source.exists():
                    shutil.copy2(source, attempt_dir / f"{name}.{filename}")
        finally:
            await cleanup_images(attempt_dir, initial_images)
            trial._close_logger_handler()


def summarize_run(run_dir, selected, run_id, results, errors, *, dataset, timeout_profile):
    resolved, unresolved = [], []
    for name, result in results.items():
        try:
            reward = graded_reward(result)
        except RuntimeError:
            continue
        (resolved if reward == 1.0 else unresolved).append(name)
    completed = set(resolved + unresolved)
    summary = {
        "dataset": dataset,
        "timeout_profile": timeout_profile,
        "total_instances": len(selected),
        "resolved_ids": sorted(resolved),
        "unresolved_ids": sorted(unresolved),
        "incomplete_ids": sorted({task_id(task) for task in selected} - completed),
        "error_ids": sorted(errors),
        "accuracy": len(resolved) / len(selected),
    }
    report_path = run_dir / "logs/run_evaluation" / run_id / "results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                **summary,
                "results": [json.loads(result.model_dump_json()) for result in results.values()],
            },
            indent=2,
        )
    )
    return report_path


async def evaluate(args, bundles):
    tasks, metadata = await load_tasks(args)
    selected = select_tasks(tasks, count=args.count, task_ids=args.task_ids, seed=args.seed)
    if not selected:
        raise ValueError("selected dataset is empty")
    initial_images = set((await docker("image", "ls", "--no-trunc", "-q")).split())
    arch = (await docker("info", "--format", "{{.Architecture}}")).strip()
    native = "linux/" + {"aarch64": "arm64", "x86_64": "amd64"}.get(arch, arch)
    run_dir = (args.output_dir / args.run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()
    profile = {
        "mode": "official" if args.official_timeouts else "swebench",
        "agent_timeout_sec": None if args.official_timeouts else args.timeout,
        "verifier_timeout_sec": None if args.official_timeouts else args.test_timeout,
    }
    dataset = (
        str(args.dataset_path.resolve())
        if args.dataset_path
        else (f"{args.dataset}@{args.dataset_version}")
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "resolved_dataset": metadata,
                "task_ids": [task_id(task) for task in selected],
                "seed": args.seed,
                "n_attempts": 1,
                "n_concurrent_trials": 1,
                "timeout_profile": profile,
            },
            indent=2,
        )
    )
    failures_path = run_dir / "logs" / f"{args.run_id}.failures.json"
    errors, results = {}, {}
    with (run_dir / "predictions.jsonl").open("w") as output:
        for number, task in enumerate(selected, 1):
            name = task_id(task)
            print(f"[{number}/{len(selected)}] {name}", flush=True)
            attempt_dir = run_dir / "logs/attempts" / name / "1"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            try:
                result = await run_trial(
                    task,
                    args=args,
                    run_dir=run_dir,
                    bundles=bundles,
                    initial_images=initial_images,
                    native=native,
                )
                results[name] = result
                reward = graded_reward(result)
                prediction = json.dumps(
                    {
                        "instance_id": name,
                        "model_name_or_path": "lhagent",
                        "reward": reward,
                        "resolved": reward == 1.0,
                    }
                )
                (attempt_dir / "prediction.jsonl").write_text(prediction + "\n")
                output.write(prediction + "\n")
                output.flush()
            except Exception as exc:
                errors[name] = str(exc)
                (attempt_dir / "error.log").write_text(str(exc))
                print(f"{name}: {exc}", flush=True)
            finally:
                cleanup_path = attempt_dir / "cleanup-errors.json"
                if cleanup_path.exists():
                    errors[name] = (
                        errors.get(name, "")
                        + "\nDocker cleanup failed: "
                        + ("; ".join(json.loads(cleanup_path.read_text())))
                    )
                failures_path.write_text(json.dumps(errors, indent=2))
            if cleanup_path.exists():
                print("Stopping before the next task because Docker cleanup failed.", flush=True)
                break
    report = summarize_run(
        run_dir, selected, args.run_id, results, errors, dataset=dataset, timeout_profile=profile
    )
    print(f"Terminal-Bench 4.0 report: {report}", flush=True)
    return int(bool(errors))


def build_parser():
    parser = argparse.ArgumentParser(description="Run LHAgent on Terminal-Bench 4.0 using Harbor")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--dataset-path", type=Path, help="local exported Harbor task packages")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--count", type=int)
    mode.add_argument("--task-ids", "--instance-ids", nargs="+")
    mode.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instruction")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--test-timeout", type=int, default=1800)
    parser.add_argument(
        "--official-timeouts",
        action="store_true",
        help="use official task timeouts instead of SWE's 1800-second defaults",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path, default=Path("terminalbench"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.timeout < 1 or args.test_timeout < 1:
        raise SystemExit("timeouts must be positive")
    if not args.config.is_file():
        raise SystemExit("--config must be a file")
    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    if "coding_agent" not in config:
        raise SystemExit("config must include [coding_agent]")
    cwd = config["coding_agent"].get("cwd")
    if cwd is not None and (not isinstance(cwd, str) or not cwd.startswith("/")):
        raise SystemExit("config cwd must be an absolute container path, or omitted")
    args.run_id = args.run_id or "lhagent-" + uuid.uuid4().hex[:10]
    if re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id) is None:
        raise SystemExit("--run-id must contain only ASCII letters, digits, '_' or '-'")
    if (args.output_dir / args.run_id).exists():
        raise SystemExit("run directory already exists; choose another --run-id")
    try:
        return asyncio.run(evaluate(args, discover_bundles(args.bundle)))
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
