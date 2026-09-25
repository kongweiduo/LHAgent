"""在 SWE-bench Lite 或 Verified 上运行 LHAgent，并调用官方评分器。

适配器使用 Docker CLI：任务镜像已包含仓库和测试依赖，运行时将
LHAgent 运行包复制到每个新建的容器中。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tomllib
import uuid
from collections.abc import Sequence
from contextlib import chdir
from pathlib import Path
from typing import Any

DATASETS = {
    "lite": "SWE-bench/SWE-bench_Lite",
    "verified": "SWE-bench/SWE-bench_Verified",
}
DEFAULT_INSTRUCTION = (
    "Implement the requested change in this repository. Read the issue below, "
    "inspect the code and tests, make the fix, and verify it with focused tests.\n\n"
)


def select_instances(
    instances: Sequence[dict[str, Any]],
    *,
    count: int | None = None,
    instance_ids: Sequence[str] | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """选择全部任务、可复现的随机样本或指定任务。

    ``count`` 与 ``instance_ids`` 互斥；指定任务按传入顺序返回，
    仅随机抽样会打乱原始顺序。
    """
    if count is not None and instance_ids:
        raise ValueError("count and instance_ids cannot be combined")
    items = list(instances)
    if instance_ids:
        wanted = list(instance_ids)
        by_id = {item["instance_id"]: item for item in items}
        missing = [item_id for item_id in wanted if item_id not in by_id]
        if missing:
            raise ValueError("instance IDs not found: " + " ".join(missing))
        return [by_id[item_id] for item_id in wanted]
    if count is None:
        return items
    if count < 1:
        raise ValueError("count must be greater than zero")
    if count > len(items):
        raise ValueError(f"count {count} exceeds dataset size {len(items)}")
    selected = items[:]
    random.Random(seed).shuffle(selected)
    return selected[:count]


def _run(command: Sequence[str], *, capture: bool = False) -> str:
    result = subprocess.run(command, check=True, text=True, capture_output=capture)
    return result.stdout if capture else ""


def _docker_exec(container: str, command: Sequence[str], *, workdir: str | None = None) -> str:
    args = ["docker", "exec"]
    if workdir:
        args += ["-w", workdir]
    args += [container, *command]
    return _run(args, capture=True)


def _instance_image(instance: dict[str, Any]) -> str:
    image = instance.get("image")
    if image:
        return str(image)
    raise ValueError(f"{instance['instance_id']} has no image in the dataset")


def discover_bundles(path: Path) -> dict[str, Path]:
    """读取运行包内的 TARGET 标记，不依赖文件名判断平台。"""
    paths = sorted(path.glob("*.tar.gz")) if path.is_dir() else [path]
    bundles = {}
    for archive in paths:
        if not archive.is_file():
            raise ValueError(f"bundle does not exist: {archive}")
        target = _run(["tar", "-xOzf", str(archive), "lhagent/TARGET"], capture=True).strip()
        if target not in {"linux/amd64", "linux/arm64"}:
            raise ValueError(f"unsupported bundle target in {archive}: {target}")
        if target in bundles:
            raise ValueError(
                f"multiple bundles for {target}; use a directory with one per platform"
            )
        bundles[target] = archive
    if not bundles:
        raise ValueError(f"no .tar.gz bundles in {path}")
    return bundles


def image_platforms(image: str) -> set[str]:
    """检查镜像清单；不可用时回退到本地镜像信息。"""
    try:
        data = json.loads(_run(["docker", "manifest", "inspect", "--verbose", image], capture=True))
    except subprocess.CalledProcessError:
        data = json.loads(_run(["docker", "image", "inspect", image], capture=True))
    entries = data if isinstance(data, list) else data.get("manifests", [data])
    platforms = set()
    for entry in entries:
        info = entry.get("Descriptor", {}).get("platform", entry.get("platform", entry))
        os_name = info.get("os", info.get("Os"))
        arch = info.get("architecture", info.get("Architecture"))
        if os_name == "linux" and arch in {"amd64", "arm64"}:
            platforms.add(f"linux/{arch}")
    if not platforms:
        raise ValueError(f"no supported Linux platform found for image {image}")
    return platforms


def select_platform(platforms: set[str], bundles: dict[str, Path], native: str) -> str:
    available = platforms & bundles.keys()
    for platform in (native, "linux/amd64", "linux/arm64"):
        if platform in available:
            return platform
    raise ValueError(f"image platforms {sorted(platforms)} have no matching bundle")


def run_agent(
    instance: dict[str, Any],
    *,
    bundle: Path,
    config: Path,
    instruction: str | None,
    timeout: int,
    log_dir: Path,
    trace_dir: Path,
    platform: str,
    container_label: str = "lhagent.swebench=standalone",
) -> str:
    """在任务镜像中运行 LHAgent，返回统一格式的补丁。"""
    instance_id = instance["instance_id"]
    name = "lhagent-swe-" + instance_id.lower().replace("/", "-") + "-" + uuid.uuid4().hex[:8]
    image = _instance_image(instance)
    prompt = (
        instruction
        if instruction is not None
        else DEFAULT_INSTRUCTION + str(instance.get("problem_statement", ""))
    )
    trace_ready = False
    try:
        _run(
            [
                "docker",
                "create",
                "--platform",
                platform,
                "--name",
                name,
                "--label",
                container_label,
                "-e",
                "LHAGENT_API_KEY",
                "-e",
                "LHAGENT_BASE_URL",
                image,
                "tail",
                "-f",
                "/dev/null",
            ]
        )
        _run(["docker", "start", name])
        _run(["docker", "cp", str(bundle), f"{name}:/tmp/lhagent.tar.gz"])
        _docker_exec(name, ["sh", "-c", "mkdir -p /opt && tar -xzf /tmp/lhagent.tar.gz -C /opt"])
        _docker_exec(name, ["/opt/lhagent/lhagent", "--bundle-check"])
        _run(["docker", "cp", str(config), f"{name}:/tmp/lhagent.toml"])
        _docker_exec(name, ["mkdir", "-p", "/tmp/.lhagent"])
        trace_ready = True
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-w",
                "/tmp",
                name,
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                # Docker exec 不读取镜像的 .bashrc；先激活任务环境，
                # 让 Agent 工具继承仓库的 Python 与依赖。
                "if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then "
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                'conda activate testbed || exit; fi; exec "$@"',
                "lhagent-swebench",
                "/opt/lhagent/lhagent",
                "--config",
                "/tmp/lhagent.toml",
                "--instruction",
                prompt,
            ],
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{instance_id}.stdout.log").write_text(result.stdout)
        (log_dir / f"{instance_id}.stderr.log").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(
                f"LHAgent exited {result.returncode}; see {log_dir / (instance_id + '.stderr.log')}"
            )
        # 使用 intent-to-add 纳入新文件，不修改其内容。
        _docker_exec(name, ["git", "add", "-N", "."], workdir="/testbed")
        return _docker_exec(
            name,
            ["git", "-c", "core.fileMode=false", "diff", "HEAD", "--binary"],
            workdir="/testbed",
        )
    finally:
        try:
            if trace_ready:
                trace_dir.mkdir(parents=True, exist_ok=True)
                _run(["docker", "cp", f"{name}:/tmp/.lhagent/.", str(trace_dir)])
        finally:
            subprocess.run(["docker", "rm", "-f", name], check=False, stdout=subprocess.DEVNULL)


def cleanup_task(image: str, container_label: str, initial_images: set[str]) -> None:
    """清理本轮容器，以及运行前不存在的任务镜像。

    不强制删除镜像，以免影响其他容器；清理失败则停止后续任务，
    避免继续占用磁盘空间。
    """
    containers = _run(
        ["docker", "ps", "-aq", "--filter", f"label={container_label}"], capture=True
    ).split()
    for container in containers:
        _run(["docker", "rm", "-f", container])
    image_ids = set(
        _run(["docker", "image", "ls", "--no-trunc", "-q", image], capture=True).split()
    )
    if image_ids and not image_ids.intersection(initial_images):
        _run(["docker", "image", "rm", image])


def summarize_run(predictions_path: Path, selected: list[dict[str, Any]], run_id: str) -> Path:
    """汇总已保存的官方报告，不启动容器或拉取镜像。"""
    from swebench.harness.reporting import make_run_report

    predictions = {
        item["instance_id"]: item
        for line in predictions_path.read_text().splitlines()
        if (item := json.loads(line))
    }
    with chdir(predictions_path.parent):
        return Path(make_run_report(predictions, selected, run_id)).resolve()


def _load_dataset(name: str) -> list[dict[str, Any]]:
    if name.endswith((".json", ".jsonl")):
        path = Path(name)
        if name.endswith(".jsonl"):
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return json.loads(path.read_text())
    from datasets import load_dataset

    return [dict(row) for row in load_dataset(name, split="test")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LHAgent on SWE-bench Lite or Verified")
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="LHAgent tar.gz bundle or directory with one bundle per platform",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="LHAgent config with cwd=/testbed"
    )
    parser.add_argument("--variant", choices=DATASETS, default="lite", help="SWE-bench variant")
    parser.add_argument("--dataset", help="override the dataset selected by --variant")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--count", type=int, help="randomly select N instances")
    mode.add_argument("--instance-ids", nargs="+", help="run these instance IDs")
    mode.add_argument("--all", action="store_true", help="run the complete test split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instruction", help="override the generated issue prompt")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("swebench"))
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="grader worker limit (only one task is submitted at a time)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_name = args.dataset or DATASETS[args.variant]
    if not args.bundle.exists() or not args.config.is_file():
        raise SystemExit("--bundle must be a file/directory and --config must be a file")
    if args.timeout < 1 or args.max_workers < 1:
        raise SystemExit("--timeout and --max-workers must be positive")
    with args.config.open("rb") as stream:
        config_data = tomllib.load(stream)
    if config_data.get("coding_agent", {}).get("cwd") != "/testbed":
        raise SystemExit("config [coding_agent].cwd must be /testbed")
    try:
        bundles = discover_bundles(args.bundle)
    except (ValueError, subprocess.SubprocessError) as exc:
        raise SystemExit(str(exc)) from exc
    arch = _run(["docker", "info", "--format", "{{.Architecture}}"], capture=True).strip()
    native = "linux/" + {"aarch64": "arm64", "x86_64": "amd64"}.get(arch, arch)
    dataset = _load_dataset(dataset_name)
    selected = select_instances(
        dataset,
        count=args.count,
        instance_ids=args.instance_ids,
        seed=args.seed,
    )
    if not selected:
        raise SystemExit("selected dataset is empty")
    missing_images = [item["instance_id"] for item in selected if not item.get("image")]
    if missing_images:
        raise SystemExit("selected instances have no image: " + " ".join(missing_images))
    run_id = args.run_id or "lhagent-" + uuid.uuid4().hex[:10]
    if re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise SystemExit("--run-id must contain only ASCII letters, digits, '_' or '-'")
    run_dir = (args.output_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    output_path = run_dir / "predictions.jsonl"
    log_dir = run_dir / "logs"
    initial_images = set(_run(["docker", "image", "ls", "--no-trunc", "-q"], capture=True).split())
    # 独立于用户指定的 run ID，避免并发运行复用容器标签。
    container_label = "lhagent.swebench=" + uuid.uuid4().hex
    log_dir.mkdir(parents=True, exist_ok=True)
    plan_path = log_dir / f"{run_id}.platforms.json"
    failures_path = log_dir / f"{run_id}.failures.json"
    plans = {}
    errors = {}
    with output_path.open("w", encoding="utf-8") as output:
        for number, instance in enumerate(selected, 1):
            instance_id = instance["instance_id"]
            image = _instance_image(instance)
            print(f"[{number}/{len(selected)}] {instance_id}", flush=True)
            cleanup_failed = False
            try:
                if image not in plans:
                    plans[image] = select_platform(image_platforms(image), bundles, native)
                platform = plans[image]
                plan_path.write_text(json.dumps(plans, indent=2))
                print(f"  {platform} -> {bundles[platform].name}", flush=True)
                patch = run_agent(
                    instance,
                    bundle=bundles[platform],
                    platform=platform,
                    config=args.config,
                    instruction=args.instruction,
                    timeout=args.timeout,
                    log_dir=log_dir,
                    trace_dir=run_dir / ".lhagent" / instance_id,
                    container_label=container_label,
                )
                output.write(
                    json.dumps(
                        {
                            "instance_id": instance_id,
                            "model_name_or_path": "lhagent",
                            "model_patch": patch,
                        }
                    )
                    + "\n"
                )
                # 下一题开始前，评分器进程需要读取已写入的预测文件。
                output.flush()
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "lhagent.evals.benchmarks.swebench.grader",
                        "--platform-plan",
                        str(plan_path),
                        "--container-label",
                        container_label,
                        "--dataset_name",
                        dataset_name,
                        "--split",
                        "test",
                        "--predictions_path",
                        str(output_path),
                        "--run_id",
                        run_id,
                        "--max_workers",
                        str(args.max_workers),
                        "--timeout",
                        str(args.timeout),
                        "--instance_ids",
                        instance_id,
                    ],
                    check=True,
                    cwd=run_dir,
                )
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                errors[instance_id] = str(exc)
                (log_dir / f"{instance_id}.error.log").write_text(str(exc))
                print(f"{instance_id}: {exc}", file=sys.stderr, flush=True)
            finally:
                try:
                    cleanup_task(image, container_label, initial_images)
                except (OSError, subprocess.SubprocessError) as exc:
                    cleanup_failed = True
                    message = f"Docker cleanup failed: {exc}"
                    errors[instance_id] = errors.get(instance_id, "") + "\n" + message
                    (log_dir / f"{instance_id}.error.log").write_text(errors[instance_id])
                    print(message, file=sys.stderr, flush=True)
                failures_path.write_text(json.dumps(errors, indent=2))
            if cleanup_failed:
                print("Stopping before the next task because Docker cleanup failed.", flush=True)
                break
    report_path = summarize_run(output_path, selected, run_id)
    report = json.loads(report_path.read_text())
    for instance_id in report.get("error_ids", []):
        errors.setdefault(instance_id, "Official grading failed; see the evaluation logs")
    failures_path.write_text(json.dumps(errors, indent=2))
    print(f"SWE-bench report: {report_path}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
