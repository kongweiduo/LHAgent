from __future__ import annotations

import json
import subprocess

import pytest

from lhagent.evals.benchmarks.swebench import adapter
from lhagent.evals.benchmarks.swebench.adapter import run_agent, select_instances

GRADE_LOG_DIR = adapter.RUN_EVALUATION_LOG_DIR


@pytest.fixture
def instances():
    return [{"instance_id": f"repo__issue-{number}"} for number in range(5)]


def test_select_all_random_and_named(instances):
    assert select_instances(instances) == instances
    first = select_instances(instances, count=3, seed=42)
    assert first == select_instances(instances, count=3, seed=42)
    assert len({item["instance_id"] for item in first}) == 3
    assert select_instances(instances, instance_ids=["repo__issue-4", "repo__issue-1"]) == [
        instances[4],
        instances[1],
    ]


@pytest.mark.parametrize("kwargs", [{"count": 0}, {"count": 6}, {"instance_ids": ["missing"]}])
def test_reject_invalid_selection(instances, kwargs):
    with pytest.raises(ValueError):
        select_instances(instances, **kwargs)


def test_run_agent_captures_diff_without_agent_stdout(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:2] == ["docker", "exec"] and "git" in command:
            if "diff" in command:
                return subprocess.CompletedProcess(command, 0, "diff --git a/a b/a\n", "")
        if command[:2] == ["docker", "exec"] and "/opt/lhagent/lhagent" in command:
            return subprocess.CompletedProcess(command, 0, "model response", "session trace")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    patch = run_agent(
        {
            "instance_id": "repo__issue-1",
            "image": "example/image:latest",
            "problem_statement": "Fix it",
        },
        bundle=tmp_path / "bundle.tar.gz",
        platform="linux/amd64",
        config=tmp_path / "config.toml",
        instruction=None,
        timeout=60,
        log_dir=tmp_path / "logs",
        trace_dir=tmp_path / ".lhagent/repo__issue-1",
    )
    assert commands[0][2:4] == ["--platform", "linux/amd64"]
    assert patch == "diff --git a/a b/a\n"
    assert (tmp_path / "logs/repo__issue-1.stdout.log").read_text() == "model response"
    assert any(
        command[:3] == ["docker", "exec", "-w"] and "/tmp" in command for command in commands
    )
    assert any(
        command[:2] == ["docker", "cp"]
        and command[2].endswith(":/tmp/.lhagent/.")
        and command[3] == str(tmp_path / ".lhagent/repo__issue-1")
        for command in commands
    )
    assert any("diff" in command and "HEAD" in command for command in commands)
    assert any(command[:3] == ["docker", "rm", "-f"] for command in commands)


def write_grade(run_dir, name, resolved=True):
    folder = run_dir / GRADE_LOG_DIR / "test/lhagent" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps({name: {"resolved": resolved}}))


@pytest.fixture
def mock_summary(monkeypatch, tmp_path):
    def summarize(*args):
        path = tmp_path / "results.json"
        path.write_text('{"error_ids": []}')
        return path

    monkeypatch.setattr(adapter, "summarize_run", summarize)
    monkeypatch.setattr(adapter, "cleanup_task", lambda *args: None)


@pytest.mark.parametrize(
    ("variant_args", "expected_dataset"),
    [
        ([], adapter.DATASETS["lite"]),
        (["--variant", "verified"], adapter.DATASETS["verified"]),
        (["--variant", "verified", "--dataset", "custom.jsonl"], "custom.jsonl"),
    ],
)
def test_main_passes_only_selected_ids_to_official_grader(
    monkeypatch, tmp_path, mock_summary, variant_args, expected_dataset
):
    bundle = tmp_path / "bundle.tar.gz"
    bundle.touch()
    config = tmp_path / "config.toml"
    config.write_text('[coding_agent]\ncwd = "/testbed"\n')
    instances = [
        {"instance_id": f"repo__issue-{number}", "image": f"swebench/x86_64.{number}"}
        for number in range(3)
    ]
    loaded_datasets = []

    def load_dataset(name):
        loaded_datasets.append(name)
        return instances

    monkeypatch.setattr(adapter, "_load_dataset", load_dataset)
    monkeypatch.setattr(adapter, "_run", lambda *_args, **_kwargs: "linux/amd64\n")
    monkeypatch.setattr(adapter, "image_platforms", lambda _: {"linux/amd64"})
    grader_calls = []
    grader_kwargs = []
    solve_kwargs = []

    def solve(instance, **kwargs):
        solve_kwargs.append(kwargs)
        return instance["instance_id"]

    monkeypatch.setattr(adapter, "run_agent", solve)

    def fake_grader(command, **kwargs):
        grader_calls.append(command)
        grader_kwargs.append(kwargs)
        write_grade(kwargs["cwd"], command[-1])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(adapter.subprocess, "run", fake_grader)
    output = tmp_path / "swebench/test/predictions.jsonl"
    assert (
        adapter.main(
            [
                "--bundle",
                str(bundle),
                "--config",
                str(config),
                "--instance-ids",
                "repo__issue-2",
                "repo__issue-0",
                "--run-id",
                "test",
                "--output-dir",
                str(tmp_path / "swebench"),
                *variant_args,
            ]
        )
        == 0
    )
    assert loaded_datasets == [expected_dataset]
    assert all(call[call.index("--dataset_name") + 1] == expected_dataset for call in grader_calls)
    predictions = [json.loads(line) for line in output.read_text().splitlines()]
    assert [item["instance_id"] for item in predictions] == ["repo__issue-2", "repo__issue-0"]
    assert [call[-2:] for call in grader_calls] == [
        ["--instance_ids", "repo__issue-2"],
        ["--instance_ids", "repo__issue-0"],
    ]
    assert all(kw["cwd"] == output.parent for kw in grader_kwargs)
    assert [kw["trace_dir"] for kw in solve_kwargs] == [
        output.parent / ".lhagent" / name / "1" for name in ("repo__issue-2", "repo__issue-0")
    ]


@pytest.mark.parametrize(
    ("supported", "native", "expected"),
    [
        ({"linux/amd64", "linux/arm64"}, "linux/arm64", "linux/arm64"),
        ({"linux/amd64"}, "linux/arm64", "linux/amd64"),
        ({"linux/arm64"}, "linux/amd64", "linux/arm64"),
    ],
)
def test_pair_bundle_with_image(supported, native, expected, tmp_path):
    bundles = {platform: tmp_path / platform for platform in supported}
    assert adapter.select_platform(supported, bundles, native) == expected


def test_missing_matching_bundle(tmp_path):
    with pytest.raises(ValueError, match="no matching bundle"):
        adapter.select_platform({"linux/arm64"}, {"linux/amd64": tmp_path}, "linux/arm64")


def test_discover_bundles_uses_embedded_target(monkeypatch, tmp_path):
    (tmp_path / "one.tar.gz").touch()
    (tmp_path / "two.tar.gz").touch()
    monkeypatch.setattr(
        adapter,
        "_run",
        lambda command, **_: "linux/arm64" if "one.tar.gz" in command[2] else "linux/amd64",
    )
    assert set(adapter.discover_bundles(tmp_path)) == {"linux/arm64", "linux/amd64"}
    (tmp_path / "three.tar.gz").touch()
    with pytest.raises(ValueError, match="multiple bundles"):
        adapter.discover_bundles(tmp_path)


@pytest.mark.parametrize("as_list", [False, True])
def test_manifest_platforms_ignore_attestations(monkeypatch, as_list):
    entries = [
        {"Descriptor": {"platform": {"os": "linux", "architecture": "amd64"}}},
        {"Descriptor": {"platform": {"os": "unknown", "architecture": "unknown"}}},
    ]
    data = entries if as_list else entries[0]
    monkeypatch.setattr(adapter, "_run", lambda *_, **__: json.dumps(data))
    assert adapter.image_platforms("image") == {"linux/amd64"}


def test_local_image_fallback(monkeypatch):
    def run(command, **kwargs):
        if command[1] == "manifest":
            raise subprocess.CalledProcessError(1, command)
        return json.dumps([{"Os": "linux", "Architecture": "arm64"}])

    monkeypatch.setattr(adapter, "_run", run)
    assert adapter.image_platforms("local") == {"linux/arm64"}


def test_grader_uses_platform_for_each_image(monkeypatch):
    from docker.models.containers import ContainerCollection
    from docker.models.images import ImageCollection

    from lhagent.evals.benchmarks.swebench.grader import docker_platforms

    calls = []
    monkeypatch.setattr(ContainerCollection, "create", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(ImageCollection, "pull", lambda *a, **kw: calls.append(kw))
    with docker_platforms({"arm:latest": "linux/arm64", "amd:latest": "linux/amd64"}):
        ContainerCollection.create(None, "arm:latest")
        ContainerCollection.create(None, image="amd:latest")
        ImageCollection.pull(None, "amd", tag="latest")
        with pytest.raises(ValueError, match="absent"):
            ContainerCollection.create(None, "unknown")
    assert [call["platform"] for call in calls] == [
        "linux/arm64",
        "linux/amd64",
        "linux/amd64",
    ]


def test_failed_instances_are_not_submitted(monkeypatch, tmp_path, mock_summary):
    bundle = tmp_path / "bundle.tar.gz"
    bundle.touch()
    config = tmp_path / "config.toml"
    config.write_text('[coding_agent]\ncwd = "/testbed"\n')
    monkeypatch.setattr(adapter, "discover_bundles", lambda _: {"linux/amd64": bundle})
    monkeypatch.setattr(adapter, "_run", lambda *_, **__: "aarch64")
    monkeypatch.setattr(
        adapter,
        "_load_dataset",
        lambda _: [
            {"instance_id": "bad", "image": "arm"},
            {"instance_id": "good", "image": "amd"},
        ],
    )
    monkeypatch.setattr(
        adapter,
        "image_platforms",
        lambda image: {"linux/arm64" if image == "arm" else "linux/amd64"},
    )
    monkeypatch.setattr(adapter, "run_agent", lambda *_, **__: "patch")
    calls = []

    def grade(command, **kwargs):
        calls.append(command)
        write_grade(kwargs["cwd"], command[-1])

    monkeypatch.setattr(subprocess, "run", grade)
    output = tmp_path / "swebench/test/predictions.jsonl"
    assert (
        adapter.main(
            [
                "--bundle",
                str(bundle),
                "--config",
                str(config),
                "--output-dir",
                str(tmp_path / "swebench"),
                "--run-id",
                "test",
            ]
        )
        == 1
    )
    assert [json.loads(line)["instance_id"] for line in output.read_text().splitlines()] == ["good"]
    assert calls[0][-2:] == ["--instance_ids", "good"]
    assert "bad" in json.loads((tmp_path / "swebench/test/logs/test.failures.json").read_text())


@pytest.mark.parametrize("failure", [None, "solve", "grade", "cleanup"])
def test_task_lifecycle_order_and_failure_cleanup(monkeypatch, tmp_path, mock_summary, failure):
    bundle = tmp_path / "bundle.tar.gz"
    bundle.touch()
    config = tmp_path / "config.toml"
    config.write_text('[coding_agent]\ncwd = "/testbed"\n')
    tasks = [{"instance_id": name, "image": name} for name in ["first", "second"]]
    events = []
    monkeypatch.setattr(adapter, "discover_bundles", lambda _: {"linux/amd64": bundle})
    monkeypatch.setattr(adapter, "_load_dataset", lambda _: tasks)
    monkeypatch.setattr(adapter, "_run", lambda *a, **kw: "")
    monkeypatch.setattr(adapter, "image_platforms", lambda _: {"linux/amd64"})

    def solve(instance, **kwargs):
        name = instance["instance_id"]
        events.append(("solve", name))
        if failure == "solve" and name == "first":
            raise RuntimeError("agent failed")
        return "patch"

    def grade(command, **kwargs):
        name = command[-1]
        events.append(("grade", name))
        # Each attempt has its own prediction file, written before grading.
        predictions = [
            json.loads(line)
            for line in (kwargs["cwd"] / "logs/attempts" / name / "1/prediction.jsonl")
            .read_text()
            .splitlines()
        ]
        assert predictions[-1]["instance_id"] == name
        assert "--container-label" in command
        if failure == "grade" and name == "first":
            raise subprocess.CalledProcessError(1, command)
        write_grade(kwargs["cwd"], name)

    def cleanup(image, label, initial):
        events.append(("cleanup", image))
        if failure == "cleanup":
            raise subprocess.CalledProcessError(1, ["docker", "image", "rm", image])

    monkeypatch.setattr(adapter, "run_agent", solve)
    monkeypatch.setattr(adapter.subprocess, "run", grade)
    monkeypatch.setattr(adapter, "cleanup_task", cleanup)
    result = adapter.main(
        [
            "--bundle",
            str(bundle),
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "swebench"),
            "--run-id",
            "test",
            "--retries",
            "0",
        ]
    )
    expected = [("solve", "first")]
    if failure != "solve":
        expected.append(("grade", "first"))
    expected.append(("cleanup", "first"))
    if failure != "cleanup":
        expected += [("solve", "second"), ("grade", "second"), ("cleanup", "second")]
    assert events == expected
    assert result == (1 if failure else 0)
    errors = json.loads((tmp_path / "swebench/test/logs/test.failures.json").read_text())
    assert ("first" in errors) == bool(failure)


@pytest.mark.parametrize(
    "image_ids,initial,removed",
    [
        ("sha256:new", {"sha256:old"}, True),
        ("sha256:old", {"sha256:old"}, False),
        ("", {"sha256:old"}, False),
    ],
)
def test_cleanup_scopes_containers_and_preserves_existing_images(
    monkeypatch, image_ids, initial, removed
):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1] == "ps":
            return "owned-container\n"
        if command[1:3] == ["image", "ls"]:
            return image_ids
        return ""

    monkeypatch.setattr(adapter, "_run", run)
    adapter.cleanup_task("task:latest", "lhagent.swebench=unique", initial)
    assert calls[0] == ["docker", "ps", "-aq", "--filter", "label=lhagent.swebench=unique"]
    assert calls[1] == ["docker", "rm", "-f", "owned-container"]
    assert (["docker", "image", "rm", "task:latest"] in calls) == removed


def test_grader_labels_only_created_containers(monkeypatch):
    from docker.models.containers import ContainerCollection

    from lhagent.evals.benchmarks.swebench.grader import docker_platforms

    calls = []
    monkeypatch.setattr(ContainerCollection, "create", lambda *a, **kw: calls.append(kw))
    with docker_platforms({"image": "linux/amd64"}, "lhagent.swebench=unique"):
        ContainerCollection.create(None, "image", labels={"existing": "value"})
    assert calls[0]["labels"] == {"existing": "value", "lhagent.swebench": "unique"}


def test_summary_aggregates_saved_reports_without_docker(monkeypatch, tmp_path):
    run_dir = tmp_path / "swebench/test"
    run_dir.mkdir(parents=True)
    predictions = run_dir / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            json.dumps(
                {
                    "instance_id": name,
                    "model_name_or_path": "lhagent",
                    "model_patch": "patch",
                }
            )
            for name in ["first", "second"]
        )
    )
    for name, resolved in [("first", True), ("second", False)]:
        folder = run_dir / GRADE_LOG_DIR / "test/lhagent" / name
        folder.mkdir(parents=True)
        (folder / "report.json").write_text(json.dumps({name: {"resolved": resolved}}))
    report_path = adapter.summarize_run(
        predictions, [{"instance_id": n} for n in ["first", "second", "failed"]], "test"
    )
    report = json.loads(report_path.read_text())
    assert report_path == run_dir / GRADE_LOG_DIR / "test/results.json"
    assert report["resolved_ids"] == ["first"]
    assert report["unresolved_ids"] == ["second"]
    assert report["incomplete_ids"] == ["failed"]
    assert report["total_instances"] == 3


def test_retry_summary_counts_only_successful_grades(monkeypatch, tmp_path):
    bundle = tmp_path / "bundle.tar.gz"
    bundle.touch()
    config = tmp_path / "config.toml"
    config.write_text('[coding_agent]\ncwd = "/testbed"\n')
    names = ["recovered", "unresolved", "exhausted", "missing"]
    monkeypatch.setattr(adapter, "discover_bundles", lambda _: {"linux/amd64": bundle})
    monkeypatch.setattr(
        adapter, "_load_dataset", lambda _: [{"instance_id": name, "image": name} for name in names]
    )
    monkeypatch.setattr(adapter, "_run", lambda *a, **kw: "")
    monkeypatch.setattr(adapter, "image_platforms", lambda _: {"linux/amd64"})
    monkeypatch.setattr(adapter, "cleanup_task", lambda *a: None)
    attempts = {name: 0 for name in names}

    def solve(instance, **kwargs):
        name = instance["instance_id"]
        attempts[name] += 1
        if name == "exhausted":
            raise RuntimeError("agent failed")
        return "patch"

    def grade(command, **kwargs):
        name = command[-1]
        prediction = json.loads(
            (
                kwargs["cwd"] / "logs/attempts" / name / str(attempts[name]) / "prediction.jsonl"
            ).read_text()
        )
        assert prediction["instance_id"] == name
        if name == "recovered" and attempts[name] == 1:
            folder = kwargs["cwd"] / GRADE_LOG_DIR / "test/lhagent" / name
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "report.json").write_text("invalid json")
            return
        if name == "missing":
            return  # successful process but no report
        write_grade(kwargs["cwd"], name, resolved=name == "recovered")

    monkeypatch.setattr(adapter, "run_agent", solve)
    monkeypatch.setattr(adapter.subprocess, "run", grade)
    assert (
        adapter.main(
            [
                "--bundle",
                str(bundle),
                "--config",
                str(config),
                "--output-dir",
                str(tmp_path),
                "--run-id",
                "test",
            ]
        )
        == 1
    )
    run_dir = tmp_path / "test"
    report = json.loads((run_dir / GRADE_LOG_DIR / "test/results.json").read_text())
    assert report["resolved_ids"] == ["recovered"]
    assert report["unresolved_ids"] == ["unresolved"]
    assert report["incomplete_ids"] == ["exhausted", "missing"]
    assert report["total_instances"] == 4
    assert attempts == {"recovered": 2, "unresolved": 1, "exhausted": 3, "missing": 3}
    predictions = [
        json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert [item["instance_id"] for item in predictions] == ["recovered", "unresolved"]
    assert (run_dir / "logs/attempts/recovered/1/grade/report.json").exists()
    failures = json.loads((run_dir / "logs/test.failures.json").read_text())
    assert set(failures) == {"exhausted", "missing"}


def test_server_report_directory_accepts_first_grade(monkeypatch, tmp_path, mock_summary):
    bundle = tmp_path / "bundle.tar.gz"
    bundle.touch()
    config = tmp_path / "config.toml"
    config.write_text('[coding_agent]\ncwd = "/testbed"\n')
    monkeypatch.setattr(adapter, "RUN_EVALUATION_LOG_DIR", adapter.Path("logs/run_evaluation"))
    monkeypatch.setattr(adapter, "discover_bundles", lambda _: {"linux/amd64": bundle})
    monkeypatch.setattr(adapter, "_load_dataset", lambda _: [{"instance_id": "issue", "image": "image"}])
    monkeypatch.setattr(adapter, "_run", lambda *a, **kw: "linux/amd64")
    monkeypatch.setattr(adapter, "image_platforms", lambda _: {"linux/amd64"})
    attempts = []

    def solve(*args, **kwargs):
        attempts.append("solve")
        return "patch"

    def grade(command, **kwargs):
        attempts.append("grade")
        folder = kwargs["cwd"] / "logs/run_evaluation/test/lhagent/issue"
        folder.mkdir(parents=True)
        (folder / "report.json").write_text('{"issue": {"resolved": true}}')

    monkeypatch.setattr(adapter, "run_agent", solve)
    monkeypatch.setattr(adapter.subprocess, "run", grade)
    assert adapter.main([
        "--bundle", str(bundle), "--config", str(config), "--output-dir", str(tmp_path),
        "--run-id", "test",
    ]) == 0
    assert attempts == ["solve", "grade"]
    assert json.loads((tmp_path / "test/predictions.jsonl").read_text())["instance_id"] == "issue"


def test_no_successful_grades_skips_summary(monkeypatch, tmp_path, mock_summary, capsys):
    bundle = tmp_path / "bundle.tar.gz"
    bundle.touch()
    config = tmp_path / "config.toml"
    config.write_text('[coding_agent]\ncwd = "/testbed"\n')
    monkeypatch.setattr(adapter, "discover_bundles", lambda _: {"linux/amd64": bundle})
    monkeypatch.setattr(adapter, "_load_dataset", lambda _: [{"instance_id": "issue", "image": "image"}])
    monkeypatch.setattr(adapter, "_run", lambda *a, **kw: "linux/amd64")
    monkeypatch.setattr(adapter, "image_platforms", lambda _: {"linux/amd64"})
    monkeypatch.setattr(adapter, "run_agent", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("agent failed")))
    monkeypatch.setattr(adapter, "summarize_run", lambda *a: pytest.fail("empty summary called"))
    assert adapter.main([
        "--bundle", str(bundle), "--config", str(config), "--output-dir", str(tmp_path),
        "--run-id", "test", "--retries", "0",
    ]) == 1
    assert "No successfully graded predictions" in capsys.readouterr().err
    assert json.loads((tmp_path / "test/logs/test.failures.json").read_text()) == {
        "issue": "agent failed"
    }


@pytest.mark.parametrize("retries", ["-1", "oops"])
def test_invalid_retry_count(retries, tmp_path):
    bundle = tmp_path / "bundle.tar.gz"
    bundle.touch()
    config = tmp_path / "config.toml"
    config.write_text('[coding_agent]\ncwd = "/testbed"\n')
    with pytest.raises(SystemExit):
        adapter.main(["--bundle", str(bundle), "--config", str(config), "--retries", retries])


@pytest.mark.parametrize("timed_out", [False, True])
def test_agent_failure_removes_container(monkeypatch, tmp_path, timed_out):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "exec"] and "--instruction" in command:
            if timed_out:
                raise subprocess.TimeoutExpired(command, 1)
            return subprocess.CompletedProcess(command, 1, "partial output", "agent error")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired if timed_out else RuntimeError):
        run_agent(
            {"instance_id": "failure", "image": "task:latest"},
            bundle=tmp_path / "bundle.tar.gz",
            config=tmp_path / "config.toml",
            instruction="test",
            timeout=1,
            log_dir=tmp_path,
            trace_dir=tmp_path / ".lhagent/failure",
            platform="linux/amd64",
            container_label="lhagent.swebench=test",
        )
    assert calls[-2][0:2] == ["docker", "cp"]
    assert calls[-2][2].endswith(":/tmp/.lhagent/.")
    assert calls[-2][3] == str(tmp_path / ".lhagent/failure")
    assert calls[-1][:3] == ["docker", "rm", "-f"]
    assert "lhagent.swebench=test" in calls[0]
