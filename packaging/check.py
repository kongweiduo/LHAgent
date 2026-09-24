"""Offline bundle smoke test; exercises real tools without credentials or a model."""

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from lhagent.harness.tools.builtin.bash import create_bash_tool
from lhagent.harness.tools.builtin.find import create_find_tool
from lhagent.harness.tools.builtin.grep import create_grep_tool
from lhagent.harness.tools.execution import execute_tool_call


async def check():
    assert sys.version_info[:2] == (3, 12), sys.version
    for executable in ("/bin/bash",):
        assert os.access(executable, os.X_OK), f"task image requires {executable}"
    with tempfile.TemporaryDirectory(prefix="lhagent-check-") as directory:
        root = Path(directory)
        (root / "example.txt").write_text("bundle needle\n", encoding="utf-8")
        context = {
            "cwd": directory,
            "timeout_seconds": 10,
            "cancel_event": asyncio.Event(),
            "max_output_lines": 200,
            "max_output_bytes": 50000,
        }

        async def invoke(tool, arguments):
            result = await execute_tool_call(
                {"call_id": "check", "name": tool["name"], "arguments": arguments},
                [tool],
                context,
            )
            assert result["status"] == "success", result
            return result["output"]["content"][0]["text"]

        assert json.loads(await invoke(create_find_tool(), {"pattern": "*.txt"})) == "example.txt"
        match = json.loads(await invoke(create_grep_tool(), {"pattern": "needle"}))
        assert match == {"path": "example.txt", "line": 1, "text": "bundle needle"}
        # Exercise cleanup with a background descendant; Linux must do this
        # through procfs even when the task image has no ps command.
        await invoke(create_bash_tool(), {"command": "sleep 60 >/dev/null 2>&1 & echo $! > child"})
        pid = (root / "child").read_text().strip()
        stat = Path("/proc") / pid / "stat"
        try:
            assert stat.read_text().rsplit(")", 1)[1].split()[0] in ("Z", "X")
        except FileNotFoundError:
            pass
        # Bash must inherit task PATH/Python variables unchanged. BASH_ENV is
        # intentionally filtered by the existing bash tool.
        names = ("PATH", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
        command = 'printf \'%s\\0\' "$PATH" "${PYTHONPATH-}" "${PYTHONHOME-}" "${VIRTUAL_ENV-}"'
        actual = await invoke(create_bash_tool(), {"command": command})
        assert actual == "".join(os.environ.get(name, "") + "\0" for name in names)
        command_path = (
            await invoke(create_bash_tool(), {"command": "command -v python || true"})
        ).strip()
        expected = shutil.which("python") or ""
        assert command_path == expected, (command_path, expected)
        if version := os.environ.get("LHAGENT_CHECK_TASK_PYTHON"):
            observed = await invoke(create_bash_tool(), {"command": "python --version"})
            assert observed.startswith(f"Python {version}."), observed
        print(
            json.dumps(
                {
                    "status": "ok",
                    "agent_python": sys.executable,
                    "task_python": command_path or None,
                    "checks": ["imports", "find", "grep", "bash", "cleanup", "task_environment"],
                }
            )
        )


if __name__ == "__main__":
    asyncio.run(check())
