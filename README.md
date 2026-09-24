# LHAgent

A terminal coding agent with streaming model responses, tools, and persistent sessions.

## Run

Configure the model and token budgets in `lhagent.toml` (see
`lhagent.example.toml`). Set `LHAGENT_BASE_URL` and `LHAGENT_API_KEY` in the
environment or in `.env` in the directory where you start the command.
Environment variables take precedence over `.env`.

```sh
uv run lhagent --config lhagent.toml
uv run lhagent --config lhagent.toml --instruction "Hello"
```

Omitting `tools` enables all built-in tools; `tools = []` disables them.
The configured `cwd` is a working directory, not a filesystem sandbox.

For an offline Linux container bundle with a private Python runtime, see
[packaging/README.md](packaging/README.md). Build with
`./packaging/build.sh linux/amd64` (or `linux/arm64`); extracting the resulting
archive is sufficient to run LHAgent without changing the task environment's PATH.
The [SWE-bench Lite adapter](src/lhagent/evals/benchmarks/swebench/README.md)
runs the bundle inside benchmark images and grades its patches with the official harness.

## Sessions And History

Every conversation is saved automatically as
`.lhagent/sessions/<session-id>.jsonl` under the startup working directory
(inside `LHAgent/` when launched from the project). The startup output shows its exact
path; single-instruction mode prints the path to stderr so stdout remains the
model answer. A new invocation starts a new session unless `--session` is given.

In the terminal, `/session` shows the current path, `/history` redisplays its
saved conversation, `/resume` selects an older conversation, and `/new` starts
a new one. `/clear` only clears the display. `/help` lists all commands.

```sh
uv run lhagent --list-sessions
uv run lhagent --history --session .lhagent/sessions/<session-id>.jsonl
uv run lhagent --config lhagent.toml --session .lhagent/sessions/<session-id>.jsonl
```

Listing and viewing history require neither model configuration nor API credentials.
Older sessions under `~/.lhagent/sessions/` can still be opened with `--session`
and their full path; changing the default does not move or delete existing files.
The history viewer uses bounded previews for tool output; the JSONL file contains
the complete stored results.

The JSONL log is the durable session trace: run starts and finishes, user messages,
terminal model responses (including errors, reasoning, tool calls and usage),
tool results, compaction summaries, and history exclusions. Records are appended
and flushed during the run, including failed runs. Individual streaming deltas
and raw HTTP requests are not stored. A process killed during streaming may
leave an interrupted run without its unfinished response. Resuming or viewing
a session validates the log and can recover an incomplete trailing record.

## Tests

```sh
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Tests use temporary working directories and fake credentials, independently of
personal `.env` and `lhagent.toml` files. The model transport tests use mock HTTP.
Development checks use pytest and Ruff; mypy is not required.

## 0.51.0

Client tool definitions use the internal `name`, `description`, and `parameters`
fields. Transport adds the OpenAI `type`/`function` wrapper; passing a prewrapped
definition is no longer supported.
