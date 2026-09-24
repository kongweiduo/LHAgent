#!/bin/sh
set -eu
[ ! -x /bin/ps ]
mkdir -p '/tmp/relocated path' /tmp/workspace /tmp/commands
cd /tmp/workspace
tar -xzf /bundle.tar.gz -C '/tmp/relocated path'
BUNDLE='/tmp/relocated path/lhagent'
chmod -R a-w "$BUNDLE"
"$BUNDLE/lhagent" --help >/dev/null
"$BUNDLE/lhagent" --list-sessions
"$BUNDLE/lhagent" --bundle-check
# Poison task Python settings must not affect the agent or its search worker.
PYTHONHOME=/nonexistent/task-python PYTHONPATH=/nonexistent/task-modules \
  VIRTUAL_ENV=/task/venv LHAGENT_CHECK_TASK_PYTHON= "$BUNDLE/lhagent" --bundle-check
"$BUNDLE/install.sh" /tmp/commands
/tmp/commands/lhagent --help >/dev/null
if "$BUNDLE/install.sh" /tmp/commands; then
    echo 'installer unexpectedly replaced an existing command' >&2
    exit 1
fi
case "$1" in
  no-python) ! command -v python; ! command -v python3 ;;
  task-python)
    python --version | /bin/grep 'Python 3.11'
    LHAGENT_CHECK_TASK_PYTHON=3.11 "$BUNDLE/lhagent" --bundle-check
    ;;
esac
printf 'PASS: %s\n' "$1"
