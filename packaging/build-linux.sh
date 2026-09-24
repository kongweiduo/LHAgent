#!/usr/bin/env bash
# Internal Docker build stage. No task image is used or modified here.
set -euo pipefail
ARCH=${1:?target architecture required}
case "$ARCH" in amd64|arm64) ;; *) exit 2 ;; esac
STAGE=/bundle/lhagent
mkdir -p "$STAGE" /output
uv python install --install-dir /python --no-bin 3.12.14
PYTHON_DIR=(/python/cpython-3.12.14-*)
cp -a "${PYTHON_DIR[0]}" "$STAGE/runtime"
PYTHON="$STAGE/runtime/bin/python3.12"
uv export --python "$PYTHON" --locked --no-dev --no-emit-project --no-editable \
  --format requirements.txt --output-file /requirements.txt >/dev/null
uv pip install --python "$PYTHON" --target "$STAGE/runtime/lib/python3.12/site-packages" \
  --require-hashes --only-binary :all: --requirement /requirements.txt
uv build --python "$PYTHON" --wheel --out-dir /wheels
uv pip install --python "$PYTHON" --target "$STAGE/runtime/lib/python3.12/site-packages" \
  --no-deps /wheels/*.whl
cp packaging/lhagent "$STAGE/lhagent"
cp packaging/check.py "$STAGE/check.py"
cp packaging/install.sh "$STAGE/install.sh"
cp packaging/README.md "$STAGE/README.md"
cp /requirements.txt "$STAGE/requirements.txt"
chmod +x "$STAGE/lhagent" "$STAGE/install.sh"
VERSION=$("$PYTHON" -I -c 'from importlib.metadata import version; print(version("lhagent"))')
printf 'linux/%s\n' "$ARCH" > "$STAGE/TARGET"
"$PYTHON" -I -c 'import json,platform,sys; from importlib.metadata import distributions; print(json.dumps({"python":sys.version,"machine":platform.machine(),"libc":platform.libc_ver(),"packages":{d.metadata["Name"]:d.version for d in distributions()}},indent=2))' > "$STAGE/manifest.json"
"$STAGE/lhagent" --help >/dev/null
"$STAGE/lhagent" --bundle-check
# Exclude bytecode/build paths and create a reproducible file ordering.
find "$STAGE" -type d -name __pycache__ -prune -exec rm -r '{}' +
NAME="lhagent-$VERSION-linux-$ARCH.tar.gz"
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C /bundle -cf - lhagent | gzip -n > "/output/$NAME"
(cd /output && sha256sum "$NAME" > "$NAME.sha256")
