#!/usr/bin/env bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ARCHIVE=${1:?usage: test.sh BUNDLE.tar.gz [linux/amd64|linux/arm64]}
PLATFORM=${2:-linux/amd64}
ARCHIVE=$(CDPATH= cd -- "$(dirname -- "$ARCHIVE")" && pwd)/$(basename -- "$ARCHIVE")
for target in no-python task-python; do
    tag="lhagent-bundle-test:${target}-${PLATFORM##*/}"
    docker build --platform "$PLATFORM" -f "$ROOT/test.Dockerfile" --target "$target" -t "$tag" "$ROOT"
    docker run --rm --platform "$PLATFORM" --network none --read-only \
      --user 65534:65534 --tmpfs /tmp:exec,mode=1777 \
      --mount "type=bind,src=$ARCHIVE,dst=/bundle.tar.gz,readonly" \
      --mount "type=bind,src=$ROOT/test-container.sh,dst=/test.sh,readonly" \
      "$tag" sh /test.sh "$target"
done
