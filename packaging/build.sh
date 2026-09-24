#!/usr/bin/env bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PLATFORM=${1:-linux/amd64}
OUT=${2:-"$ROOT/packaging/dist"}
case "$PLATFORM" in linux/amd64|linux/arm64) ;; *) echo 'usage: build.sh [linux/amd64|linux/arm64] [output-directory]' >&2; exit 2 ;; esac
command -v docker >/dev/null || { echo 'Docker with buildx is required' >&2; exit 2; }
mkdir -p "$OUT"
docker buildx build --platform "$PLATFORM" \
  --file "$ROOT/packaging/Dockerfile" --output "type=local,dest=$OUT" "$ROOT"
