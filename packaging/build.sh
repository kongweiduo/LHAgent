#!/usr/bin/env bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PLATFORM=${1:-all}
OUT=${2:-"$ROOT/packaging/dist"}
case "$PLATFORM" in
  all) PLATFORMS=(linux/amd64 linux/arm64) ;;
  linux/amd64|linux/arm64) PLATFORMS=("$PLATFORM") ;;
  *) echo 'usage: build.sh [all|linux/amd64|linux/arm64] [output-directory]' >&2; exit 2 ;;
esac
command -v docker >/dev/null || { echo 'Docker with buildx is required' >&2; exit 2; }
docker buildx version >/dev/null || { echo 'Docker buildx is required' >&2; exit 2; }
mkdir -p "$OUT"
for target in "${PLATFORMS[@]}"; do
  echo "Building $target bundle"
  docker buildx build --platform "$target" \
    --file "$ROOT/packaging/Dockerfile" --output "type=local,dest=$OUT" "$ROOT"
done
