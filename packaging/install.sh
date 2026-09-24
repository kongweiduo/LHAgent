#!/bin/sh
# Run after extracting the bundle. Installation is optional: ./lhagent works too.
set -eu
BUNDLE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BIN_DIR=${1:-/usr/local/bin}
case "$BIN_DIR" in /*) ;; *) echo 'command directory must be absolute' >&2; exit 2 ;; esac
case "$(uname -m)" in x86_64) ARCH=amd64 ;; aarch64|arm64) ARCH=arm64 ;; *) echo 'unsupported CPU' >&2; exit 2 ;; esac
[ "$(uname -s)/$ARCH" = "Linux/$(cut -d/ -f2 "$BUNDLE/TARGET")" ] || {
    echo 'bundle OS/architecture does not match this container' >&2; exit 2;
}
[ ! -e "$BIN_DIR/lhagent" ] && [ ! -L "$BIN_DIR/lhagent" ] || {
    echo "$BIN_DIR/lhagent already exists; refusing to overwrite" >&2; exit 1;
}
# Check before creating the command link. This never contacts the model provider.
"$BUNDLE/lhagent" --bundle-check
mkdir -p "$BIN_DIR"
ln -s "$BUNDLE/lhagent" "$BIN_DIR/lhagent"
printf 'Installed %s/lhagent -> %s/lhagent\n' "$BIN_DIR" "$BUNDLE"
