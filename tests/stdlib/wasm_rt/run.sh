#!/usr/bin/env bash
# Builds this directory's wasm32 test module with safeguard and runs it
# under Node with NO imports at all: std's libc/compiler-rt needs must all
# be met by std/wasm/wasm_rt.
#   usage: run.sh            (env: SAFEGUARD, NODE)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SAFEGUARD="${SAFEGUARD:-safeguard}"; NODE="${NODE:-node}"
export SAFEC_HOME="${SAFEC_HOME:-$ROOT}"
cd "$HERE"
"$SAFEGUARD" build --target wasm32 >/dev/null
"$NODE" "$HERE/run.mjs" "$HERE/build/wasm_rt_tests"
