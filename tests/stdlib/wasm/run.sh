#!/usr/bin/env bash
# Builds this directory's wasm32 test module with safeguard and runs it
# under Node against a fake DOM through std/wasm/safec_dom.js.
#   usage: run.sh            (env: SAFEGUARD, NODE)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SAFEGUARD="${SAFEGUARD:-safeguard}"; NODE="${NODE:-node}"
export SAFEC_HOME="${SAFEC_HOME:-$ROOT}"
cd "$HERE"
"$SAFEGUARD" build --target wasm32 >/dev/null
"$NODE" "$HERE/run.mjs" "$HERE/build/wasm_dom_tests" "$ROOT/std/wasm/safec_dom.js"
