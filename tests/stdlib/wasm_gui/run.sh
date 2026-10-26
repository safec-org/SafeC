#!/usr/bin/env bash
# Builds this directory's wasm32 module with safeguard and runs it under
# Node against a fake canvas through std/gui/gui_web.js — standalone and
# instantiated through std/wasm/safec_dom.js's loadSafeC.
#   usage: run.sh            (env: SAFEGUARD, NODE)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SAFEGUARD="${SAFEGUARD:-safeguard}"; NODE="${NODE:-node}"
export SAFEC_HOME="${SAFEC_HOME:-$ROOT}"
cd "$HERE"
"$SAFEGUARD" build --target wasm32 >/dev/null
"$NODE" "$HERE/run.mjs" "$HERE/build/wasm_gui_tests" "$ROOT/std/gui/gui_web.js"
"$NODE" "$HERE/run.mjs" "$HERE/build/wasm_gui_tests" "$ROOT/std/gui/gui_web.js" "$ROOT/std/wasm/safec_dom.js" | tail -1
