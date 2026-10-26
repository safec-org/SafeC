#!/usr/bin/env bash
# Everything CI runs for scc (Linux): build the native and the browser
# compiler with the C++ safec, run the differential tests with both, and
# check the bootstrap fixpoint (with the self-compiled scc tested too).
#   SAFEC (default: safec on PATH), SAFEC_HOME (the SafeC checkout),
#   SAFEC_CLANG / CLANG, WASM_LD, NODE
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${SAFEC_HOME:=$(dirname "$HERE")}"
export SAFEC_HOME
echo "── build: native scc"
"$HERE/build.sh"
echo "── build: browser scc.wasm"
"$HERE/build-web.sh"
echo "── differential tests: native scc"
python3 "$HERE/tests/difftest.py" --json "$HERE/build/difftest-native.json"
echo "── differential tests: scc.wasm under Node (the playground path)"
python3 "$HERE/tests/difftest.py" --scc-wasm "$HERE/web/scc.wasm" --json "$HERE/build/difftest-wasm.json"
echo "── playground API"
"${NODE:-node}" "$HERE/web/test-api.mjs"
echo "── bootstrap"
"$HERE/bootstrap.sh" --test
