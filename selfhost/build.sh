#!/usr/bin/env bash
# Builds the native scc (build/scc) with the C++ safec.
#   SAFEC=path/to/safec (default: safec on PATH)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SAFEC="${SAFEC:-safec}"
mkdir -p "$HERE/build"
python3 "$HERE/tools/genvfs.py"
cd "$HERE/src"
SRCS=(util.sc lex.sc pp.sc parse.sc types.sc sema.sc expr.sc call.sc stmt.sc wasm.sc gen.sc genx.sc vfs.sc memflow.sc core.sc driver_host.sc)
"$SAFEC" -I . "${SRCS[@]}" --emit-bin -O2 -o "$HERE/build/scc" 2> "$HERE/build/build.log" || { cat "$HERE/build/build.log"; exit 1; }
echo "built $HERE/build/scc"
