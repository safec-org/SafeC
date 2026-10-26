#!/usr/bin/env bash
# Builds the browser compiler: web/dist/scc.wasm (+ copies scc.js, worker.js)
# with the C++ safec, clang and wasm-ld.
#   SAFEC, SAFEC_HOME (for <std/stddef.h>), CLANG (default: $SAFEC_CLANG or clang), WASM_LD
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SAFEC="${SAFEC:-safec}"
CLANG="${CLANG:-${SAFEC_CLANG:-clang}}"
WASM_LD="${WASM_LD:-wasm-ld}"
: "${SAFEC_HOME:?set SAFEC_HOME to the SafeC checkout (for std/stddef.h)}"
OUT="$HERE/build/web"
DIST="$HERE/web/dist"
mkdir -p "$OUT" "$DIST"
python3 "$HERE/tools/genvfs.py"
SRCS=(util lex pp parse types sema expr call stmt wasm gen genx vfs memflow core driver_wasm)
CFLAGS=$("$SAFEC" --print-target-clang-flags --target=wasm32)
objs=()
for s in "${SRCS[@]}"; do
    "$SAFEC" -I "$HERE/src" -I "$SAFEC_HOME" -D SCC_SMALL "$HERE/src/$s.sc" --target=wasm32 \
        --emit-llvm -O2 -o "$OUT/$s.ll"
    "$CLANG" $CFLAGS -O2 -c "$OUT/$s.ll" -o "$OUT/$s.o" -Wno-override-module
    objs+=("$OUT/$s.o")
done
"$WASM_LD" --no-entry --allow-undefined -z stack-size=8388608 --gc-sections \
    -o "$DIST/scc.wasm" "${objs[@]}"
cp "$HERE/web/scc.js" "$HERE/web/worker.js" "$DIST/" 2>/dev/null || true
echo "built $DIST/scc.wasm ($(wc -c < "$DIST/scc.wasm") bytes)"
