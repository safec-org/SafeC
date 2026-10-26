#!/usr/bin/env bash
# Builds the browser compiler, web/scc.wasm, next to web/scc.js and the
# rest of the playground files, so web/ is the directory to deploy (the docs
# site copies it to assets/scc/). Needs the C++ safec, clang and wasm-ld.
#   SAFEC      default: safec on PATH, else compiler/build/safec
#   SAFEC_HOME default: the SafeC checkout this script is in
#   CLANG      default: $SAFEC_CLANG, else clang (or clang-NN) on PATH
#   WASM_LD    default: wasm-ld (or wasm-ld-NN) on PATH
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SAFEC_HOME="${SAFEC_HOME:-$(dirname "$HERE")}"
pick() { for c in "$@"; do if command -v "$c" >/dev/null 2>&1; then command -v "$c"; return; fi; done; echo "$1"; }
if [[ -z "${SAFEC:-}" ]]; then
    if command -v safec >/dev/null 2>&1; then SAFEC=safec
    elif [[ -x "$SAFEC_HOME/compiler/build/safec" ]]; then SAFEC="$SAFEC_HOME/compiler/build/safec"
    else SAFEC=safec; fi
fi
CLANG="${CLANG:-${SAFEC_CLANG:-$(pick clang clang-22 clang-21)}}"
WASM_LD="${WASM_LD:-$(pick wasm-ld wasm-ld-22 wasm-ld-21)}"
OUT="$HERE/build/web"
mkdir -p "$OUT"
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
    -o "$HERE/web/scc.wasm" "${objs[@]}"
echo "built $HERE/web/scc.wasm ($(wc -c < "$HERE/web/scc.wasm") bytes)" >&2
