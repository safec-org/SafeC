#!/usr/bin/env bash
# Builds and runs the CUDA backend checks (needs an NVIDIA GPU + CUDA 12 or newer).
#   usage: run.sh            Env: SAFEC, CLANG, CUDA_PATH (default /usr/local/cuda)
# Links against tests/stdlib/build/deps/libsafec_std.a — run
# 'safeguard test' in tests/stdlib first to build it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SAFEC="${SAFEC:-safec}"; CLANG="${CLANG:-clang}"
CUDA="${CUDA_PATH:-/usr/local/cuda}"
OUT="${OUT:-$HERE/build}"
mkdir -p "$OUT"
STD="$ROOT/tests/stdlib/build/deps/libsafec_std.a"
[[ -f "$STD" ]] || { echo "missing $STD — run 'safeguard test' in tests/stdlib first"; exit 1; }
for t in cuda_check; do
    "$SAFEC" -I "$ROOT" "$HERE/$t.sc" -O2 -o "$OUT/$t.ll"
    "$CLANG" -O2 -w "$OUT/$t.ll" "$STD" -L"$CUDA/lib64" -Wl,-rpath,"$CUDA/lib64" \
        -lcuda -lcublas -lcublasLt -lnvrtc -lm -lpthread -o "$OUT/$t"
    "$OUT/$t"
done
