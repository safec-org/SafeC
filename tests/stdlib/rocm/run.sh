#!/usr/bin/env bash
# Builds and runs the ROCm backend checks (needs an AMD GPU + ROCm).
#   usage: run.sh            Env: SAFEC, CLANG, ROCM_PATH (default /opt/rocm)
# Links against tests/stdlib/build/deps/libsafec_std.a — run
# 'safeguard test' in tests/stdlib first to build it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SAFEC="${SAFEC:-safec}"; CLANG="${CLANG:-clang}"
ROCM="${ROCM_PATH:-/opt/rocm}"
OUT="${OUT:-$HERE/build}"
mkdir -p "$OUT"
STD="$ROOT/tests/stdlib/build/deps/libsafec_std.a"
[[ -f "$STD" ]] || { echo "missing $STD — run 'safeguard test' in tests/stdlib first"; exit 1; }
for t in rocm_check; do
    "$SAFEC" -I "$ROOT" "$HERE/$t.sc" -O2 -o "$OUT/$t.ll"
    "$CLANG" -O2 -w "$OUT/$t.ll" "$STD" -L"$ROCM/lib" -Wl,-rpath,"$ROCM/lib" \
        -lamdhip64 -lhiprtc -lrocblas -lm -lpthread -o "$OUT/$t"
    "$OUT/$t"
done
