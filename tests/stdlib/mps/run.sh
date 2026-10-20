#!/usr/bin/env bash
# Builds and runs the MPS (Metal) backend checks (needs an Apple GPU and
# Homebrew LLVM's clang, which safec's IR targets).
#   usage: run.sh            Env: SAFEC (default compiler/build/safec), OUT
# Compiles the std sources the checks reach directly rather than linking
# libsafec_std.a, which also needs the CUDA/ROCm toolchains.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SAFEC="${SAFEC:-$ROOT/compiler/build/safec}"
OUT="${OUT:-$HERE/build}"
mkdir -p "$OUT"
CLANG=clang
for c in /opt/homebrew/opt/llvm/bin/clang /usr/local/opt/llvm/bin/clang; do
    if [[ -x "$c" ]]; then CLANG="$c"; break; fi
done
WRAP="$OUT/clang_frameworks.sh"
printf '#!/bin/sh\nexec "%s" "$@" -framework Metal -framework Foundation -framework Accelerate -lobjc\n' "$CLANG" > "$WRAP"
chmod +x "$WRAP"
S="$ROOT/std"
DEPS=("$S/ml/gpu_mps.sc" "$S/ml/float16.sc" "$S/ml/attention.sc" "$S/ml/tensor.sc" "$S/ml/activations.sc"
      "$S/math.sc" "$S/mem.sc" "$S/atomic.sc" "$S/panic.sc" "$S/sync/parallel.sc" "$S/thread.sc"
      "$S/simd/gemm.sc" "$S/simd/vmath.sc" "$S/simd/cpu.sc" "$S/str.sc" "$S/sys.sc" "$S/time.sc")
for t in mps_attention_check; do
    SAFEC_CLANG="$WRAP" "$SAFEC" "$HERE/$t.sc" "${DEPS[@]}" -I "$ROOT" --emit-bin -o "$OUT/$t"
    "$OUT/$t"
done
