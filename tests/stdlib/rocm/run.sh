#!/usr/bin/env bash
# Builds and runs the ROCm backend checks (needs an AMD GPU + ROCm with
# MIOpen and hipFFT).
#   usage: run.sh            Env: SAFEC, CLANG, ROCM_PATH (default /opt/rocm)
# Links against tests/stdlib/build/deps/libsafec_std.a — run
# 'safeguard test' in tests/stdlib first to build it.
#
# train_backend_check.sc is built twice — for the CPU (no features) and with
# the rocm feature (SAFEC_FEATURE_ROCM) — and the two runs' RESULT lines
# must agree: one program, either backend, through train.h.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SAFEC="${SAFEC:-safec}"; CLANG="${CLANG:-clang}"
ROCM="${ROCM_PATH:-/opt/rocm}"
OUT="${OUT:-$HERE/build}"
mkdir -p "$OUT"
STD="$ROOT/tests/stdlib/build/deps/libsafec_std.a"
[[ -f "$STD" ]] || { echo "missing $STD — run 'safeguard test' in tests/stdlib first"; exit 1; }
for t in rocm_check rocm_dnn_fft_check nn_rocm_check; do
    "$SAFEC" -I "$ROOT" "$HERE/$t.sc" -O2 -o "$OUT/$t.ll"
    "$CLANG" -O2 -w "$OUT/$t.ll" "$STD" -L"$ROCM/lib" -Wl,-rpath,"$ROCM/lib" \
        -lMIOpen -lhipfft -lamdhip64 -lhiprtc -lrocblas -lm -lpthread -o "$OUT/$t"
    "$OUT/$t"
done

LIBS=(-L"$ROCM/lib" -Wl,-rpath,"$ROCM/lib" -lMIOpen -lhipfft -lamdhip64 -lhiprtc -lrocblas -lm -lpthread)
"$SAFEC" -I "$ROOT" "$HERE/train_backend_check.sc" -O2 -o "$OUT/train_backend_cpu.ll"
"$CLANG" -O2 -w "$OUT/train_backend_cpu.ll" "$STD" "${LIBS[@]}" -o "$OUT/train_backend_cpu"
"$SAFEC" -I "$ROOT" -DSAFEC_FEATURE_ROCM=1 "$HERE/train_backend_check.sc" -O2 -o "$OUT/train_backend_rocm.ll"
"$CLANG" -O2 -w "$OUT/train_backend_rocm.ll" "$STD" "${LIBS[@]}" -o "$OUT/train_backend_rocm"
"$OUT/train_backend_cpu" | tee "$OUT/train_backend_cpu.txt"
"$OUT/train_backend_rocm" | tee "$OUT/train_backend_rocm.txt"
# value b (rocm) against a (cpu): |a - b| <= tol * (1 + |a|), every name present in both
awk '$1 == "RESULT" { if (FILENAME == ARGV[1]) { a[$2] = $3; tol[$2] = $4 } else b[$2] = $3 }
     END {
         bad = 0; n = 0
         for (k in a) {
             n++
             if (!(k in b)) { printf "  FAIL %s: missing from the rocm run\n", k; bad++; continue }
             d = a[k] - b[k]; if (d < 0) d = -d
             m = a[k] < 0 ? -a[k] : a[k]
             if (d > tol[k] * (1 + m)) { printf "  FAIL %s: cpu %s rocm %s (tolerance %s)\n", k, a[k], b[k], tol[k]; bad++ }
         }
         printf "train_backend_check: cpu vs rocm, %d values, %d differ\n", n, bad
         exit bad > 0 || n == 0
     }' "$OUT/train_backend_cpu.txt" "$OUT/train_backend_rocm.txt"
