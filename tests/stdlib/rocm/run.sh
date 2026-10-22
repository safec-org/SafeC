#!/usr/bin/env bash
# Builds and runs the ROCm backend checks (needs an AMD GPU + ROCm with
# MIOpen and hipFFT).
#   usage: run.sh            Env: SAFEC, CLANG, ROCM_PATH (default /opt/rocm)
# Links an archive of all of std (../std_archive.sh, through safeguard:
# SAFEGUARD, default on PATH), or STD_ARCHIVE if set.
#
# ../device/device_check.sc is built with the rocm feature
# (SAFEC_FEATURE_ROCM, which links device_rocm.sc in) and trains every
# std::ml workload on the CPU and on the GPU through the same code: the
# losses must agree.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SAFEC="${SAFEC:-safec}"; CLANG="${CLANG:-clang}"
ROCM="${ROCM_PATH:-/opt/rocm}"
OUT="${OUT:-$HERE/build}"
mkdir -p "$OUT"
STD="${STD_ARCHIVE:-$(SAFEC="$SAFEC" "$HERE/../std_archive.sh" "$OUT/_std")}"
LIBS=(-L"$ROCM/lib" -Wl,-rpath,"$ROCM/lib" -lMIOpen -lhipfft -lamdhip64 -lhiprtc -lrocblas -lm -lpthread)
for t in rocm_check rocm_dnn_fft_check; do
    "$SAFEC" -I "$ROOT" "$HERE/$t.sc" -O2 -o "$OUT/$t.ll"
    "$CLANG" -O2 -w "$OUT/$t.ll" "$STD" "${LIBS[@]}" -o "$OUT/$t"
    "$OUT/$t"
done
"$SAFEC" -I "$ROOT" -DSAFEC_FEATURE_ROCM=1 -DCHECK_DEVICE=DEVICE_ROCM "$HERE/../device/device_check.sc" -O2 \
    -o "$OUT/device_check.ll"
"$CLANG" -O2 -w "$OUT/device_check.ll" "$STD" "${LIBS[@]}" -o "$OUT/device_check"
"$OUT/device_check"
