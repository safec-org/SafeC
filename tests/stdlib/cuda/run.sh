#!/usr/bin/env bash
# Builds and runs the CUDA backend checks (needs an NVIDIA GPU, CUDA 12 or newer, cuDNN 9).
#   usage: run.sh            Env: SAFEC, CLANG, CUDA_PATH (default /usr/local/cuda)
# Links an archive of all of std (../std_archive.sh, through safeguard:
# SAFEGUARD, default on PATH), or STD_ARCHIVE if set.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SAFEC="${SAFEC:-safec}"; CLANG="${CLANG:-clang}"
CUDA="${CUDA_PATH:-/usr/local/cuda}"
OUT="${OUT:-$HERE/build}"
mkdir -p "$OUT"
STD="${STD_ARCHIVE:-$(SAFEC="$SAFEC" "$HERE/../std_archive.sh" "$OUT/_std")}"
for t in cuda_check cuda_attention_check; do
    "$SAFEC" -I "$ROOT" "$HERE/$t.sc" -O2 -o "$OUT/$t.ll"
    "$CLANG" -O2 -w "$OUT/$t.ll" "$STD" -L"$CUDA/lib64" -Wl,-rpath,"$CUDA/lib64" \
        -lcuda -lcublas -lcublasLt -lnvrtc -lcudnn -lcufft -lm -lpthread -o "$OUT/$t"
    "$OUT/$t"
done

# ../device/device_check.sc with the cuda feature (SAFEC_FEATURE_CUDA links
# device_cuda.sc in): every std::ml workload on the CPU and on the GPU.
"$SAFEC" -I "$ROOT" -DSAFEC_FEATURE_CUDA=1 -DCHECK_DEVICE=DEVICE_CUDA "$HERE/../device/device_check.sc" -O2 \
    -o "$OUT/device_check.ll"
"$CLANG" -O2 -w "$OUT/device_check.ll" "$STD" -L"$CUDA/lib64" -Wl,-rpath,"$CUDA/lib64" \
    -lcuda -lcublas -lcublasLt -lnvrtc -lcudnn -lcufft -lm -lpthread -o "$OUT/device_check"
"$OUT/device_check"
