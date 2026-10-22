#!/usr/bin/env bash
# Builds and runs the WebGPU backend checks against a wgpu-native build (and
# a GPU it can use; Homebrew LLVM's clang on macOS, which safec's IR
# targets): webgpu_api.h's struct layouts against the real webgpu.h,
# gpu_webgpu.h's host-array API, and every std::ml training workload on the
# CPU and on the WebGPU device backend.
#   usage: run.sh
#   Env: WGPU_NATIVE_DIR (required: a wgpu-native release, with
#          include/webgpu/webgpu.h and lib/libwgpu_native.a or .dylib / .so),
#        SAFEC (default compiler/build/safec), SAFEGUARD (default: on PATH),
#        OUT (build dir), STD_ARCHIVE (a prebuilt release libsafec_std.a).
# Links the release std archive 'safeguard build --release' makes for this
# checkout, as tests/stdlib/mps/run.sh does.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SAFEC="${SAFEC:-$ROOT/compiler/build/safec}"
SAFEGUARD="${SAFEGUARD:-safeguard}"
OUT="${OUT:-$HERE/build}"
WGPU="${WGPU_NATIVE_DIR:?set WGPU_NATIVE_DIR to a wgpu-native release directory (include/, lib/)}"
mkdir -p "$OUT"
CLANG=clang
for c in /opt/homebrew/opt/llvm/bin/clang /usr/local/opt/llvm/bin/clang; do
    if [[ -x "$c" ]]; then CLANG="$c"; break; fi
done

# wgpu-native, statically when the release has the archive
if [[ -f "$WGPU/lib/libwgpu_native.a" ]]; then
    WLIB="$WGPU/lib/libwgpu_native.a"
    if [[ "$(uname)" == Darwin ]]; then
        WLIB="$WLIB -framework Metal -framework QuartzCore -framework Foundation -lobjc"
    else
        WLIB="$WLIB -ldl -lm -lpthread"
    fi
else
    WLIB="-L$WGPU/lib -lwgpu_native -Wl,-rpath,$WGPU/lib"
fi

STD="${STD_ARCHIVE:-}"
if [[ -z "$STD" ]]; then
    P="$OUT/_std"
    mkdir -p "$P/src"
    printf '[package]\nname = "webgpu_checks_std"\nversion = "0.1.0"\n\n[build]\nedition = "2026"\n' > "$P/Package.toml"
    echo 'int main() { return 0; }' > "$P/src/main.sc"
    (cd "$P" && SAFEC="$SAFEC" SAFEC_HOME="$ROOT" "$SAFEGUARD" build --release >/dev/null)
    STD="$P/build/deps/libsafec_std.a"
fi

FW=""
[[ "$(uname)" == Darwin ]] && FW="-framework Foundation -framework Accelerate"
WRAP="$OUT/clang_link.sh"
printf '#!/bin/sh\nexec "%s" -w "$@" "%s" %s %s -lm -lpthread\n' "$CLANG" "$STD" "$WLIB" "$FW" > "$WRAP"
chmod +x "$WRAP"

# webgpu_api.h against the real header: the same sizes and offsets
"$CLANG" -w "$HERE/layout_check.c" -I "$WGPU/include" -o "$OUT/layout_check_c"
SAFEC_CLANG="$CLANG" "$SAFEC" "$HERE/layout_check.sc" -I "$ROOT" --emit-bin -o "$OUT/layout_check_sc" 2>/dev/null
if diff <("$OUT/layout_check_c") <("$OUT/layout_check_sc"); then
    echo "webgpu_api.h: layouts match webgpu.h"
else
    echo "webgpu_api.h: layouts differ from webgpu.h"
    exit 1
fi

# gpu_webgpu.h's array API (with the webgpu feature: and a tensor's buffer)
SAFEC_CLANG="$WRAP" "$SAFEC" "$HERE/webgpu_check.sc" -I "$ROOT" -DSAFEC_FEATURE_WEBGPU=1 --emit-bin -O2 -o "$OUT/webgpu_check"
"$OUT/webgpu_check"

# ../device/device_check.sc with the webgpu feature (SAFEC_FEATURE_WEBGPU
# links device_webgpu.sc in): every std::ml workload on the CPU and on the GPU.
SAFEC_CLANG="$WRAP" "$SAFEC" "$HERE/../device/device_check.sc" -I "$ROOT" -DSAFEC_FEATURE_WEBGPU=1 \
    -DCHECK_DEVICE=DEVICE_WEBGPU --emit-bin -O2 -o "$OUT/device_check"
"$OUT/device_check"
