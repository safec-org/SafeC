#!/usr/bin/env bash
# Builds and runs the Vulkan backend checks (needs a Vulkan 1.2 GPU with the
# loader, e.g. Homebrew's vulkan-loader + molten-vk on macOS, and Homebrew
# LLVM's clang, which safec's IR targets): gpu_spirv.h's host-array API, and
# every std::ml training workload on the CPU and on the Vulkan device backend
# in fp32, fp16 and bf16.
#   usage: run.sh
#   Env: SAFEC (default compiler/build/safec), SAFEGUARD (default: on PATH),
#        OUT (build dir), STD_ARCHIVE (a prebuilt release libsafec_std.a),
#        VULKAN_LIB (default -L/opt/homebrew/lib -lvulkan, else -lvulkan).
# Links the release std archive 'safeguard build --release' makes for this
# checkout, as tests/stdlib/mps/run.sh does.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SAFEC="${SAFEC:-$ROOT/compiler/build/safec}"
SAFEGUARD="${SAFEGUARD:-safeguard}"
OUT="${OUT:-$HERE/build}"
mkdir -p "$OUT"
CLANG=clang
for c in /opt/homebrew/opt/llvm/bin/clang /usr/local/opt/llvm/bin/clang; do
    if [[ -x "$c" ]]; then CLANG="$c"; break; fi
done
VULKAN_LIB="${VULKAN_LIB:-}"
if [[ -z "$VULKAN_LIB" ]]; then
    VULKAN_LIB="-lvulkan"
    for d in /opt/homebrew/lib /usr/local/lib; do
        if [[ -e "$d/libvulkan.dylib" ]]; then VULKAN_LIB="-L$d -lvulkan"; break; fi
    done
fi

STD="${STD_ARCHIVE:-}"
if [[ -z "$STD" ]]; then
    P="$OUT/_std"
    mkdir -p "$P/src"
    printf '[package]\nname = "vulkan_checks_std"\nversion = "0.1.0"\n\n[build]\nedition = "2026"\n' > "$P/Package.toml"
    echo 'int main() { return 0; }' > "$P/src/main.sc"
    (cd "$P" && SAFEC="$SAFEC" SAFEC_HOME="$ROOT" "$SAFEGUARD" build --release >/dev/null)
    STD="$P/build/deps/libsafec_std.a"
fi

FRAMEWORKS=""
if [[ "$(uname)" == "Darwin" ]]; then FRAMEWORKS="-framework Metal -framework Foundation -framework Accelerate -lobjc"; fi
WRAP="$OUT/clang_link.sh"
printf '#!/bin/sh\nexec "%s" -w "$@" "%s" %s %s -lm -lpthread\n' "$CLANG" "$STD" "$VULKAN_LIB" "$FRAMEWORKS" > "$WRAP"
chmod +x "$WRAP"
for t in vulkan_spirv_check; do
    SAFEC_CLANG="$WRAP" "$SAFEC" "$HERE/$t.sc" -I "$ROOT" --emit-bin -O2 -o "$OUT/$t"
    "$OUT/$t"
done

# ../device/device_check.sc with the vulkan feature (SAFEC_FEATURE_VULKAN
# links device_vulkan.sc in): every std::ml workload on the CPU and on the GPU.
SAFEC_CLANG="$WRAP" "$SAFEC" "$HERE/../device/device_check.sc" -I "$ROOT" -DSAFEC_FEATURE_VULKAN=1 \
    -DCHECK_DEVICE=DEVICE_VULKAN --emit-bin -O2 -o "$OUT/device_check"
"$OUT/device_check"
