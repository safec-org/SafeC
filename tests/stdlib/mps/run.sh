#!/usr/bin/env bash
# Builds and runs the MPS (Metal) backend checks (needs an Apple GPU and
# Homebrew LLVM's clang, which safec's IR targets): attention, and every
# std::ml training workload on the CPU and on the Metal device backend.
#   usage: run.sh
#   Env: SAFEC (default compiler/build/safec), SAFEGUARD (default: on PATH),
#        OUT (build dir), STD_ARCHIVE (a prebuilt release libsafec_std.a).
# Links the release archive of all of std ../std_archive.sh makes for this
# checkout through safeguard (rebuilt incrementally on every run, so it
# follows std edits) rather than listing std sources by hand: the checks
# reach std/ml, whose source set changes as the library grows.
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

STD="${STD_ARCHIVE:-}"
if [[ -z "$STD" ]]; then
    STD="$(SAFEC="$SAFEC" SAFEGUARD="$SAFEGUARD" SAFEC_HOME="$ROOT" "$HERE/../std_archive.sh" "$OUT/_std" 2>/dev/null)"
fi

WRAP="$OUT/clang_link.sh"
printf '#!/bin/sh\nexec "%s" -w "$@" "%s" -framework Metal -framework Foundation -framework Accelerate -lobjc -lm -lpthread\n' \
    "$CLANG" "$STD" > "$WRAP"
chmod +x "$WRAP"
for t in mps_attention_check; do
    SAFEC_CLANG="$WRAP" "$SAFEC" "$HERE/$t.sc" -I "$ROOT" --emit-bin -O2 -o "$OUT/$t"
    "$OUT/$t"
done

# ../device/device_check.sc with the mps feature (SAFEC_FEATURE_MPS links
# device_mps.sc in): every std::ml workload on the CPU and on the GPU.
SAFEC_CLANG="$WRAP" "$SAFEC" "$HERE/../device/device_check.sc" -I "$ROOT" -DSAFEC_FEATURE_MPS=1 \
    -DCHECK_DEVICE=DEVICE_MPS --emit-bin -O2 -o "$OUT/device_check"
"$OUT/device_check"
