#!/usr/bin/env bash
# Builds the SafeC std library natively (for the differential tests' C++
# safec side): build/native-std/libsafec_std.a. Needs SAFEC_HOME.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$HERE/build/native-std"
SAFEC="${SAFEC:-safec}"; CLANG="${SAFEC_CLANG:-clang}"
[[ -f "$OUT/libsafec_std.a" ]] && exit 0
mkdir -p "$OUT/o"
find "$SAFEC_HOME/std" -name '*.sc' -not -path '*/wasm/*' | sort | while read -r f; do
    case "$(basename "$f" .sc)" in *_win32|*_cocoa|*_macos|*_kqueue|*_bsd) continue ;; esac
    case "$f" in */gui/*|*/ml/*|*/audio/*|*/hal/*|*/kernel/*|*/interrupt/*|*/simd/*) continue ;; esac
    o="$OUT/o/$(echo "${f#$SAFEC_HOME/std/}" | tr '/.' '__').o"
    "$SAFEC" -I "$SAFEC_HOME/std" -I "$SAFEC_HOME" "$f" --emit-llvm --compat-preprocessor -O1 -o "$o.ll" 2>/dev/null &&
        "$CLANG" -c "$o.ll" -o "$o" -Wno-override-module 2>/dev/null || rm -f "$o"
    rm -f "$o.ll"
done
ar rcs "$OUT/libsafec_std.a" "$OUT"/o/*.o
echo "built $OUT/libsafec_std.a ($(ls "$OUT"/o/*.o | wc -l) objects)"
