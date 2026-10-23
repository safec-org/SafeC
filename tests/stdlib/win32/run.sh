#!/usr/bin/env bash
# Builds the Win32 checks for x86_64-windows (MinGW, linked with zig cc)
# and runs them under Wine — or natively, with RUN="" on Windows.
#   usage: run.sh            Env: SAFEC, CLANG, ZIG (default zig),
#                            WINE (default wine64), DISPLAY for Wine's X11 driver
# win32_audio_check streams through WASAPI endpoints when Wine has any: on a
# machine without sound hardware, give Wine's ALSA driver a null default
# PCM (ALSA_CONFIG_PATH naming a file with '</usr/share/alsa/alsa.conf>' and
# 'pcm.!default { type plug slave.pcm "null" }').
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SAFEC="${SAFEC:-safec}"; CLANG="${CLANG:-clang}"; ZIG="${ZIG:-zig}"; AR="${AR:-llvm-ar}"
RUN="${RUN-${WINE:-wine64}}"
OUT="${OUT:-$HERE/build}"
TRIPLE=x86_64-pc-windows-gnu
mkdir -p "$OUT/std"
# std archive for Windows: everything except the other platforms' backends.
if [[ ! -f "$OUT/libsafec_std.a" ]]; then
    while IFS= read -r -d '' f; do
        case "$(basename "$f" .sc)" in *_linux|*_bsd|*_posix|*_epoll|*_kqueue|*_cocoa|*_macos|*_wasm|*_x11|*_fb) continue ;; esac
        o="$OUT/std/$(echo "${f#$ROOT/std/}" | tr '/.' '__').o"
        "$SAFEC" -I "$ROOT/std" -I "$ROOT" "$f" --target=$TRIPLE --emit-llvm --compat-preprocessor -o "${o%.o}.ll" 2>/dev/null &&
            "$CLANG" -c --target=$TRIPLE "${o%.o}.ll" -o "$o" -Wno-override-module 2>/dev/null || rm -f "$o"
    done < <(find "$ROOT/std" -name '*.sc' -not -path '*/ml/*' -not -path '*/wasm/*' -not -path '*/hal/*' -print0 | sort -z)
    "$AR" rcs "$OUT/libsafec_std.a" "$OUT"/std/*.o
fi
fails=0
for t in win32_gui_check win32_thread_check win32_audio_check win32_net_check; do
    "$SAFEC" -I "$ROOT/std" -I "$ROOT" "$HERE/$t.sc" --target=$TRIPLE --emit-llvm --compat-preprocessor -o "$OUT/$t.ll"
    "$CLANG" -c --target=$TRIPLE "$OUT/$t.ll" -o "$OUT/$t.o" -Wno-override-module
    "$ZIG" cc -target x86_64-windows-gnu "$OUT/$t.o" "$OUT/libsafec_std.a" -lgdi32 -luser32 -lws2_32 -lole32 -lwinmm -o "$OUT/$t.exe"
    $RUN "$OUT/$t.exe" || fails=$((fails + 1))
done
exit $(( fails > 0 ))
