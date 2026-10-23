#!/usr/bin/env bash
# Builds the Win32 checks for x86_64-windows (MinGW, linked with zig cc, or
# x86_64-w64-mingw32-gcc when there's no zig) and runs them under Wine — or
# natively, with RUN="" on Windows.
#   usage: run.sh            Env: SAFEC, CLANG, ZIG (default zig), MINGW_CC
#                            (default x86_64-w64-mingw32-gcc), WINE (default
#                            wine64), DISPLAY for Wine's X11 driver
# win32_tls_check talks to an `openssl s_server` this script starts on the
# host (Wine shares its network) when openssl is installed; without one it
# only checks that the TLS backends load.
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
LIBS=(-lgdi32 -luser32 -lws2_32 -lole32 -lwinmm -lsecur32 -lcrypt32 -lbcrypt)
link() {   # <obj> <exe>
    if command -v "$ZIG" >/dev/null 2>&1; then
        "$ZIG" cc -target x86_64-windows-gnu "$1" "$OUT/libsafec_std.a" "${LIBS[@]}" -o "$2"
    else
        "${MINGW_CC:-x86_64-w64-mingw32-gcc}" "$1" "$OUT/libsafec_std.a" "${LIBS[@]}" -o "$2"
    fi
}
# A local TLS server for win32_tls_check.
SSERVER_PID=""
if command -v openssl >/dev/null 2>&1; then
    export SAFEC_TLS_TEST_PORT=$(( 40000 + RANDOM % 20000 ))
    cp "$ROOT/tests/stdlib/fixtures/tls/root.pem" "$OUT/tls_root.pem"
    export SAFEC_TLS_TEST_ROOT="$OUT/tls_root.pem"
    (cd "$ROOT/tests/stdlib/fixtures/tls" && exec openssl s_server -accept "$SAFEC_TLS_TEST_PORT" -www -quiet \
        -cert leaf-ec.pem -key leaf-ec.key -cert_chain int.pem -alpn http/1.1 >/dev/null 2>&1) &
    SSERVER_PID=$!
    trap '[[ -n "$SSERVER_PID" ]] && kill "$SSERVER_PID" 2>/dev/null' EXIT
    sleep 0.5
fi
fails=0
for t in win32_gui_check win32_thread_check win32_audio_check win32_tls_check; do
    "$SAFEC" -I "$ROOT/std" -I "$ROOT" "$HERE/$t.sc" --target=$TRIPLE --emit-llvm --compat-preprocessor -o "$OUT/$t.ll"
    "$CLANG" -c --target=$TRIPLE "$OUT/$t.ll" -o "$OUT/$t.o" -Wno-override-module
    link "$OUT/$t.o" "$OUT/$t.exe"
    $RUN "$OUT/$t.exe" || fails=$((fails + 1))
done
exit $(( fails > 0 ))
