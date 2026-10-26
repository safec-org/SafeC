#!/usr/bin/env bash
# Runs the X11 GUI backend check against a real X server: $DISPLAY if set,
# otherwise a private Xvfb (must be on PATH).
#   usage: run.sh            Env: SAFEC, CLANG, XVFB (default Xvfb)
# Links an archive of all of std (../std_archive.sh, through safeguard:
# SAFEGUARD, default on PATH), or STD_ARCHIVE if set.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SAFEC="${SAFEC:-safec}"; CLANG="${CLANG:-clang}"; XVFB="${XVFB:-Xvfb}"
OUT="${OUT:-$HERE/build}"
mkdir -p "$OUT"
STD="${STD_ARCHIVE:-$(SAFEC="$SAFEC" "$HERE/../std_archive.sh" "$OUT/_std")}"
"$SAFEC" -I "$ROOT/std" -I "$ROOT" "$HERE/x11_check.sc" --emit-llvm --compat-preprocessor -o "$OUT/x11_check.ll"
"$CLANG" -w "$OUT/x11_check.ll" "$STD" -lX11 -ldl -lm -o "$OUT/x11_check"
xpid=""
if [[ -z "${DISPLAY:-}" ]]; then
    n=$(( 90 + RANDOM % 50 ))
    "$XVFB" ":$n" -screen 0 320x240x24 >"$OUT/xvfb.log" 2>&1 & xpid=$!
    trap '[[ -n "$xpid" ]] && kill "$xpid"' EXIT
    for _ in $(seq 50); do [[ -S /tmp/.X11-unix/X$n ]] && break; sleep 0.1; done
    export DISPLAY=":$n"
fi
"$OUT/x11_check"
