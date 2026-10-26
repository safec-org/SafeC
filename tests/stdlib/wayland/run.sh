#!/usr/bin/env bash
# Runs the Wayland GUI backend check against real compositors:
#   1. sway, headless (wlroots; no GPU needed: the pixman renderer), at
#      output scale 1 and again at a fractional 1.5 — windows, frames read
#      back with grim, input injected through the virtual-keyboard and
#      virtual-pointer protocols (wtype, wlrctl), the clipboard against
#      wl-copy/wl-paste, resizes and closes through swaymsg, xdg-decoration,
#      fractional-scale, cursor-shape;
#      plus the EGL/OpenGL ES presenter (Mesa llvmpipe, LIBGL_ALWAYS_SOFTWARE);
#   2. Weston, headless (no xdg-decoration, no cursor-shape): windows,
#      presenting, multi-window, close.
# Needs sway, grim, wtype, wlrctl, wl-clipboard, swaymsg and weston on PATH
# (apt-get install sway grim wtype wlrctl wl-clipboard weston); a missing
# compositor's part is skipped with a note. With $WAYLAND_DISPLAY already
# set, runs part 1 against that compositor instead.
#   usage: run.sh            Env: SAFEC, CLANG, SAFEGUARD
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SAFEC="${SAFEC:-safec}"; CLANG="${CLANG:-clang}"
OUT="${OUT:-$HERE/build}"
mkdir -p "$OUT"
STD="${STD_ARCHIVE:-$(SAFEC="$SAFEC" "$HERE/../std_archive.sh" "$OUT/_std")}"
"$SAFEC" -I "$ROOT/std" -I "$ROOT" "$HERE/wayland_check.sc" --emit-llvm --compat-preprocessor -o "$OUT/wayland_check.ll"
"$CLANG" -w "$OUT/wayland_check.ll" "$STD" -lX11 -lwayland-client -lxkbcommon -ldl -lm -o "$OUT/wayland_check"
unset DISPLAY
# Mesa without a GPU: EGL on Wayland through llvmpipe and wl_shm.
export LIBGL_ALWAYS_SOFTWARE=1

# Compositor sockets must fit in sun_path: a short private runtime dir.
RT="$(mktemp -d /tmp/safec-wl.XXXXXX)"
chmod 700 "$RT"
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; rm -rf "$RT"; }
trap cleanup EXIT
wait_socket() { for _ in $(seq 100); do [[ -S "$RT/$1" ]] && return 0; sleep 0.05; done; return 1; }

status=0
if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
    "$OUT/wayland_check" full || status=1
    exit $status
fi

run_sway() {   # $1: output scale
    local cfg="$RT/sway-$1.cfg"
    cat > "$cfg" <<CFG
output HEADLESS-1 resolution 640x480 position 0 0 scale $1
default_border none
default_floating_border none
focus_follows_mouse no
for_window [app_id=".*"] floating enable
for_window [title="wl first"] move position 0 0
for_window [title="wl second"] move position 320 0
seat seat0 hide_cursor 0
CFG
    rm -f "$RT"/wayland-*
    XDG_RUNTIME_DIR="$RT" WLR_BACKENDS=headless WLR_LIBINPUT_NO_DEVICES=1 WLR_RENDERER=pixman \
        sway -c "$cfg" >"$OUT/sway-$1.log" 2>&1 &
    local sp=$!
    pids+=("$sp")
    wait_socket wayland-1 || { echo "sway did not start (see $OUT/sway-$1.log)"; return 1; }
    local rc=0
    XDG_RUNTIME_DIR="$RT" WAYLAND_DISPLAY=wayland-1 SWAYSOCK="$(ls "$RT"/sway-ipc.*.sock)" \
        SAFEC_WL_OUTPUT_SCALE="$1" "$OUT/wayland_check" "$2" || rc=1
    kill "$sp" 2>/dev/null || true
    wait "$sp" 2>/dev/null || true
    return $rc
}

if command -v sway >/dev/null && command -v grim >/dev/null && command -v wtype >/dev/null &&
   command -v wlrctl >/dev/null && command -v wl-copy >/dev/null; then
    echo "== sway (headless), scale 1"
    run_sway 1 full || status=1
    echo "== sway (headless), scale 1: the EGL presenter"
    run_sway 1 gl || status=1
    echo "== sway (headless), scale 1.5 (+ the EGL presenter)"
    run_sway 1.5 scaled || status=1
else
    echo "== sway/grim/wtype/wlrctl/wl-clipboard not installed: skipped"
fi

if command -v weston >/dev/null; then
    echo "== weston (headless)"
    rm -f "$RT"/wayland-*
    XDG_RUNTIME_DIR="$RT" weston --backend=headless --socket=wayland-w --idle-time=0 \
        >"$OUT/weston.log" 2>&1 &
    wp=$!
    pids+=("$wp")
    if wait_socket wayland-w; then
        XDG_RUNTIME_DIR="$RT" WAYLAND_DISPLAY=wayland-w "$OUT/wayland_check" basic || status=1
    else
        echo "weston did not start (see $OUT/weston.log)"; status=1
    fi
    kill "$wp" 2>/dev/null || true
else
    echo "== weston not installed: skipped"
fi
exit $status
