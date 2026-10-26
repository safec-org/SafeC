#!/usr/bin/env bash
# std/gui on wasm32 in headless Chromium (Playwright): builds
# examples/gui_gallery natively (its --snapshot frame is the reference,
# rendered with the DejaVu fonts it finds) and for wasm32, and
# examples/gui_web for wasm32, then runs browser.mjs. Screenshots land in
# $OUT (default build/browser). Skips (exit 0) without Playwright.
#   usage: browser.sh        Env: SAFEGUARD, NODE, NODE_PATH, PLAYWRIGHT_BROWSERS_PATH, OUT
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SAFEGUARD="${SAFEGUARD:-safeguard}"; NODE="${NODE:-node}"
export SAFEC_HOME="${SAFEC_HOME:-$ROOT}"
export NODE_PATH="${NODE_PATH:-$(npm root -g 2>/dev/null || true)}"
OUT="${OUT:-$HERE/build/browser}"
if ! "$NODE" -e "require('playwright')" 2>/dev/null; then echo "  [SKIP] playwright not installed"; exit 0; fi
FONTS=(/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf)
for f in "${FONTS[@]}"; do [[ -f "$f" ]] || { echo "  [SKIP] $f missing (the reference frame needs DejaVu Sans)"; exit 0; }; done
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT
cp -R "$ROOT/examples/gui_gallery" "$W/gallery"
cp -R "$ROOT/examples/gui_web" "$W/web"
rm -rf "$W/gallery/build" "$W/web/build"
(cd "$W/gallery" && "$SAFEGUARD" build >/dev/null && SAFEC_GUI_FONT_DIR= ./build/gui_gallery --snapshot "$W" >/dev/null &&
    "$SAFEGUARD" build --target wasm32 >/dev/null 2>&1)
(cd "$W/web" && "$SAFEGUARD" build --target wasm32 >/dev/null 2>&1 && "$SAFEGUARD" bundle --target wasm32 >/dev/null 2>&1)
mkdir -p "$OUT"
"$NODE" "$HERE/browser.mjs" "$W/gallery/build/gui_gallery" "$W" "$W/web/build/gui_web" "$ROOT/std/gui/gui_web.js" "$OUT" "$ROOT/examples/gui_web/web/index.html" "$W/web/build/GuiWeb.web" "${FONTS[@]}"
