#!/usr/bin/env bash
# Builds audio_web_check.sc for wasm32 with safeguard and runs it under Node
# against std/audio/audio_web.js (run.mjs supplies the browser stand-ins).
#   usage: run.sh            Env: SAFEGUARD (default safeguard), NODE (default node)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="${OUT:-$HERE/build}"
P="$OUT/proj"
mkdir -p "$P/src"
printf '[package]\nname = "audio_web_check"\nversion = "0.1.0"\n\n[build]\nedition = "2026"\n' > "$P/Package.toml"
cp "$HERE/audio_web_check.sc" "$P/src/main.sc"
(cd "$P" && SAFEC_HOME="${SAFEC_HOME:-$ROOT}" "${SAFEGUARD:-safeguard}" build --release --target wasm32 >&2)
"${NODE:-node}" "$HERE/run.mjs" "$P/build/audio_web_check" "$ROOT/std/audio/audio_web.js"
