#!/usr/bin/env bash
# Prints the path of an archive of every std module, for the checks here
# that link std by hand (mps/, rocm/, cuda/, x11/): 'safeguard build
# --release' of a throwaway project under $1 with [build] std = "all" —
# a normal build compiles only the std modules its sources include, which
# for this project's empty main() is next to nothing. Rebuilt
# incrementally, so it follows std edits.
#   usage: std_archive.sh <dir>     Env: SAFEC, SAFEGUARD (default: on PATH)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
P="$1"
mkdir -p "$P/src"
printf '[package]\nname = "std_all"\nversion = "0.1.0"\n\n[build]\nedition = "2026"\nstd = "all"\n' > "$P/Package.toml"
[[ -f "$P/src/main.sc" ]] || echo 'int main() { return 0; }' > "$P/src/main.sc"
(cd "$P" && SAFEC_HOME="${SAFEC_HOME:-$(cd "$HERE/../.." && pwd)}" "${SAFEGUARD:-safeguard}" build --release >&2)
echo "$P/build/deps/libsafec_std.a"
