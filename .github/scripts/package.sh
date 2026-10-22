#!/usr/bin/env bash
# Lays out a SafeC install the way a release archive ships it (and
# install.sh / install.ps1 copy it to the install prefix):
#
#   <out>/bin/safec, safeguard[, sc-lsp]          (.exe on Windows)
#   <out>/std/...                                  SafeC's std sources
#   <out>/lib/<triple>/libsafec_std.a              std, prebuilt for this machine
#   <out>/lib/<triple>/libsafec_std.stamp          the compiler and target it is for
#   <out>/lib/<triple>/libsafec_std.units          its units' input hashes and symbols
#   <out>/README.md
#
# The archive is every hosted std unit no feature gates, compiled in release
# mode by the packaged safec from the packaged std/ ('safeguard
# std-prebuilt'); a native build with that safec links it instead of
# compiling std (see safeguard's ensureStdLib).
#
#   usage: package.sh <out> <safec> <safeguard> [<sc-lsp>]
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo "usage: $0 <out> <safec> <safeguard> [<sc-lsp>]" >&2
    exit 2
fi
OUT="$1"; SAFEC_IN="$2"; SAFEGUARD_IN="$3"; LSP_IN="${4:-}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

mkdir -p "$OUT/bin" "$OUT/std"
cp "$SAFEC_IN" "$SAFEGUARD_IN" "$OUT/bin/"
if [[ -n "$LSP_IN" ]]; then cp "$LSP_IN" "$OUT/bin/"; fi
cp -R "$ROOT/std/." "$OUT/std/"
cp "$ROOT/README.md" "$OUT/"

# Built with the packaged copies, as the installed toolchain will run: the
# stamp names that safec, and the units' hashes that std/.
SAFEC="$OUT/bin/$(basename "$SAFEC_IN")" SAFEC_HOME="$OUT" SAFEC_STD_ARCHIVE="" \
    "$OUT/bin/$(basename "$SAFEGUARD_IN")" std-prebuilt "$OUT"
ls -l "$OUT"/lib/*/
