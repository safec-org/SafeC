#!/usr/bin/env bash
# Lays out a SafeC install the way a release archive ships it (and
# install.sh / install.ps1 copy it to the install prefix):
#
#   <out>/bin/safec, safeguard[, sc-lsp]          (.exe on Windows)
#   <out>/std/...                                  SafeC's std sources
#   <out>/lib/<triple>/std/<section>.a             std, prebuilt for this machine,
#   <out>/lib/<triple>/std/<section>.units         one archive per section: its stamp
#                                                  (compiler, target) and its units'
#                                                  input hashes and symbols
#   <out>/README.md
#   <out>/bin/safec-doc                            the API reference generator
#   <out>/share/safec/doctool/                     ('safeguard doc'): its SafeC
#                                                  sources, and theme/ its assets
#                                                  (style.css, doc.js, doc.wasm,
#                                                  safec_dom.js)
#
# A section is a top-level directory of std/ (collections, gui, ml, ...),
# 'core' the files directly in std/. Each archive holds the section's
# hosted units that no feature gates, compiled optimized (and, on macOS, for
# macOS 11.0 and later) by the packaged safec from the packaged std/
# ('safeguard std-prebuilt'). A native build with that safec links the
# sections holding its units instead of compiling them (see safeguard's
# ensureStdLib); install.sh --std=<sections> installs only some of them.
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

# The API reference generator 'safeguard doc' runs (safeguard/doctool, a
# SafeC program), built by the packaged safeguard against the packaged std:
# bin/safec-doc, its assets in share/safec/doctool/theme, and its sources
# beside them, so a safeguard whose prebuilt one can't run rebuilds it.
# The search client is wasm32 (needs clang's wasm-ld); without it the
# reference works, only search is missing.
EXE=""; [[ "$SAFEGUARD_IN" == *.exe ]] && EXE=".exe"
SG="$OUT/bin/$(basename "$SAFEGUARD_IN")"
DOCSRC="$OUT/share/safec/doctool"
DOCW="$(mktemp -d)"
mkdir -p "$DOCSRC"
(cd "$ROOT/safeguard/doctool" && tar cf - --exclude=build --exclude=deps --exclude=doc.wasm --exclude=safec_dom.js \
    --exclude=Package.lock --exclude=.safec_cache .) | (cd "$DOCSRC" && tar xf -)
cp -R "$DOCSRC/." "$DOCW/"
sg_doc() { (cd "$1" && shift && SAFEC="$OUT/bin/$(basename "$SAFEC_IN")" SAFEC_HOME="$OUT" SAFEC_STD_ARCHIVE="" "$SG" "$@"); }
sg_doc "$DOCW" build --release
cp "$DOCW/build/safec-doc$EXE" "$OUT/bin/"
if sg_doc "$DOCW/client" build --release --target wasm32 && [[ -f "$DOCW/client/build/safec-doc-client" ]]; then
    cp "$DOCW/client/build/safec-doc-client" "$DOCSRC/theme/doc.wasm"
else
    echo "package.sh: warning: the safec-doc search client (wasm32) did not build; no doc.wasm" >&2
fi
cp "$ROOT/std/wasm/safec_dom.js" "$DOCSRC/theme/"
rm -rf "$DOCW"
ls -l "$OUT/bin/" "$DOCSRC/theme/"
