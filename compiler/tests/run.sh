#!/usr/bin/env bash
# Compiler regression tests: compiles every tests/**/*.sc at -O2 to LLVM
# IR and checks the output against the '// CHECK:' lines in the source
# (see filecheck.py). A test with '// EXPECT-ERROR: <text>' lines must
# instead fail to compile with exactly those errors (one per line, each
# matched as a substring).
#
#   SAFEC_BIN=path/to/safec compiler/tests/run.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SAFEC_BIN="${SAFEC_BIN:-$HERE/../build/safec}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

PASS=0
FAIL=0
while IFS= read -r -d '' t; do
    name="${t#"$HERE"/}"
    ll="$OUT/$(echo "$name" | tr '/' '_').ll"
    if grep -q '^// EXPECT-ERROR:' "$t"; then
        # Diagnostics test: must fail to compile, with every expected
        # message (a substring) among the errors, and no other errors.
        "$SAFEC_BIN" "$t" --emit-llvm -o "$ll" >"$OUT/err.txt" 2>&1
        want=$(sed -n 's|^// EXPECT-ERROR: ||p' "$t")
        got=$(grep -c 'error:' "$OUT/err.txt" | head -1)
        miss=""
        while IFS= read -r w; do
            grep -qF -- "$w" "$OUT/err.txt" || miss="$miss\n      missing: $w"
        done <<< "$want"
        nwant=$(printf '%s\n' "$want" | grep -c .)
        if [[ -n "$miss" || "$got" -ne "$nwant" ]]; then
            echo "  [FAIL] $name (expected $nwant errors, got $got)"
            [[ -n "$miss" ]] && printf "$miss\n"
            grep 'error:' "$OUT/err.txt" | sed 's/^/      /'
            FAIL=$((FAIL + 1))
        else
            echo "  [PASS] $name"
            PASS=$((PASS + 1))
        fi
        continue
    fi
    if ! "$SAFEC_BIN" "$t" -O2 --emit-llvm -o "$ll" >"$OUT/err.txt" 2>&1; then
        echo "  [FAIL] $name (compile)"
        sed 's/^/      /' "$OUT/err.txt"
        FAIL=$((FAIL + 1))
    elif ! python3 "$HERE/filecheck.py" "$t" "$ll" >"$OUT/check.txt" 2>&1; then
        echo "  [FAIL] $name"
        sed 's/^/      /' "$OUT/check.txt"
        FAIL=$((FAIL + 1))
    else
        echo "  [PASS] $name"
        PASS=$((PASS + 1))
    fi
done < <(find "$HERE" -name '*.sc' -print0 | sort -z)

echo "$PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]
