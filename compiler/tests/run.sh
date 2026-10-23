#!/usr/bin/env bash
# Compiler regression tests: compiles every tests/**/*.sc at -O2 to LLVM
# IR and checks the output against the '// CHECK:' lines in the source
# (see filecheck.py). A test with '// EXPECT-ERROR: <text>' directives must
# instead fail to compile with exactly those errors (see filecheck.py for
# line-tied vs. free-floating directives).
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
    if grep -q 'EXPECT-ERROR:' "$t"; then
        # Negative test: must be rejected with exactly the expected errors.
        if "$SAFEC_BIN" "$t" --emit-llvm -o "$ll" >"$OUT/err.txt" 2>&1; then
            echo "  [FAIL] $name (compiled, expected errors)"
            FAIL=$((FAIL + 1))
        elif ! python3 "$HERE/filecheck.py" --errors "$t" "$OUT/err.txt" >"$OUT/check.txt" 2>&1; then
            echo "  [FAIL] $name"
            sed 's/^/      /' "$OUT/check.txt"
            FAIL=$((FAIL + 1))
        else
            echo "  [PASS] $name"
            PASS=$((PASS + 1))
        fi
        continue
    fi
    # '// SAFEC-ARGS: <flags>' adds compiler flags (e.g. --target <triple>).
    read -r -a args <<<"$(sed -n 's|^// SAFEC-ARGS: *||p' "$t" | head -1)"
    if ! "$SAFEC_BIN" "$t" -O2 --emit-llvm "${args[@]}" -o "$ll" >"$OUT/err.txt" 2>&1; then
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
