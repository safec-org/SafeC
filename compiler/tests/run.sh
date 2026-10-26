#!/usr/bin/env bash
# Compiler regression tests: compiles every tests/**/*.sc at -O2 to LLVM
# IR and checks the output against the '// CHECK:' lines in the source
# (see filecheck.py). A test with '// EXPECT-ERROR: <text>' directives must
# instead fail to compile with exactly those errors (see filecheck.py for
# line-tied vs. free-floating directives).
#
#   // SAFEC-ARGS: <flags>  extra compiler flags; '%root' is the repository
#                           root ('-I %root' for '#include <std/...>')
#   // SAFEC-RUN            also build it with --emit-bin (safec finds a clang
#                           new enough for its IR) and run it: the program
#                           must exit 0
#
#   SAFEC_BIN=path/to/safec compiler/tests/run.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SAFEC_BIN="${SAFEC_BIN:-$HERE/../build/safec}"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

PASS=0
FAIL=0
while IFS= read -r -d '' t; do
    name="${t#"$HERE"/}"
    ll="$OUT/$(echo "$name" | tr '/' '_').ll"
    # '// SAFEC-ARGS: <flags>' adds compiler flags (e.g. --target <triple>).
    read -r -a args <<<"$(sed -n 's|^// SAFEC-ARGS: *||p' "$t" | head -1 | sed "s|%root|$ROOT|g")"
    if grep -q 'EXPECT-ERROR:' "$t"; then
        # Negative test: must be rejected with exactly the expected errors.
        if "$SAFEC_BIN" "$t" --emit-llvm ${args[@]+"${args[@]}"} -o "$ll" >"$OUT/err.txt" 2>&1; then
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
    # '// SAFEC-DOC-JSON': checks the --emit-doc-json output instead of IR.
    if grep -q '^// SAFEC-DOC-JSON' "$t"; then
        json="$OUT/$(echo "$name" | tr '/' '_').json"
        if ! "$SAFEC_BIN" --emit-doc-json --doc-root "$(dirname "$t")" "$t" -o "$json" >"$OUT/err.txt" 2>&1; then
            echo "  [FAIL] $name (--emit-doc-json)"
            sed 's/^/      /' "$OUT/err.txt"
            FAIL=$((FAIL + 1))
        elif ! python3 "$HERE/filecheck.py" "$t" "$json" >"$OUT/check.txt" 2>&1; then
            echo "  [FAIL] $name"
            sed 's/^/      /' "$OUT/check.txt"
            FAIL=$((FAIL + 1))
        else
            echo "  [PASS] $name"
            PASS=$((PASS + 1))
        fi
        continue
    fi
    if ! "$SAFEC_BIN" "$t" -O2 --emit-llvm ${args[@]+"${args[@]}"} -o "$ll" >"$OUT/err.txt" 2>&1; then
        echo "  [FAIL] $name (compile)"
        sed 's/^/      /' "$OUT/err.txt"
        FAIL=$((FAIL + 1))
    elif ! python3 "$HERE/filecheck.py" "$t" "$ll" >"$OUT/check.txt" 2>&1; then
        echo "  [FAIL] $name"
        sed 's/^/      /' "$OUT/check.txt"
        FAIL=$((FAIL + 1))
    elif grep -q '^// SAFEC-RUN' "$t"; then
        # '// SAFEC-RUN': the built program must also exit 0.
        exe="${ll%.ll}"
        case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) exe="$exe.exe" ;; esac
        if ! "$SAFEC_BIN" "$t" -O2 --emit-bin ${args[@]+"${args[@]}"} -o "$exe" >"$OUT/err.txt" 2>&1; then
            echo "  [FAIL] $name (link)"
            sed 's/^/      /' "$OUT/err.txt"
            FAIL=$((FAIL + 1))
        elif "$exe" >"$OUT/run.txt" 2>&1; rc=$?; [[ "$rc" -ne 0 ]]; then
            echo "  [FAIL] $name (run: exit $rc)"
            sed 's/^/      /' "$OUT/run.txt"
            FAIL=$((FAIL + 1))
        else
            echo "  [PASS] $name (ran)"
            PASS=$((PASS + 1))
        fi
    else
        echo "  [PASS] $name"
        PASS=$((PASS + 1))
    fi
done < <(find "$HERE" -name '*.sc' -print0 | sort -z)

echo "$PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]
