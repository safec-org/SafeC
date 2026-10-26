#!/usr/bin/env bash
# Compiler regression tests: compiles every tests/**/*.sc at -O2 to LLVM
# IR and checks the output against the '// CHECK:' lines in the source
# (see filecheck.py). A test with '// EXPECT-ERROR: <text>' directives must
# instead fail to compile with exactly those errors (see filecheck.py for
# line-tied vs. free-floating directives). A test with a '// EXEC' line is
# also built with --emit-bin and run, and must exit 0 (skipped, with a note,
# when safec finds no clang to link with).
#
#   // SAFEC-ARGS: <flags>  extra compiler flags; '%root' is the repository
#                           root ('-I %root' for '#include <std/...>')
#
#   SAFEC_BIN=path/to/safec compiler/tests/run.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SAFEC_BIN="${SAFEC_BIN:-$HERE/../build/safec}"
ROOT="$(cd "$HERE/../.." && pwd)"
# The clang safec links with (--emit-bin): one new enough for its IR, which
# the system's default clang may not be (macOS).
CLANG_BIN="$("$SAFEC_BIN" --print-clang 2>/dev/null)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

PASS=0
FAIL=0
while IFS= read -r -d '' t; do
    name="${t#"$HERE"/}"
    ll="$OUT/$(echo "$name" | tr '/' '_').ll"
    # '// REQUIRES-HOST: <uname -s>' (e.g. Linux): a test of the host's own
    # system headers, run only there.
    host="$(sed -n 's|^// REQUIRES-HOST: *||p' "$t" | head -1)"
    if [[ -n "$host" && "$host" != "$(uname -s)" ]]; then
        echo "  [SKIP] $name (host is not $host)"
        continue
    fi
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
    elif grep -q '^// EXEC$' "$t" && command -v "$CLANG_BIN" >/dev/null 2>&1; then
        exe="${ll%.ll}.exe"
        if ! "$SAFEC_BIN" "$t" -O2 --emit-bin ${args[@]+"${args[@]}"} -o "$exe" >"$OUT/err.txt" 2>&1; then
            echo "  [FAIL] $name (link)"
            sed 's/^/      /' "$OUT/err.txt"
            FAIL=$((FAIL + 1))
        elif "$exe" >"$OUT/run.txt" 2>&1; rc=$?; [[ "$rc" -ne 0 ]]; then
            echo "  [FAIL] $name (exited $rc)"
            sed 's/^/      /' "$OUT/run.txt"
            FAIL=$((FAIL + 1))
        else
            echo "  [PASS] $name"
            PASS=$((PASS + 1))
        fi
    else
        grep -q '^// EXEC$' "$t" && echo "  [NOTE] $name: no clang to link with, not run"
        echo "  [PASS] $name"
        PASS=$((PASS + 1))
    fi
done < <(find "$HERE" -name '*.sc' -print0 | sort -z)

echo "$PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]
