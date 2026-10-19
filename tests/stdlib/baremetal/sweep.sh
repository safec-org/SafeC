#!/usr/bin/env bash
# Compiles + assembles every std/*.sc for each bare-metal target alias
# (safec --list-targets) and prints a pass/fail matrix.
#   usage: sweep.sh [alias...]        (default: every alias)
# Env: SAFEC (default: safec on PATH), CLANG (default: clang).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SAFEC="${SAFEC:-safec}"
CLANG="${CLANG:-clang}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
if [[ $# -gt 0 ]]; then TARGETS=("$@"); else mapfile -t TARGETS < <("$SAFEC" --list-targets | awk '{print $1}'); fi

# Sources that only make sense on one architecture family (or need an OS),
# and are therefore expected to fail elsewhere.
expected_fail() {  # $1 = target alias, $2 = path relative to std/
    case "$2" in
        hal/aarch64.sc)   [[ $1 != aarch64 ]] ;;
        hal/riscv.sc)     [[ $1 != rv* ]] ;;
        hal/cortex_m.sc)  [[ $1 != thumb* && $1 != arm-*m ]] ;;
        simd/cortex_m.sc) [[ $1 != arm-7em && $1 != arm-8m ]] ;;   # DSP extension: v7E-M / v8-M Mainline only
        *) false ;;
    esac
}

total_unexpected=0
for t in "${TARGETS[@]}"; do
    mapfile -t CFLAGS < <("$SAFEC" --print-target-clang-flags --target="$t")
    pass=0; fail=0; xfail=0; unexpected=()
    while IFS= read -r -d '' f; do
        rel="${f#$ROOT/std/}"
        ll="$WORK/$t.ll"; obj="$WORK/$t.o"
        if "$SAFEC" -I "$ROOT/std" -I "$ROOT" "$f" --target="$t" --emit-llvm --compat-preprocessor \
                -o "$ll" >"$WORK/err" 2>&1 &&
           "$CLANG" -c "$ll" -o "$obj" "${CFLAGS[@]}" -Wno-override-module >>"$WORK/err" 2>&1; then
            pass=$((pass + 1))
        elif expected_fail "$t" "$rel"; then
            xfail=$((xfail + 1))
        else
            fail=$((fail + 1))
            unexpected+=("$rel: $(grep -m1 -E 'error' "$WORK/err" | cut -c1-160)")
        fi
    done < <(find "$ROOT/std" -name '*.sc' -not -path '*/wasm/*' -print0 | sort -z)
    printf '%-8s pass=%-4d expected-fail=%-3d UNEXPECTED=%d\n' "$t" "$pass" "$xfail" "$fail"
    for u in "${unexpected[@]}"; do echo "    $u"; done
    total_unexpected=$((total_unexpected + fail))
done
exit $(( total_unexpected > 0 ))
