#!/usr/bin/env bash
set -uo pipefail
# Deliberately not 'set -e' — this script tracks pass/fail per check itself
# and reports a summary, rather than dying on the first failure, so one CI
# run surfaces every broken check instead of just the first one.

# Expects these environment variables (set by the calling workflow step):
#   SAFEC_BIN       path to the safec binary
#   SAFEGUARD_BIN   path to the safeguard binary
#   SC_LSP_BIN      path to the sc-lsp binary
#   SAFEC_HOME      SafeC repo root (for -I std, and exported for safeguard)
#   CLANG_BIN       clang binary name/path to link with
export SAFEC_HOME

FAILURES=0
pass() { echo "  [PASS] $1"; }
fail() { echo "  [FAIL] $1"; FAILURES=$((FAILURES + 1)); }

section() { echo ""; echo "── $1 ──────────────────────────────────────────"; }

# ── 1. safec --help ───────────────────────────────────────────────────────────
section "safec --help"
if "$SAFEC_BIN" --help >/tmp/safec_help.txt 2>&1; then
    pass "safec --help exits 0"
else
    fail "safec --help exited non-zero"
    cat /tmp/safec_help.txt
fi

# ── 2. Compile a trivial hello-world ──────────────────────────────────────────
section "Compile hello.sc"
cat > /tmp/hello_smoke.sc <<'EOF'
extern int printf(const char* fmt, ...);
int main() {
    unsafe { printf("hello from CI\n"); }
    return 0;
}
EOF
if "$SAFEC_BIN" /tmp/hello_smoke.sc --emit-llvm -o /tmp/hello_smoke.ll >/tmp/hello_out.txt 2>&1; then
    pass "compiled hello.sc to LLVM IR"
else
    fail "failed to compile hello.sc"
    cat /tmp/hello_out.txt
fi

if command -v "$CLANG_BIN" >/dev/null 2>&1 && [[ -f /tmp/hello_smoke.ll ]]; then
    if "$CLANG_BIN" /tmp/hello_smoke.ll -o /tmp/hello_smoke >/tmp/hello_link.txt 2>&1; then
        OUT="$(/tmp/hello_smoke 2>&1)"
        if [[ "$OUT" == "hello from CI" ]]; then
            pass "linked and ran hello.sc, output matched"
        else
            fail "hello.sc ran but output was: '$OUT'"
        fi
    else
        fail "failed to link hello.sc with $CLANG_BIN"
        cat /tmp/hello_link.txt
    fi
fi

# ── 3. Standard library compile sweep ─────────────────────────────────────────
# --emit-llvm only (Preprocess/Lex/Parse/Sema/ConstEval/CodeGen-to-IR) — no
# assembler stage, so this only catches front-end regressions, not "does
# this target's assembler accept the resulting inline asm" (that needs the
# matching --target too, e.g. std/hal/riscv.sc only assembles when
# targeting riscv, which is a separate, expected condition, not a bug).
# The one file expected to fail even at this stage on a non-ARM host is
# std/simd/cortex_m.sc, whose ARM-only DSP builtins are rejected by Sema
# itself when there's no ARM target selected — by design, verified
# extensively during development (see reference/baremetal.md).
section "Standard library compile sweep"
STD_TOTAL=0
STD_FAILED=0
while IFS= read -r -d '' f; do
    STD_TOTAL=$((STD_TOTAL + 1))
    if "$SAFEC_BIN" -I "$SAFEC_HOME/std" -I "$SAFEC_HOME" "$f" \
            --emit-llvm --compat-preprocessor -o /tmp/std_sweep.ll >/tmp/std_sweep_err.txt 2>&1; then
        continue
    fi
    if [[ "$f" == *"std/simd/cortex_m.sc" ]]; then
        continue  # expected — see comment above
    fi
    STD_FAILED=$((STD_FAILED + 1))
    echo "    unexpected failure: $f"
    sed 's/^/      /' /tmp/std_sweep_err.txt
done < <(find "$SAFEC_HOME/std" -name "*.sc" -print0)

if [[ "$STD_FAILED" -eq 0 ]]; then
    pass "standard library: $STD_TOTAL files, 0 unexpected failures"
else
    fail "standard library: $STD_TOTAL files, $STD_FAILED unexpected failure(s)"
fi

# ── 4. safeguard: scaffold, build, run ────────────────────────────────────────
section "safeguard new / build / run"
WORKDIR="$(mktemp -d)"
pushd "$WORKDIR" >/dev/null

if "$SAFEGUARD_BIN" new ci_smoke_proj >/tmp/sg_new.txt 2>&1; then
    pass "safeguard new"
else
    fail "safeguard new failed"
    cat /tmp/sg_new.txt
fi

cd ci_smoke_proj 2>/dev/null || { fail "project directory not created"; popd >/dev/null; exit 1; }

if "$SAFEGUARD_BIN" build >/tmp/sg_build.txt 2>&1; then
    pass "safeguard build"
else
    fail "safeguard build failed"
    cat /tmp/sg_build.txt
fi

BIN_NAME="ci_smoke_proj"
[[ -f "build/${BIN_NAME}.exe" ]] && BIN_NAME="${BIN_NAME}.exe"
if [[ -f "build/${BIN_NAME}" ]]; then
    RUN_OUT="$("./build/${BIN_NAME}" 2>&1)"
    if [[ "$RUN_OUT" == *"Hello from ci_smoke_proj"* ]]; then
        pass "safeguard-built binary ran, output matched"
    else
        fail "safeguard-built binary ran but output was: '$RUN_OUT'"
    fi
else
    fail "safeguard build did not produce build/${BIN_NAME}"
fi

# ── 5. safeguard check / lint / format / test ─────────────────────────────────
section "safeguard check / lint / format / test"

if "$SAFEGUARD_BIN" check >/tmp/sg_check.txt 2>&1; then
    pass "safeguard check"
else
    fail "safeguard check failed"
    cat /tmp/sg_check.txt
fi

if "$SAFEGUARD_BIN" lint >/tmp/sg_lint.txt 2>&1; then
    pass "safeguard lint"
else
    fail "safeguard lint reported error-level diagnostics"
    cat /tmp/sg_lint.txt
fi

if "$SAFEGUARD_BIN" format --check >/tmp/sg_fmt.txt 2>&1; then
    pass "safeguard format --check (freshly scaffolded project is already clean)"
else
    fail "safeguard format --check found unformatted files in a fresh scaffold"
    cat /tmp/sg_fmt.txt
fi

if "$SAFEGUARD_BIN" test >/tmp/sg_test.txt 2>&1; then
    pass "safeguard test (no tests/ directory — trivially passes)"
else
    fail "safeguard test failed unexpectedly with no tests/ directory"
    cat /tmp/sg_test.txt
fi

popd >/dev/null
rm -rf "$WORKDIR"

# ── 6. sc-lsp wire-protocol smoke test ────────────────────────────────────────
section "sc-lsp initialize handshake"
REQUEST='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"rootUri":null,"capabilities":{}}}'
REQUEST_LEN=${#REQUEST}
LSP_OUT="$(printf 'Content-Length: %d\r\n\r\n%s' "$REQUEST_LEN" "$REQUEST" | "$SC_LSP_BIN" 2>&1 | head -c 2000)"
if [[ "$LSP_OUT" == *'"id"'*'1'* || "$LSP_OUT" == *"capabilities"* ]]; then
    pass "sc-lsp responded to initialize"
else
    fail "sc-lsp did not respond as expected to initialize"
    echo "    got: $LSP_OUT"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
if [[ "$FAILURES" -eq 0 ]]; then
    echo "  All smoke tests passed."
    exit 0
else
    echo "  $FAILURES smoke test(s) failed."
    exit 1
fi
