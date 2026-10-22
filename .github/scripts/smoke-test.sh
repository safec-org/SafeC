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
# Run safeguard in a directory, output to a log; true if it succeeded.
sg_in() { local dir="$1" log="$2"; shift 2; (cd "$dir" && "$SAFEGUARD_BIN" "$@") >"$log" 2>&1; }

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

# ── 3b. Compiler IR regression tests ──────────────────────────────────────────
# compiler/tests/**/*.sc compiled at -O2 and checked against their
# '// CHECK:' lines (e.g. volatile accesses surviving optimization).
section "Compiler IR tests"
if command -v python3 >/dev/null 2>&1; then
    if SAFEC_BIN="$SAFEC_BIN" bash "$SAFEC_HOME/compiler/tests/run.sh" >/tmp/ir_tests.txt 2>&1; then
        pass "compiler IR tests: $(tail -1 /tmp/ir_tests.txt)"
    else
        fail "compiler IR tests"
        cat /tmp/ir_tests.txt
    fi
else
    echo "  [SKIP] python3 not available"
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

# std modules: the build compiled only the std it uses — for the scaffold,
# which includes no std header and calls nothing safec routes to std's
# runtime ('new', channels, 'spawn'), none of it.
STD_OBJS=$(ls build/deps/std/*.o 2>/dev/null | wc -l | tr -d ' ')
if [[ "$STD_OBJS" -eq 0 ]] && "$SAFEGUARD_BIN" std-modules >/tmp/sg_stdmods.txt 2>&1 &&
   ! grep -q "^unit " /tmp/sg_stdmods.txt; then
    pass "safeguard build compiled no std module for a program using none (std-modules agrees)"
else
    fail "std modules of the scaffold ($STD_OBJS objects)"
    cat /tmp/sg_stdmods.txt 2>/dev/null
fi

# The runtime units come with what the code calls into: a hello world on
# std/io.h has no thread or channel module (nor -lpthread on Linux); a
# channel ('chan_create', no #include) brings sync/channel.sc, thread.sc and
# -lpthread; so does 'spawn' (pthread_create) with thread.sc.
mkdir -p "$WORKDIR/ci_rt/src"
printf '[package]\nname = "ci_rt"\nversion = "0.1.0"\n\n[build]\nedition = "2026"\n' > "$WORKDIR/ci_rt/Package.toml"
printf '#include <std/io.h>\nint main() { std::println("hi"); return 0; }\n' > "$WORKDIR/ci_rt/src/hello.sc"
printf 'int main() {\n    void* ch = (void*)0;\n    unsafe { ch = chan_create(1); chan_close(ch); }\n    return 0;\n}\n' > "$WORKDIR/ci_rt/src/chan.sc"
printf 'void* work(void* a) { return a; }\nint main() {\n    long long h = spawn(work, 0);\n    join(h);\n    return 0;\n}\n' > "$WORKDIR/ci_rt/src/spawn.sc"
RT_OK=1
for f in hello chan spawn; do
    sg_in "$WORKDIR/ci_rt" "/tmp/sg_rt_$f.txt" std-modules --target x86_64-unknown-linux-gnu "src/$f.sc" || RT_OK=0
done
grep -qE "^unit std/(thread|sync/channel)\.sc|^link -lpthread" /tmp/sg_rt_hello.txt && RT_OK=0
grep -q "^unit std/sync/channel.sc" /tmp/sg_rt_chan.txt && grep -q "^unit std/thread.sc" /tmp/sg_rt_chan.txt &&
    grep -q "^link -lpthread" /tmp/sg_rt_chan.txt || RT_OK=0
grep -q "^unit std/thread.sc" /tmp/sg_rt_spawn.txt && grep -q "^link -lpthread" /tmp/sg_rt_spawn.txt &&
    ! grep -q "^unit std/sync/channel.sc" /tmp/sg_rt_spawn.txt || RT_OK=0
if [[ "$RT_OK" -eq 1 ]]; then
    pass "runtime std modules follow the code: no thread/channel/-lpthread for hello, both for channels"
else
    fail "runtime std modules (std-modules for Linux)"
    head -50 /tmp/sg_rt_hello.txt /tmp/sg_rt_chan.txt /tmp/sg_rt_spawn.txt
fi

# std/crypto/rng seeds from the OS (BCryptGenRandom on Windows, which has
# neither /dev/urandom nor, on every CPU, rdrand): two contexts and two runs
# differ, and std's annotation links bcrypt.
mkdir -p "$WORKDIR/ci_rng/src"
printf '[package]\nname = "ci_rng"\nversion = "0.1.0"\n\n[build]\nedition = "2026"\n' > "$WORKDIR/ci_rng/Package.toml"
cat > "$WORKDIR/ci_rng/src/main.sc" <<'EOF'
#include <std/crypto/rng.h>
extern int printf(const char* fmt, ...);
int main() {
    struct RngCtx a;
    struct RngCtx b;
    a.init();
    b.init();
    unsigned long long x = a.rand64();
    unsigned long long y = b.rand64();
    unsigned long long z = a.rand64();
    unsafe { printf("rng %llx %llx %llx\n", x, y, z); }
    if (x == y || x == z) { return 1; }
    return 0;
}
EOF
RNG1=""; RNG2=""
if sg_in "$WORKDIR/ci_rng" /tmp/sg_rng.txt build; then
    RNG_BIN="$WORKDIR/ci_rng/build/ci_rng"; [[ -f "$RNG_BIN.exe" ]] && RNG_BIN="$RNG_BIN.exe"
    RNG1="$("$RNG_BIN" 2>&1)" && RNG2="$("$RNG_BIN" 2>&1)" || RNG1="failed: $RNG1"
fi
if [[ "$RNG1" == "rng "* && "$RNG2" == "rng "* && "$RNG1" != "$RNG2" ]]; then
    pass "std/crypto/rng seeds from the OS ($RNG1)"
else
    fail "std/crypto/rng seeding (runs: '$RNG1' / '$RNG2')"
    cat /tmp/sg_rng.txt
fi

# A library build exports its headers (include/ and src/'s .h files) into
# build/include beside build/lib<name>.a, so build/ is a prebuilt dependency.
mkdir -p "$WORKDIR/ci_lib/src" "$WORKDIR/ci_lib/include/ci_lib"
printf '[package]\nname = "ci_lib"\nversion = "0.1.0"\n\n[build]\nedition = "2026"\ncrate_type = "staticlib"\n' \
    > "$WORKDIR/ci_lib/Package.toml"
printf '#pragma once\n#define CI_LIB_BASE 40\n' > "$WORKDIR/ci_lib/include/ci_lib/config.h"
printf '#pragma once\n#include <ci_lib/config.h>\nint ci_lib_answer();\n' > "$WORKDIR/ci_lib/src/ci_lib.h"
printf '#include "ci_lib.h"\nint ci_lib_answer() { return CI_LIB_BASE + 2; }\n' > "$WORKDIR/ci_lib/src/ci_lib.sc"
if (cd "$WORKDIR/ci_lib" && "$SAFEGUARD_BIN" build >/tmp/sg_lib.txt 2>&1) &&
   [[ -f "$WORKDIR/ci_lib/build/libci_lib.a" && -f "$WORKDIR/ci_lib/build/include/ci_lib.h" &&
      -f "$WORKDIR/ci_lib/build/include/ci_lib/config.h" ]]; then
    pass "safeguard build (staticlib) exports build/include beside build/libci_lib.a"
else
    fail "safeguard build (staticlib) did not produce build/libci_lib.a with build/include"
    cat /tmp/sg_lib.txt
fi

# ── 5b. safeguard: other languages, built dependencies, cdylibs, bundles ─────
section "safeguard: C/C++/ObjC packages and dependencies, cdylib, bundle"
case "$(uname -s)" in
    Darwin) OS=macos ;;
    Linux) OS=linux ;;
    MINGW*|MSYS*|CYGWIN*) OS=windows ;;
    *) OS=other ;;
esac
EXE=""; [[ "$OS" == windows ]] && EXE=".exe"
# An LLVM tool for inspecting binaries: beside $CLANG_BIN or its realpath
# (apt.llvm.org's /usr/bin/clang-21 links into /usr/lib/llvm-21/bin), then
# <tool>-<clang major>, then <tool> on PATH. Prints nothing if none is found.
llvm_tool() {
    local clang real d major
    clang="$(command -v "${CLANG_BIN:-clang}" 2>/dev/null || true)"
    if [[ -n "$clang" ]]; then
        real="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$clang" 2>/dev/null || echo "$clang")"
        for d in "$(dirname "$clang")" "$(dirname "$real")"; do
            if [[ -x "$d/$1" ]]; then echo "$d/$1"; return; fi
            if [[ -x "$d/$1.exe" ]]; then echo "$d/$1.exe"; return; fi
        done
    fi
    major="$(basename "${CLANG_BIN:-clang}" .exe | sed -nE 's/.*-([0-9]+)$/\1/p')"
    if [[ -z "$major" && -n "$clang" ]]; then
        major="$("$clang" --version 2>/dev/null | sed -nE 's/.*clang version ([0-9]+).*/\1/p' | head -1)"
    fi
    if [[ -n "$major" ]] && command -v "$1-$major" >/dev/null 2>&1; then command -v "$1-$major"; return; fi
    command -v "$1" 2>/dev/null || true
}
# Run a command, killed after $1 seconds (a GUI program must exit by itself).
with_timeout() { local t="$1"; shift; perl -e 'alarm shift; exec @ARGV' "$t" "$@"; }

# A plain C package (no SafeC): built without std, run.
if (cd "$WORKDIR" && "$SAFEGUARD_BIN" new ci_cpkg --lang c >/dev/null 2>&1) &&
   sg_in "$WORKDIR/ci_cpkg" /tmp/sg_cpkg.txt run && grep -q "Hello from ci_cpkg!" /tmp/sg_cpkg.txt &&
   [[ ! -e "$WORKDIR/ci_cpkg/build/deps/libsafec_std.a" ]]; then
    pass "safeguard new --lang c: a C package builds and runs without std"
else
    fail "safeguard new --lang c package"
    cat /tmp/sg_cpkg.txt
fi

# A C static library exporting its headers, used by a SafeC program.
mkdir -p "$WORKDIR/ci_cstat/src" "$WORKDIR/ci_cstat/include" "$WORKDIR/ci_cstat_app/src"
cat > "$WORKDIR/ci_cstat/Package.toml" <<'EOF'
[package]
name = "ci_cstat"
version = "0.1.0"

[build]
edition = "2026"
crate_type = "staticlib"
EOF
cat > "$WORKDIR/ci_cstat/include/ci_cstat.h" <<'EOF'
#pragma once
int ci_cstat_sum(const int* values, int n);
EOF
cat > "$WORKDIR/ci_cstat/src/ci_cstat.c" <<'EOF'
#include <ci_cstat.h>
int ci_cstat_sum(const int* values, int n) {
    int s = 0;
    for (int i = 0; i < n; ++i) s += values[i];
    return s;
}
EOF
cat > "$WORKDIR/ci_cstat_app/Package.toml" <<'EOF'
[package]
name = "ci_cstat_app"
version = "0.1.0"

[build]
edition = "2026"

[[dependencies]]
name = "ci_cstat"
path = "../ci_cstat/build"
EOF
cat > "$WORKDIR/ci_cstat_app/src/main.sc" <<'EOF'
#include <ci_cstat.h>
extern int printf(const char* fmt, ...);
int main() {
    int v[3] = { 1, 2, 39 };
    unsafe { printf("ci_cstat sum %d\n", ci_cstat_sum(&v[0], 3)); }
    return 0;
}
EOF
if sg_in "$WORKDIR/ci_cstat" /tmp/sg_cstat.txt build &&
   [[ -f "$WORKDIR/ci_cstat/build/libci_cstat.a" && -f "$WORKDIR/ci_cstat/build/include/ci_cstat.h" ]] &&
   sg_in "$WORKDIR/ci_cstat_app" /tmp/sg_cstat.txt run && grep -q "ci_cstat sum 42" /tmp/sg_cstat.txt; then
    pass "a C package built as a staticlib (with build/include) links into a SafeC program"
else
    fail "C staticlib package"
    cat /tmp/sg_cstat.txt
fi

# A cdylib and a program using it: lib<name>.so/.dylib with an rpath, or on
# Windows <name>.dll + its import library <name>.lib, the DLL copied beside
# the program.
mkdir -p "$WORKDIR/ci_dylib/src" "$WORKDIR/ci_dylib/include" "$WORKDIR/ci_dyapp/src"
cat > "$WORKDIR/ci_dylib/Package.toml" <<'EOF'
[package]
name = "ci_dylib"
version = "0.1.0"

[build]
edition = "2026"
crate_type = "cdylib"
EOF
cat > "$WORKDIR/ci_dylib/include/ci_dylib.h" <<'EOF'
#pragma once
int ci_dylib_add(int a, int b);
EOF
cat > "$WORKDIR/ci_dylib/src/ci_dylib.sc" <<'EOF'
#include <ci_dylib.h>
static int twice(int x) { return x * 2; }
int ci_dylib_add(int a, int b) { return twice(a) + b; }
EOF
cat > "$WORKDIR/ci_dyapp/Package.toml" <<'EOF'
[package]
name = "ci_dyapp"
version = "0.1.0"
author = "SafeC CI <ci@example.org>"

[build]
edition = "2026"

[[dependencies]]
name = "ci_dylib"
path = "../ci_dylib/build"

[bundle]
identifier = "org.safec.ci-dyapp"
subsystem  = "console"
EOF
cat > "$WORKDIR/ci_dyapp/src/main.sc" <<'EOF'
#include <ci_dylib.h>
extern int printf(const char* fmt, ...);
int main() {
    unsafe { printf("ci_dylib says %d\n", ci_dylib_add(20, 2)); }
    return 0;
}
EOF
case "$OS" in
    windows) DYLIB_FILES=(ci_dylib.dll ci_dylib.lib include/ci_dylib.h) ;;
    macos)   DYLIB_FILES=(libci_dylib.dylib include/ci_dylib.h) ;;
    *)       DYLIB_FILES=(libci_dylib.so include/ci_dylib.h) ;;
esac
DYLIB_OK=1
sg_in "$WORKDIR/ci_dylib" /tmp/sg_dylib.txt build || DYLIB_OK=0
for f in "${DYLIB_FILES[@]}"; do [[ -f "$WORKDIR/ci_dylib/build/$f" ]] || { DYLIB_OK=0; echo "    missing build/$f"; }; done
if [[ "$DYLIB_OK" -eq 1 ]]; then
    pass "safeguard build (cdylib): ${DYLIB_FILES[*]}"
else
    fail "safeguard build (cdylib)"
    cat /tmp/sg_dylib.txt
fi
if [[ "$OS" == windows ]]; then
    READOBJ="$(llvm_tool llvm-readobj)"
    if [[ -n "$READOBJ" ]]; then
        "$READOBJ" --coff-exports "$WORKDIR/ci_dylib/build/ci_dylib.dll" >/tmp/sg_exports.txt 2>&1
        if grep -q "Name: ci_dylib_add" /tmp/sg_exports.txt && ! grep -q "Name: twice" /tmp/sg_exports.txt &&
           [[ "$(grep -c 'Name:' /tmp/sg_exports.txt)" -lt 5 ]]; then
            pass "ci_dylib.dll exports ci_dylib_add (and not std's or static functions)"
        else
            fail "ci_dylib.dll export table"
            cat /tmp/sg_exports.txt
        fi
    fi
fi
if sg_in "$WORKDIR/ci_dyapp" /tmp/sg_dyapp.txt run && grep -q "ci_dylib says 42" /tmp/sg_dyapp.txt &&
   { [[ "$OS" != windows ]] || [[ -f "$WORKDIR/ci_dyapp/build/ci_dylib.dll" ]]; }; then
    pass "a program using the cdylib builds and runs (safeguard run)"
else
    fail "a program using the cdylib"
    cat /tmp/sg_dyapp.txt
fi

# Its bundle carries the shared library: Contents/Frameworks (macOS), the
# app folder (Windows), /usr/lib/<pkg> on the rpath (Linux .deb). Run with
# the library's own build directory moved away, so only the bundled copy
# can satisfy it.
if sg_in "$WORKDIR/ci_dyapp" /tmp/sg_dybundle.txt bundle; then
    mv "$WORKDIR/ci_dylib/build" "$WORKDIR/ci_dylib/build.away"
    BOUT=""
    case "$OS" in
        macos)
            APP="$WORKDIR/ci_dyapp/build/ci_dyapp.app"
            [[ -f "$APP/Contents/Frameworks/libci_dylib.dylib" ]] && codesign --verify --strict "$APP" &&
                BOUT="$("$APP/Contents/MacOS/ci_dyapp" 2>&1)" ;;
        windows)
            [[ -f "$WORKDIR/ci_dyapp/build/ci_dyapp/ci_dylib.dll" ]] &&
                BOUT="$("$WORKDIR/ci_dyapp/build/ci_dyapp/ci_dyapp.exe" 2>&1)" ;;
        linux)
            DEB="$(ls "$WORKDIR"/ci_dyapp/build/ci-dyapp_0.1.0_*.deb 2>/dev/null | head -1)"
            if [[ -n "$DEB" ]] && dpkg-deb --contents "$DEB" | grep -q "usr/lib/ci-dyapp/libci_dylib.so"; then
                rm -rf "$WORKDIR/deb_x" && dpkg -x "$DEB" "$WORKDIR/deb_x" &&
                    BOUT="$("$WORKDIR/deb_x/usr/bin/ci_dyapp" 2>&1)"
            fi ;;
    esac
    mv "$WORKDIR/ci_dylib/build.away" "$WORKDIR/ci_dylib/build"
    if [[ "$BOUT" == *"ci_dylib says 42"* ]]; then
        pass "safeguard bundle: the bundled program runs on its bundled cdylib"
    else
        fail "safeguard bundle with a cdylib dependency (output: '$BOUT')"
        cat /tmp/sg_dybundle.txt
    fi
else
    fail "safeguard bundle (cdylib dependent)"
    cat /tmp/sg_dybundle.txt
fi

# A dependency that is a directory of C++ sources (build = "sources"),
# called through extern "C": compiled with its defines, linked with clang++.
mkdir -p "$WORKDIR/ci_vec" "$WORKDIR/ci_vec_app/src"
cat > "$WORKDIR/ci_vec/ci_vec.h" <<'EOF'
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
int ci_vec_max(const int* values, int n);
#ifdef __cplusplus
}
#endif
EOF
cat > "$WORKDIR/ci_vec/ci_vec.cpp" <<'EOF'
#include "ci_vec.h"
#include <algorithm>
#include <vector>
extern "C" int ci_vec_max(const int* values, int n) {
    std::vector<int> v(values, values + n);
    return *std::max_element(v.begin(), v.end()) + CI_VEC_BIAS;
}
EOF
cat > "$WORKDIR/ci_vec_app/Package.toml" <<'EOF'
[package]
name = "ci_vec_app"
version = "0.1.0"

[build]
edition = "2026"

[[dependencies]]
name    = "ci_vec"
path    = "../ci_vec"
build   = "sources"
defines = ["CI_VEC_BIAS=2"]
EOF
cat > "$WORKDIR/ci_vec_app/src/main.sc" <<'EOF'
#include <ci_vec.h>
extern int printf(const char* fmt, ...);
int main() {
    int v[4] = { 7, 40, 3, 9 };
    unsafe { printf("ci_vec max %d\n", ci_vec_max(&v[0], 4)); }
    return 0;
}
EOF
if sg_in "$WORKDIR/ci_vec_app" /tmp/sg_vec.txt run && grep -q "ci_vec max 42" /tmp/sg_vec.txt; then
    pass "a C++ source dependency (build = \"sources\") links into a SafeC program"
else
    fail "C++ source dependency"
    cat /tmp/sg_vec.txt
fi

# A CMake C++ library (examples/cpp_dep): configured, built and installed
# into build/deps/, cached, rebuilt when its sources change.
if command -v cmake >/dev/null 2>&1; then
    cp -R "$SAFEC_HOME/examples/cpp_dep" "$WORKDIR/cpp_dep"
    CMAKE_OK=1
    sg_in "$WORKDIR/cpp_dep/app" /tmp/sg_cmake.txt run && grep -q "median 4.0, distinct 3" /tmp/sg_cmake.txt || CMAKE_OK=0
    sg_in "$WORKDIR/cpp_dep/app" /tmp/sg_cmake2.txt build && ! grep -q "building dependency" /tmp/sg_cmake2.txt || CMAKE_OK=0
    sleep 1
    sed 's/\.size();/.size() + 100;/' "$WORKDIR/cpp_dep/cpplib/src/cppstats.cpp" > "$WORKDIR/cppstats.cpp" &&
        mv "$WORKDIR/cppstats.cpp" "$WORKDIR/cpp_dep/cpplib/src/cppstats.cpp"
    sg_in "$WORKDIR/cpp_dep/app" /tmp/sg_cmake3.txt run; grep -q "distinct 103" /tmp/sg_cmake3.txt || CMAKE_OK=0
    if [[ "$CMAKE_OK" -eq 1 ]]; then
        pass "a CMake C++ dependency (build = \"cmake\") builds, is cached, and rebuilds on change"
    else
        fail "CMake C++ dependency"
        cat /tmp/sg_cmake.txt /tmp/sg_cmake2.txt /tmp/sg_cmake3.txt
    fi
else
    echo "  [SKIP] cmake not available"
fi

# Objective-C: a package of its own and a dependency (examples/objc_dep).
if [[ "$OS" == macos ]]; then
    if (cd "$WORKDIR" && "$SAFEGUARD_BIN" new ci_objc --lang objc >/dev/null 2>&1) &&
       sg_in "$WORKDIR/ci_objc" /tmp/sg_objc.txt run && grep -q "Hello from ci_objc!" /tmp/sg_objc.txt; then
        pass "safeguard new --lang objc: an Objective-C package builds and runs"
    else
        fail "Objective-C package"
        cat /tmp/sg_objc.txt
    fi
    cp -R "$SAFEC_HOME/examples/objc_dep" "$WORKDIR/objc_dep"
    if sg_in "$WORKDIR/objc_dep/app" /tmp/sg_objcdep.txt run &&
       grep -q "length 5, uppercase H.*LLO (6 bytes)" /tmp/sg_objcdep.txt; then
        pass "an Objective-C dependency (Foundation, ARC) is callable from SafeC"
    else
        fail "Objective-C dependency"
        cat /tmp/sg_objcdep.txt
    fi
else
    echo "  [SKIP] Objective-C (macOS only)"
fi

# std/async (examples/async_demo): a Runtime serving HTTP to its own
# clients over loopback, a ThreadPool job awaited through a Oneshot -- the
# OS readiness backend (epoll / kqueue / WSAPoll) at run time, and the
# manifest's compat_preprocessor for the coroutine macros.
cp -R "$SAFEC_HOME/examples/async_demo" "$WORKDIR/async_demo"
if sg_in "$WORKDIR/async_demo" /tmp/sg_async.txt build; then
    ASYNC_EXE="$WORKDIR/async_demo/build/async_demo"
    [[ -f "$ASYNC_EXE.exe" ]] && ASYNC_EXE="$ASYNC_EXE.exe"
    if with_timeout 60 "$ASYNC_EXE" >/tmp/sg_async_run.txt 2>&1 &&
       grep -q "8/8 clients ok" /tmp/sg_async_run.txt; then
        pass "examples/async_demo: $(tail -1 /tmp/sg_async_run.txt)"
    else
        fail "examples/async_demo run"
        cat /tmp/sg_async_run.txt
    fi
else
    fail "examples/async_demo build"
    cat /tmp/sg_async.txt
fi

# A GUI app bundle (examples/gui_hello: a std/gui window that closes itself).
GUI_CAN_BUILD=1
[[ "$OS" == linux && ! -e /usr/include/X11/Xlib.h ]] && GUI_CAN_BUILD=0
if [[ "$OS" == other || "$GUI_CAN_BUILD" -eq 0 ]]; then
    echo "  [SKIP] GUI bundle (no X11 headers / unsupported OS)"
else
    cp -R "$SAFEC_HOME/examples/gui_hello" "$WORKDIR/gui_hello"
    G="$WORKDIR/gui_hello/build"
    # Its manifest names no system library: std/gui's backend brings them.
    if grep -qE '^(libs|frameworks|depends)' "$WORKDIR/gui_hello/Package.toml"; then
        fail "examples/gui_hello/Package.toml lists libraries by hand"
    fi
    if sg_in "$WORKDIR/gui_hello" /tmp/sg_gui.txt bundle; then
        GUI_OK=1
        case "$OS" in
            macos)
                C="$G/GuiHello.app/Contents"
                for f in MacOS/gui_hello Info.plist Resources/icon.icns Resources/greeting.txt; do
                    [[ -f "$C/$f" ]] || { GUI_OK=0; echo "    missing $f"; }
                done
                [[ "$(plutil -extract CFBundleIdentifier raw "$C/Info.plist")" == org.safec.gui-hello ]] || GUI_OK=0
                [[ "$(plutil -extract CFBundleExecutable raw "$C/Info.plist")" == gui_hello ]] || GUI_OK=0
                codesign --verify --strict "$G/GuiHello.app" || GUI_OK=0
                with_timeout 60 "$C/MacOS/gui_hello" >/tmp/sg_gui_run.txt 2>&1 || GUI_OK=0
                ;;
            windows)
                for f in gui_hello.exe greeting.txt; do
                    [[ -f "$G/GuiHello/$f" ]] || { GUI_OK=0; echo "    missing GuiHello/$f"; }
                done
                READOBJ="$(llvm_tool llvm-readobj)"
                if [[ -n "$READOBJ" ]]; then
                    "$READOBJ" --file-headers --coff-resources "$G/GuiHello/gui_hello.exe" >/tmp/sg_gui_res.txt 2>&1
                    for k in IMAGE_SUBSYSTEM_WINDOWS_GUI "Type: ICON" "Type: VERSIONINFO" "Type: MANIFEST"; do
                        grep -q "$k" /tmp/sg_gui_res.txt || { GUI_OK=0; echo "    gui_hello.exe lacks $k"; }
                    done
                fi
                with_timeout 60 "$G/GuiHello/gui_hello.exe" >/tmp/sg_gui_run.txt 2>&1 || GUI_OK=0
                ;;
            linux)
                DEB="$G/gui-hello_1.0.0_$(dpkg --print-architecture).deb"
                dpkg-deb --info "$DEB" >/tmp/sg_gui_deb.txt 2>&1 || GUI_OK=0
                dpkg-deb --contents "$DEB" >>/tmp/sg_gui_deb.txt 2>&1 || GUI_OK=0
                for k in "Package: gui-hello" "Depends: libx11-6" ./usr/bin/gui_hello \
                         ./usr/share/applications/org.safec.gui-hello.desktop \
                         ./usr/share/icons/hicolor/256x256/apps/org.safec.gui-hello.png \
                         ./usr/share/gui-hello/greeting.txt; do
                    grep -q -- "$k" /tmp/sg_gui_deb.txt || { GUI_OK=0; echo "    .deb lacks $k"; }
                done
                [[ -x "$G/GuiHello.AppDir/AppRun" ]] || { GUI_OK=0; echo "    no AppDir/AppRun"; }
                rm -rf "$WORKDIR/gui_x" && dpkg -x "$DEB" "$WORKDIR/gui_x" || GUI_OK=0
                if command -v xvfb-run >/dev/null 2>&1; then
                    with_timeout 60 xvfb-run -a "$WORKDIR/gui_x/usr/bin/gui_hello" >/tmp/sg_gui_run.txt 2>&1 || GUI_OK=0
                else
                    with_timeout 60 "$WORKDIR/gui_x/usr/bin/gui_hello" >/tmp/sg_gui_run.txt 2>&1 || GUI_OK=0
                fi
                ;;
        esac
        if [[ "$OS" != windows ]] && ! grep -q "gui_hello:" /tmp/sg_gui_run.txt; then GUI_OK=0; fi
        if [[ "$GUI_OK" -eq 1 ]]; then
            pass "safeguard bundle: GUI app bundle built, inspected and run ($(head -1 /tmp/sg_gui_run.txt))"
        else
            fail "GUI app bundle"
            cat /tmp/sg_gui.txt /tmp/sg_gui_run.txt 2>/dev/null
        fi
    else
        fail "safeguard bundle (GUI app)"
        cat /tmp/sg_gui.txt
        # Diagnostics: the link line, and what the std archive it links holds.
        sg_in "$WORKDIR/gui_hello" /tmp/sg_gui_v.txt build --verbose
        grep -iE 'llvm-ar|[^a-z]ar rcs|libsafec_std|-o .*gui_hello' /tmp/sg_gui_v.txt | tail -6 | cut -c1-600
        NM="$(llvm_tool llvm-nm)"; NM="${NM:-nm}"
        for a in $(find "$WORKDIR/gui_hello/build/deps" -name 'libsafec_std.a' 2>/dev/null); do
            echo "    $a: $(wc -c < "$a") bytes"
            "$NM" "$a" 2>&1 | grep -cE ' T ' | sed 's/^/    defined functions: /'
            "$NM" "$a" 2>&1 | grep -E 'GuiWindow_create|std_thread_sleep_ms' | head -4 | sed 's/^/    /'
            "$NM" "$a" 2>&1 | grep -iE 'error|warning' | head -3 | sed 's/^/    /'
        done
    fi
fi

popd >/dev/null
rm -rf "$WORKDIR"

# ── 5c. std module completeness ──────────────────────────────────────────────
# Every std header's functions are implemented by the std modules a build
# including it compiles (tests/std_modules/check.py: safec's declarations
# of each header against llvm-nm of the subset's objects) -- and by the
# prebuilt std sections such a build links: package.sh lays out an install
# as release.yml packages it, with lib/<triple>/std/<section>.a, and
# check.py --prebuilt checks those sections too.
section "std module completeness"
PRE="$(mktemp -d)"
PKG_OK=0
if bash "$SAFEC_HOME/.github/scripts/package.sh" "$PRE/inst" "$SAFEC_BIN" "$SAFEGUARD_BIN" >/tmp/sg_pkg.txt 2>&1 &&
   ls "$PRE"/inst/lib/*/std/core.a "$PRE"/inst/lib/*/std/core.units "$PRE"/inst/lib/*/std/collections.a >/dev/null 2>&1; then
    PKG_OK=1
    PRE_STD="$(dirname "$(ls "$PRE"/inst/lib/*/std/core.a)")"
    pass "package.sh lays out an install with lib/$(basename "$(dirname "$PRE_STD")")/std/: $(ls "$PRE_STD" | grep -c '\.units$') sections ($(ls "$PRE_STD" | sed -n 's/\.units$//p' | tr '\n' ' '))"
else
    fail "package.sh"
    cat /tmp/sg_pkg.txt
fi
if command -v python3 >/dev/null 2>&1; then
    PREARG=()
    [[ "$PKG_OK" -eq 1 ]] && PREARG=(--prebuilt "$PRE_STD")
    if SAFEC="$SAFEC_BIN" SAFEGUARD_BIN="$SAFEGUARD_BIN" CLANG_BIN="$CLANG_BIN" \
            python3 "$SAFEC_HOME/tests/std_modules/check.py" ${PREARG[@]+"${PREARG[@]}"} >/tmp/std_modules.txt 2>&1; then
        pass "$(head -1 /tmp/std_modules.txt)"
    else
        fail "std module completeness"
        cat /tmp/std_modules.txt
    fi
else
    echo "  [SKIP] python3 not available"
fi

# ── 5d. prebuilt std sections ────────────────────────────────────────────────
# A native build with the packaged safec links the prebuilt sections holding
# its units (core and collections for io.h + vec.h; not gui, ml, ...) and
# compiles no std. Without a section installed (install.sh --std=...), it
# compiles that section's units from source and links the rest; a section
# stamped for another compiler is ignored the same way. std is compiled
# optimized in a debug build too.
section "prebuilt std sections"
if [[ "$PKG_OK" -eq 1 ]]; then
    IBIN="$PRE/inst/bin/$(basename "$SAFEGUARD_BIN")"
    mkdir -p "$PRE/app/src"
    printf '[package]\nname = "pre_app"\nversion = "0.1.0"\n\n[build]\nedition = "2026"\n' > "$PRE/app/Package.toml"
    printf '#include <std/io.h>\n#include <std/collections/vec.h>\nint main() { std::println("prebuilt std ok"); return 0; }\n' \
        > "$PRE/app/src/main.sc"
    pre_build() { (cd "$PRE/app" && rm -rf build && env -u SAFEC -u SAFEC_STD_ARCHIVE SAFEC_HOME="$PRE/inst" "$IBIN" run --verbose); }
    if pre_build >/tmp/sg_pre.txt 2>&1 && grep -q "linking the prebuilt std sections collections core from" /tmp/sg_pre.txt &&
       grep -q "prebuilt std ok" /tmp/sg_pre.txt && [[ -z "$(ls "$PRE"/app/build/deps/std/*.o 2>/dev/null)" ]] &&
       ! grep -qE 'std/(gui|ml|net)\.a' /tmp/sg_pre.txt; then
        pass "a native build links only the prebuilt std sections it uses (core, collections), compiling no std"
    else
        fail "a native build with the prebuilt std sections"
        cat /tmp/sg_pre.txt
    fi
    mkdir -p "$PRE/aside" && mv "$PRE_STD"/collections.* "$PRE/aside/"
    if pre_build >/tmp/sg_pre1.txt 2>&1 &&
       grep -q "std section 'collections' is not installed in .*compiling its units from source: std/collections/vec.sc" /tmp/sg_pre1.txt &&
       grep -q "linking the prebuilt std sections core from" /tmp/sg_pre1.txt &&
       grep -q "prebuilt std ok" /tmp/sg_pre1.txt &&
       [[ "$(cd "$PRE/app/build/deps/std" && ls *.o 2>/dev/null)" == "collections_vec_sc.o" ]]; then
        pass "a section that isn't installed is compiled from source, the others linked prebuilt"
    else
        fail "a build with a std section missing"
        cat /tmp/sg_pre1.txt
    fi
    if grep -E 'clang(\.exe)? -c .*collections_vec_sc\.ll' /tmp/sg_pre1.txt | grep -q -- ' -O2'; then
        pass "std is compiled optimized (-O2) in a debug build"
    else
        fail "std's optimization flags in a debug build"
        grep -E 'collections_vec_sc' /tmp/sg_pre1.txt
    fi
    mv "$PRE/aside"/collections.* "$PRE_STD/"
    for u in "$PRE_STD"/*.units; do
        sed 's/^stamp safec=.*/stamp safec=0000000000000000/' "$u" > "$PRE/stamp" && cp "$PRE/stamp" "$u"
    done
    if pre_build >/tmp/sg_pre2.txt 2>&1 && grep -q "not using the prebuilt std sections .*different safec" /tmp/sg_pre2.txt &&
       grep -q "prebuilt std ok" /tmp/sg_pre2.txt && [[ -n "$(ls "$PRE"/app/build/deps/std/*.o 2>/dev/null)" ]]; then
        pass "prebuilt std sections stamped for another compiler are ignored (std compiled from source)"
    else
        fail "stale prebuilt std sections"
        cat /tmp/sg_pre2.txt
    fi
else
    echo "  [SKIP] no package (see above)"
fi
rm -rf "$PRE"

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
