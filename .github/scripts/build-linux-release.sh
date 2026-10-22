#!/usr/bin/env bash
# Builds the Linux release binaries -- safec, safeguard and, when its
# checkout is there, sc-lsp -- so that they run on glibc 2.25 and newer.
#
# Runs inside quay.io/pypa/manylinux2014_x86_64 (CentOS 7: glibc 2.17, and
# devtoolset's GCC, whose libstdc++ is linked statically here), started by
# release.yml with the workspace mounted at <workspace>:
#
#   <workspace>/SafeC                   this repository
#   <workspace>/sc-language-server      (optional)
#   <workspace>/llvm-install            LLVM's libraries, built here by
#                                       build-llvm.sh unless cached
#
# A binary linked against glibc 2.17 imports no symbol version newer than
# that, so it runs on any later glibc; check-glibc.py verifies it (and the
# prebuilt std objects) after the build. Compiling in an old container is
# what keeps the versions old: linking a new glibc with flags doesn't.
#
#   usage: build-linux-release.sh <workspace> <llvm version>
set -euo pipefail
W="$1"; LLVM_VER="$2"
export HOME="${HOME:-/tmp}"
[[ -w "$HOME" ]] || export HOME=/tmp

# CMake and Ninja from PyPI (manylinux wheels): the image's own are older
# than LLVM 21 needs, or absent.
PY="$(ls -d /opt/python/cp312-*/bin/python 2>/dev/null | head -1)"
[[ -n "$PY" ]] || PY=python3
"$PY" -m venv /tmp/buildtools
/tmp/buildtools/bin/pip install --quiet --disable-pip-version-check "cmake>=3.20,<4" ninja
export PATH="/tmp/buildtools/bin:$PATH"
echo "$(gcc --version | head -1); $(cmake --version | head -1); glibc $(ldd --version | head -1 | awk '{print $NF}')"

LLVM_PREFIX="$W/llvm-install"
bash "$W/SafeC/.github/scripts/build-llvm.sh" "$LLVM_VER" "$LLVM_PREFIX"

# libstdc++ and libgcc linked in, so the binaries need no particular
# libstdc++.so.6 (GLIBCXX_*) either -- if this GCC has a static libstdc++.
# devtoolset's otherwise links the parts CentOS 7's libstdc++.so.6 lacks
# statically, which needs GLIBCXX_3.4.19 at most: still fine.
LDFLAGS_STATIC="-static-libstdc++ -static-libgcc"
if ! echo 'int main() { return 0; }' | g++ -x c++ - $LDFLAGS_STATIC -o /tmp/static-probe 2>/dev/null; then
    echo "note: no static libstdc++ here; linking devtoolset's libstdc++"
    LDFLAGS_STATIC=""
fi
cmake -G Ninja -S "$W/SafeC/compiler" -B "$W/SafeC/compiler/build" \
    -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm" -DSAFEC_STATIC_LLVM=ON \
    -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS_STATIC"
cmake --build "$W/SafeC/compiler/build"

cmake -G Ninja -S "$W/SafeC/safeguard" -B "$W/SafeC/safeguard/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS_STATIC"
cmake --build "$W/SafeC/safeguard/build"

if [[ -d "$W/sc-language-server" ]]; then
    cmake -G Ninja -S "$W/sc-language-server" -B "$W/sc-language-server/build" \
        -DCMAKE_BUILD_TYPE=Release -DSAFEC_DIR="$W/SafeC/compiler" \
        -DLLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm" -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS_STATIC"
    cmake --build "$W/sc-language-server/build"
fi

for b in "$W/SafeC/compiler/build/safec" "$W/SafeC/safeguard/build/safeguard" "$W/sc-language-server/build/sc-lsp"; do
    if [[ -f "$b" ]]; then echo "== $b"; ldd "$b" || true; fi
done
