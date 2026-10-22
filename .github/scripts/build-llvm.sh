#!/usr/bin/env bash
# Builds the LLVM libraries the release's safec links statically, from
# LLVM.org's source release, into <prefix> (lib/, include/, lib/cmake/llvm).
#
# The release builds LLVM themselves instead of taking Homebrew's or
# apt.llvm.org's, whose libraries are built for the runner's OS: Homebrew's
# for the runner's macOS (a safec linking them needs that macOS, whatever
# its own deployment target says), apt.llvm.org's against Ubuntu's current
# glibc (fstat@GLIBC_2.33, __isoc23_strtol@GLIBC_2.38, ...). Here they are
# compiled for MACOSX_DEPLOYMENT_TARGET (macOS, release.yml sets 11.0) or
# against the glibc of the build container (manylinux2014's 2.17, see
# build-linux-release.sh), and with no optional dependencies (zlib, zstd,
# libxml2, terminfo), so safec links nothing a user's machine may lack.
#
# Only the libraries are built, no tools or clang, for every target LLVM
# builds by default -- as Homebrew's and apt.llvm.org's have, since safec
# compiles for any LLVM target (NVPTX, AMDGPU and SPIR-V for std::ml too).
# $LLVM_TARGETS overrides the list, for a quicker local try.
# Nothing is done if <prefix> already has an LLVM (a cache hit).
#
#   usage: build-llvm.sh <version> <prefix>      e.g. 21.1.8 "$PWD/llvm-install"
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: $0 <version> <prefix>" >&2
    exit 2
fi
VER="$1"; PREFIX="$2"
if [[ -f "$PREFIX/lib/cmake/llvm/LLVMConfig.cmake" ]]; then
    echo "LLVM $VER is already in $PREFIX"
    exit 0
fi
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
SRC="llvm-project-${VER}.src"
echo "Fetching LLVM $VER sources..."
curl -fsSL --retry 3 -o "$WORK/src.tar.xz" \
    "https://github.com/llvm/llvm-project/releases/download/llvmorg-${VER}/${SRC}.tar.xz"
# llvm/ and what its build reads from beside it.
tar -xJf "$WORK/src.tar.xz" -C "$WORK" --exclude="$SRC/llvm/test" --exclude="$SRC/llvm/unittests" \
    --exclude="$SRC/llvm/docs" "$SRC/llvm" "$SRC/cmake" "$SRC/third-party"
rm -f "$WORK/src.tar.xz"

GEN=()
if command -v ninja >/dev/null 2>&1; then GEN=(-G Ninja); fi
EXTRA=()
if [[ "$(uname -s)" == Darwin ]]; then
    EXTRA+=(-DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}")
    EXTRA+=(-DCMAKE_OSX_ARCHITECTURES="$(uname -m)")
fi
cmake ${GEN[@]+"${GEN[@]}"} -S "$WORK/$SRC/llvm" -B "$WORK/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DLLVM_TARGETS_TO_BUILD="${LLVM_TARGETS:-all}" \
    -DLLVM_ENABLE_PROJECTS="" \
    -DLLVM_BUILD_TOOLS=OFF -DLLVM_BUILD_UTILS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_DOCS=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_LIBXML2=OFF \
    -DLLVM_ENABLE_LIBEDIT=OFF -DLLVM_ENABLE_LIBPFM=OFF -DLLVM_ENABLE_CURL=OFF \
    -DLLVM_ENABLE_HTTPLIB=OFF \
    -DLLVM_BUILD_LLVM_DYLIB=OFF -DLLVM_LINK_LLVM_DYLIB=OFF \
    ${EXTRA[@]+"${EXTRA[@]}"}
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
cmake --build "$WORK/build" --target install -j "$JOBS"
echo "LLVM $VER installed in $PREFIX"
