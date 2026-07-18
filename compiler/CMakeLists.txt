cmake_minimum_required(VERSION 3.20)
project(SafeC VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── LLVM ──────────────────────────────────────────────────────────────────────
# Locate LLVM via its CMake config.  Set LLVM_DIR if LLVM is not on the
# default CMake search path.
#
#   macOS  (Homebrew):   cmake .. -DLLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm
#   Ubuntu / Debian:     cmake .. -DLLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm
#   Fedora / RHEL:       cmake .. -DLLVM_DIR=/usr/lib64/llvm17/lib/cmake/llvm
#   Windows (vcpkg):     cmake .. -DLLVM_DIR=<vcpkg_root>/installed/x64-windows/share/llvm
#   Windows (installer): cmake .. -DLLVM_DIR="C:/Program Files/LLVM/lib/cmake/llvm"
#
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# ── SafeC compiler ────────────────────────────────────────────────────────────
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SAFEC_SOURCES
  src/Preprocessor.cpp
  src/CHeaderImporter.cpp
  src/Lexer.cpp
  src/AST.cpp
  src/Type.cpp
  src/Parser.cpp
  src/Sema.cpp
  src/Clone.cpp
  src/ConstEval.cpp
  src/CodeGen.cpp
  src/main.cpp
)

add_executable(safec ${SAFEC_SOURCES})
target_compile_definitions(safec PRIVATE SAFEC_VERSION="${PROJECT_VERSION}")

# ── Compile options ───────────────────────────────────────────────────────────
if(MSVC)
  # MSVC equivalents for -fno-exceptions / -fno-rtti required by LLVM headers
  target_compile_options(safec PRIVATE /W3 /EHs-c- /GR-)
else()
  target_compile_options(safec PRIVATE
    -Wall -Wextra
    -Wno-unused-parameter
    -Wno-unused-variable
    # LLVM headers require these
    -fno-exceptions -fno-rtti
  )
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC-only flag
    target_compile_options(safec PRIVATE -Wno-unused-but-set-variable)
  endif()
endif()

# ── LLVM linking ──────────────────────────────────────────────────────────────
# Prefer the monolithic shared library (libLLVM.so / LLVM.dylib) when
# available — it avoids linking hundreds of individual static archives.
# Fall back to individual component libraries otherwise.
if(LLVM_LINK_LLVM_DYLIB)
  target_link_libraries(safec PRIVATE LLVM)
else()
  # safec is a multi-target cross-compiler, not host-only: main.cpp calls
  # llvm::InitializeAllTargets()/InitializeAllTargetMCs()/
  # InitializeAllAsmPrinters()/InitializeAllAsmParsers() so a single binary
  # can codegen for any ISA this LLVM was built with (--target flag;
  # std::simd needs genuinely different per-ISA vector codegen, not just
  # the host's). That requires every target's libraries to be linked, not
  # just the host's — plain 'nativecodegen'/'nativedisassembler'
  # (llvm_map_components_to_libnames' aliases for "the host arch's
  # CodeGen/Disassembler library", resolved via LLVM_NATIVE_ARCH) were
  # already too narrow for that even before LLVM_NATIVE_ARCH turned out to
  # come back empty from LLVM.org's official prebuilt Windows archive (that
  # emptiness turned the alias into a literal, nonexistent
  # 'LLVMnativedisassembler.lib' instead of even the single host library).
  # There's no single "all targets" pseudo-component either — 'all-targets'
  # was tried and hits that exact same empty-substitution failure mode,
  # producing a bogus 'LLVMall-targets'.
  #
  # What actually works, verified locally: LLVM_TARGETS_TO_BUILD is a
  # plain list CMake variable the LLVM package config exports (e.g.
  # "AArch64;ARM;X86;...", every target that LLVM build has), and passing
  # its entries as bare component names expands each into its full
  # CodeGen/AsmParser/Desc/Disassembler/Info set — exactly what
  # InitializeAllTargets() et al. need.
  #
  # This is only reached on Windows today (LLVM_LINK_LLVM_DYLIB is true on
  # macOS/Linux, where the monolithic library already contains every target
  # regardless of what's requested here — this branch was never actually
  # exercised on those platforms before).
  # Diagnostic: the immediately preceding fix (this same ${LLVM_TARGETS_TO_BUILD}
  # expansion) produced byte-for-byte the same unresolved-symbol list as
  # before it was added, on a real Windows CI run — meaning either this
  # variable comes back narrower than expected from this specific LLVM
  # package's exported config (same failure family as LLVM_NATIVE_ARCH
  # earlier), or something else is dropping these libraries before the
  # final link line. Printing both the input and the resolved output here
  # so the next CI run's CMake-configure log settles which, instead of
  # guessing a third time.
  message(STATUS "safec: LLVM_TARGETS_TO_BUILD = '${LLVM_TARGETS_TO_BUILD}'")
  llvm_map_components_to_libnames(LLVM_COMPONENT_LIBS
    core support irreader bitwriter bitreader passes
    target mc mcparser asmprinter
    ${LLVM_TARGETS_TO_BUILD}
  )
  message(STATUS "safec: LLVM_COMPONENT_LIBS = '${LLVM_COMPONENT_LIBS}'")
  target_link_libraries(safec PRIVATE ${LLVM_COMPONENT_LIBS})
endif()

# ── Platform-specific system libraries ───────────────────────────────────────
if(APPLE)
  target_link_libraries(safec PRIVATE
    "-framework CoreFoundation"
    z
    curses
  )
elseif(UNIX)
  # zlib is needed by LLVM on Linux; tinfo (ncurses) is needed by LLVM's
  # terminal support.  Both are optional if the LLVM build already links them.
  find_library(ZLIB_LIB z)
  find_library(TINFO_LIB NAMES tinfo ncurses)
  if(ZLIB_LIB)
    target_link_libraries(safec PRIVATE ${ZLIB_LIB})
  endif()
  if(TINFO_LIB)
    target_link_libraries(safec PRIVATE ${TINFO_LIB})
  endif()
endif()
