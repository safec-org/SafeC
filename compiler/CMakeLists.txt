cmake_minimum_required(VERSION 3.20)
project(SafeC VERSION 0.1.0 LANGUAGES CXX)

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
  src/Lexer.cpp
  src/AST.cpp
  src/Type.cpp
  src/Parser.cpp
  src/Sema.cpp
  src/ConstEval.cpp
  src/CodeGen.cpp
  src/main.cpp
)

add_executable(safec ${SAFEC_SOURCES})

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
  llvm_map_components_to_libnames(LLVM_COMPONENT_LIBS
    core support irreader bitwriter passes
    target nativecodegen nativedisassembler
    mc mcparser asmprinter
  )
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
