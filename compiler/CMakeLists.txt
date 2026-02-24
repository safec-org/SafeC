cmake_minimum_required(VERSION 3.20)
project(SafeC VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── LLVM ─────────────────────────────────────────────────────────────────────
# Detect LLVM installation. Prefer user-supplied LLVM_DIR.
# On this system: Homebrew x86_64 LLVM at /usr/local/Cellar/llvm/21.1.8_1
# built as x86_64; compile safec as x86_64 (runs via Rosetta on Apple Silicon).
if(NOT DEFINED LLVM_DIR AND EXISTS "/usr/local/Cellar/llvm/21.1.8_1")
  set(LLVM_DIR "/usr/local/Cellar/llvm/21.1.8_1/lib/cmake/llvm")
endif()

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# ── SafeC compiler ────────────────────────────────────────────────────────────
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SAFEC_SOURCES
  src/Lexer.cpp
  src/AST.cpp
  src/Type.cpp
  src/Parser.cpp
  src/Sema.cpp
  src/CodeGen.cpp
  src/main.cpp
)

add_executable(safec ${SAFEC_SOURCES})

# Warnings (suppress some for generated/prototype code)
target_compile_options(safec PRIVATE
  -Wall -Wextra
  -Wno-unused-parameter
  -Wno-unused-but-set-variable
  -Wno-unused-variable
)

# LLVM headers require -fno-exceptions and -fno-rtti
# Use the system libc++ (Apple clang's default) instead of LLVM's own
target_compile_options(safec PRIVATE -fno-exceptions -fno-rtti)

# Link against the LLVM monolithic shared library.
# The dylib bundles all LLVM components and avoids needing to link
# 200+ individual static archives.
set(LLVM_CELLAR "/usr/local/Cellar/llvm/21.1.8_1")
target_link_libraries(safec PRIVATE
  "${LLVM_CELLAR}/lib/libLLVM-21.dylib"
)

# Ensure the dylib is found at runtime
target_link_options(safec PRIVATE
  "-Wl,-rpath,${LLVM_CELLAR}/lib"
)

if(APPLE)
  target_link_libraries(safec PRIVATE "-framework CoreFoundation" "-lz" "-lcurses")
endif()

# ── Build helper script ───────────────────────────────────────────────────────
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/build.sh.in"
  "${CMAKE_CURRENT_BINARY_DIR}/build.sh"
  @ONLY
)
