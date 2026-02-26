cmake_minimum_required(VERSION 3.16)
project(safeguard VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# All source files
set(SOURCES
    src/main.cpp
    src/Manifest.cpp
    src/Builder.cpp
    src/Project.cpp
    src/Lock.cpp
    src/Analyzer.cpp
)

add_executable(safeguard ${SOURCES})

target_include_directories(safeguard PRIVATE include)

# std::filesystem needs explicit linking on some toolchains
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_link_libraries(safeguard stdc++fs)
endif()

# Warn aggressively
target_compile_options(safeguard PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall -Wextra -Wshadow -Wpedantic>
)

install(TARGETS safeguard RUNTIME DESTINATION bin)
