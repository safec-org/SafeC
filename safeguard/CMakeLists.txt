cmake_minimum_required(VERSION 3.16)
# The release version is ../VERSION (e.g. 1.0.0-beta-7), as for safec;
# 'safeguard --version' prints it. project() takes only its numeric part.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION" SAFEGUARD_RELEASE_VERSION LIMIT_COUNT 1)
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" SAFEGUARD_NUMERIC_VERSION "${SAFEGUARD_RELEASE_VERSION}")
if(NOT SAFEGUARD_NUMERIC_VERSION)
  message(FATAL_ERROR "../VERSION must start with MAJOR.MINOR.PATCH, has '${SAFEGUARD_RELEASE_VERSION}'")
endif()
project(safeguard VERSION ${SAFEGUARD_NUMERIC_VERSION} LANGUAGES CXX)
# Re-run CMake when VERSION changes (a version bump rebuilds with it).
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION")

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
    src/Formatter.cpp
    src/ScxTranspiler.cpp
    src/StdModules.cpp
    src/Doc.cpp
    src/Publish.cpp
)

add_executable(safeguard ${SOURCES})

target_include_directories(safeguard PRIVATE include)
target_compile_definitions(safeguard PRIVATE SAFEGUARD_VERSION="${SAFEGUARD_RELEASE_VERSION}")

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
