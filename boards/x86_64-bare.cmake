# SafeC Board Toolchain — x86-64 bare-metal (bootloader / kernel development)
# Targets the x86-64 System V ABI without any hosted runtime.
# Intended for use with a custom linker script (e.g., linker.ld).
# Usage:
#   cmake -S . -B build \
#         -DCMAKE_TOOLCHAIN_FILE=<safec-root>/boards/x86_64-bare.cmake \
#         -DSAFEC_ROOT=<safec-root>

set(CMAKE_SYSTEM_NAME      Generic)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# Use the host x86-64 cross-compiler (or a dedicated bare-metal one).
# On macOS with Homebrew: brew install x86_64-elf-gcc
find_program(X86_GCC     x86_64-elf-gcc     NAMES gcc   REQUIRED)
find_program(X86_GXX     x86_64-elf-g++     NAMES g++   REQUIRED)
find_program(X86_OBJCOPY x86_64-elf-objcopy NAMES objcopy REQUIRED)

set(CMAKE_C_COMPILER   ${X86_GCC})
set(CMAKE_CXX_COMPILER ${X86_GXX})
set(CMAKE_ASM_COMPILER ${X86_GCC})
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# ── Compile flags ─────────────────────────────────────────────────────────────
set(CPU_FLAGS -m64 -mno-red-zone -mno-mmx -mno-sse -mno-sse2)

set(BARE_FLAGS
    -ffreestanding
    -nostdlib
    -fno-stack-protector
    -fno-pic
    -fno-exceptions
    -fno-rtti
    -ffunction-sections
    -fdata-sections
    -mcmodel=kernel
)

string(JOIN " " CPU_FLAGS_STR  ${CPU_FLAGS})
string(JOIN " " BARE_FLAGS_STR ${BARE_FLAGS})

set(CMAKE_C_FLAGS_INIT   "${CPU_FLAGS_STR} ${BARE_FLAGS_STR}")
set(CMAKE_CXX_FLAGS_INIT "${CPU_FLAGS_STR} ${BARE_FLAGS_STR}")
set(CMAKE_ASM_FLAGS_INIT "${CPU_FLAGS_STR}")

# Kernel code model requires a linker script; supply via target_link_options.
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "${CPU_FLAGS_STR} -nostdlib -z max-page-size=0x1000")

# ── SafeC compiler flags ──────────────────────────────────────────────────────
set(SAFEC_FLAGS
    --freestanding
    -D__SAFEC_FREESTANDING__
    -D__x86_64__
    -D__SAFEC_MMU_X86_64__
)

# ── Helper: create a flat binary from ELF ────────────────────────────────────
function(safec_flat_binary TARGET)
    add_custom_command(TARGET ${TARGET} POST_BUILD
        COMMAND ${X86_OBJCOPY} -O binary $<TARGET_FILE:${TARGET}>
                               $<TARGET_FILE_DIR:${TARGET}>/${TARGET}.bin
        COMMENT "Generating flat binary ${TARGET}.bin"
    )
endfunction()
