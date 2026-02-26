# SafeC Board Toolchain — ARM Cortex-M4 (bare-metal, hard-float ABI)
# Usage:
#   cmake -S . -B build \
#         -DCMAKE_TOOLCHAIN_FILE=<safec-root>/boards/cortex-m4.cmake \
#         -DSAFEC_ROOT=<safec-root>

set(CMAKE_SYSTEM_NAME      Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

# ── Toolchain binaries ────────────────────────────────────────────────────────
find_program(ARM_GCC     arm-none-eabi-gcc     REQUIRED)
find_program(ARM_GXX     arm-none-eabi-g++     REQUIRED)
find_program(ARM_OBJCOPY arm-none-eabi-objcopy REQUIRED)
find_program(ARM_SIZE    arm-none-eabi-size     REQUIRED)

set(CMAKE_C_COMPILER   ${ARM_GCC})
set(CMAKE_CXX_COMPILER ${ARM_GXX})
set(CMAKE_ASM_COMPILER ${ARM_GCC})

# Prevent CMake from testing the compiler with a full hosted program.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# ── CPU / FPU flags ───────────────────────────────────────────────────────────
set(CPU_FLAGS
    -mcpu=cortex-m4
    -mthumb
    -mfpu=fpv4-sp-d16
    -mfloat-abi=hard
)

# ── Freestanding compile flags ────────────────────────────────────────────────
set(BARE_FLAGS
    -ffreestanding
    -fno-hosted
    -nostdlib
    -fno-exceptions
    -fno-rtti
    -ffunction-sections
    -fdata-sections
)

string(JOIN " " CPU_FLAGS_STR  ${CPU_FLAGS})
string(JOIN " " BARE_FLAGS_STR ${BARE_FLAGS})

set(CMAKE_C_FLAGS_INIT   "${CPU_FLAGS_STR} ${BARE_FLAGS_STR}")
set(CMAKE_CXX_FLAGS_INIT "${CPU_FLAGS_STR} ${BARE_FLAGS_STR}")
set(CMAKE_ASM_FLAGS_INIT "${CPU_FLAGS_STR}")

set(CMAKE_EXE_LINKER_FLAGS_INIT
    "${CPU_FLAGS_STR} -nostdlib -Wl,--gc-sections -Wl,--print-memory-usage")

# ── SafeC compiler flags ──────────────────────────────────────────────────────
# Pass to safec via add_custom_command:
#   safec foo.sc --freestanding -D__SAFEC_FREESTANDING__ --emit-llvm -o foo.ll
set(SAFEC_FLAGS --freestanding -D__SAFEC_FREESTANDING__ -D__ARM_ARCH_7EM__)

# ── Helper: convert ELF → binary / hex ───────────────────────────────────────
function(safec_bin_target TARGET)
    add_custom_command(TARGET ${TARGET} POST_BUILD
        COMMAND ${ARM_OBJCOPY} -O binary $<TARGET_FILE:${TARGET}>
                               $<TARGET_FILE_DIR:${TARGET}>/${TARGET}.bin
        COMMAND ${ARM_OBJCOPY} -O ihex   $<TARGET_FILE:${TARGET}>
                               $<TARGET_FILE_DIR:${TARGET}>/${TARGET}.hex
        COMMAND ${ARM_SIZE} $<TARGET_FILE:${TARGET}>
        COMMENT "Generating ${TARGET}.bin and ${TARGET}.hex"
    )
endfunction()
