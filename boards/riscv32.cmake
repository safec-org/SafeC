# SafeC Board Toolchain — RISC-V 32-bit (RV32IMAC, bare-metal)
# Usage:
#   cmake -S . -B build \
#         -DCMAKE_TOOLCHAIN_FILE=<safec-root>/boards/riscv32.cmake \
#         -DSAFEC_ROOT=<safec-root>

set(CMAKE_SYSTEM_NAME      Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv)

# ── Toolchain binaries ────────────────────────────────────────────────────────
find_program(RISCV_GCC     riscv32-unknown-elf-gcc     REQUIRED)
find_program(RISCV_GXX     riscv32-unknown-elf-g++     REQUIRED)
find_program(RISCV_OBJCOPY riscv32-unknown-elf-objcopy REQUIRED)
find_program(RISCV_SIZE    riscv32-unknown-elf-size     REQUIRED)

set(CMAKE_C_COMPILER   ${RISCV_GCC})
set(CMAKE_CXX_COMPILER ${RISCV_GXX})
set(CMAKE_ASM_COMPILER ${RISCV_GCC})
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# ── CPU / ABI flags ───────────────────────────────────────────────────────────
set(CPU_FLAGS -march=rv32imac -mabi=ilp32)

# ── Freestanding compile flags ────────────────────────────────────────────────
set(BARE_FLAGS
    -ffreestanding
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
    "${CPU_FLAGS_STR} -nostdlib -Wl,--gc-sections")

# ── SafeC compiler flags ──────────────────────────────────────────────────────
set(SAFEC_FLAGS --freestanding -D__SAFEC_FREESTANDING__ -D__riscv -D__riscv_xlen=32)

# ── Helper: convert ELF → binary / hex ───────────────────────────────────────
function(safec_bin_target TARGET)
    add_custom_command(TARGET ${TARGET} POST_BUILD
        COMMAND ${RISCV_OBJCOPY} -O binary $<TARGET_FILE:${TARGET}>
                                 $<TARGET_FILE_DIR:${TARGET}>/${TARGET}.bin
        COMMAND ${RISCV_OBJCOPY} -O ihex   $<TARGET_FILE:${TARGET}>
                                 $<TARGET_FILE_DIR:${TARGET}>/${TARGET}.hex
        COMMAND ${RISCV_SIZE} $<TARGET_FILE:${TARGET}>
        COMMENT "Generating ${TARGET}.bin and ${TARGET}.hex"
    )
endfunction()
