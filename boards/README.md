# SafeC Board Toolchain Files

CMake toolchain files for cross-compiling SafeC programs to bare-metal targets.

## Available Targets

| File | Target | Toolchain prefix |
|------|--------|-----------------|
| `cortex-m4.cmake` | ARM Cortex-M4 (hard-float) | `arm-none-eabi-` |
| `riscv32.cmake` | RISC-V 32-bit RV32IMAC | `riscv32-unknown-elf-` |
| `riscv64.cmake` | RISC-V 64-bit RV64GC (Sv39) | `riscv64-unknown-elf-` |
| `x86_64-bare.cmake` | x86-64 bare-metal kernel | `x86_64-elf-` |

## Usage

### 1. Install a cross-compilation toolchain

**macOS (Homebrew):**
```sh
brew install --cask gcc-arm-embedded          # Cortex-M4
brew install riscv-gnu-toolchain              # RISC-V 32 + 64
brew install x86_64-elf-gcc                  # x86-64 bare-metal
```

**Ubuntu/Debian:**
```sh
sudo apt install gcc-arm-none-eabi            # Cortex-M4
sudo apt install gcc-riscv64-unknown-elf      # RISC-V 64
```

### 2. Compile your SafeC source to LLVM IR

```sh
# With freestanding mode enabled
safec main.sc \
  --freestanding \
  -D__SAFEC_FREESTANDING__ \
  -I <safec-root>/std \
  --emit-llvm -o main.ll
```

### 3. Lower LLVM IR to object file with the cross-compiler

```sh
# Cortex-M4 example
llc -mtriple=thumbv7em-none-eabi -mcpu=cortex-m4 \
    -mattr=+vfp4,+d16 -float-abi=hard \
    main.ll -filetype=obj -o main.o

# RISC-V 64 example
llc -mtriple=riscv64-unknown-elf -mcpu=generic-rv64 \
    -mattr=+m,+a,+f,+d,+c \
    main.ll -filetype=obj -o main.o
```

### 4. Link with CMake using the toolchain file

```sh
cmake -S . -B build \
      -DCMAKE_TOOLCHAIN_FILE=<safec-root>/boards/cortex-m4.cmake \
      -DCMAKE_BUILD_TYPE=MinSizeRel
cmake --build build
```

### 5. Flash (Cortex-M4 example)

```sh
# Generate .bin from ELF (the safec_bin_target() helper does this automatically)
arm-none-eabi-objcopy -O binary build/firmware.elf build/firmware.bin

# Flash with OpenOCD
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg \
        -c "program build/firmware.bin verify reset exit 0x08000000"
```

## Linker Script

Each board target expects a linker script (`linker.ld`) that describes the memory
layout of your specific device. Pass it via:

```cmake
target_link_options(my_firmware PRIVATE -T ${CMAKE_SOURCE_DIR}/linker.ld)
```

A minimal Cortex-M4 linker script template:

```ld
MEMORY {
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 512K
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}
SECTIONS {
  .text   : { *(.text*) }   > FLASH
  .rodata : { *(.rodata*) } > FLASH
  .data   : { *(.data*) }   > RAM AT > FLASH
  .bss    : { *(.bss*) }    > RAM
  .heap   : { . = ALIGN(8); _heap_start = .; . += 0x10000; } > RAM
}
```

## SafeC Freestanding Defines

| Define | Meaning |
|--------|---------|
| `__SAFEC_FREESTANDING__` | Enable freestanding standard library path (`prelude.h` gates HAL/kernel headers) |
| `__ARM_ARCH_7EM__` | Target is Cortex-M4/M7 (Thumb-2 + DSP) |
| `__riscv` | Target is any RISC-V |
| `__riscv_xlen=32` | RISC-V 32-bit |
| `__riscv_xlen=64` | RISC-V 64-bit |
| `__x86_64__` | Target is x86-64 |
| `HEAP_SIZE=N` | Override freestanding heap buffer size (default 1 MiB) |
