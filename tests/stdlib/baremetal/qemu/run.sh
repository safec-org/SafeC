#!/usr/bin/env bash
# Runs bare-metal std code on QEMU's emulated hardware: the native MMU format
# with translation really on, and the HAL chip layouts against QEMU's models
# of those chips. Each program prints over its UART and ends with
# "QEMU-TEST PASS" or "QEMU-TEST FAIL".
#   usage: run.sh [config...]   (default: every config below)
# Env: SAFEC, CLANG, LLD, AR, BUILTINS (dir from ../build_builtins.sh,
#      default ~/opt/bm-builtins), OUT (build dir), QEMU_TIMEOUT (seconds).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
SAFEC="${SAFEC:-safec}"; CLANG="${CLANG:-clang}"; LLD="${LLD:-ld.lld}"; AR="${AR:-llvm-ar}"
BUILTINS="${BUILTINS:-$HOME/opt/bm-builtins}"
OUT="${OUT:-$HERE/build}"
TIMEOUT="${QEMU_TIMEOUT:-20}"

# config: name | target alias | load address | RAM address (empty: right after the
# image) | test.sc | start.S | extra clang flags | safec -D flags for bm_rt.sc | qemu command
configs=(
  "riscv64|rv64i|0x80000000||qemu_riscv.sc|start_riscv.S|-mcmodel=medany||qemu-system-riscv64 -M virt -bios none -nographic -m 256M"
  "riscv32|rv32i|0x80000000||qemu_riscv.sc|start_riscv.S|||qemu-system-riscv32 -M virt -bios none -nographic -m 256M"
  "aarch64|aarch64|0x40000000||qemu_arm.sc|start_aarch64.S|||qemu-system-aarch64 -M virt -cpu cortex-a53 -nographic -m 256M -semihosting"
  "armv7|aarch32|0x40000000||qemu_arm.sc|start_arm.S|||qemu-system-arm -M virt -cpu cortex-a15 -nographic -m 256M -semihosting"
  "stm32f4|arm-7em|0x08000000|0x20000000|qemu_stm32.sc|start_cortexm.S||-DBM_HEAP_SIZE=16384|qemu-system-arm -M netduinoplus2 -nographic -semihosting"
  "x86|x86|0x00100000||qemu_x86.sc|start_x86.S|||qemu-system-x86_64 -nographic -m 64M -device isa-debug-exit,iobase=0xf4,iosize=0x01"
)
if [[ $# -gt 0 ]]; then
  sel=(); for c in "${configs[@]}"; do for a in "$@"; do [[ ${c%%|*} == "$a" ]] && sel+=("$c"); done; done
  configs=("${sel[@]}")
fi

build_std() {   # <alias> <extra-flags> -> $OUT/<alias>/libsafec_std.a
  local t=$1 extra=$2 d="$OUT/$1"
  [[ -f "$d/libsafec_std.a" ]] && return 0
  mkdir -p "$d/std"
  while IFS= read -r -d '' f; do
    case "$(basename "$f" .sc)" in *_linux|*_bsd|*_win32|*_posix|*_epoll|*_kqueue|*_cocoa|*_x11) continue ;; esac
    local o="$d/std/$(echo "${f#$ROOT/std/}" | tr '/.' '__').o"
    "$SAFEC" -I "$ROOT/std" -I "$ROOT" "$f" --target="$t" --emit-llvm --compat-preprocessor -o "${o%.o}.ll" 2>/dev/null &&
      "$CLANG" -c "${o%.o}.ll" -o "$o" "${CFLAGS[@]}" $extra -Wno-override-module 2>/dev/null || rm -f "$o"
  done < <(find "$ROOT/std" -name '*.sc' -not -path '*/wasm/*' -not -path '*/ml/*' -print0 | sort -z)
  "$AR" rcs "$d/libsafec_std.a" "$d"/std/*.o
}

fails=0
for c in "${configs[@]}"; do
  IFS='|' read -r name t base ram test start extra rtdefs qemu <<< "$c"
  mapfile -t CFLAGS < <("$SAFEC" --print-target-clang-flags --target="$t")
  build_std "$t" "$extra"
  d="$OUT/$name"; mkdir -p "$d"
  cat > "$d/link.ld" <<LD
ENTRY(_start)
SECTIONS {
  . = $base;
  .text   : { KEEP(*(.vectors)) KEEP(*(.text.start)) *(.text .text.*) }
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) }
  ${ram:+. = $ram;}
  .data   : { *(.data .data.* .sdata .sdata.*) }
  .bss    : { *(.bss .bss.* .sbss .sbss.* COMMON) }
  . = ALIGN(16);
  . = . + 0x10000;
  __stack_top = .;
  /DISCARD/ : { *(.ARM.exidx*) *(.eh_frame*) *(.comment) }
}
LD
  if ! "$SAFEC" -I "$ROOT/std" -I "$ROOT" "$HERE/$test" --target="$t" --emit-llvm --compat-preprocessor -o "$d/test.ll" 2>"$d/err" ||
     ! "$CLANG" -c "$d/test.ll" -o "$d/test.o" "${CFLAGS[@]}" $extra -Wno-override-module 2>>"$d/err" ||
     ! "$SAFEC" -I "$ROOT/std" -I "$ROOT" "$HERE/../bm_rt.sc" --target="$t" $rtdefs --emit-llvm --compat-preprocessor -o "$d/rt.ll" 2>>"$d/err" ||
     ! "$CLANG" -c "$d/rt.ll" -o "$d/rt.o" "${CFLAGS[@]}" $extra -Wno-override-module 2>>"$d/err" ||
     ! "$CLANG" -c "$HERE/$start" -o "$d/start.o" "${CFLAGS[@]}" $extra 2>>"$d/err" ||
     ! "$LLD" -o "$d/image.elf" -T "$d/link.ld" -static -nostdlib "$d/start.o" "$d/test.o" "$d/rt.o" \
          "$d/../$t/libsafec_std.a" "$BUILTINS/$t/libclang_rt.builtins.a" 2>>"$d/err"; then
    echo "$name BUILD-FAILED: $(grep -m2 -E 'error' "$d/err" | tr '\n' ' ' | cut -c1-300)"; fails=$((fails + 1)); continue
  fi
  timeout "$TIMEOUT" $qemu -kernel "$d/image.elf" > "$d/out.txt" 2>&1 < /dev/null
  if grep -q "QEMU-TEST PASS" "$d/out.txt"; then
    echo "$name ok     $(tr -d '\r' < "$d/out.txt" | grep -a -E '^(uart|mmu|tim|spi):' | tr '\n' ';' | cut -c1-300)"
  else
    echo "$name FAILED $(tr -d '\r' < "$d/out.txt" | grep -a -E '^(uart|mmu|tim|spi|trap):|FAIL' | tr '\n' ';' | cut -c1-300)"; fails=$((fails + 1))
  fi
done
exit $(( fails > 0 ))
