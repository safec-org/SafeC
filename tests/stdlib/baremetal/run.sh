#!/usr/bin/env bash
# Builds the bare-metal stdlib test image (bm_main.sc + bm_rt.sc + a std
# archive for the target) for each target alias and runs it in the Unicorn
# emulator via run.py.
#   usage: run.sh [alias...]            (default: every alias)
# Env: SAFEC, CLANG, LLD (ld.lld), BUILTINS (dir from build_builtins.sh,
#      default ~/opt/bm-builtins), PYTHONPATH must make 'unicorn' importable.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SAFEC="${SAFEC:-safec}"; CLANG="${CLANG:-clang}"; LLD="${LLD:-ld.lld}"
OBJCOPY="${OBJCOPY:-llvm-objcopy}"; NM="${NM:-llvm-nm}"; AR="${AR:-llvm-ar}"
BUILTINS="${BUILTINS:-$HOME/opt/bm-builtins}"
OUT="${OUT:-$HERE/build}"
if [[ $# -gt 0 ]]; then TARGETS=("$@"); else mapfile -t TARGETS < <("$SAFEC" --list-targets | awk '{print $1}'); fi

LDS="$OUT/link.ld"
mkdir -p "$OUT"
cat > "$LDS" <<'LD'
ENTRY(main)
SECTIONS {
  . = 0x10000;
  .text   : { *(.text .text.*) }
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) }
  .data   : { *(.data .data.* .sdata .sdata.*) }
  .bss    : { *(.bss .bss.* .sbss .sbss.* COMMON) }
  /DISCARD/ : { *(.ARM.exidx*) *(.eh_frame*) *(.comment) }
}
LD

compile() {  # <target> <src> <obj>
    local ll="${3%.o}.ll"
    "$SAFEC" -I "$ROOT/std" -I "$ROOT" "$2" --target="$1" --emit-llvm --compat-preprocessor \
        -o "$ll" 2>/dev/null &&
    "$CLANG" -c "$ll" -o "$3" "${CFLAGS[@]}" -Wno-override-module 2>/dev/null
}

fails=0
for t in "${TARGETS[@]}"; do
    mapfile -t CFLAGS < <("$SAFEC" --print-target-clang-flags --target="$t")
    d="$OUT/$t"; mkdir -p "$d/std"; rm -f "$d"/std/*.o "$d/libsafec_std.a"
    # std archive: every source that builds for this target.
    while IFS= read -r -d '' f; do
        case "$(basename "$f" .sc)" in   # OS-specific backends never apply on bare metal
            *_linux|*_bsd|*_win32|*_posix|*_epoll|*_kqueue|*_cocoa|*_x11) continue ;;
        esac
        o="$d/std/$(echo "${f#$ROOT/std/}" | tr '/.' '__').o"
        compile "$t" "$f" "$o" || rm -f "$o"
    done < <(find "$ROOT/std" -name '*.sc' -not -path '*/wasm/*' -print0 | sort -z)
    "$AR" rcs "$d/libsafec_std.a" "$d"/std/*.o
    if ! compile "$t" "$HERE/bm_main.sc" "$d/bm_main.o" || ! compile "$t" "$HERE/bm_rt.sc" "$d/bm_rt.o"; then
        echo "$t BUILD-FAILED (test sources)"; fails=$((fails + 1)); continue
    fi
    if ! "$LLD" -o "$d/image.elf" -T "$LDS" --gc-sections -static -nostdlib \
            "$d/bm_main.o" "$d/bm_rt.o" "$d/libsafec_std.a" \
            "$BUILTINS/$t/libclang_rt.builtins.a" 2>"$d/link.err"; then
        echo "$t LINK-FAILED: $(head -3 "$d/link.err" | tr '\n' ' ')"; fails=$((fails + 1)); continue
    fi
    "$OBJCOPY" -O binary "$d/image.elf" "$d/image.bin"
    "$NM" "$d/image.elf" > "$d/syms.txt"
    python3 "$HERE/run.py" "$t" "$d/image.bin" "$d/syms.txt" || fails=$((fails + 1))
done
exit $(( fails > 0 ))
