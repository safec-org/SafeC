#!/usr/bin/env bash
# Builds libclang_rt.builtins.a for every bare-metal target alias from a
# compiler-rt source tree (bare-metal images need __udivdi3, __aeabi_idiv,
# __mulsi3 on RV32I/E, soft-float helpers, ...; LLVM's release tarballs only
# ship the host's).
#   usage: build_builtins.sh <compiler-rt-src> <outdir> [alias...]
# Produces <outdir>/<alias>/libclang_rt.builtins.a
set -uo pipefail
CRT="${1:?compiler-rt source dir}"; OUT="${2:?output dir}"; shift 2
SAFEC="${SAFEC:-safec}"; CLANG="${CLANG:-clang}"; AR="${AR:-llvm-ar}"
B="$CRT/lib/builtins"
if [[ $# -gt 0 ]]; then TARGETS=("$@"); else mapfile -t TARGETS < <("$SAFEC" --list-targets | awk '{print $1}'); fi
for t in "${TARGETS[@]}"; do
    mapfile -t FL < <("$SAFEC" --print-target-clang-flags --target="$t")
    dir="$OUT/$t"; mkdir -p "$dir/obj"; rm -f "$dir/obj/"*.o "$dir/libclang_rt.builtins.a"
    arch=()
    case "$t" in
        aarch32|thumb|thumb2|arm-*) for f in "$B"/arm/*.S "$B"/arm/*.c; do [[ $f == *vfp* || $f == *chkstk* || $f == *sync* ]] || arch+=("$f"); done ;;
        rv*)                  arch+=("$B"/riscv/*.S) ;;
        x86)                  arch+=("$B"/i386/*.S) ;;
    esac
    build() {  # compile each listed source; drop objects that fail
        printf '%s\n' "$@" | xargs -P "$(nproc)" -I{} sh -c '
            f="$1"; shift; o="'"$dir"'/obj/$(basename "${f%.*}").o"
            "'"$CLANG"'" "$@" -c "$f" -o "$o" -O2 -ffreestanding -fno-builtin -I"'"$B"'" \
                -Wno-everything 2>/dev/null || rm -f "$o"' _ {} "${FL[@]}"
    }
    # Arch-specific implementations first; the generic C version of the same
    # routine only fills in where no arch object built (e.g. riscv/muldi3.S
    # is RV64-only; arm/clzsi2.S needs Thumb-2). Generic muldi3.c must not
    # shadow riscv/muldi3.S on RV64: without M it compiles to a call to
    # __muldi3 itself.
    [[ ${#arch[@]} -gt 0 ]] && build "${arch[@]}"
    # Arch sources guarded out for this sub-target assemble to empty objects;
    # those mustn't count as "implemented".
    for o in "$dir"/obj/*.o; do
        [[ -e $o ]] || continue
        [[ -n "$("${NM:-llvm-nm}" --defined-only -g "$o" 2>/dev/null)" ]] || rm -f "$o"
    done
    gen=()
    for f in "$B"/*.c; do
        [[ -f "$dir/obj/$(basename "${f%.*}").o" ]] || gen+=("$f")
    done
    build "${gen[@]}"
    "$AR" rcs "$dir/libclang_rt.builtins.a" "$dir"/obj/*.o
    echo "$t: $(ls "$dir"/obj/*.o | wc -l) objects"
done
