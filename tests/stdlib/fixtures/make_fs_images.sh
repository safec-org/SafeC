#!/usr/bin/env bash
# Builds the filesystem images test_fs.sc reads (no root needed):
#   $1/fat32.img — FAT32, 512-byte clusters: HELLO.TXT (1300 bytes, spans
#                  3 clusters) and SUB/A.TXT
#   $1/ext2.img  — ext2, 1 KiB blocks, same tree (populated via mke2fs -d)
# Exits non-zero if the tools aren't available; the test then skips.
set -euo pipefail
OUT="${1:?usage: make_fs_images.sh <outdir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PATH="$PATH:/usr/sbin:/sbin"
command -v mkfs.vfat >/dev/null && command -v mke2fs >/dev/null && command -v python3 >/dev/null
mkdir -p "$OUT"
TREE="$(mktemp -d)"
trap 'rm -rf "$TREE"' EXIT
python3 -c "import sys; sys.stdout.write(''.join(chr(65 + i % 26) for i in range(1300)))" > "$TREE/HELLO.TXT"
mkdir "$TREE/SUB"
printf 'nested file\n' > "$TREE/SUB/A.TXT"

rm -f "$OUT/fat32.img" "$OUT/ext2.img"
truncate -s 40M "$OUT/fat32.img"
mkfs.vfat -F 32 -S 512 -s 1 -n SAFEC "$OUT/fat32.img" >/dev/null
python3 -I "$HERE/fat32_add.py" "$OUT/fat32.img" "$TREE"

mke2fs -q -t ext2 -b 1024 -d "$TREE" "$OUT/ext2.img" 4M
