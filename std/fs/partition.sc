// SafeC Standard Library — Partition Table Implementation
#pragma once
#include "partition.h"

// MBR layout: 446 bytes bootstrap + 4×16-byte partition entries + 2-byte signature
#define MBR_ENTRY_OFFSET  446
#define MBR_SIG_OFFSET    510
#define MBR_SIG_0         0x55
#define MBR_SIG_1         0xAA

const struct PartEntry* PartTable::get(int idx) const {
    if (idx < 0 || idx >= PART_MAX) { return (const struct PartEntry*)0; }
    unsafe { return (const struct PartEntry*)&self.entries[idx]; }
    return (const struct PartEntry*)0;
}

int partition_read(&stack BlockDevice dev, &stack PartTable table_out) {
    unsigned char mbr[512];
    if (dev.read((unsigned long)0, (unsigned char*)mbr, (unsigned long)1) != 0) {
        return -1;
    }
    // Validate signature.
    unsafe {
        if (mbr[MBR_SIG_OFFSET]   != (unsigned char)MBR_SIG_0 ||
            mbr[MBR_SIG_OFFSET+1] != (unsigned char)MBR_SIG_1) {
            return -1;
        }
    }
    table_out.count = 0;
    int i = 0;
    while (i < PART_MAX) {
        unsafe {
            unsigned char* e = (unsigned char*)mbr + MBR_ENTRY_OFFSET + i * 16;
            table_out.entries[i].status = e[0];
            table_out.entries[i].type   = e[4];
            // LBA start at bytes 8-11 (little-endian)
            table_out.entries[i].lba_start =
                (unsigned long)e[8]  |
                ((unsigned long)e[9]  << 8) |
                ((unsigned long)e[10] << 16) |
                ((unsigned long)e[11] << 24);
            // sector count at bytes 12-15 (little-endian)
            table_out.entries[i].sector_count =
                (unsigned long)e[12] |
                ((unsigned long)e[13] << 8) |
                ((unsigned long)e[14] << 16) |
                ((unsigned long)e[15] << 24);
            if (table_out.entries[i].type != (unsigned char)PART_TYPE_EMPTY) {
                table_out.count = table_out.count + 1;
            }
        }
        i = i + 1;
    }
    return 0;
}
