// SafeC Standard Library — MBR Partition Table Parser
// Freestanding-safe.
#pragma once
#include "block.h"

#define PART_MAX  4       // MBR has 4 primary partitions

#define PART_TYPE_EMPTY   0x00
#define PART_TYPE_FAT32   0x0B
#define PART_TYPE_FAT32X  0x0C   // FAT32 with LBA
#define PART_TYPE_LINUX   0x83
#define PART_TYPE_SWAP    0x82

struct PartEntry {
    unsigned char  status;      // 0x80 = bootable
    unsigned char  type;        // partition type
    unsigned long  lba_start;   // first LBA (host byte order)
    unsigned long  sector_count;
};

struct PartTable {
    struct PartEntry entries[PART_MAX];
    int              count;     // number of valid (non-empty) entries

    // Access entry by index (0-3).
    // Returns pointer into entries[]; check entry.type != PART_TYPE_EMPTY.
    const struct PartEntry* get(int idx) const;
};

// Read and parse the MBR from sector 0 of `dev`.
// Returns 0 on success, -1 on read error or bad signature.
int partition_read(&stack BlockDevice dev, &stack PartTable table_out);
