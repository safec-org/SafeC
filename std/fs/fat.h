// SafeC Standard Library — FAT32 Read-Only Driver
// Freestanding-safe. Uses a BlockDevice for sector reads.
#pragma once
#include "block.h"
#include "vfs.h"

// Maximum path components parsed
#define FAT_MAX_DEPTH  8

struct FatBpb {
    unsigned char  oem_name[8];
    unsigned short bytes_per_sector;
    unsigned char  sectors_per_cluster;
    unsigned short reserved_sectors;
    unsigned char  num_fats;
    unsigned short root_entry_count;    // 0 for FAT32
    unsigned short total_sectors_16;
    unsigned char  media;
    unsigned short fat_size_16;         // 0 for FAT32
    unsigned short sectors_per_track;
    unsigned short num_heads;
    unsigned int   hidden_sectors;
    unsigned int   total_sectors_32;
    // FAT32 extended
    unsigned int   fat_size_32;
    unsigned short ext_flags;
    unsigned short fs_version;
    unsigned int   root_cluster;
    unsigned short fs_info;
    unsigned short backup_boot;
};

struct FatCtx {
    struct BlockDevice dev;
    unsigned long  partition_lba;      // LBA of FAT32 partition start
    struct FatBpb  bpb;
    unsigned long  fat_lba;            // LBA of FAT region
    unsigned long  data_lba;           // LBA of first data cluster
    unsigned long  root_cluster;       // cluster of root directory
    unsigned long  bytes_per_cluster;

    // Read a cluster into `buf` (must be >= bytes_per_cluster bytes).
    int read_cluster(unsigned long cluster, unsigned char* buf);

    // Follow FAT chain to find cluster at position `idx` from `start_cluster`.
    unsigned long follow_fat(unsigned long start_cluster, unsigned long idx);
};

// Initialise FAT32 context from a block device partition starting at `lba`.
// Returns 0 on success.
int fat_init(&stack FatCtx ctx, &stack BlockDevice dev, unsigned long lba);

// Return VfsOps for the FAT32 driver (ctx must be a FatCtx*).
struct VfsOps fat_ops();
