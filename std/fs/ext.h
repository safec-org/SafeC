// SafeC Standard Library — ext2 Read-Only Filesystem Driver
// Implements read-only access to ext2 (and compatible ext3/ext4) volumes.
// Freestanding-safe.
#pragma once
#include "block.h"
#include "vfs.h"

// ext2 constants
#define EXT2_SUPER_MAGIC   0xEF53
#define EXT2_ROOT_INODE    2
#define EXT2_BLOCK_SIZE_BASE 1024

// Inode type masks
#define EXT2_S_IFREG  0x8000
#define EXT2_S_IFDIR  0x4000

// Superblock (relevant fields only, 1024-byte block at offset 1024).
struct Ext2Super {
    unsigned int  inodes_count;
    unsigned int  blocks_count;
    unsigned int  reserved_blocks;
    unsigned int  free_blocks;
    unsigned int  free_inodes;
    unsigned int  first_data_block;
    unsigned int  log_block_size;   // block_size = 1024 << log_block_size
    unsigned int  log_frag_size;
    unsigned int  blocks_per_group;
    unsigned int  frags_per_group;
    unsigned int  inodes_per_group;
    unsigned int  mtime;
    unsigned int  wtime;
    unsigned short mnt_count;
    unsigned short max_mnt_count;
    unsigned short magic;           // must be EXT2_SUPER_MAGIC
    unsigned short state;
    unsigned short errors;
    unsigned short minor_rev;
    unsigned int  lastcheck;
    unsigned int  checkinterval;
    unsigned int  creator_os;
    unsigned int  rev_level;
};

// Block group descriptor (32 bytes).
struct Ext2GroupDesc {
    unsigned int  block_bitmap;
    unsigned int  inode_bitmap;
    unsigned int  inode_table;
    unsigned short free_blocks;
    unsigned short free_inodes;
    unsigned short used_dirs;
};

// Inode (128 bytes, fields we need).
struct Ext2Inode {
    unsigned short mode;
    unsigned short uid;
    unsigned int   size_lo;
    unsigned int   atime;
    unsigned int   ctime;
    unsigned int   mtime;
    unsigned int   dtime;
    unsigned short gid;
    unsigned short links_count;
    unsigned int   blocks_count;  // 512-byte blocks
    unsigned int   flags;
    unsigned int   reserved1;
    unsigned int   block[15];     // 12 direct + 1 indirect + 1 dbl + 1 trpl
};

struct Ext2Ctx {
    struct BlockDevice dev;
    struct Ext2Super   super;
    unsigned long      block_size;
    unsigned long      inodes_per_group;
    unsigned long      blocks_per_group;
    unsigned long      group_count;

    // Read block `block_no` into `buf` (must be >= block_size bytes).
    int  read_block(unsigned long block_no, unsigned char* buf);

    // Read inode `ino` into `inode_out`.  Returns 0 on success.
    int  read_inode(unsigned long ino, &stack Ext2Inode inode_out);

    // Read file data for inode at byte offset `offset` into `buf` for `len` bytes.
    unsigned long read_file(unsigned long ino, unsigned long offset,
                             unsigned char* buf, unsigned long len);
};

// Initialise ext2 context from a block device.  Returns 0 on success.
int  ext2_init(&stack Ext2Ctx ctx, &stack BlockDevice dev);

// Return VfsOps for the ext2 driver.
struct VfsOps ext2_ops();
