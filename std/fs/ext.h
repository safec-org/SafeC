// SafeC Standard Library — ext2 Read-Only Filesystem Driver
// Implements read-only access to ext2 (and compatible ext3/ext4) volumes.
// Freestanding-safe.
#pragma once
#include <std/fs/block.h>
#include <std/fs/vfs.h>

// ext2 constants
#define EXT2_SUPER_MAGIC   0xEF53
#define EXT2_ROOT_INODE    2
#define EXT2_BLOCK_SIZE_BASE 1024
#define EXT2_MAX_BLOCK_SIZE  4096   // largest supported block size (Ext2Ctx::init rejects bigger)

// Inode type masks
#define EXT2_S_IFREG  0x8000
#define EXT2_S_IFDIR  0x4000

// Superblock (relevant fields only, 1024-byte block at offset 1024).
namespace std {

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
    unsigned short inode_size;     // bytes per on-disk inode (128 for rev 0; s_inode_size for rev >= 1)
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

    // Initialise this context in place from a block device (copied into
    // `dev`).  Returns 0 on success.
    int  init(&BlockDevice dev);

    // Read block `block_no` into the first block_size bytes of `buf`
    // (-1 if it is shorter).
    int  read_block(unsigned long block_no, []unsigned char buf);

    // Read inode `ino` into `inode_out`.  Returns 0 on success.
    int  read_inode(unsigned long ino, &Ext2Inode inode_out);

    // Read up to buf.len bytes of inode `ino`'s data at byte `offset`.
    unsigned long read_file(unsigned long ino, unsigned long offset, []unsigned char buf);

    // ── FsRead (vfs.h): mount with vfs.mount_ro(mountpoint, &ctx) ──────────
    int           open(const &char path, int flags, &VfsNode node_out);
    unsigned long read(unsigned long inode, unsigned long offset, []unsigned char buf);
    generic<T> int readdir(unsigned long inode, fn void(&VfsNode, ?&T) cb, ?&T user);
    int           readdir_raw_(unsigned long inode, fn void(&VfsNode, ?&Erased) cb, ?&Erased user);
};

generic<T>
int Ext2Ctx::readdir(unsigned long inode, fn void(&VfsNode, ?&T) cb, ?&T user) {
    unsafe { return self.readdir_raw_(inode, (fn void(&VfsNode, ?&Erased))cb, (?&Erased)user); }
}

} // namespace std
