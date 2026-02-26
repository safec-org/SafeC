// SafeC Standard Library — tmpfs (in-memory filesystem)
// Fixed inode table; no dynamic allocation (backed by static arrays).
// Freestanding-safe.
#pragma once
#include "vfs.h"

#define TMPFS_MAX_FILES   32
#define TMPFS_MAX_DATA    (64 * 1024)   // 64 KiB total data pool

struct TmpfsInode {
    char           name[VFS_MAX_NAME];
    int            type;            // VFS_TYPE_FILE / VFS_TYPE_DIR
    unsigned long  parent;          // parent inode index (0 = root)
    unsigned long  data_off;        // offset into data pool (file only)
    unsigned long  size;
    int            used;
};

struct TmpfsCtx {
    struct TmpfsInode inodes[TMPFS_MAX_FILES];
    unsigned char     data[TMPFS_MAX_DATA];
    unsigned long     data_used;
    int               inode_count;
};

// Initialise a tmpfs context (zero-fills).
void tmpfs_init(&stack TmpfsCtx ctx);

// Return VfsOps for the tmpfs driver (ctx must be a TmpfsCtx*).
struct VfsOps tmpfs_ops();
