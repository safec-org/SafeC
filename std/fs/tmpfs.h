// SafeC Standard Library — tmpfs (in-memory filesystem)
// Fixed inode table; no dynamic allocation (backed by static arrays).
// Freestanding-safe.
#pragma once
#include <std/fs/vfs.h>

#define TMPFS_MAX_FILES   32
#define TMPFS_MAX_DATA    (64 * 1024)   // 64 KiB total data pool

namespace std {

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

    // Initialise this context in place (zero-fills, creates the root).
    void init();

    // ── FileSystem (vfs.h): mount with vfs.mount(mountpoint, &ctx) ─────────
    int           open(const &char path, int flags, &VfsNode node_out);
    unsigned long read(unsigned long inode, unsigned long offset, &unsigned char buf, unsigned long len);
    int           readdir(unsigned long inode, fn void(&VfsNode, ?&void) cb, ?&void user);
    unsigned long write(unsigned long inode, unsigned long offset, const &unsigned char buf, unsigned long len);
    int           unlink(const &char path);
    int           mkdir(const &char path);
};

} // namespace std
