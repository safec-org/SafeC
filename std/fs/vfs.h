// SafeC Standard Library — Virtual Filesystem (VFS) Layer
// Minimal inode-based VFS with driver registration. Freestanding-safe.
#pragma once

#define VFS_MAX_MOUNTS  8
#define VFS_MAX_PATH    256
#define VFS_MAX_NAME    64

// Node types
#define VFS_TYPE_FILE   0
#define VFS_TYPE_DIR    1

// Open flags
#define VFS_O_READ      0x01
#define VFS_O_WRITE     0x02
#define VFS_O_CREATE    0x04
#define VFS_O_TRUNC     0x08

// Seek origins
#define VFS_SEEK_SET    0
#define VFS_SEEK_CUR    1
#define VFS_SEEK_END    2

namespace std {

struct VfsNode {
    char          name[VFS_MAX_NAME];
    int           type;             // VFS_TYPE_FILE / VFS_TYPE_DIR
    unsigned long size;             // file size in bytes (0 for dirs)
    unsigned long inode;            // driver-specific inode number
    ?&void        fs_ctx;           // driver-internal: the owning VfsMount

    // Read `len` bytes from file at `offset` into `buf`.  Returns bytes read.
    unsigned long read(unsigned long offset, &unsigned char buf, unsigned long len);

    // Write `len` bytes from `buf` to file at `offset`.  Returns bytes written.
    unsigned long write(unsigned long offset, const &unsigned char buf, unsigned long len);

    // Calls cb(entry, user) for each child of this directory.
    // Returns the entry count, or -1 on error.
    generic<T> int readdir(fn void(&VfsNode, &T) cb, &T user);

    // Type-erased listing behind readdir().
    int readdir_raw_(void* cb, void* user);
};

// Driver interface: filled by each filesystem.
struct VfsOps {
    // Mount from a raw byte buffer (e.g. RAM disk) or opaque driver handle.
    fn int(?&void ctx, unsigned long size) mount;
    fn int(?&void ctx) unmount;
    fn int(?&void ctx, const &char path, int flags, &VfsNode node_out) open;
    fn int(?&void ctx, const &char path) unlink;
    fn int(?&void ctx, const &char path) mkdir;
    fn unsigned long(?&void ctx, unsigned long inode, unsigned long off,
                      &unsigned char buf, unsigned long len) read;
    fn unsigned long(?&void ctx, unsigned long inode, unsigned long off,
                      const &unsigned char buf, unsigned long len) write;
    fn int(?&void ctx, unsigned long inode,
           fn void(&VfsNode, ?&void) cb, ?&void user) readdir;
};

struct VfsMount {
    char            mountpoint[VFS_MAX_NAME];
    struct VfsOps   ops;
    void*           ctx;
    int             active;
};

struct Vfs {
    struct VfsMount mounts[VFS_MAX_MOUNTS];
    int             mount_count;

    // Register a filesystem at `mountpoint` (e.g. "/" or "/tmp"); ctx is the
    // driver context its ops expect (TmpFs, FatCtx, Ext2Ctx, ...).
    generic<T> int mount(const &char mountpoint, struct VfsOps ops, &T ctx);

    // Type-erased registration behind mount().
    int  mount_raw_(const char* mountpoint, struct VfsOps ops, void* ctx);

    // Unmount filesystem at `mountpoint`.
    int  unmount(const &char mountpoint);

    // Open path; fills `node_out`.  Returns 0 on success.
    int  open(const &char path, int flags, &VfsNode node_out);

    // Remove a file.
    int  unlink(const &char path);

    // Create a directory.
    int  mkdir(const &char path);
};

// Global VFS instance (single-root for embedded use).
extern struct Vfs vfs_root;
void vfs_init();

generic<T>
int Vfs::mount(const &char mountpoint, struct VfsOps ops, &T ctx) {
    unsafe { return self.mount_raw_(mountpoint, ops, (void*)ctx); }
}

generic<T>
int VfsNode::readdir(fn void(&VfsNode, &T) cb, &T user) {
    unsafe { return self.readdir_raw_((void*)cb, (void*)user); }
}

} // namespace std
