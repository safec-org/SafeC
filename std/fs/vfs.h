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

struct VfsNode {
    char          name[VFS_MAX_NAME];
    int           type;             // VFS_TYPE_FILE / VFS_TYPE_DIR
    unsigned long size;             // file size in bytes (0 for dirs)
    unsigned long inode;            // driver-specific inode number
    void*         fs_ctx;           // pointer back to mounted filesystem

    // Read `len` bytes from file at `offset` into `buf`.  Returns bytes read.
    unsigned long read(unsigned long offset, &stack unsigned char buf, unsigned long len);

    // Write `len` bytes from `buf` to file at `offset`.  Returns bytes written.
    unsigned long write(unsigned long offset, const &stack unsigned char buf, unsigned long len);

    // List directory entries.  Calls `cb(entry, user)` for each child.
    // Returns count of entries, -1 on error.
    int readdir(void* cb, void* user);
};

// Driver interface: filled by each filesystem.
struct VfsOps {
    // Mount from a raw byte buffer (e.g. RAM disk) or opaque driver handle.
    int (*mount)(void* ctx, unsigned long size);
    int (*unmount)(void* ctx);
    int (*open)(void* ctx, const char* path, int flags, &stack VfsNode node_out);
    int (*unlink)(void* ctx, const char* path);
    int (*mkdir)(void* ctx, const char* path);
    unsigned long (*read)(void* ctx, unsigned long inode, unsigned long off,
                          unsigned char* buf, unsigned long len);
    unsigned long (*write)(void* ctx, unsigned long inode, unsigned long off,
                           const unsigned char* buf, unsigned long len);
    int (*readdir)(void* ctx, unsigned long inode, void* cb, void* user);
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

    // Register a filesystem at `mountpoint` (e.g. "/" or "/tmp").
    int  mount(const char* mountpoint, struct VfsOps ops, void* ctx);

    // Unmount filesystem at `mountpoint`.
    int  unmount(const char* mountpoint);

    // Open path; fills `node_out`.  Returns 0 on success.
    int  open(const char* path, int flags, &stack VfsNode node_out);

    // Remove a file.
    int  unlink(const char* path);

    // Create a directory.
    int  mkdir(const char* path);
};

// Global VFS instance (single-root for embedded use).
extern struct Vfs vfs_root;
void vfs_init();
