// SafeC Standard Library — Virtual Filesystem (VFS) Layer
// Minimal inode-based VFS with driver registration. Freestanding-safe.
//
// A filesystem driver is a context struct with methods; the traits below
// name them. Vfs::mount / mount_ro build the VfsOps table from those
// methods at compile time (one thunk per method, instantiated for the
// driver's type), so a driver writes no function table of its own:
//
//   FsRead      open, read, readdir                  FatCtx, Ext2Ctx
//   FsWrite     write, unlink, mkdir
//   FileSystem  FsRead + FsWrite                     TmpfsCtx
//
//   struct TmpfsCtx tmp; tmp.init();
//   vfs_init();
//   vfs_root.mount("/tmp", &tmp);       // read-write: needs a FileSystem
//   vfs_root.mount_ro("/sd", &fat);     // read-only: any FsRead
//
// Paths a driver sees are relative to its mountpoint ("/" for the root).
// A driver need not set VfsNode.fs_ctx: the VFS fills it in.
#pragma once
#include <std/erased.h>

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
    ?&struct VfsMount fs_ctx;       // driver-internal: the owning VfsMount

    // Reads up to buf.len bytes of the file at `offset` into `buf`.
    // Returns bytes read.
    unsigned long read(unsigned long offset, []unsigned char buf);

    // Writes `buf` to the file at `offset`.  Returns bytes written.
    unsigned long write(unsigned long offset, []const unsigned char buf);

    // Calls cb(entry, user) for each child of this directory.
    // Returns the entry count, or -1 on error.
    generic<T> int readdir(fn void(&VfsNode, &T) cb, &T user);

    // Type-erased listing behind readdir() (see std/erased.h).
    int readdir_raw_(fn void(&VfsNode, ?&Erased) cb, ?&Erased user);
};

trait FsRead {
    // Looks up path (VFS_O_CREATE: creates a missing file) and fills
    // node_out's name, type, size and inode. 0 on success.
    int open(const &char path, int flags, &VfsNode node_out);
    // Up to buf.len bytes of file `inode` from offset; returns the count read.
    unsigned long read(unsigned long inode, unsigned long offset, []unsigned char buf);
    // cb(entry, user) per child of directory `inode`; the entry count, or
    // -1. Drivers declare it generic<T>: (fn void(&VfsNode, ?&T), ?&T).
    int readdir(unsigned long inode, fn void(&VfsNode, ?&Erased) cb, ?&Erased user);
}

trait FsWrite {
    unsigned long write(unsigned long inode, unsigned long offset, []const unsigned char buf);
    int unlink(const &char path);   // 0 on success
    int mkdir(const &char path);    // 0 on success
}

trait FileSystem : FsRead + FsWrite {}

// The mounted driver's entry points, type-erased so one Vfs holds drivers
// of different context types. Built by Vfs::mount / mount_ro; a null
// entry (a read-only mount's write/unlink/mkdir) makes that call fail.
// (See std/erased.h: ctx is the driver's &F.)
struct VfsOps {
    ?fn int(&Erased ctx, const &char path, int flags, &VfsNode node_out) open;
    ?fn int(&Erased ctx, const &char path) unlink;
    ?fn int(&Erased ctx, const &char path) mkdir;
    ?fn unsigned long(&Erased ctx, unsigned long inode, unsigned long off, []unsigned char buf) read;
    ?fn unsigned long(&Erased ctx, unsigned long inode, unsigned long off, []const unsigned char buf) write;
    ?fn int(&Erased ctx, unsigned long inode,
            fn void(&VfsNode, ?&Erased) cb, ?&Erased user) readdir;
};

struct VfsMount {
    char            mountpoint[VFS_MAX_NAME];
    struct VfsOps   ops;
    ?&Erased        ctx;      // the mounted driver (see VfsOps)
    int             active;
};

struct Vfs {
    struct VfsMount mounts[VFS_MAX_MOUNTS];
    int             mount_count;

    // Mount fs at `mountpoint` (e.g. "/" or "/tmp"). 0 on success, -1 if
    // every mount slot is taken. fs must outlive the mount.
    generic<F: FileSystem> int mount(const &char mountpoint, &F fs);
    // Read-only: write, unlink and mkdir through this mount fail.
    generic<F: FsRead>     int mount_ro(const &char mountpoint, &F fs);

    // Type-erased registration behind mount() / mount_ro().
    int  mount_raw_(const &char mountpoint, struct VfsOps ops, &Erased ctx);

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

// An all-null table (vfs.sc).
struct VfsOps vfs_ops_none_();

// One thunk per driver method: called through the type-erased VfsOps entry
// with the mount's ctx, which mount() took as an &F.
generic<F: FsRead>
int vfs_open_thunk_(&F fs, const &char path, int flags, &VfsNode node_out) {
    return fs.open(path, flags, node_out);
}
generic<F: FsRead>
unsigned long vfs_read_thunk_(&F fs, unsigned long inode, unsigned long off, []unsigned char buf) {
    return fs.read(inode, off, buf);
}
generic<F: FsRead>
int vfs_readdir_thunk_(&F fs, unsigned long inode, fn void(&VfsNode, ?&Erased) cb, ?&Erased user) {
    return fs.readdir(inode, cb, user);
}
generic<F: FileSystem>
unsigned long vfs_write_thunk_(&F fs, unsigned long inode, unsigned long off, []const unsigned char buf) {
    return fs.write(inode, off, buf);
}
generic<F: FileSystem>
int vfs_unlink_thunk_(&F fs, const &char path) { return fs.unlink(path); }
generic<F: FileSystem>
int vfs_mkdir_thunk_(&F fs, const &char path) { return fs.mkdir(path); }

// The read entries of F's table (fs only selects F).
generic<F: FsRead>
struct VfsOps vfs_ops_read_(&F fs) {
    struct VfsOps ops = vfs_ops_none_();
    fn int(&F, const &char, int, &VfsNode) o = vfs_open_thunk_;
    fn unsigned long(&F, unsigned long, unsigned long, []unsigned char) r = vfs_read_thunk_;
    fn int(&F, unsigned long, fn void(&VfsNode, ?&Erased), ?&Erased) d = vfs_readdir_thunk_;
    unsafe {
        ops.open    = (?fn int(&Erased, const &char, int, &VfsNode))o;
        ops.read    = (?fn unsigned long(&Erased, unsigned long, unsigned long, []unsigned char))r;
        ops.readdir = (?fn int(&Erased, unsigned long, fn void(&VfsNode, ?&Erased), ?&Erased))d;
    }
    return ops;
}

generic<F: FileSystem>
int Vfs::mount(const &char mountpoint, &F fs) {
    struct VfsOps ops = vfs_ops_read_(fs);
    fn unsigned long(&F, unsigned long, unsigned long, []const unsigned char) w = vfs_write_thunk_;
    fn int(&F, const &char) u = vfs_unlink_thunk_;
    fn int(&F, const &char) m = vfs_mkdir_thunk_;
    unsafe {
        ops.write  = (?fn unsigned long(&Erased, unsigned long, unsigned long, []const unsigned char))w;
        ops.unlink = (?fn int(&Erased, const &char))u;
        ops.mkdir  = (?fn int(&Erased, const &char))m;
        return self.mount_raw_(mountpoint, ops, (&Erased)fs);
    }
}

generic<F: FsRead>
int Vfs::mount_ro(const &char mountpoint, &F fs) {
    struct VfsOps ops = vfs_ops_read_(fs);
    unsafe { return self.mount_raw_(mountpoint, ops, (&Erased)fs); }
}

generic<T>
int VfsNode::readdir(fn void(&VfsNode, &T) cb, &T user) {
    unsafe { return self.readdir_raw_((fn void(&VfsNode, ?&Erased))cb, (?&Erased)user); }
}

} // namespace std
