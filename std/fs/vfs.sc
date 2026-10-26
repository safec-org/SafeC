// SafeC Standard Library — VFS Implementation
#pragma once
#include <std/stddef.h>
#include <std/mem.h>
#include <std/fs/vfs.h>

namespace std {

// Definitions of the globals vfs.h declares extern.
struct Vfs vfs_root;


extern int   strcmp(const char* a, const char* b);
extern size_t strlen(const char* s);
extern char* strncpy(char* d, const char* s, size_t n);


inline struct VfsOps vfs_ops_none_() {
    struct VfsOps ops;
    safe_memset_obj(&ops, 0);
    return ops;
}

inline void vfs_init() {
    safe_memset_obj(&vfs_root, 0);
    vfs_root.mount_count = 0;
}

// ── VfsNode forwarding methods ────────────────────────────────────────────────

inline unsigned long VfsNode::read(unsigned long offset, []unsigned char buf) {
    match (self.fs_ctx) {
        case some(m): {
            match (m.ops.read) {
                case some(f): {
                    match (m.ctx) { case some(c): return f(c, self.inode, offset, buf); case null: { } }
                }
                case null: { }
            }
        }
        case null: { }
    }
    return (unsigned long)0;
}

inline unsigned long VfsNode::write(unsigned long offset, []const unsigned char buf) {
    match (self.fs_ctx) {
        case some(m): {
            match (m.ops.write) {
                case some(f): {
                    match (m.ctx) { case some(c): return f(c, self.inode, offset, buf); case null: { } }
                }
                case null: { }
            }
        }
        case null: { }
    }
    return (unsigned long)0;
}

// Hands each entry to the caller's callback with fs_ctx set to the mount,
// so the entry can be read or listed like a node from Vfs::open.
struct VfsListing_ {
    fn void(&VfsNode, ?&Erased) cb;     // the caller's callback
    ?&Erased user;
    ?&VfsMount mount;
};

static void vfs_listing_entry_(&VfsNode node, ?&VfsListing_ lp) {
    match (lp) {
        case some(l): {
            node.fs_ctx = l.mount;
            l.cb(node, l.user);
        }
        case null: { }
    }
}

inline int VfsNode::readdir_raw_(fn void(&VfsNode, ?&Erased) cb, ?&Erased user) {
    match (self.fs_ctx) {
        case some(m): {
            match (m.ops.readdir) {
                case some(f): {
                    match (m.ctx) {
                        case some(c): {
                            struct VfsListing_ l;
                            l.cb = cb;
                            l.user = user;
                            l.mount = m;
                            fn void(&VfsNode, ?&VfsListing_) entry = vfs_listing_entry_;
                            unsafe {
                                return f(c, self.inode, (fn void(&VfsNode, ?&Erased))entry, (?&Erased)&l);
                            }
                        }
                        case null: { }
                    }
                }
                case null: { }
            }
        }
        case null: { }
    }
    return -1;
}

// ── Vfs methods ───────────────────────────────────────────────────────────────

// Find the mount whose mountpoint is longest prefix of `path`.
static struct VfsMount* vfs_find_mount_(struct Vfs* v, const char* path) {
    struct VfsMount* best;
    unsafe { best = (struct VfsMount*)0; }
    unsigned long best_len = (unsigned long)0;
    int i = 0;
    unsafe {
    while (i < v->mount_count) {
        struct VfsMount* m = (struct VfsMount*)&v->mounts[i];
        if (m->active != 0) {
            unsigned long mlen = strlen(m->mountpoint);
            int matches = 0;
            {
                // strncmp equivalent
                int j = 0;
                matches = 1;
                while ((unsigned long)j < mlen) {
                    if (m->mountpoint[j] != path[j]) { matches = 0; break; }
                    j = j + 1;
                }
            }
            if (matches != 0 && mlen > best_len) {
                best = m;
                best_len = mlen;
            }
        }
        i = i + 1;
    }
    }
    return best;
}

// Strip the mountpoint prefix from path to get the relative path.
static const char* vfs_rel_path_(const char* path, const char* mountpoint) {
    unsigned long mlen;
    unsafe { mlen = strlen(mountpoint); }
    unsafe {
        const char* rel = path + mlen;
        if (rel[0] == (char)0) { return (const char*)"/"; }
        return rel;
    }
    return path;
}

int Vfs::mount_raw_(const &char mountpoint, struct VfsOps ops, &Erased ctx) {
    if (self.mount_count >= VFS_MAX_MOUNTS) { return -1; }
    int i = 0;
    while (i < VFS_MAX_MOUNTS) {
        if (self.mounts[i].active == 0) {
            unsafe {
                strncpy((char*)self.mounts[i].mountpoint, mountpoint,
                        (unsigned long)VFS_MAX_NAME - (unsigned long)1);
                self.mounts[i].mountpoint[VFS_MAX_NAME - 1] = (char)0;
                self.mounts[i].ops = ops;
            }
            self.mounts[i].ctx    = ctx;
            self.mounts[i].active = 1;
            self.mount_count      = self.mount_count + 1;
            return 0;
        }
        i = i + 1;
    }
    return -1;
}

inline int Vfs::unmount(const &char mountpoint) {
    int i = 0;
    while (i < VFS_MAX_MOUNTS) {
        if (self.mounts[i].active != 0) {
            int eq = 0;
            unsafe { eq = (strcmp(self.mounts[i].mountpoint, mountpoint) == 0); }
            if (eq != 0) {
                self.mounts[i].active = 0;
                self.mount_count = self.mount_count - 1;
                return 0;
            }
        }
        i = i + 1;
    }
    return -1;
}

inline int Vfs::open(const &char path, int flags, &VfsNode node_out) {
    struct VfsMount* m;
    unsafe { m = vfs_find_mount_((struct Vfs*)self, path); }
    if (m == (struct VfsMount*)0) { return -1; }
    int rc = -1;
    unsafe {
        const char* rel = vfs_rel_path_(path, (const char*)m->mountpoint);
        match (m->ops.open) {
            case some(f): {
                match (m->ctx) { case some(c): rc = f(c, rel, flags, node_out); case null: { } }
            }
            case null: return -1;
        }
    }
    if (rc == 0) {
        unsafe { node_out.fs_ctx = (?&VfsMount)m; }
    }
    return rc;
}

inline int Vfs::unlink(const &char path) {
    struct VfsMount* m;
    unsafe { m = vfs_find_mount_((struct Vfs*)self, path); }
    if (m == (struct VfsMount*)0) { return -1; }
    unsafe {
        const char* rel = vfs_rel_path_(path, (const char*)m->mountpoint);
        match (m->ops.unlink) {
            case some(f): {
                match (m->ctx) { case some(c): return f(c, rel); case null: { } }
            }
            case null: { }
        }
    }
    return -1;
}

inline int Vfs::mkdir(const &char path) {
    struct VfsMount* m;
    unsafe { m = vfs_find_mount_((struct Vfs*)self, path); }
    if (m == (struct VfsMount*)0) { return -1; }
    unsafe {
        const char* rel = vfs_rel_path_(path, (const char*)m->mountpoint);
        match (m->ops.mkdir) {
            case some(f): {
                match (m->ctx) { case some(c): return f(c, rel); case null: { } }
            }
            case null: { }
        }
    }
    return -1;
}

} // namespace std
