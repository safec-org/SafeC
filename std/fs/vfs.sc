// SafeC Standard Library — VFS Implementation
#pragma once
#include "vfs.h"

extern void* memset(void* p, int v, unsigned long n);
extern void* memcpy(void* d, const void* s, unsigned long n);
extern int   strcmp(const char* a, const char* b);
extern unsigned long strlen(const char* s);
extern void* strncpy(char* d, const char* s, unsigned long n);

struct Vfs vfs_root;

void vfs_init() {
    unsafe { memset((void*)&vfs_root, 0, sizeof(struct Vfs)); }
    vfs_root.mount_count = 0;
}

// ── VfsNode forwarding methods ────────────────────────────────────────────────

unsigned long VfsNode::read(unsigned long offset, &stack unsigned char buf, unsigned long len) {
    if (self.fs_ctx == (void*)0) { return (unsigned long)0; }
    struct VfsMount* m = (struct VfsMount*)self.fs_ctx;
    if (m->ops.read == (void*)0) { return (unsigned long)0; }
    unsafe { return m->ops.read(m->ctx, self.inode, offset, (unsigned char*)buf, len); }
    return (unsigned long)0;
}

unsigned long VfsNode::write(unsigned long offset, const &stack unsigned char buf, unsigned long len) {
    if (self.fs_ctx == (void*)0) { return (unsigned long)0; }
    struct VfsMount* m = (struct VfsMount*)self.fs_ctx;
    if (m->ops.write == (void*)0) { return (unsigned long)0; }
    unsafe { return m->ops.write(m->ctx, self.inode, offset, (const unsigned char*)buf, len); }
    return (unsigned long)0;
}

int VfsNode::readdir(void* cb, void* user) {
    if (self.fs_ctx == (void*)0) { return -1; }
    struct VfsMount* m = (struct VfsMount*)self.fs_ctx;
    if (m->ops.readdir == (void*)0) { return -1; }
    unsafe { return m->ops.readdir(m->ctx, self.inode, cb, user); }
    return -1;
}

// ── Vfs methods ───────────────────────────────────────────────────────────────

// Find the mount whose mountpoint is longest prefix of `path`.
static struct VfsMount* vfs_find_mount_(struct Vfs* v, const char* path) {
    struct VfsMount* best = (struct VfsMount*)0;
    unsigned long best_len = (unsigned long)0;
    int i = 0;
    while (i < v->mount_count) {
        struct VfsMount* m = &v->mounts[i];
        if (m->active != 0) {
            unsigned long mlen;
            unsafe { mlen = strlen(m->mountpoint); }
            int matches = 0;
            unsafe {
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

int Vfs::mount(const char* mountpoint, struct VfsOps ops, void* ctx) {
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
            if (ops.mount != (void*)0) {
                unsafe { ops.mount(ctx, (unsigned long)0); }
            }
            return 0;
        }
        i = i + 1;
    }
    return -1;
}

int Vfs::unmount(const char* mountpoint) {
    int i = 0;
    while (i < VFS_MAX_MOUNTS) {
        if (self.mounts[i].active != 0) {
            int eq = 0;
            unsafe { eq = (strcmp(self.mounts[i].mountpoint, mountpoint) == 0); }
            if (eq != 0) {
                if (self.mounts[i].ops.unmount != (void*)0) {
                    unsafe { self.mounts[i].ops.unmount(self.mounts[i].ctx); }
                }
                self.mounts[i].active = 0;
                self.mount_count = self.mount_count - 1;
                return 0;
            }
        }
        i = i + 1;
    }
    return -1;
}

int Vfs::open(const char* path, int flags, &stack VfsNode node_out) {
    struct VfsMount* m = vfs_find_mount_(&self, path);
    if (m == (struct VfsMount*)0) { return -1; }
    if (m->ops.open == (void*)0) { return -1; }
    const char* rel = vfs_rel_path_(path, (const char*)m->mountpoint);
    int rc;
    unsafe { rc = m->ops.open(m->ctx, rel, flags, node_out); }
    if (rc == 0) {
        node_out.fs_ctx = (void*)m;
    }
    return rc;
}

int Vfs::unlink(const char* path) {
    struct VfsMount* m = vfs_find_mount_(&self, path);
    if (m == (struct VfsMount*)0) { return -1; }
    if (m->ops.unlink == (void*)0) { return -1; }
    const char* rel = vfs_rel_path_(path, (const char*)m->mountpoint);
    unsafe { return m->ops.unlink(m->ctx, rel); }
    return -1;
}

int Vfs::mkdir(const char* path) {
    struct VfsMount* m = vfs_find_mount_(&self, path);
    if (m == (struct VfsMount*)0) { return -1; }
    if (m->ops.mkdir == (void*)0) { return -1; }
    const char* rel = vfs_rel_path_(path, (const char*)m->mountpoint);
    unsafe { return m->ops.mkdir(m->ctx, rel); }
    return -1;
}
