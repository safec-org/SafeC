// SafeC Standard Library — tmpfs Implementation
#pragma once
#include "tmpfs.h"

extern void* memset(void* p, int v, unsigned long n);
extern void* memcpy(void* d, const void* s, unsigned long n);
extern int   strcmp(const char* a, const char* b);
extern int   strncmp(const char* a, const char* b, unsigned long n);
extern unsigned long strlen(const char* s);
extern void* strncpy(char* d, const char* s, unsigned long n);

void tmpfs_init(&stack TmpfsCtx ctx) {
    unsafe { memset((void*)&ctx, 0, sizeof(struct TmpfsCtx)); }
    // Create root inode at index 0.
    ctx.inodes[0].used   = 1;
    ctx.inodes[0].type   = VFS_TYPE_DIR;
    ctx.inodes[0].parent = (unsigned long)0;
    ctx.inodes[0].size   = (unsigned long)0;
    unsafe {
        ctx.inodes[0].name[0] = '/';
        ctx.inodes[0].name[1] = (char)0;
    }
    ctx.inode_count = 1;
    ctx.data_used   = (unsigned long)0;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// Tokenise a path and resolve down from root; returns inode index or -1.
static int tmpfs_resolve_(struct TmpfsCtx* tc, const char* path) {
    if (path[0] == '/' && path[1] == (char)0) { return 0; }
    // Walk components.
    unsigned long cur = (unsigned long)0;
    unsigned long i   = (unsigned long)0;
    while (path[i] == '/') { i = i + (unsigned long)1; }
    while (path[i] != (char)0) {
        // Extract component.
        char comp[VFS_MAX_NAME];
        unsigned long clen = (unsigned long)0;
        unsafe {
            while (path[i] != (char)0 && path[i] != '/' &&
                   clen < (unsigned long)VFS_MAX_NAME - (unsigned long)1) {
                comp[clen] = path[i];
                clen = clen + (unsigned long)1;
                i = i + (unsigned long)1;
            }
            comp[clen] = (char)0;
            while (path[i] == '/') { i = i + (unsigned long)1; }
        }
        // Find child with this name under `cur`.
        int found = -1;
        int j = 0;
        while (j < TMPFS_MAX_FILES) {
            if (tc->inodes[j].used != 0 &&
                (unsigned long)tc->inodes[j].parent == cur) {
                int eq = 0;
                unsafe { eq = (strcmp(tc->inodes[j].name, (const char*)comp) == 0); }
                if (eq != 0) { found = j; break; }
            }
            j = j + 1;
        }
        if (found < 0) { return -1; }
        cur = (unsigned long)found;
    }
    return (int)cur;
}

// Allocate a fresh inode; returns index or -1.
static int tmpfs_alloc_inode_(struct TmpfsCtx* tc) {
    int i = 0;
    while (i < TMPFS_MAX_FILES) {
        if (tc->inodes[i].used == 0) { return i; }
        i = i + 1;
    }
    return -1;
}

// Split path into parent path + leaf component.
static void tmpfs_split_path_(const char* path,
                               char* parent_out, char* leaf_out) {
    unsafe {
        unsigned long len = strlen(path);
        // Find last slash.
        long last = -1;
        long k = (long)len - 1L;
        while (k >= 0L) {
            if (path[k] == '/') { last = k; break; }
            k = k - 1L;
        }
        if (last <= 0L) {
            parent_out[0] = '/'; parent_out[1] = (char)0;
            strncpy(leaf_out, path + (last + 1L), (unsigned long)VFS_MAX_NAME - 1);
            leaf_out[VFS_MAX_NAME - 1] = (char)0;
        } else {
            strncpy(parent_out, path, (unsigned long)last);
            parent_out[last] = (char)0;
            strncpy(leaf_out, path + last + 1L,
                    (unsigned long)VFS_MAX_NAME - 1);
            leaf_out[VFS_MAX_NAME - 1] = (char)0;
        }
    }
}

// ── VfsOps callbacks ──────────────────────────────────────────────────────────

static int tmpfs_vfs_open_(void* ctx, const char* path, int flags,
                            &stack VfsNode node_out) {
    struct TmpfsCtx* tc = (struct TmpfsCtx*)ctx;
    int idx = tmpfs_resolve_(tc, path);
    if (idx < 0) {
        // Create if VFS_O_CREATE.
        if ((flags & VFS_O_CREATE) == 0) { return -1; }
        char parent[VFS_MAX_PATH];
        char leaf[VFS_MAX_NAME];
        tmpfs_split_path_(path, (char*)parent, (char*)leaf);
        int pidx = tmpfs_resolve_(tc, (const char*)parent);
        if (pidx < 0) { return -1; }
        int new_idx = tmpfs_alloc_inode_(tc);
        if (new_idx < 0) { return -1; }
        tc->inodes[new_idx].used     = 1;
        tc->inodes[new_idx].type     = VFS_TYPE_FILE;
        tc->inodes[new_idx].parent   = (unsigned long)pidx;
        tc->inodes[new_idx].size     = (unsigned long)0;
        tc->inodes[new_idx].data_off = tc->data_used;
        unsafe {
            strncpy(tc->inodes[new_idx].name, (const char*)leaf,
                    (unsigned long)VFS_MAX_NAME - 1);
            tc->inodes[new_idx].name[VFS_MAX_NAME - 1] = (char)0;
        }
        tc->inode_count = tc->inode_count + 1;
        idx = new_idx;
    }
    unsafe {
        unsigned long nlen = strlen(tc->inodes[idx].name);
        unsigned long k = (unsigned long)0;
        while (k < nlen && k < (unsigned long)VFS_MAX_NAME - (unsigned long)1) {
            node_out.name[k] = tc->inodes[idx].name[k];
            k = k + (unsigned long)1;
        }
        node_out.name[k] = (char)0;
    }
    node_out.type  = tc->inodes[idx].type;
    node_out.size  = tc->inodes[idx].size;
    node_out.inode = (unsigned long)idx;
    return 0;
}

static int tmpfs_vfs_unlink_(void* ctx, const char* path) {
    struct TmpfsCtx* tc = (struct TmpfsCtx*)ctx;
    int idx = tmpfs_resolve_(tc, path);
    if (idx < 0) { return -1; }
    tc->inodes[idx].used = 0;
    tc->inode_count = tc->inode_count - 1;
    return 0;
}

static int tmpfs_vfs_mkdir_(void* ctx, const char* path) {
    struct TmpfsCtx* tc = (struct TmpfsCtx*)ctx;
    char parent[VFS_MAX_PATH];
    char leaf[VFS_MAX_NAME];
    tmpfs_split_path_(path, (char*)parent, (char*)leaf);
    int pidx = tmpfs_resolve_(tc, (const char*)parent);
    if (pidx < 0) { return -1; }
    int new_idx = tmpfs_alloc_inode_(tc);
    if (new_idx < 0) { return -1; }
    tc->inodes[new_idx].used   = 1;
    tc->inodes[new_idx].type   = VFS_TYPE_DIR;
    tc->inodes[new_idx].parent = (unsigned long)pidx;
    tc->inodes[new_idx].size   = (unsigned long)0;
    unsafe {
        strncpy(tc->inodes[new_idx].name, (const char*)leaf,
                (unsigned long)VFS_MAX_NAME - 1);
        tc->inodes[new_idx].name[VFS_MAX_NAME - 1] = (char)0;
    }
    tc->inode_count = tc->inode_count + 1;
    return 0;
}

static unsigned long tmpfs_vfs_read_(void* ctx, unsigned long inode,
                                      unsigned long offset,
                                      unsigned char* buf, unsigned long len) {
    struct TmpfsCtx* tc = (struct TmpfsCtx*)ctx;
    int idx = (int)inode;
    if (idx < 0 || idx >= TMPFS_MAX_FILES) { return (unsigned long)0; }
    if (tc->inodes[idx].used == 0 || tc->inodes[idx].type != VFS_TYPE_FILE) {
        return (unsigned long)0;
    }
    unsigned long fsize = tc->inodes[idx].size;
    if (offset >= fsize) { return (unsigned long)0; }
    unsigned long avail = fsize - offset;
    if (avail > len) { avail = len; }
    unsafe {
        unsigned long doff = tc->inodes[idx].data_off + offset;
        unsigned long k = (unsigned long)0;
        while (k < avail) {
            buf[k] = tc->data[doff + k];
            k = k + (unsigned long)1;
        }
    }
    return avail;
}

static unsigned long tmpfs_vfs_write_(void* ctx, unsigned long inode,
                                       unsigned long offset,
                                       const unsigned char* buf, unsigned long len) {
    struct TmpfsCtx* tc = (struct TmpfsCtx*)ctx;
    int idx = (int)inode;
    if (idx < 0 || idx >= TMPFS_MAX_FILES) { return (unsigned long)0; }
    if (tc->inodes[idx].used == 0 || tc->inodes[idx].type != VFS_TYPE_FILE) {
        return (unsigned long)0;
    }
    unsigned long doff = tc->inodes[idx].data_off + offset;
    unsigned long end  = doff + len;
    if (end > (unsigned long)TMPFS_MAX_DATA) {
        len = (unsigned long)TMPFS_MAX_DATA - doff;
    }
    unsafe {
        unsigned long k = (unsigned long)0;
        while (k < len) {
            tc->data[doff + k] = buf[k];
            k = k + (unsigned long)1;
        }
    }
    unsigned long new_size = offset + len;
    if (new_size > tc->inodes[idx].size) { tc->inodes[idx].size = new_size; }
    if (end > tc->data_used) { tc->data_used = end; }
    return len;
}

static int tmpfs_vfs_readdir_(void* ctx, unsigned long inode,
                               void* cb, void* user) {
    struct TmpfsCtx* tc = (struct TmpfsCtx*)ctx;
    int count = 0;
    int j = 0;
    while (j < TMPFS_MAX_FILES) {
        if (tc->inodes[j].used != 0 &&
            (unsigned long)tc->inodes[j].parent == inode &&
            j != (int)inode) {
            struct VfsNode node;
            unsafe {
                unsigned long nlen = strlen(tc->inodes[j].name);
                unsigned long k = (unsigned long)0;
                while (k < nlen && k < (unsigned long)VFS_MAX_NAME - (unsigned long)1) {
                    node.name[k] = tc->inodes[j].name[k];
                    k = k + (unsigned long)1;
                }
                node.name[k] = (char)0;
            }
            node.type  = tc->inodes[j].type;
            node.size  = tc->inodes[j].size;
            node.inode = (unsigned long)j;
            node.fs_ctx = ctx;
            unsafe {
                void (*callback)(struct VfsNode*, void*) =
                    (void (*)(struct VfsNode*, void*))cb;
                callback(&node, user);
            }
            count = count + 1;
        }
        j = j + 1;
    }
    return count;
}

struct VfsOps tmpfs_ops() {
    struct VfsOps ops;
    unsafe {
        memset((void*)&ops, 0, sizeof(struct VfsOps));
        ops.open    = tmpfs_vfs_open_;
        ops.unlink  = tmpfs_vfs_unlink_;
        ops.mkdir   = tmpfs_vfs_mkdir_;
        ops.read    = tmpfs_vfs_read_;
        ops.write   = tmpfs_vfs_write_;
        ops.readdir = tmpfs_vfs_readdir_;
    }
    return ops;
}
