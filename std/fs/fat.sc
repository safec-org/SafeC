// SafeC Standard Library — FAT32 Read-Only Driver Implementation
#pragma once
#include "fat.h"

extern void* memset(void* p, int v, unsigned long n);
extern void* memcpy(void* d, const void* s, unsigned long n);
extern int   strncmp(const char* a, const char* b, unsigned long n);
extern unsigned long strlen(const char* s);

#define FAT32_EOC   0x0FFFFFF8U   // end-of-chain marker
#define FAT32_FREE  0x00000000U
#define FAT_ATTR_DIR  0x10

// Read a 32-bit little-endian value.
static unsigned int fat_read32_(const unsigned char* p) {
    unsafe {
        return (unsigned int)p[0] | ((unsigned int)p[1] << 8)
             | ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
    }
    return (unsigned int)0;
}

// Read a 16-bit little-endian value.
static unsigned short fat_read16_(const unsigned char* p) {
    unsafe { return (unsigned short)p[0] | ((unsigned short)p[1] << 8); }
    return (unsigned short)0;
}

int fat_init(&stack FatCtx ctx, &stack BlockDevice dev, unsigned long lba) {
    ctx.dev = dev;
    ctx.partition_lba = lba;
    unsigned char boot[512];
    if (dev.read(lba, (unsigned char*)boot, (unsigned long)1) != 0) { return -1; }
    // BPB parsing
    unsafe {
        ctx.bpb.bytes_per_sector     = fat_read16_(boot + 11);
        ctx.bpb.sectors_per_cluster  = boot[13];
        ctx.bpb.reserved_sectors     = fat_read16_(boot + 14);
        ctx.bpb.num_fats             = boot[16];
        ctx.bpb.root_entry_count     = fat_read16_(boot + 17);
        ctx.bpb.fat_size_16          = fat_read16_(boot + 22);
        ctx.bpb.fat_size_32          = fat_read32_(boot + 36);
        ctx.bpb.root_cluster         = fat_read32_(boot + 44);
    }
    // Validate FAT32 signature
    unsigned char sig0; unsigned char sig1;
    unsafe { sig0 = boot[510]; sig1 = boot[511]; }
    if (sig0 != (unsigned char)0x55 || sig1 != (unsigned char)0xAA) { return -1; }

    unsigned long fat_size = (unsigned long)ctx.bpb.fat_size_32;
    ctx.fat_lba  = lba + (unsigned long)ctx.bpb.reserved_sectors;
    ctx.data_lba = ctx.fat_lba + (unsigned long)ctx.bpb.num_fats * fat_size;
    ctx.root_cluster = (unsigned long)ctx.bpb.root_cluster;
    ctx.bytes_per_cluster = (unsigned long)ctx.bpb.bytes_per_sector
                          * (unsigned long)ctx.bpb.sectors_per_cluster;
    return 0;
}

// Convert cluster number to LBA.
static unsigned long fat_cluster_to_lba_(struct FatCtx* ctx, unsigned long cluster) {
    return ctx->data_lba + (cluster - (unsigned long)2)
           * (unsigned long)ctx->bpb.sectors_per_cluster;
}

int FatCtx::read_cluster(unsigned long cluster, unsigned char* buf) {
    unsigned long lba = fat_cluster_to_lba_(&self, cluster);
    return self.dev.read(lba, buf,
                         (unsigned long)self.bpb.sectors_per_cluster);
}

unsigned long FatCtx::follow_fat(unsigned long start_cluster, unsigned long idx) {
    unsigned long cluster = start_cluster;
    unsigned long i = (unsigned long)0;
    unsigned char fat_sect[512];
    while (i < idx) {
        unsigned long fat_byte_off = cluster * (unsigned long)4;
        unsigned long fat_sector   = fat_byte_off / (unsigned long)512;
        unsigned long fat_off_in   = fat_byte_off % (unsigned long)512;
        if (self.dev.read(self.fat_lba + fat_sector,
                          (unsigned char*)fat_sect, (unsigned long)1) != 0) {
            return (unsigned long)0;
        }
        unsigned int next;
        unsafe { next = fat_read32_((const unsigned char*)fat_sect + fat_off_in) & (unsigned int)0x0FFFFFFF; }
        if ((unsigned long)next >= (unsigned long)FAT32_EOC) { return (unsigned long)0; }
        cluster = (unsigned long)next;
        i = i + (unsigned long)1;
    }
    return cluster;
}

// Convert 8.3 directory-entry name to null-terminated string.
static void fat_name83_(const unsigned char* entry, char* out) {
    unsafe {
        int i = 0;
        int pos = 0;
        while (i < 8 && entry[i] != (unsigned char)0x20) {
            out[pos] = (char)entry[i];
            pos = pos + 1;
            i = i + 1;
        }
        if (entry[8] != (unsigned char)0x20) {
            out[pos] = '.'; pos = pos + 1;
            i = 8;
            while (i < 11 && entry[i] != (unsigned char)0x20) {
                out[pos] = (char)entry[i];
                pos = pos + 1;
                i = i + 1;
            }
        }
        out[pos] = (char)0;
    }
}

// Case-insensitive ASCII compare for FAT names.
static int fat_namecmp_(const char* a, const char* b) {
    unsafe {
        int i = 0;
        while (a[i] != (char)0 && b[i] != (char)0) {
            char ca = a[i];
            char cb = b[i];
            if (ca >= 'a' && ca <= 'z') { ca = (char)(ca - 32); }
            if (cb >= 'a' && cb <= 'z') { cb = (char)(cb - 32); }
            if (ca != cb) { return (int)ca - (int)cb; }
            i = i + 1;
        }
        return (int)a[i] - (int)b[i];
    }
    return 0;
}

// Walk one path component; returns cluster of found entry, 0 on miss.
// `is_dir_out` set to 1 if the entry is a directory.
static unsigned long fat_walk_component_(struct FatCtx* ctx,
                                          unsigned long dir_cluster,
                                          const char* component,
                                          int* is_dir_out) {
    unsigned char buf[4096]; // up to 8 sectors (assume ≤ 4K cluster)
    unsigned long cluster = dir_cluster;
    while (cluster != (unsigned long)0) {
        if (ctx->read_cluster(cluster, (unsigned char*)buf) != 0) { break; }
        unsigned long entries = ctx->bytes_per_cluster / (unsigned long)32;
        unsigned long e = (unsigned long)0;
        while (e < entries) {
            unsafe {
                unsigned char* entry = (unsigned char*)buf + e * (unsigned long)32;
                if (entry[0] == (unsigned char)0x00) { return (unsigned long)0; }
                if (entry[0] == (unsigned char)0xE5) { e = e + (unsigned long)1; continue; }
                unsigned char attr = entry[11];
                if ((attr & (unsigned char)0x0F) == (unsigned char)0x0F) {
                    // Long file name entry — skip
                    e = e + (unsigned long)1;
                    continue;
                }
                char name[13];
                fat_name83_(entry, (char*)name);
                if (fat_namecmp_((const char*)name, component) == 0) {
                    *is_dir_out = ((attr & (unsigned char)FAT_ATTR_DIR) != (unsigned char)0) ? 1 : 0;
                    unsigned long hi = (unsigned long)fat_read16_(entry + 20);
                    unsigned long lo = (unsigned long)fat_read16_(entry + 26);
                    return (hi << 16) | lo;
                }
            }
            e = e + (unsigned long)1;
        }
        cluster = ctx->follow_fat(cluster, (unsigned long)1);
    }
    return (unsigned long)0;
}

// Tokenise path into components; write into `parts`; return count.
static int fat_split_path_(const char* path, char parts[][VFS_MAX_NAME], int max_parts) {
    int count = 0;
    unsigned long i = (unsigned long)0;
    unsigned long plen = (unsigned long)0;
    unsafe {
        while (path[i] != (char)0 && count < max_parts) {
            if (path[i] == '/') {
                if (plen > (unsigned long)0) {
                    parts[count][plen] = (char)0;
                    count = count + 1;
                    plen = (unsigned long)0;
                }
            } else {
                if (plen < (unsigned long)VFS_MAX_NAME - (unsigned long)1) {
                    parts[count][plen] = path[i];
                    plen = plen + (unsigned long)1;
                }
            }
            i = i + (unsigned long)1;
        }
        if (plen > (unsigned long)0) {
            parts[count][plen] = (char)0;
            count = count + 1;
        }
    }
    return count;
}

// ── VfsOps callbacks ──────────────────────────────────────────────────────────

static int fat_vfs_open_(void* ctx, const char* path, int flags,
                          &stack VfsNode node_out) {
    struct FatCtx* fc = (struct FatCtx*)ctx;
    // Tokenise path.
    char parts[FAT_MAX_DEPTH][VFS_MAX_NAME];
    int n = fat_split_path_(path, (char(*)[VFS_MAX_NAME])parts, FAT_MAX_DEPTH);
    if (n == 0) {
        // Root directory.
        unsafe {
            node_out.inode = fc->root_cluster;
            node_out.type  = VFS_TYPE_DIR;
            node_out.size  = (unsigned long)0;
            node_out.name[0] = '/'; node_out.name[1] = (char)0;
        }
        return 0;
    }
    unsigned long cluster = fc->root_cluster;
    int i = 0;
    int is_dir = 1;
    while (i < n) {
        int d = 0;
        unsigned long next = fat_walk_component_(fc, cluster,
                                                  (const char*)parts[i], &d);
        if (next == (unsigned long)0) { return -1; }
        cluster = next;
        is_dir  = d;
        i = i + 1;
    }
    unsafe {
        node_out.inode = cluster;
        node_out.type  = (is_dir != 0) ? VFS_TYPE_DIR : VFS_TYPE_FILE;
        node_out.size  = (unsigned long)0;   // could parse from dir entry
        unsigned long nlen = strlen(parts[n - 1]);
        unsigned long k = (unsigned long)0;
        while (k < nlen && k < (unsigned long)VFS_MAX_NAME - (unsigned long)1) {
            node_out.name[k] = parts[n-1][k];
            k = k + (unsigned long)1;
        }
        node_out.name[k] = (char)0;
    }
    return 0;
}

static unsigned long fat_vfs_read_(void* ctx, unsigned long inode,
                                    unsigned long offset,
                                    unsigned char* buf, unsigned long len) {
    struct FatCtx* fc = (struct FatCtx*)ctx;
    unsigned long bpc = fc->bytes_per_cluster;
    unsigned long cluster_idx = offset / bpc;
    unsigned long off_in = offset % bpc;
    unsigned long cluster = fc->follow_fat(inode, cluster_idx);
    if (cluster == (unsigned long)0) { return (unsigned long)0; }

    unsigned char cbuf[4096];
    if (fc->read_cluster(cluster, (unsigned char*)cbuf) != 0) {
        return (unsigned long)0;
    }
    unsigned long avail = bpc - off_in;
    if (avail > len) { avail = len; }
    unsafe {
        unsigned long k = (unsigned long)0;
        while (k < avail) {
            buf[k] = cbuf[off_in + k];
            k = k + (unsigned long)1;
        }
    }
    return avail;
}

static int fat_vfs_readdir_(void* ctx, unsigned long inode, void* cb, void* user) {
    struct FatCtx* fc = (struct FatCtx*)ctx;
    int count = 0;
    unsigned long cluster = inode;
    unsigned char buf[4096];
    while (cluster != (unsigned long)0) {
        if (fc->read_cluster(cluster, (unsigned char*)buf) != 0) { break; }
        unsigned long entries = fc->bytes_per_cluster / (unsigned long)32;
        unsigned long e = (unsigned long)0;
        while (e < entries) {
            unsafe {
                unsigned char* entry = (unsigned char*)buf + e * (unsigned long)32;
                if (entry[0] == (unsigned char)0x00) { goto done_; }
                if (entry[0] == (unsigned char)0xE5) { e = e + (unsigned long)1; continue; }
                unsigned char attr = entry[11];
                if ((attr & (unsigned char)0x0F) == (unsigned char)0x0F) {
                    e = e + (unsigned long)1;
                    continue;
                }
                struct VfsNode node;
                char name[13];
                fat_name83_(entry, (char*)name);
                unsigned long nlen = strlen((const char*)name);
                unsigned long k = (unsigned long)0;
                while (k < nlen && k < (unsigned long)VFS_MAX_NAME - (unsigned long)1) {
                    node.name[k] = name[k]; k = k + (unsigned long)1;
                }
                node.name[k] = (char)0;
                node.type     = ((attr & (unsigned char)FAT_ATTR_DIR) != (unsigned char)0)
                                  ? VFS_TYPE_DIR : VFS_TYPE_FILE;
                node.size     = (unsigned long)fat_read32_(entry + 28);
                unsigned long hi = (unsigned long)fat_read16_(entry + 20);
                unsigned long lo = (unsigned long)fat_read16_(entry + 26);
                node.inode    = (hi << 16) | lo;
                node.fs_ctx   = ctx;
                void (*callback)(struct VfsNode*, void*) =
                    (void (*)(struct VfsNode*, void*))cb;
                callback(&node, user);
                count = count + 1;
            }
            e = e + (unsigned long)1;
        }
        cluster = fc->follow_fat(cluster, (unsigned long)1);
    }
    done_:
    return count;
}

struct VfsOps fat_ops() {
    struct VfsOps ops;
    unsafe {
        memset((void*)&ops, 0, sizeof(struct VfsOps));
        ops.open    = fat_vfs_open_;
        ops.read    = fat_vfs_read_;
        ops.readdir = fat_vfs_readdir_;
    }
    return ops;
}
