// SafeC Standard Library — ext2 Read-Only Filesystem Driver Implementation
#pragma once
#include <std/stddef.h>
#include <std/fs/ext.h>
#include <std/fs/block.h>

namespace std {

extern void* memset(void* p, int v, size_t n);
extern void* memcpy(void* d, const void* s, size_t n);
extern int   strncmp(const char* a, const char* b, size_t n);
extern size_t strlen(const char* s);

// ── Little-endian read helpers ────────────────────────────────────────────────

static unsigned int ext2_read32_(const unsigned char* p) {
    unsafe {
        return (unsigned int)p[0]
             | ((unsigned int)p[1] << 8)
             | ((unsigned int)p[2] << 16)
             | ((unsigned int)p[3] << 24);
    }
    return (unsigned int)0;
}

static unsigned short ext2_read16_(const unsigned char* p) {
    unsafe {
        return (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    }
    return (unsigned short)0;
}

// ── ext2_init ─────────────────────────────────────────────────────────────────
// Superblock lives at byte offset 1024 = sector 2 on a 512-byte device.

int ext2_init(&Ext2Ctx ctx, &BlockDevice dev) {
    ctx.dev = *dev;

    // Read two sectors starting at LBA 2 to get the 1024-byte superblock.
    unsigned char sb_buf[1024];
    if (dev.read((unsigned long)2, (&stack unsigned char)sb_buf, (unsigned long)2) != 0) {
        return -1;
    }

    // Parse superblock fields.
    unsafe {
        ctx.super.inodes_count     = ext2_read32_(sb_buf + 0);
        ctx.super.blocks_count     = ext2_read32_(sb_buf + 4);
        ctx.super.reserved_blocks  = ext2_read32_(sb_buf + 8);
        ctx.super.free_blocks      = ext2_read32_(sb_buf + 12);
        ctx.super.free_inodes      = ext2_read32_(sb_buf + 16);
        ctx.super.first_data_block = ext2_read32_(sb_buf + 20);
        ctx.super.log_block_size   = ext2_read32_(sb_buf + 24);
        ctx.super.log_frag_size    = ext2_read32_(sb_buf + 28);
        ctx.super.blocks_per_group = ext2_read32_(sb_buf + 32);
        ctx.super.frags_per_group  = ext2_read32_(sb_buf + 36);
        ctx.super.inodes_per_group = ext2_read32_(sb_buf + 40);
        ctx.super.mtime            = ext2_read32_(sb_buf + 44);
        ctx.super.wtime            = ext2_read32_(sb_buf + 48);
        ctx.super.mnt_count        = ext2_read16_(sb_buf + 52);
        ctx.super.max_mnt_count    = ext2_read16_(sb_buf + 54);
        ctx.super.magic            = ext2_read16_(sb_buf + 56);
        ctx.super.state            = ext2_read16_(sb_buf + 58);
        ctx.super.errors           = ext2_read16_(sb_buf + 60);
        ctx.super.minor_rev        = ext2_read16_(sb_buf + 62);
        ctx.super.lastcheck        = ext2_read32_(sb_buf + 64);
        ctx.super.checkinterval    = ext2_read32_(sb_buf + 68);
        ctx.super.creator_os       = ext2_read32_(sb_buf + 72);
        ctx.super.rev_level        = ext2_read32_(sb_buf + 76);
        // Revision 0 has fixed 128-byte inodes; dynamic-revision (>= 1)
        // filesystems — everything mke2fs creates — record the size here.
        ctx.super.inode_size = (ctx.super.rev_level == (unsigned int)0)
            ? (unsigned short)128 : ext2_read16_(sb_buf + 88);
    }

    // Validate magic.
    if ((unsigned int)ctx.super.magic != (unsigned int)EXT2_SUPER_MAGIC) { return -1; }

    // Compute derived values.
    ctx.block_size        = (unsigned long)EXT2_BLOCK_SIZE_BASE
                            << (unsigned long)ctx.super.log_block_size;
    // Blocks are staged in EXT2_MAX_BLOCK_SIZE-byte stack buffers.
    if (ctx.block_size > (unsigned long)EXT2_MAX_BLOCK_SIZE) { return -1; }
    if (ctx.super.inode_size < (unsigned short)128 ||
        (unsigned long)ctx.super.inode_size > ctx.block_size) { return -1; }
    ctx.inodes_per_group  = (unsigned long)ctx.super.inodes_per_group;
    ctx.blocks_per_group  = (unsigned long)ctx.super.blocks_per_group;

    // group_count = ceil(blocks_count / blocks_per_group)
    unsigned long bc  = (unsigned long)ctx.super.blocks_count;
    unsigned long bpg = ctx.blocks_per_group;
    ctx.group_count = (bc + bpg - (unsigned long)1) / bpg;

    return 0;
}

// ── Ext2Ctx::read_block ───────────────────────────────────────────────────────
// Convert block number to LBA (512-byte sectors) and read block_size bytes.

inline int Ext2Ctx::read_block(unsigned long block_no, &stack unsigned char buf) {
    unsigned long sectors_per_block = self.block_size / (unsigned long)512;
    unsigned long lba = block_no * sectors_per_block;
    return self.dev.read(lba, buf, sectors_per_block);
}

// ── Ext2Ctx::read_inode ───────────────────────────────────────────────────────
// Locate the group, read the inode table block, extract the 128-byte inode.

int Ext2Ctx::read_inode(unsigned long ino, &stack Ext2Inode inode_out) {
    if (ino < (unsigned long)1) { return -1; }
    unsigned long idx     = ino - (unsigned long)1;
    unsigned long group   = idx / self.inodes_per_group;
    unsigned long local   = idx % self.inodes_per_group;

    // Read the block group descriptor table.
    // BGDT is always in the block immediately after the superblock.
    unsigned long bgdt_block = (unsigned long)self.super.first_data_block + (unsigned long)1;
    unsigned char block_buf[EXT2_MAX_BLOCK_SIZE];
    if (self.read_block(bgdt_block, block_buf) != 0) { return -1; }

    // Each group descriptor is 32 bytes.
    unsigned long gd_off = group * (unsigned long)32;
    unsigned long inode_table_block;
    unsafe {
        inode_table_block = (unsigned long)ext2_read32_((const unsigned char*)block_buf + gd_off + 8);
    }

    // On-disk inode stride comes from the superblock (128 for rev 0, usually
    // 256 for rev 1+); only the first 128 bytes (the classic fields) are parsed.
    unsigned long inode_size   = (unsigned long)self.super.inode_size;
    unsigned long inodes_per_blk = self.block_size / inode_size;
    unsigned long inode_blk    = inode_table_block + local / inodes_per_blk;
    unsigned long inode_off    = (local % inodes_per_blk) * inode_size;

    unsigned char ibuf[EXT2_MAX_BLOCK_SIZE];
    if (self.read_block(inode_blk, ibuf) != 0) { return -1; }

    unsafe {
        unsigned char* p = (unsigned char*)ibuf + inode_off;
        inode_out.mode        = ext2_read16_(p + 0);
        inode_out.uid         = ext2_read16_(p + 2);
        inode_out.size_lo     = ext2_read32_(p + 4);
        inode_out.atime       = ext2_read32_(p + 8);
        inode_out.ctime       = ext2_read32_(p + 12);
        inode_out.mtime       = ext2_read32_(p + 16);
        inode_out.dtime       = ext2_read32_(p + 20);
        inode_out.gid         = ext2_read16_(p + 24);
        inode_out.links_count = ext2_read16_(p + 26);
        inode_out.blocks_count= ext2_read32_(p + 28);
        inode_out.flags       = ext2_read32_(p + 32);
        inode_out.reserved1   = ext2_read32_(p + 36);
        int bi = 0;
        while (bi < 15) {
            inode_out.block[bi] = ext2_read32_(p + 40 + (unsigned long)bi * (unsigned long)4);
            bi = bi + 1;
        }
    }
    return 0;
}

// Helper: return first_data_block from superblock inside a method where 'ctx'
// refers to the struct via self.
static unsigned long ctx_super_first_data_block_(struct Ext2Ctx ctx) {
    return (unsigned long)ctx.super.first_data_block;
}

// ── Ext2Ctx::read_file ────────────────────────────────────────────────────────
// Reads up to `len` bytes from file inode `ino` at byte `offset`.
// Only handles the 12 direct blocks (block[0..11]).

unsigned long Ext2Ctx::read_file(unsigned long ino, unsigned long offset,
                                  unsigned char* buf, unsigned long len) {
    struct Ext2Inode inode;
    if (self.read_inode(ino, &inode) != 0) { return (unsigned long)0; }

    unsigned long file_size = (unsigned long)inode.size_lo;
    if (offset >= file_size) { return (unsigned long)0; }
    unsigned long avail = file_size - offset;
    if (avail < len) { len = avail; }

    unsigned long bs        = self.block_size;
    unsigned long total_read= (unsigned long)0;
    unsigned char blk_buf[EXT2_MAX_BLOCK_SIZE];

    while (total_read < len) {
        unsigned long cur_off   = offset + total_read;
        unsigned long blk_idx   = cur_off / bs;
        unsigned long blk_off   = cur_off % bs;

        // Only direct blocks supported (indices 0–11).
        if (blk_idx >= (unsigned long)12) { break; }

        unsigned long blk_no;
        unsafe { blk_no = (unsigned long)inode.block[blk_idx]; }
        if (blk_no == (unsigned long)0) { break; }

        if (self.read_block(blk_no, blk_buf) != 0) { break; }

        unsigned long chunk = bs - blk_off;
        unsigned long remain = len - total_read;
        if (chunk > remain) { chunk = remain; }

        unsafe {
            unsigned long k = (unsigned long)0;
            while (k < chunk) {
                buf[total_read + k] = blk_buf[blk_off + k];
                k = k + (unsigned long)1;
            }
        }
        total_read = total_read + chunk;
    }
    return total_read;
}

// ── Directory entry iteration ─────────────────────────────────────────────────
// Each directory entry: u32 inode, u16 rec_len, u8 name_len, u8 file_type,
//                       char name[name_len]  (not null-terminated in disk)

// Walk a directory for a single path component.
// Returns the inode of the matched entry, or 0 if not found.
// Sets *type_out to 1 if the entry is a directory, 0 otherwise.
static unsigned long ext2_dir_find_(struct Ext2Ctx* ctx, unsigned long dir_ino,
                                     const char* component, unsigned long comp_len,
                                     int* type_out) {
    struct Ext2Inode dir_inode;
    if (ctx->read_inode(dir_ino, &dir_inode) != 0) { return (unsigned long)0; }

    unsigned long dir_size = (unsigned long)dir_inode.size_lo;
    unsigned long bs;
    unsafe { bs = ctx->block_size; }
    unsigned long blk_idx  = (unsigned long)0;
    unsigned long read_pos = (unsigned long)0;
    unsigned char blk_buf[EXT2_MAX_BLOCK_SIZE];

    while (read_pos < dir_size && blk_idx < (unsigned long)12) {
        unsigned long blk_no;
        unsafe { blk_no = (unsigned long)dir_inode.block[blk_idx]; }
        if (blk_no == (unsigned long)0) { break; }
        if (ctx->read_block(blk_no, blk_buf) != 0) { break; }

        unsigned long blk_limit = bs;
        if (read_pos + bs > dir_size) { blk_limit = dir_size - read_pos; }
        unsigned long pos_in_blk = (unsigned long)0;

        while (pos_in_blk < blk_limit) {
            unsafe {
                unsigned char* entry = (unsigned char*)blk_buf + pos_in_blk;
                unsigned int   e_ino     = ext2_read32_(entry + 0);
                unsigned short e_rec_len = ext2_read16_(entry + 4);
                unsigned char  e_name_len= entry[6];
                unsigned char  e_ftype   = entry[7];

                if (e_rec_len == (unsigned short)0) { goto blk_done_; }

                if (e_ino != (unsigned int)0
                    && (unsigned long)e_name_len == comp_len) {
                    // Compare name (not null-terminated on disk).
                    if (strncmp((const char*)(entry + 8), component, comp_len) == 0) {
                        *type_out = (e_ftype == (unsigned char)2) ? 1 : 0;
                        return (unsigned long)e_ino;
                    }
                }
                pos_in_blk = pos_in_blk + (unsigned long)e_rec_len;
            }
        }
        blk_done_:
        read_pos = read_pos + bs;
        blk_idx  = blk_idx + (unsigned long)1;
    }
    return (unsigned long)0;
}

// ── Path tokeniser ────────────────────────────────────────────────────────────

#define EXT2_MAX_DEPTH  16

// Split `path` on '/'; store each component pointer+length into parallel arrays.
// Returns number of components.
static int ext2_split_path_(const char* path,
                             const char* comp_ptrs[EXT2_MAX_DEPTH],
                             unsigned long comp_lens[EXT2_MAX_DEPTH]) {
    int count = 0;
    unsigned long i = (unsigned long)0;
    unsafe {
        while (path[i] != (char)0 && count < EXT2_MAX_DEPTH) {
            // Skip leading slashes.
            while (path[i] == '/') { i = i + (unsigned long)1; }
            if (path[i] == (char)0) { break; }
            // Component start.
            comp_ptrs[count] = path + i;
            unsigned long start = i;
            while (path[i] != (char)0 && path[i] != '/') {
                i = i + (unsigned long)1;
            }
            comp_lens[count] = i - start;
            count = count + 1;
        }
    }
    return count;
}

// ── VfsOps callbacks ──────────────────────────────────────────────────────────

static int ext2_vfs_open_(void* ctx, const char* path, int flags,
                           &stack VfsNode node_out) {
    struct Ext2Ctx* ec;
    unsafe { ec = (struct Ext2Ctx*)ctx; }

    const char* comp_ptrs[EXT2_MAX_DEPTH];
    unsigned long comp_lens[EXT2_MAX_DEPTH];
    int n = ext2_split_path_(path, comp_ptrs, comp_lens);

    if (n == 0) {
        // Root directory.
        unsafe {
            node_out.inode    = (unsigned long)EXT2_ROOT_INODE;
            node_out.type     = VFS_TYPE_DIR;
            node_out.size     = (unsigned long)0;
            node_out.name[0]  = '/';
            node_out.name[1]  = (char)0;
            node_out.fs_ctx   = ctx;
        }
        return 0;
    }

    unsigned long cur_ino = (unsigned long)EXT2_ROOT_INODE;
    int i = 0;
    int is_dir = 1;
    while (i < n) {
        unsafe {
            int tp = 0;
            unsigned long next = ext2_dir_find_(ec, cur_ino,
                                                comp_ptrs[i], comp_lens[i], (int*)&tp);
            if (next == (unsigned long)0) { return -1; }
            cur_ino = next;
            is_dir  = tp;
        }
        i = i + 1;
    }

    // Stat the found inode for size.
    struct Ext2Inode inode;
    if (ec->read_inode(cur_ino, &inode) != 0) { return -1; }

    unsafe {
        node_out.inode  = cur_ino;
        node_out.type   = (is_dir != 0) ? VFS_TYPE_DIR : VFS_TYPE_FILE;
        node_out.size   = (unsigned long)inode.size_lo;
        node_out.fs_ctx = ctx;
        // Copy last component as name.
        unsigned long nlen = comp_lens[n - 1];
        unsigned long k = (unsigned long)0;
        while (k < nlen && k < (unsigned long)VFS_MAX_NAME - (unsigned long)1) {
            node_out.name[k] = comp_ptrs[n - 1][k];
            k = k + (unsigned long)1;
        }
        node_out.name[k] = (char)0;
    }
    return 0;
}

static unsigned long ext2_vfs_read_(void* ctx, unsigned long inode,
                                     unsigned long offset,
                                     unsigned char* buf, unsigned long len) {
    struct Ext2Ctx* ec;
    unsafe { ec = (struct Ext2Ctx*)ctx; }
    return ec->read_file(inode, offset, buf, len);
}

static int ext2_vfs_readdir_(void* ctx, unsigned long dir_ino,
                              void* cb, void* user) {
    struct Ext2Ctx* ec;
    unsafe { ec = (struct Ext2Ctx*)ctx; }

    struct Ext2Inode dir_inode;
    if (ec->read_inode(dir_ino, &dir_inode) != 0) { return -1; }

    unsigned long dir_size = (unsigned long)dir_inode.size_lo;
    unsigned long bs;
    unsafe { bs = ec->block_size; }
    unsigned long blk_idx  = (unsigned long)0;
    unsigned long read_pos = (unsigned long)0;
    unsigned char blk_buf[EXT2_MAX_BLOCK_SIZE];
    int count = 0;

    while (read_pos < dir_size && blk_idx < (unsigned long)12) {
        unsigned long blk_no;
        unsafe { blk_no = (unsigned long)dir_inode.block[blk_idx]; }
        if (blk_no == (unsigned long)0) { break; }
        if (ec->read_block(blk_no, blk_buf) != 0) { break; }

        unsigned long blk_limit = bs;
        if (read_pos + bs > dir_size) { blk_limit = dir_size - read_pos; }
        unsigned long pos_in_blk = (unsigned long)0;

        while (pos_in_blk < blk_limit) {
            unsafe {
                unsigned char* entry    = (unsigned char*)blk_buf + pos_in_blk;
                unsigned int   e_ino    = ext2_read32_(entry + 0);
                unsigned short e_rec   = ext2_read16_(entry + 4);
                unsigned char  e_nlen  = entry[6];
                unsigned char  e_ftype = entry[7];

                if (e_rec == (unsigned short)0) { goto blk_end_; }

                if (e_ino != (unsigned int)0 && e_nlen > (unsigned char)0) {
                    struct VfsNode node;
                    // Copy name.
                    unsigned long k = (unsigned long)0;
                    while (k < (unsigned long)e_nlen
                           && k < (unsigned long)VFS_MAX_NAME - (unsigned long)1) {
                        node.name[k] = (char)entry[(unsigned long)8 + k];
                        k = k + (unsigned long)1;
                    }
                    node.name[k]  = (char)0;
                    node.type     = (e_ftype == (unsigned char)2) ? VFS_TYPE_DIR
                                                                   : VFS_TYPE_FILE;
                    node.inode    = (unsigned long)e_ino;
                    node.fs_ctx   = ctx;

                    // Stat to get size for files.
                    struct Ext2Inode child_inode;
                    if (ec->read_inode((unsigned long)e_ino, &child_inode) == 0) {
                        node.size = (unsigned long)child_inode.size_lo;
                    } else {
                        node.size = (unsigned long)0;
                    }

                    fn void(struct VfsNode*, void*) callback =
                        (fn void(struct VfsNode*, void*))cb;
                    callback((struct VfsNode*)&node, user);
                    count = count + 1;
                }
                pos_in_blk = pos_in_blk + (unsigned long)e_rec;
            }
        }
        blk_end_:
        read_pos = read_pos + bs;
        blk_idx  = blk_idx + (unsigned long)1;
    }
    return count;
}

// ── ext2_ops ──────────────────────────────────────────────────────────────────

inline struct VfsOps ext2_ops() {
    struct VfsOps ops;
    unsafe {
        memset((void*)&ops, 0, sizeof(struct VfsOps));
        ops.open    = ext2_vfs_open_;
        ops.read    = ext2_vfs_read_;
        ops.readdir = ext2_vfs_readdir_;
    }
    return ops;
}

} // namespace std
