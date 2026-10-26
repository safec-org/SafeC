// std::fs — vfs + tmpfs, block device, MBR partitions, FAT32, ext2.
// FAT32/ext2 run against real images built by fixtures/make_fs_images.sh
// (mkfs.vfat + mke2fs -d); those two cases skip if the tools are missing.
#include <std/test/test.h>
#include <std/mem.h>
#include <std/panic.h>
#include <std/str.h>
#include <std/sys.h>
#include <std/io.h>
#include <std/io_file.h>
#include <std/fs/vfs.h>
#include <std/fs/tmpfs.h>
#include <std/fs/block.h>
#include <std/fs/partition.h>
#include <std/fs/fat.h>
#include <std/fs/ext.h>

// ── In-memory block device: a BlockDriver ────────────────────────────────────
struct MemDisk {
    ?&heap unsigned char data;
    unsigned long   size;

    int read(unsigned long lba, []unsigned char buf);
    int write(unsigned long lba, []const unsigned char buf);
};

int MemDisk::read(unsigned long lba, []unsigned char buf) {
    unsigned long off = lba * 512UL;
    unsigned long n = buf.len;
    if (off + n > self.size || self.data.is_null()) return -1;
    unsafe { safe_memcpy(buf, ((const unsigned char*)self.data)[off..off + n]); }
    return 0;
}

int MemDisk::write(unsigned long lba, []const unsigned char buf) {
    unsigned long off = lba * 512UL;
    unsigned long n = buf.len;
    if (off + n > self.size || self.data.is_null()) return -1;
    unsafe { safe_memcpy(((unsigned char*)self.data)[off..off + n], buf); }
    return 0;
}

static struct BlockDevice memdisk_device(&MemDisk d) {
    return BlockDevice::new(d, d.size / 512UL, 512UL);
}

// FsRead, straight on a driver (no VFS): the root directory's entry count.
static void count_any(&VfsNode n, ?&int seen) {
    match (seen) { case some(s): *s = *s + 1; case null: { } }
}
generic<F: FsRead>
int root_entries(&F fs) {
    struct VfsNode root;
    if (fs.open("/", VFS_O_READ, &root) != 0) return -1;
    int seen = 0;
    fs.readdir(root.inode, count_any, (?&int)&seen);
    return seen;
}

// A directory entry handed to a readdir callback reads like an opened node.
static void read_entry(&VfsNode n, &int bytes) {
    unsigned char buf[16];
    *bytes = *bytes + (int)n.read(0UL, buf[0..16]);
}

// Loads a whole file into a MemDisk; data stays null if it can't be read.
static struct MemDisk memdisk_load(const &char path) {
    struct MemDisk d;
    d.data = null;
    d.size = 0UL;
    ?&File fq = File::open(path, "rb");
    if (fq.is_null()) return d;
    &File f = std::expect_ref(fq, "memdisk file");
    long long sz = f.size();
    ?&heap unsigned char buf = alloc((unsigned long)sz);
    match (buf) {
        case some(b): {
            long long got;
            unsafe { got = f.read(((unsigned char*)b)[0..(unsigned long)sz]); }
            if (got == sz) {
                d.data = buf;
                d.size = (unsigned long)sz;
            }
        }
        case null: { }
    }
    f.close();
    return d;
}

static int g_fixtures_ok = -1;

static int fixtures_ready() {
    if (g_fixtures_ok < 0) {
        // safeguard test runs from the package root; also allow the repo root.
        int rc = sys_system("bash fixtures/make_fs_images.sh /tmp/safec_stdlib_fs_fixtures >/dev/null 2>&1 || "
                            "bash tests/stdlib/fixtures/make_fs_images.sh /tmp/safec_stdlib_fs_fixtures >/dev/null 2>&1");
        g_fixtures_ok = rc == 0 ? 1 : 0;
        if (g_fixtures_ok == 0) println("    (skipping: mkfs.vfat/mke2fs/python3 not available)");
    }
    return g_fixtures_ok;
}

static void count_entry(&VfsNode n, &int seen) {
    unsafe { *seen = *seen + 1; }
}

// ── tmpfs through the VFS ────────────────────────────────────────────────────
void t_tmpfs_vfs() {
    unsafe {
        ?&heap TmpfsCtx cq = alloc(sizeof(struct TmpfsCtx));
        ASSERT_FALSE(cq.is_null());
        if (cq.is_null()) return;
        &TmpfsCtx ctx = std::expect_ref(cq, "cq is null");
        ctx.init();
        vfs_init();
        ASSERT_EQ(vfs_root.mount("/tmp", ctx), 0);
        ASSERT_EQ(vfs_root.mkdir("/tmp/dir"), 0);
        struct VfsNode node;
        ASSERT_EQ(vfs_root.open("/tmp/dir/hello.txt", VFS_O_CREATE | VFS_O_WRITE, &node), 0);
        unsigned char msg[12];
        unsafe { str_copy(((char*)msg)[0..12], "hello tmpfs"); }
        ASSERT_EQ(node.write(0UL, msg[0..11]), 11);
        struct VfsNode again;
        ASSERT_EQ(vfs_root.open("/tmp/dir/hello.txt", VFS_O_READ, &again), 0);
        ASSERT_EQ(again.size, 11);
        unsigned char back[16];
        safe_memset(back[..], 0);
        ASSERT_EQ(again.read(6UL, back[0..5]), 5);
        ASSERT_STR_EQ((const char*)back, "tmpfs");
        struct VfsNode dir;
        ASSERT_EQ(vfs_root.open("/tmp/dir", VFS_O_READ, &dir), 0);
        int seen = 0;
        dir.readdir(count_entry, (&int)&seen);
        ASSERT_EQ(seen, 1);
        int bytes = 0;
        dir.readdir(read_entry, (&int)&bytes);      // entries carry their mount
        ASSERT_EQ(bytes, 11);
        ASSERT_EQ(root_entries(ctx), 1);            // FsRead without the VFS: "dir"
        // read-only mount of the same driver: writes are refused
        ASSERT_EQ(vfs_root.mount_ro("/ro", ctx), 0);
        ASSERT_NE(vfs_root.mkdir("/ro/other"), 0);
        struct VfsNode ro;
        ASSERT_EQ(vfs_root.open("/ro/dir/hello.txt", VFS_O_READ, &ro), 0);
        ASSERT_EQ(ro.write(0UL, msg[0..1]), 0);
        ASSERT_EQ(vfs_root.unmount("/ro"), 0);
        ASSERT_EQ(vfs_root.unlink("/tmp/dir/hello.txt"), 0);
        ASSERT_NE(vfs_root.open("/tmp/dir/hello.txt", VFS_O_READ, &again), 0);
        ASSERT_NE(vfs_root.open("/nowhere/x", VFS_O_READ, &again), 0);
        ASSERT_EQ(vfs_root.unmount("/tmp"), 0);
        dealloc(cq);
    }
}

// ── MBR partition table ──────────────────────────────────────────────────────
void t_partition() {
    unsafe {
        ?&heap unsigned char img = alloc_zeroed(4096UL);
        ASSERT_FALSE(img.is_null());
        if (img.is_null()) return;
        &unsigned char m = std::expect_ref(img, "img is null");
        // Entry 0: bootable FAT32-LBA at LBA 2048, 100000 sectors.
        m[446] = (unsigned char)0x80;
        m[446 + 4] = (unsigned char)PART_TYPE_FAT32X;
        m[446 + 8] = (unsigned char)0x00; m[446 + 9] = (unsigned char)0x08;
        m[446 + 12] = (unsigned char)0xA0; m[446 + 13] = (unsigned char)0x86; m[446 + 14] = (unsigned char)0x01;
        // Entry 1: Linux at LBA 200000.
        m[462 + 4] = (unsigned char)PART_TYPE_LINUX;
        m[462 + 8] = (unsigned char)0x40; m[462 + 9] = (unsigned char)0x0D; m[462 + 10] = (unsigned char)0x03;
        m[462 + 12] = (unsigned char)0x10;
        m[510] = (unsigned char)0x55; m[511] = (unsigned char)0xAA;
        struct MemDisk d;
        d.data = img;
        d.size = 4096UL;
        struct BlockDevice dev = memdisk_device(&d);
        ASSERT_TRUE(dev.valid());
        struct PartTable pt;
        ASSERT_EQ(pt.read(&dev), 0);
        ASSERT_EQ(pt.count, 2);
        ASSERT_EQ(pt.entries[0].lba_start, 2048);
        ASSERT_EQ(pt.entries[0].sector_count, 100000);
        ASSERT_EQ(pt.entries[0].type, PART_TYPE_FAT32X);
        ASSERT_EQ(pt.entries[1].lba_start, 200000);
        m[511] = (unsigned char)0;
        ASSERT_NE(pt.read(&dev), 0);   // bad signature
        dealloc(img);
    }
}

// Reads a whole file from a mounted filesystem through the VFS.
static unsigned long read_all(&VfsNode node, []unsigned char out) {
    unsigned long cap = out.len;
    unsigned long total = 0UL;
    while (total < cap) {
        unsigned char chunk[512];
        unsigned long want = cap - total;
        if (want > 512UL) want = 512UL;
        unsigned long got = node.read(total, chunk[0..want]);
        if (got == 0UL) break;
        safe_memcpy(out[total..total + got], chunk[0..got]);
        total = total + got;
    }
    return total;
}

static int hello_matches(&unsigned char data, unsigned long n) {
    if (n != 1300UL) return 0;
    unsigned long i = 0UL;
    while (i < n) {
        unsafe { if (data[i] != (unsigned char)(65 + (int)(i % 26UL))) return 0; }
        i = i + 1UL;
    }
    return 1;
}

void t_fat32() {
    if (!fixtures_ready()) return;
    unsafe {
        struct MemDisk d = memdisk_load("/tmp/safec_stdlib_fs_fixtures/fat32.img");
        ASSERT_FALSE(d.data.is_null());
        if (d.data.is_null()) return;
        struct BlockDevice dev = memdisk_device(&d);
        struct FatCtx fc;
        ASSERT_EQ(fc.init(&dev, 0UL), 0);
        ASSERT_EQ(fc.bytes_per_cluster, 512);
        vfs_init();
        ASSERT_EQ(vfs_root.mount_ro("/fat", (&FatCtx)&fc), 0);
        struct VfsNode f;
        ASSERT_EQ(vfs_root.open("/fat/hello.txt", VFS_O_READ, &f), 0);   // 8.3 names match case-insensitively
        ASSERT_EQ(f.type, VFS_TYPE_FILE);
        ASSERT_EQ(f.size, 1300);
        unsigned char buf[2048];
        unsigned long n = read_all(&f, buf[0..f.size]);
        ASSERT_TRUE(hello_matches(buf, n));
        struct VfsNode a;
        ASSERT_EQ(vfs_root.open("/fat/SUB/A.TXT", VFS_O_READ, &a), 0);
        safe_memset(buf[0..64], 0);
        read_all(&a, buf[0..a.size]);
        ASSERT_STR_EQ((const char*)buf, "nested file\n");
        struct VfsNode root;
        ASSERT_EQ(vfs_root.open("/fat", VFS_O_READ, &root), 0);
        int seen = 0;
        root.readdir(count_entry, (&int)&seen);
        ASSERT_EQ(seen, 3);   // volume label + HELLO.TXT + SUB
        ASSERT_EQ(root_entries((&FatCtx)&fc), 3);
        ASSERT_NE(vfs_root.open("/fat/missing.txt", VFS_O_READ, &a), 0);
        dealloc(d.data);
    }
}

void t_ext2() {
    if (!fixtures_ready()) return;
    unsafe {
        struct MemDisk d = memdisk_load("/tmp/safec_stdlib_fs_fixtures/ext2.img");
        ASSERT_FALSE(d.data.is_null());
        if (d.data.is_null()) return;
        struct BlockDevice dev = memdisk_device(&d);
        ?&heap Ext2Ctx eq = alloc(sizeof(struct Ext2Ctx));
        if (eq.is_null()) return;
        &Ext2Ctx ec = std::expect_ref(eq, "eq is null");
        ASSERT_EQ(ec.init(&dev), 0);
        ASSERT_EQ(ec.block_size, 1024);
        vfs_init();
        ASSERT_EQ(vfs_root.mount_ro("/ext", ec), 0);
        struct VfsNode f;
        ASSERT_EQ(vfs_root.open("/ext/HELLO.TXT", VFS_O_READ, &f), 0);
        ASSERT_EQ(f.size, 1300);
        unsigned char buf[2048];
        unsigned long n = read_all(&f, buf[0..f.size]);
        ASSERT_TRUE(hello_matches(buf, n));
        struct VfsNode a;
        ASSERT_EQ(vfs_root.open("/ext/SUB/A.TXT", VFS_O_READ, &a), 0);
        safe_memset(buf[0..64], 0);
        read_all(&a, buf[0..a.size]);
        ASSERT_STR_EQ((const char*)buf, "nested file\n");
        struct VfsNode root;
        ASSERT_EQ(vfs_root.open("/ext", VFS_O_READ, &root), 0);
        int seen = 0;
        root.readdir(count_entry, (&int)&seen);
        ASSERT_EQ(seen, 5);   // . .. lost+found HELLO.TXT SUB
        ASSERT_EQ(root_entries(ec), 5);
        dealloc(eq);
        dealloc(d.data);
    }
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("tmpfs+vfs", t_tmpfs_vfs);
        t.add("partition (MBR)", t_partition);
        t.add("fat32 image", t_fat32);
        t.add("ext2 image", t_ext2);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
