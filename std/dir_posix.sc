// SafeC Standard Library — Directories: POSIX backend (Linux, macOS, BSD)
// (see dir.h; the tree operations shared with Windows are in dir.sc)
//
// struct dirent and struct stat have no portable layout, so the fields
// read here are at per-platform offsets:
//   - readdir: d_type / d_name at 18/19 (glibc), 20/21 (macOS arm64,
//     64-bit inodes), 6/8 (macOS x86_64's legacy 'readdir' symbol),
//     18/24 (FreeBSD 12+, also the fallback for other BSDs);
//   - file metadata: Linux uses the statx syscall (one fixed layout on
//     every architecture, and no dependency on glibc's stat symbols, which
//     only exist as real symbols from glibc 2.33); macOS uses lstat
//     (64-bit-inode layout on arm64, the legacy layout on x86_64); other
//     systems get kind and size without stat (opendir/fopen probes) and a
//     zero mtime_ns.
#pragma once
#include <std/stddef.h>
#include <std/dir.h>
#include <std/mem.h>
#include <std/io_file.h>
#include <std/collections/string.h>

#ifdef __linux__
#define SAFEC_DIRENT_TYPE_OFF 18UL
#define SAFEC_DIRENT_NAME_OFF 19UL
#else
#ifdef __APPLE__
#ifdef __aarch64__
#define SAFEC_DIRENT_TYPE_OFF 20UL
#define SAFEC_DIRENT_NAME_OFF 21UL
#else
#define SAFEC_DIRENT_TYPE_OFF 6UL
#define SAFEC_DIRENT_NAME_OFF 8UL
#endif
#else
#define SAFEC_DIRENT_TYPE_OFF 18UL
#define SAFEC_DIRENT_NAME_OFF 24UL
#endif
#endif

// d_type values (identical on Linux, macOS and the BSDs).
#define SAFEC_DT_UNKNOWN 0
#define SAFEC_DT_DIR     4
#define SAFEC_DT_REG     8
#define SAFEC_DT_LNK     10

// st_mode file type bits (identical everywhere).
#define SAFEC_S_IFMT  0xF000
#define SAFEC_S_IFDIR 0x4000
#define SAFEC_S_IFREG 0x8000
#define SAFEC_S_IFLNK 0xA000

namespace std {

extern void* opendir(const char* path);
extern void* readdir(void* dirp);
extern int   closedir(void* dirp);
extern int   mkdir(const char* path, unsigned int mode);
extern int   rmdir(const char* path);
extern int   unlink(const char* path);

#ifdef __linux__
extern long  syscall(long number, ...);
#ifdef __x86_64__
#define SAFEC_SYS_STATX 332L
#else
#define SAFEC_SYS_STATX 291L   // aarch64, riscv64 and the other generic-syscall ABIs
#endif
#define SAFEC_AT_FDCWD            (-100)
#define SAFEC_AT_SYMLINK_NOFOLLOW 0x100
#define SAFEC_STATX_BASIC_STATS   0x7ff
#else
#ifdef __APPLE__
extern int   lstat(const char* path, void* buf);
#endif
#endif

// What a Dir handle points at.
struct DirPosix_ {
    void* dirp;
    struct String path;   // for entries whose d_type is DT_UNKNOWN
};

?&Dir Dir::open(const &char path) {
    void* d;
    unsafe { d = opendir(path); }
    if (d == (void*)0) { return (?&Dir)null; }
    struct DirPosix_* h;
    unsafe {
        h = (struct DirPosix_*)alloc_raw_((unsigned long)sizeof(struct DirPosix_));
        h->dirp = d;
        h->path = String::from(path);
        return (?&Dir)(void*)h;
    }
}

int Dir::next(&DirEntry out) {
    unsafe {
        struct DirPosix_* h = (struct DirPosix_*)(void*)self;
        for (;;) {
            unsigned char* ent = (unsigned char*)readdir(h->dirp);
            if (ent == (unsigned char*)0) { return 0; }
            const char* name = (const char*)(ent + SAFEC_DIRENT_NAME_OFF);
            if (name[0] == '.' && (name[1] == '\0' || (name[1] == '.' && name[2] == '\0'))) { continue; }
            int t = (int)ent[SAFEC_DIRENT_TYPE_OFF];
            out.name.clear();
            out.name.push(name);
            if (t == SAFEC_DT_DIR) { out.kind = DIR_DIR; }
            else if (t == SAFEC_DT_REG) { out.kind = DIR_FILE; }
            else if (t == SAFEC_DT_LNK) { out.kind = DIR_LINK; }
            else if (t == SAFEC_DT_UNKNOWN) {
                // Some file systems don't fill d_type: ask stat.
                struct String full = h->path.clone();
                full.push_char('/');
                full.push(name);
                struct FileStat st;
                file_stat(full.as_ptr(), &st);
                out.kind = st.kind == DIR_NONE ? DIR_OTHER : st.kind;
                full.free();
            }
            else { out.kind = DIR_OTHER; }
            return 1;
        }
    }
    return 0;
}

void Dir::close() {
    unsafe {
        struct DirPosix_* h = (struct DirPosix_*)(void*)self;
        closedir(h->dirp);
        h->path.free();
        dealloc_raw_((void*)h);
    }
}

static int dir_kind_from_mode_(unsigned int mode) {
    unsigned int t = mode & (unsigned int)SAFEC_S_IFMT;
    if (t == (unsigned int)SAFEC_S_IFDIR) { return DIR_DIR; }
    if (t == (unsigned int)SAFEC_S_IFREG) { return DIR_FILE; }
    if (t == (unsigned int)SAFEC_S_IFLNK) { return DIR_LINK; }
    return DIR_OTHER;
}

// Little-endian field reads out of a stat buffer.
static unsigned long long dir_le_(const &unsigned char b, unsigned long off, unsigned long n) {
    unsigned long long v = 0ULL;
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL) {
            v = v | ((unsigned long long)b[off + i] << (8UL * i));
        }
    }
    return v;
}

// Without a stat call: a directory if opendir succeeds, a file if fopen does.
static int dir_probe_stat_(const &char path, &FileStat out) {
    ?&Dir d = Dir::open(path);
    match (d) {
        case some(h): { h.close(); out.kind = DIR_DIR; return 1; }
        case null: { }
    }
    ?&File f = File::open(path, "rb");
    match (f) {
        case some(h): {
            out.kind = DIR_FILE;
            out.size = h.size();
            h.close();
            return 1;
        }
        case null: { }
    }
    return 0;
}

int file_stat(const &char path, &FileStat out) {
    out.kind = DIR_NONE;
    out.size = 0LL;
    out.mtime_ns = 0LL;
    unsigned char buf[256];
#ifdef __linux__
    long rc;
    unsafe { rc = syscall(SAFEC_SYS_STATX, SAFEC_AT_FDCWD, path, SAFEC_AT_SYMLINK_NOFOLLOW,
                          SAFEC_STATX_BASIC_STATS, (void*)&buf[0]); }
    if (rc == 0L) {
        out.kind = dir_kind_from_mode_((unsigned int)dir_le_(&buf[0], 28UL, 2UL));
        out.size = (long long)dir_le_(&buf[0], 40UL, 8UL);
        out.mtime_ns = (long long)dir_le_(&buf[0], 112UL, 8UL) * 1000000000LL +
                       (long long)dir_le_(&buf[0], 120UL, 4UL);
        return 1;
    }
    return dir_probe_stat_(path, out);   // kernels before 4.11 have no statx
#else
#ifdef __APPLE__
    int rc;
    unsafe { rc = lstat(path, (void*)&buf[0]); }
    if (rc != 0) { return 0; }
#ifdef __aarch64__
    out.kind = dir_kind_from_mode_((unsigned int)dir_le_(&buf[0], 4UL, 2UL));
    out.mtime_ns = (long long)dir_le_(&buf[0], 48UL, 8UL) * 1000000000LL + (long long)dir_le_(&buf[0], 56UL, 8UL);
    out.size = (long long)dir_le_(&buf[0], 96UL, 8UL);
#else
    out.kind = dir_kind_from_mode_((unsigned int)dir_le_(&buf[0], 8UL, 2UL));
    out.mtime_ns = (long long)dir_le_(&buf[0], 40UL, 8UL) * 1000000000LL + (long long)dir_le_(&buf[0], 48UL, 8UL);
    out.size = (long long)dir_le_(&buf[0], 72UL, 8UL);
#endif
    return 1;
#else
    buf[0] = (unsigned char)0;
    return dir_probe_stat_(path, out);
#endif
#endif
}

int dir_create(const &char path) {
    int rc;
    unsafe { rc = mkdir(path, 511U); }   // 0777, narrowed by the umask
    if (rc == 0) { return 1; }
    return path_kind(path) == DIR_DIR ? 1 : 0;
}

int dir_remove(const &char path) {
    int rc;
    unsafe { rc = rmdir(path); }
    return rc == 0 ? 1 : 0;
}

int file_remove(const &char path) {
    int rc;
    unsafe { rc = unlink(path); }
    return rc == 0 ? 1 : 0;
}

} // namespace std
