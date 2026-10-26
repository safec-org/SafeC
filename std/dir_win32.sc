// SafeC Standard Library — Directories: Win32 backend (see dir.h; the tree
// operations shared with POSIX are in dir.sc). kernel32 only — always
// linked on Windows. ANSI (A) entry points: paths are passed through in
// the active code page.
#pragma once
#include <std/stddef.h>
#include <std/dir.h>
#include <std/mem.h>
#include <std/collections/string.h>

// WIN32_FIND_DATAA: dwFileAttributes @0, ftLastWriteTime @20,
// nFileSizeHigh @28, nFileSizeLow @32, cFileName[260] @44; 320 bytes.
#define SAFEC_FIND_DATA_SIZE   320UL
#define SAFEC_FIND_NAME_OFF    44UL
// WIN32_FILE_ATTRIBUTE_DATA: dwFileAttributes @0, ftLastWriteTime @20,
// nFileSizeHigh @28, nFileSizeLow @32; 36 bytes.
#define SAFEC_FILE_ATTR_SIZE   36UL
#define SAFEC_FILE_ATTRIBUTE_DIRECTORY     0x10U
#define SAFEC_FILE_ATTRIBUTE_REPARSE_POINT 0x400U
#define SAFEC_FILE_ATTRIBUTE_DEVICE        0x40U
// FILETIME ticks (100 ns since 1601-01-01) at the Unix epoch.
#define SAFEC_FILETIME_UNIX_EPOCH 116444736000000000LL

namespace std {

extern void* FindFirstFileA(const char* pattern, void* find_data);
extern int   FindNextFileA(void* handle, void* find_data);
extern int   FindClose(void* handle);
extern int   GetFileAttributesExA(const char* path, int info_level, void* info);
extern int   CreateDirectoryA(const char* path, void* security);
extern int   RemoveDirectoryA(const char* path);
extern int   DeleteFileA(const char* path);

struct DirWin32_ {
    void* handle;
    int   pending;   // find_data already holds an unread entry (from FindFirstFileA)
    unsigned char find_data[320];
};

static unsigned long long dir_le_(const &unsigned char b, unsigned long off, unsigned long n) {
    unsigned long long v = 0ULL;
    unsafe {
        for (unsigned long i = 0UL; i < n; i = i + 1UL) {
            v = v | ((unsigned long long)b[off + i] << (8UL * i));
        }
    }
    return v;
}

static int dir_kind_from_attrs_(unsigned int attrs) {
    if ((attrs & SAFEC_FILE_ATTRIBUTE_REPARSE_POINT) != 0U) { return DIR_LINK; }
    if ((attrs & SAFEC_FILE_ATTRIBUTE_DIRECTORY) != 0U) { return DIR_DIR; }
    if ((attrs & SAFEC_FILE_ATTRIBUTE_DEVICE) != 0U) { return DIR_OTHER; }
    return DIR_FILE;
}

// ftLastWriteTime (two little-endian DWORDs at 'off') as ns since the Unix epoch.
static long long dir_filetime_ns_(const &unsigned char b, unsigned long off) {
    long long ticks = (long long)dir_le_(b, off, 8UL);
    return (ticks - SAFEC_FILETIME_UNIX_EPOCH) * 100LL;
}

?&Dir Dir::open(const &char path) {
    struct String pattern = String::from(path);
    if (pattern.len > 0UL && !pattern.ends_with("/") && !pattern.ends_with("\\")) { pattern.push_char('/'); }
    pattern.push_char('*');
    struct DirWin32_* h;
    unsafe {
        h = (struct DirWin32_*)alloc_raw_((unsigned long)sizeof(struct DirWin32_));
        h->handle = FindFirstFileA(pattern.as_ptr(), (void*)&h->find_data[0]);
    }
    pattern.free();
    unsafe {
        if (h->handle == (void*)(long long)-1) {
            dealloc(h);
            return (?&Dir)null;
        }
        h->pending = 1;
        return (?&Dir)(void*)h;
    }
}

int Dir::next(&DirEntry out) {
    unsafe {
        struct DirWin32_* h = (struct DirWin32_*)(void*)self;
        for (;;) {
            if (!h->pending) {
                if (FindNextFileA(h->handle, (void*)&h->find_data[0]) == 0) { return 0; }
            }
            h->pending = 0;
            const char* name = (const char*)&h->find_data[SAFEC_FIND_NAME_OFF];
            if (name[0] == '.' && (name[1] == '\0' || (name[1] == '.' && name[2] == '\0'))) { continue; }
            out.name.clear();
            out.name.push(name);
            out.kind = dir_kind_from_attrs_((unsigned int)dir_le_(&h->find_data[0], 0UL, 4UL));
            return 1;
        }
    }
    return 0;
}

void Dir::close() { self.drop(); }

void Dir::drop() {
    unsafe {
        struct DirWin32_* h = (struct DirWin32_*)(void*)self;
        FindClose(h->handle);
        dealloc(h);
    }
}

int file_stat(const &char path, &FileStat out) {
    out.kind = DIR_NONE;
    out.size = 0LL;
    out.mtime_ns = 0LL;
    unsigned char buf[36];
    int ok;
    unsafe { ok = GetFileAttributesExA(path, 0, (void*)&buf[0]); }   // GetFileExInfoStandard
    if (ok == 0) { return 0; }
    out.kind = dir_kind_from_attrs_((unsigned int)dir_le_(&buf[0], 0UL, 4UL));
    out.size = (long long)((dir_le_(&buf[0], 28UL, 4UL) << 32UL) | dir_le_(&buf[0], 32UL, 4UL));
    out.mtime_ns = dir_filetime_ns_(&buf[0], 20UL);
    return 1;
}

int dir_create(const &char path) {
    int ok;
    unsafe { ok = CreateDirectoryA(path, (void*)0); }
    if (ok != 0) { return 1; }
    return path_kind(path) == DIR_DIR ? 1 : 0;
}

int dir_remove(const &char path) {
    int ok;
    unsafe { ok = RemoveDirectoryA(path); }
    return ok != 0 ? 1 : 0;
}

int file_remove(const &char path) {
    int ok;
    unsafe { ok = DeleteFileA(path); }
    return ok != 0 ? 1 : 0;
}

} // namespace std
