// SafeC Standard Library — File I/O implementation
// &File handles wrap the C FILE (stderr_compat.h's opaque CFile_).
#pragma once
#include <std/stderr_compat.h>
#include <std/stddef.h>
#include <std/io_file.h>
#include <std/io_traits.h>

namespace std {

extern CFile_*          fopen(const char* path, const char* mode);
extern int            fclose(CFile_* stream);
extern size_t  fread(unsigned char* buf, size_t size, size_t count, CFile_* stream);
extern int            fgetc(CFile_* stream);
extern char*          fgets(char* buf, int n, CFile_* stream);
extern size_t  fwrite(const unsigned char* buf, size_t size, size_t count, CFile_* stream);
extern int            fputc(int c, CFile_* stream);
extern int            fputs(const char* s, CFile_* stream);
extern int            fseek(CFile_* stream, long offset, int whence);
extern long           ftell(CFile_* stream);
extern void           rewind(CFile_* stream);
extern int            feof(CFile_* stream);
extern int            ferror(CFile_* stream);
extern int            fflush(CFile_* stream);

inline ?&File File::open(const &char path, const &char mode) {
    unsafe { return (?&File)fopen(path, mode); }
}

inline void File::drop() { self.close(); }

inline int File::close() {
    unsafe { return fclose((CFile_*)self); }
}

inline long long File::read([]unsigned char buf_s) {
    unsigned long n = buf_s.len;
    if (n == 0UL) { return 0LL; }
    &unsigned char buf;
    unsafe { buf = (&unsigned char)buf_s.ptr; }
    unsafe {
        unsigned long got = fread((unsigned char*)buf, (unsigned long)1, n, (CFile_*)self);
        if (got == 0UL && n > 0UL && ferror((CFile_*)self) != 0) { return -1LL; }
        return (long long)got;
    }
}

inline int File::getc() {
    unsafe { return fgetc((CFile_*)self); }
}

inline ?&char File::gets([]char buf) {
    if (buf.len == 0UL) { return null; }
    int n = buf.len > 2147483647UL ? 2147483647 : (int)buf.len;
    unsafe { return fgets((char*)buf.ptr, n, (CFile_*)self); }
}

inline long long File::write([]const unsigned char buf_s) {
    unsigned long n = buf_s.len;
    if (n == 0UL) { return 0LL; }
    const &unsigned char buf;
    unsafe { buf = (const &unsigned char)buf_s.ptr; }
    unsafe {
        unsigned long put = fwrite((const unsigned char*)buf, (unsigned long)1, n, (CFile_*)self);
        if (put == 0UL && n > 0UL) { return -1LL; }
        return (long long)put;
    }
}

inline int File::putc(int c) {
    unsafe { return fputc(c, (CFile_*)self); }
}

inline int File::puts(const &char s) {
    unsafe { return fputs(s, (CFile_*)self); }
}

inline int File::seek(long long offset, int whence) {
    unsafe { return fseek((CFile_*)self, (long)offset, whence); }
}

inline long long File::tell() {
    unsafe { return (long long)ftell((CFile_*)self); }
}

inline void File::rewind() {
    unsafe { rewind((CFile_*)self); }
}

inline int File::eof() {
    unsafe { return feof((CFile_*)self); }
}

inline int File::error() {
    unsafe { return ferror((CFile_*)self); }
}

inline int File::flush() {
    unsafe { return fflush((CFile_*)self); }
}

inline long long File::size() {
    return stream_size(self);
}

// Forwarding wrappers kept for std/ml (see io_file.h).
// kept for std/ml until its refactor moves to the static method
inline ?&File        file_open(const &char path, const &char mode) { return File::open(path, mode); }
// Closes *f in place (File::open hands out references, not owned values).
inline int           file_close(&File f) { unsafe { return f.close(); } }
inline unsigned long file_read(&File f, []unsigned char buf) {
    long long r = f.read(buf); return r < 0LL ? 0UL : (unsigned long)r;
}
inline unsigned long file_write(&File f, []const unsigned char buf) {
    long long r = f.write(buf); return r < 0LL ? 0UL : (unsigned long)r;
}
inline int           file_seek(&File f, long long offset, int whence) { return f.seek(offset, whence); }
inline long long     file_tell(&File f) { return f.tell(); }
inline long long     file_size(&File f) { return f.size(); }

int File::read_to_string(const &char path, &String out) {
    ?&File fo = File::open(path, "rb");
    match (fo) {
        case some(f): {
            long long total = f.size();
            if (total > 0LL) { out.reserve((unsigned long)total); }
            unsigned char buf[16384];
            int ok = 1;
            for (;;) {
                long long n = f.read(buf[0..16384UL]);
                if (n < 0LL) { ok = 0; break; }
                if (n == 0LL) { break; }
                out.push_bytes(buf[0..(unsigned long)n]);
            }
            f.close();
            return ok;
        }
        case null: { return 0; }
    }
    return 0;
}

int File::write_bytes(const &char path, []const unsigned char data) {
    ?&File fo = File::open(path, "wb");
    match (fo) {
        case some(f): {
            long long w = data.len == 0UL ? 0LL : f.write(data);
            int closed = f.close();
            return (w == (long long)data.len && closed == 0) ? 1 : 0;
        }
        case null: { return 0; }
    }
    return 0;
}

} // namespace std
