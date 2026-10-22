// SafeC Standard Library — File I/O implementation
// &File handles wrap the C FILE* (the libc externs below stay untyped).
#pragma once
#include <std/stddef.h>
#include <std/io_file.h>
#include <std/io_traits.h>

namespace std {

extern void*          fopen(const char* path, const char* mode);
extern int            fclose(void* stream);
extern size_t  fread(void* buf, size_t size, size_t count, void* stream);
extern int            fgetc(void* stream);
extern char*          fgets(char* buf, int n, void* stream);
extern size_t  fwrite(const void* buf, size_t size, size_t count, void* stream);
extern int            fputc(int c, void* stream);
extern int            fputs(const char* s, void* stream);
extern int            fseek(void* stream, long offset, int whence);
extern long           ftell(void* stream);
extern void           rewind(void* stream);
extern int            feof(void* stream);
extern int            ferror(void* stream);
extern int            fflush(void* stream);

inline ?&File File::open(const &char path, const &char mode) {
    unsafe { return (?&File)fopen(path, mode); }
}

inline int File::close() {
    unsafe { return fclose((void*)self); }
}

inline long long File::read(&unsigned char buf, unsigned long n) {
    unsafe {
        unsigned long got = fread((void*)buf, (unsigned long)1, n, (void*)self);
        if (got == 0UL && n > 0UL && ferror((void*)self) != 0) { return -1LL; }
        return (long long)got;
    }
}

inline int File::getc() {
    unsafe { return fgetc((void*)self); }
}

inline ?&char File::gets(&char buf, int n) {
    unsafe { return fgets(buf, n, (void*)self); }
}

inline long long File::write(const &unsigned char buf, unsigned long n) {
    unsafe {
        unsigned long put = fwrite((const void*)buf, (unsigned long)1, n, (void*)self);
        if (put == 0UL && n > 0UL) { return -1LL; }
        return (long long)put;
    }
}

inline int File::putc(int c) {
    unsafe { return fputc(c, (void*)self); }
}

inline int File::puts(const &char s) {
    unsafe { return fputs(s, (void*)self); }
}

inline int File::seek(long long offset, int whence) {
    unsafe { return fseek((void*)self, (long)offset, whence); }
}

inline long long File::tell() {
    unsafe { return (long long)ftell((void*)self); }
}

inline void File::rewind() {
    unsafe { rewind((void*)self); }
}

inline int File::eof() {
    unsafe { return feof((void*)self); }
}

inline int File::error() {
    unsafe { return ferror((void*)self); }
}

inline int File::flush() {
    unsafe { return fflush((void*)self); }
}

inline long long File::size() {
    return stream_size(self);
}

// Forwarding wrappers kept for std/ml (see io_file.h).
// kept for std/ml until its refactor moves to the static method
inline ?&File        file_open(const &char path, const &char mode) { return File::open(path, mode); }
inline int           file_close(&File f) { return f.close(); }
inline unsigned long file_read(&File f, &unsigned char buf, unsigned long n) {
    long long r = f.read(buf, n); return r < 0LL ? 0UL : (unsigned long)r;
}
inline unsigned long file_write(&File f, const &unsigned char buf, unsigned long n) {
    long long r = f.write(buf, n); return r < 0LL ? 0UL : (unsigned long)r;
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
                long long n = f.read(&buf[0], 16384UL);
                if (n < 0LL) { ok = 0; break; }
                if (n == 0LL) { break; }
                unsafe { out.push_n((const char*)&buf[0], (unsigned long)n); }
            }
            f.close();
            return ok;
        }
        case null: { return 0; }
    }
    return 0;
}

int File::write_bytes(const &char path, const &char data, unsigned long n) {
    ?&File fo = File::open(path, "wb");
    match (fo) {
        case some(f): {
            long long w;
            unsafe { w = f.write((const unsigned char*)data, n); }
            int closed = f.close();
            return (w == (long long)n && closed == 0) ? 1 : 0;
        }
        case null: { return 0; }
    }
    return 0;
}

} // namespace std
