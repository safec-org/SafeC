// SafeC Standard Library — File I/O implementation
// &File handles wrap the C FILE* (the libc externs below stay untyped).
#pragma once
#include <std/stddef.h>
#include <std/io_file.h>

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

inline ?&File file_open(const &char path, const &char mode) {
    unsafe { return (?&File)fopen(path, mode); }
}

inline int File::close() {
    unsafe { return fclose((void*)self); }
}

inline unsigned long File::read(&unsigned char buf, unsigned long n) {
    unsafe { return fread((void*)buf, (unsigned long)1, n, (void*)self); }
}

inline int File::getc() {
    unsafe { return fgetc((void*)self); }
}

inline ?&char File::gets(&char buf, int n) {
    unsafe { return fgets(buf, n, (void*)self); }
}

inline unsigned long File::write(const &unsigned char buf, unsigned long n) {
    unsafe { return fwrite((const void*)buf, (unsigned long)1, n, (void*)self); }
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
    unsafe {
        long cur = ftell((void*)self);
        if (cur < 0) return (long long)-1;
        if (fseek((void*)self, 0, 2) != 0) return (long long)-1;  // SEEK_END = 2
        long long end = (long long)ftell((void*)self);
        fseek((void*)self, cur, 0);                                 // SEEK_SET = 0
        return end;
    }
}

// Forwarding wrappers kept for std/ml (see io_file.h).
inline int           file_close(&File f) { return f.close(); }
inline unsigned long file_read(&File f, &unsigned char buf, unsigned long n) { return f.read(buf, n); }
inline unsigned long file_write(&File f, const &unsigned char buf, unsigned long n) { return f.write(buf, n); }
inline int           file_seek(&File f, long long offset, int whence) { return f.seek(offset, whence); }
inline long long     file_tell(&File f) { return f.tell(); }
inline long long     file_size(&File f) { return f.size(); }

} // namespace std
