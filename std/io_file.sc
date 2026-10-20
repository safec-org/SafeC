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

inline int file_close(&File f) {
    unsafe { return fclose((void*)f); }
}

inline unsigned long file_read(&File f, &unsigned char buf, unsigned long n) {
    unsafe { return fread((void*)buf, (unsigned long)1, n, (void*)f); }
}

inline int file_getc(&File f) {
    unsafe { return fgetc((void*)f); }
}

inline ?&char file_gets(&char buf, int n, &File f) {
    unsafe { return fgets(buf, n, (void*)f); }
}

inline unsigned long file_write(&File f, const &unsigned char buf, unsigned long n) {
    unsafe { return fwrite((const void*)buf, (unsigned long)1, n, (void*)f); }
}

inline int file_putc(int c, &File f) {
    unsafe { return fputc(c, (void*)f); }
}

inline int file_puts(const &char s, &File f) {
    unsafe { return fputs(s, (void*)f); }
}

inline int file_seek(&File f, long long offset, int whence) {
    unsafe { return fseek((void*)f, (long)offset, whence); }
}

inline long long file_tell(&File f) {
    unsafe { return (long long)ftell((void*)f); }
}

inline void file_rewind(&File f) {
    unsafe { rewind((void*)f); }
}

inline int file_eof(&File f) {
    unsafe { return feof((void*)f); }
}

inline int file_error(&File f) {
    unsafe { return ferror((void*)f); }
}

inline int file_flush(&File f) {
    unsafe { return fflush((void*)f); }
}

inline long long file_size(&File f) {
    unsafe {
        long cur = ftell((void*)f);
        if (cur < 0) return (long long)-1;
        if (fseek((void*)f, 0, 2) != 0) return (long long)-1;  // SEEK_END = 2
        long long end = (long long)ftell((void*)f);
        fseek((void*)f, cur, 0);                                 // SEEK_SET = 0
        return end;
    }
}

} // namespace std
