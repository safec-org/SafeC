// SafeC Standard Library — File I/O implementation
// Uses void* for FILE* handles to avoid exposing the C FILE struct.
#pragma once
#include "io_file.h"

extern void*          fopen(const char* path, const char* mode);
extern int            fclose(void* stream);
extern unsigned long  fread(void* buf, unsigned long size, unsigned long count, void* stream);
extern int            fgetc(void* stream);
extern char*          fgets(char* buf, int n, void* stream);
extern unsigned long  fwrite(const void* buf, unsigned long size, unsigned long count, void* stream);
extern int            fputc(int c, void* stream);
extern int            fputs(const char* s, void* stream);
extern int            fseek(void* stream, long offset, int whence);
extern long           ftell(void* stream);
extern void           rewind(void* stream);
extern int            feof(void* stream);
extern int            ferror(void* stream);
extern int            fflush(void* stream);

void* file_open(const char* path, const char* mode) {
    unsafe { return fopen(path, mode); }
}

int file_close(void* f) {
    unsafe { return fclose(f); }
}

unsigned long file_read(void* f, void* buf, unsigned long n) {
    unsafe { return fread(buf, (unsigned long)1, n, f); }
}

int file_getc(void* f) {
    unsafe { return fgetc(f); }
}

char* file_gets(char* buf, int n, void* f) {
    unsafe { return fgets(buf, n, f); }
}

unsigned long file_write(void* f, const void* buf, unsigned long n) {
    unsafe { return fwrite(buf, (unsigned long)1, n, f); }
}

int file_putc(int c, void* f) {
    unsafe { return fputc(c, f); }
}

int file_puts(const char* s, void* f) {
    unsafe { return fputs(s, f); }
}

int file_seek(void* f, long long offset, int whence) {
    unsafe { return fseek(f, (long)offset, whence); }
}

long long file_tell(void* f) {
    unsafe { return (long long)ftell(f); }
}

void file_rewind(void* f) {
    unsafe { rewind(f); }
}

int file_eof(void* f) {
    unsafe { return feof(f); }
}

int file_error(void* f) {
    unsafe { return ferror(f); }
}

int file_flush(void* f) {
    unsafe { return fflush(f); }
}

long long file_size(void* f) {
    unsafe {
        long cur = ftell(f);
        if (cur < 0) return (long long)-1;
        if (fseek(f, 0, 2) != 0) return (long long)-1;  // SEEK_END = 2
        long long end = (long long)ftell(f);
        fseek(f, cur, 0);                                 // SEEK_SET = 0
        return end;
    }
}
