// SafeC Standard Library — File I/O implementation
#pragma once
#include "io_file.h"
#include <stdio.h>

void* file_open(const char* path, const char* mode) {
    unsafe { return (void*)fopen(path, mode); }
}

int file_close(void* f) {
    unsafe { return fclose((FILE*)f); }
}

unsigned long file_read(void* f, void* buf, unsigned long n) {
    unsafe { return fread(buf, (unsigned long)1, n, (FILE*)f); }
}

int file_getc(void* f) {
    unsafe { return fgetc((FILE*)f); }
}

char* file_gets(char* buf, int n, void* f) {
    unsafe { return fgets(buf, n, (FILE*)f); }
}

unsigned long file_write(void* f, const void* buf, unsigned long n) {
    unsafe { return fwrite(buf, (unsigned long)1, n, (FILE*)f); }
}

int file_putc(int c, void* f) {
    unsafe { return fputc(c, (FILE*)f); }
}

int file_puts(const char* s, void* f) {
    unsafe { return fputs(s, (FILE*)f); }
}

int file_seek(void* f, long long offset, int whence) {
    unsafe { return fseeko((FILE*)f, (off_t)offset, whence); }
}

long long file_tell(void* f) {
    unsafe { return (long long)ftello((FILE*)f); }
}

void file_rewind(void* f) {
    unsafe { rewind((FILE*)f); }
}

int file_eof(void* f) {
    unsafe { return feof((FILE*)f); }
}

int file_error(void* f) {
    unsafe { return ferror((FILE*)f); }
}

int file_flush(void* f) {
    unsafe { return fflush((FILE*)f); }
}

long long file_size(void* f) {
    unsafe {
        long long cur = (long long)ftello((FILE*)f);
        if (cur < 0) return -1;
        if (fseeko((FILE*)f, 0, 2) != 0) return -1;  // SEEK_END = 2
        long long end = (long long)ftello((FILE*)f);
        fseeko((FILE*)f, (off_t)cur, 0);              // SEEK_SET = 0
        return end;
    }
}
