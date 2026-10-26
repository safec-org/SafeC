// stdio.h — the playground's C stdio subset (implemented by the scc runtime)
#pragma once
#include <stddef.h>
struct __FILE { int fd; };
typedef struct __FILE FILE;
extern struct __FILE __rt_files[3];
// real globals, not macros, so std sources that declare 'extern void* stdin'
// (std/io.sc, std/stderr_compat.h) compile in the same translation unit
extern FILE* stdin;
extern FILE* stdout;
extern FILE* stderr;
#define EOF (-1)
extern int printf(const char* fmt, ...);
extern int fprintf(FILE* f, const char* fmt, ...);
extern int sprintf(char* buf, const char* fmt, ...);
extern int snprintf(char* buf, size_t n, const char* fmt, ...);
extern int puts(const char* s);
extern int fputs(const char* s, FILE* f);
extern int putchar(int c);
extern int fputc(int c, FILE* f);
extern int putc(int c, FILE* f);
extern int getchar();
extern char* fgets(char* buf, int n, FILE* f);
extern int scanf(const char* fmt, ...);
extern int fflush(FILE* f);
extern int sscanf(const char* s, const char* fmt, ...);
extern int fscanf(FILE* f, const char* fmt, ...);
extern int fgetc(FILE* f);
extern int getc(FILE* f);
extern int ungetc(int c, FILE* f);
extern size_t fread(void* buf, size_t size, size_t count, FILE* f);
extern size_t fwrite(const void* buf, size_t size, size_t count, FILE* f);
// there is no file system: fopen and tmpfile return null
extern FILE* fopen(const char* path, const char* mode);
extern FILE* tmpfile();
extern int fclose(FILE* f);
extern int feof(FILE* f);
extern int ferror(FILE* f);
extern int fseek(FILE* f, long offset, int whence);
extern long ftell(FILE* f);
extern void rewind(FILE* f);
extern int remove(const char* path);
extern int rename(const char* from, const char* to);
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
