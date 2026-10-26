// stdio.h — the playground's C stdio subset (implemented by the scc runtime)
#pragma once
#include <stddef.h>
struct __FILE { int fd; };
typedef struct __FILE FILE;
extern struct __FILE __rt_files[3];
#define stdin (&__rt_files[0])
#define stdout (&__rt_files[1])
#define stderr (&__rt_files[2])
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
