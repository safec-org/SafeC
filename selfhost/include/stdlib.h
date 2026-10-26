// stdlib.h — playground subset (implemented by the scc runtime)
#pragma once
#include <stddef.h>
#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
#define RAND_MAX 2147483647
extern void* malloc(size_t n);
extern void* calloc(size_t n, size_t size);
extern void* realloc(void* p, size_t n);
extern void free(void* p);
extern void exit(int code);
extern void abort();
extern int atoi(const char* s);
extern long atol(const char* s);
extern long long atoll(const char* s);
extern double atof(const char* s);
extern long strtol(const char* s, char** end, int base);
extern unsigned long strtoul(const char* s, char** end, int base);
extern long long strtoll(const char* s, char** end, int base);
extern double strtod(const char* s, char** end);
extern int abs(int x);
extern long labs(long x);
extern long long llabs(long long x);
extern int rand();
extern void srand(unsigned int seed);
extern unsigned long long strtoull(const char* s, char** end, int base);
extern float strtof(const char* s, char** end);
extern int atexit(void* fnp);
extern char* getenv(const char* name);   // always null: there is no environment
extern int system(const char* cmd);      // always -1
extern void qsort(void* base, size_t n, size_t size, void* cmp);
extern void* bsearch(const void* key, const void* base, size_t n, size_t size, void* cmp);
