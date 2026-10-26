// string.h — playground subset (implemented by the scc runtime)
#pragma once
#include <stddef.h>
extern void* memcpy(void* dst, const void* src, size_t n);
extern void* memmove(void* dst, const void* src, size_t n);
extern void* memset(void* p, int v, size_t n);
extern int memcmp(const void* a, const void* b, size_t n);
extern size_t strlen(const char* s);
extern int strcmp(const char* a, const char* b);
extern int strncmp(const char* a, const char* b, size_t n);
extern char* strcpy(char* d, const char* s);
extern char* strncpy(char* d, const char* s, size_t n);
extern char* strcat(char* d, const char* s);
extern char* strncat(char* d, const char* s, size_t n);
extern char* strchr(const char* s, int c);
extern char* strrchr(const char* s, int c);
extern char* strstr(const char* h, const char* n);
extern char* strdup(const char* s);
extern void* memchr(const void* s, int c, size_t n);
extern size_t strspn(const char* s, const char* accept);
extern size_t strcspn(const char* s, const char* reject);
extern char* strpbrk(const char* s, const char* accept);
extern char* strtok(char* s, const char* delim);
extern char* strtok_r(char* s, const char* delim, char** save);
