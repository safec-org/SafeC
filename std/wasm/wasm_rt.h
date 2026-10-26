#pragma once
// SafeC Standard Library — freestanding wasm32 runtime.
//
// A wasm32-unknown-unknown build is linked with -nostdlib (see
// std/wasm/dom.h's header comment for the full toolchain invocation) —
// there is no libc, so the libc functions std's own sources call (malloc
// & co., the mem*/str* family, strtod/strtoll, qsort) and the compiler-rt
// builtins LLVM emits calls to (__multi3 & co. for 128-bit arithmetic)
// have no definition unless something in the module provides one. This
// file *is* that something, so a module needs no host imports beyond
// the ones its own code (and std/wasm/dom) declares.
//
// The heap is a size-class allocator over the module's linear memory:
// blocks are powers of two (header included), freed blocks are reused by
// later allocations of the same class, and the memory grows (memory.grow)
// on demand from the linker's __heap_base — a long-running page can
// allocate and free indefinitely. Memory is never returned to the host.
//
// Only meaningful for wasm32 freestanding targets — a hosted build
// (Cocoa/Win32/X11/bare-metal GUI, the RPC server, etc.) must keep using
// the real libc via std/mem.sc, not this file; including both in the
// same link would duplicate-symbol on malloc/free/etc.
#include <std/stddef.h>

unsigned char* malloc(size_t size);
void  free(unsigned char* ptr);
unsigned char* calloc(size_t count, size_t size);
unsigned char* realloc(unsigned char* ptr, size_t new_size);
unsigned char* memcpy(unsigned char* dst, const unsigned char* src, size_t n);
unsigned char* memmove(unsigned char* dst, const unsigned char* src, size_t n);
unsigned char* memset(unsigned char* ptr, int val, size_t n);
int   memcmp(const unsigned char* a, const unsigned char* b, size_t n);
unsigned char* memchr(const unsigned char* s, int c, size_t n);
// Payload bytes usable in a malloc'd block (0 for any other pointer).
size_t malloc_usable_size(unsigned char* ptr);
// Alignments above 8 over-allocate; such a block is never freed.
int   posix_memalign(unsigned char** memptr, size_t alignment, size_t size);

size_t strlen(const char* s);
int    strcmp(const char* a, const char* b);
int    strncmp(const char* a, const char* b, size_t n);
char*  strchr(const char* s, int c);
char*  strrchr(const char* s, int c);
char*  strstr(const char* haystack, const char* needle);
size_t strspn(const char* s, const char* accept);
size_t strcspn(const char* s, const char* reject);
char*  strpbrk(const char* s, const char* accept);
char*  strncpy(char* dst, const char* src, size_t n);
char*  strcat(char* dst, const char* src);
char*  strncat(char* dst, const char* src, size_t n);
char*  strtok_r(char* s, const char* delim, char** saveptr);

unsigned long long strtoull(const char* s, char** end, int base);
long long          strtoll(const char* s, char** end, int base);
int                atoi(const char* s);
long long          atoll(const char* s);
// Decimal only ('1.5e-3'), good to ~15 significant digits; no inf/nan/hex.
double             strtod(const char* s, char** end);

// Heapsort (not stable), the same contract as C's qsort.
void qsort(unsigned char* base, size_t n, size_t size, unsigned char* cmp);

// compiler-rt: 128-bit multiply and shifts (result through 'ret' = lo, hi).
void __multi3(unsigned long long* ret, unsigned long long alo, unsigned long long ahi,
              unsigned long long blo, unsigned long long bhi);
void __ashlti3(unsigned long long* ret, unsigned long long lo, unsigned long long hi, int b);
void __lshrti3(unsigned long long* ret, unsigned long long lo, unsigned long long hi, int b);
void __ashrti3(unsigned long long* ret, unsigned long long lo, unsigned long long hi, int b);

// abort() is wasm's native trap — the host sees a RuntimeError
// ("unreachable") — and the printf family drops its output (SafeC has no
// va_arg to decode a varargs list with). snprintf isn't provided: std's
// float formatting (fmt_float / String::push_float) needs a host on wasm.
void abort();
int  fprintf(unsigned char* stream, const char* fmt, ...);
int  printf(const char* fmt, ...);
