// SafeC Standard Library — Memory
// Safe wrappers around malloc / free / realloc / mem* from libc.
#pragma once
#include "mem.h"

// ── Explicit extern declarations for libc memory functions ───────────────────
extern void* malloc(unsigned long size);
extern void* calloc(unsigned long count, unsigned long size);
extern void  free(void* ptr);
extern void* realloc(void* ptr, unsigned long new_size);
extern void* memcpy(void* dst, const void* src, unsigned long n);
extern void* memmove(void* dst, const void* src, unsigned long n);
extern void* memset(void* ptr, int val, unsigned long n);
extern int   memcmp(const void* a, const void* b, unsigned long n);

// Allocate `size` bytes.  Returns NULL on failure; callers should check.
// Use inside unsafe{} when storing the result in a raw pointer.
void* alloc(unsigned long size) {
    unsafe { return malloc(size); }
}

// Allocate `size` bytes and zero-initialize them.
void* alloc_zeroed(unsigned long size) {
    unsafe { return calloc((unsigned long)1, size); }
}

// Free memory previously returned by alloc / alloc_zeroed / realloc_buf.
void dealloc(void* ptr) {
    unsafe { free(ptr); }
}

// Resize an allocation.  Returns NULL on failure (old block is NOT freed).
void* realloc_buf(void* ptr, unsigned long new_size) {
    unsafe { return realloc(ptr, new_size); }
}

// Copy `n` bytes from `src` to `dst`.  Regions must not overlap.
void safe_memcpy(void* dst, const void* src, unsigned long n) {
    unsafe { memcpy(dst, src, n); }
}

// Copy `n` bytes from `src` to `dst`.  Handles overlapping regions.
void safe_memmove(void* dst, const void* src, unsigned long n) {
    unsafe { memmove(dst, src, n); }
}

// Fill `n` bytes starting at `ptr` with byte value `val`.
void safe_memset(void* ptr, int val, unsigned long n) {
    unsafe { memset(ptr, val, n); }
}

// Compare `n` bytes of `a` and `b`.
// Returns <0, 0, or >0 (same semantics as C memcmp).
int safe_memcmp(const void* a, const void* b, unsigned long n) {
    unsafe { return memcmp(a, b, n); }
}
