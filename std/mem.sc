// SafeC Standard Library — Memory
// Safe wrappers around malloc / free / realloc / mem* from libc.
#pragma once
#include <std/mem.h>

// ── Explicit extern declarations for libc memory functions ───────────────────
namespace std {

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
inline void* alloc(unsigned long size) {
    unsafe { return malloc(size); }
}

// Allocate `size` bytes and zero-initialize them.
inline void* alloc_zeroed(unsigned long size) {
    unsafe { return calloc((unsigned long)1, size); }
}

// Free memory previously returned by alloc / alloc_zeroed / realloc_buf.
inline void dealloc(void* ptr) {
    unsafe { free(ptr); }
}

// Resize an allocation.  Returns NULL on failure (old block is NOT freed).
inline void* realloc_buf(void* ptr, unsigned long new_size) {
    unsafe { return realloc(ptr, new_size); }
}

// Copy `n` bytes from `src` to `dst`.  Regions must not overlap.
inline void safe_memcpy(void* dst, const void* src, unsigned long n) {
    unsafe { memcpy(dst, src, n); }
}

// Copy `n` bytes from `src` to `dst`.  Handles overlapping regions.
inline void safe_memmove(void* dst, const void* src, unsigned long n) {
    unsafe { memmove(dst, src, n); }
}

// Fill `n` bytes starting at `ptr` with byte value `val`.
inline void safe_memset(void* ptr, int val, unsigned long n) {
    unsafe { memset(ptr, val, n); }
}

// Compare `n` bytes of `a` and `b`.
// Returns <0, 0, or >0 (same semantics as C memcmp).
inline int safe_memcmp(const void* a, const void* b, unsigned long n) {
    unsafe { return memcmp(a, b, n); }
}

// ── Cache-line helpers ────────────────────────────────────────────────────────

inline const unsigned long mem_align_up(unsigned long addr, unsigned long align) {
    unsigned long mask = align - (unsigned long)1;
    return (addr + mask) & ~mask;
}

inline const unsigned long mem_align_down(unsigned long addr, unsigned long align) {
    unsigned long mask = align - (unsigned long)1;
    return addr & ~mask;
}

inline const int mem_is_aligned(unsigned long addr, unsigned long align) {
    unsigned long mask = align - (unsigned long)1;
    if ((addr & mask) == (unsigned long)0) { return 1; }
    return 0;
}

inline void mem_prefetch(const void* addr, int write, int locality) {
    unsafe {
#ifdef __GNUC__
        __builtin_prefetch(addr, write, locality);
#else
        (void)addr; (void)write; (void)locality;
#endif
    }
}

inline void mem_zero_secure(void* ptr, unsigned long n) {
    // Volatile write-through to prevent compiler elimination — the whole
    // point of a "secure" zero (e.g. wiping a key before it goes out of
    // scope) is that it survives dead-store elimination, so the bulk word
    // path below must stay just as volatile as the byte loop it replaces.
    unsafe {
        volatile unsigned char* p = (volatile unsigned char*)ptr;
        unsigned long i = (unsigned long)0;

        // Byte-at-a-time until the write pointer is 8-byte aligned (no-op
        // if it already is), so the bulk loop never issues a misaligned
        // volatile access.
        while (i < n && mem_is_aligned((unsigned long)(p + i), (unsigned long)8) == 0) {
            p[i] = (unsigned char)0;
            i = i + (unsigned long)1;
        }

        volatile unsigned long* pw = (volatile unsigned long*)(p + i);
        unsigned long words = (n - i) / (unsigned long)8;
        unsigned long w = (unsigned long)0;
        while (w < words) {
            pw[w] = (unsigned long)0;
            w = w + (unsigned long)1;
        }
        i = i + words * (unsigned long)8;

        while (i < n) {
            p[i] = (unsigned char)0;
            i = i + (unsigned long)1;
        }
    }
}

inline void mem_clflush(const void* addr) {
    unsafe {
#ifdef __x86_64__
        asm volatile ("clflush (%0)" : : "r"(addr) : "memory");
#else
        (void)addr;
#endif
    }
}

// ── Alignment utilities ───────────────────────────────────────────────────────

inline void* mem_align_ptr(void* ptr, unsigned long align) {
    unsafe {
        unsigned long p = (unsigned long)ptr;
        unsigned long a = mem_align_up(p, align);
        return (void*)a;
    }
}

inline const int mem_fits_page(unsigned long addr, unsigned long size) {
    unsigned long page_base = mem_align_down(addr, (unsigned long)4096);
    if (addr + size <= page_base + (unsigned long)4096) { return 1; }
    return 0;
}

} // namespace std
