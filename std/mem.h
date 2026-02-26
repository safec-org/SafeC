// SafeC Standard Library — Memory declarations
#pragma once

void*         alloc(unsigned long size);
void*         alloc_zeroed(unsigned long size);
void          dealloc(void* ptr);
void*         realloc_buf(void* ptr, unsigned long new_size);
void          safe_memcpy(void* dst, const void* src, unsigned long n);
void          safe_memmove(void* dst, const void* src, unsigned long n);
void          safe_memset(void* ptr, int val, unsigned long n);
int           safe_memcmp(const void* a, const void* b, unsigned long n);

// ── Cache-line helpers ────────────────────────────────────────────────────────
#define MEM_CACHE_LINE_SIZE  64   // bytes (typical x86/ARM/RISC-V L1)

// Round `addr` up to the next multiple of `align` (must be power of two).
unsigned long mem_align_up(unsigned long addr, unsigned long align);

// Round `addr` down to the previous multiple of `align`.
unsigned long mem_align_down(unsigned long addr, unsigned long align);

// Return 1 if `addr` is aligned to `align` bytes.
int           mem_is_aligned(unsigned long addr, unsigned long align);

// Software prefetch hint (maps to __builtin_prefetch on GCC/Clang, no-op elsewhere).
// locality: 0=no temporal, 1=low, 2=moderate, 3=high
void          mem_prefetch(const void* addr, int write, int locality);

// Zero a cache-line-aligned region (flushes after zero for security).
void          mem_zero_secure(void* ptr, unsigned long n);

// Flush cache line containing `addr` to memory (no-op on non-x86 without CLFLUSH).
void          mem_clflush(const void* addr);

// ── Alignment utilities ───────────────────────────────────────────────────────

// Align a pointer to `align` bytes (power of two). Returns NULL if impossible
// within `slack` bytes of extra headroom.
void*         mem_align_ptr(void* ptr, unsigned long align);

// Check if `size` bytes fit within a page (4096 bytes) from `addr`.
int           mem_fits_page(unsigned long addr, unsigned long size);
