// SafeC Standard Library — Memory declarations
#pragma once

namespace std {

// Heap allocation. T comes from the variable the result is assigned to
// ('?&heap char buf = std::alloc(64UL);'); use 'void' for untyped bytes.
// Sizes are in bytes.
generic<T> ?&heap T alloc(unsigned long size);
generic<T> ?&heap T alloc_zeroed(unsigned long size);
generic<T> void     dealloc(?&heap T ptr);
generic<T> ?&heap T realloc_buf(?&heap T ptr, unsigned long new_size);

// Multiply two allocation-size factors (typically an element count and an
// element size) and abort with a diagnostic instead of silently wrapping
// if the product overflows unsigned long — the classic "count * elem_size"
// integer-overflow-to-undersized-allocation bug. Use this instead of a
// bare '*' anywhere the multiplicands aren't both compile-time constants
// small enough to prove can't overflow.
unsigned long checked_mul_size(unsigned long a, unsigned long b);
// Same idea as checked_mul_size but for addition (e.g. summing several
// fields' sizes into one combined allocation) -- abort instead of silently
// wrapping if the sum overflows unsigned long.
unsigned long checked_add_size(unsigned long a, unsigned long b);
generic<T> void safe_memcpy(&T dst, const &T src, unsigned long n);
generic<T> void safe_memmove(&T dst, const &T src, unsigned long n);
generic<T> void safe_memset(&T ptr, int val, unsigned long n);
generic<T> int  safe_memcmp(const &T a, const &T b, unsigned long n);

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
generic<T> void mem_prefetch(const &T addr, int write, int locality);

// Zero a cache-line-aligned region (flushes after zero for security).
generic<T> void mem_zero_secure(&T ptr, unsigned long n);

// Flush cache line containing `addr` to memory (no-op on non-x86 without CLFLUSH).
generic<T> void mem_clflush(const &T addr);

// ── Alignment utilities ───────────────────────────────────────────────────────

// Round a reference up to the next `align`-byte boundary (power of two).
generic<T> ?&T mem_align_ptr(&T ptr, unsigned long align);

// Check if `size` bytes fit within a page (4096 bytes) from `addr`.
int           mem_fits_page(unsigned long addr, unsigned long size);

// ── Type-erased forms ─────────────────────────────────────────────────────────
// What the generic wrappers above forward to, and what std's own
// type-erased containers use internally. 'new T' (std_alloc_raw_) and
// 'malloc_defer' are backed by alloc_raw_/dealloc_raw_.
void*         alloc_raw_(unsigned long size);
void*         alloc_zeroed_raw_(unsigned long size);
void          dealloc_raw_(void* ptr);
void*         realloc_buf_raw_(void* ptr, unsigned long new_size);
void          safe_memcpy_raw_(void* dst, const void* src, unsigned long n);
void          safe_memmove_raw_(void* dst, const void* src, unsigned long n);
void          safe_memset_raw_(void* ptr, int val, unsigned long n);
int           safe_memcmp_raw_(const void* a, const void* b, unsigned long n);
void          mem_prefetch_raw_(const void* addr, int write, int locality);
void          mem_zero_secure_raw_(void* ptr, unsigned long n);
void          mem_clflush_raw_(const void* addr);
void*         mem_align_ptr_raw_(void* ptr, unsigned long align);

generic<T> ?&heap T alloc(unsigned long size) { return alloc_raw_(size); }
generic<T> ?&heap T alloc_zeroed(unsigned long size) { return alloc_zeroed_raw_(size); }
generic<T> void dealloc(?&heap T ptr) { dealloc_raw_(ptr); }
generic<T> ?&heap T realloc_buf(?&heap T ptr, unsigned long new_size) {
    return realloc_buf_raw_(ptr, new_size);
}
generic<T> void safe_memcpy(&T dst, const &T src, unsigned long n) { safe_memcpy_raw_(dst, src, n); }
generic<T> void safe_memmove(&T dst, const &T src, unsigned long n) { safe_memmove_raw_(dst, src, n); }
generic<T> void safe_memset(&T ptr, int val, unsigned long n) { safe_memset_raw_(ptr, val, n); }
generic<T> int safe_memcmp(const &T a, const &T b, unsigned long n) { return safe_memcmp_raw_(a, b, n); }
generic<T> void mem_prefetch(const &T addr, int write, int locality) {
    mem_prefetch_raw_(addr, write, locality);
}
generic<T> void mem_zero_secure(&T ptr, unsigned long n) { mem_zero_secure_raw_(ptr, n); }
generic<T> void mem_clflush(const &T addr) { mem_clflush_raw_(addr); }
generic<T> ?&T mem_align_ptr(&T ptr, unsigned long align) { return mem_align_ptr_raw_(ptr, align); }

} // namespace std
