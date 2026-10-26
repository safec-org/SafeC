// SafeC Standard Library — Memory declarations
#pragma once

namespace std {

// Heap allocation. T comes from the variable the result is assigned to
// ('?&heap char buf = std::alloc(64UL);'); 'unsigned char' for untyped bytes.
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
// Element-wise copies between slices: min(dst.len, src.len) elements, the
// count returned. safe_memmove allows dst and src to overlap.
generic<T> unsigned long safe_memcpy([]T dst, []const T src);
generic<T> unsigned long safe_memmove([]T dst, []const T src);
// Sets every byte of s's elements to val (memset).
generic<T> void safe_memset([]T s, int val);
// The same, for one object: safe_memset_obj(&ev, 0) clears a struct.
generic<T> void safe_memset_obj(&T v, int val);
// Byte-wise comparison (memcmp) of a and b over their common length; when
// that prefix is equal, the shorter slice compares less.
generic<T> int  safe_memcmp([]const T a, []const T b);

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

// Zero s (all its elements' bytes) with volatile stores the optimizer can't
// remove (for secrets); mem_zero_secure_obj zeroes one object the same way.
generic<T> void mem_zero_secure([]T s);
generic<T> void mem_zero_secure_obj(&T v);

// Write back and invalidate the cache line containing `addr` (x86 CLFLUSH,
// AArch64 DC CIVAC; a no-op on other targets).
generic<T> void mem_clflush(const &T addr);

// ── Alignment utilities ───────────────────────────────────────────────────────

// Round a reference up to the next `align`-byte boundary (power of two).
generic<T> ?&T mem_align_ptr(&T ptr, unsigned long align);

// Check if `size` bytes fit within a page (4096 bytes) from `addr`.
int           mem_fits_page(unsigned long addr, unsigned long size);

// ── Type-erased forms ─────────────────────────────────────────────────────────
// What the generic wrappers above forward to, and what std's own
// type-erased containers use internally: untyped memory is a byte pointer
// ('unsigned char*'), which a generic wrapper turns into its '?&heap T' /
// '[]T' with a cast inside 'unsafe'. 'new T' (std_alloc_raw_) and
// 'malloc_defer' are backed by alloc_raw_/dealloc_raw_.
unsigned char*  alloc_raw_(unsigned long size);
unsigned char*  alloc_zeroed_raw_(unsigned long size);
void            dealloc_raw_(unsigned char* ptr);
unsigned char*  realloc_buf_raw_(unsigned char* ptr, unsigned long new_size);
// realloc_buf_raw_ to 'count' elements of 'elemSize' bytes (overflow-checked),
// out of line and cold: containers call it on their growth path so that
// the inlined fast path (push, enqueue, ...) keeps its state in registers.
unsigned char*  realloc_array_raw_(unsigned char* ptr, unsigned long count, unsigned long elemSize);
void            safe_memcpy_raw_(unsigned char* dst, const unsigned char* src, unsigned long n);
void            safe_memmove_raw_(unsigned char* dst, const unsigned char* src, unsigned long n);
void            safe_memset_raw_(unsigned char* ptr, int val, unsigned long n);
int             safe_memcmp_raw_(const unsigned char* a, const unsigned char* b, unsigned long n);
void            mem_prefetch_raw_(const unsigned char* addr, int write, int locality);
void            mem_zero_secure_raw_(unsigned char* ptr, unsigned long n);
void            mem_clflush_raw_(const unsigned char* addr);
unsigned char*  mem_align_ptr_raw_(unsigned char* ptr, unsigned long align);

generic<T> ?&heap T alloc(unsigned long size) { unsafe { return (?&heap T)alloc_raw_(size); } }
generic<T> ?&heap T alloc_zeroed(unsigned long size) { unsafe { return (?&heap T)alloc_zeroed_raw_(size); } }
generic<T> void dealloc(?&heap T ptr) { unsafe { dealloc_raw_((unsigned char*)ptr); } }
generic<T> ?&heap T realloc_buf(?&heap T ptr, unsigned long new_size) {
    unsafe { return (?&heap T)realloc_buf_raw_((unsigned char*)ptr, new_size); }
}
generic<T> unsigned long safe_memcpy([]T dst, []const T src) {
    unsigned long n = dst.len < src.len ? dst.len : src.len;
    if (n > 0UL) { unsafe { safe_memcpy_raw_((unsigned char*)dst.ptr, (const unsigned char*)src.ptr, n * sizeof(T)); } }
    return n;
}
generic<T> unsigned long safe_memmove([]T dst, []const T src) {
    unsigned long n = dst.len < src.len ? dst.len : src.len;
    if (n > 0UL) { unsafe { safe_memmove_raw_((unsigned char*)dst.ptr, (const unsigned char*)src.ptr, n * sizeof(T)); } }
    return n;
}
generic<T> void safe_memset([]T s, int val) {
    if (s.len > 0UL) { unsafe { safe_memset_raw_((unsigned char*)s.ptr, val, s.len * sizeof(T)); } }
}
generic<T> void safe_memset_obj(&T v, int val) { unsafe { safe_memset_raw_((unsigned char*)v, val, sizeof(T)); } }
generic<T> int safe_memcmp([]const T a, []const T b) {
    unsigned long n = a.len < b.len ? a.len : b.len;
    int c = 0;
    if (n > 0UL) { unsafe { c = safe_memcmp_raw_((const unsigned char*)a.ptr, (const unsigned char*)b.ptr, n * sizeof(T)); } }
    if (c != 0) { return c; }
    if (a.len < b.len) { return -1; }
    if (a.len > b.len) { return 1; }
    return 0;
}
generic<T> void mem_prefetch(const &T addr, int write, int locality) {
    unsafe { mem_prefetch_raw_((const unsigned char*)addr, write, locality); }
}
generic<T> void mem_zero_secure([]T s) {
    if (s.len > 0UL) { unsafe { mem_zero_secure_raw_((unsigned char*)s.ptr, s.len * sizeof(T)); } }
}
generic<T> void mem_zero_secure_obj(&T v) { unsafe { mem_zero_secure_raw_((unsigned char*)v, sizeof(T)); } }
generic<T> void mem_clflush(const &T addr) { unsafe { mem_clflush_raw_((const unsigned char*)addr); } }
generic<T> ?&T mem_align_ptr(&T ptr, unsigned long align) {
    unsafe { return (?&T)mem_align_ptr_raw_((unsigned char*)ptr, align); }
}

} // namespace std
