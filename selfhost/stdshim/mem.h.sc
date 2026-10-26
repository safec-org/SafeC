// stdshim/mem.h.sc — playground implementation of <std/mem.h>'s out-of-line
// functions on top of the scc runtime allocator (appended to the header).
namespace std {

unsigned long checked_mul_size(unsigned long a, unsigned long b) {
    if (a != 0UL && b > 18446744073709551615UL / a) { abort(); }
    return a * b;
}
unsigned long checked_add_size(unsigned long a, unsigned long b) {
    if (a + b < a) { abort(); }
    return a + b;
}
unsigned long mem_align_up(unsigned long addr, unsigned long align) {
    return (addr + align - 1UL) & ~(align - 1UL);
}
unsigned long mem_align_down(unsigned long addr, unsigned long align) {
    return addr & ~(align - 1UL);
}
int mem_is_aligned(unsigned long addr, unsigned long align) {
    return (addr & (align - 1UL)) == 0UL;
}
int mem_fits_page(unsigned long addr, unsigned long size) {
    return (addr & 4095UL) + size <= 4096UL;
}
void* alloc_raw_(unsigned long size) { return malloc(size); }
void* alloc_zeroed_raw_(unsigned long size) { return calloc(1UL, size); }
void dealloc_raw_(void* ptr) { free(ptr); }
void* realloc_buf_raw_(void* ptr, unsigned long new_size) { return realloc(ptr, new_size); }
void* realloc_array_raw_(void* ptr, unsigned long count, unsigned long elemSize) {
    return realloc(ptr, checked_mul_size(count, elemSize));
}
void safe_memcpy_raw_(void* dst, const void* src, unsigned long n) { memcpy(dst, src, n); }
void safe_memmove_raw_(void* dst, const void* src, unsigned long n) { memmove(dst, src, n); }
void safe_memset_raw_(void* ptr, int val, unsigned long n) { memset(ptr, val, n); }
int safe_memcmp_raw_(const void* a, const void* b, unsigned long n) { return memcmp(a, b, n); }
void mem_prefetch_raw_(const void* addr, int write, int locality) { }
void mem_zero_secure_raw_(void* ptr, unsigned long n) { memset(ptr, 0, n); }
void mem_clflush_raw_(const void* addr) { }
void* mem_align_ptr_raw_(void* ptr, unsigned long align) {
    unsigned long a = 0UL;
    unsafe { a = (unsigned long)ptr; }
    unsigned long r = mem_align_up(a, align);
    unsafe { return (void*)r; }
}

} // namespace std
