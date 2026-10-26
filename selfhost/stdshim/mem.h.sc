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
unsigned char* alloc_raw_(unsigned long size) { unsafe { return (unsigned char*)malloc(size); } }
unsigned char* alloc_zeroed_raw_(unsigned long size) { unsafe { return (unsigned char*)calloc(1UL, size); } }
void dealloc_raw_(unsigned char* ptr) { free(ptr); }
unsigned char* realloc_buf_raw_(unsigned char* ptr, unsigned long new_size) { unsafe { return (unsigned char*)realloc(ptr, new_size); } }
unsigned char* realloc_array_raw_(unsigned char* ptr, unsigned long count, unsigned long elemSize) {
    unsafe { return (unsigned char*)realloc(ptr, checked_mul_size(count, elemSize)); }
}
void safe_memcpy_raw_(unsigned char* dst, const unsigned char* src, unsigned long n) { memcpy(dst, src, n); }
void safe_memmove_raw_(unsigned char* dst, const unsigned char* src, unsigned long n) { memmove(dst, src, n); }
void safe_memset_raw_(unsigned char* ptr, int val, unsigned long n) { memset(ptr, val, n); }
int safe_memcmp_raw_(const unsigned char* a, const unsigned char* b, unsigned long n) { return memcmp(a, b, n); }
void mem_prefetch_raw_(const unsigned char* addr, int write, int locality) { }
void mem_zero_secure_raw_(unsigned char* ptr, unsigned long n) { memset(ptr, 0, n); }
void mem_clflush_raw_(const unsigned char* addr) { }
unsigned char* mem_align_ptr_raw_(unsigned char* ptr, unsigned long align) {
    unsigned long a = 0UL;
    unsafe { a = (unsigned long)ptr; }
    unsigned long r = mem_align_up(a, align);
    unsafe { return (unsigned char*)r; }
}

} // namespace std
