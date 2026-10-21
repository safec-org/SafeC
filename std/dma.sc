// SafeC Standard Library — DMA Implementation
#pragma once
#include <std/dma.h>
#include <std/mem.h>

// ── DmaBuffer ─────────────────────────────────────────────────────────────────

namespace std {

inline struct DmaBuffer dma_buf_wrap(&unsigned char virt, unsigned long phys, unsigned long size) {
    struct DmaBuffer buf;
    buf.virt  = virt;
    buf.phys  = (phys != (unsigned long)0) ? phys : (unsigned long)virt;
    buf.size  = size;
    buf.owner = 0;  // CPU owns initially
    return buf;
}

void DmaBuffer::to_device() {
    // Flush all cache lines in the buffer range.
    unsafe {
        // Start at the line holding the first byte: stepping from an
        // unaligned address would skip the line holding the last one.
        unsigned long p = mem_align_down((unsigned long)self.virt, (unsigned long)MEM_CACHE_LINE_SIZE);
        unsigned long end = (unsigned long)self.virt + self.size;
        while (p < end) {
            mem_clflush_raw_((const void*)p);
            p = p + (unsigned long)MEM_CACHE_LINE_SIZE;
        }
        // Memory barrier to ensure writes are visible before DMA starts.
#ifdef __x86_64__
        asm volatile ("mfence" : : : "memory");
#elif defined(__aarch64__)
        asm volatile ("dsb st" : : : "memory");
#elif defined(__riscv)
        asm volatile ("fence w,o" : : : "memory");
#else
        asm volatile ("" : : : "memory");
#endif
    }
    self.owner = 1;
}

void DmaBuffer::to_cpu() {
    // Invalidate cache lines so CPU sees DMA-written data.
    unsafe {
#ifdef __x86_64__
        asm volatile ("mfence" : : : "memory");
#elif defined(__aarch64__)
        asm volatile ("dsb ld" : : : "memory");
#elif defined(__riscv)
        asm volatile ("fence i,r" : : : "memory");
#else
        asm volatile ("" : : : "memory");
#endif
        // Start at the line holding the first byte: stepping from an
        // unaligned address would skip the line holding the last one.
        unsigned long p = mem_align_down((unsigned long)self.virt, (unsigned long)MEM_CACHE_LINE_SIZE);
        unsigned long end = (unsigned long)self.virt + self.size;
        while (p < end) {
            mem_clflush_raw_((const void*)p);
            p = p + (unsigned long)MEM_CACHE_LINE_SIZE;
        }
        asm volatile ("" : : : "memory");
    }
    self.owner = 0;
}

inline int DmaBuffer::cpu_accessible() const {
    if (self.owner == 0) { return 1; }
    return 0;
}

inline unsigned long DmaBuffer::phys_at(unsigned long offset) const {
    return self.phys + offset;
}


// ── DmaSgList ─────────────────────────────────────────────────────────────────

inline void DmaSgList::add(unsigned long phys, unsigned long len) {
    if (self.count >= DMA_SG_MAX) { return; }
    self.entries[self.count].phys = phys;
    self.entries[self.count].len  = len;
    self.count = self.count + 1;
}

inline void DmaSgList::clear() {
    self.count = 0;
}

inline unsigned long DmaSgList::total_len() const {
    unsigned long total = (unsigned long)0;
    int i = 0;
    while (i < self.count) {
        total = total + self.entries[i].len;
        i = i + 1;
    }
    return total;
}

} // namespace std
