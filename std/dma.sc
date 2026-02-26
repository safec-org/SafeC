// SafeC Standard Library — DMA Implementation
#pragma once
#include "dma.h"
#include "mem.h"

// ── DmaBuffer ─────────────────────────────────────────────────────────────────

struct DmaBuffer dma_buf_wrap(void* virt, unsigned long phys, unsigned long size) {
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
        unsigned long p = (unsigned long)self.virt;
        unsigned long end = p + self.size;
        while (p < end) {
            mem_clflush((const void*)p);
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
        unsigned long p = (unsigned long)self.virt;
        unsigned long end = p + self.size;
        while (p < end) {
            mem_clflush((const void*)p);
            p = p + (unsigned long)MEM_CACHE_LINE_SIZE;
        }
        asm volatile ("" : : : "memory");
    }
    self.owner = 0;
}

int DmaBuffer::cpu_accessible() const {
    if (self.owner == 0) { return 1; }
    return 0;
}

unsigned long DmaBuffer::phys_at(unsigned long offset) const {
    return self.phys + offset;
}

// ── DmaChannel ────────────────────────────────────────────────────────────────

int DmaChannel::start(unsigned long src_phys, unsigned long dst_phys, unsigned long len) {
    if (self.start_fn == (void*)0) { return -1; }
    if (self.busy != 0) { return -1; }
    unsafe {
        int (*fn)(void*, unsigned long, unsigned long, unsigned long) =
            (int (*)(void*, unsigned long, unsigned long, unsigned long))self.start_fn;
        int rc = fn(self.ctx, src_phys, dst_phys, len);
        if (rc == 0) { self.busy = 1; }
        return rc;
    }
    return -1;
}

int DmaChannel::wait(unsigned int max_polls) {
    if (self.poll_fn == (void*)0) {
        self.busy = 0;
        return 0;
    }
    unsigned int i = (unsigned int)0;
    while (i < max_polls) {
        int done;
        unsafe {
            int (*fn)(void*) = (int (*)(void*))self.poll_fn;
            done = fn(self.ctx);
        }
        if (done != 0) {
            self.busy = 0;
            return 0;
        }
        i = i + (unsigned int)1;
    }
    return -1;
}

int DmaChannel::is_busy() const {
    if (self.busy != 0) { return 1; }
    return 0;
}

// ── DmaSgList ─────────────────────────────────────────────────────────────────

void DmaSgList::add(unsigned long phys, unsigned long len) {
    if (self.count >= DMA_SG_MAX) { return; }
    self.entries[self.count].phys = phys;
    self.entries[self.count].len  = len;
    self.count = self.count + 1;
}

void DmaSgList::clear() {
    self.count = 0;
}

unsigned long DmaSgList::total_len() const {
    unsigned long total = (unsigned long)0;
    int i = 0;
    while (i < self.count) {
        total = total + self.entries[i].len;
        i = i + 1;
    }
    return total;
}
