// SafeC Standard Library — MMU Context
#pragma once
#include "mmu.h"

extern void* memset(void* ptr, int val, unsigned long n);

// ── Helpers ───────────────────────────────────────────────────────────────────

// Extract L1 index from a 30-bit virtual address (bits [29:21]).
unsigned long mmu_l1_idx_(unsigned long virt) {
    return (virt >> (unsigned long)21) & (unsigned long)0x1FF;
}

// Extract L2 index from a 30-bit virtual address (bits [20:12]).
unsigned long mmu_l2_idx_(unsigned long virt) {
    return (virt >> (unsigned long)12) & (unsigned long)0x1FF;
}

// ── Constructor ───────────────────────────────────────────────────────────────

struct MmuContext mmu_init(unsigned long root, void* frames) {
    struct MmuContext ctx;
    ctx.root   = root;
    ctx.frames = frames;
    return ctx;
}

// ── MmuContext methods ────────────────────────────────────────────────────────

int MmuContext::map(unsigned long virt, unsigned long phys, unsigned int flags) {
    unsigned long l1_idx = mmu_l1_idx_(virt);
    unsigned long l2_idx = mmu_l2_idx_(virt);

    unsafe {
        // L1 table lives at physical address self.root.
        unsigned long* l1 = (unsigned long*)self.root;

        // If L1 entry absent, allocate a new L2 frame.
        if ((l1[l1_idx] & (unsigned long)PAGE_PRESENT) == (unsigned long)0) {
            struct FrameAllocator* fa = (struct FrameAllocator*)self.frames;
            long long frame = fa->alloc();
            if (frame < 0) { return 0; }                   // OOM
            unsigned long l2_phys = (unsigned long)frame * (unsigned long)PAGE_SIZE;
            // Zero the new L2 frame.
            memset((void*)l2_phys, 0, (unsigned long)PAGE_SIZE);
            // Write L1 entry: physical address | PRESENT | WRITABLE.
            l1[l1_idx] = (l2_phys & ~(unsigned long)0xFFF)
                       | (unsigned long)(PAGE_PRESENT | PAGE_WRITABLE);
        }

        // Resolve L2 table physical address from L1 entry.
        unsigned long l2_phys  = l1[l1_idx] & ~(unsigned long)0xFFF;
        unsigned long* l2      = (unsigned long*)l2_phys;

        // Write L2 entry: physical address | caller flags (must include PAGE_PRESENT).
        l2[l2_idx] = (phys & ~(unsigned long)0xFFF) | (unsigned long)flags;
    }
    return 1;
}

void MmuContext::unmap(unsigned long virt) {
    unsigned long l1_idx = mmu_l1_idx_(virt);
    unsigned long l2_idx = mmu_l2_idx_(virt);

    unsafe {
        unsigned long* l1 = (unsigned long*)self.root;
        if ((l1[l1_idx] & (unsigned long)PAGE_PRESENT) == (unsigned long)0) { return; }
        unsigned long l2_phys = l1[l1_idx] & ~(unsigned long)0xFFF;
        unsigned long* l2     = (unsigned long*)l2_phys;
        l2[l2_idx] = (unsigned long)0;
    }
}

int MmuContext::walk(unsigned long virt, &stack unsigned long phys_out) const {
    unsigned long l1_idx = mmu_l1_idx_(virt);
    unsigned long l2_idx = mmu_l2_idx_(virt);

    unsafe {
        unsigned long* l1 = (unsigned long*)self.root;
        if ((l1[l1_idx] & (unsigned long)PAGE_PRESENT) == (unsigned long)0) { return 0; }

        unsigned long l2_phys = l1[l1_idx] & ~(unsigned long)0xFFF;
        unsigned long* l2     = (unsigned long*)l2_phys;

        if ((l2[l2_idx] & (unsigned long)PAGE_PRESENT) == (unsigned long)0) { return 0; }

        unsigned long page_base  = l2[l2_idx] & ~(unsigned long)0xFFF;
        unsigned long page_offset = virt & (unsigned long)0xFFF;
        phys_out = page_base + page_offset;
    }
    return 1;
}

void MmuContext::tlb_flush_all() {
    unsafe {
#ifdef __x86_64__
        // Reload CR3 — flushes all non-global TLB entries.
        asm volatile (
            "mov %%cr3, %%rax\n\t"
            "mov %%rax, %%cr3"
            : : : "rax"
        );
#elif defined(__riscv)
        asm volatile ("sfence.vma x0, x0" : : : "memory");
#elif defined(__aarch64__)
        asm volatile (
            "dsb ishst\n\t"
            "tlbi vmalle1\n\t"
            "dsb ish\n\t"
            "isb"
            : : : "memory"
        );
#endif
    }
}

void MmuContext::tlb_flush_page(unsigned long virt) {
    unsafe {
#ifdef __x86_64__
        asm volatile ("invlpg (%0)" : : "r"(virt) : "memory");
#elif defined(__riscv)
        asm volatile ("sfence.vma %0, x0" : : "r"(virt) : "memory");
#elif defined(__aarch64__)
        // VAE1 operates on the VPN (virtual address >> 12).
        unsigned long vpn = virt >> (unsigned long)12;
        asm volatile (
            "dsb ishst\n\t"
            "tlbi vae1, %0\n\t"
            "dsb ish\n\t"
            "isb"
            : : "r"(vpn) : "memory"
        );
#endif
    }
}

void MmuContext::activate() {
    unsafe {
#ifdef __x86_64__
        asm volatile ("mov %0, %%cr3" : : "r"(self.root) : "memory");
#elif defined(__riscv)
        // Sv30: mode = 8 (Sv39 = 8 in satp), ASID = 0, PPN = root >> 12.
        unsigned long satp = ((unsigned long)8 << (unsigned long)60)
                           | (self.root >> (unsigned long)12);
        asm volatile ("csrw satp, %0\n\t"
                      "sfence.vma x0, x0"
                      : : "r"(satp) : "memory");
#elif defined(__aarch64__)
        asm volatile (
            "msr ttbr0_el1, %0\n\t"
            "isb"
            : : "r"(self.root) : "memory"
        );
#endif
    }
}
