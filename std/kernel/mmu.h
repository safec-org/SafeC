// SafeC Standard Library — MMU Context (2-level page table)
// Virtual-to-physical address translation for freestanding targets.
// Freestanding-safe; TLB/activate operations use inline asm inside unsafe{}.
//
// Page table layout (30-bit virtual address, Sv30-style):
//   L1 (root, 512 entries): index = virt[29:21]  (9 bits)
//   L2 (per-L1,  512 entries): index = virt[20:12]  (9 bits)
//   Page offset: virt[11:0] (12 bits)  →  4 KiB pages
//   Total addressable VA: 512 × 512 × 4 KiB = 1 GiB
//
// Each page table entry uses the same PageEntry format as paging.h:
//   bits [63:12] = physical frame number (physical address >> 12)
//   bits [11:0]  = flags (PAGE_PRESENT, PAGE_WRITABLE, etc.)
#pragma once
#include "paging.h"
#include "frame.h"

// Maximum number of L2 tables cached in one MmuContext.
#define MMU_L2_MAX 512

struct MmuContext {
    unsigned long root;          // physical address of L1 page table (4 KiB aligned)
    void*         frames;        // raw ptr to FrameAllocator — kernel-lifetime object

    // Map virtual address `virt` to physical `phys` with `flags`.
    // Allocates a new L2 frame via `frames` if the L1 entry is absent.
    // Returns 1 on success, 0 on OOM.
    int           map(unsigned long virt, unsigned long phys, unsigned int flags);

    // Unmap virtual address `virt`.  Does not reclaim L2 frames.
    void          unmap(unsigned long virt);

    // Walk the page tables and resolve `virt` to a physical address.
    // Returns 1 and writes *phys_out on success; returns 0 if not mapped.
    int           walk(unsigned long virt, &stack unsigned long phys_out) const;

    // Flush the entire TLB.
    // x86-64: reload CR3.  RISC-V: sfence.vma x0,x0.  ARM64: TLBI VMALLE1.
    void          tlb_flush_all();

    // Flush a single TLB entry for virtual address `virt`.
    // x86-64: invlpg.  RISC-V: sfence.vma virt,x0.  ARM64: TLBI VAE1, virt>>12.
    void          tlb_flush_page(unsigned long virt);

    // Load this context as the active address space.
    // x86-64: mov cr3, root.  RISC-V: csrw satp, (8<<60)|root>>12.
    // ARM64:  msr ttbr0_el1, root; isb.
    void          activate();
};

// Initialise an MmuContext.
// `root` must be the physical address of a zeroed, 4 KiB-aligned L1 page table.
// `frames` must point to a live FrameAllocator with PAGE_SIZE == 4096.
struct MmuContext mmu_init(unsigned long root, void* frames);
