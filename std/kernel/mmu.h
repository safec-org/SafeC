// SafeC Standard Library — MMU Context (real hardware page tables)
// Virtual-to-physical address translation for freestanding targets.
// Freestanding-safe; TLB/activate operations use inline asm inside unsafe{}.
//
// MmuContext<F> builds the page tables the CPU itself walks. F is a
// compile-time page-table format type — every table walk step is a direct,
// inlinable call to F's methods, the same pattern as BST<K, V, C>'s
// ordering. One format per architecture:
//
//   MmuX86_64   x86-64 4-level paging (48-bit VA; PML4/PDPT/PD/PT, 8-byte entries)
//   MmuX86_32   x86 32-bit paging without PAE (2 levels, 4-byte entries)
//   MmuSv39     RISC-V Sv39 (RV64, 3 levels, 8-byte PTEs)
//   MmuSv32     RISC-V Sv32 (RV32, 2 levels, 4-byte PTEs)
//   MmuAArch64  AArch64 stage 1, 4 KiB granule, 48-bit VA (4 levels)
//   MmuArmV7    ARMv7-A short-descriptor (4096-entry L1 + 256-entry L2)
//
// MmuNative names the format of the CPU being compiled for (none on
// Cortex-M, which has an MPU but no MMU, or on wasm). All formats can be
// built and walked on any target — e.g. on the host, for tests or for an
// image builder — but activate()/tlb_flush_*() only touch the CPU when
// the format matches the running bare-metal target (they return 0
// otherwise, and always in a hosted process).
//
// Mappings are 4 KiB pages. walk() also follows larger (block/superpage)
// leaves built by someone else.
//
// Tables are read and written at (physical address + phys_offset): 0 for
// an identity-mapped kernel, the direct-map base for a higher-half one.
// New intermediate tables come from 'frames' (one 4 KiB frame each,
// zeroed); the root table is the caller's.
#pragma once
#include <std/kernel/frame.h>

// Portable mapping flags (map()'s 'flags'). Every mapping is readable.
#define MMU_WRITE   1   // writable
#define MMU_EXEC    2   // executable
#define MMU_USER    4   // reachable from user mode (x86 U/S, RISC-V U, ARM EL0/PL0)
#define MMU_DEVICE  8   // uncached device memory (x86 PCD|PWT, ARM device
                        // attributes; RISC-V: no effect — PMAs decide)

namespace std {

// ── Formats ──────────────────────────────────────────────────────────────────
// Each format has the same methods (unused_ only gives the struct a size):
//   levels()            number of table levels
//   shift(l)            VA bit where level l's index starts (l = 0 is the root)
//   entries(l)          entries in a level-l table
//   esize()             bytes per entry (4 or 8)
//   table_desc(p)       entry pointing at the next-level table at physical p
//   leaf_desc(p, f)     4 KiB page entry for physical p with MMU_* flags f
//   valid(e)            entry present
//   is_leaf(e, l)       valid entry that maps memory (not a table pointer)
//   addr(e, l)          physical address an entry points at
//   load_root(root)     install root on this CPU (CR3/satp/TTBR0/TTBR0)
//   flush_all()/flush_page(v)  TLB maintenance
// load_root/flush_* return 1 if they ran, 0 if this isn't the running CPU.

// Non-leaf entries carry P|RW|US so the leaf alone decides access; non-exec
// pages set NX (bit 63), which needs EFER.NXE enabled.
struct MmuX86_64 {
    int                unused_;
    int                levels() const;
    int                shift(int level) const;
    unsigned long      entries(int level) const;
    unsigned long      esize() const;
    unsigned long long table_desc(unsigned long phys) const;
    unsigned long long leaf_desc(unsigned long phys, unsigned int flags) const;
    int                valid(unsigned long long e) const;
    int                is_leaf(unsigned long long e, int level) const;
    unsigned long      addr(unsigned long long e, int level) const;
    int                load_root(unsigned long root) const;
    int                flush_all() const;
    int                flush_page(unsigned long virt) const;
};
// No NX without PAE: MMU_EXEC is implied.
struct MmuX86_32 {
    int                unused_;
    int                levels() const;
    int                shift(int level) const;
    unsigned long      entries(int level) const;
    unsigned long      esize() const;
    unsigned long long table_desc(unsigned long phys) const;
    unsigned long long leaf_desc(unsigned long phys, unsigned int flags) const;
    int                valid(unsigned long long e) const;
    int                is_leaf(unsigned long long e, int level) const;
    unsigned long      addr(unsigned long long e, int level) const;
    int                load_root(unsigned long root) const;
    int                flush_all() const;
    int                flush_page(unsigned long virt) const;
};
// Leaves set A and D up front (no faults on hardware without A/D updates).
struct MmuSv39 {
    int                unused_;
    int                levels() const;
    int                shift(int level) const;
    unsigned long      entries(int level) const;
    unsigned long      esize() const;
    unsigned long long table_desc(unsigned long phys) const;
    unsigned long long leaf_desc(unsigned long phys, unsigned int flags) const;
    int                valid(unsigned long long e) const;
    int                is_leaf(unsigned long long e, int level) const;
    unsigned long      addr(unsigned long long e, int level) const;
    int                load_root(unsigned long root) const;
    int                flush_all() const;
    int                flush_page(unsigned long virt) const;
};
struct MmuSv32 {
    int                unused_;
    int                levels() const;
    int                shift(int level) const;
    unsigned long      entries(int level) const;
    unsigned long      esize() const;
    unsigned long long table_desc(unsigned long phys) const;
    unsigned long long leaf_desc(unsigned long phys, unsigned int flags) const;
    int                valid(unsigned long long e) const;
    int                is_leaf(unsigned long long e, int level) const;
    unsigned long      addr(unsigned long long e, int level) const;
    int                load_root(unsigned long root) const;
    int                flush_all() const;
    int                flush_page(unsigned long virt) const;
};
// Leaves use MAIR attribute index 0 for normal memory and 1 for device
// memory (program MAIR_EL1 with Attr0 = 0xFF, Attr1 = 0x04), inner-shareable,
// AF set. TCR_EL1 must select a 4 KiB granule and T0SZ = 16.
struct MmuAArch64 {
    int                unused_;
    int                levels() const;
    int                shift(int level) const;
    unsigned long      entries(int level) const;
    unsigned long      esize() const;
    unsigned long long table_desc(unsigned long phys) const;
    unsigned long long leaf_desc(unsigned long phys, unsigned int flags) const;
    int                valid(unsigned long long e) const;
    int                is_leaf(unsigned long long e, int level) const;
    unsigned long      addr(unsigned long long e, int level) const;
    int                load_root(unsigned long root) const;
    int                flush_all() const;
    int                flush_page(unsigned long virt) const;
};
// Small pages; the root L1 table is 16 KiB and must be 16 KiB aligned.
// Normal memory is TEX=001 C=1 B=1 (write-back), device is TEX=000 C=0 B=1.
struct MmuArmV7 {
    int                unused_;
    int                levels() const;
    int                shift(int level) const;
    unsigned long      entries(int level) const;
    unsigned long      esize() const;
    unsigned long long table_desc(unsigned long phys) const;
    unsigned long long leaf_desc(unsigned long phys, unsigned int flags) const;
    int                valid(unsigned long long e) const;
    int                is_leaf(unsigned long long e, int level) const;
    unsigned long      addr(unsigned long long e, int level) const;
    int                load_root(unsigned long root) const;
    int                flush_all() const;
    int                flush_page(unsigned long virt) const;
};

#if defined(__x86_64__)
typedef struct MmuX86_64 MmuNative;
#elif defined(__i386__)
typedef struct MmuX86_32 MmuNative;
#elif defined(__riscv) && __riscv_xlen == 64
typedef struct MmuSv39 MmuNative;
#elif defined(__riscv)
typedef struct MmuSv32 MmuNative;
#elif defined(__aarch64__)
typedef struct MmuAArch64 MmuNative;
#elif defined(__arm__) && defined(__ARM_ARCH_PROFILE) && __ARM_ARCH_PROFILE == 'A'
typedef struct MmuArmV7 MmuNative;
#endif

// ── Context ──────────────────────────────────────────────────────────────────

generic<F>
struct MmuContext {
    unsigned long          root;         // physical address of the root table
    unsigned long          phys_offset;  // tables live at phys + phys_offset
    &static FrameAllocator frames;       // source of new intermediate tables
    F                      fmt;

    // Map the 4 KiB page at 'virt' to 'phys' (both 4 KiB aligned; low bits
    // ignored) with MMU_* flags, allocating intermediate tables as needed.
    // Returns 1, or 0 if a table frame couldn't be allocated or a larger
    // leaf already covers 'virt'. Remapping a page overwrites it; flush
    // the TLB entry afterwards if the context is live.
    int  map(unsigned long virt, unsigned long phys, unsigned int flags);

    // Remove the page at 'virt' (no-op if unmapped). Tables aren't freed.
    void unmap(unsigned long virt);

    // Resolve 'virt': 1 and *phys_out on success, 0 if unmapped.
    int  walk(unsigned long virt, &unsigned long phys_out) const;

    // The raw leaf entry mapping 'virt': 1 and *entry_out, or 0 if unmapped.
    int  entry(unsigned long virt, &unsigned long long entry_out) const;

    // TLB maintenance and switching to this address space: x86 reloads CR3 /
    // invlpg; RISC-V writes satp (which also turns translation on for S/U
    // mode) / sfence.vma; AArch64 writes TTBR0_EL1 / TLBI; ARMv7 writes
    // TTBR0 / TLBIALL. Turning paging on (CR0.PG, SCTLR.M) and the MAIR/TCR
    // setup are the boot code's. Each returns 1 if it ran, 0 if F isn't the
    // running CPU's format (or in a hosted process).
    int  tlb_flush_all();
    int  tlb_flush_page(unsigned long virt);
    int  activate();

    // Internal: pointer to the table at physical address 'phys'.
    unsigned long table_(unsigned long phys) const;
    unsigned long long read_(unsigned long table, unsigned long idx) const;
    void write_(unsigned long table, unsigned long idx, unsigned long long e);
    unsigned long index_(unsigned long virt, int level) const;
};

// A context over an existing root table at physical 'root' — zeroed, and
// aligned as the format needs (4 KiB; 16 KiB for MmuArmV7). 'frames' must be
// a live, &static-lifetime FrameAllocator (the kernel's own, usually),
// handing out 4 KiB frames whose memory is at frame * 4096 + phys_offset.
// F comes from the target type:
//   struct MmuContext<MmuNative> mmu = std::mmu_init(root, &g_frames, 0UL);
generic<F>
struct MmuContext<F> mmu_init(unsigned long root, &static FrameAllocator frames,
                              unsigned long phys_offset) {
    struct MmuContext<F> m;
    m.root = root;
    m.phys_offset = phys_offset;
    m.frames = frames;
    return m;
}

generic<F>
unsigned long MmuContext::table_(unsigned long phys) const {
    return phys + self.phys_offset;
}

generic<F>
unsigned long long MmuContext::read_(unsigned long table, unsigned long idx) const {
    unsafe {
        if (self.fmt.esize() == 8UL) { return ((unsigned long long*)table)[idx]; }
        return (unsigned long long)((unsigned int*)table)[idx];
    }
}

generic<F>
void MmuContext::write_(unsigned long table, unsigned long idx, unsigned long long e) {
    unsafe {
        if (self.fmt.esize() == 8UL) { ((unsigned long long*)table)[idx] = e; }
        else { ((unsigned int*)table)[idx] = (unsigned int)e; }
    }
}

generic<F>
unsigned long MmuContext::index_(unsigned long virt, int level) const {
    unsigned long sh = (unsigned long)self.fmt.shift(level);
    if (sh >= 8UL * sizeof(unsigned long)) { return 0UL; }   // 64-bit format on a 32-bit build
    return (virt >> sh) & (self.fmt.entries(level) - 1UL);
}

generic<F>
int MmuContext::map(unsigned long virt, unsigned long phys, unsigned int flags) {
    unsigned long table = self.table_(self.root);
    int last = self.fmt.levels() - 1;
    int l = 0;
    while (l < last) {
        unsigned long idx = self.index_(virt, l);
        unsigned long long e = self.read_(table, idx);
        if (!self.fmt.valid(e)) {
            long long frame = self.frames.alloc();
            if (frame < 0LL) { return 0; }
            unsigned long next = (unsigned long)frame * 4096UL;
            unsafe {
                unsigned char* t = (unsigned char*)self.table_(next);
                unsigned long k = 0UL;
                while (k < 4096UL) { t[k] = (unsigned char)0; k = k + 1UL; }
            }
            e = self.fmt.table_desc(next);
            self.write_(table, idx, e);
        } else if (self.fmt.is_leaf(e, l)) {
            return 0;                    // a larger page already maps this range
        }
        table = self.table_(self.fmt.addr(e, l));
        l = l + 1;
    }
    self.write_(table, self.index_(virt, last), self.fmt.leaf_desc(phys & ~4095UL, flags));
    return 1;
}

generic<F>
void MmuContext::unmap(unsigned long virt) {
    unsigned long table = self.table_(self.root);
    int last = self.fmt.levels() - 1;
    int l = 0;
    while (l < last) {
        unsigned long long e = self.read_(table, self.index_(virt, l));
        if (!self.fmt.valid(e) || self.fmt.is_leaf(e, l)) { return; }
        table = self.table_(self.fmt.addr(e, l));
        l = l + 1;
    }
    self.write_(table, self.index_(virt, last), 0ULL);
}

generic<F>
int MmuContext::entry(unsigned long virt, &unsigned long long entry_out) const {
    unsigned long table = self.table_(self.root);
    int last = self.fmt.levels() - 1;
    int l = 0;
    while (1) {
        unsigned long long e = self.read_(table, self.index_(virt, l));
        if (!self.fmt.valid(e)) { return 0; }
        if (l == last || self.fmt.is_leaf(e, l)) {
            *entry_out = e;
            return l + 1;                // 1 + the level the leaf is at
        }
        table = self.table_(self.fmt.addr(e, l));
        l = l + 1;
    }
    return 0;
}

generic<F>
int MmuContext::walk(unsigned long virt, &unsigned long phys_out) const {
    unsigned long table = self.table_(self.root);
    int last = self.fmt.levels() - 1;
    int l = 0;
    while (1) {
        unsigned long long e = self.read_(table, self.index_(virt, l));
        if (!self.fmt.valid(e)) { return 0; }
        if (l == last || self.fmt.is_leaf(e, l)) {
            unsigned long sh = (unsigned long)self.fmt.shift(l);       // page or block size
            unsigned long low = sh >= 8UL * sizeof(unsigned long) ? ~0UL : (1UL << sh) - 1UL;
            *phys_out = self.fmt.addr(e, l) + (virt & low);
            return 1;
        }
        table = self.table_(self.fmt.addr(e, l));
        l = l + 1;
    }
    return 0;
}

generic<F>
int MmuContext::tlb_flush_all() { return self.fmt.flush_all(); }

generic<F>
int MmuContext::tlb_flush_page(unsigned long virt) { return self.fmt.flush_page(virt); }

generic<F>
int MmuContext::activate() { return self.fmt.load_root(self.root); }

} // namespace std
