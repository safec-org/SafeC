// SafeC Standard Library — MMU page-table formats (see mmu.h)
#pragma once
#include <std/kernel/mmu.h>

namespace std {

// Mask keeping the bits of 'addr' above a 'shift'-bit page/block size.
static unsigned long mmu_align_(unsigned long long addr, int shift) {
    unsigned long long low = ((unsigned long long)1 << (unsigned long long)shift) - 1ULL;
    return (unsigned long)(addr & ~low);
}

// ── x86-64: 4-level, 512 x 8-byte entries ────────────────────────────────────
#define X86_P_    0x001ULL
#define X86_RW_   0x002ULL
#define X86_US_   0x004ULL
#define X86_PWT_  0x008ULL
#define X86_PCD_  0x010ULL
#define X86_A_    0x020ULL
#define X86_D_    0x040ULL
#define X86_PS_   0x080ULL
#define X86_NX_   0x8000000000000000ULL
#define X86_64_ADDR_ 0x000FFFFFFFFFF000ULL

static unsigned long long x86_leaf_bits_(unsigned int flags) {
    unsigned long long e = X86_P_ | X86_A_;
    if ((flags & MMU_WRITE) != 0u)  { e = e | X86_RW_ | X86_D_; }
    if ((flags & MMU_USER) != 0u)   { e = e | X86_US_; }
    if ((flags & MMU_DEVICE) != 0u) { e = e | X86_PCD_ | X86_PWT_; }
    return e;
}

int                MmuX86_64::levels() const { return 4; }
int                MmuX86_64::shift(int level) const { return 39 - 9 * level; }
unsigned long      MmuX86_64::entries(int level) const { return 512UL; }
unsigned long      MmuX86_64::esize() const { return 8UL; }
unsigned long long MmuX86_64::table_desc(unsigned long phys) const {
    return ((unsigned long long)phys & X86_64_ADDR_) | X86_P_ | X86_RW_ | X86_US_;
}
unsigned long long MmuX86_64::leaf_desc(unsigned long phys, unsigned int flags) const {
    unsigned long long e = ((unsigned long long)phys & X86_64_ADDR_) | x86_leaf_bits_(flags);
    if ((flags & MMU_EXEC) == 0u) { e = e | X86_NX_; }
    return e;
}
int MmuX86_64::valid(unsigned long long e) const { return (e & X86_P_) != 0ULL; }
int MmuX86_64::is_leaf(unsigned long long e, int level) const {
    if ((e & X86_P_) == 0ULL) { return 0; }
    return level == 3 || ((level == 1 || level == 2) && (e & X86_PS_) != 0ULL);
}
unsigned long MmuX86_64::addr(unsigned long long e, int level) const {
    if (level < 3 && (e & X86_PS_) != 0ULL) { return mmu_align_(e & X86_64_ADDR_, 39 - 9 * level); }
    return (unsigned long)(e & X86_64_ADDR_);
}
int MmuX86_64::load_root(unsigned long root) const {
#if defined(__x86_64__) && defined(__SAFEC_FREESTANDING__)
    unsafe { asm volatile ("mov %0, %%cr3" : : "r"(root) : "memory"); }
    return 1;
#else
    return 0;
#endif
}
int MmuX86_64::flush_all() const {
#if defined(__x86_64__) && defined(__SAFEC_FREESTANDING__)
    unsafe { asm volatile ("mov %%cr3, %%rax\n\tmov %%rax, %%cr3" : : : "rax", "memory"); }
    return 1;
#else
    return 0;
#endif
}
int MmuX86_64::flush_page(unsigned long virt) const {
#if defined(__x86_64__) && defined(__SAFEC_FREESTANDING__)
    unsafe { asm volatile ("invlpg (%0)" : : "r"(virt) : "memory"); }
    return 1;
#else
    return 0;
#endif
}

// ── x86 32-bit, no PAE: 2-level, 1024 x 4-byte entries ───────────────────────
int                MmuX86_32::levels() const { return 2; }
int                MmuX86_32::shift(int level) const { return level == 0 ? 22 : 12; }
unsigned long      MmuX86_32::entries(int level) const { return 1024UL; }
unsigned long      MmuX86_32::esize() const { return 4UL; }
unsigned long long MmuX86_32::table_desc(unsigned long phys) const {
    return ((unsigned long long)phys & 0xFFFFF000ULL) | X86_P_ | X86_RW_ | X86_US_;
}
unsigned long long MmuX86_32::leaf_desc(unsigned long phys, unsigned int flags) const {
    return ((unsigned long long)phys & 0xFFFFF000ULL) | x86_leaf_bits_(flags);
}
int MmuX86_32::valid(unsigned long long e) const { return (e & X86_P_) != 0ULL; }
int MmuX86_32::is_leaf(unsigned long long e, int level) const {
    if ((e & X86_P_) == 0ULL) { return 0; }
    return level == 1 || (e & X86_PS_) != 0ULL;            // PSE 4 MiB page at level 0
}
unsigned long MmuX86_32::addr(unsigned long long e, int level) const {
    if (level == 0 && (e & X86_PS_) != 0ULL) { return (unsigned long)(e & 0xFFC00000ULL); }
    return (unsigned long)(e & 0xFFFFF000ULL);
}
int MmuX86_32::load_root(unsigned long root) const {
#if defined(__i386__) && defined(__SAFEC_FREESTANDING__)
    unsafe { asm volatile ("mov %0, %%cr3" : : "r"(root) : "memory"); }
    return 1;
#else
    return 0;
#endif
}
int MmuX86_32::flush_all() const {
#if defined(__i386__) && defined(__SAFEC_FREESTANDING__)
    unsafe { asm volatile ("mov %%cr3, %%eax\n\tmov %%eax, %%cr3" : : : "eax", "memory"); }
    return 1;
#else
    return 0;
#endif
}
int MmuX86_32::flush_page(unsigned long virt) const {
#if defined(__i386__) && defined(__SAFEC_FREESTANDING__)
    unsafe { asm volatile ("invlpg (%0)" : : "r"(virt) : "memory"); }
    return 1;
#else
    return 0;
#endif
}

// ── RISC-V Sv39 / Sv32 ───────────────────────────────────────────────────────
#define RV_V_ 0x01ULL
#define RV_R_ 0x02ULL
#define RV_W_ 0x04ULL
#define RV_X_ 0x08ULL
#define RV_U_ 0x10ULL
#define RV_A_ 0x40ULL
#define RV_D_ 0x80ULL

static unsigned long long rv_leaf_bits_(unsigned int flags) {
    unsigned long long e = RV_V_ | RV_R_ | RV_A_ | RV_D_;
    if ((flags & MMU_WRITE) != 0u) { e = e | RV_W_; }
    if ((flags & MMU_EXEC) != 0u)  { e = e | RV_X_; }
    if ((flags & MMU_USER) != 0u)  { e = e | RV_U_; }
    return e;                                      // MMU_DEVICE: PMAs decide
}
static int rv_is_leaf_(unsigned long long e) {
    return (e & RV_V_) != 0ULL && (e & (RV_R_ | RV_W_ | RV_X_)) != 0ULL;
}

int                MmuSv39::levels() const { return 3; }
int                MmuSv39::shift(int level) const { return 30 - 9 * level; }
unsigned long      MmuSv39::entries(int level) const { return 512UL; }
unsigned long      MmuSv39::esize() const { return 8UL; }
unsigned long long MmuSv39::table_desc(unsigned long phys) const {
    return (((unsigned long long)phys >> 12ULL) << 10ULL) | RV_V_;
}
unsigned long long MmuSv39::leaf_desc(unsigned long phys, unsigned int flags) const {
    return (((unsigned long long)phys >> 12ULL) << 10ULL) | rv_leaf_bits_(flags);
}
int MmuSv39::valid(unsigned long long e) const { return (e & RV_V_) != 0ULL; }
int MmuSv39::is_leaf(unsigned long long e, int level) const { return rv_is_leaf_(e); }
unsigned long MmuSv39::addr(unsigned long long e, int level) const {
    unsigned long long pa = ((e >> 10ULL) & 0xFFFFFFFFFFFULL) << 12ULL;   // 44-bit PPN
    if (rv_is_leaf_(e)) { return mmu_align_(pa, 30 - 9 * level); }
    return (unsigned long)pa;
}
int MmuSv39::load_root(unsigned long root) const {
#if defined(__riscv) && __riscv_xlen == 64 && defined(__SAFEC_FREESTANDING__)
    unsigned long satp = (8UL << 60UL) | (root >> 12UL);   // MODE = Sv39
    unsafe { asm volatile ("csrw satp, %0\n\tsfence.vma x0, x0" : : "r"(satp) : "memory"); }
    return 1;
#else
    return 0;
#endif
}
int MmuSv39::flush_all() const {
#if defined(__riscv) && __riscv_xlen == 64 && defined(__SAFEC_FREESTANDING__)
    unsafe { asm volatile ("sfence.vma x0, x0" : : : "memory"); }
    return 1;
#else
    return 0;
#endif
}
int MmuSv39::flush_page(unsigned long virt) const {
#if defined(__riscv) && __riscv_xlen == 64 && defined(__SAFEC_FREESTANDING__)
    unsafe { asm volatile ("sfence.vma %0, x0" : : "r"(virt) : "memory"); }
    return 1;
#else
    return 0;
#endif
}

int                MmuSv32::levels() const { return 2; }
int                MmuSv32::shift(int level) const { return level == 0 ? 22 : 12; }
unsigned long      MmuSv32::entries(int level) const { return 1024UL; }
unsigned long      MmuSv32::esize() const { return 4UL; }
unsigned long long MmuSv32::table_desc(unsigned long phys) const {
    return (((unsigned long long)phys >> 12ULL) << 10ULL) | RV_V_;
}
unsigned long long MmuSv32::leaf_desc(unsigned long phys, unsigned int flags) const {
    return (((unsigned long long)phys >> 12ULL) << 10ULL) | rv_leaf_bits_(flags);
}
int MmuSv32::valid(unsigned long long e) const { return (e & RV_V_) != 0ULL; }
int MmuSv32::is_leaf(unsigned long long e, int level) const { return rv_is_leaf_(e); }
unsigned long MmuSv32::addr(unsigned long long e, int level) const {
    unsigned long long pa = ((e >> 10ULL) & 0x3FFFFFULL) << 12ULL;        // 22-bit PPN
    if (rv_is_leaf_(e)) { return mmu_align_(pa, level == 0 ? 22 : 12); }
    return (unsigned long)pa;
}
int MmuSv32::load_root(unsigned long root) const {
#if defined(__riscv) && __riscv_xlen == 32 && defined(__SAFEC_FREESTANDING__)
    unsigned long satp = 0x80000000UL | (root >> 12UL);   // MODE = Sv32
    unsafe { asm volatile ("csrw satp, %0\n\tsfence.vma x0, x0" : : "r"(satp) : "memory"); }
    return 1;
#else
    return 0;
#endif
}
int MmuSv32::flush_all() const {
#if defined(__riscv) && __riscv_xlen == 32 && defined(__SAFEC_FREESTANDING__)
    unsafe { asm volatile ("sfence.vma x0, x0" : : : "memory"); }
    return 1;
#else
    return 0;
#endif
}
int MmuSv32::flush_page(unsigned long virt) const {
#if defined(__riscv) && __riscv_xlen == 32 && defined(__SAFEC_FREESTANDING__)
    unsafe { asm volatile ("sfence.vma %0, x0" : : "r"(virt) : "memory"); }
    return 1;
#else
    return 0;
#endif
}

// ── AArch64 stage 1, 4 KiB granule, 48-bit VA ────────────────────────────────
#define A64_ADDR_  0x0000FFFFFFFFF000ULL
#define A64_AF_    0x400ULL                 // access flag
#define A64_ISH_   0x300ULL                 // SH = inner shareable
#define A64_DEV_   0x004ULL                 // AttrIndx = 1 (device)
#define A64_EL0_   0x040ULL                 // AP[1]: EL0 access
#define A64_RO_    0x080ULL                 // AP[2]: read-only
#define A64_PXN_   0x0020000000000000ULL
#define A64_UXN_   0x0040000000000000ULL

int                MmuAArch64::levels() const { return 4; }
int                MmuAArch64::shift(int level) const { return 39 - 9 * level; }
unsigned long      MmuAArch64::entries(int level) const { return 512UL; }
unsigned long      MmuAArch64::esize() const { return 8UL; }
unsigned long long MmuAArch64::table_desc(unsigned long phys) const {
    return ((unsigned long long)phys & A64_ADDR_) | 3ULL;
}
unsigned long long MmuAArch64::leaf_desc(unsigned long phys, unsigned int flags) const {
    unsigned long long e = ((unsigned long long)phys & A64_ADDR_) | 3ULL | A64_AF_;
    if ((flags & MMU_DEVICE) != 0u) { e = e | A64_DEV_ | A64_PXN_ | A64_UXN_; }
    else { e = e | A64_ISH_; }
    if ((flags & MMU_USER) != 0u)  { e = e | A64_EL0_; }
    if ((flags & MMU_WRITE) == 0u) { e = e | A64_RO_; }
    if ((flags & MMU_EXEC) == 0u) { e = e | A64_PXN_ | A64_UXN_; }
    else if ((flags & MMU_USER) != 0u) { e = e | A64_PXN_; }   // user code: not kernel-executable
    else { e = e | A64_UXN_; }
    return e;
}
int MmuAArch64::valid(unsigned long long e) const { return (e & 1ULL) != 0ULL; }
int MmuAArch64::is_leaf(unsigned long long e, int level) const {
    if ((e & 1ULL) == 0ULL) { return 0; }
    if (level == 3) { return 1; }
    return (e & 2ULL) == 0ULL;                             // block descriptor (L1/L2)
}
unsigned long MmuAArch64::addr(unsigned long long e, int level) const {
    if (level < 3 && (e & 2ULL) == 0ULL) { return mmu_align_(e & A64_ADDR_, 39 - 9 * level); }
    return (unsigned long)(e & A64_ADDR_);
}
int MmuAArch64::load_root(unsigned long root) const {
#if defined(__aarch64__) && defined(__SAFEC_FREESTANDING__)
    unsafe {
        asm volatile ("msr ttbr0_el1, %0\n\tisb\n\tdsb ishst\n\ttlbi vmalle1\n\tdsb ish\n\tisb"
                      : : "r"(root) : "memory");
    }
    return 1;
#else
    return 0;
#endif
}
int MmuAArch64::flush_all() const {
#if defined(__aarch64__) && defined(__SAFEC_FREESTANDING__)
    unsafe { asm volatile ("dsb ishst\n\ttlbi vmalle1\n\tdsb ish\n\tisb" : : : "memory"); }
    return 1;
#else
    return 0;
#endif
}
int MmuAArch64::flush_page(unsigned long virt) const {
#if defined(__aarch64__) && defined(__SAFEC_FREESTANDING__)
    unsigned long vpn = virt >> 12UL;
    unsafe { asm volatile ("dsb ishst\n\ttlbi vae1, %0\n\tdsb ish\n\tisb" : : "r"(vpn) : "memory"); }
    return 1;
#else
    return 0;
#endif
}

// ── ARMv7-A short descriptors: 4096-entry L1, 256-entry L2, 4-byte entries ───
int                MmuArmV7::levels() const { return 2; }
int                MmuArmV7::shift(int level) const { return level == 0 ? 20 : 12; }
unsigned long      MmuArmV7::entries(int level) const { return level == 0 ? 4096UL : 256UL; }
unsigned long      MmuArmV7::esize() const { return 4UL; }
unsigned long long MmuArmV7::table_desc(unsigned long phys) const {
    return ((unsigned long long)phys & 0xFFFFFC00ULL) | 1ULL;     // page table, domain 0
}
unsigned long long MmuArmV7::leaf_desc(unsigned long phys, unsigned int flags) const {
    unsigned long long e = ((unsigned long long)phys & 0xFFFFF000ULL) | 2ULL;   // small page
    if ((flags & MMU_EXEC) == 0u)   { e = e | 1ULL; }                           // XN
    if ((flags & MMU_DEVICE) != 0u) { e = e | 0x004ULL; }                       // TEX=000 C=0 B=1
    else { e = e | 0x040ULL | 0x008ULL | 0x004ULL | 0x400ULL; }                // TEX=001 C B, S
    // AP[2:0]: 011 full RW, 111 RO everyone, 001 PL1 RW only, 101 PL1 RO only.
    unsigned long long ap = 0x010ULL;                                          // AP[1:0] = 01
    if ((flags & MMU_USER) != 0u) { ap = 0x030ULL; }                           // AP[1:0] = 11
    if ((flags & MMU_WRITE) == 0u) { ap = ap | 0x200ULL; }                     // AP[2]
    return e | ap;
}
int MmuArmV7::valid(unsigned long long e) const { return (e & 3ULL) != 0ULL; }
int MmuArmV7::is_leaf(unsigned long long e, int level) const {
    if ((e & 3ULL) == 0ULL) { return 0; }
    return level == 1 || (e & 3ULL) == 2ULL;               // L1 section
}
unsigned long MmuArmV7::addr(unsigned long long e, int level) const {
    if (level == 1) { return (unsigned long)(e & 0xFFFFF000ULL); }
    if ((e & 3ULL) == 2ULL) { return (unsigned long)(e & 0xFFF00000ULL); }
    return (unsigned long)(e & 0xFFFFFC00ULL);
}
int MmuArmV7::load_root(unsigned long root) const {
#if defined(__arm__) && defined(__ARM_ARCH_PROFILE) && __ARM_ARCH_PROFILE == 'A' && defined(__SAFEC_FREESTANDING__)
    unsigned long zero = 0UL;
    unsafe {
        asm volatile ("mcr p15, 0, %0, c2, c0, 0\n\tmcr p15, 0, %1, c8, c7, 0\n\tdsb\n\tisb"
                      : : "r"(root), "r"(zero) : "memory");
    }
    return 1;
#else
    return 0;
#endif
}
int MmuArmV7::flush_all() const {
#if defined(__arm__) && defined(__ARM_ARCH_PROFILE) && __ARM_ARCH_PROFILE == 'A' && defined(__SAFEC_FREESTANDING__)
    unsigned long zero = 0UL;
    unsafe { asm volatile ("mcr p15, 0, %0, c8, c7, 0\n\tdsb\n\tisb" : : "r"(zero) : "memory"); }
    return 1;
#else
    return 0;
#endif
}
int MmuArmV7::flush_page(unsigned long virt) const {
#if defined(__arm__) && defined(__ARM_ARCH_PROFILE) && __ARM_ARCH_PROFILE == 'A' && defined(__SAFEC_FREESTANDING__)
    unsigned long mva = virt & ~4095UL;
    unsafe { asm volatile ("mcr p15, 0, %0, c8, c7, 1\n\tdsb\n\tisb" : : "r"(mva) : "memory"); }
    return 1;
#else
    return 0;
#endif
}

} // namespace std
