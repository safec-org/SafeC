// SafeC Standard Library — MMIO Helpers
#pragma once
#include <std/interrupt/mmio.h>

// ── MmioReg constructor ───────────────────────────────────────────────────────
namespace std {

// Mask of the low (hi - lo + 1) bits; a full 32-bit field can't use 1 << 32.
static unsigned int mmio_field_mask_(int lo, int hi) {
    int width = hi - lo + 1;
    if (width >= 32) { return ~(unsigned int)0; }
    return ((unsigned int)1 << width) - (unsigned int)1;
}

inline struct MmioReg MmioReg::at(&volatile unsigned int addr) {
    struct MmioReg r;
    r.addr = addr;
    return r;
}

// ── MmioReg methods ───────────────────────────────────────────────────────────
inline unsigned int MmioReg::read32() const {
    unsafe { return volatile_load((unsigned int*)self.addr); }
}

inline void MmioReg::write32(unsigned int val) {
    unsafe { volatile_store((unsigned int*)self.addr, val); }
}

inline void MmioReg::set_bits(unsigned int mask) {
    unsafe {
        unsigned int val = volatile_load((unsigned int*)self.addr);
        volatile_store((unsigned int*)self.addr, val | mask);
    }
}

inline void MmioReg::clear_bits(unsigned int mask) {
    unsafe {
        unsigned int val = volatile_load((unsigned int*)self.addr);
        volatile_store((unsigned int*)self.addr, val & ~mask);
    }
}

inline unsigned int MmioReg::read_field(int lo, int hi) const {
    unsafe {
        unsigned int val   = volatile_load((unsigned int*)self.addr);
        unsigned int mask  = mmio_field_mask_(lo, hi);
        return (val >> lo) & mask;
    }
}

inline void MmioReg::write_field(int lo, int hi, unsigned int val) {
    unsafe {
        unsigned int reg   = volatile_load((unsigned int*)self.addr);
        unsigned int mask  = mmio_field_mask_(lo, hi);
        reg = reg & ~(mask << lo);
        reg = reg | ((val & mask) << lo);
        volatile_store((unsigned int*)self.addr, reg);
    }
}

// ── Free-function API ─────────────────────────────────────────────────────────
inline unsigned int mmio_read32(&volatile unsigned int addr) {
    unsafe { return volatile_load((unsigned int*)addr); }
}

inline void mmio_write32(&volatile unsigned int addr, unsigned int val) {
    unsafe { volatile_store((unsigned int*)addr, val); }
}

// 8/16-bit registers get 8/16-bit accesses: a 32-bit store would also write
// the neighbouring registers (and a 32-bit load can fault or have read side
// effects on them).
inline unsigned int mmio_read16(&volatile unsigned short addr) {
    unsafe { return (unsigned int)volatile_load((unsigned short*)addr); }
}

inline void mmio_write16(&volatile unsigned short addr, unsigned int val) {
    unsafe { volatile_store((unsigned short*)addr, (unsigned short)(val & (unsigned int)0xFFFF)); }
}

inline unsigned int mmio_read8(&volatile unsigned char addr) {
    unsafe { return (unsigned int)volatile_load((unsigned char*)addr); }
}

inline void mmio_write8(&volatile unsigned char addr, unsigned int val) {
    unsafe { volatile_store((unsigned char*)addr, (unsigned char)(val & (unsigned int)0xFF)); }
}

inline void mmio_set_bits32(&volatile unsigned int addr, unsigned int mask) {
    unsafe {
        unsigned int val = volatile_load((unsigned int*)addr);
        volatile_store((unsigned int*)addr, val | mask);
    }
}

inline void mmio_clear_bits32(&volatile unsigned int addr, unsigned int mask) {
    unsafe {
        unsigned int val = volatile_load((unsigned int*)addr);
        volatile_store((unsigned int*)addr, val & ~mask);
    }
}

inline unsigned int mmio_read_field32(&volatile unsigned int addr, int lo, int hi) {
    unsafe {
        unsigned int val   = volatile_load((unsigned int*)addr);
        unsigned int mask  = mmio_field_mask_(lo, hi);
        return (val >> lo) & mask;
    }
}

inline void mmio_write_field32(&volatile unsigned int addr, int lo, int hi, unsigned int val) {
    unsafe {
        unsigned int reg   = volatile_load((unsigned int*)addr);
        unsigned int mask  = mmio_field_mask_(lo, hi);
        reg = reg & ~(mask << lo);
        reg = reg | ((val & mask) << lo);
        volatile_store((unsigned int*)addr, reg);
    }
}

} // namespace std
