// SafeC Standard Library — MMIO Helpers
#pragma once
#include <std/interrupt/mmio.h>

// ── MmioReg constructor ───────────────────────────────────────────────────────
namespace std {

inline struct MmioReg mmio_reg(void* addr) {
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
        unsigned int width = (unsigned int)(hi - lo + 1);
        unsigned int mask  = ((unsigned int)1 << width) - (unsigned int)1;
        return (val >> lo) & mask;
    }
}

inline void MmioReg::write_field(int lo, int hi, unsigned int val) {
    unsafe {
        unsigned int reg   = volatile_load((unsigned int*)self.addr);
        unsigned int width = (unsigned int)(hi - lo + 1);
        unsigned int mask  = ((unsigned int)1 << width) - (unsigned int)1;
        reg = reg & ~(mask << lo);
        reg = reg | ((val & mask) << lo);
        volatile_store((unsigned int*)self.addr, reg);
    }
}

// ── Free-function API ─────────────────────────────────────────────────────────
inline unsigned int mmio_read32(void* addr) {
    unsafe { return volatile_load((unsigned int*)addr); }
}

inline void mmio_write32(void* addr, unsigned int val) {
    unsafe { volatile_store((unsigned int*)addr, val); }
}

inline unsigned int mmio_read16(void* addr) {
    unsafe {
        unsigned int val = volatile_load((unsigned int*)addr);
        return val & (unsigned int)0xFFFF;
    }
}

inline void mmio_write16(void* addr, unsigned int val) {
    unsafe { volatile_store((unsigned int*)addr, val & (unsigned int)0xFFFF); }
}

inline unsigned int mmio_read8(void* addr) {
    unsafe {
        unsigned int val = volatile_load((unsigned int*)addr);
        return val & (unsigned int)0xFF;
    }
}

inline void mmio_write8(void* addr, unsigned int val) {
    unsafe { volatile_store((unsigned int*)addr, val & (unsigned int)0xFF); }
}

inline void mmio_set_bits32(void* addr, unsigned int mask) {
    unsafe {
        unsigned int val = volatile_load((unsigned int*)addr);
        volatile_store((unsigned int*)addr, val | mask);
    }
}

inline void mmio_clear_bits32(void* addr, unsigned int mask) {
    unsafe {
        unsigned int val = volatile_load((unsigned int*)addr);
        volatile_store((unsigned int*)addr, val & ~mask);
    }
}

inline unsigned int mmio_read_field32(void* addr, int lo, int hi) {
    unsafe {
        unsigned int val   = volatile_load((unsigned int*)addr);
        unsigned int width = (unsigned int)(hi - lo + 1);
        unsigned int mask  = ((unsigned int)1 << width) - (unsigned int)1;
        return (val >> lo) & mask;
    }
}

inline void mmio_write_field32(void* addr, int lo, int hi, unsigned int val) {
    unsafe {
        unsigned int reg   = volatile_load((unsigned int*)addr);
        unsigned int width = (unsigned int)(hi - lo + 1);
        unsigned int mask  = ((unsigned int)1 << width) - (unsigned int)1;
        reg = reg & ~(mask << lo);
        reg = reg | ((val & mask) << lo);
        volatile_store((unsigned int*)addr, reg);
    }
}

} // namespace std
