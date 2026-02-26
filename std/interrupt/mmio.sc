// SafeC Standard Library — MMIO Helpers
#pragma once
#include "mmio.h"

// ── MmioReg constructor ───────────────────────────────────────────────────────
struct MmioReg mmio_reg(void* addr) {
    struct MmioReg r;
    r.addr = addr;
    return r;
}

// ── MmioReg methods ───────────────────────────────────────────────────────────
unsigned int MmioReg::read32() const {
    unsafe { return volatile_load((unsigned int*)self.addr); }
}

void MmioReg::write32(unsigned int val) {
    unsafe { volatile_store((unsigned int*)self.addr, val); }
}

void MmioReg::set_bits(unsigned int mask) {
    unsafe {
        unsigned int val = volatile_load((unsigned int*)self.addr);
        volatile_store((unsigned int*)self.addr, val | mask);
    }
}

void MmioReg::clear_bits(unsigned int mask) {
    unsafe {
        unsigned int val = volatile_load((unsigned int*)self.addr);
        volatile_store((unsigned int*)self.addr, val & ~mask);
    }
}

unsigned int MmioReg::read_field(int lo, int hi) const {
    unsafe {
        unsigned int val   = volatile_load((unsigned int*)self.addr);
        unsigned int width = (unsigned int)(hi - lo + 1);
        unsigned int mask  = ((unsigned int)1 << width) - (unsigned int)1;
        return (val >> lo) & mask;
    }
}

void MmioReg::write_field(int lo, int hi, unsigned int val) {
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
unsigned int mmio_read32(void* addr) {
    unsafe { return volatile_load((unsigned int*)addr); }
}

void mmio_write32(void* addr, unsigned int val) {
    unsafe { volatile_store((unsigned int*)addr, val); }
}

unsigned int mmio_read16(void* addr) {
    unsafe {
        unsigned int val = volatile_load((unsigned int*)addr);
        return val & (unsigned int)0xFFFF;
    }
}

void mmio_write16(void* addr, unsigned int val) {
    unsafe { volatile_store((unsigned int*)addr, val & (unsigned int)0xFFFF); }
}

unsigned int mmio_read8(void* addr) {
    unsafe {
        unsigned int val = volatile_load((unsigned int*)addr);
        return val & (unsigned int)0xFF;
    }
}

void mmio_write8(void* addr, unsigned int val) {
    unsafe { volatile_store((unsigned int*)addr, val & (unsigned int)0xFF); }
}

void mmio_set_bits32(void* addr, unsigned int mask) {
    unsafe {
        unsigned int val = volatile_load((unsigned int*)addr);
        volatile_store((unsigned int*)addr, val | mask);
    }
}

void mmio_clear_bits32(void* addr, unsigned int mask) {
    unsafe {
        unsigned int val = volatile_load((unsigned int*)addr);
        volatile_store((unsigned int*)addr, val & ~mask);
    }
}

unsigned int mmio_read_field32(void* addr, int lo, int hi) {
    unsafe {
        unsigned int val   = volatile_load((unsigned int*)addr);
        unsigned int width = (unsigned int)(hi - lo + 1);
        unsigned int mask  = ((unsigned int)1 << width) - (unsigned int)1;
        return (val >> lo) & mask;
    }
}

void mmio_write_field32(void* addr, int lo, int hi, unsigned int val) {
    unsafe {
        unsigned int reg   = volatile_load((unsigned int*)addr);
        unsigned int width = (unsigned int)(hi - lo + 1);
        unsigned int mask  = ((unsigned int)1 << width) - (unsigned int)1;
        reg = reg & ~(mask << lo);
        reg = reg | ((val & mask) << lo);
        volatile_store((unsigned int*)addr, reg);
    }
}
