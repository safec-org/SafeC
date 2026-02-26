// SafeC Standard Library — MMIO Helpers
// Memory-mapped I/O read/write primitives. Freestanding-safe.
#pragma once

// ── MmioReg: single-register wrapper with methods ────────────────────────────
struct MmioReg {
    void* addr;  // MMIO register address

    unsigned int read32() const;
    void         write32(unsigned int val);
    void         set_bits(unsigned int mask);
    void         clear_bits(unsigned int mask);
    unsigned int read_field(int lo, int hi) const;
    void         write_field(int lo, int hi, unsigned int val);
};

// Create a MmioReg pointing at `addr`.
struct MmioReg mmio_reg(void* addr);

// ── Free-function API (backward-compatible) ───────────────────────────────────
unsigned int mmio_read32(void* addr);
void         mmio_write32(void* addr, unsigned int val);
unsigned int mmio_read16(void* addr);
void         mmio_write16(void* addr, unsigned int val);
unsigned int mmio_read8(void* addr);
void         mmio_write8(void* addr, unsigned int val);
void         mmio_set_bits32(void* addr, unsigned int mask);
void         mmio_clear_bits32(void* addr, unsigned int mask);
unsigned int mmio_read_field32(void* addr, int lo, int hi);
void         mmio_write_field32(void* addr, int lo, int hi, unsigned int val);
