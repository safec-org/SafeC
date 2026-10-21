// SafeC Standard Library — MMIO Helpers
// Memory-mapped I/O read/write primitives. Freestanding-safe.
#pragma once

// ── MmioReg: single-register wrapper with methods ────────────────────────────
namespace std {

struct MmioReg {
    &volatile unsigned int addr;  // MMIO register address

    unsigned int read32() const;
    void         write32(unsigned int val);
    void         set_bits(unsigned int mask);
    void         clear_bits(unsigned int mask);
    unsigned int read_field(int lo, int hi) const;
    void         write_field(int lo, int hi, unsigned int val);

    // Create a MmioReg pointing at `addr`.
    static struct MmioReg at(&volatile unsigned int addr);
};

// ── Free-function API (backward-compatible) ───────────────────────────────────
unsigned int mmio_read32(&volatile unsigned int addr);
void         mmio_write32(&volatile unsigned int addr, unsigned int val);
unsigned int mmio_read16(&volatile unsigned short addr);
void         mmio_write16(&volatile unsigned short addr, unsigned int val);
unsigned int mmio_read8(&volatile unsigned char addr);
void         mmio_write8(&volatile unsigned char addr, unsigned int val);
void         mmio_set_bits32(&volatile unsigned int addr, unsigned int mask);
void         mmio_clear_bits32(&volatile unsigned int addr, unsigned int mask);
unsigned int mmio_read_field32(&volatile unsigned int addr, int lo, int hi);
void         mmio_write_field32(&volatile unsigned int addr, int lo, int hi, unsigned int val);

} // namespace std
