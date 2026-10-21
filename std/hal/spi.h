// SafeC Standard Library — SPI HAL
// Polling-based SPI master driver. Freestanding-safe.
//
// SpiDevice<C> is generic over the SPI block, a compile-time chip type:
//   SpiGeneric    reference layout: DATA +0x00, STATUS +0x04 (bit 0 busy,
//                 bit 1 TX ready), CTRL +0x08 (mode | 0x80 enable),
//                 CS +0x0C (0 = asserted)
//   Stm32F4Spi    STM32F1/F2/F4 SPI as master with software NSS; chip select
//                 is an ordinary GPIO there (cs_assert/cs_deassert do
//                 nothing — drive a GpioPin)
#pragma once

// SPI modes (CPOL/CPHA)
#define SPI_MODE_0 0  // CPOL=0, CPHA=0
#define SPI_MODE_1 1  // CPOL=0, CPHA=1
#define SPI_MODE_2 2  // CPOL=1, CPHA=0
#define SPI_MODE_3 3  // CPOL=1, CPHA=1

namespace std {

struct SpiGeneric {
    int           unused_;
    void          init(unsigned long base, int mode) const;
    unsigned char transfer(unsigned long base, unsigned char tx) const;
    void          cs(unsigned long base, int asserted) const;
};
struct Stm32F4Spi {
    int           unused_;
    void          init(unsigned long base, int mode) const;   // master, fPCLK/16, 8-bit, MSB first
    unsigned char transfer(unsigned long base, unsigned char tx) const;
    void          cs(unsigned long base, int asserted) const;
};

generic<C>
struct SpiDevice {
    &volatile unsigned int base; // MMIO base address
    int          mode;  // SPI mode (0-3)
    C            chip;

    // Initialize an SPI master at `base` with the given mode; the chip type
    // comes from the target.
    static struct SpiDevice<C> new(&volatile unsigned int base, int mode);

    // Transfer one byte (full-duplex). Returns the received byte.
    unsigned char transfer(unsigned char tx);
    // Assert chip select (drive low).
    void          cs_assert();
    // Deassert chip select (drive high).
    void          cs_deassert();
    // Transfer a buffer of bytes. `rx` receives incoming data (may be null).
    void          write(const &unsigned char tx, ?&unsigned char rx, unsigned long len);
    // Read len bytes (sends 0xFF as dummy TX).
    void          read(&unsigned char rx, unsigned long len);
    unsigned long addr_() const;
};

generic<C>
struct SpiDevice<C> SpiDevice::new(&volatile unsigned int base, int mode) {
    struct SpiDevice<C> s;
    s.base = base;
    s.mode = mode;
    s.chip.init(s.addr_(), mode);
    return s;
}

generic<C>
unsigned long SpiDevice::addr_() const { unsafe { return (unsigned long)self.base; } }
generic<C>
unsigned char SpiDevice::transfer(unsigned char tx) { return self.chip.transfer(self.addr_(), tx); }
generic<C>
void SpiDevice::cs_assert() { self.chip.cs(self.addr_(), 1); }
generic<C>
void SpiDevice::cs_deassert() { self.chip.cs(self.addr_(), 0); }
generic<C>
void SpiDevice::write(const &unsigned char tx, ?&unsigned char rx, unsigned long len) {
    unsigned long i = 0UL;
    while (i < len) {
        unsigned char b;
        unsafe { b = self.transfer(tx[i]); }
        match (rx) {
            case null: { }
            case some(r): { unsafe { r[i] = b; } }
        }
        i = i + 1UL;
    }
}
generic<C>
void SpiDevice::read(&unsigned char rx, unsigned long len) {
    unsigned long i = 0UL;
    while (i < len) {
        unsafe { rx[i] = self.transfer((unsigned char)0xFF); }
        i = i + 1UL;
    }
}

} // namespace std
