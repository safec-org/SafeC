// SafeC Standard Library — I2C HAL
// Polling-based I2C master driver. Freestanding-safe.
//
// I2cBus<C> is generic over the I2C block, a compile-time chip type:
//   I2cGeneric    reference layout: DATA +0x00, STATUS +0x04 (bit 0 busy,
//                 bit 1 ACK, bit 2 error), CTRL +0x08 (1 START, 2 STOP,
//                 4 SEND, 8 READ+ACK, 16 READ+NACK), ADDR +0x0C, SPEED +0x10
//   Stm32F4I2c    STM32F1/F2/F4 I2C ("v1": CR1/CR2/DR/SR1/SR2/CCR/TRISE),
//                 standard (<= 100 kHz) and fast (<= 400 kHz) mode
// Every transfer ends with STOP, also when it fails, so the bus is released.
#pragma once

// I2C result codes
#define I2C_OK     0
#define I2C_NACK   1
#define I2C_ERROR  2

namespace std {

struct I2cGeneric {
    int unused_;
    void init(unsigned long base, unsigned int speed, unsigned int clock_hz) const;
    int  write(unsigned long base, unsigned char addr, const &unsigned char data, unsigned long len) const;
    int  read(unsigned long base, unsigned char addr, &unsigned char data, unsigned long len) const;
    int  probe(unsigned long base, unsigned char addr) const;
};
struct Stm32F4I2c {
    int unused_;
    void init(unsigned long base, unsigned int speed, unsigned int clock_hz) const;
    int  write(unsigned long base, unsigned char addr, const &unsigned char data, unsigned long len) const;
    int  read(unsigned long base, unsigned char addr, &unsigned char data, unsigned long len) const;
    int  probe(unsigned long base, unsigned char addr) const;
};

generic<C>
struct I2cBus {
    &volatile unsigned int base; // MMIO base address
    unsigned int speed;  // bus speed in Hz
    C            chip;

    // Initialize an I2C master at `base` for `speed` Hz, from a `clock_hz`
    // peripheral clock (STM32: the APB1 clock; ignored by I2cGeneric). The chip
    // type comes from the target.
    static struct I2cBus<C> new(&volatile unsigned int base, unsigned int speed, unsigned int clock_hz);

    // Write `len` bytes from `data` to the 7-bit address `addr`. Returns I2C_OK on success.
    int          write(unsigned char addr, const &unsigned char data, unsigned long len);
    // Read `len` bytes into `data` from device at `addr`. Returns I2C_OK on success.
    int          read(unsigned char addr, &unsigned char data, unsigned long len);
    // Write, then read (e.g. a register address, then its value). Returns I2C_OK on success.
    int          write_read(unsigned char addr,
                            const &unsigned char wr, unsigned long wr_len,
                            &unsigned char rd,       unsigned long rd_len);
    // Probe: address the device with no data. Returns 1 if it acknowledges.
    int          probe(unsigned char addr);
    unsigned long addr_() const;
};

generic<C>
struct I2cBus<C> I2cBus::new(&volatile unsigned int base, unsigned int speed, unsigned int clock_hz) {
    struct I2cBus<C> bus;
    bus.base = base;
    bus.speed = speed;
    bus.chip.init(bus.addr_(), speed, clock_hz);
    return bus;
}

generic<C>
unsigned long I2cBus::addr_() const { unsafe { return (unsigned long)self.base; } }
generic<C>
int I2cBus::write(unsigned char addr, const &unsigned char data, unsigned long len) {
    return self.chip.write(self.addr_(), addr, data, len);
}
generic<C>
int I2cBus::read(unsigned char addr, &unsigned char data, unsigned long len) {
    return self.chip.read(self.addr_(), addr, data, len);
}
generic<C>
int I2cBus::write_read(unsigned char addr, const &unsigned char wr, unsigned long wr_len,
                       &unsigned char rd, unsigned long rd_len) {
    int r = self.chip.write(self.addr_(), addr, wr, wr_len);
    if (r != I2C_OK) { return r; }
    return self.chip.read(self.addr_(), addr, rd, rd_len);
}
generic<C>
int I2cBus::probe(unsigned char addr) { return self.chip.probe(self.addr_(), addr); }

} // namespace std
