// SafeC Standard Library — I2C HAL
// Polling-based I2C master driver. Freestanding-safe.
#pragma once

// I2C result codes
#define I2C_OK     0
#define I2C_NACK   1
#define I2C_ERROR  2

struct I2cBus {
    void*        base;   // MMIO base address
    unsigned int speed;  // bus speed in Hz

    // Write `len` bytes from `data` to device at `addr`. Returns I2C_OK on success.
    int          write(unsigned char addr, const unsigned char* data, unsigned long len);

    // Read `len` bytes into `data` from device at `addr`. Returns I2C_OK on success.
    int          read(unsigned char addr, unsigned char* data, unsigned long len);

    // Write then read (combined transaction). Returns I2C_OK on success.
    int          write_read(unsigned char addr,
                            const unsigned char* wr, unsigned long wr_len,
                            unsigned char* rd,       unsigned long rd_len);

    // Probe: attempt a zero-length write to `addr`. Returns 1 if device responds.
    int          probe(unsigned char addr);

    // Internal: wait until bus is not busy.
    int          wait_();
};

// Initialize I2C master at `base` with the given bus speed.
struct I2cBus i2c_init(void* base, unsigned int speed);
