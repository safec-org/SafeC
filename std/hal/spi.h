// SafeC Standard Library — SPI HAL
// Polling-based SPI master driver. Freestanding-safe.
#pragma once

// SPI modes (CPOL/CPHA)
#define SPI_MODE_0 0  // CPOL=0, CPHA=0
#define SPI_MODE_1 1  // CPOL=0, CPHA=1
#define SPI_MODE_2 2  // CPOL=1, CPHA=0
#define SPI_MODE_3 3  // CPOL=1, CPHA=1

struct SpiDevice {
    void*        base;  // MMIO base address
    int          mode;  // SPI mode (0-3)

    // Transfer one byte (full-duplex). Returns the received byte.
    unsigned char transfer(unsigned char tx);

    // Assert chip select (drive low).
    void          cs_assert();

    // Deassert chip select (drive high).
    void          cs_deassert();

    // Transfer a buffer of bytes. `rx` receives incoming data (may be NULL).
    void          write(const unsigned char* tx, unsigned char* rx, unsigned long len);

    // Read len bytes (sends 0xFF as dummy TX).
    void          read(unsigned char* rx, unsigned long len);
};

// Initialize SPI master at `base` with the given mode.
struct SpiDevice spi_init(void* base, int mode);
