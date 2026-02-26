// SafeC Standard Library — UART HAL
// Polling-based UART serial driver. Freestanding-safe.
#pragma once

struct Uart {
    void*        base;  // MMIO base address
    unsigned int baud;  // configured baud rate

    // Send a single byte (blocks until TX ready).
    void          write_byte(unsigned char c);

    // Read a single byte (blocks until RX available).
    unsigned char read_byte();

    // Send a null-terminated string.
    void          write_str(const char* s);

    // Check if a byte is available to read. Returns 1 if ready.
    int           rx_ready() const;

    // Check if transmitter is ready. Returns 1 if ready.
    int           tx_ready() const;
};

// Initialize UART at `base` with the given baud rate.
struct Uart uart_init(void* base, unsigned int baud);
