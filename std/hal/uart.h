// SafeC Standard Library — UART HAL
// Polling-based UART serial driver. Freestanding-safe.
//
// Uart<C> is generic over the UART block, a compile-time chip type:
//   UartGeneric    reference layout: DATA +0x00, STATUS +0x04 (bit 0 RX
//                  ready, bit 1 TX ready), BAUD +0x08 (written with 'baud')
//   Pl011          ARM PrimeCell PL011 (QEMU virt, Raspberry Pi, many SoCs)
//   Ns16550        16550-compatible, byte-wide registers at consecutive
//                  addresses (QEMU RISC-V virt, many SoCs)
//   Stm32F4Usart   STM32F1/F2/F4 USART (SR/DR/BRR/CR1)
//
//   struct Uart<Pl011> u = std::Uart::new(uart0, 115200u, 24000000u);
//   u.write_str("hello\n");
#pragma once

namespace std {

// What a UART block type provides; Uart<C> requires it.
trait UartChip {
    void          init(unsigned long base, unsigned int baud, unsigned int clock_hz) const;
    int           tx_ready(unsigned long base) const;
    int           rx_ready(unsigned long base) const;
    void          put(unsigned long base, unsigned char c) const;
    unsigned char get(unsigned long base) const;
}

// Each chip: init (set the baud rate from the input clock and enable TX/RX;
// clock_hz == 0 keeps whatever the boot firmware configured), the two
// ready flags, and one-byte data access. The driver does the waiting.
struct UartGeneric {
    int           unused_;
    void          init(unsigned long base, unsigned int baud, unsigned int clock_hz) const;
    int           tx_ready(unsigned long base) const;
    int           rx_ready(unsigned long base) const;
    void          put(unsigned long base, unsigned char c) const;
    unsigned char get(unsigned long base) const;
};
struct Pl011 {
    int           unused_;
    void          init(unsigned long base, unsigned int baud, unsigned int clock_hz) const;
    int           tx_ready(unsigned long base) const;
    int           rx_ready(unsigned long base) const;
    void          put(unsigned long base, unsigned char c) const;
    unsigned char get(unsigned long base) const;
};
struct Ns16550 {
    int           unused_;
    void          init(unsigned long base, unsigned int baud, unsigned int clock_hz) const;
    int           tx_ready(unsigned long base) const;
    int           rx_ready(unsigned long base) const;
    void          put(unsigned long base, unsigned char c) const;
    unsigned char get(unsigned long base) const;
};
struct Stm32F4Usart {
    int           unused_;
    void          init(unsigned long base, unsigned int baud, unsigned int clock_hz) const;
    int           tx_ready(unsigned long base) const;
    int           rx_ready(unsigned long base) const;
    void          put(unsigned long base, unsigned char c) const;
    unsigned char get(unsigned long base) const;
};

generic<C: UartChip>
struct Uart {
    &volatile unsigned int base; // MMIO base address
    unsigned int baud;  // configured baud rate
    C            chip;

    // Initialize the UART at `base`: 8N1 at `baud` from a `clock_hz` input clock
    // (0 = leave the rate the firmware set). The chip type comes from the target.
    static struct Uart<C> new(&volatile unsigned int base, unsigned int baud, unsigned int clock_hz);

    // Send a single byte (blocks until TX ready).
    void          write_byte(unsigned char c);
    // Read a single byte (blocks until RX available).
    unsigned char read_byte();
    // Send a null-terminated string.
    void          write_str(const &char s);
    // Check if a byte is available to read. Returns 1 if ready.
    int           rx_ready() const;
    // Check if transmitter is ready. Returns 1 if ready.
    int           tx_ready() const;
    unsigned long addr_() const;
};

generic<C: UartChip>
struct Uart<C> Uart::new(&volatile unsigned int base, unsigned int baud, unsigned int clock_hz) {
    struct Uart<C> u;
    u.base = base;
    u.baud = baud;
    u.chip.init(u.addr_(), baud, clock_hz);
    return u;
}

generic<C: UartChip>
unsigned long Uart::addr_() const { unsafe { return (unsigned long)self.base; } }
generic<C: UartChip>
int Uart::tx_ready() const { return self.chip.tx_ready(self.addr_()); }
generic<C: UartChip>
int Uart::rx_ready() const { return self.chip.rx_ready(self.addr_()); }
generic<C: UartChip>
void Uart::write_byte(unsigned char c) {
    while (self.tx_ready() == 0) { }
    self.chip.put(self.addr_(), c);
}
generic<C: UartChip>
unsigned char Uart::read_byte() {
    while (self.rx_ready() == 0) { }
    return self.chip.get(self.addr_());
}
generic<C: UartChip>
void Uart::write_str(const &char s) {
    unsigned long i = 0UL;
    unsafe {
        while (s[i] != (char)0) { self.write_byte((unsigned char)s[i]); i = i + 1UL; }
    }
}

} // namespace std
