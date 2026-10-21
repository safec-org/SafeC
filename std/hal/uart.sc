// SafeC Standard Library — UART HAL chips (see uart.h)
#pragma once
#include <std/hal/uart.h>
#include <std/hal/reg.h>

namespace std {

// ── UartGeneric: DATA +0, STATUS +4 (bit0 RX ready, bit1 TX ready), BAUD +8 ─
void UartGeneric::init(unsigned long base, unsigned int baud, unsigned int clock_hz) const {
    hal_wr32_(base, 0x08UL, baud);
}
int UartGeneric::tx_ready(unsigned long base) const { return (hal_rd32_(base, 0x04UL) & 2u) != 0u; }
int UartGeneric::rx_ready(unsigned long base) const { return (hal_rd32_(base, 0x04UL) & 1u) != 0u; }
void UartGeneric::put(unsigned long base, unsigned char c) const { hal_wr32_(base, 0x00UL, (unsigned int)c); }
unsigned char UartGeneric::get(unsigned long base) const { return (unsigned char)hal_rd32_(base, 0x00UL); }

// ── PL011: DR +0, FR +0x18 (TXFF bit5, RXFE bit4), IBRD +0x24, FBRD +0x28,
//    LCR_H +0x2C, CR +0x30 ─────────────────────────────────────────────────────
void Pl011::init(unsigned long base, unsigned int baud, unsigned int clock_hz) const {
    if (clock_hz == 0u || baud == 0u) { return; }
    hal_wr32_(base, 0x30UL, 0u);                               // disable while configuring
    // Divisor = clock / (16 * baud), as 16.6 fixed point, rounded.
    unsigned long long div64 = ((unsigned long long)clock_hz * 4ULL + (unsigned long long)baud / 2ULL)
                             / (unsigned long long)baud;
    hal_wr32_(base, 0x24UL, (unsigned int)(div64 >> 6ULL));
    hal_wr32_(base, 0x28UL, (unsigned int)(div64 & 63ULL));
    hal_wr32_(base, 0x2CUL, 0x70u);                            // 8 bits, FIFOs on
    hal_wr32_(base, 0x30UL, 0x301u);                           // UARTEN | TXE | RXE
}
int Pl011::tx_ready(unsigned long base) const { return (hal_rd32_(base, 0x18UL) & 0x20u) == 0u; }
int Pl011::rx_ready(unsigned long base) const { return (hal_rd32_(base, 0x18UL) & 0x10u) == 0u; }
void Pl011::put(unsigned long base, unsigned char c) const { hal_wr32_(base, 0x00UL, (unsigned int)c); }
unsigned char Pl011::get(unsigned long base) const { return (unsigned char)hal_rd32_(base, 0x00UL); }

// ── 16550: RBR/THR +0, IER +1, FCR +2, LCR +3, LSR +5 (byte registers) ───────
void Ns16550::init(unsigned long base, unsigned int baud, unsigned int clock_hz) const {
    if (clock_hz == 0u || baud == 0u) { return; }
    unsigned int div = (clock_hz + 8u * baud) / (16u * baud);
    hal_wr8_(base, 1UL, 0u);                                   // no interrupts
    hal_wr8_(base, 3UL, 0x80u);                                // DLAB: divisor latch
    hal_wr8_(base, 0UL, div & 0xFFu);
    hal_wr8_(base, 1UL, (div >> 8) & 0xFFu);
    hal_wr8_(base, 3UL, 0x03u);                                // 8N1, DLAB off
    hal_wr8_(base, 2UL, 0x07u);                                // FIFOs on, cleared
}
int Ns16550::tx_ready(unsigned long base) const { return (hal_rd8_(base, 5UL) & 0x20u) != 0u; }
int Ns16550::rx_ready(unsigned long base) const { return (hal_rd8_(base, 5UL) & 0x01u) != 0u; }
void Ns16550::put(unsigned long base, unsigned char c) const { hal_wr8_(base, 0UL, (unsigned int)c); }
unsigned char Ns16550::get(unsigned long base) const { return (unsigned char)hal_rd8_(base, 0UL); }

// ── STM32F4 USART: SR +0 (TXE bit7, RXNE bit5), DR +4, BRR +8, CR1 +0xC ──────
void Stm32F4Usart::init(unsigned long base, unsigned int baud, unsigned int clock_hz) const {
    if (clock_hz != 0u && baud != 0u) {
        hal_wr32_(base, 0x08UL, (clock_hz + baud / 2u) / baud);   // USARTDIV, 16x oversampling
    }
    hal_wr32_(base, 0x0CUL, 0x200Cu);                          // UE | TE | RE
}
int Stm32F4Usart::tx_ready(unsigned long base) const { return (hal_rd32_(base, 0x00UL) & 0x80u) != 0u; }
int Stm32F4Usart::rx_ready(unsigned long base) const { return (hal_rd32_(base, 0x00UL) & 0x20u) != 0u; }
void Stm32F4Usart::put(unsigned long base, unsigned char c) const { hal_wr32_(base, 0x04UL, (unsigned int)c); }
unsigned char Stm32F4Usart::get(unsigned long base) const { return (unsigned char)hal_rd32_(base, 0x04UL); }

} // namespace std
