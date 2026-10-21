// SafeC Standard Library — SPI HAL chips (see spi.h)
#pragma once
#include <std/hal/spi.h>
#include <std/hal/reg.h>

namespace std {

// ── SpiGeneric: DATA +0, STATUS +4 (bit0 busy, bit1 TX ready), CTRL +8, CS +0xC
void SpiGeneric::init(unsigned long base, int mode) const {
    hal_wr32_(base, 0x08UL, ((unsigned int)mode & 3u) | 0x80u);
}
unsigned char SpiGeneric::transfer(unsigned long base, unsigned char tx) const {
    while ((hal_rd32_(base, 0x04UL) & 2u) == 0u) { }
    hal_wr32_(base, 0x00UL, (unsigned int)tx);
    while ((hal_rd32_(base, 0x04UL) & 1u) != 0u) { }
    return (unsigned char)hal_rd32_(base, 0x00UL);
}
void SpiGeneric::cs(unsigned long base, int asserted) const {
    hal_wr32_(base, 0x0CUL, asserted != 0 ? 0u : 1u);
}

// ── STM32F4 SPI: CR1 +0, SR +8 (RXNE bit0, TXE bit1), DR +0xC ───────────────
void Stm32F4Spi::init(unsigned long base, int mode) const {
    // CPHA/CPOL = mode, MSTR, BR = fPCLK/16, SSI|SSM (software NSS), then SPE.
    unsigned int cr1 = ((unsigned int)mode & 3u) | 0x04u | (3u << 3) | 0x300u;
    hal_wr32_(base, 0x00UL, cr1);
    hal_wr32_(base, 0x00UL, cr1 | 0x40u);
}
unsigned char Stm32F4Spi::transfer(unsigned long base, unsigned char tx) const {
    while ((hal_rd32_(base, 0x08UL) & 2u) == 0u) { }          // TXE
    hal_wr8_(base, 0x0CUL, (unsigned int)tx);                   // 8-bit frame
    while ((hal_rd32_(base, 0x08UL) & 1u) == 0u) { }          // RXNE
    return (unsigned char)hal_rd8_(base, 0x0CUL);
}
void Stm32F4Spi::cs(unsigned long base, int asserted) const { }   // NSS is a GPIO

} // namespace std
