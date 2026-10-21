// SafeC Standard Library — GPIO HAL chips (see gpio.h)
#pragma once
#include <std/hal/gpio.h>
#include <std/hal/reg.h>

namespace std {

// ── GpioGeneric: DIR +0, OUT +4, IN +8, PULL_EN +0xC, PULL_SEL +0x10 ─────────
void GpioGeneric::set_direction(unsigned long base, int pin, int direction) const {
    unsigned int m = (unsigned int)1 << pin;
    if (direction == GPIO_OUTPUT) { hal_set32_(base, 0x00UL, m); } else { hal_clr32_(base, 0x00UL, m); }
}
void GpioGeneric::write(unsigned long base, int pin, int value) const {
    unsigned int m = (unsigned int)1 << pin;
    if (value != 0) { hal_set32_(base, 0x04UL, m); } else { hal_clr32_(base, 0x04UL, m); }
}
int GpioGeneric::read(unsigned long base, int pin) const {
    return (hal_rd32_(base, 0x08UL) >> pin) & 1u;
}
void GpioGeneric::toggle(unsigned long base, int pin) const {
    hal_wr32_(base, 0x04UL, hal_rd32_(base, 0x04UL) ^ ((unsigned int)1 << pin));
}
void GpioGeneric::set_pull(unsigned long base, int pin, int pull) const {
    unsigned int m = (unsigned int)1 << pin;
    if (pull == GPIO_PULL_NONE) { hal_clr32_(base, 0x0CUL, m); return; }
    if (pull == GPIO_PULL_UP) { hal_set32_(base, 0x10UL, m); } else { hal_clr32_(base, 0x10UL, m); }
    hal_set32_(base, 0x0CUL, m);
}

// ── Stm32F4Gpio: MODER +0, PUPDR +0xC, IDR +0x10, ODR +0x14, BSRR +0x18 ──────
// Two bits per pin in MODER/PUPDR; BSRR sets (bits 0-15) or resets (16-31)
// a pin in one write, so write() needs no read-modify-write.
static void stm32_field2_(unsigned long base, unsigned long off, int pin, unsigned int v) {
    unsigned int shift = (unsigned int)(pin * 2);
    unsigned int r = hal_rd32_(base, off) & ~(3u << shift);
    hal_wr32_(base, off, r | ((v & 3u) << shift));
}
void Stm32F4Gpio::set_direction(unsigned long base, int pin, int direction) const {
    stm32_field2_(base, 0x00UL, pin, direction == GPIO_OUTPUT ? 1u : 0u);   // 01 output, 00 input
}
void Stm32F4Gpio::write(unsigned long base, int pin, int value) const {
    hal_wr32_(base, 0x18UL, value != 0 ? (1u << pin) : (1u << (pin + 16)));
}
int Stm32F4Gpio::read(unsigned long base, int pin) const {
    return (hal_rd32_(base, 0x10UL) >> pin) & 1u;
}
void Stm32F4Gpio::toggle(unsigned long base, int pin) const {
    int high = (hal_rd32_(base, 0x14UL) >> pin) & 1u;
    self.write(base, pin, high == 0);
}
void Stm32F4Gpio::set_pull(unsigned long base, int pin, int pull) const {
    unsigned int v = pull == GPIO_PULL_UP ? 1u : (pull == GPIO_PULL_DOWN ? 2u : 0u);
    stm32_field2_(base, 0x0CUL, pin, v);
}

} // namespace std
