// SafeC Standard Library — Timer HAL chips (see timer.h)
#pragma once
#include <std/hal/timer.h>
#include <std/hal/reg.h>

namespace std {

// ── TimerGeneric: CTRL +0, PRESCALER +4, PERIOD +8, COUNT +0xC, STATUS +0x10 ─
void TimerGeneric::init(unsigned long base, unsigned int prescaler) const { hal_wr32_(base, 0x04UL, prescaler); }
void TimerGeneric::set_period(unsigned long base, unsigned int period) const { hal_wr32_(base, 0x08UL, period); }
void TimerGeneric::run(unsigned long base, int on) const {
    if (on != 0) { hal_set32_(base, 0x00UL, 1u); } else { hal_clr32_(base, 0x00UL, 1u); }
}
unsigned int TimerGeneric::count(unsigned long base) const { return hal_rd32_(base, 0x0CUL); }
int TimerGeneric::flag(unsigned long base) const { return (hal_rd32_(base, 0x10UL) & 1u) != 0u; }
void TimerGeneric::clear_flag(unsigned long base) const { hal_wr32_(base, 0x10UL, 0u); }

// ── STM32 TIMx: CR1 +0 (CEN bit0), SR +0x10 (UIF bit0, write 0 to clear),
//    EGR +0x14 (UG bit0), CNT +0x24, PSC +0x28, ARR +0x2C ────────────────────
void Stm32F4Tim::init(unsigned long base, unsigned int prescaler) const {
    hal_wr32_(base, 0x28UL, prescaler & 0xFFFFu);
    hal_wr32_(base, 0x14UL, 1u);                         // UG: load PSC now
    hal_wr32_(base, 0x10UL, 0u);                         // ...and drop the UIF that set
}
void Stm32F4Tim::set_period(unsigned long base, unsigned int period) const { hal_wr32_(base, 0x2CUL, period); }
void Stm32F4Tim::run(unsigned long base, int on) const {
    if (on != 0) { hal_set32_(base, 0x00UL, 1u); } else { hal_clr32_(base, 0x00UL, 1u); }
}
unsigned int Stm32F4Tim::count(unsigned long base) const { return hal_rd32_(base, 0x24UL); }
int Stm32F4Tim::flag(unsigned long base) const { return (hal_rd32_(base, 0x10UL) & 1u) != 0u; }
void Stm32F4Tim::clear_flag(unsigned long base) const { hal_wr32_(base, 0x10UL, ~1u); }   // rc_w0

} // namespace std
