// SafeC Standard Library — Watchdog HAL chips (see watchdog.h)
#pragma once
#include <std/hal/watchdog.h>
#include <std/hal/reg.h>

namespace std {

// ── WatchdogGeneric: CTRL +0, TIMEOUT +4, FEED +8, STATUS +0xC ───────────────
void WatchdogGeneric::init(unsigned long base, unsigned int timeout) const { hal_wr32_(base, 0x04UL, timeout); }
void WatchdogGeneric::enable(unsigned long base) const { hal_set32_(base, 0x00UL, 1u); }
void WatchdogGeneric::feed(unsigned long base) const { hal_wr32_(base, 0x08UL, 0x6969u); }
int WatchdogGeneric::caused_reset(unsigned long base) const { return (hal_rd32_(base, 0x0CUL) & 1u) != 0u; }

// ── STM32 IWDG: KR +0, PR +4, RLR +8, SR +0xC ────────────────────────────────
void Stm32F4Iwdg::init(unsigned long base, unsigned int timeout) const {
    hal_wr32_(base, 0x00UL, 0x5555u);                    // unlock PR/RLR
    hal_wr32_(base, 0x04UL, 6u);                         // LSI / 256
    hal_wr32_(base, 0x08UL, timeout & 0xFFFu);
    hal_wait32_(base, 0x0CUL, 3u, 0u, 100000);           // PVU/RVU: values taken
    hal_wr32_(base, 0x00UL, 0xAAAAu);                    // reload the counter
}
void Stm32F4Iwdg::enable(unsigned long base) const { hal_wr32_(base, 0x00UL, 0xCCCCu); }
void Stm32F4Iwdg::feed(unsigned long base) const { hal_wr32_(base, 0x00UL, 0xAAAAu); }
int Stm32F4Iwdg::caused_reset(unsigned long base) const {
    return (hal_rd32_(STM32F4_RCC_CSR, 0UL) & (1u << 29)) != 0u;   // RCC_CSR.IWDGRSTF
}

} // namespace std
