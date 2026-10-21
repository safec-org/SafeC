// SafeC Standard Library — Watchdog Timer HAL
// Hardware watchdog feed/enable. Freestanding-safe.
//
// Watchdog<C> is generic over the watchdog block, a compile-time chip type:
//   WatchdogGeneric  reference layout: CTRL +0x00 (bit 0 enable), TIMEOUT
//                    +0x04, FEED +0x08 (write 0x6969), STATUS +0x0C (bit 0:
//                    last reset was a watchdog reset)
//   Stm32F4Iwdg      STM32 independent watchdog (KR/PR/RLR/SR): 'timeout'
//                    is the 12-bit reload value, counted at LSI / 256
//                    (~8 ms per unit at 32 kHz); caused_reset() reads
//                    RCC_CSR.IWDGRSTF at STM32F4's RCC address
#pragma once

#define STM32F4_RCC_CSR 0x40023874UL

namespace std {

struct WatchdogGeneric {
    int  unused_;
    void init(unsigned long base, unsigned int timeout) const;
    void enable(unsigned long base) const;
    void feed(unsigned long base) const;
    int  caused_reset(unsigned long base) const;
};
struct Stm32F4Iwdg {
    int  unused_;
    void init(unsigned long base, unsigned int timeout) const;
    void enable(unsigned long base) const;
    void feed(unsigned long base) const;
    int  caused_reset(unsigned long base) const;
};

generic<C>
struct Watchdog {
    &volatile unsigned int base; // MMIO base address
    unsigned int timeout;  // timeout period (chip-specific units, see above)
    C            chip;

    // Initialize the watchdog at `base` with the given timeout; the chip type
    // comes from the target.
    static struct Watchdog<C> new(&volatile unsigned int base, unsigned int timeout);

    // Enable the watchdog timer (cannot be disabled on most hardware).
    void         enable();
    // Feed (kick) the watchdog to prevent reset.
    void         feed();
    // Check if a watchdog reset occurred. Returns 1 if the previous reset was a WDT reset.
    int          caused_reset() const;
    unsigned long addr_() const;
};

generic<C>
struct Watchdog<C> Watchdog::new(&volatile unsigned int base, unsigned int timeout) {
    struct Watchdog<C> w;
    w.base = base;
    w.timeout = timeout;
    w.chip.init(w.addr_(), timeout);
    return w;
}

generic<C>
unsigned long Watchdog::addr_() const { unsafe { return (unsigned long)self.base; } }
generic<C>
void Watchdog::enable() { self.chip.enable(self.addr_()); }
generic<C>
void Watchdog::feed() { self.chip.feed(self.addr_()); }
generic<C>
int Watchdog::caused_reset() const { return self.chip.caused_reset(self.addr_()); }

} // namespace std
