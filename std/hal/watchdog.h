// SafeC Standard Library — Watchdog Timer HAL
// Hardware watchdog feed/enable. Freestanding-safe.
#pragma once

namespace std {

struct Watchdog {
    &volatile unsigned int base; // MMIO base address
    unsigned int timeout;  // timeout period (hardware-specific units)

    // Enable the watchdog timer (cannot be disabled on most hardware).
    void         enable();

    // Feed (kick) the watchdog to prevent reset.
    void         feed();

    // Check if a watchdog reset occurred. Returns 1 if the previous reset was a WDT reset.
    int          caused_reset() const;
};

// Initialize watchdog at `base` with the given timeout period.
struct Watchdog watchdog_init(&volatile unsigned int base, unsigned int timeout);

} // namespace std
