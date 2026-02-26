// SafeC Standard Library — Watchdog Timer HAL
// Generic MMIO layout:
//   base + 0x00 = control register (bit 0 = enable)
//   base + 0x04 = timeout/reload register
//   base + 0x08 = feed register (write magic value to feed)
//   base + 0x0C = status register (bit 0 = WDT reset occurred)
#pragma once
#include "watchdog.h"

// Magic feed value (common on ARM Cortex-M)
unsigned int wdt_feed_magic_() {
    return (unsigned int)0x6969;
}

struct Watchdog watchdog_init(void* base, unsigned int timeout) {
    struct Watchdog w;
    w.base    = base;
    w.timeout = timeout;
    unsafe {
        unsigned int* tmo_reg = (unsigned int*)((unsigned long)base + (unsigned long)4);
        volatile_store(tmo_reg, timeout);
    }
    return w;
}

void Watchdog::enable() {
    unsafe {
        unsigned int* ctrl = (unsigned int*)self.base;
        unsigned int val = volatile_load(ctrl);
        val = val | (unsigned int)1;
        volatile_store(ctrl, val);
    }
}

void Watchdog::feed() {
    unsafe {
        unsigned int* feed_reg = (unsigned int*)((unsigned long)self.base + (unsigned long)8);
        volatile_store(feed_reg, wdt_feed_magic_());
    }
}

int Watchdog::caused_reset() const {
    unsafe {
        unsigned int* status = (unsigned int*)((unsigned long)self.base + (unsigned long)12);
        unsigned int val = volatile_load(status);
        if ((val & (unsigned int)1) != (unsigned int)0) { return 1; }
        return 0;
    }
}
