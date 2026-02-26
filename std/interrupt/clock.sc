// SafeC Standard Library — System Clock Configuration
// Generic MMIO layout (loosely based on STM32 RCC):
//   base + 0x00 = clock control register (source select, PLL enable)
//   base + 0x04 = PLL configuration (multiplier, divider)
//   base + 0x08 = clock config register (AHB/APB prescalers)
//   base + 0x0C = status register (PLL ready, clock ready)
#pragma once
#include "clock.h"

struct ClockConfig clock_init(void* base) {
    struct ClockConfig c;
    c.base     = base;
    c.source   = 0; // HSI default
    c.sys_freq = (unsigned int)8000000; // 8 MHz HSI typical
    c.ahb_div  = (unsigned int)1;
    c.apb1_div = (unsigned int)1;
    c.apb2_div = (unsigned int)1;
    return c;
}

void ClockConfig::set_source(int source) {
    self.source = source;
    unsafe {
        unsigned int* ctrl = (unsigned int*)self.base;
        unsigned int val   = volatile_load(ctrl);
        val = val & ~(unsigned int)3;
        val = val | ((unsigned int)source & (unsigned int)3);
        volatile_store(ctrl, val);

        // Wait for clock switch to complete
        unsigned int* status = (unsigned int*)((unsigned long)self.base + (unsigned long)12);
        int timeout = 100000;
        while (timeout > 0) {
            unsigned int st = volatile_load(status);
            if (((st >> 2) & (unsigned int)3) == ((unsigned int)source & (unsigned int)3)) {
                return;
            }
            timeout = timeout - 1;
        }
    }
}

int ClockConfig::configure_pll(unsigned int mul, unsigned int div) {
    if (div == (unsigned int)0) { return 0; }
    unsafe {
        unsigned int* ctrl = (unsigned int*)self.base;
        unsigned int val   = volatile_load(ctrl);
        val = val & ~((unsigned int)1 << 24); // disable PLL
        volatile_store(ctrl, val);

        unsigned int* pll_cfg = (unsigned int*)((unsigned long)self.base + (unsigned long)4);
        unsigned int cfg = (mul & (unsigned int)0xFF) | ((div & (unsigned int)0xFF) << 8);
        volatile_store(pll_cfg, cfg);

        val = val | ((unsigned int)1 << 24); // enable PLL
        volatile_store(ctrl, val);

        // Wait for PLL ready
        unsigned int* status = (unsigned int*)((unsigned long)self.base + (unsigned long)12);
        int timeout = 100000;
        while (timeout > 0) {
            unsigned int st = volatile_load(status);
            if ((st & ((unsigned int)1 << 25)) != (unsigned int)0) {
                self.sys_freq = (self.sys_freq * mul) / div;
                return 1;
            }
            timeout = timeout - 1;
        }
    }
    return 0;
}

void ClockConfig::set_ahb_div(unsigned int div) {
    self.ahb_div = div;
    unsafe {
        unsigned int* cfg = (unsigned int*)((unsigned long)self.base + (unsigned long)8);
        unsigned int val  = volatile_load(cfg);
        val = val & ~((unsigned int)0xF << 4);
        val = val | ((div & (unsigned int)0xF) << 4);
        volatile_store(cfg, val);
    }
}

void ClockConfig::set_apb1_div(unsigned int div) {
    self.apb1_div = div;
    unsafe {
        unsigned int* cfg = (unsigned int*)((unsigned long)self.base + (unsigned long)8);
        unsigned int val  = volatile_load(cfg);
        val = val & ~((unsigned int)0x7 << 8);
        val = val | ((div & (unsigned int)0x7) << 8);
        volatile_store(cfg, val);
    }
}

void ClockConfig::set_apb2_div(unsigned int div) {
    self.apb2_div = div;
    unsafe {
        unsigned int* cfg = (unsigned int*)((unsigned long)self.base + (unsigned long)8);
        unsigned int val  = volatile_load(cfg);
        val = val & ~((unsigned int)0x7 << 11);
        val = val | ((div & (unsigned int)0x7) << 11);
        volatile_store(cfg, val);
    }
}

unsigned int ClockConfig::get_freq() const {
    return self.sys_freq / self.ahb_div;
}
