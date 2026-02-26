// SafeC Standard Library — Timer HAL
// Generic MMIO layout:
//   base + 0x00 = control register (bit 0 = enable)
//   base + 0x04 = prescaler register
//   base + 0x08 = period/auto-reload register
//   base + 0x0C = counter register (read-only)
//   base + 0x10 = status register (bit 0 = overflow flag)
#pragma once
#include "timer.h"

struct Timer timer_init(void* base, unsigned int prescaler) {
    struct Timer t;
    t.base      = base;
    t.prescaler = prescaler;
    unsafe {
        unsigned int* psc_reg = (unsigned int*)((unsigned long)base + (unsigned long)4);
        volatile_store(psc_reg, prescaler);
    }
    return t;
}

void Timer::set_period(unsigned int period) {
    unsafe {
        unsigned int* arr_reg = (unsigned int*)((unsigned long)self.base + (unsigned long)8);
        volatile_store(arr_reg, period);
    }
}

void Timer::start() {
    unsafe {
        unsigned int* ctrl = (unsigned int*)self.base;
        unsigned int val = volatile_load(ctrl);
        val = val | (unsigned int)1;
        volatile_store(ctrl, val);
    }
}

void Timer::stop() {
    unsafe {
        unsigned int* ctrl = (unsigned int*)self.base;
        unsigned int val = volatile_load(ctrl);
        val = val & ~(unsigned int)1;
        volatile_store(ctrl, val);
    }
}

unsigned int Timer::read() const {
    unsafe {
        unsigned int* cnt = (unsigned int*)((unsigned long)self.base + (unsigned long)12);
        return volatile_load(cnt);
    }
}

void Timer::clear_flag() {
    unsafe {
        unsigned int* status = (unsigned int*)((unsigned long)self.base + (unsigned long)16);
        volatile_store(status, (unsigned int)0);
    }
}

int Timer::flag_set() const {
    unsafe {
        unsigned int* status = (unsigned int*)((unsigned long)self.base + (unsigned long)16);
        unsigned int val = volatile_load(status);
        if ((val & (unsigned int)1) != (unsigned int)0) { return 1; }
        return 0;
    }
}
