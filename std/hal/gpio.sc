// SafeC Standard Library — GPIO HAL
// Assumes standard MMIO layout:
//   base + 0x00 = direction register
//   base + 0x04 = output data register
//   base + 0x08 = input data register
//   base + 0x0C = pull config register
#pragma once
#include "gpio.h"

struct GpioPin gpio_init(void* base, int pin, int direction) {
    struct GpioPin p;
    p.base      = base;
    p.pin_mask  = (unsigned int)1 << pin;
    p.direction = direction;
    unsafe {
        unsigned int* dir_reg = (unsigned int*)base;
        unsigned int val = volatile_load(dir_reg);
        if (direction == 1) {
            val = val | p.pin_mask;
        } else {
            val = val & ~p.pin_mask;
        }
        volatile_store(dir_reg, val);
    }
    return p;
}

void GpioPin::set_direction(int direction) {
    self.direction = direction;
    unsafe {
        unsigned int* dir_reg = (unsigned int*)self.base;
        unsigned int val = volatile_load(dir_reg);
        if (direction == 1) {
            val = val | self.pin_mask;
        } else {
            val = val & ~self.pin_mask;
        }
        volatile_store(dir_reg, val);
    }
}

void GpioPin::write(int value) {
    unsafe {
        unsigned int* out_reg = (unsigned int*)((unsigned long)self.base + (unsigned long)4);
        unsigned int val = volatile_load(out_reg);
        if (value != 0) {
            val = val | self.pin_mask;
        } else {
            val = val & ~self.pin_mask;
        }
        volatile_store(out_reg, val);
    }
}

int GpioPin::read() const {
    unsafe {
        unsigned int* in_reg = (unsigned int*)((unsigned long)self.base + (unsigned long)8);
        unsigned int val = volatile_load(in_reg);
        if ((val & self.pin_mask) != (unsigned int)0) {
            return 1;
        }
        return 0;
    }
}

void GpioPin::toggle() {
    unsafe {
        unsigned int* out_reg = (unsigned int*)((unsigned long)self.base + (unsigned long)4);
        unsigned int val = volatile_load(out_reg);
        val = val ^ self.pin_mask;
        volatile_store(out_reg, val);
    }
}

void GpioPin::set_pull(int pull) {
    unsafe {
        unsigned int* pull_reg = (unsigned int*)((unsigned long)self.base + (unsigned long)12);
        unsigned int val = volatile_load(pull_reg);
        val = val & ~self.pin_mask;
        if (pull == 1 || pull == 2) {
            val = val | self.pin_mask;
        }
        volatile_store(pull_reg, val);
    }
}
