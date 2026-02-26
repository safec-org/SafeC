// SafeC Standard Library — ISR Vector Table
#pragma once
#include "isr.h"

struct IsrTable isr_init() {
    struct IsrTable t;
    t.count = 0;
    int i = 0;
    while (i < 256) {
        t.handlers[i] = (void*)0;
        i = i + 1;
    }
    return t;
}

int IsrTable::register_(int irq, void* handler) {
    if (irq < 0 || irq >= 256) { return 0; }
    self.handlers[irq] = handler;
    if (handler != (void*)0) { self.count = self.count + 1; }
    return 1;
}

void IsrTable::unregister_(int irq) {
    if (irq >= 0 && irq < 256) {
        if (self.handlers[irq] != (void*)0) {
            self.handlers[irq] = (void*)0;
            self.count = self.count - 1;
        }
    }
}

int IsrTable::dispatch(int irq) {
    if (irq < 0 || irq >= 256) { return 0; }
    if (self.handlers[irq] == (void*)0) { return 0; }
    unsafe { ((void(*)(void))self.handlers[irq])(); }
    return 1;
}

void irq_disable() {
    unsafe { asm volatile("" ::: "memory"); }
}

void irq_enable() {
    unsafe { asm volatile("" ::: "memory"); }
}
