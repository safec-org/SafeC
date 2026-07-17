// SafeC Standard Library — ISR Vector Table
#pragma once
#include <std/interrupt/isr.h>

namespace std {

inline struct IsrTable isr_init() {
    struct IsrTable t;
    t.count = 0;
    int i = 0;
    while (i < 256) {
        t.handlers[i] = (void*)0;
        i = i + 1;
    }
    return t;
}

inline int IsrTable::register_(int irq, void* handler) {
    if (irq < 0 || irq >= 256) { return 0; }
    self.handlers[irq] = handler;
    if (handler != (void*)0) { self.count = self.count + 1; }
    return 1;
}

inline void IsrTable::unregister_(int irq) {
    if (irq >= 0 && irq < 256) {
        if (self.handlers[irq] != (void*)0) {
            self.handlers[irq] = (void*)0;
            self.count = self.count - 1;
        }
    }
}

inline int IsrTable::dispatch(int irq) {
    if (irq < 0 || irq >= 256) { return 0; }
    if (self.handlers[irq] == (void*)0) { return 0; }
    unsafe {
        fn void(void) handler = (fn void(void))self.handlers[irq];
        handler();
    }
    return 1;
}

inline void irq_disable() {
    unsafe { asm volatile("" ::: "memory"); }
}

inline void irq_enable() {
    unsafe { asm volatile("" ::: "memory"); }
}

} // namespace std
