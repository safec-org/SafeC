// SafeC Standard Library — ISR Vector Table
#pragma once
#include <std/interrupt/isr.h>

namespace std {

inline struct IsrTable IsrTable::new() {
    struct IsrTable t;
    t.count = 0;
    int i = 0;
    while (i < ISR_MAX) {
        t.handlers[i] = (void*)0;
        i = i + 1;
    }
    return t;
}

inline int IsrTable::register_(int irq, fn void() handler) {
    if (irq < 0 || irq >= ISR_MAX) { return 0; }
    // Re-registering an IRQ replaces its handler; count only new slots.
    if (self.handlers[irq] == (void*)0) { self.count = self.count + 1; }
    unsafe { self.handlers[irq] = (void*)handler; }
    return 1;
}

inline void IsrTable::unregister_(int irq) {
    if (irq >= 0 && irq < ISR_MAX) {
        if (self.handlers[irq] != (void*)0) {
            self.handlers[irq] = (void*)0;
            self.count = self.count - 1;
        }
    }
}

inline int IsrTable::dispatch(int irq) {
    if (irq < 0 || irq >= ISR_MAX) { return 0; }
    if (self.handlers[irq] == (void*)0) { return 0; }
    unsafe {
        fn void(void) handler = (fn void(void))self.handlers[irq];
        handler();
    }
    return 1;
}

// Masks/unmasks IRQs at the CPU on bare metal (privileged code). A hosted
// process can't touch the interrupt flag, so there (and on wasm) these are
// only compiler barriers.
inline void irq_disable() {
    unsafe {
#if !defined(__SAFEC_FREESTANDING__)
        asm volatile("" ::: "memory");
#elif defined(__aarch64__)
        asm volatile("msr daifset, #2" ::: "memory");
#elif defined(__arm__)
        asm volatile("cpsid i" ::: "memory");
#elif defined(__riscv)
        asm volatile("csrci mstatus, 8" ::: "memory");   // clear MIE
#elif defined(__x86_64__) || defined(__i386__)
        asm volatile("cli" ::: "memory");
#else
        asm volatile("" ::: "memory");
#endif
    }
}

inline void irq_enable() {
    unsafe {
#if !defined(__SAFEC_FREESTANDING__)
        asm volatile("" ::: "memory");
#elif defined(__aarch64__)
        asm volatile("msr daifclr, #2" ::: "memory");
#elif defined(__arm__)
        asm volatile("cpsie i" ::: "memory");
#elif defined(__riscv)
        asm volatile("csrsi mstatus, 8" ::: "memory");   // set MIE
#elif defined(__x86_64__) || defined(__i386__)
        asm volatile("sti" ::: "memory");
#else
        asm volatile("" ::: "memory");
#endif
    }
}

} // namespace std
