// SafeC Standard Library — Vector Table Management Implementation
#pragma once
#include "vector_table.h"

static IrqHandler vtable_default_ = (IrqHandler)0;

struct VectorTable vtable;

void vtable_default_handler() {
    // Infinite loop — override in application.
    unsafe { while (1) {} }
}

void vtable_init() {
    vtable.count = 0;
    if (vtable_default_ == (IrqHandler)0) {
        vtable_default_ = vtable_default_handler;
    }
    int i = 0;
    while (i < VTABLE_MAX_VECTORS) {
        vtable.handlers[i] = vtable_default_;
        i = i + 1;
    }
}

void VectorTable::install(int index, IrqHandler handler) {
    if (index < 0 || index >= VTABLE_MAX_VECTORS) { return; }
    if (handler == (IrqHandler)0) { return; }
    self.handlers[index] = handler;
    if (index >= self.count) { self.count = index + 1; }
}

void VectorTable::remove(int index) {
    if (index < 0 || index >= VTABLE_MAX_VECTORS) { return; }
    self.handlers[index] = vtable_default_;
}

IrqHandler VectorTable::get(int index) const {
    if (index < 0 || index >= VTABLE_MAX_VECTORS) { return (IrqHandler)0; }
    return self.handlers[index];
}

void VectorTable::dispatch(int index) {
    if (index < 0 || index >= VTABLE_MAX_VECTORS) {
        vtable_default_handler();
        return;
    }
    IrqHandler h = self.handlers[index];
    if (h == (IrqHandler)0) {
        vtable_default_handler();
        return;
    }
    unsafe { h(); }
}

void VectorTable::set_default(IrqHandler handler) {
    if (handler != (IrqHandler)0) { vtable_default_ = handler; }
}

void vtable_activate() {
    unsafe {
#if defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_8M_MAIN__)
        // Cortex-M: set VTOR to address of vtable.handlers[0]
        volatile unsigned int* vtor = (volatile unsigned int*)0xE000ED08;
        *vtor = (unsigned int)(unsigned long)&vtable.handlers[0];
#elif defined(__riscv)
        // RISC-V: vectored mode — base = &vtable.handlers[0], mode bits = 01
        unsigned long base = (unsigned long)&vtable.handlers[0];
        asm volatile ("csrw mtvec, %0" : : "r"(base | (unsigned long)1));
#elif defined(__aarch64__)
        // AArch64: write VBAR_EL1
        unsigned long base = (unsigned long)&vtable.handlers[0];
        asm volatile ("msr vbar_el1, %0" : : "r"(base));
        asm volatile ("isb");
#else
        // No-op on unsupported architectures.
        (void)vtable;
#endif
    }
}
