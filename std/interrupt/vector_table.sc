// SafeC Standard Library — Vector Table Management Implementation
#pragma once
#include <std/interrupt/vector_table.h>

namespace std {

// Definitions of the globals vector_table.h declares extern.
struct VectorTable vtable;


static IrqHandler vtable_default_ = (IrqHandler)0;


inline void vtable_default_handler() {
    // Infinite loop — override in application.
    unsafe { while (1) {} }
}

inline void vtable_init() {
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

inline void VectorTable::install(int index, IrqHandler handler) {
    if (index < 0 || index >= VTABLE_MAX_VECTORS) { return; }
    if (handler == (IrqHandler)0) { return; }
    self.handlers[index] = handler;
    if (index >= self.count) { self.count = index + 1; }
}

inline void VectorTable::remove(int index) {
    if (index < 0 || index >= VTABLE_MAX_VECTORS) { return; }
    self.handlers[index] = vtable_default_;
}

inline IrqHandler VectorTable::get(int index) const {
    if (index < 0 || index >= VTABLE_MAX_VECTORS) { return (IrqHandler)0; }
    return self.handlers[index];
}

inline void VectorTable::dispatch(int index) {
    IrqHandler h = (IrqHandler)0;
    if (index >= 0 && index < VTABLE_MAX_VECTORS) { h = self.handlers[index]; }
    if (h == (IrqHandler)0) { h = vtable_default_; }   // the set_default() handler
    if (h == (IrqHandler)0) {
        vtable_default_handler();
        return;
    }
    unsafe { h(); }
}

// Also retargets every slot still holding the previous default, so vectors
// that were never installed reach the new handler.
inline void VectorTable::set_default(IrqHandler handler) {
    if (handler == (IrqHandler)0) { return; }
    IrqHandler old = vtable_default_;
    int i = 0;
    while (i < VTABLE_MAX_VECTORS) {
        if (self.handlers[i] == old || self.handlers[i] == (IrqHandler)0) { self.handlers[i] = handler; }
        i = i + 1;
    }
    vtable_default_ = handler;
}

inline void vtable_activate() {
    unsafe {
#if defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_8M_MAIN__) || defined(__ARM_ARCH_8M_BASE__) || defined(__ARM_ARCH_8_1M_MAIN__)
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

} // namespace std
