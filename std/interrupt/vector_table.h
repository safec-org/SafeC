// SafeC Standard Library — Vector Table Management
// Provides a software-managed vector table for Cortex-M, RISC-V, and AArch64.
// Link section placement is controlled by the linker script; this header provides
// the C/SafeC interface to populate and activate the table.
// Freestanding-safe.
#pragma once

#define VTABLE_MAX_VECTORS  256   // Cortex-M can have up to 256 external IRQs + 16 core

// Handler function type: void(*)(void)
typedef void (*IrqHandler)(void);

struct VectorTable {
    IrqHandler handlers[VTABLE_MAX_VECTORS];
    int        count;          // number of valid slots installed

    // Install a handler for vector `index`.
    void  install(int index, IrqHandler handler);

    // Remove handler for vector `index` (replaces with default_handler).
    void  remove(int index);

    // Return the handler at `index`, or NULL if not installed.
    IrqHandler get(int index) const;

    // Dispatch vector `index` (calls installed handler or default).
    void  dispatch(int index);

    // Set the default handler called for unregistered vectors.
    void  set_default(IrqHandler handler);
};

// Global vector table instance.
extern struct VectorTable vtable;

// Initialise all slots to the default handler.
void vtable_init();

// Activate the vector table on the current CPU:
//   Cortex-M: writes VTOR (SCB->VTOR = &vtable)
//   RISC-V:   writes mtvec (vectored mode, base = &vtable)
//   AArch64:  writes VBAR_EL1
void vtable_activate();

// Default (weak) fault handler — infinite loop; override per project.
void vtable_default_handler();
