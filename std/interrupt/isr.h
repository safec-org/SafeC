// SafeC Standard Library — ISR Vector Table
// Interrupt service routine registration and dispatch. Freestanding-safe.
#pragma once

#define ISR_MAX 256

struct IsrTable {
    void* handlers[ISR_MAX];  // function pointers: void(*)(void)
    int   count;              // number of registered handlers

    // Register a handler for IRQ number `irq`. Returns 1 on success, 0 if out of range.
    // Handler signature: void handler(void)
    int   register_(int irq, void* handler);

    // Unregister the handler for IRQ `irq`.
    void  unregister_(int irq);

    // Dispatch: call the handler for IRQ `irq` if registered.
    // Returns 1 if a handler was called, 0 if no handler registered.
    int   dispatch(int irq);
};

// Initialize an ISR vector table (all entries set to NULL).
struct IsrTable isr_init();

// Disable all interrupts (platform-specific inline asm).
void irq_disable();

// Enable all interrupts (platform-specific inline asm).
void irq_enable();
