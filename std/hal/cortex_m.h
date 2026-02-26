// SafeC Standard Library — Cortex-M HAL (NVIC, SysTick, SCB)
// Freestanding-safe. Register addresses follow the ARM-v7M/v8-M memory map.
#pragma once

// ── NVIC — Nested Vectored Interrupt Controller ───────────────────────────────
#define NVIC_BASE  0xE000E100UL

struct Nvic {
    void* base;   // NVIC_BASE cast to void* for MMIO access

    // Enable interrupt `irq` (0–239).
    void enable(unsigned int irq);

    // Disable interrupt `irq`.
    void disable(unsigned int irq);

    // Return 1 if interrupt `irq` is currently pending.
    int  is_pending(unsigned int irq) const;

    // Clear pending flag for `irq`.
    void clear_pending(unsigned int irq);

    // Set priority (0–255; lower = higher priority) for `irq`.
    void set_priority(unsigned int irq, unsigned char priority);

    // Return current priority for `irq`.
    unsigned char get_priority(unsigned int irq) const;
};

// Global NVIC instance at the standard Cortex-M base address.
extern struct Nvic nvic;
void nvic_init();

// ── SysTick ───────────────────────────────────────────────────────────────────
#define SYSTICK_BASE  0xE000E010UL

struct SysTick {
    void* base;

    // Configure and enable SysTick with `reload` counts per tick.
    void  start(unsigned int reload, int use_core_clock);

    // Stop SysTick.
    void  stop();

    // Return current counter value.
    unsigned int read() const;

    // Return 1 if COUNTFLAG is set (counter reached zero since last read).
    int  flag_set() const;
};

extern struct SysTick systick;
void systick_init();

// ── SCB — System Control Block ────────────────────────────────────────────────
#define SCB_BASE  0xE000ED00UL

struct Scb {
    void* base;

    // Software reset.
    void  system_reset();

    // Enable / disable UsageFault, BusFault, MemManage.
    void  enable_fault(unsigned int fault_mask);
    void  disable_fault(unsigned int fault_mask);

    // Configure VTOR (vector table offset register).
    void  set_vtor(unsigned int addr);

    // Return 1 if the core is in Handler mode.
    int   in_handler_mode() const;

    // Set process-stack pointer.
    void  set_psp(unsigned int addr);

    // Set main-stack pointer.
    void  set_msp(unsigned int addr);
};

extern struct Scb scb;
void scb_init();

// Fault mask bits for enable_fault / disable_fault
#define SCB_FAULT_USAGE  (1u << 18)
#define SCB_FAULT_BUS    (1u << 17)
#define SCB_FAULT_MEM    (1u << 16)
