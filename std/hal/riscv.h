// SafeC Standard Library — RISC-V HAL (CLINT, PLIC, CSRs)
// Freestanding-safe. Addresses follow the SiFive FE310 / standard RISC-V MMIO map.
#pragma once

// ── CSR helpers ───────────────────────────────────────────────────────────────
// Read / write arbitrary CSRs by name (uses inline asm).

unsigned long rv_csr_read_mstatus();
void          rv_csr_write_mstatus(unsigned long val);
unsigned long rv_csr_read_mie();
void          rv_csr_write_mie(unsigned long val);
unsigned long rv_csr_read_mip();
unsigned long rv_csr_read_mcause();
unsigned long rv_csr_read_mepc();
void          rv_csr_write_mepc(unsigned long val);
unsigned long rv_csr_read_mtvec();
void          rv_csr_write_mtvec(unsigned long val);
unsigned long rv_csr_read_time();
unsigned long rv_csr_read_cycle();
unsigned long rv_csr_read_instret();

// Global interrupt enable / disable.
void rv_global_irq_enable();
void rv_global_irq_disable();

// ── CLINT — Core-Local Interruptor ────────────────────────────────────────────
#define CLINT_BASE  0x02000000UL

struct Clint {
    unsigned long base;

    // Software interrupt (msip) for hart `hart_id`.
    void  set_msip(unsigned int hart_id);
    void  clear_msip(unsigned int hart_id);

    // Machine timer compare for hart 0.
    void          set_mtimecmp(unsigned long long cmp);
    unsigned long long read_mtime() const;

    // Schedule a timer interrupt `delta` ticks in the future.
    void  schedule(unsigned long delta);
};

extern struct Clint clint;
void clint_init(unsigned long base_addr);

// ── PLIC — Platform-Level Interrupt Controller ───────────────────────────────
#define PLIC_BASE  0x0C000000UL
#define PLIC_MAX_IRQ  53

struct Plic {
    unsigned long base;

    // Set interrupt priority (1–7; 0 = disabled).
    void  set_priority(unsigned int irq, unsigned int priority);

    // Enable interrupt `irq` for target context 0 (M-mode hart 0).
    void  enable(unsigned int irq);
    void  disable(unsigned int irq);

    // Set priority threshold for context 0.
    void  set_threshold(unsigned int threshold);

    // Claim highest-priority pending interrupt; returns IRQ number (0 = none).
    unsigned int claim();

    // Complete interrupt handling for `irq`.
    void  complete(unsigned int irq);
};

extern struct Plic plic;
void plic_init(unsigned long base_addr);
