// SafeC Standard Library — AArch64 HAL (GIC, Generic Timer, System Registers)
// Freestanding-safe. Targets ARMv8-A (Cortex-A53/A57/A72/A55/A76).
#pragma once

// ── System register helpers ───────────────────────────────────────────────────

unsigned long aa64_read_mpidr();     // processor affinity / CPU ID
unsigned long aa64_read_currentel(); // current exception level (EL0–EL3)
unsigned long aa64_read_daif();      // interrupt mask bits
void          aa64_write_daif(unsigned long val);
void          aa64_irq_enable();     // clear DAIF.I
void          aa64_irq_disable();    // set DAIF.I
void          aa64_fiq_enable();     // clear DAIF.F
void          aa64_fiq_disable();    // set DAIF.F
void          aa64_isb();
void          aa64_dsb_sy();
void          aa64_dmb_sy();

// ── Generic Timer ─────────────────────────────────────────────────────────────

struct Aa64Timer {
    // EL1 Physical Counter / Timer.
    unsigned long long read_cntpct();    // system counter value
    unsigned long      read_cntfrq();    // counter frequency in Hz
    void               set_tval(unsigned int tval);   // EL1 physical tval (countdown)
    void               enable();
    void               disable();
    int                fire_pending() const;
};

extern struct Aa64Timer aa64_timer;

// ── GIC — Generic Interrupt Controller (GICv2 compatible) ────────────────────

#define GIC_DIST_BASE  0x08000000UL   // typical QEMU virt / Juno board
#define GIC_CPU_BASE   0x08010000UL

struct GicDist {
    unsigned long base;

    void          enable_group0();
    void          enable_irq(unsigned int irq);
    void          disable_irq(unsigned int irq);
    void          set_priority(unsigned int irq, unsigned char priority);
    void          set_target(unsigned int irq, unsigned char cpu_mask);
    void          set_config(unsigned int irq, int edge_triggered);
    int           is_pending(unsigned int irq) const;
    void          clear_pending(unsigned int irq);
};

struct GicCpu {
    unsigned long base;

    void          enable(unsigned char min_priority);  // set priority mask
    void          disable();
    unsigned int  ack();                 // IAR — returns IRQ number
    void          eoi(unsigned int irq); // EOIR
    unsigned int  running_priority() const;
};

extern struct GicDist  gic_dist;
extern struct GicCpu   gic_cpu;
void gic_init(unsigned long dist_base, unsigned long cpu_base);
