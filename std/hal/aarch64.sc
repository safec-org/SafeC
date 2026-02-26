// SafeC Standard Library — AArch64 HAL Implementation
#pragma once
#include "aarch64.h"

// ── System registers ──────────────────────────────────────────────────────────

unsigned long aa64_read_mpidr() {
    unsigned long v = (unsigned long)0;
    unsafe { asm volatile ("mrs %0, mpidr_el1" : "=r"(v)); }
    return v;
}
unsigned long aa64_read_currentel() {
    unsigned long v = (unsigned long)0;
    unsafe { asm volatile ("mrs %0, CurrentEL" : "=r"(v)); }
    return (v >> 2) & (unsigned long)3;
}
unsigned long aa64_read_daif() {
    unsigned long v = (unsigned long)0;
    unsafe { asm volatile ("mrs %0, daif" : "=r"(v)); }
    return v;
}
void aa64_write_daif(unsigned long val) {
    unsafe { asm volatile ("msr daif, %0" : : "r"(val)); }
}
void aa64_irq_enable()  { unsafe { asm volatile ("msr daifclr, #2"); } }
void aa64_irq_disable() { unsafe { asm volatile ("msr daifset, #2"); } }
void aa64_fiq_enable()  { unsafe { asm volatile ("msr daifclr, #1"); } }
void aa64_fiq_disable() { unsafe { asm volatile ("msr daifset, #1"); } }
void aa64_isb()   { unsafe { asm volatile ("isb" : : : "memory"); } }
void aa64_dsb_sy(){ unsafe { asm volatile ("dsb sy" : : : "memory"); } }
void aa64_dmb_sy(){ unsafe { asm volatile ("dmb sy" : : : "memory"); } }

// ── Generic Timer ─────────────────────────────────────────────────────────────

struct Aa64Timer aa64_timer;

unsigned long long Aa64Timer::read_cntpct() {
    unsigned long v = (unsigned long)0;
    unsafe { asm volatile ("mrs %0, cntpct_el0" : "=r"(v)); }
    return (unsigned long long)v;
}

unsigned long Aa64Timer::read_cntfrq() {
    unsigned long v = (unsigned long)0;
    unsafe { asm volatile ("mrs %0, cntfrq_el0" : "=r"(v)); }
    return v;
}

void Aa64Timer::set_tval(unsigned int tval) {
    unsafe { asm volatile ("msr cntp_tval_el0, %0" : : "r"((unsigned long)tval)); }
}

void Aa64Timer::enable() {
    // CNTP_CTL_EL0: bit0=ENABLE, bit1=IMASK (clear), bit2=ISTATUS
    unsafe { asm volatile ("msr cntp_ctl_el0, %0" : : "r"((unsigned long)1)); }
}

void Aa64Timer::disable() {
    unsafe { asm volatile ("msr cntp_ctl_el0, %0" : : "r"((unsigned long)0)); }
}

int Aa64Timer::fire_pending() const {
    unsigned long v = (unsigned long)0;
    unsafe { asm volatile ("mrs %0, cntp_ctl_el0" : "=r"(v)); }
    // ISTATUS = bit 2
    if ((v >> 2) & (unsigned long)1) { return 1; }
    return 0;
}

// ── GIC helpers ───────────────────────────────────────────────────────────────

static unsigned int gic_read32_(unsigned long addr) {
    unsafe {
        volatile unsigned int* p = (volatile unsigned int*)addr;
        return *p;
    }
    return (unsigned int)0;
}
static void gic_write32_(unsigned long addr, unsigned int val) {
    unsafe {
        volatile unsigned int* p = (volatile unsigned int*)addr;
        *p = val;
    }
}

// ── GicDist ───────────────────────────────────────────────────────────────────
// GICD_CTLR   +0x000  GICD_ISENABLER +0x100  GICD_ICENABLER +0x180
// GICD_ISPENDR+0x200  GICD_ICPENDR   +0x280
// GICD_IPRIORITYR +0x400  GICD_ITARGETSR +0x800  GICD_ICFGR +0xC00

struct GicDist gic_dist;
struct GicCpu  gic_cpu;

void GicDist::enable_group0() {
    gic_write32_(self.base, (unsigned int)1);
}

void GicDist::enable_irq(unsigned int irq) {
    unsigned long reg = self.base + (unsigned long)0x100 + (unsigned long)(irq / 32) * 4;
    gic_write32_(reg, (unsigned int)1 << (irq % 32));
}

void GicDist::disable_irq(unsigned int irq) {
    unsigned long reg = self.base + (unsigned long)0x180 + (unsigned long)(irq / 32) * 4;
    gic_write32_(reg, (unsigned int)1 << (irq % 32));
}

void GicDist::set_priority(unsigned int irq, unsigned char priority) {
    unsigned long reg = self.base + (unsigned long)0x400 + (unsigned long)irq;
    unsafe {
        volatile unsigned char* p = (volatile unsigned char*)reg;
        *p = priority;
    }
}

void GicDist::set_target(unsigned int irq, unsigned char cpu_mask) {
    unsigned long reg = self.base + (unsigned long)0x800 + (unsigned long)irq;
    unsafe {
        volatile unsigned char* p = (volatile unsigned char*)reg;
        *p = cpu_mask;
    }
}

void GicDist::set_config(unsigned int irq, int edge_triggered) {
    unsigned long reg = self.base + (unsigned long)0xC00 + (unsigned long)(irq / 16) * 4;
    unsigned int  shift = (irq % 16) * 2;
    unsigned int val = gic_read32_(reg);
    if (edge_triggered != 0) {
        val = val | ((unsigned int)2 << shift);
    } else {
        val = val & ~((unsigned int)3 << shift);
    }
    gic_write32_(reg, val);
}

int GicDist::is_pending(unsigned int irq) const {
    unsigned long reg = self.base + (unsigned long)0x200 + (unsigned long)(irq / 32) * 4;
    unsigned int val = gic_read32_(reg);
    if ((val >> (irq % 32)) & (unsigned int)1) { return 1; }
    return 0;
}

void GicDist::clear_pending(unsigned int irq) {
    unsigned long reg = self.base + (unsigned long)0x280 + (unsigned long)(irq / 32) * 4;
    gic_write32_(reg, (unsigned int)1 << (irq % 32));
}

// ── GicCpu ────────────────────────────────────────────────────────────────────
// GICC_CTLR +0x000  GICC_PMR +0x004  GICC_IAR +0x00C  GICC_EOIR +0x010
// GICC_RPR  +0x014

void GicCpu::enable(unsigned char min_priority) {
    gic_write32_(self.base + (unsigned long)4, (unsigned int)min_priority);
    gic_write32_(self.base, (unsigned int)1);
}

void GicCpu::disable() {
    gic_write32_(self.base, (unsigned int)0);
}

unsigned int GicCpu::ack() {
    return gic_read32_(self.base + (unsigned long)0xC) & (unsigned int)0x3FF;
}

void GicCpu::eoi(unsigned int irq) {
    gic_write32_(self.base + (unsigned long)0x10, irq & (unsigned int)0x3FF);
}

unsigned int GicCpu::running_priority() const {
    return gic_read32_(self.base + (unsigned long)0x14);
}

void gic_init(unsigned long dist_base, unsigned long cpu_base) {
    gic_dist.base = dist_base;
    gic_cpu.base  = cpu_base;
}
