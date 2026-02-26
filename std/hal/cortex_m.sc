// SafeC Standard Library — Cortex-M HAL Implementation
#pragma once
#include "cortex_m.h"

// MMIO read/write helpers for 32-bit registers.
static unsigned int cm_read32_(unsigned long addr) {
    unsafe {
        volatile unsigned int* p = (volatile unsigned int*)addr;
        return *p;
    }
    return (unsigned int)0;
}

static void cm_write32_(unsigned long addr, unsigned int val) {
    unsafe {
        volatile unsigned int* p = (volatile unsigned int*)addr;
        *p = val;
    }
}

// ── NVIC ──────────────────────────────────────────────────────────────────────
// NVIC_ISER0: base + 0x000  (32 IRQs per word)
// NVIC_ICER0: base + 0x080
// NVIC_ISPR0: base + 0x100
// NVIC_ICPR0: base + 0x180
// NVIC_IPR0:  base + 0x300  (8-bit priority per IRQ)

struct Nvic nvic;

void nvic_init() {
    unsafe { nvic.base = (void*)NVIC_BASE; }
}

void Nvic::enable(unsigned int irq) {
    unsigned long reg = (unsigned long)self.base + (unsigned long)0 + (unsigned long)(irq / 32) * 4;
    cm_write32_(reg, (unsigned int)1 << (irq % 32));
}

void Nvic::disable(unsigned int irq) {
    unsigned long reg = (unsigned long)self.base + (unsigned long)0x80 + (unsigned long)(irq / 32) * 4;
    cm_write32_(reg, (unsigned int)1 << (irq % 32));
}

int Nvic::is_pending(unsigned int irq) const {
    unsigned long reg = (unsigned long)self.base + (unsigned long)0x100 + (unsigned long)(irq / 32) * 4;
    unsigned int val = cm_read32_(reg);
    if ((val >> (irq % 32)) & (unsigned int)1) { return 1; }
    return 0;
}

void Nvic::clear_pending(unsigned int irq) {
    unsigned long reg = (unsigned long)self.base + (unsigned long)0x180 + (unsigned long)(irq / 32) * 4;
    cm_write32_(reg, (unsigned int)1 << (irq % 32));
}

void Nvic::set_priority(unsigned int irq, unsigned char priority) {
    unsigned long reg = (unsigned long)self.base + (unsigned long)0x300 + (unsigned long)irq;
    unsafe {
        volatile unsigned char* p = (volatile unsigned char*)reg;
        *p = priority;
    }
}

unsigned char Nvic::get_priority(unsigned int irq) const {
    unsigned long reg = (unsigned long)self.base + (unsigned long)0x300 + (unsigned long)irq;
    unsafe {
        volatile unsigned char* p = (volatile unsigned char*)reg;
        return *p;
    }
    return (unsigned char)0;
}

// ── SysTick ───────────────────────────────────────────────────────────────────
// SYST_CSR  = base + 0x0
// SYST_RVR  = base + 0x4
// SYST_CVR  = base + 0x8

struct SysTick systick;

void systick_init() {
    unsafe { systick.base = (void*)SYSTICK_BASE; }
}

void SysTick::start(unsigned int reload, int use_core_clock) {
    unsigned long base = (unsigned long)self.base;
    cm_write32_(base + (unsigned long)4, reload & (unsigned int)0x00FFFFFF);
    cm_write32_(base + (unsigned long)8, (unsigned int)0);  // clear CVR
    unsigned int csr = (unsigned int)0x3;  // TICKINT | ENABLE
    if (use_core_clock != 0) { csr = csr | (unsigned int)0x4; }
    cm_write32_(base, csr);
}

void SysTick::stop() {
    unsigned long base = (unsigned long)self.base;
    cm_write32_(base, (unsigned int)0);
}

unsigned int SysTick::read() const {
    return cm_read32_((unsigned long)self.base + (unsigned long)8);
}

int SysTick::flag_set() const {
    unsigned int csr = cm_read32_((unsigned long)self.base);
    if ((csr >> 16) & (unsigned int)1) { return 1; }
    return 0;
}

// ── SCB ───────────────────────────────────────────────────────────────────────
// SCB: CPUID=+0, ICSR=+4, VTOR=+8, AIRCR=+0xC, SCR=+0x10, CCR=+0x14, SHCSR=+0x24

struct Scb scb;

void scb_init() {
    unsafe { scb.base = (void*)SCB_BASE; }
}

void Scb::system_reset() {
    unsafe {
        unsigned long base = (unsigned long)self.base;
        unsigned int val = (unsigned int)0x05FA0004;  // VECTKEY | SYSRESETREQ
        cm_write32_(base + (unsigned long)0xC, val);
        asm volatile ("dsb");
        while (1) {}
    }
}

void Scb::enable_fault(unsigned int fault_mask) {
    unsigned long reg = (unsigned long)self.base + (unsigned long)0x24;
    unsigned int val = cm_read32_(reg);
    cm_write32_(reg, val | fault_mask);
}

void Scb::disable_fault(unsigned int fault_mask) {
    unsigned long reg = (unsigned long)self.base + (unsigned long)0x24;
    unsigned int val = cm_read32_(reg);
    cm_write32_(reg, val & ~fault_mask);
}

void Scb::set_vtor(unsigned int addr) {
    cm_write32_((unsigned long)self.base + (unsigned long)8, addr);
}

int Scb::in_handler_mode() const {
    unsigned int icsr = cm_read32_((unsigned long)self.base + (unsigned long)4);
    if ((icsr & (unsigned int)0x1FF) != (unsigned int)0) { return 1; }
    return 0;
}

void Scb::set_psp(unsigned int addr) {
    unsafe { asm volatile ("msr psp, %0" : : "r"(addr)); }
}

void Scb::set_msp(unsigned int addr) {
    unsafe { asm volatile ("msr msp, %0" : : "r"(addr)); }
}
