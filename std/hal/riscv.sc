// SafeC Standard Library — RISC-V HAL Implementation
#pragma once
#include "riscv.h"

// ── CSR accessors ─────────────────────────────────────────────────────────────

unsigned long rv_csr_read_mstatus() {
    unsigned long v = (unsigned long)0;
    unsafe { asm volatile ("csrr %0, mstatus" : "=r"(v)); }
    return v;
}
void rv_csr_write_mstatus(unsigned long val) {
    unsafe { asm volatile ("csrw mstatus, %0" : : "r"(val)); }
}
unsigned long rv_csr_read_mie() {
    unsigned long v = (unsigned long)0;
    unsafe { asm volatile ("csrr %0, mie" : "=r"(v)); }
    return v;
}
void rv_csr_write_mie(unsigned long val) {
    unsafe { asm volatile ("csrw mie, %0" : : "r"(val)); }
}
unsigned long rv_csr_read_mip() {
    unsigned long v = (unsigned long)0;
    unsafe { asm volatile ("csrr %0, mip" : "=r"(v)); }
    return v;
}
unsigned long rv_csr_read_mcause() {
    unsigned long v = (unsigned long)0;
    unsafe { asm volatile ("csrr %0, mcause" : "=r"(v)); }
    return v;
}
unsigned long rv_csr_read_mepc() {
    unsigned long v = (unsigned long)0;
    unsafe { asm volatile ("csrr %0, mepc" : "=r"(v)); }
    return v;
}
void rv_csr_write_mepc(unsigned long val) {
    unsafe { asm volatile ("csrw mepc, %0" : : "r"(val)); }
}
unsigned long rv_csr_read_mtvec() {
    unsigned long v = (unsigned long)0;
    unsafe { asm volatile ("csrr %0, mtvec" : "=r"(v)); }
    return v;
}
void rv_csr_write_mtvec(unsigned long val) {
    unsafe { asm volatile ("csrw mtvec, %0" : : "r"(val)); }
}
unsigned long rv_csr_read_time() {
    unsigned long v = (unsigned long)0;
    unsafe { asm volatile ("csrr %0, time" : "=r"(v)); }
    return v;
}
unsigned long rv_csr_read_cycle() {
    unsigned long v = (unsigned long)0;
    unsafe { asm volatile ("csrr %0, cycle" : "=r"(v)); }
    return v;
}
unsigned long rv_csr_read_instret() {
    unsigned long v = (unsigned long)0;
    unsafe { asm volatile ("csrr %0, instret" : "=r"(v)); }
    return v;
}

void rv_global_irq_enable() {
    // Set MIE bit (bit 3) in mstatus.
    unsafe { asm volatile ("csrsi mstatus, 0x8"); }
}

void rv_global_irq_disable() {
    unsafe { asm volatile ("csrci mstatus, 0x8"); }
}

// ── CLINT ─────────────────────────────────────────────────────────────────────
// Layout: msip[0..MAX_HARTS] @ +0, mtimecmp[hart] @ +0x4000, mtime @ +0xBFF8

struct Clint clint;

void clint_init(unsigned long base_addr) {
    clint.base = base_addr;
}

void Clint::set_msip(unsigned int hart_id) {
    unsafe {
        volatile unsigned int* p = (volatile unsigned int*)(self.base + (unsigned long)hart_id * 4);
        *p = (unsigned int)1;
    }
}

void Clint::clear_msip(unsigned int hart_id) {
    unsafe {
        volatile unsigned int* p = (volatile unsigned int*)(self.base + (unsigned long)hart_id * 4);
        *p = (unsigned int)0;
    }
}

void Clint::set_mtimecmp(unsigned long long cmp) {
    // Split into lo/hi to avoid non-atomic 64-bit write on 32-bit cores.
    unsafe {
        volatile unsigned int* lo = (volatile unsigned int*)(self.base + (unsigned long)0x4000);
        volatile unsigned int* hi = (volatile unsigned int*)(self.base + (unsigned long)0x4004);
        *lo = (unsigned int)0xFFFFFFFF;  // prevent spurious match
        *hi = (unsigned int)((unsigned long long)cmp >> 32);
        *lo = (unsigned int)(cmp & (unsigned long long)0xFFFFFFFF);
    }
}

unsigned long long Clint::read_mtime() const {
    unsafe {
        volatile unsigned int* lo = (volatile unsigned int*)(self.base + (unsigned long)0xBFF8);
        volatile unsigned int* hi = (volatile unsigned int*)(self.base + (unsigned long)0xBFFC);
        unsigned int l, h;
        // Re-read if hi changed during lo read (rollover guard).
        do {
            h = *hi;
            l = *lo;
        } while (*hi != h);
        return ((unsigned long long)h << 32) | (unsigned long long)l;
    }
    return (unsigned long long)0;
}

void Clint::schedule(unsigned long delta) {
    unsigned long long now = self.read_mtime();
    self.set_mtimecmp(now + (unsigned long long)delta);
}

// ── PLIC ──────────────────────────────────────────────────────────────────────
// Priority: base + irq*4
// Enable (ctx0): base + 0x2000
// Threshold (ctx0): base + 0x200000
// Claim/complete (ctx0): base + 0x200004

struct Plic plic;

void plic_init(unsigned long base_addr) {
    plic.base = base_addr;
}

void Plic::set_priority(unsigned int irq, unsigned int priority) {
    unsafe {
        volatile unsigned int* p = (volatile unsigned int*)(self.base + (unsigned long)irq * 4);
        *p = priority & (unsigned int)7;
    }
}

void Plic::enable(unsigned int irq) {
    unsafe {
        unsigned long reg = self.base + (unsigned long)0x2000 + (unsigned long)(irq / 32) * 4;
        volatile unsigned int* p = (volatile unsigned int*)reg;
        *p = *p | ((unsigned int)1 << (irq % 32));
    }
}

void Plic::disable(unsigned int irq) {
    unsafe {
        unsigned long reg = self.base + (unsigned long)0x2000 + (unsigned long)(irq / 32) * 4;
        volatile unsigned int* p = (volatile unsigned int*)reg;
        *p = *p & ~((unsigned int)1 << (irq % 32));
    }
}

void Plic::set_threshold(unsigned int threshold) {
    unsafe {
        volatile unsigned int* p = (volatile unsigned int*)(self.base + (unsigned long)0x200000);
        *p = threshold;
    }
}

unsigned int Plic::claim() {
    unsafe {
        volatile unsigned int* p = (volatile unsigned int*)(self.base + (unsigned long)0x200004);
        return *p;
    }
    return (unsigned int)0;
}

void Plic::complete(unsigned int irq) {
    unsafe {
        volatile unsigned int* p = (volatile unsigned int*)(self.base + (unsigned long)0x200004);
        *p = irq;
    }
}
