// SafeC Standard Library — Hardware Debug Helpers Implementation
#pragma once
#include "jtag.h"

// ── debug_break ───────────────────────────────────────────────────────────────
// Emits a software breakpoint instruction for the current architecture.
// If no debugger is attached the CPU will:
//   - ARM Cortex-M: trigger a DebugMon or HardFault exception (config-dependent)
//   - x86-64: raise SIGTRAP (caught by the OS; usually terminates process without dbg)
//   - RISC-V: trigger a breakpoint exception (handled by debug module or M-mode trap)

void debug_break() {
#if defined(__aarch64__) || defined(__arm64__)
    // AArch64 — BRK #0
    unsafe { asm volatile ("brk #0"); }

#elif defined(__arm__) || defined(__thumb__)
    // ARM / Thumb — BKPT #0
    unsafe { asm volatile ("bkpt #0"); }

#elif defined(__x86_64__) || defined(_M_X64)
    // x86-64 — INT3 (single-byte 0xCC)
    unsafe { asm volatile ("int3"); }

#elif defined(__riscv)
    // RISC-V — EBREAK
    unsafe { asm volatile ("ebreak"); }

#else
    // Fallback: no-op on unrecognised architectures.
#endif
}

// ── debug_semihost_puts ───────────────────────────────────────────────────────
// ARM Cortex-M semihosting protocol:
//   r0 = operation (0x04 = SYS_WRITE0, writes null-terminated string)
//   r1 = pointer to string
//   Trigger via BKPT #0xAB.
//
// On AArch64 the semihosting HLT #0xF000 is used.
// x86-64 and RISC-V have no standard semihosting — no-op.

void debug_semihost_puts(const char* s) {
#if defined(__arm__) || defined(__thumb__)
    unsafe {
        // Cortex-M semihosting: r0=4 (SYS_WRITE0), r1=s.
        asm volatile (
            "mov r0, #4\n"
            "mov r1, %0\n"
            "bkpt #0xAB"
            :
            : "r"(s)
            : "r0", "r1"
        );
    }

#elif defined(__aarch64__) || defined(__arm64__)
    unsafe {
        // AArch64 semihosting: w0=4 (SYS_WRITE0), x1=s, HLT #0xF000.
        asm volatile (
            "mov w0, #4\n"
            "mov x1, %0\n"
            "hlt #0xF000"
            :
            : "r"(s)
            : "x0", "x1"
        );
    }

#else
    // x86-64, RISC-V: no semihosting standard — no-op.
    (void)s;
#endif
}

// ── debug_semihost_putc ───────────────────────────────────────────────────────
// SYS_WRITEC (0x03): write a single character.  r1 points to the char.

void debug_semihost_putc(char c) {
#if defined(__arm__) || defined(__thumb__)
    unsafe {
        char buf[1];
        buf[0] = c;
        asm volatile (
            "mov r0, #3\n"
            "mov r1, %0\n"
            "bkpt #0xAB"
            :
            : "r"((const char*)buf)
            : "r0", "r1"
        );
    }

#elif defined(__aarch64__) || defined(__arm64__)
    unsafe {
        char buf[1];
        buf[0] = c;
        asm volatile (
            "mov w0, #3\n"
            "mov x1, %0\n"
            "hlt #0xF000"
            :
            : "r"((const char*)buf)
            : "x0", "x1"
        );
    }

#else
    (void)c;
#endif
}

// ── ITM helpers ───────────────────────────────────────────────────────────────
// ITM_TCR_BASE + 0 is ITMTCR; bit 0 is ITMENA (ITM enable).
// ITM_TCR_BASE + 0xE00 - 0xE80 + 0x00 = ITM_TER (trace enable register).
// ITM_TER: each bit i enables port i.  Address: ITM_STIMULUS_BASE + 0xE00.
// We use the TER at 0xE0000E00 (ITM_TER for ports 0-31).

#define ITM_TER_ADDR  0xE0000E00UL
#define ITM_TCR_ADDR  0xE0000E80UL

static unsigned int itm_reg_read_(unsigned long addr) {
    unsafe {
        volatile unsigned int* p = (volatile unsigned int*)addr;
        return *p;
    }
    return (unsigned int)0;
}

static void itm_reg_write_(unsigned long addr, unsigned int val) {
    unsafe {
        volatile unsigned int* p = (volatile unsigned int*)addr;
        *p = val;
    }
}

// Enable ITM stimulus port `port` (0-31).
void itm_enable_port(unsigned char port) {
    unsigned int mask = (unsigned int)1 << ((unsigned int)port & (unsigned int)31);
    unsigned int ter  = itm_reg_read_((unsigned long)ITM_TER_ADDR);
    itm_reg_write_((unsigned long)ITM_TER_ADDR, ter | mask);
}

// Write a byte to ITM stimulus port `port`.
// Only writes if ITM is enabled (ITMTCR bit 0) and the port is enabled (TER).
void itm_putc(unsigned char port, unsigned char c) {
    // Check ITM enabled.
    if ((itm_reg_read_((unsigned long)ITM_TCR_ADDR) & (unsigned int)1) == (unsigned int)0) {
        return;
    }
    // Check port enabled in TER.
    unsigned int port_mask = (unsigned int)1 << ((unsigned int)port & (unsigned int)31);
    if ((itm_reg_read_((unsigned long)ITM_TER_ADDR) & port_mask) == (unsigned int)0) {
        return;
    }
    // Stimulus port register: ITM_STIMULUS_BASE + port*4.
    unsigned long stim_addr = (unsigned long)ITM_STIMULUS_BASE
                            + (unsigned long)port * (unsigned long)4;
    // Poll until port not busy (bit 0 of stimulus register must be 1).
    unsafe {
        volatile unsigned int* stim = (volatile unsigned int*)stim_addr;
        while ((*stim & (unsigned int)1) == (unsigned int)0) {}
        // Write byte via byte-width write (use byte pointer).
        volatile unsigned char* stim8 = (volatile unsigned char*)stim_addr;
        *stim8 = c;
    }
}

// Write a 32-bit word to ITM stimulus port `port`.
void itm_put32(unsigned char port, unsigned int word) {
    if ((itm_reg_read_((unsigned long)ITM_TCR_ADDR) & (unsigned int)1) == (unsigned int)0) {
        return;
    }
    unsigned int port_mask = (unsigned int)1 << ((unsigned int)port & (unsigned int)31);
    if ((itm_reg_read_((unsigned long)ITM_TER_ADDR) & port_mask) == (unsigned int)0) {
        return;
    }
    unsigned long stim_addr = (unsigned long)ITM_STIMULUS_BASE
                            + (unsigned long)port * (unsigned long)4;
    unsafe {
        volatile unsigned int* stim = (volatile unsigned int*)stim_addr;
        while ((*stim & (unsigned int)1) == (unsigned int)0) {}
        *stim = word;
    }
}
