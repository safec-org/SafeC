// SafeC Standard Library — Hardware Debug Helpers (JTAG / SWD / Semihosting)
// Provides software breakpoint, semihosting output, and OpenOCD-compatible
// trace via ITM (Cortex-M) or T-trace stubs.
// Freestanding-safe.
#pragma once

// ── Software breakpoint ───────────────────────────────────────────────────────
// Inserts a software breakpoint instruction (BKPT on ARM, INT3 on x86-64).
// No-op if no debugger is connected (on most targets; may fault on some).
void  debug_break();

// ── Semihosting output ────────────────────────────────────────────────────────
// Write a null-terminated string to the debugger console via semihosting.
// No-op if running without a debugger (semihosting not enabled).
void  debug_semihost_puts(const char* s);

// Write one byte to the debugger console.
void  debug_semihost_putc(char c);

// ── ITM Stimulus Port (Cortex-M) ──────────────────────────────────────────────
#define ITM_STIMULUS_BASE  0xE0000000UL
#define ITM_TCR_BASE       0xE0000E80UL

// Write a byte to ITM stimulus port `port` (0-31).
// No-op if ITM not enabled (checks TER register).
void  itm_putc(unsigned char port, unsigned char c);

// Write a 32-bit word to ITM stimulus port `port`.
void  itm_put32(unsigned char port, unsigned int word);

// Enable ITM stimulus port `port`.
void  itm_enable_port(unsigned char port);

// ── Assert with debug break ───────────────────────────────────────────────────
#define DBG_ASSERT(cond)  do { if (!(cond)) { debug_break(); } } while(0)
