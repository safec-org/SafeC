#pragma once
// SafeC Standard Library — Floating-point environment (C99 <fenv.h>)
// Read and set floating-point exception flags and rounding mode.

// Rounding modes (C99 FE_* constants, platform values)
const int FE_TONEAREST()  { return 0; }
const int FE_DOWNWARD()   { return 1024; }
const int FE_UPWARD()     { return 2048; }
const int FE_TOWARDZERO() { return 3072; }

// Exception flags (combined bitmask)
const int FE_INVALID()   { return 1; }
const int FE_DIVBYZERO() { return 4; }
const int FE_OVERFLOW()  { return 8; }
const int FE_UNDERFLOW() { return 16; }
const int FE_INEXACT()   { return 32; }
const int FE_ALL_EXCEPT() { return 63; }

// Clear specified exception flags (pass FE_ALL_EXCEPT() to clear all).
int fenv_clear(int excepts);

// Test whether any of the specified exceptions are raised.
int fenv_test(int excepts);

// Raise specified exceptions.
int fenv_raise(int excepts);

// Get current rounding mode (returns FE_* constant).
int fenv_get_round();

// Set rounding mode (pass FE_* constant). Returns 0 on success.
int fenv_set_round(int mode);

// Save and restore the complete FP environment.
// env must point to a 32-byte buffer.
int fenv_save(void* env);
int fenv_restore(const void* env);
int fenv_save_clear(void* env); // save + clear all exceptions
