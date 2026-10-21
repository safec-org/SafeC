#pragma once
// SafeC Standard Library — Floating-point environment (C99 <fenv.h>)
// Read and set floating-point exception flags and rounding mode.

// Rounding modes and exception flags (C99 FE_* constants). Their values
// are the platform's <fenv.h> values, which differ per architecture.
namespace std {

#if defined(__aarch64__) || defined(__arm__)
static const int FE_TONEAREST()  { return 0; }
static const int FE_UPWARD()     { return 0x400000; }
static const int FE_DOWNWARD()   { return 0x800000; }
static const int FE_TOWARDZERO() { return 0xC00000; }

static const int FE_INVALID()   { return 1; }
static const int FE_DIVBYZERO() { return 2; }
static const int FE_OVERFLOW()  { return 4; }
static const int FE_UNDERFLOW() { return 8; }
static const int FE_INEXACT()   { return 16; }
static const int FE_ALL_EXCEPT() { return 31; }
#elif defined(__riscv)
static const int FE_TONEAREST()  { return 0; }
static const int FE_TOWARDZERO() { return 1; }
static const int FE_DOWNWARD()   { return 2; }
static const int FE_UPWARD()     { return 3; }

static const int FE_INEXACT()   { return 1; }
static const int FE_UNDERFLOW() { return 2; }
static const int FE_OVERFLOW()  { return 4; }
static const int FE_DIVBYZERO() { return 8; }
static const int FE_INVALID()   { return 16; }
static const int FE_ALL_EXCEPT() { return 31; }
#else   // x86 / x86-64
static const int FE_TONEAREST()  { return 0; }
static const int FE_DOWNWARD()   { return 1024; }
static const int FE_UPWARD()     { return 2048; }
static const int FE_TOWARDZERO() { return 3072; }

static const int FE_INVALID()   { return 1; }
static const int FE_DIVBYZERO() { return 4; }
static const int FE_OVERFLOW()  { return 8; }
static const int FE_UNDERFLOW() { return 16; }
static const int FE_INEXACT()   { return 32; }
static const int FE_ALL_EXCEPT() { return 63; }
#endif

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
int fenv_save(&unsigned char env);
int fenv_restore(const &unsigned char env);
int fenv_save_clear(&unsigned char env); // save + clear all exceptions

} // namespace std
