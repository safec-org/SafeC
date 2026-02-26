// SafeC Standard Library — DSP / Fixed-Point Arithmetic (Q16.16)
// Q16.16: 16 integer bits, 16 fractional bits, stored as a 32-bit signed int.
#pragma once

// Newtype so Fixed is distinct from plain int in SafeC's type system.
newtype Fixed = int;

// ── Constants ───────────────────────────────────────────────────────────────
// 1.0 in Q16.16
#define FIXED_ONE   65536
// 0.5 in Q16.16
#define FIXED_HALF  32768
// π ≈ 3.14159265 in Q16.16  (3.14159265 * 65536 ≈ 205887)
#define FIXED_PI    205887

// ── Constructors / Conversions ───────────────────────────────────────────────

// Convert integer to Fixed (shift left by 16).
Fixed fixed_from_int(int x);

// Convert double to Fixed (multiply by 65536 and truncate).
Fixed fixed_from_float(double f);

// Convert Fixed to integer (arithmetic right shift by 16 — rounds toward -∞).
int   fixed_to_int(Fixed x);

// Convert Fixed to double.
double fixed_to_float(Fixed x);

// ── Arithmetic ───────────────────────────────────────────────────────────────

// Addition and subtraction (same scale, just add/sub the raw values).
Fixed fixed_add(Fixed a, Fixed b);
Fixed fixed_sub(Fixed a, Fixed b);

// Multiplication: (a * b) >> 16 using 64-bit intermediate.
Fixed fixed_mul(Fixed a, Fixed b);

// Division: (a << 16) / b using 64-bit intermediate.
Fixed fixed_div(Fixed a, Fixed b);

// Absolute value.
Fixed fixed_abs(Fixed x);

// Negation.
Fixed fixed_neg(Fixed x);

// Integer square root (Newton-Raphson, 3 iterations). Argument must be >= 0.
Fixed fixed_sqrt(Fixed x);
