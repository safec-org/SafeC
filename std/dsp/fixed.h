// SafeC Standard Library — DSP / Fixed-Point Arithmetic (Q8.24)
// Q8.24: 8 integer bits, 24 fractional bits, stored as a 32-bit signed int.
// Matches 24-bit audio sample depth for sub-sample precision.
#pragma once

// Newtype so Fixed is distinct from plain int in SafeC's type system.
namespace std {

newtype Fixed = int;

// ── Constants ───────────────────────────────────────────────────────────────
// 1.0 in Q8.24
#define FIXED_ONE   16777216
// 0.5 in Q8.24
#define FIXED_HALF  8388608
// π ≈ 3.14159265 in Q8.24  (3.14159265 * 16777216 ≈ 52707178)
#define FIXED_PI    52707178

// ── Constructors / Conversions ───────────────────────────────────────────────

// Convert integer to Fixed (shift left by 24).
Fixed fixed_from_int(int x);

// Convert double to Fixed (multiply by 2^24 and truncate).
Fixed fixed_from_float(double f);

// Convert Fixed to integer (arithmetic right shift by 24 — rounds toward -∞).
int   fixed_to_int(Fixed x);

// Convert Fixed to double.
double fixed_to_float(Fixed x);

// ── Arithmetic ───────────────────────────────────────────────────────────────

// Addition and subtraction (same scale, just add/sub the raw values).
Fixed fixed_add(Fixed a, Fixed b);
Fixed fixed_sub(Fixed a, Fixed b);

// Multiplication: (a * b) >> 24 using 64-bit intermediate.
Fixed fixed_mul(Fixed a, Fixed b);

// Division: (a << 24) / b using 64-bit intermediate.
Fixed fixed_div(Fixed a, Fixed b);

// Absolute value.
Fixed fixed_abs(Fixed x);

// Negation.
Fixed fixed_neg(Fixed x);

// Integer square root (Newton-Raphson, 4 iterations). Argument must be >= 0.
Fixed fixed_sqrt(Fixed x);

} // namespace std
