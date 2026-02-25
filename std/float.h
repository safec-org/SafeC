// SafeC Standard Library — Floating-point limit constants (C89/C99/C11/C17/C23)
// All values assume IEEE 754 binary32 (float) and binary64 (double).
#pragma once

// Radix of floating-point representation (always 2 for IEEE 754).
#define FLT_RADIX  2

// float — mantissa bits (including implicit leading 1 bit)
#define FLT_MANT_DIG     24
// float — significant decimal digits
#define FLT_DIG           6
// float — minimum exponent (unbiased)
#define FLT_MIN_EXP    (-125)
// float — maximum exponent
#define FLT_MAX_EXP     128
// float — minimum decimal exponent
#define FLT_MIN_10_EXP (-37)
// float — maximum decimal exponent
#define FLT_MAX_10_EXP  38
// float — machine epsilon (2^-23)
#define FLT_EPSILON    1.1920929e-07f
// float — smallest positive normalised value
#define FLT_MIN        1.1754944e-38f
// float — largest finite value
#define FLT_MAX        3.4028235e+38f
// float — smallest positive value (subnormal)
#define FLT_TRUE_MIN   1.4012985e-45f

// double — mantissa bits
#define DBL_MANT_DIG       53
// double — significant decimal digits
#define DBL_DIG            15
// double — minimum exponent (unbiased)
#define DBL_MIN_EXP     (-1021)
// double — maximum exponent
#define DBL_MAX_EXP      1024
// double — minimum decimal exponent
#define DBL_MIN_10_EXP  (-307)
// double — maximum decimal exponent
#define DBL_MAX_10_EXP   308
// double — machine epsilon (2^-52)
#define DBL_EPSILON     2.2204460492503131e-16
// double — smallest positive normalised value
#define DBL_MIN         2.2250738585072014e-308
// double — largest finite value
#define DBL_MAX         1.7976931348623157e+308
// double — smallest positive value (subnormal)
#define DBL_TRUE_MIN    5.0e-324

// long double — treated as double on LP64 / LLP64 without 80-bit FPU extension
#define LDBL_MANT_DIG   DBL_MANT_DIG
#define LDBL_DIG        DBL_DIG
#define LDBL_MIN_EXP    DBL_MIN_EXP
#define LDBL_MAX_EXP    DBL_MAX_EXP
#define LDBL_EPSILON    DBL_EPSILON
#define LDBL_MIN        DBL_MIN
#define LDBL_MAX        DBL_MAX

// Decimal rounding mode: 1 = round to nearest (IEEE 754 default).
#define FLT_ROUNDS  1

// Evaluation method: 0 = each operation evaluated in its declared type.
#define FLT_EVAL_METHOD  0

// Decimal digits needed to represent any long double value uniquely.
#define DECIMAL_DIG  21
