// SafeC Standard Library — Integer limit constants (C89/C99/C11/C17/C23)
// Implementation assumes LP64 (64-bit POSIX) or LLP64 (Windows 64-bit).
#pragma once

// ── Bit width of a byte ───────────────────────────────────────────────────────
#define CHAR_BIT   8

// ── char limits ───────────────────────────────────────────────────────────────
#define SCHAR_MIN   (-128)
#define SCHAR_MAX   127
#define UCHAR_MAX   255
// char is signed on most platforms; adjust if building for unsigned-char target
#define CHAR_MIN    SCHAR_MIN
#define CHAR_MAX    SCHAR_MAX

// ── short limits ──────────────────────────────────────────────────────────────
#define SHRT_MIN    (-32768)
#define SHRT_MAX    32767
#define USHRT_MAX   65535

// ── int limits ────────────────────────────────────────────────────────────────
#define INT_MIN     (-2147483647 - 1)
#define INT_MAX     2147483647
#define UINT_MAX    4294967295U

// ── long limits (LP64: long = 64-bit; LLP64: long = 32-bit) ──────────────────
// We follow the LP64 convention (Linux/macOS 64-bit).
// On Windows x64 (LLP64), long is 32-bit — include platform checks if needed.
#define LONG_MIN    (-9223372036854775807LL - (long long)1)
#define LONG_MAX    9223372036854775807LL
#define ULONG_MAX   18446744073709551615ULL

// ── long long limits (always 64-bit) ─────────────────────────────────────────
#define LLONG_MIN   (-9223372036854775807LL - (long long)1)
#define LLONG_MAX   9223372036854775807LL
#define ULLONG_MAX  18446744073709551615ULL

// ── Multibyte character limit ─────────────────────────────────────────────────
#define MB_LEN_MAX  16
