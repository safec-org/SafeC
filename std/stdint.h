// SafeC Standard Library — Fixed-width integer types (C99/C11/C17/C23)
// Provides exact-width, minimum-width, and fastest integer type aliases.
#pragma once

// ── Exact-width signed integers ───────────────────────────────────────────────
#define int8_t   signed char
#define int16_t  short
#define int32_t  int
#define int64_t  long long

// ── Exact-width unsigned integers ─────────────────────────────────────────────
#define uint8_t   unsigned char
#define uint16_t  unsigned short
#define uint32_t  unsigned int
#define uint64_t  unsigned long long

// ── Pointer-sized integers ────────────────────────────────────────────────────
#define intptr_t   long long
#define uintptr_t  unsigned long long
#define ptrdiff_t  long long

// ── Maximum-width integers ────────────────────────────────────────────────────
#define intmax_t   long long
#define uintmax_t  unsigned long long

// ── size_t ────────────────────────────────────────────────────────────────────
#define size_t  unsigned long long

// ── Exact-width signed limits ─────────────────────────────────────────────────
#define INT8_MIN    (-128)
#define INT8_MAX    127
#define INT16_MIN   (-32768)
#define INT16_MAX   32767
#define INT32_MIN   (-2147483647 - 1)
#define INT32_MAX   2147483647
#define INT64_MIN   (-9223372036854775807LL - (long long)1)
#define INT64_MAX   9223372036854775807LL

// ── Exact-width unsigned limits ───────────────────────────────────────────────
#define UINT8_MAX   255
#define UINT16_MAX  65535
#define UINT32_MAX  4294967295U
#define UINT64_MAX  18446744073709551615ULL

// ── Pointer / ptrdiff / size limits ───────────────────────────────────────────
#define INTPTR_MIN   INT64_MIN
#define INTPTR_MAX   INT64_MAX
#define UINTPTR_MAX  UINT64_MAX
#define PTRDIFF_MIN  INT64_MIN
#define PTRDIFF_MAX  INT64_MAX
#define SIZE_MAX     UINT64_MAX

// ── Maximum-width limits ──────────────────────────────────────────────────────
#define INTMAX_MIN   INT64_MIN
#define INTMAX_MAX   INT64_MAX
#define UINTMAX_MAX  UINT64_MAX

// ── Minimum-width type aliases (commonly INT_LEAST*_t) ────────────────────────
#define int_least8_t    signed char
#define int_least16_t   short
#define int_least32_t   int
#define int_least64_t   long long
#define uint_least8_t   unsigned char
#define uint_least16_t  unsigned short
#define uint_least32_t  unsigned int
#define uint_least64_t  unsigned long long

// ── Fast integer aliases (at least N bits, fastest representation) ────────────
#define int_fast8_t    int
#define int_fast16_t   int
#define int_fast32_t   int
#define int_fast64_t   long long
#define uint_fast8_t   unsigned int
#define uint_fast16_t  unsigned int
#define uint_fast32_t  unsigned int
#define uint_fast64_t  unsigned long long

// ── Integer constant suffixes (note: SafeC safe mode bans function-like macros) ──
// Use integer suffixes directly:
//   8-bit  : cast, e.g. (int8_t)42
//   16-bit : cast, e.g. (int16_t)42
//   32-bit : plain literal, e.g. 42
//   64-bit : LL suffix, e.g. 42LL
//   unsigned 32-bit : U suffix, e.g. 42U
//   unsigned 64-bit : ULL suffix, e.g. 42ULL
