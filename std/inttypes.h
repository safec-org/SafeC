// SafeC Standard Library — Fixed-width integer format specifiers (C99/C11/C17/C23)
// Use these macros with printf/scanf and the io_printf/io_scan family.
#pragma once
#include "stdint.h"

// ── printf: signed decimal ────────────────────────────────────────────────────
#define PRId8    "hhd"
#define PRId16   "hd"
#define PRId32   "d"
#define PRId64   "lld"
#define PRIdMAX  "lld"
#define PRIdPTR  "lld"

// ── printf: unsigned decimal ──────────────────────────────────────────────────
#define PRIu8    "hhu"
#define PRIu16   "hu"
#define PRIu32   "u"
#define PRIu64   "llu"
#define PRIuMAX  "llu"
#define PRIuPTR  "llu"

// ── printf: unsigned octal ────────────────────────────────────────────────────
#define PRIo8    "hho"
#define PRIo16   "ho"
#define PRIo32   "o"
#define PRIo64   "llo"
#define PRIoMAX  "llo"

// ── printf: unsigned hex (lowercase) ─────────────────────────────────────────
#define PRIx8    "hhx"
#define PRIx16   "hx"
#define PRIx32   "x"
#define PRIx64   "llx"
#define PRIxMAX  "llx"
#define PRIxPTR  "llx"

// ── printf: unsigned hex (uppercase) ─────────────────────────────────────────
#define PRIX8    "hhX"
#define PRIX16   "hX"
#define PRIX32   "X"
#define PRIX64   "llX"
#define PRIXMAX  "llX"

// ── scanf: signed decimal ─────────────────────────────────────────────────────
#define SCNd8    "hhd"
#define SCNd16   "hd"
#define SCNd32   "d"
#define SCNd64   "lld"
#define SCNdMAX  "lld"
#define SCNdPTR  "lld"

// ── scanf: unsigned decimal ───────────────────────────────────────────────────
#define SCNu8    "hhu"
#define SCNu16   "hu"
#define SCNu32   "u"
#define SCNu64   "llu"
#define SCNuMAX  "llu"
#define SCNuPTR  "llu"

// ── scanf: unsigned hex (lowercase) ──────────────────────────────────────────
#define SCNx8    "hhx"
#define SCNx16   "hx"
#define SCNx32   "x"
#define SCNx64   "llx"
#define SCNxMAX  "llx"
