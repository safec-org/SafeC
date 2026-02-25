// SafeC Standard Library — Standard type definitions (C99/C11/C17/C23)
// Fundamental types, NULL, offsetof, max_align_t.
#pragma once

// ── Fundamental type aliases ───────────────────────────────────────────────────
#define size_t      unsigned long long
#define ptrdiff_t   long long
#define max_align_t long double

// ── NULL pointer ──────────────────────────────────────────────────────────────
#ifndef NULL
#define NULL  0
#endif

// ── offsetof — byte offset of a struct member ─────────────────────────────────
// SafeC safe mode does not allow function-like macros.
// Use the sys_offsetof() function provided in sys.h for runtime member offsets,
// or rely on SafeC struct layout which is deterministic and matches C ABI.
// (Compile with --compat-preprocessor if you need C's offsetof macro.)

// ── unreachable (C23) ─────────────────────────────────────────────────────────
// Mark unreachable paths with: unsafe { __builtin_unreachable(); }
// or simply: sys_abort();  (in sys.h)
