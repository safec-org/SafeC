// SafeC Standard Library — Standard type definitions (C99/C11/C17/C23)
// Fundamental types, NULL, offsetof, max_align_t.
#pragma once

// ── Fundamental type aliases ───────────────────────────────────────────────────
// size_t & co. are pointer-width (the C ABI's types: libc's size_t
// parameters are 32-bit on 32-bit targets); SafeC's own 'long' is always
// 64-bit. __SIZEOF_POINTER__ is predefined by safec from the target.
#ifndef SAFEC_PTRWIDTH_TYPES_
#define SAFEC_PTRWIDTH_TYPES_
#if __SIZEOF_POINTER__ == 4
#define size_t      unsigned int
#define ssize_t     int
#define ptrdiff_t   int
#define intptr_t    int
#define uintptr_t   unsigned int
#else
#define size_t      unsigned long long
#define ssize_t     long long
#define ptrdiff_t   long long
#define intptr_t    long long
#define uintptr_t   unsigned long long
#endif
#endif
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
