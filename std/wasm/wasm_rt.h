#pragma once
// SafeC Standard Library — freestanding wasm32 allocator runtime.
//
// A wasm32-unknown-unknown build is linked with -nostdlib (see
// std/wasm/dom.h's header comment for the full toolchain invocation) —
// there is no libc, so std/mem.h's 'extern void* malloc(...)' etc. have
// no definition to link against unless something in the module provides
// one. This file *is* that something: a bump allocator over one static
// linear-memory pool, standing in for a browser/Node-provided malloc that
// doesn't exist (JS has no such export). free() is a no-op (a pure bump
// allocator can't reclaim without a header + free list, and a hydrated
// page's allocation volume — a handful of short struct String/struct
// Value/struct Signal buffers per interaction — doesn't warrant one) —
// documented, not hidden: long-running pages that allocate unboundedly
// will exhaust WASM_RT_POOL_SIZE. realloc() copies into a fresh
// allocation (the old block is leaked like any other free()).
//
// Only meaningful for wasm32 freestanding targets — a hosted build
// (Cocoa/Win32/X11/bare-metal GUI, the RPC server, etc.) must keep using
// real libc malloc/free via std/mem.sc, not this file; including both in
// the same link would duplicate-symbol on malloc/free/etc.
#include <std/stddef.h>

#define WASM_RT_POOL_SIZE (4UL * 1024UL * 1024UL) // 4 MiB

void* malloc(size_t size);
void  free(void* ptr);
void* calloc(size_t count, size_t size);
void* realloc(void* ptr, size_t new_size);
void* memcpy(void* dst, const void* src, size_t n);
void* memmove(void* dst, const void* src, size_t n);
void* memset(void* ptr, int val, size_t n);
int   memcmp(const void* a, const void* b, size_t n);
