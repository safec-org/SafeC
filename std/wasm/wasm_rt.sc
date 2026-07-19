// SafeC Standard Library — freestanding wasm32 allocator runtime (see
// wasm_rt.h). Top-level (non-namespaced) definitions on purpose: these
// must link against the *bare*, unmangled 'malloc'/'free'/etc. symbols
// that std/mem.h's 'extern' declarations reference (SafeC only mangles
// function *definitions* inside a namespace — 'extern' declarations and
// top-level definitions both keep the bare C-ABI name, which is how this
// file's malloc() satisfies std::malloc's extern declaration at link
// time without either side needing to know about the other).
#pragma once
#include <std/wasm/wasm_rt.h>
//
// NOT part of the general stdlib archive (libsafec_std.a): this file
// defines the bare libc-shaped 'malloc'/'free'/'calloc'/'memcpy'/
// 'memmove'/'memset'/'memcmp'/'realloc' symbols wasm32 freestanding
// builds need (see the header comment below) — linking those same
// symbol names into a *hosted* build's archive collides with the real
// libc there (confirmed in CI: Windows' libucrt raised LNK2005 "already
// defined" for exactly these five symbols once this file made it into
// libsafec_std.a). safeguard's Builder.cpp excludes 'std/wasm/' from the
// stdlib archive build specifically because of this file — see the
// 'kExcludedStdSubdirs' comment there. A wasm32 build never links
// against that archive anyway (it's a host-arch .a); it reaches this
// file the same way every test in this repo does, via a direct
// '#include <std/wasm/wasm_rt.sc>' in wasm32-target source.

static unsigned char gWasmRtPool[WASM_RT_POOL_SIZE];
static unsigned long gWasmRtOffset = 0UL;

void* malloc(unsigned long size) {
    unsafe {
        unsigned long headerSize = sizeof(unsigned long);
        unsigned long total = headerSize + size;
        unsigned long rem = total % 8UL;
        if (rem != 0UL) {
            total = total + (8UL - rem);
        }
        if (gWasmRtOffset + total > WASM_RT_POOL_SIZE) {
            return (void*)0;
        }
        unsigned char* base = (unsigned char*)&gWasmRtPool[gWasmRtOffset];
        unsigned long* header = (unsigned long*)base;
        *header = size;
        gWasmRtOffset = gWasmRtOffset + total;
        return (void*)(base + headerSize);
    }
}

void free(void* ptr) {
    // Bump allocator: no reclamation. See wasm_rt.h for rationale.
}

void* calloc(unsigned long count, unsigned long size) {
    unsigned long total = count * size;
    void* p = malloc(total);
    if (p != (void*)0) {
        unsafe { memset(p, 0, total); }
    }
    return p;
}

void* realloc(void* ptr, unsigned long new_size) {
    if (ptr == (void*)0) {
        return malloc(new_size);
    }
    unsigned long oldSize;
    unsafe {
        unsigned char* base = (unsigned char*)ptr;
        unsigned long headerSize = sizeof(unsigned long);
        unsigned long* header = (unsigned long*)(base - headerSize);
        oldSize = *header;
    }
    void* newPtr = malloc(new_size);
    if (newPtr == (void*)0) {
        return (void*)0;
    }
    unsigned long copyLen = oldSize < new_size ? oldSize : new_size;
    unsafe { memcpy(newPtr, ptr, copyLen); }
    return newPtr;
}

void* memcpy(void* dst, const void* src, unsigned long n) {
    unsafe {
        unsigned char* d = (unsigned char*)dst;
        const unsigned char* s = (const unsigned char*)src;
        unsigned long i = 0UL;
        while (i < n) {
            d[i] = s[i];
            i = i + 1UL;
        }
    }
    return dst;
}

void* memmove(void* dst, const void* src, unsigned long n) {
    unsafe {
        unsigned char* d = (unsigned char*)dst;
        const unsigned char* s = (const unsigned char*)src;
        if (d < s) {
            unsigned long i = 0UL;
            while (i < n) {
                d[i] = s[i];
                i = i + 1UL;
            }
        } else {
            unsigned long i = n;
            while (i > 0UL) {
                i = i - 1UL;
                d[i] = s[i];
            }
        }
    }
    return dst;
}

void* memset(void* ptr, int val, unsigned long n) {
    unsafe {
        unsigned char* p = (unsigned char*)ptr;
        unsigned char b = (unsigned char)val;
        unsigned long i = 0UL;
        while (i < n) {
            p[i] = b;
            i = i + 1UL;
        }
    }
    return ptr;
}

int memcmp(const void* a, const void* b, unsigned long n) {
    unsafe {
        const unsigned char* pa = (const unsigned char*)a;
        const unsigned char* pb = (const unsigned char*)b;
        unsigned long i = 0UL;
        while (i < n) {
            if (pa[i] != pb[i]) {
                return (int)pa[i] - (int)pb[i];
            }
            i = i + 1UL;
        }
    }
    return 0;
}
