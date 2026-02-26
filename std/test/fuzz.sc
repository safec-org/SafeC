// SafeC Standard Library — Fuzz Harness
#pragma once
#include "fuzz.h"

extern void* malloc(unsigned long size);
extern void  free(void* ptr);
extern void* memcpy(void* d, const void* s, unsigned long n);

// ── XorShift64 PRNG ───────────────────────────────────────────────────────────
static unsigned long fuzz_rng_(unsigned long* state) {
    unsigned long x = *state;
    x = x ^ (x << (unsigned long)13);
    x = x ^ (x >> (unsigned long)7);
    x = x ^ (x << (unsigned long)17);
    *state = x;
    return x;
}

struct FuzzTarget fuzz_target_init(void* fn, unsigned long iters) {
    struct FuzzTarget t;
    t.fn    = fn;
    t.seed  = (unsigned long)0xDEADBEEFCAFEBABE;
    t.iters = iters;
    return t;
}

void FuzzTarget::run(const unsigned char* corpus, unsigned long corpus_size) {
    if (corpus_size == (unsigned long)0) { return; }

    unsigned long state = self.seed;

    // Working buffer: start as a copy of the corpus.
    unsafe {
        unsigned char* buf = (unsigned char*)malloc(corpus_size);
        if (buf == (unsigned char*)0) { return; }
        memcpy((void*)buf, (const void*)corpus, corpus_size);

        // Call target on the original corpus first.
        ((void(*)(const unsigned char*, unsigned long))self.fn)(buf, corpus_size);

        unsigned long i = (unsigned long)0;
        while (i < self.iters) {
            // Pick a random mutation: bit flip, byte set, truncate, or extend.
            unsigned long r   = fuzz_rng_(&state);
            unsigned long mut = r & (unsigned long)3;
            unsigned long pos = fuzz_rng_(&state) % corpus_size;

            if (mut == (unsigned long)0) {
                // Bit flip at random position.
                unsigned long bit = fuzz_rng_(&state) & (unsigned long)7;
                buf[pos] = buf[pos] ^ (unsigned char)(1 << bit);
            } else if (mut == (unsigned long)1) {
                // Random byte set.
                buf[pos] = (unsigned char)(fuzz_rng_(&state) & (unsigned long)0xFF);
            } else if (mut == (unsigned long)2) {
                // Restore to original corpus.
                memcpy((void*)buf, (const void*)corpus, corpus_size);
            } else {
                // Overwrite two adjacent bytes.
                buf[pos] = (unsigned char)(fuzz_rng_(&state) & (unsigned long)0xFF);
                unsigned long pos2 = (pos + (unsigned long)1) % corpus_size;
                buf[pos2] = (unsigned char)(fuzz_rng_(&state) & (unsigned long)0xFF);
            }

            // Vary the size: sometimes pass a prefix, sometimes full buffer.
            unsigned long sz = corpus_size;
            if ((fuzz_rng_(&state) & (unsigned long)3) == (unsigned long)0) {
                sz = (fuzz_rng_(&state) % corpus_size) + (unsigned long)1;
            }

            ((void(*)(const unsigned char*, unsigned long))self.fn)(buf, sz);
            i = i + (unsigned long)1;
        }

        free((void*)buf);
    }
}
