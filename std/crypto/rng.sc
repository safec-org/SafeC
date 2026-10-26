// SafeC Standard Library — Cryptographic RNG (ChaCha20-based)
// safeguard: link windows -lbcrypt          (BCryptGenRandom: the seed on Windows)
#pragma once
#include <std/stddef.h>
#include <std/crypto/rng.h>
#include <std/stderr_compat.h>

namespace std {

extern unsigned char* memcpy(unsigned char* d, const unsigned char* s, size_t n);
// Hosted-mode seeding from /dev/urandom. Declared here at file scope: this
// parser only accepts 'extern' declarations at top level, not nested inside
// a function body.
extern CFile_* fopen(const char* path, const char* mode);
extern size_t fread(unsigned char* buf, size_t size, size_t n, CFile_* f);
extern int fclose(CFile_* f);
// Same rationale as mem.sc/pool.sc/slab.sc/tlsf.sc for not using
// panic.h/panic_at here (no forced --compat-preprocessor link dependency
// for every RNG user).
extern void  abort();
extern int   fprintf(CFile_* stream, const char* fmt, ...);
// The OS's own CSPRNG, the first seed source tried (see RngCtx::init).
#if defined(_WIN32)
// bcrypt.dll: NTSTATUS BCryptGenRandom(BCRYPT_ALG_HANDLE, PUCHAR, ULONG, ULONG);
// a null algorithm with BCRYPT_USE_SYSTEM_PREFERRED_RNG (2) needs no handle.
extern int BCryptGenRandom(unsigned char* alg, unsigned char* buf, unsigned int len, unsigned int flags);
#elif defined(__APPLE__)
extern int getentropy(unsigned char* buf, size_t len);              // libSystem, at most 256 bytes
#elif defined(__linux__)
extern long getrandom(unsigned char* buf, size_t len, unsigned int flags);   // glibc 2.25+, musl
#endif

static void rng_abort_(const char* msg) {
    unsafe { fprintf(SAFEC_STDERR_, "std::crypto::rng fatal: %s\n", msg); }
    unsafe { abort(); }
}

// ── ChaCha20 core ─────────────────────────────────────────────────────────────

static unsigned int chacha_rotl_(unsigned int v, int n) {
    return (v << n) | (v >> (32 - n));
}

static void chacha_qr_(unsigned int* a, unsigned int* b,
                        unsigned int* c, unsigned int* d) {
    unsafe {
        *a = *a + *b; *d = chacha_rotl_(*d ^ *a, 16);
        *c = *c + *d; *b = chacha_rotl_(*b ^ *c, 12);
        *a = *a + *b; *d = chacha_rotl_(*d ^ *a,  8);
        *c = *c + *d; *b = chacha_rotl_(*b ^ *c,  7);
    }
}

// Produce 64 bytes of keystream into `out` from state `s`.
static void chacha20_block_(const unsigned int* s, unsigned char* out) {
    unsigned int x[16];
    int i = 0;
    unsafe { while (i < 16) { x[i] = s[i]; i = i + 1; } }

    // 10 double-rounds
    int r = 0;
    while (r < 10) {
        unsafe {
            chacha_qr_((unsigned int*)&x[ 0],(unsigned int*)&x[ 4],(unsigned int*)&x[ 8],(unsigned int*)&x[12]);
            chacha_qr_((unsigned int*)&x[ 1],(unsigned int*)&x[ 5],(unsigned int*)&x[ 9],(unsigned int*)&x[13]);
            chacha_qr_((unsigned int*)&x[ 2],(unsigned int*)&x[ 6],(unsigned int*)&x[10],(unsigned int*)&x[14]);
            chacha_qr_((unsigned int*)&x[ 3],(unsigned int*)&x[ 7],(unsigned int*)&x[11],(unsigned int*)&x[15]);
            chacha_qr_((unsigned int*)&x[ 0],(unsigned int*)&x[ 5],(unsigned int*)&x[10],(unsigned int*)&x[15]);
            chacha_qr_((unsigned int*)&x[ 1],(unsigned int*)&x[ 6],(unsigned int*)&x[11],(unsigned int*)&x[12]);
            chacha_qr_((unsigned int*)&x[ 2],(unsigned int*)&x[ 7],(unsigned int*)&x[ 8],(unsigned int*)&x[13]);
            chacha_qr_((unsigned int*)&x[ 3],(unsigned int*)&x[ 4],(unsigned int*)&x[ 9],(unsigned int*)&x[14]);
        }
        r = r + 1;
    }

    i = 0;
    unsafe { while (i < 16) { x[i] = x[i] + s[i]; i = i + 1; } }

    // Serialise little-endian
    unsafe {
        i = 0;
        while (i < 16) {
            out[i*4+0] = (unsigned char)( x[i]       & 0xFF);
            out[i*4+1] = (unsigned char)((x[i] >>  8) & 0xFF);
            out[i*4+2] = (unsigned char)((x[i] >> 16) & 0xFF);
            out[i*4+3] = (unsigned char)((x[i] >> 24) & 0xFF);
            i = i + 1;
        }
    }
}

// ── Initialisation ─────────────────────────────────────────────────────────────

// ChaCha20 "expa nd 32-byte k" constant
static unsigned int chacha_const_[4] = {
    0x61707865, 0x3320646e, 0x79622d32, 0x6b206574
};

static void rng_setup_(unsigned int* state, const unsigned char* key32,
                        unsigned int ctr, unsigned int nonce) {
    int i = 0;
    unsafe { while (i < 4) { state[i] = chacha_const_[i]; i = i + 1; } }
    unsafe {
        i = 0;
        while (i < 8) {
            state[4+i] = ((unsigned int)key32[i*4+0])
                       | ((unsigned int)key32[i*4+1] << 8)
                       | ((unsigned int)key32[i*4+2] << 16)
                       | ((unsigned int)key32[i*4+3] << 24);
            i = i + 1;
        }
    }
    unsafe {
        state[12] = ctr;
        state[13] = (unsigned int)0;
        state[14] = nonce;
        state[15] = (unsigned int)0;
    }
}

inline void RngCtx::init_seed(const unsigned char seed[32]) {
    rng_setup_(self.state, seed, (unsigned int)0, (unsigned int)1);
    chacha20_block_(self.state, (unsigned char*)self.buf);
    self.state[12] = self.state[12] + (unsigned int)1;
    self.buf_pos = (unsigned long)0;
}

int RngCtx::init() {
    // The OS's CSPRNG first: BCryptGenRandom on Windows (where there is no
    // /dev/urandom), getentropy on macOS, getrandom on Linux.
#if defined(_WIN32) || defined(__APPLE__) || defined(__linux__)
    unsafe {
        unsigned char os_seed[32];
        int os_ok = 0;
#if defined(_WIN32)
        if (BCryptGenRandom((unsigned char*)0, (unsigned char*)os_seed, (unsigned int)32, (unsigned int)2) == 0) { os_ok = 1; }
#elif defined(__APPLE__)
        if (getentropy((unsigned char*)os_seed, (size_t)32) == 0) { os_ok = 1; }
#else
        if (getrandom((unsigned char*)os_seed, (size_t)32, (unsigned int)0) == (long)32) { os_ok = 1; }
#endif
        if (os_ok != 0) {
            self.init_seed(os_seed);
            return 1;
        }
    }
#endif

    // Else rdrand (x86-64), else /dev/urandom (a Linux kernel without
    // getrandom, the BSDs).
#ifdef __x86_64__
    // Try rdrand for seeding
    unsigned int seed_words[8];
    int ok = 1;
    unsafe {
        int i = 0;
        while (i < 8) {
            unsigned int val = (unsigned int)0;
            int success = 0;
            int tries = 0;
            while (tries < 10 && success == 0) {
                asm volatile (
                    "xorl %0, %0\n\t"
                    "rdrand %1\n\t"
                    "setc %b0"
                    : "=r"(success), "=r"(val) : : "cc"
                );
                tries = tries + 1;
            }
            if (success == 0) { ok = 0; }
            seed_words[i] = val;
            i = i + 1;
        }
    }
    if (ok != 0) {
        rng_setup_(self.state, (const unsigned char*)seed_words, (unsigned int)0, (unsigned int)1);
        chacha20_block_(self.state, (unsigned char*)self.buf);
        self.state[12] = self.state[12] + (unsigned int)1;
        self.buf_pos = (unsigned long)0;
        return 1;
    }
#endif

#ifndef __SAFEC_FREESTANDING__
    // Hosted: seed from /dev/urandom
    unsafe {
        CFile_* f = fopen("/dev/urandom", "rb");
        if (f != (CFile_*)0) {
            unsigned char seed[32];
            unsigned long got = fread((unsigned char*)seed, (unsigned long)1, (unsigned long)32, f);
            fclose(f);
            if (got == (unsigned long)32) {      // a short read would seed from stack garbage
                self.init_seed(seed);
                return 1;
            }
        }
    }
#endif

    // No real entropy source was available (no OS CSPRNG, no rdrand, no
    // /dev/urandom — e.g. a freestanding target without either). Previously this silently
    // fell back to a fixed, fully-predictable seed and returned 0, trusting
    // every caller to check that return value before treating the "random"
    // stream as safe to use for keys/nonces — nothing enforced that, so a
    // caller that (reasonably) assumed init() always produces usable
    // randomness would get deterministic output with no warning. Aborting
    // here instead makes the failure impossible to miss. Callers that
    // genuinely have their own entropy source (e.g. a hardware TRNG driver
    // on a microcontroller with no rdrand/urandom) should call
    // init_seed() directly with that source instead of init().
    rng_abort_("RngCtx::init() found no entropy source (no OS random source, no rdrand, no /dev/urandom) — "
               "refusing to seed with a predictable fallback; call RngCtx::init_seed() "
               "with your own entropy source instead");
    return 0;
}

// ── RngCtx methods ─────────────────────────────────────────────────────────────

inline void RngCtx::fill_n_(&unsigned char out, unsigned long len) {
    unsigned long i = (unsigned long)0;
    while (i < len) {
        if (self.buf_pos >= (unsigned long)RNG_BLOCK_SIZE) {
            chacha20_block_(self.state, (unsigned char*)self.buf);
            self.state[12] = self.state[12] + (unsigned int)1;
            self.buf_pos = (unsigned long)0;
        }
        unsafe { out[i] = self.buf[self.buf_pos]; }
        self.buf_pos = self.buf_pos + (unsigned long)1;
        i = i + (unsigned long)1;
    }
}

inline void RngCtx::fill([]unsigned char out_s) {
    unsigned long len = out_s.len;
    &unsigned char out = crypto_out_(out_s, 0UL);
    self.fill_n_(out, len);
}

inline unsigned int RngCtx::rand32() {
    unsigned char b[4];
    self.fill_n_(&b[0], 4UL);
    unsafe {
        return ((unsigned int)b[0])
             | ((unsigned int)b[1] << 8)
             | ((unsigned int)b[2] << 16)
             | ((unsigned int)b[3] << 24);
    }
}

inline unsigned long long RngCtx::rand64() {
    unsigned long long lo = (unsigned long long)self.rand32();
    unsigned long long hi = (unsigned long long)self.rand32();
    return (hi << 32ULL) | lo;
}

inline unsigned long RngCtx::rand_range(unsigned long bound) {
    if (bound <= (unsigned long)1) { return (unsigned long)0; }
    // Rejection sampling to eliminate bias
    unsigned long threshold = (~bound + (unsigned long)1) % bound;
    unsafe {
        while (1) {
            unsigned long r = (unsigned long)self.rand64();   // 32 bits on 32-bit targets: threshold matches
            if (r >= threshold) { return r % bound; }
        }
    }
    return (unsigned long)0;
}

} // namespace std
