// SafeC Standard Library — convolution core, written once for both precisions.
//
// Internal: included by convolution.sc (double) and convolution_f32.sc
// (float), each its own translation unit. The includer defines CV_R (the
// real type), CV_C / CV_CNEW (its complex type and constructor), CV_DOT,
// CV_SDOT, CV_FFT / CV_IFFT / CV_FFT_BATCH (that precision's dsp_dot,
// dsp_sliding_dot and fft
// functions) and CV_DIRECT / CV_FFTCONV (the public names to define).
#pragma once

// Direct convolution's inner sum is sum_k x[k] * h[n-k]: x is walked
// forward, h is walked in reverse, so it isn't literally a dot product of
// two contiguous ranges as written. Reversing h once up front (hrev[i] =
// h[len_h-1-i]) turns h[n-k] into hrev[(len_h-1-n)+k] — now increasing
// with k, same direction as x[k] — so the whole inner loop becomes one
// contiguous-range call into dsp_dot_f64's SIMD FMA accumulation instead
// of a scalar multiply-add per sample.
// Long convolutions split the outputs across the thread pool.
struct ConvDirectJob {
    const CV_R* x;
    const CV_R* hrev;
    CV_R*       out;
    unsigned long lenX;
    unsigned long lenH;
};

static void conv_direct_outs_(&ConvDirectJob j, unsigned long lo, unsigned long hi) {
    // Outputs whose window lies wholly inside x (lenH - 1 <= n < lenX) are
    // one sliding dot product: out[n] = dot(x[n - lenH + 1 ..], hrev).
    unsigned long i0 = j.lenH - 1UL;
    unsigned long i1 = j.lenX;
    if (i0 < lo) i0 = lo;
    if (i1 > hi) i1 = hi;
    if (i0 < i1) {
        unsafe { CV_SDOT(j.hrev, &j.x[i0 + 1UL - j.lenH], &j.out[i0], i1 - i0, j.lenH); }
    } else {
        i0 = hi;
        i1 = hi;
    }
    unsigned long n = lo;
    while (n < hi) {
        if (n == i0) { n = i1; continue; }
        unsigned long kStart = n >= j.lenH ? n - j.lenH + 1UL : 0UL;
        unsigned long kEnd = n >= j.lenX ? j.lenX - 1UL : n;
        unsigned long segLen = kEnd - kStart + 1UL;
        unsigned long base = j.lenH - 1UL + kStart - n;
        unsafe { j.out[n] = CV_DOT(&j.x[kStart], &j.hrev[base], segLen); }
        n = n + 1UL;
    }
}

void CV_DIRECT(const &CV_R x, unsigned long len_x,
                  const &CV_R h, unsigned long len_h, &CV_R out) {
    if (len_x == 0UL || len_h == 0UL) {
        return;
    }
    CV_R* hrev;
    unsafe { hrev = (CV_R*)alloc_raw_(len_h * (unsigned long)sizeof(CV_R)); }
    if (hrev == (CV_R*)0) return;
    unsigned long r = 0UL;
    while (r < len_h) {
        unsafe { hrev[r] = h[len_h - 1UL - r]; }
        r = r + 1UL;
    }
    struct ConvDirectJob j;
    unsafe {
        j.x = (const CV_R*)x;
        j.out = (CV_R*)out;
    }
    j.hrev = (const CV_R*)hrev;
    j.lenX = len_x;
    j.lenH = len_h;
    unsigned long outLen = len_x + len_h - 1UL;
    if (outLen * len_h >= 1048576UL) parallel_for(outLen, 4096UL, conv_direct_outs_, &j);
    else conv_direct_outs_(&j, 0UL, outLen);
    unsafe { dealloc_raw_((void*)hrev); }
}

// Two strategies, both on fft_batch / fft (threaded, vendor FFT if linked):
//  - kernel much shorter than the signal: overlap-add. The kernel's
//    spectrum is computed once at a block size N of about 8x the kernel;
//    signal blocks of N - len_h + 1 samples go two to a complex FFT (one in
//    the real part, one in the imaginary part: H is a real signal's
//    spectrum, so the two results come back apart), in batches through
//    fft_batch, multiplied on the pool, transformed back and added in.
//  - otherwise one transform of the whole thing, with x and h packed into
//    one complex FFT (x + i h) and separated in the frequency domain:
//    two transforms instead of three.
#define CONV_OLA_BATCH 64UL

static unsigned long conv_pow2_(unsigned long n) {
    unsigned long p = 1UL;
    while (p < n) p = p * 2UL;
    return p;
}

// Pointwise multiply of every row of the batch by H, on the thread pool.
struct ConvMulJob { CV_C* rows; const CV_C* h; unsigned long n; };

static void conv_mul_body_(const &ConvMulJob j, unsigned long lo, unsigned long hi) __attribute__((always_inline)) {
    unsafe {
        unsigned long r = lo;
        while (r < hi) {
            CV_R* row = (CV_R*)&j.rows[r * j.n];
            const CV_R* hh = (const CV_R*)j.h;
            unsigned long q = 0UL;
            while (q < j.n) {
                CV_R ar = row[2UL * q]; CV_R ai = row[2UL * q + 1UL];
                CV_R br = hh[2UL * q];  CV_R bi = hh[2UL * q + 1UL];
                row[2UL * q] = __builtin_fmuladd(ar, br, -ai * bi);
                row[2UL * q + 1UL] = __builtin_fmuladd(ar, bi, ai * br);
                q = q + 1UL;
            }
            r = r + 1UL;
        }
    }
}

static void conv_mul_base_(const &ConvMulJob j, unsigned long lo, unsigned long hi) { conv_mul_body_(j, lo, hi); }
#if defined(__x86_64__)
static void conv_mul_fma_(const &ConvMulJob j, unsigned long lo, unsigned long hi)
    __attribute__((target("avx2,fma"))) { conv_mul_body_(j, lo, hi); }
#endif

static void conv_mul_rows_(&ConvMulJob j, unsigned long lo, unsigned long hi) {
#if defined(__x86_64__)
    if (simd_level() >= SIMD_AVX2) { conv_mul_fma_(j, lo, hi); return; }
#endif
    conv_mul_base_(j, lo, hi);
}

// Filling rows from the signal, and adding results back, on the pool. A
// pair's output overlaps only its neighbours', so even pairs and then odd
// pairs can be added concurrently.
struct ConvOlaJob {
    const CV_R* x;
    CV_R*       out;
    CV_R*       rows;      // batch of interleaved complex rows
    unsigned long lenX;
    unsigned long outLen;
    unsigned long N;
    unsigned long L;
    unsigned long p0;        // first pair in this batch
    unsigned long parity;    // accumulate: 0 = even rows, 1 = odd rows
};

static void conv_ola_fill_(&ConvOlaJob j, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long r = lo;
        while (r < hi) {
            unsigned long sa = 2UL * (j.p0 + r) * j.L;
            unsigned long sb = sa + j.L;
            CV_R* row = &j.rows[2UL * r * j.N];
            unsigned long q = 0UL;
            while (q < j.N) {
                row[2UL * q] = q < j.L && sa + q < j.lenX ? j.x[sa + q] : (CV_R)0.0;
                row[2UL * q + 1UL] = q < j.L && sb + q < j.lenX ? j.x[sb + q] : (CV_R)0.0;
                q = q + 1UL;
            }
            r = r + 1UL;
        }
    }
}

static void conv_ola_add_(&ConvOlaJob j, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long k = lo;
        while (k < hi) {
            unsigned long r = 2UL * k + j.parity;
            unsigned long sa = 2UL * (j.p0 + r) * j.L;
            unsigned long sb = sa + j.L;
            const CV_R* row = (const CV_R*)&j.rows[2UL * r * j.N];
            unsigned long q = 0UL;
            while (q < j.N && sa + q < j.outLen) { j.out[sa + q] = j.out[sa + q] + row[2UL * q]; q = q + 1UL; }
            q = 0UL;
            while (q < j.N && sb + q < j.outLen) { j.out[sb + q] = j.out[sb + q] + row[2UL * q + 1UL]; q = q + 1UL; }
            k = k + 1UL;
        }
    }
}

static void conv_fft_ola_(const CV_R* x, unsigned long lenX, const CV_R* h, unsigned long lenH, CV_R* out) {
    unsigned long N = conv_pow2_(8UL * lenH);
    if (N < 1024UL) N = 1024UL;
    unsigned long L = N - lenH + 1UL;
    unsigned long blocks = (lenX + L - 1UL) / L;
    unsigned long pairs = (blocks + 1UL) / 2UL;    // two real blocks per complex FFT
    unsigned long outLen = lenX + lenH - 1UL;
    unsafe {
        CV_C* H = (CV_C*)alloc_raw_(N * (unsigned long)sizeof(CV_C));
        unsigned long batch = pairs < CONV_OLA_BATCH ? pairs : CONV_OLA_BATCH;
        CV_C* B = (CV_C*)alloc_raw_(batch * N * (unsigned long)sizeof(CV_C));
        if (H == (CV_C*)0 || B == (CV_C*)0) {
            dealloc_raw_((void*)H); dealloc_raw_((void*)B);
            return;
        }
        unsigned long i = 0UL;
        while (i < N) { H[i] = CV_CNEW(i < lenH ? h[i] : (CV_R)0.0, (CV_R)0.0); i = i + 1UL; }
        CV_FFT(H, N);
        i = 0UL;
        while (i < outLen) { out[i] = (CV_R)0.0; i = i + 1UL; }
        unsigned long p0 = 0UL;
        while (p0 < pairs) {
            unsigned long np = pairs - p0 < batch ? pairs - p0 : batch;
            // row r: block 2(p0+r) in the real part, the next block in the imaginary part
            struct ConvOlaJob oj;
            oj.x = x;
            oj.out = out;
            oj.rows = (CV_R*)B;
            oj.lenX = lenX;
            oj.outLen = outLen;
            oj.N = N;
            oj.L = L;
            oj.p0 = p0;
            oj.parity = 0UL;
            if (np > 1UL) parallel_for(np, 1UL, conv_ola_fill_, &oj);
            else conv_ola_fill_(&oj, 0UL, np);
            CV_FFT_BATCH(B, N, np, 0);
            struct ConvMulJob mj;
            mj.rows = B;
            mj.h = (const CV_C*)H;
            mj.n = N;
            if (np > 1UL) parallel_for(np, 1UL, conv_mul_rows_, &mj);
            else conv_mul_rows_(&mj, 0UL, np);
            CV_FFT_BATCH(B, N, np, 1);
            unsigned long par = 0UL;
            while (par < 2UL) {
                oj.parity = par;
                unsigned long cnt = np > par ? (np - par + 1UL) / 2UL : 0UL;
                if (cnt > 1UL) parallel_for(cnt, 1UL, conv_ola_add_, &oj);
                else conv_ola_add_(&oj, 0UL, cnt);
                par = par + 1UL;
            }
            p0 = p0 + np;
        }
        dealloc_raw_((void*)H);
        dealloc_raw_((void*)B);
    }
}

static void conv_fft_packed_(const CV_R* x, unsigned long lenX, const CV_R* h, unsigned long lenH, CV_R* out) {
    unsigned long outLen = lenX + lenH - 1UL;
    unsigned long N = conv_pow2_(outLen);
    unsafe {
        CV_C* Z = (CV_C*)alloc_raw_(N * (unsigned long)sizeof(CV_C));
        if (Z == (CV_C*)0) return;
        unsigned long i = 0UL;
        while (i < N) { Z[i] = CV_CNEW(i < lenX ? x[i] : (CV_R)0.0, i < lenH ? h[i] : (CV_R)0.0); i = i + 1UL; }
        CV_FFT(Z, N);
        // X[k] = (Z[k] + conj Z[-k]) / 2, H[k] = (Z[k] - conj Z[-k]) / 2i,
        // so X[k] H[k] = (Z[k]^2 - conj(Z[-k])^2) / 4i. Pairs (k, N-k) are
        // updated together (both products come from the same two values).
        unsigned long k = 0UL;
        while (k <= N / 2UL) {
            unsigned long m = k == 0UL ? 0UL : N - k;
            CV_C zk = Z[k];
            CV_C zm = Z[m];
            // p(k) = (zk^2 - conj(zm)^2) / (4i);  p(m) = (zm^2 - conj(zk)^2) / (4i)
            CV_R ar = zk.re * zk.re - zk.im * zk.im - (zm.re * zm.re - zm.im * zm.im);
            CV_R ai = (CV_R)2.0 * zk.re * zk.im + (CV_R)2.0 * zm.re * zm.im;
            CV_R br = zm.re * zm.re - zm.im * zm.im - (zk.re * zk.re - zk.im * zk.im);
            CV_R bi = (CV_R)2.0 * zm.re * zm.im + (CV_R)2.0 * zk.re * zk.im;
            // divide by 4i: (r + i im) / 4i = (im - i r) / 4
            Z[k] = CV_CNEW(ai * (CV_R)0.25, -ar * (CV_R)0.25);
            if (m != k) Z[m] = CV_CNEW(bi * (CV_R)0.25, -br * (CV_R)0.25);
            k = k + 1UL;
        }
        CV_IFFT(Z, N);
        i = 0UL;
        while (i < outLen) { out[i] = Z[i].re; i = i + 1UL; }
        dealloc_raw_((void*)Z);
    }
}

void CV_FFTCONV(const &CV_R x, unsigned long len_x,
              const &CV_R h, unsigned long len_h, &CV_R out) {
    if (len_x == 0UL || len_h == 0UL) {
        return;
    }
    unsafe {
        if (len_h * 16UL <= len_x) conv_fft_ola_((const CV_R*)x, len_x, (const CV_R*)h, len_h, (CV_R*)out);
        else conv_fft_packed_((const CV_R*)x, len_x, (const CV_R*)h, len_h, (CV_R*)out);
    }
}
