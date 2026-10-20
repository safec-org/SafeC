// SafeC Standard Library — FFT core, written once for both precisions.
//
// Internal: included by fft.sc (double, Complex) and fft_f32.sc (float,
// ComplexF), each its own translation unit, so every static here exists
// once per precision. The includer defines, before including:
//   FFT_R           the real type (double / float)
//   FFT_V2, FFT_V4, FFT_V8  2-, 4- and 8-lane vectors of it (one, two, four
//                   complex values)
//   FFT_SINGLE      0 for double, 1 for float
//   FFT_DFTI_CREATE oneMKL's descriptor constructor for that precision
// and includes the headers fft.sc lists. Not a public header.
#pragma once

static int dsp_log2_(unsigned long n) {
    int bits = 0;
    unsigned long v = n;
    while (v > 1UL) {
        v = v >> 1;
        bits = bits + 1;
    }
    return bits;
}

static unsigned long dsp_bit_reverse_(unsigned long x, int numBits) {
    unsigned long result = 0UL;
    int k = 0;
    unsigned long v = x;
    while (k < numBits) {
        result = (result << 1) | (v & 1UL);
        v = v >> 1;
        k = k + 1;
    }
    return result;
}
// ── Twiddle table ────────────────────────────────────────────────────────────
// One table serves every length: entry h + k (1 <= h < N a power of two,
// 0 <= k < h) is exp(-i*pi*k/h), the k-th twiddle of a length-2h stage,
// which doesn't depend on the transform length. So a table built for N
// covers every n <= N, and growing it only ever appends.
//
// It is shared by all threads. Readers take the published length and
// pointer without locking; a grow builds the new table aside, publishes the
// pointer, then the length, and never frees the old one (a reader may still
// be using it — the old tables add up to less than the newest). Call
// fft_prepare(n) up front so a real-time thread never builds or allocates.

#if defined(__SAFEC_FREESTANDING__) || defined(__wasm__)
#define FFT_HAS_THREADS 0
#else
#define FFT_HAS_THREADS 1
#endif

static long long g_fft_tw_ptr = 0LL;   // (FFT_R*) table, interleaved re/im per entry
static long long g_fft_tw_n = 0LL;     // the table covers lengths <= this
static int g_fft_tw_lock = 0;

// ── Compile-time twiddles ────────────────────────────────────────────────────
// cos/sin of 2*pi*m/n evaluated by the compiler: exact integer reduction to
// an octant, then Taylor series on [0, pi/4) (terms to x^29, error ~2 ulp).
consteval double fft_ce_series_(double x, int wantSin) {
    double x2 = x * x;
    double term = wantSin ? x : 1.0;
    double sum = term;
    int i = 0;
    while (i < 14) {
        double d = wantSin ? (double)((2 * i + 2) * (2 * i + 3)) : (double)((2 * i + 1) * (2 * i + 2));
        term = -term * x2 / d;
        sum = sum + term;
        i = i + 1;
    }
    return sum;
}

consteval double fft_ce_trig_(long long m, long long n, int wantSin) {
    long long r = m % n;
    if (r < 0LL) r = r + n;
    long long q8 = 8LL * r;
    long long oct = q8 / n;
    double x = 6.283185307179586476925286766559 * ((double)(q8 - oct * n) / (8.0 * (double)n));
    double cx = fft_ce_series_(x, 0);
    double sx = fft_ce_series_(x, 1);
    double h = 0.70710678118654752440084436210485;
    double c = 0.0;
    double sn = 0.0;
    if (oct == 0LL) { c = cx; sn = sx; }
    else if (oct == 1LL) { c = h * (cx - sx); sn = h * (cx + sx); }
    else if (oct == 2LL) { c = -sx; sn = cx; }
    else if (oct == 3LL) { c = -h * (cx + sx); sn = h * (cx - sx); }
    else if (oct == 4LL) { c = -cx; sn = -sx; }
    else if (oct == 5LL) { c = -h * (cx - sx); sn = -h * (cx + sx); }
    else if (oct == 6LL) { c = sx; sn = -cx; }
    else { c = h * (cx + sx); sn = -h * (cx - sx); }
    return wantSin ? sn : c;
}

// The twiddle table for every length up to FFT_CONST_N, in the same layout
// as the runtime one (entry h + k = exp(-i pi k / h)), computed at compile
// time: transforms up to this length need no table build, allocation or
// lock, so fft_serial is real-time safe there without fft_prepare.
#if defined(__SAFEC_FREESTANDING__)
#define FFT_CONST_N 512
#else
#define FFT_CONST_N 4096
#endif

struct FftConstTable { FFT_R v[2 * FFT_CONST_N]; };

consteval struct FftConstTable fft_ce_table_() {
    struct FftConstTable t;
    t.v[0] = (FFT_R)1.0;
    t.v[1] = (FFT_R)0.0;
    long long h = 1LL;
    while (h < (long long)FFT_CONST_N) {
        long long k = 0LL;
        while (k < h) {
            t.v[2LL * (h + k)] = (FFT_R)fft_ce_trig_(k, 2LL * h, 0);
            t.v[2LL * (h + k) + 1LL] = (FFT_R)-fft_ce_trig_(k, 2LL * h, 1);
            k = k + 1LL;
        }
        h = h * 2LL;
    }
    return t;
}

static const struct FftConstTable g_fft_const_tw = fft_ce_table_();

static unsigned long fft_tw_len_() {
#if FFT_HAS_THREADS
    return (unsigned long)atomic_load_ll(&g_fft_tw_n);
#else
    return (unsigned long)g_fft_tw_n;
#endif
}

static FFT_R* fft_tw_ptr_() {
#if FFT_HAS_THREADS
    long long p = atomic_load_ll(&g_fft_tw_ptr);
#else
    long long p = g_fft_tw_ptr;
#endif
    unsafe { return (FFT_R*)p; }
}

static void fft_tw_build_(FFT_R* t, unsigned long N) {
    unsafe {
        unsigned long h = N / 2UL;
        // Top level, h = N/2: exact cos/sin every 32 entries, complex
        // rotation by exp(-2*pi*i/N) in between (error stays ~1e-15).
        // (computed in double for either precision, stored as FFT_R)
        double sr = cos_d(-SAFEC_DSP_TWO_PI / (double)N);
        double si = sin_d(-SAFEC_DSP_TWO_PI / (double)N);
        unsigned long k0 = 0UL;
        while (k0 < h) {
            double th = -SAFEC_DSP_TWO_PI * (double)k0 / (double)N;
            double wr = cos_d(th);
            double wi = sin_d(th);
            unsigned long k = k0;
            unsigned long end = k0 + 32UL < h ? k0 + 32UL : h;
            while (k < end) {
                t[2UL * (h + k)] = (FFT_R)wr;
                t[2UL * (h + k) + 1UL] = (FFT_R)wi;
                double nr = wr * sr - wi * si;
                wi = wr * si + wi * sr;
                wr = nr;
                k = k + 1UL;
            }
            k0 = end;
        }
        // Lower levels are every other entry of the level above.
        h = h / 2UL;
        while (h >= 1UL) {
            unsigned long k = 0UL;
            while (k < h) {
                t[2UL * (h + k)] = t[2UL * (2UL * h + 2UL * k)];
                t[2UL * (h + k) + 1UL] = t[2UL * (2UL * h + 2UL * k) + 1UL];
                k = k + 1UL;
            }
            h = h / 2UL;
        }
    }
}

// Twiddles covering length n (a power of two), or null if out of memory.
static const FFT_R* fft_twiddles_(unsigned long n) {
    if (n <= (unsigned long)FFT_CONST_N) { unsafe { return (const FFT_R*)&g_fft_const_tw.v[0]; } }
    if (fft_tw_len_() >= n) return (const FFT_R*)fft_tw_ptr_();
#if FFT_HAS_THREADS
    while (atomic_exchange_int(&g_fft_tw_lock, 1) != 0) { thread_yield(); }
#endif
    const FFT_R* r = (const FFT_R*)0;
    if (fft_tw_len_() >= n) {
        r = (const FFT_R*)fft_tw_ptr_();
    } else {
        unsigned long N = n < 64UL ? 64UL : n;
        unsafe {
            FFT_R* t = (FFT_R*)alloc_raw_(sizeof(FFT_R) * 2UL * N);
            if (t != (FFT_R*)0) {
                fft_tw_build_(t, N);
#if FFT_HAS_THREADS
                atomic_store_ll(&g_fft_tw_ptr, (long long)t);
                atomic_store_ll(&g_fft_tw_n, (long long)N);
#else
                g_fft_tw_ptr = (long long)t;
                g_fft_tw_n = (long long)N;
#endif
                r = (const FFT_R*)t;
            }
        }
    }
#if FFT_HAS_THREADS
    atomic_store_int(&g_fft_tw_lock, 0);
#endif
    return r;
}
// ── Radix-2 core on interleaved (re, im) doubles ─────────────────────────────

// Stages up to this length run block by block, so a block (2^11 complex =
// 32 KB) stays in L1/L2 through all of them.
#define FFT_BLOCK   2048UL
// Transforms up to this length run as one block (64 KB in double: L2), so
// every stage is in the radix-4 pairing and there is no separate outer pass.
#define FFT_ONE_BLOCK 4096UL
// Single transforms at least this long use the thread pool.
#define FFT_PAR_MIN 32768UL

// Compile-time bit-reversal swap lists for every length up to FFT_CONST_N:
// for n = 2^L, the pairs (i, rev i) with i < rev i, in increasing i, at
// p[2 * off[L] ..]. Swapping from a list skips the per-element index walk
// and the unpredictable i < j branch of the loop below.
#define FFT_REV_LOGS 14
struct FftRevTable { unsigned short p[2 * FFT_CONST_N]; int off[FFT_REV_LOGS]; };

consteval struct FftRevTable fft_ce_rev_() {
    struct FftRevTable t;
    int c = 0;
    int L = 0;
    while (L < FFT_REV_LOGS) {
        t.off[L] = c;
        long long n = 1LL << L;
        if (L >= 1 && n <= (long long)FFT_CONST_N) {
            long long i = 0LL;
            long long r = 0LL;   // rev(i), stepped like fft_bitrev_'s j
            while (i < n) {
                if (i < r) {
                    t.p[2 * c] = (unsigned short)i;
                    t.p[2 * c + 1] = (unsigned short)r;
                    c = c + 1;
                }
                long long m = n >> 1;
                while (m != 0LL && (r & m) != 0LL) { r = r ^ m; m = m >> 1; }
                r = r | m;
                i = i + 1LL;
            }
        }
        L = L + 1;
    }
    return t;
}

static const struct FftRevTable g_fft_const_rev = fft_ce_rev_();

static void fft_bitrev_table_(FFT_R* d, int L) {
    unsafe {
        const unsigned short* p = (const unsigned short*)&g_fft_const_rev.p[2 * g_fft_const_rev.off[L]];
        int cnt = g_fft_const_rev.off[L + 1] - g_fft_const_rev.off[L];
        int c = 0;
        while (c < cnt) {
            unsigned long i = (unsigned long)p[2 * c];
            unsigned long j = (unsigned long)p[2 * c + 1];
            FFT_V2 a = *(FFT_V2*)&d[2UL * i];
            *(FFT_V2*)&d[2UL * i] = *(FFT_V2*)&d[2UL * j];
            *(FFT_V2*)&d[2UL * j] = a;
            c = c + 1;
        }
    }
}

// Bit-reversal swaps for i in [lo, hi). Each pair (i, rev i) is swapped by
// the range holding the smaller index, so disjoint ranges never touch the
// same element.
static void fft_bitrev_(FFT_R* d, unsigned long n, unsigned long lo, unsigned long hi) {
    int bits = dsp_log2_(n);
    if (lo == 0UL && hi == n && n <= (unsigned long)FFT_CONST_N && bits + 1 < FFT_REV_LOGS) {
        fft_bitrev_table_(d, bits);
        return;
    }
    unsigned long j = dsp_bit_reverse_(lo, bits);
    unsigned long i = lo;
    unsafe {
        while (i < hi) {
            if (i < j) {
                FFT_R tr = d[2UL * i];
                FFT_R ti = d[2UL * i + 1UL];
                d[2UL * i] = d[2UL * j];
                d[2UL * i + 1UL] = d[2UL * j + 1UL];
                d[2UL * j] = tr;
                d[2UL * j + 1UL] = ti;
            }
            // j = rev(i + 1): add one at the top bit, carrying downwards.
            unsigned long m = n >> 1;
            while (m != 0UL && (j & m) != 0UL) { j = j ^ m; m = m >> 1; }
            j = j | m;
            i = i + 1UL;
        }
    }
}

// Butterflies g in [g0, g1) of the length-len stage (butterfly g pairs
// element (g / half) * len + g % half with the one 'half' after it).
static void fft_pass_body_(FFT_R* d, const FFT_R* t, unsigned long len, unsigned long g0, unsigned long g1, int inv)
    __attribute__((always_inline)) {
    unsigned long half = len / 2UL;
    unsafe {
        if (half == 1UL) {
            unsigned long g = g0;
            while (g < g1) {
                FFT_R* a = &d[4UL * g];
                FFT_R ur = a[0]; FFT_R ui = a[1];
                FFT_R vr = a[2]; FFT_R vi = a[3];
                a[0] = ur + vr; a[1] = ui + vi;
                a[2] = ur - vr; a[3] = ui - vi;
                g = g + 1UL;
            }
            return;
        }
        FFT_R sgn = inv ? (FFT_R)-1.0 : (FFT_R)1.0;
        if (half == 2UL) {
            // len 4: twiddles 1 and -i (forward) or +i (inverse), no multiplies
            unsigned long g = g0;
            while (g < g1) {
                unsigned long blk = g / 2UL;
                unsigned long k = g % 2UL;
                FFT_R* a = &d[8UL * blk + 2UL * k];
                FFT_R ur = a[0]; FFT_R ui = a[1];
                FFT_R br = a[4]; FFT_R bi = a[5];
                FFT_R vr = br;
                FFT_R vi = bi;
                if (k == 1UL) { vr = sgn * bi; vi = -sgn * br; }
                a[0] = ur + vr; a[1] = ui + vi;
                a[4] = ur - vr; a[5] = ui - vi;
                g = g + 1UL;
            }
            return;
        }
        unsigned long g = g0;
        while (g < g1) {
            unsigned long blk = g / half;
            unsigned long k = g % half;
            unsigned long kend = half - k < g1 - g ? half : k + (g1 - g);
            FFT_R* a = &d[2UL * (blk * len)];
            FFT_R* b = &a[2UL * half];
            const FFT_R* w = &t[2UL * half];
            unsigned long run = kend - k;
            // Two complex values per FFT_V4 while at least two remain (one
            // ymm register from AVX on): v = b * w as
            // (br, bi) * (wr, wr) + (bi, br) * (-wi, wi), per pair.
            while (k + 4UL <= kend) {
                FFT_V8 v = fft_cmul8_(*(FFT_V8*)&b[2UL * k], *(const FFT_V8*)&w[2UL * k], sgn);
                FFT_V8 u = *(FFT_V8*)&a[2UL * k];
                *(FFT_V8*)&a[2UL * k] = u + v;
                *(FFT_V8*)&b[2UL * k] = u - v;
                k = k + 4UL;
            }
            while (k + 2UL <= kend) {
                FFT_V4 bv = *(FFT_V4*)&b[2UL * k];
                FFT_V4 bs;
                bs[0] = bv[1]; bs[1] = bv[0]; bs[2] = bv[3]; bs[3] = bv[2];
                FFT_V4 wv = *(const FFT_V4*)&w[2UL * k];   // wr0 wi0 wr1 wi1
                FFT_V4 vr;
                vr[0] = wv[0]; vr[1] = wv[0]; vr[2] = wv[2]; vr[3] = wv[2];
                FFT_V4 vi;
                vi[0] = -sgn * wv[1]; vi[1] = sgn * wv[1]; vi[2] = -sgn * wv[3]; vi[3] = sgn * wv[3];
                FFT_V4 v = __builtin_fmuladd(bs, vi, bv * vr);
                FFT_V4 u = *(FFT_V4*)&a[2UL * k];
                *(FFT_V4*)&a[2UL * k] = u + v;
                *(FFT_V4*)&b[2UL * k] = u - v;
                k = k + 2UL;
            }
            while (k < kend) {
                // One complex value per FFT_V2: v = b * w as
                // (br, bi) * wr + (bi, br) * (-wi, wi).
                FFT_V2 bv = *(FFT_V2*)&b[2UL * k];
                FFT_V2 bs;
                bs[0] = bv[1];
                bs[1] = bv[0];
                FFT_R wr = w[2UL * k];
                FFT_R wi = sgn * w[2UL * k + 1UL];
                FFT_V2 vr; vr[0] = wr; vr[1] = wr;
                FFT_V2 vi; vi[0] = -wi; vi[1] = wi;
                FFT_V2 v = __builtin_fmuladd(bs, vi, bv * vr);
                FFT_V2 u = *(FFT_V2*)&a[2UL * k];
                *(FFT_V2*)&a[2UL * k] = u + v;
                *(FFT_V2*)&b[2UL * k] = u - v;
                k = k + 1UL;
            }
            g = g + run;
        }
    }
}

// Radix-4: the radix-2 stages of length 2m and 4m fused, so each group of
// four values is loaded and stored once instead of twice. For q in
// [q0, q1) (n/4 groups per stage pair), group q is block q / m, k = q % m:
//   a0..a3 = d[base + k + {0, m, 2m, 3m}]
//   stage 2m: b0, b1 = a0 +- a1 W2m^k;  b2, b3 = a2 +- a3 W2m^k
//   stage 4m: c0, c2 = b0 +- b2 W4m^k;  c1, c3 = b1 +- b3 W4m^k (-i)
// (+i for the inverse). Twiddle W_{2h}^k is table entry h + k.
static FFT_V4 fft_cmul4_(FFT_V4 b, FFT_V4 w, FFT_R sgn) __attribute__((always_inline)) {
    FFT_V4 bs;
    bs[0] = b[1]; bs[1] = b[0]; bs[2] = b[3]; bs[3] = b[2];
    FFT_V4 wr;
    wr[0] = w[0]; wr[1] = w[0]; wr[2] = w[2]; wr[3] = w[2];
    FFT_V4 wi;
    wi[0] = -sgn * w[1]; wi[1] = sgn * w[1]; wi[2] = -sgn * w[3]; wi[3] = sgn * w[3];
    return __builtin_fmuladd(bs, wi, b * wr);
}

// The same on four complex values (FFT_V8: a zmm of doubles, a ymm of floats).
static FFT_V8 fft_cmul8_(FFT_V8 b, FFT_V8 w, FFT_R sgn) __attribute__((always_inline)) {
    FFT_V8 bs;
    bs[0] = b[1]; bs[1] = b[0]; bs[2] = b[3]; bs[3] = b[2];
    bs[4] = b[5]; bs[5] = b[4]; bs[6] = b[7]; bs[7] = b[6];
    FFT_V8 wr;
    wr[0] = w[0]; wr[1] = w[0]; wr[2] = w[2]; wr[3] = w[2];
    wr[4] = w[4]; wr[5] = w[4]; wr[6] = w[6]; wr[7] = w[6];
    FFT_V8 wi;
    wi[0] = -sgn * w[1]; wi[1] = sgn * w[1]; wi[2] = -sgn * w[3]; wi[3] = sgn * w[3];
    wi[4] = -sgn * w[5]; wi[5] = sgn * w[5]; wi[6] = -sgn * w[7]; wi[7] = sgn * w[7];
    return __builtin_fmuladd(bs, wi, b * wr);
}

static FFT_V8 fft_rot8_(FFT_V8 x, FFT_R sgn) __attribute__((always_inline)) {
    FFT_V8 r;
    r[0] = sgn * x[1]; r[1] = -sgn * x[0]; r[2] = sgn * x[3]; r[3] = -sgn * x[2];
    r[4] = sgn * x[5]; r[5] = -sgn * x[4]; r[6] = sgn * x[7]; r[7] = -sgn * x[6];
    return r;
}

// x * (-i) forward, x * (+i) inverse, on two complex values.
static FFT_V4 fft_rot4_(FFT_V4 x, FFT_R sgn) __attribute__((always_inline)) {
    FFT_V4 r;
    r[0] = sgn * x[1]; r[1] = -sgn * x[0]; r[2] = sgn * x[3]; r[3] = -sgn * x[2];
    return r;
}

static void fft_pass4_body_(FFT_R* d, const FFT_R* t, unsigned long m, unsigned long q0, unsigned long q1, int inv)
    __attribute__((always_inline)) {
    FFT_R sgn = inv ? (FFT_R)-1.0 : (FFT_R)1.0;
    unsafe {
        if (m == 1UL) {
            // stages 2 and 4: twiddles 1, 1 and -i
            unsigned long q = q0;
            while (q < q1) {
                FFT_R* a = &d[8UL * q];
                FFT_R a0r = a[0]; FFT_R a0i = a[1]; FFT_R a1r = a[2]; FFT_R a1i = a[3];
                FFT_R a2r = a[4]; FFT_R a2i = a[5]; FFT_R a3r = a[6]; FFT_R a3i = a[7];
                FFT_R b0r = a0r + a1r; FFT_R b0i = a0i + a1i;
                FFT_R b1r = a0r - a1r; FFT_R b1i = a0i - a1i;
                FFT_R b2r = a2r + a3r; FFT_R b2i = a2i + a3i;
                FFT_R b3r = a2r - a3r; FFT_R b3i = a2i - a3i;
                FFT_R vr = sgn * b3i; FFT_R vi = -sgn * b3r;     // b3 * (-i)
                a[0] = b0r + b2r; a[1] = b0i + b2i;
                a[4] = b0r - b2r; a[5] = b0i - b2i;
                a[2] = b1r + vr;  a[3] = b1i + vi;
                a[6] = b1r - vr;  a[7] = b1i - vi;
                q = q + 1UL;
            }
            return;
        }
        const FFT_R* w2 = &t[2UL * m];         // W_2m^k
        const FFT_R* w4 = &t[4UL * m];         // W_4m^k
        unsigned long q = q0;
        while (q < q1) {
            unsigned long blk = q / m;
            unsigned long k = q % m;
            unsigned long kend = m - k < q1 - q ? m : k + (q1 - q);
            unsigned long run = kend - k;
            FFT_R* p0 = &d[2UL * (blk * 4UL * m)];
            FFT_R* p1 = &p0[2UL * m];
            FFT_R* p2 = &p0[4UL * m];
            FFT_R* p3 = &p0[6UL * m];
            while (k + 4UL <= kend) {
                FFT_V8 tw2 = *(const FFT_V8*)&w2[2UL * k];
                FFT_V8 tw4 = *(const FFT_V8*)&w4[2UL * k];
                FFT_V8 a0 = *(FFT_V8*)&p0[2UL * k];
                FFT_V8 a1 = fft_cmul8_(*(FFT_V8*)&p1[2UL * k], tw2, sgn);
                FFT_V8 a2 = *(FFT_V8*)&p2[2UL * k];
                FFT_V8 a3 = fft_cmul8_(*(FFT_V8*)&p3[2UL * k], tw2, sgn);
                FFT_V8 b0 = a0 + a1; FFT_V8 b1 = a0 - a1;
                FFT_V8 b2 = fft_cmul8_(a2 + a3, tw4, sgn);
                FFT_V8 b3 = fft_rot8_(fft_cmul8_(a2 - a3, tw4, sgn), sgn);
                *(FFT_V8*)&p0[2UL * k] = b0 + b2;
                *(FFT_V8*)&p2[2UL * k] = b0 - b2;
                *(FFT_V8*)&p1[2UL * k] = b1 + b3;
                *(FFT_V8*)&p3[2UL * k] = b1 - b3;
                k = k + 4UL;
            }
            while (k + 2UL <= kend) {
                FFT_V4 tw2 = *(const FFT_V4*)&w2[2UL * k];
                FFT_V4 tw4 = *(const FFT_V4*)&w4[2UL * k];
                FFT_V4 a0 = *(FFT_V4*)&p0[2UL * k];
                FFT_V4 a1 = fft_cmul4_(*(FFT_V4*)&p1[2UL * k], tw2, sgn);
                FFT_V4 a2 = *(FFT_V4*)&p2[2UL * k];
                FFT_V4 a3 = fft_cmul4_(*(FFT_V4*)&p3[2UL * k], tw2, sgn);
                FFT_V4 b0 = a0 + a1; FFT_V4 b1 = a0 - a1;
                FFT_V4 b2 = fft_cmul4_(a2 + a3, tw4, sgn);
                FFT_V4 b3 = fft_rot4_(fft_cmul4_(a2 - a3, tw4, sgn), sgn);
                *(FFT_V4*)&p0[2UL * k] = b0 + b2;
                *(FFT_V4*)&p2[2UL * k] = b0 - b2;
                *(FFT_V4*)&p1[2UL * k] = b1 + b3;
                *(FFT_V4*)&p3[2UL * k] = b1 - b3;
                k = k + 2UL;
            }
            while (k < kend) {
                // one complex value: same math, scalar
                FFT_R w2r = w2[2UL * k]; FFT_R w2i = sgn * w2[2UL * k + 1UL];
                FFT_R w4r = w4[2UL * k]; FFT_R w4i = sgn * w4[2UL * k + 1UL];
                FFT_R a0r = p0[2UL * k]; FFT_R a0i = p0[2UL * k + 1UL];
                FFT_R xr = p1[2UL * k]; FFT_R xi = p1[2UL * k + 1UL];
                FFT_R a1r = xr * w2r - xi * w2i; FFT_R a1i = xr * w2i + xi * w2r;
                FFT_R a2r = p2[2UL * k]; FFT_R a2i = p2[2UL * k + 1UL];
                xr = p3[2UL * k]; xi = p3[2UL * k + 1UL];
                FFT_R a3r = xr * w2r - xi * w2i; FFT_R a3i = xr * w2i + xi * w2r;
                FFT_R b0r = a0r + a1r; FFT_R b0i = a0i + a1i;
                FFT_R b1r = a0r - a1r; FFT_R b1i = a0i - a1i;
                FFT_R sr = a2r + a3r; FFT_R si = a2i + a3i;
                FFT_R dr = a2r - a3r; FFT_R di = a2i - a3i;
                FFT_R b2r = sr * w4r - si * w4i; FFT_R b2i = sr * w4i + si * w4r;
                FFT_R er = dr * w4r - di * w4i; FFT_R ei = dr * w4i + di * w4r;
                FFT_R b3r = sgn * ei; FFT_R b3i = -sgn * er;
                p0[2UL * k] = b0r + b2r; p0[2UL * k + 1UL] = b0i + b2i;
                p2[2UL * k] = b0r - b2r; p2[2UL * k + 1UL] = b0i - b2i;
                p1[2UL * k] = b1r + b3r; p1[2UL * k + 1UL] = b1i + b3i;
                p3[2UL * k] = b1r - b3r; p3[2UL * k + 1UL] = b1i - b3i;
                k = k + 1UL;
            }
            q = q + run;
        }
    }
}

// Radix-8: the radix-2 stages of length 2m, 4m and 8m fused (m >= 4), so
// each group of eight values is loaded and stored once instead of three
// times (radix-8 passes as in SPIRAL/FFTE's 8^q 16^r factorizations).
// For q in [q0, q1) (n/8 groups per pass), block q / m, k = q % m:
//   a_r = d[base + k + r m], r = 0..7
//   stage 2m: (a0,a1) (a2,a3) (a4,a5) (a6,a7)        twiddle W2m^k
//   stage 4m: (b0,b2) (b1,b3) (b4,b6) (b5,b7)        W4m^k, W4m^k * (-i)
//   stage 8m: (c0,c4) (c1,c5) (c2,c6) (c3,c7)        W8m^k * W8^r
// with W8^1 = (1 - i)/sqrt2, W8^2 = -i, W8^3 = -(1 + i)/sqrt2 (conjugated
// for the inverse). 'wide' (a constant per ISA build) picks four complex
// values per vector (FFT_V8) or two (FFT_V4): two keep the 8 inputs of a
// double AVX2 build in 16 registers.
#define FFT_SQRT1_2 0.70710678118654752440

static FFT_V2 fft_cmul2_(FFT_V2 b, FFT_R wr, FFT_R wi) __attribute__((always_inline)) {
    FFT_V2 bs; bs[0] = b[1]; bs[1] = b[0];
    FFT_V2 vr; vr[0] = wr; vr[1] = wr;
    FFT_V2 vi; vi[0] = -wi; vi[1] = wi;
    return __builtin_fmuladd(bs, vi, b * vr);
}

static FFT_V2 fft_rot2_(FFT_V2 x, FFT_R sgn) __attribute__((always_inline)) {
    FFT_V2 r; r[0] = sgn * x[1]; r[1] = -sgn * x[0];
    return r;
}

static void fft_pass8_body_(FFT_R* d, const FFT_R* t, unsigned long m, unsigned long q0, unsigned long q1, int inv, int wide)
    __attribute__((always_inline)) {
    FFT_R sgn = inv ? (FFT_R)-1.0 : (FFT_R)1.0;
    FFT_R h = (FFT_R)FFT_SQRT1_2;
    unsafe {
        const FFT_R* w2 = &t[2UL * m];   // W_2m^k
        const FFT_R* w4 = &t[4UL * m];   // W_4m^k
        const FFT_R* w8 = &t[8UL * m];   // W_8m^k
        FFT_V8 h8; h8[0] = h; h8[1] = h; h8[2] = h; h8[3] = h; h8[4] = h; h8[5] = h; h8[6] = h; h8[7] = h;
        FFT_V4 h4; h4[0] = h; h4[1] = h; h4[2] = h; h4[3] = h;
        FFT_V2 h2; h2[0] = h; h2[1] = h;
        unsigned long q = q0;
        while (q < q1) {
            unsigned long blk = q / m;
            unsigned long k = q % m;
            unsigned long kend = m - k < q1 - q ? m : k + (q1 - q);
            unsigned long run = kend - k;
            FFT_R* p = &d[2UL * (blk * 8UL * m)];
            unsigned long s = 2UL * m;           // element stride r -> r + 1, in reals
            if (wide) {
                while (k + 4UL <= kend) {
                    unsigned long o = 2UL * k;
                    FFT_V8 tw2 = *(const FFT_V8*)&w2[o];
                    FFT_V8 tw4 = *(const FFT_V8*)&w4[o];
                    FFT_V8 tw8 = *(const FFT_V8*)&w8[o];
                    FFT_V8 a0 = *(FFT_V8*)&p[o];
                    FFT_V8 a1 = fft_cmul8_(*(FFT_V8*)&p[o + s], tw2, sgn);
                    FFT_V8 a2 = *(FFT_V8*)&p[o + 2UL * s];
                    FFT_V8 a3 = fft_cmul8_(*(FFT_V8*)&p[o + 3UL * s], tw2, sgn);
                    FFT_V8 a4 = *(FFT_V8*)&p[o + 4UL * s];
                    FFT_V8 a5 = fft_cmul8_(*(FFT_V8*)&p[o + 5UL * s], tw2, sgn);
                    FFT_V8 a6 = *(FFT_V8*)&p[o + 6UL * s];
                    FFT_V8 a7 = fft_cmul8_(*(FFT_V8*)&p[o + 7UL * s], tw2, sgn);
                    FFT_V8 b0 = a0 + a1; FFT_V8 b1 = a0 - a1;
                    FFT_V8 b2 = fft_cmul8_(a2 + a3, tw4, sgn); FFT_V8 b3 = fft_rot8_(fft_cmul8_(a2 - a3, tw4, sgn), sgn);
                    FFT_V8 b4 = a4 + a5; FFT_V8 b5 = a4 - a5;
                    FFT_V8 b6 = fft_cmul8_(a6 + a7, tw4, sgn); FFT_V8 b7 = fft_rot8_(fft_cmul8_(a6 - a7, tw4, sgn), sgn);
                    FFT_V8 c0 = b0 + b2; FFT_V8 c2 = b0 - b2;
                    FFT_V8 c1 = b1 + b3; FFT_V8 c3 = b1 - b3;
                    FFT_V8 c4 = fft_cmul8_(b4 + b6, tw8, sgn);
                    FFT_V8 c6 = fft_rot8_(fft_cmul8_(b4 - b6, tw8, sgn), sgn);              // * W8^2
                    FFT_V8 e5 = fft_cmul8_(b5 + b7, tw8, sgn);
                    FFT_V8 c5 = (e5 + fft_rot8_(e5, sgn)) * h8;                              // * W8^1
                    FFT_V8 e7 = fft_cmul8_(b5 - b7, tw8, sgn);
                    FFT_V8 c7 = (fft_rot8_(e7, sgn) - e7) * h8;                              // * W8^3
                    *(FFT_V8*)&p[o] = c0 + c4;
                    *(FFT_V8*)&p[o + 4UL * s] = c0 - c4;
                    *(FFT_V8*)&p[o + s] = c1 + c5;
                    *(FFT_V8*)&p[o + 5UL * s] = c1 - c5;
                    *(FFT_V8*)&p[o + 2UL * s] = c2 + c6;
                    *(FFT_V8*)&p[o + 6UL * s] = c2 - c6;
                    *(FFT_V8*)&p[o + 3UL * s] = c3 + c7;
                    *(FFT_V8*)&p[o + 7UL * s] = c3 - c7;
                    k = k + 4UL;
                }
            }
            while (k + 2UL <= kend) {
                unsigned long o = 2UL * k;
                FFT_V4 tw2 = *(const FFT_V4*)&w2[o];
                FFT_V4 tw4 = *(const FFT_V4*)&w4[o];
                FFT_V4 tw8 = *(const FFT_V4*)&w8[o];
                FFT_V4 a0 = *(FFT_V4*)&p[o];
                FFT_V4 a1 = fft_cmul4_(*(FFT_V4*)&p[o + s], tw2, sgn);
                FFT_V4 a2 = *(FFT_V4*)&p[o + 2UL * s];
                FFT_V4 a3 = fft_cmul4_(*(FFT_V4*)&p[o + 3UL * s], tw2, sgn);
                FFT_V4 a4 = *(FFT_V4*)&p[o + 4UL * s];
                FFT_V4 a5 = fft_cmul4_(*(FFT_V4*)&p[o + 5UL * s], tw2, sgn);
                FFT_V4 a6 = *(FFT_V4*)&p[o + 6UL * s];
                FFT_V4 a7 = fft_cmul4_(*(FFT_V4*)&p[o + 7UL * s], tw2, sgn);
                FFT_V4 b0 = a0 + a1; FFT_V4 b1 = a0 - a1;
                FFT_V4 b2 = fft_cmul4_(a2 + a3, tw4, sgn); FFT_V4 b3 = fft_rot4_(fft_cmul4_(a2 - a3, tw4, sgn), sgn);
                FFT_V4 b4 = a4 + a5; FFT_V4 b5 = a4 - a5;
                FFT_V4 b6 = fft_cmul4_(a6 + a7, tw4, sgn); FFT_V4 b7 = fft_rot4_(fft_cmul4_(a6 - a7, tw4, sgn), sgn);
                FFT_V4 c0 = b0 + b2; FFT_V4 c2 = b0 - b2;
                FFT_V4 c1 = b1 + b3; FFT_V4 c3 = b1 - b3;
                FFT_V4 c4 = fft_cmul4_(b4 + b6, tw8, sgn);
                FFT_V4 c6 = fft_rot4_(fft_cmul4_(b4 - b6, tw8, sgn), sgn);
                FFT_V4 e5 = fft_cmul4_(b5 + b7, tw8, sgn);
                FFT_V4 c5 = (e5 + fft_rot4_(e5, sgn)) * h4;
                FFT_V4 e7 = fft_cmul4_(b5 - b7, tw8, sgn);
                FFT_V4 c7 = (fft_rot4_(e7, sgn) - e7) * h4;
                *(FFT_V4*)&p[o] = c0 + c4;
                *(FFT_V4*)&p[o + 4UL * s] = c0 - c4;
                *(FFT_V4*)&p[o + s] = c1 + c5;
                *(FFT_V4*)&p[o + 5UL * s] = c1 - c5;
                *(FFT_V4*)&p[o + 2UL * s] = c2 + c6;
                *(FFT_V4*)&p[o + 6UL * s] = c2 - c6;
                *(FFT_V4*)&p[o + 3UL * s] = c3 + c7;
                *(FFT_V4*)&p[o + 7UL * s] = c3 - c7;
                k = k + 2UL;
            }
            while (k < kend) {
                unsigned long o = 2UL * k;
                FFT_R w2r = w2[o]; FFT_R w2i = sgn * w2[o + 1UL];
                FFT_R w4r = w4[o]; FFT_R w4i = sgn * w4[o + 1UL];
                FFT_R w8r = w8[o]; FFT_R w8i = sgn * w8[o + 1UL];
                FFT_V2 a0 = *(FFT_V2*)&p[o];
                FFT_V2 a1 = fft_cmul2_(*(FFT_V2*)&p[o + s], w2r, w2i);
                FFT_V2 a2 = *(FFT_V2*)&p[o + 2UL * s];
                FFT_V2 a3 = fft_cmul2_(*(FFT_V2*)&p[o + 3UL * s], w2r, w2i);
                FFT_V2 a4 = *(FFT_V2*)&p[o + 4UL * s];
                FFT_V2 a5 = fft_cmul2_(*(FFT_V2*)&p[o + 5UL * s], w2r, w2i);
                FFT_V2 a6 = *(FFT_V2*)&p[o + 6UL * s];
                FFT_V2 a7 = fft_cmul2_(*(FFT_V2*)&p[o + 7UL * s], w2r, w2i);
                FFT_V2 b0 = a0 + a1; FFT_V2 b1 = a0 - a1;
                FFT_V2 b2 = fft_cmul2_(a2 + a3, w4r, w4i); FFT_V2 b3 = fft_rot2_(fft_cmul2_(a2 - a3, w4r, w4i), sgn);
                FFT_V2 b4 = a4 + a5; FFT_V2 b5 = a4 - a5;
                FFT_V2 b6 = fft_cmul2_(a6 + a7, w4r, w4i); FFT_V2 b7 = fft_rot2_(fft_cmul2_(a6 - a7, w4r, w4i), sgn);
                FFT_V2 c0 = b0 + b2; FFT_V2 c2 = b0 - b2;
                FFT_V2 c1 = b1 + b3; FFT_V2 c3 = b1 - b3;
                FFT_V2 c4 = fft_cmul2_(b4 + b6, w8r, w8i);
                FFT_V2 c6 = fft_rot2_(fft_cmul2_(b4 - b6, w8r, w8i), sgn);
                FFT_V2 e5 = fft_cmul2_(b5 + b7, w8r, w8i);
                FFT_V2 c5 = (e5 + fft_rot2_(e5, sgn)) * h2;
                FFT_V2 e7 = fft_cmul2_(b5 - b7, w8r, w8i);
                FFT_V2 c7 = (fft_rot2_(e7, sgn) - e7) * h2;
                *(FFT_V2*)&p[o] = c0 + c4;
                *(FFT_V2*)&p[o + 4UL * s] = c0 - c4;
                *(FFT_V2*)&p[o + s] = c1 + c5;
                *(FFT_V2*)&p[o + 5UL * s] = c1 - c5;
                *(FFT_V2*)&p[o + 2UL * s] = c2 + c6;
                *(FFT_V2*)&p[o + 6UL * s] = c2 - c6;
                *(FFT_V2*)&p[o + 3UL * s] = c3 + c7;
                *(FFT_V2*)&p[o + 7UL * s] = c3 - c7;
                k = k + 1UL;
            }
            q = q + run;
        }
    }
}

static void fft_pass8_base_(FFT_R* d, const FFT_R* t, unsigned long m, unsigned long q0, unsigned long q1, int inv) {
    fft_pass8_body_(d, t, m, q0, q1, inv, 0);
}
#if defined(__x86_64__)
static void fft_pass8_avx2_(FFT_R* d, const FFT_R* t, unsigned long m, unsigned long q0, unsigned long q1, int inv)
    __attribute__((target("avx2,fma"))) {
    fft_pass8_body_(d, t, m, q0, q1, inv, FFT_SINGLE);
}
static void fft_pass8_avx512_(FFT_R* d, const FFT_R* t, unsigned long m, unsigned long q0, unsigned long q1, int inv)
    __attribute__((target("avx512f,avx512vl,avx2,fma"))) {
    fft_pass8_body_(d, t, m, q0, q1, inv, 1);
}
#endif

static void fft_pass8_(FFT_R* d, const FFT_R* t, unsigned long m, unsigned long q0, unsigned long q1, int inv) {
#if defined(__x86_64__)
    int l = simd_level();
    if (l >= SIMD_AVX512) { fft_pass8_avx512_(d, t, m, q0, q1, inv); return; }
    if (l >= SIMD_AVX2) { fft_pass8_avx2_(d, t, m, q0, q1, inv); return; }
#endif
    fft_pass8_base_(d, t, m, q0, q1, inv);
}

static void fft_pass4_base_(FFT_R* d, const FFT_R* t, unsigned long m, unsigned long q0, unsigned long q1, int inv) {
    fft_pass4_body_(d, t, m, q0, q1, inv);
}
#if defined(__x86_64__)
static void fft_pass4_avx2_(FFT_R* d, const FFT_R* t, unsigned long m, unsigned long q0, unsigned long q1, int inv)
    __attribute__((target("avx2,fma"))) {
    fft_pass4_body_(d, t, m, q0, q1, inv);
}
static void fft_pass4_avx512_(FFT_R* d, const FFT_R* t, unsigned long m, unsigned long q0, unsigned long q1, int inv)
    __attribute__((target("avx512f,avx512vl,avx2,fma"))) {
    fft_pass4_body_(d, t, m, q0, q1, inv);
}
#endif

static void fft_pass4_(FFT_R* d, const FFT_R* t, unsigned long m, unsigned long q0, unsigned long q1, int inv) {
#if defined(__x86_64__)
    int l = simd_level();
    if (l >= SIMD_AVX512) { fft_pass4_avx512_(d, t, m, q0, q1, inv); return; }
    if (l >= SIMD_AVX2) { fft_pass4_avx2_(d, t, m, q0, q1, inv); return; }
#endif
    fft_pass4_base_(d, t, m, q0, q1, inv);
}

// The butterflies per ISA level (std/simd/cpu.h): AVX2 adds FMA to the
// complex multiply; the base build is SSE2 / Neon.
static void fft_pass_base_(FFT_R* d, const FFT_R* t, unsigned long len, unsigned long g0, unsigned long g1, int inv) {
    fft_pass_body_(d, t, len, g0, g1, inv);
}
#if defined(__x86_64__)
static void fft_pass_avx2_(FFT_R* d, const FFT_R* t, unsigned long len, unsigned long g0, unsigned long g1, int inv)
    __attribute__((target("avx2,fma"))) {
    fft_pass_body_(d, t, len, g0, g1, inv);
}
static void fft_pass_avx512_(FFT_R* d, const FFT_R* t, unsigned long len, unsigned long g0, unsigned long g1, int inv)
    __attribute__((target("avx512f,avx512vl,avx2,fma"))) {
    fft_pass_body_(d, t, len, g0, g1, inv);
}
#endif

static void fft_pass_(FFT_R* d, const FFT_R* t, unsigned long len, unsigned long g0, unsigned long g1, int inv) {
#if defined(__x86_64__)
    int l = simd_level();
    if (l >= SIMD_AVX512) { fft_pass_avx512_(d, t, len, g0, g1, inv); return; }
    if (l >= SIMD_AVX2) { fft_pass_avx2_(d, t, len, g0, g1, inv); return; }
#endif
    fft_pass_base_(d, t, len, g0, g1, inv);
}

// ── Vendor FFT (oneMKL DFTI) ─────────────────────────────────────────────────
// On hosted targets (not Windows yet), a program that links oneMKL (the
// Intel choice for [build] blas, see std/simd/gemm.h) gets its DFTI
// transforms for fft/ifft/fft_real below FFT_VENDOR_SINGLE_MAX points, for
// every row of fft_batch / fft_real_batch and for fft2d's rows and
// columns. The references are weak: without oneMKL they are null and the
// built-in transform runs, and which one is used is fixed at link time.
// (FFTW-based libraries, AOCL-FFTW included, are GPL and not used.) One
// committed descriptor per length is made on first use, under a lock, and
// then computed concurrently from the thread pool, single-threaded per call.
// fft_serial and ifft_serial never use it: creating descriptors allocates
// and locks. Not on Apple targets: oneMKL has no Apple Silicon build, and
// Apple's linker requires even weak imports to resolve at link time, so
// the references would break every macOS program that reaches fft.sc.
#if FFT_HAS_THREADS && !defined(_WIN32) && !defined(__APPLE__)
#define FFT_VENDOR 1
extern long FFT_DFTI_CREATE(void** desc, int domain, long length) __attribute__((weak));
extern long DftiSetValue(void* desc, int param, ...) __attribute__((weak));
extern long DftiCommitDescriptor(void* desc) __attribute__((weak));
extern long DftiComputeForward(void* desc, void* x, ...) __attribute__((weak));
extern long DftiComputeBackward(void* desc, void* x, ...) __attribute__((weak));
extern long DftiFreeDescriptor(void** desc) __attribute__((weak));
#else
#define FFT_VENDOR 0
#endif

#define DFTI_BACKWARD_SCALE_          5
#define DFTI_CONJUGATE_EVEN_STORAGE_  10
#define DFTI_PLACEMENT_               11
#define DFTI_THREAD_LIMIT_            27
#define DFTI_COMPLEX_                 32
#define DFTI_REAL_                    33
#define DFTI_COMPLEX_COMPLEX_         39
#define DFTI_NOT_INPLACE_             44
#define FFT_PLANS                     48
// fft()/ifft() at or above this length use the built-in threaded transform
// instead of one single-threaded library call.
#define FFT_VENDOR_SINGLE_MAX         65536UL

#define FFT_KIND_C2C 0   // in place, forward and backward (backward scaled by 1/n)
#define FFT_KIND_R2C 1   // out of place, n/2 + 1 complex outputs

static int g_fftv_lock = 0;
static int g_fftv_count = 0;                     // published entries (atomic)
static int g_fftv_kind[FFT_PLANS];
static unsigned long g_fftv_n[FFT_PLANS];
static long long g_fftv_desc[FFT_PLANS];

static int fft_vendor_linked_() {
#if FFT_VENDOR
    unsafe {
        return (void*)FFT_DFTI_CREATE != (void*)0 && (void*)DftiSetValue != (void*)0 &&
               (void*)DftiCommitDescriptor != (void*)0 && (void*)DftiComputeForward != (void*)0 &&
               (void*)DftiComputeBackward != (void*)0 && (void*)DftiFreeDescriptor != (void*)0;
    }
#else
    return 0;
#endif
}


#if FFT_VENDOR
static void* fft_vendor_make_(int kind, unsigned long n) {
    void* d = (void*)0;
    unsafe {
        if (FFT_DFTI_CREATE(&d, kind == FFT_KIND_R2C ? DFTI_REAL_ : DFTI_COMPLEX_, (long)n) != 0L)
            return (void*)0;
        long ok = DftiSetValue(d, DFTI_THREAD_LIMIT_, 1L);
        if (kind == FFT_KIND_R2C) {
            ok = ok | DftiSetValue(d, DFTI_PLACEMENT_, (long)DFTI_NOT_INPLACE_);
            ok = ok | DftiSetValue(d, DFTI_CONJUGATE_EVEN_STORAGE_, (long)DFTI_COMPLEX_COMPLEX_);
        } else if (FFT_SINGLE == 0) {
            // (single precision scales after the call instead: the scale's
            // type follows the precision, and varargs can't pass a float)
            ok = ok | DftiSetValue(d, DFTI_BACKWARD_SCALE_, 1.0 / (double)n);
        }
        if (ok != 0L || DftiCommitDescriptor(d) != 0L) {
            DftiFreeDescriptor(&d);
            return (void*)0;
        }
    }
    return d;
}
#endif

// Cached committed descriptor for (kind, n), made on first use; null if
// unavailable (or the cache is full).
static void* fft_vendor_desc_(int kind, unsigned long n) {
#if FFT_VENDOR
    if (n > 2147483647UL) return (void*)0;
    int c = atomic_load_int(&g_fftv_count);
    int i = 0;
    while (i < c) {
        if (g_fftv_kind[i] == kind && g_fftv_n[i] == n) { unsafe { return (void*)g_fftv_desc[i]; } }
        i = i + 1;
    }
    while (atomic_exchange_int(&g_fftv_lock, 1) != 0) { thread_yield(); }
    void* d = (void*)0;
    c = atomic_load_int(&g_fftv_count);
    i = 0;
    while (i < c) {
        if (g_fftv_kind[i] == kind && g_fftv_n[i] == n) { unsafe { d = (void*)g_fftv_desc[i]; } }
        i = i + 1;
    }
    if (d == (void*)0 && c < FFT_PLANS) {
        d = fft_vendor_make_(kind, n);
        if (d != (void*)0) {
            g_fftv_kind[c] = kind;
            g_fftv_n[c] = n;
            unsafe { g_fftv_desc[c] = (long long)d; }
            atomic_store_int(&g_fftv_count, c + 1);   // publish after the entry is written
        }
    }
    atomic_store_int(&g_fftv_lock, 0);
    return d;
#else
    return (void*)0;
#endif
}

// In-place complex transform through the library; 0 if it can't (then the
// caller runs the built-in one). The inverse includes 1/n, like ifft.
static int fft_vendor_c2c_(FFT_R* d, unsigned long n, int inv) {
#if FFT_VENDOR
    if (!fft_vendor_linked_()) return 0;
    void* desc = fft_vendor_desc_(FFT_KIND_C2C, n);
    if (desc == (void*)0) return 0;
    unsafe {
        long r = inv ? DftiComputeBackward(desc, (void*)d) : DftiComputeForward(desc, (void*)d);
        if (r != 0L) return 0;
        if (FFT_SINGLE != 0 && inv) {
            FFT_R sc = (FFT_R)(1.0 / (double)n);
            unsigned long i = 0UL;
            while (i < 2UL * n) { d[i] = d[i] * sc; i = i + 1UL; }
        }
        return 1;
    }
#else
    return 0;
#endif
}

// Real transform through the library into the full n-point spectrum.
static int fft_vendor_r2c_(const FFT_R* in, FFT_R* out, unsigned long n, int full) {
#if FFT_VENDOR
    if (!fft_vendor_linked_() || n < 2UL) return 0;
    void* desc = fft_vendor_desc_(FFT_KIND_R2C, n);
    if (desc == (void*)0) return 0;
    unsafe {
        if (DftiComputeForward(desc, (void*)in, (void*)out) != 0L) return 0;   // bins 0 .. n/2
        unsigned long k = 1UL;
        while (full && k < n / 2UL) {
            out[2UL * (n - k)] = out[2UL * k];
            out[2UL * (n - k) + 1UL] = -out[2UL * k + 1UL];
            k = k + 1UL;
        }
    }
    return 1;
#else
    return 0;
#endif
}

// fft_run_'s 'mode': who may run the transform.
#define FFT_MODE_RT   0   // built-in, calling thread only (fft_serial)
#define FFT_MODE_TOP  1   // a top-level call: library or built-in threads
#define FFT_MODE_ROW  2   // one row of a batch already on the pool: library, no threads

struct FftJob {
    FFT_R*       d;
    const FFT_R* t;
    unsigned long n;
    unsigned long len;     // stage, for fft_stage_job_
    unsigned long block;   // block size, for fft_block_job_
    int           inv;
    FFT_R        scale;   // for fft_scale_job_
};

static void fft_bitrev_job_(&FftJob j, unsigned long lo, unsigned long hi) { fft_bitrev_(j.d, j.n, lo, hi); }

// All stages of length <= j.block within blocks [lo, hi).
// 16-point codelet: the first four radix-2 stages (lengths 2, 4, 8, 16) of
// a bit-reversed block, entirely in registers -- one load and one store per
// value instead of four passes. The x2 form keeps one complex value per
// FFT_V2; the x4 form runs two adjacent blocks side by side, the same element
// of each in one FFT_V4 (one ymm from AVX on), sharing the twiddles.
// Twiddles are entries of the compile-time table at constant indices, so
// they fold into the code. Generated, not hand-written: the straight-line
// form is what makes it fast.
static FFT_V2 fft_cl_swap_(FFT_V2 v) __attribute__((always_inline)) { FFT_V2 s; s[0] = v[1]; s[1] = v[0]; return s; }
static FFT_V4 fft_cl_swap4_(FFT_V4 v) __attribute__((always_inline)) {
    FFT_V4 s; s[0] = v[1]; s[1] = v[0]; s[2] = v[3]; s[3] = v[2]; return s;
}

static void fft_codelet16_body_(FFT_R* a, FFT_R sgn) __attribute__((always_inline)) {
    unsafe {
        const FFT_R* t = (const FFT_R*)&g_fft_const_tw.v[0];
        FFT_V2 rot; rot[0] = sgn; rot[1] = -sgn;
        FFT_V2 v0 = *(FFT_V2*)&a[0];
        FFT_V2 v1 = *(FFT_V2*)&a[2];
        FFT_V2 v2 = *(FFT_V2*)&a[4];
        FFT_V2 v3 = *(FFT_V2*)&a[6];
        FFT_V2 v4 = *(FFT_V2*)&a[8];
        FFT_V2 v5 = *(FFT_V2*)&a[10];
        FFT_V2 v6 = *(FFT_V2*)&a[12];
        FFT_V2 v7 = *(FFT_V2*)&a[14];
        FFT_V2 v8 = *(FFT_V2*)&a[16];
        FFT_V2 v9 = *(FFT_V2*)&a[18];
        FFT_V2 v10 = *(FFT_V2*)&a[20];
        FFT_V2 v11 = *(FFT_V2*)&a[22];
        FFT_V2 v12 = *(FFT_V2*)&a[24];
        FFT_V2 v13 = *(FFT_V2*)&a[26];
        FFT_V2 v14 = *(FFT_V2*)&a[28];
        FFT_V2 v15 = *(FFT_V2*)&a[30];
        { FFT_V2 m = v1;
          v1 = v0 - m; v0 = v0 + m; }
        { FFT_V2 m = v3;
          v3 = v2 - m; v2 = v2 + m; }
        { FFT_V2 m = v5;
          v5 = v4 - m; v4 = v4 + m; }
        { FFT_V2 m = v7;
          v7 = v6 - m; v6 = v6 + m; }
        { FFT_V2 m = v9;
          v9 = v8 - m; v8 = v8 + m; }
        { FFT_V2 m = v11;
          v11 = v10 - m; v10 = v10 + m; }
        { FFT_V2 m = v13;
          v13 = v12 - m; v12 = v12 + m; }
        { FFT_V2 m = v15;
          v15 = v14 - m; v14 = v14 + m; }
        { FFT_V2 m = v2;
          v2 = v0 - m; v0 = v0 + m; }
        { FFT_V2 m = fft_cl_swap_(v3) * rot;
          v3 = v1 - m; v1 = v1 + m; }
        { FFT_V2 m = v6;
          v6 = v4 - m; v4 = v4 + m; }
        { FFT_V2 m = fft_cl_swap_(v7) * rot;
          v7 = v5 - m; v5 = v5 + m; }
        { FFT_V2 m = v10;
          v10 = v8 - m; v8 = v8 + m; }
        { FFT_V2 m = fft_cl_swap_(v11) * rot;
          v11 = v9 - m; v9 = v9 + m; }
        { FFT_V2 m = v14;
          v14 = v12 - m; v12 = v12 + m; }
        { FFT_V2 m = fft_cl_swap_(v15) * rot;
          v15 = v13 - m; v13 = v13 + m; }
        { FFT_V2 m = v4;
          v4 = v0 - m; v0 = v0 + m; }
        { FFT_V2 wr; wr[0] = t[10]; wr[1] = t[10];
          FFT_V2 wi; wi[0] = -sgn * t[11]; wi[1] = sgn * t[11];
          FFT_V2 m = __builtin_fmuladd(fft_cl_swap_(v5), wi, v5 * wr);
          v5 = v1 - m; v1 = v1 + m; }
        { FFT_V2 wr; wr[0] = t[12]; wr[1] = t[12];
          FFT_V2 wi; wi[0] = -sgn * t[13]; wi[1] = sgn * t[13];
          FFT_V2 m = __builtin_fmuladd(fft_cl_swap_(v6), wi, v6 * wr);
          v6 = v2 - m; v2 = v2 + m; }
        { FFT_V2 wr; wr[0] = t[14]; wr[1] = t[14];
          FFT_V2 wi; wi[0] = -sgn * t[15]; wi[1] = sgn * t[15];
          FFT_V2 m = __builtin_fmuladd(fft_cl_swap_(v7), wi, v7 * wr);
          v7 = v3 - m; v3 = v3 + m; }
        { FFT_V2 m = v12;
          v12 = v8 - m; v8 = v8 + m; }
        { FFT_V2 wr; wr[0] = t[10]; wr[1] = t[10];
          FFT_V2 wi; wi[0] = -sgn * t[11]; wi[1] = sgn * t[11];
          FFT_V2 m = __builtin_fmuladd(fft_cl_swap_(v13), wi, v13 * wr);
          v13 = v9 - m; v9 = v9 + m; }
        { FFT_V2 wr; wr[0] = t[12]; wr[1] = t[12];
          FFT_V2 wi; wi[0] = -sgn * t[13]; wi[1] = sgn * t[13];
          FFT_V2 m = __builtin_fmuladd(fft_cl_swap_(v14), wi, v14 * wr);
          v14 = v10 - m; v10 = v10 + m; }
        { FFT_V2 wr; wr[0] = t[14]; wr[1] = t[14];
          FFT_V2 wi; wi[0] = -sgn * t[15]; wi[1] = sgn * t[15];
          FFT_V2 m = __builtin_fmuladd(fft_cl_swap_(v15), wi, v15 * wr);
          v15 = v11 - m; v11 = v11 + m; }
        { FFT_V2 m = v8;
          v8 = v0 - m; v0 = v0 + m; }
        { FFT_V2 wr; wr[0] = t[18]; wr[1] = t[18];
          FFT_V2 wi; wi[0] = -sgn * t[19]; wi[1] = sgn * t[19];
          FFT_V2 m = __builtin_fmuladd(fft_cl_swap_(v9), wi, v9 * wr);
          v9 = v1 - m; v1 = v1 + m; }
        { FFT_V2 wr; wr[0] = t[20]; wr[1] = t[20];
          FFT_V2 wi; wi[0] = -sgn * t[21]; wi[1] = sgn * t[21];
          FFT_V2 m = __builtin_fmuladd(fft_cl_swap_(v10), wi, v10 * wr);
          v10 = v2 - m; v2 = v2 + m; }
        { FFT_V2 wr; wr[0] = t[22]; wr[1] = t[22];
          FFT_V2 wi; wi[0] = -sgn * t[23]; wi[1] = sgn * t[23];
          FFT_V2 m = __builtin_fmuladd(fft_cl_swap_(v11), wi, v11 * wr);
          v11 = v3 - m; v3 = v3 + m; }
        { FFT_V2 wr; wr[0] = t[24]; wr[1] = t[24];
          FFT_V2 wi; wi[0] = -sgn * t[25]; wi[1] = sgn * t[25];
          FFT_V2 m = __builtin_fmuladd(fft_cl_swap_(v12), wi, v12 * wr);
          v12 = v4 - m; v4 = v4 + m; }
        { FFT_V2 wr; wr[0] = t[26]; wr[1] = t[26];
          FFT_V2 wi; wi[0] = -sgn * t[27]; wi[1] = sgn * t[27];
          FFT_V2 m = __builtin_fmuladd(fft_cl_swap_(v13), wi, v13 * wr);
          v13 = v5 - m; v5 = v5 + m; }
        { FFT_V2 wr; wr[0] = t[28]; wr[1] = t[28];
          FFT_V2 wi; wi[0] = -sgn * t[29]; wi[1] = sgn * t[29];
          FFT_V2 m = __builtin_fmuladd(fft_cl_swap_(v14), wi, v14 * wr);
          v14 = v6 - m; v6 = v6 + m; }
        { FFT_V2 wr; wr[0] = t[30]; wr[1] = t[30];
          FFT_V2 wi; wi[0] = -sgn * t[31]; wi[1] = sgn * t[31];
          FFT_V2 m = __builtin_fmuladd(fft_cl_swap_(v15), wi, v15 * wr);
          v15 = v7 - m; v7 = v7 + m; }
        *(FFT_V2*)&a[0] = v0;
        *(FFT_V2*)&a[2] = v1;
        *(FFT_V2*)&a[4] = v2;
        *(FFT_V2*)&a[6] = v3;
        *(FFT_V2*)&a[8] = v4;
        *(FFT_V2*)&a[10] = v5;
        *(FFT_V2*)&a[12] = v6;
        *(FFT_V2*)&a[14] = v7;
        *(FFT_V2*)&a[16] = v8;
        *(FFT_V2*)&a[18] = v9;
        *(FFT_V2*)&a[20] = v10;
        *(FFT_V2*)&a[22] = v11;
        *(FFT_V2*)&a[24] = v12;
        *(FFT_V2*)&a[26] = v13;
        *(FFT_V2*)&a[28] = v14;
        *(FFT_V2*)&a[30] = v15;
    }
}

static void fft_codelet16x2_body_(FFT_R* a, FFT_R sgn) __attribute__((always_inline)) {
    unsafe {
        const FFT_R* t = (const FFT_R*)&g_fft_const_tw.v[0];
        FFT_V4 rot; rot[0] = sgn; rot[1] = -sgn; rot[2] = sgn; rot[3] = -sgn;
        FFT_V2 lo0 = *(FFT_V2*)&a[0]; FFT_V2 hi0 = *(FFT_V2*)&a[32];
        FFT_V4 v0; v0[0] = lo0[0]; v0[1] = lo0[1]; v0[2] = hi0[0]; v0[3] = hi0[1];
        FFT_V2 lo1 = *(FFT_V2*)&a[2]; FFT_V2 hi1 = *(FFT_V2*)&a[34];
        FFT_V4 v1; v1[0] = lo1[0]; v1[1] = lo1[1]; v1[2] = hi1[0]; v1[3] = hi1[1];
        FFT_V2 lo2 = *(FFT_V2*)&a[4]; FFT_V2 hi2 = *(FFT_V2*)&a[36];
        FFT_V4 v2; v2[0] = lo2[0]; v2[1] = lo2[1]; v2[2] = hi2[0]; v2[3] = hi2[1];
        FFT_V2 lo3 = *(FFT_V2*)&a[6]; FFT_V2 hi3 = *(FFT_V2*)&a[38];
        FFT_V4 v3; v3[0] = lo3[0]; v3[1] = lo3[1]; v3[2] = hi3[0]; v3[3] = hi3[1];
        FFT_V2 lo4 = *(FFT_V2*)&a[8]; FFT_V2 hi4 = *(FFT_V2*)&a[40];
        FFT_V4 v4; v4[0] = lo4[0]; v4[1] = lo4[1]; v4[2] = hi4[0]; v4[3] = hi4[1];
        FFT_V2 lo5 = *(FFT_V2*)&a[10]; FFT_V2 hi5 = *(FFT_V2*)&a[42];
        FFT_V4 v5; v5[0] = lo5[0]; v5[1] = lo5[1]; v5[2] = hi5[0]; v5[3] = hi5[1];
        FFT_V2 lo6 = *(FFT_V2*)&a[12]; FFT_V2 hi6 = *(FFT_V2*)&a[44];
        FFT_V4 v6; v6[0] = lo6[0]; v6[1] = lo6[1]; v6[2] = hi6[0]; v6[3] = hi6[1];
        FFT_V2 lo7 = *(FFT_V2*)&a[14]; FFT_V2 hi7 = *(FFT_V2*)&a[46];
        FFT_V4 v7; v7[0] = lo7[0]; v7[1] = lo7[1]; v7[2] = hi7[0]; v7[3] = hi7[1];
        FFT_V2 lo8 = *(FFT_V2*)&a[16]; FFT_V2 hi8 = *(FFT_V2*)&a[48];
        FFT_V4 v8; v8[0] = lo8[0]; v8[1] = lo8[1]; v8[2] = hi8[0]; v8[3] = hi8[1];
        FFT_V2 lo9 = *(FFT_V2*)&a[18]; FFT_V2 hi9 = *(FFT_V2*)&a[50];
        FFT_V4 v9; v9[0] = lo9[0]; v9[1] = lo9[1]; v9[2] = hi9[0]; v9[3] = hi9[1];
        FFT_V2 lo10 = *(FFT_V2*)&a[20]; FFT_V2 hi10 = *(FFT_V2*)&a[52];
        FFT_V4 v10; v10[0] = lo10[0]; v10[1] = lo10[1]; v10[2] = hi10[0]; v10[3] = hi10[1];
        FFT_V2 lo11 = *(FFT_V2*)&a[22]; FFT_V2 hi11 = *(FFT_V2*)&a[54];
        FFT_V4 v11; v11[0] = lo11[0]; v11[1] = lo11[1]; v11[2] = hi11[0]; v11[3] = hi11[1];
        FFT_V2 lo12 = *(FFT_V2*)&a[24]; FFT_V2 hi12 = *(FFT_V2*)&a[56];
        FFT_V4 v12; v12[0] = lo12[0]; v12[1] = lo12[1]; v12[2] = hi12[0]; v12[3] = hi12[1];
        FFT_V2 lo13 = *(FFT_V2*)&a[26]; FFT_V2 hi13 = *(FFT_V2*)&a[58];
        FFT_V4 v13; v13[0] = lo13[0]; v13[1] = lo13[1]; v13[2] = hi13[0]; v13[3] = hi13[1];
        FFT_V2 lo14 = *(FFT_V2*)&a[28]; FFT_V2 hi14 = *(FFT_V2*)&a[60];
        FFT_V4 v14; v14[0] = lo14[0]; v14[1] = lo14[1]; v14[2] = hi14[0]; v14[3] = hi14[1];
        FFT_V2 lo15 = *(FFT_V2*)&a[30]; FFT_V2 hi15 = *(FFT_V2*)&a[62];
        FFT_V4 v15; v15[0] = lo15[0]; v15[1] = lo15[1]; v15[2] = hi15[0]; v15[3] = hi15[1];
        { FFT_V4 m = v1;
          v1 = v0 - m; v0 = v0 + m; }
        { FFT_V4 m = v3;
          v3 = v2 - m; v2 = v2 + m; }
        { FFT_V4 m = v5;
          v5 = v4 - m; v4 = v4 + m; }
        { FFT_V4 m = v7;
          v7 = v6 - m; v6 = v6 + m; }
        { FFT_V4 m = v9;
          v9 = v8 - m; v8 = v8 + m; }
        { FFT_V4 m = v11;
          v11 = v10 - m; v10 = v10 + m; }
        { FFT_V4 m = v13;
          v13 = v12 - m; v12 = v12 + m; }
        { FFT_V4 m = v15;
          v15 = v14 - m; v14 = v14 + m; }
        { FFT_V4 m = v2;
          v2 = v0 - m; v0 = v0 + m; }
        { FFT_V4 m = fft_cl_swap4_(v3) * rot;
          v3 = v1 - m; v1 = v1 + m; }
        { FFT_V4 m = v6;
          v6 = v4 - m; v4 = v4 + m; }
        { FFT_V4 m = fft_cl_swap4_(v7) * rot;
          v7 = v5 - m; v5 = v5 + m; }
        { FFT_V4 m = v10;
          v10 = v8 - m; v8 = v8 + m; }
        { FFT_V4 m = fft_cl_swap4_(v11) * rot;
          v11 = v9 - m; v9 = v9 + m; }
        { FFT_V4 m = v14;
          v14 = v12 - m; v12 = v12 + m; }
        { FFT_V4 m = fft_cl_swap4_(v15) * rot;
          v15 = v13 - m; v13 = v13 + m; }
        { FFT_V4 m = v4;
          v4 = v0 - m; v0 = v0 + m; }
        { FFT_V4 wr; wr[0] = t[10]; wr[1] = t[10]; wr[2] = t[10]; wr[3] = t[10];
          FFT_V4 wi; wi[0] = -sgn * t[11]; wi[1] = sgn * t[11]; wi[2] = -sgn * t[11]; wi[3] = sgn * t[11];
          FFT_V4 m = __builtin_fmuladd(fft_cl_swap4_(v5), wi, v5 * wr);
          v5 = v1 - m; v1 = v1 + m; }
        { FFT_V4 wr; wr[0] = t[12]; wr[1] = t[12]; wr[2] = t[12]; wr[3] = t[12];
          FFT_V4 wi; wi[0] = -sgn * t[13]; wi[1] = sgn * t[13]; wi[2] = -sgn * t[13]; wi[3] = sgn * t[13];
          FFT_V4 m = __builtin_fmuladd(fft_cl_swap4_(v6), wi, v6 * wr);
          v6 = v2 - m; v2 = v2 + m; }
        { FFT_V4 wr; wr[0] = t[14]; wr[1] = t[14]; wr[2] = t[14]; wr[3] = t[14];
          FFT_V4 wi; wi[0] = -sgn * t[15]; wi[1] = sgn * t[15]; wi[2] = -sgn * t[15]; wi[3] = sgn * t[15];
          FFT_V4 m = __builtin_fmuladd(fft_cl_swap4_(v7), wi, v7 * wr);
          v7 = v3 - m; v3 = v3 + m; }
        { FFT_V4 m = v12;
          v12 = v8 - m; v8 = v8 + m; }
        { FFT_V4 wr; wr[0] = t[10]; wr[1] = t[10]; wr[2] = t[10]; wr[3] = t[10];
          FFT_V4 wi; wi[0] = -sgn * t[11]; wi[1] = sgn * t[11]; wi[2] = -sgn * t[11]; wi[3] = sgn * t[11];
          FFT_V4 m = __builtin_fmuladd(fft_cl_swap4_(v13), wi, v13 * wr);
          v13 = v9 - m; v9 = v9 + m; }
        { FFT_V4 wr; wr[0] = t[12]; wr[1] = t[12]; wr[2] = t[12]; wr[3] = t[12];
          FFT_V4 wi; wi[0] = -sgn * t[13]; wi[1] = sgn * t[13]; wi[2] = -sgn * t[13]; wi[3] = sgn * t[13];
          FFT_V4 m = __builtin_fmuladd(fft_cl_swap4_(v14), wi, v14 * wr);
          v14 = v10 - m; v10 = v10 + m; }
        { FFT_V4 wr; wr[0] = t[14]; wr[1] = t[14]; wr[2] = t[14]; wr[3] = t[14];
          FFT_V4 wi; wi[0] = -sgn * t[15]; wi[1] = sgn * t[15]; wi[2] = -sgn * t[15]; wi[3] = sgn * t[15];
          FFT_V4 m = __builtin_fmuladd(fft_cl_swap4_(v15), wi, v15 * wr);
          v15 = v11 - m; v11 = v11 + m; }
        { FFT_V4 m = v8;
          v8 = v0 - m; v0 = v0 + m; }
        { FFT_V4 wr; wr[0] = t[18]; wr[1] = t[18]; wr[2] = t[18]; wr[3] = t[18];
          FFT_V4 wi; wi[0] = -sgn * t[19]; wi[1] = sgn * t[19]; wi[2] = -sgn * t[19]; wi[3] = sgn * t[19];
          FFT_V4 m = __builtin_fmuladd(fft_cl_swap4_(v9), wi, v9 * wr);
          v9 = v1 - m; v1 = v1 + m; }
        { FFT_V4 wr; wr[0] = t[20]; wr[1] = t[20]; wr[2] = t[20]; wr[3] = t[20];
          FFT_V4 wi; wi[0] = -sgn * t[21]; wi[1] = sgn * t[21]; wi[2] = -sgn * t[21]; wi[3] = sgn * t[21];
          FFT_V4 m = __builtin_fmuladd(fft_cl_swap4_(v10), wi, v10 * wr);
          v10 = v2 - m; v2 = v2 + m; }
        { FFT_V4 wr; wr[0] = t[22]; wr[1] = t[22]; wr[2] = t[22]; wr[3] = t[22];
          FFT_V4 wi; wi[0] = -sgn * t[23]; wi[1] = sgn * t[23]; wi[2] = -sgn * t[23]; wi[3] = sgn * t[23];
          FFT_V4 m = __builtin_fmuladd(fft_cl_swap4_(v11), wi, v11 * wr);
          v11 = v3 - m; v3 = v3 + m; }
        { FFT_V4 wr; wr[0] = t[24]; wr[1] = t[24]; wr[2] = t[24]; wr[3] = t[24];
          FFT_V4 wi; wi[0] = -sgn * t[25]; wi[1] = sgn * t[25]; wi[2] = -sgn * t[25]; wi[3] = sgn * t[25];
          FFT_V4 m = __builtin_fmuladd(fft_cl_swap4_(v12), wi, v12 * wr);
          v12 = v4 - m; v4 = v4 + m; }
        { FFT_V4 wr; wr[0] = t[26]; wr[1] = t[26]; wr[2] = t[26]; wr[3] = t[26];
          FFT_V4 wi; wi[0] = -sgn * t[27]; wi[1] = sgn * t[27]; wi[2] = -sgn * t[27]; wi[3] = sgn * t[27];
          FFT_V4 m = __builtin_fmuladd(fft_cl_swap4_(v13), wi, v13 * wr);
          v13 = v5 - m; v5 = v5 + m; }
        { FFT_V4 wr; wr[0] = t[28]; wr[1] = t[28]; wr[2] = t[28]; wr[3] = t[28];
          FFT_V4 wi; wi[0] = -sgn * t[29]; wi[1] = sgn * t[29]; wi[2] = -sgn * t[29]; wi[3] = sgn * t[29];
          FFT_V4 m = __builtin_fmuladd(fft_cl_swap4_(v14), wi, v14 * wr);
          v14 = v6 - m; v6 = v6 + m; }
        { FFT_V4 wr; wr[0] = t[30]; wr[1] = t[30]; wr[2] = t[30]; wr[3] = t[30];
          FFT_V4 wi; wi[0] = -sgn * t[31]; wi[1] = sgn * t[31]; wi[2] = -sgn * t[31]; wi[3] = sgn * t[31];
          FFT_V4 m = __builtin_fmuladd(fft_cl_swap4_(v15), wi, v15 * wr);
          v15 = v7 - m; v7 = v7 + m; }
        FFT_V2 so0; so0[0] = v0[0]; so0[1] = v0[1]; *(FFT_V2*)&a[0] = so0;
        FFT_V2 sh0; sh0[0] = v0[2]; sh0[1] = v0[3]; *(FFT_V2*)&a[32] = sh0;
        FFT_V2 so1; so1[0] = v1[0]; so1[1] = v1[1]; *(FFT_V2*)&a[2] = so1;
        FFT_V2 sh1; sh1[0] = v1[2]; sh1[1] = v1[3]; *(FFT_V2*)&a[34] = sh1;
        FFT_V2 so2; so2[0] = v2[0]; so2[1] = v2[1]; *(FFT_V2*)&a[4] = so2;
        FFT_V2 sh2; sh2[0] = v2[2]; sh2[1] = v2[3]; *(FFT_V2*)&a[36] = sh2;
        FFT_V2 so3; so3[0] = v3[0]; so3[1] = v3[1]; *(FFT_V2*)&a[6] = so3;
        FFT_V2 sh3; sh3[0] = v3[2]; sh3[1] = v3[3]; *(FFT_V2*)&a[38] = sh3;
        FFT_V2 so4; so4[0] = v4[0]; so4[1] = v4[1]; *(FFT_V2*)&a[8] = so4;
        FFT_V2 sh4; sh4[0] = v4[2]; sh4[1] = v4[3]; *(FFT_V2*)&a[40] = sh4;
        FFT_V2 so5; so5[0] = v5[0]; so5[1] = v5[1]; *(FFT_V2*)&a[10] = so5;
        FFT_V2 sh5; sh5[0] = v5[2]; sh5[1] = v5[3]; *(FFT_V2*)&a[42] = sh5;
        FFT_V2 so6; so6[0] = v6[0]; so6[1] = v6[1]; *(FFT_V2*)&a[12] = so6;
        FFT_V2 sh6; sh6[0] = v6[2]; sh6[1] = v6[3]; *(FFT_V2*)&a[44] = sh6;
        FFT_V2 so7; so7[0] = v7[0]; so7[1] = v7[1]; *(FFT_V2*)&a[14] = so7;
        FFT_V2 sh7; sh7[0] = v7[2]; sh7[1] = v7[3]; *(FFT_V2*)&a[46] = sh7;
        FFT_V2 so8; so8[0] = v8[0]; so8[1] = v8[1]; *(FFT_V2*)&a[16] = so8;
        FFT_V2 sh8; sh8[0] = v8[2]; sh8[1] = v8[3]; *(FFT_V2*)&a[48] = sh8;
        FFT_V2 so9; so9[0] = v9[0]; so9[1] = v9[1]; *(FFT_V2*)&a[18] = so9;
        FFT_V2 sh9; sh9[0] = v9[2]; sh9[1] = v9[3]; *(FFT_V2*)&a[50] = sh9;
        FFT_V2 so10; so10[0] = v10[0]; so10[1] = v10[1]; *(FFT_V2*)&a[20] = so10;
        FFT_V2 sh10; sh10[0] = v10[2]; sh10[1] = v10[3]; *(FFT_V2*)&a[52] = sh10;
        FFT_V2 so11; so11[0] = v11[0]; so11[1] = v11[1]; *(FFT_V2*)&a[22] = so11;
        FFT_V2 sh11; sh11[0] = v11[2]; sh11[1] = v11[3]; *(FFT_V2*)&a[54] = sh11;
        FFT_V2 so12; so12[0] = v12[0]; so12[1] = v12[1]; *(FFT_V2*)&a[24] = so12;
        FFT_V2 sh12; sh12[0] = v12[2]; sh12[1] = v12[3]; *(FFT_V2*)&a[56] = sh12;
        FFT_V2 so13; so13[0] = v13[0]; so13[1] = v13[1]; *(FFT_V2*)&a[26] = so13;
        FFT_V2 sh13; sh13[0] = v13[2]; sh13[1] = v13[3]; *(FFT_V2*)&a[58] = sh13;
        FFT_V2 so14; so14[0] = v14[0]; so14[1] = v14[1]; *(FFT_V2*)&a[28] = so14;
        FFT_V2 sh14; sh14[0] = v14[2]; sh14[1] = v14[3]; *(FFT_V2*)&a[60] = sh14;
        FFT_V2 so15; so15[0] = v15[0]; so15[1] = v15[1]; *(FFT_V2*)&a[30] = so15;
        FFT_V2 sh15; sh15[0] = v15[2]; sh15[1] = v15[3]; *(FFT_V2*)&a[62] = sh15;
    }
}


static void fft_codelet16_loop_(FFT_R* d, unsigned long b0, unsigned long b1, int inv) __attribute__((always_inline)) {
    FFT_R sgn = inv ? (FFT_R)-1.0 : (FFT_R)1.0;
    unsigned long b = b0;
    unsafe {
        while (b + 2UL <= b1) { fft_codelet16x2_body_(&d[32UL * b], sgn); b = b + 2UL; }
        while (b < b1) { fft_codelet16_body_(&d[32UL * b], sgn); b = b + 1UL; }
    }
}

static void fft_codelet16_base_(FFT_R* d, unsigned long b0, unsigned long b1, int inv) { fft_codelet16_loop_(d, b0, b1, inv); }
#if defined(__x86_64__)
static void fft_codelet16_fma_(FFT_R* d, unsigned long b0, unsigned long b1, int inv)
    __attribute__((target("avx2,fma"))) { fft_codelet16_loop_(d, b0, b1, inv); }
#endif

// Blocks of 16 values [b0, b1): stages 2..16.
static void fft_codelet16_(FFT_R* d, unsigned long b0, unsigned long b1, int inv) {
#if defined(__x86_64__)
    if (simd_level() >= SIMD_AVX2) { fft_codelet16_fma_(d, b0, b1, inv); return; }
#endif
    fft_codelet16_base_(d, b0, b1, inv);
}

// Stages of length 'from' .. 'to' (both powers of two) on groups [lo, hi)
// of 'span' elements: radix-4 pairs, with one radix-2 stage first when the
// count is odd.
static void fft_stages_(&FftJob j, unsigned long from, unsigned long to, unsigned long lo, unsigned long hi,
                        unsigned long span) {
    unsigned long stages = 0UL;
    unsigned long l = from;
    while (l <= to) { stages = stages + 1UL; l = l * 2UL; }
    unsigned long len = from;
    // Radix-8 passes while three or more stages remain (except a final
    // four, which go as two radix-4 passes), then radix-4 / radix-2. The
    // radix-8 pass needs len >= 8 (m = len / 2 >= 4).
    while (stages > 0UL) {
        if (stages >= 3UL && stages != 4UL && len >= 8UL) {
            fft_pass8_(j.d, j.t, len / 2UL, lo * (span / 8UL), hi * (span / 8UL), j.inv);
            len = len * 8UL;
            stages = stages - 3UL;
        } else if (stages >= 2UL) {
            fft_pass4_(j.d, j.t, len / 2UL, lo * (span / 4UL), hi * (span / 4UL), j.inv);
            len = len * 4UL;
            stages = stages - 2UL;
        } else {
            fft_pass_(j.d, j.t, len, lo * (span / 2UL), hi * (span / 2UL), j.inv);
            len = len * 2UL;
            stages = stages - 1UL;
        }
    }
}

static void fft_block_job_(&FftJob j, unsigned long lo, unsigned long hi) {
    if (j.block >= 16UL) {
        // stages 2..16 in one codelet pass, the rest radix-4 / radix-2
        unsigned long per = j.block / 16UL;
        fft_codelet16_(j.d, lo * per, hi * per, j.inv);
        if (j.block >= 32UL) fft_stages_(j, 32UL, j.block, lo, hi, j.block);
        return;
    }
    fft_stages_(j, 2UL, j.block, lo, hi, j.block);
}

static void fft_stage8_job_(&FftJob j, unsigned long lo, unsigned long hi) { fft_pass8_(j.d, j.t, j.len / 2UL, lo, hi, j.inv); }

static void fft_stage4_job_(&FftJob j, unsigned long lo, unsigned long hi) { fft_pass4_(j.d, j.t, j.len / 2UL, lo, hi, j.inv); }

static void fft_stage_job_(&FftJob j, unsigned long lo, unsigned long hi) { fft_pass_(j.d, j.t, j.len, lo, hi, j.inv); }

static void fft_scale_job_(&FftJob j, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long i = 2UL * lo;
        while (i < 2UL * hi) { j.d[i] = j.d[i] * j.scale; i = i + 1UL; }
    }
}

// In-place transform of n (a power of two, >= 2) complex values; the
// inverse includes 1/n. 0 if the twiddle table couldn't be allocated.
static int fft_run_(FFT_R* d, unsigned long n, int inv, int mode) {
    if (mode == FFT_MODE_ROW || (mode == FFT_MODE_TOP && n < FFT_VENDOR_SINGLE_MAX)) {
        if (fft_vendor_c2c_(d, n, inv)) return 1;
    }
    int threaded = mode == FFT_MODE_TOP;
    const FFT_R* t = fft_twiddles_(n);
    if (t == (const FFT_R*)0) return 0;
    struct FftJob j;
    j.d = d;
    j.t = t;
    j.n = n;
    j.inv = inv;
    j.block = n <= FFT_ONE_BLOCK ? n : FFT_BLOCK;
    j.len = 0UL;
    j.scale = (FFT_R)(1.0 / (double)n);
    int par = threaded && n >= FFT_PAR_MIN;
    if (par) parallel_for(n, 8192UL, fft_bitrev_job_, &j);
    else fft_bitrev_(d, n, 0UL, n);
    if (par) parallel_for(n / j.block, 1UL, fft_block_job_, &j);
    else fft_block_job_(&j, 0UL, n / j.block);
    // The stages above the block size, over the whole array.
    unsigned long len = j.block * 2UL;
    unsigned long outer = 0UL;
    unsigned long l = len;
    while (l <= n) { outer = outer + 1UL; l = l * 2UL; }
    // (radix-8 / radix-4 / radix-2 chosen as in fft_stages_)
    while (outer > 0UL) {
        j.len = len;
        if (outer >= 3UL && outer != 4UL) {
            if (par) parallel_for(n / 8UL, 1024UL, fft_stage8_job_, &j);
            else fft_pass8_(d, t, len / 2UL, 0UL, n / 8UL, inv);
            len = len * 8UL;
            outer = outer - 3UL;
        } else if (outer >= 2UL) {
            if (par) parallel_for(n / 4UL, 2048UL, fft_stage4_job_, &j);
            else fft_pass4_(d, t, len / 2UL, 0UL, n / 4UL, inv);
            len = len * 4UL;
            outer = outer - 2UL;
        } else {
            if (par) parallel_for(n / 2UL, 4096UL, fft_stage_job_, &j);
            else fft_pass_(d, t, len, 0UL, n / 2UL, inv);
            len = len * 2UL;
            outer = outer - 1UL;
        }
    }
    if (inv) {
        if (par) parallel_for(n, 16384UL, fft_scale_job_, &j);
        else fft_scale_job_(&j, 0UL, n);
    }
    return 1;
}

// ── Real input ───────────────────────────────────────────────────────────────
// A length-n real transform from one length-n/2 complex one: pack
// z[k] = x[2k] + i*x[2k+1], transform, then split Z into the spectra of the
// even and odd samples, E and O, and combine X[k] = E[k] + W^k O[k]
// (W = exp(-2*pi*i/n)). The upper half follows from X[n-k] = conj(X[k]).
// The real-input split after the half-length transform, two bins per
// FFT_V4 from each end: for bins k, k+1 and their partners m-k, m-k-1
// (loaded as one vector and swapped),
//   E = (Z[k] + conj Z[m-k]) / 2,  O = -i (Z[k] - conj Z[m-k]) / 2
//   X[k] = E + W^k O,  X[m-k] = conj(E - W^k O)
// then the upper half X[n-k] = conj X[k], also two at a time. Built per
// SIMD level like the passes.
static FFT_V4 fft_swap_pairs_(FFT_V4 v) __attribute__((always_inline)) {
    FFT_V4 r; r[0] = v[2]; r[1] = v[3]; r[2] = v[0]; r[3] = v[1];
    return r;
}

static FFT_V4 fft_conj4_(FFT_V4 v) __attribute__((always_inline)) {
    FFT_V4 r; r[0] = v[0]; r[1] = -v[1]; r[2] = v[2]; r[3] = -v[3];
    return r;
}

static void fft_real_split_body_(FFT_R* out, const FFT_R* w, unsigned long m, unsigned long n, int full)
    __attribute__((always_inline)) {
    unsafe {
        FFT_R z0r = out[0];
        FFT_R z0i = out[1];
        FFT_V4 half; half[0] = (FFT_R)0.5; half[1] = (FFT_R)0.5; half[2] = (FFT_R)0.5; half[3] = (FFT_R)0.5;
        unsigned long k = 1UL;
        // vector pairs while (k, k+1) and (m-k-1, m-k) don't meet
        while (2UL * k + 2UL < m) {
            unsigned long jj = m - k - 1UL;                     // the pair (jj, jj+1) = (m-k-1, m-k)
            FFT_V4 a = *(FFT_V4*)&out[2UL * k];                 // Z[k], Z[k+1]
            FFT_V4 b = fft_swap_pairs_(*(FFT_V4*)&out[2UL * jj]);   // Z[m-k], Z[m-k-1]
            FFT_V4 cb = fft_conj4_(b);
            FFT_V4 e = half * (a + cb);
            FFT_V4 d = half * (a - cb);
            FFT_V4 o; o[0] = d[1]; o[1] = -d[0]; o[2] = d[3]; o[3] = -d[2];   // -i * d
            FFT_V4 p = fft_cmul4_(o, *(const FFT_V4*)&w[2UL * k], (FFT_R)1.0);
            *(FFT_V4*)&out[2UL * k] = e + p;
            *(FFT_V4*)&out[2UL * jj] = fft_swap_pairs_(fft_conj4_(e - p));
            k = k + 2UL;
        }
        while (k <= m / 2UL) {
            unsigned long jj = m - k;
            FFT_R zkr = out[2UL * k];   FFT_R zki = out[2UL * k + 1UL];
            FFT_R zjr = out[2UL * jj];  FFT_R zji = out[2UL * jj + 1UL];
            FFT_R er = (FFT_R)0.5 * (zkr + zjr);
            FFT_R ei = (FFT_R)0.5 * (zki - zji);
            FFT_R or_ = (FFT_R)0.5 * (zki + zji);
            FFT_R oi = (FFT_R)-0.5 * (zkr - zjr);
            FFT_R wr = w[2UL * k];
            FFT_R wi = w[2UL * k + 1UL];
            FFT_R pr = wr * or_ - wi * oi;   // W^k * O[k]
            FFT_R pi = wr * oi + wi * or_;
            out[2UL * k] = er + pr;
            out[2UL * k + 1UL] = ei + pi;
            if (jj != k) {                  // X[m-k] = conj(E[k] - W^k O[k])
                out[2UL * jj] = er - pr;
                out[2UL * jj + 1UL] = -(ei - pi);
            }
            k = k + 1UL;
        }
        out[0] = z0r + z0i;
        out[1] = (FFT_R)0.0;
        out[2UL * m] = z0r - z0i;
        out[2UL * m + 1UL] = (FFT_R)0.0;
        if (!full) return;   // the n/2 + 1 bins only
        // X[n-k] = conj X[k], k = 1 .. m-1: pairs (k, k+1) -> (n-k-1, n-k)
        k = 1UL;
        while (k + 2UL <= m) {
            *(FFT_V4*)&out[2UL * (n - k - 1UL)] = fft_swap_pairs_(fft_conj4_(*(FFT_V4*)&out[2UL * k]));
            k = k + 2UL;
        }
        while (k < m) {
            out[2UL * (n - k)] = out[2UL * k];
            out[2UL * (n - k) + 1UL] = -out[2UL * k + 1UL];
            k = k + 1UL;
        }
    }
}

static void fft_real_split_base_(FFT_R* out, const FFT_R* w, unsigned long m, unsigned long n, int full) {
    fft_real_split_body_(out, w, m, n, full);
}
#if defined(__x86_64__)
static void fft_real_split_avx2_(FFT_R* out, const FFT_R* w, unsigned long m, unsigned long n, int full)
    __attribute__((target("avx2,fma"))) { fft_real_split_body_(out, w, m, n, full); }
#endif

// full = 0: only bins 0 .. n/2 (out holds n/2 + 1 complex values).
static int fft_real_run_(const FFT_R* in, FFT_R* out, unsigned long n, int mode, int full) {
    // (the library transform is out of place only, so not when in == out)
    if (mode != FFT_MODE_RT && in != (const FFT_R*)out && fft_vendor_r2c_(in, out, n, full)) return 1;
    unsafe {
        if (n == 1UL) { out[0] = in[0]; out[1] = (FFT_R)0.0; return 1; }
        unsigned long m = n / 2UL;
        if (in != (const FFT_R*)out) {
            unsigned long k = 0UL;
            while (k < n) { out[k] = in[k]; k = k + 1UL; }
        }
        if (m >= 2UL && fft_run_(out, m, 0, mode) == 0) return 0;
        const FFT_R* t = fft_twiddles_(n);
        if (t == (const FFT_R*)0) return 0;
        const FFT_R* w = &t[2UL * m];   // w[k] = W^k, k < m
#if defined(__x86_64__)
        if (simd_level() >= SIMD_AVX2) { fft_real_split_avx2_(out, w, m, n, full); return 1; }
#endif
        fft_real_split_base_(out, w, m, n, full);
    }
    return 1;
}

// ── Batches ──────────────────────────────────────────────────────────────────

struct FftBatchJob {
    const FFT_R* in;      // fft_real_batch only
    FFT_R*       d;
    unsigned long n;
    int           inv;
    int           ok;      // cleared (atomically) if a transform failed
};

static void fft_batch_job_(&FftBatchJob j, unsigned long lo, unsigned long hi) {
    unsigned long r = lo;
    while (r < hi) {
        unsafe {
            if (fft_run_(&j.d[2UL * r * j.n], j.n, j.inv, FFT_MODE_ROW) == 0) atomic_store_int(&j.ok, 0);
        }
        r = r + 1UL;
    }
}

static void fft_real_batch_job_(&FftBatchJob j, unsigned long lo, unsigned long hi) {
    unsigned long r = lo;
    while (r < hi) {
        unsafe {
            if (fft_real_run_(&j.in[r * j.n], &j.d[2UL * r * j.n], j.n, FFT_MODE_ROW, 1) == 0) atomic_store_int(&j.ok, 0);
        }
        r = r + 1UL;
    }
}

// Half spectrum: row r's n/2 + 1 bins at d[r * (n + 2)].
static void fft_real_half_batch_job_(&FftBatchJob j, unsigned long lo, unsigned long hi) {
    unsigned long r = lo;
    while (r < hi) {
        unsafe {
            if (fft_real_run_(&j.in[r * j.n], &j.d[r * (j.n + 2UL)], j.n, FFT_MODE_ROW, 0) == 0) atomic_store_int(&j.ok, 0);
        }
        r = r + 1UL;
    }
}

static int fft_real_half_batch_(const FFT_R* in, FFT_R* out, unsigned long n, unsigned long batch) {
    if (dsp_is_pow2(n) == 0) return 0;
    if (batch == 0UL) return 1;
    if (fft_twiddles_(n) == (const FFT_R*)0) return 0;
    struct FftBatchJob j;
    j.in = in;
    j.d = out;
    j.n = n;
    j.inv = 0;
    j.ok = 1;
    if (batch == 1UL) return fft_real_run_(in, out, n, FFT_MODE_TOP, 0);
    parallel_for(batch, 16384UL / n + 1UL, fft_real_half_batch_job_, &j);
    return j.ok;
}

// In place: row r's real input is the first n values of its own output row.
static void fft_real_inplace_job_(&FftBatchJob j, unsigned long lo, unsigned long hi) {
    unsigned long r = lo;
    while (r < hi) {
        unsafe {
            FFT_R* row = &j.d[2UL * r * j.n];
            if (fft_real_run_((const FFT_R*)row, row, j.n, FFT_MODE_ROW, 1) == 0) atomic_store_int(&j.ok, 0);
        }
        r = r + 1UL;
    }
}

static int fft_real_inplace_(FFT_R* d, unsigned long n, unsigned long batch) {
    if (dsp_is_pow2(n) == 0) return 0;
    if (batch == 0UL) return 1;
    if (fft_twiddles_(n) == (const FFT_R*)0) return 0;
    struct FftBatchJob j;
    unsafe { j.in = (const FFT_R*)0; }
    j.d = d;
    j.n = n;
    j.inv = 0;
    j.ok = 1;
    if (batch == 1UL) return fft_real_run_((const FFT_R*)d, d, n, FFT_MODE_TOP, 1);
    parallel_for(batch, 16384UL / n + 1UL, fft_real_inplace_job_, &j);
    return j.ok;
}

