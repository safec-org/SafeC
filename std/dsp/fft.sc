// SafeC Standard Library — FFT implementation (see fft.h)
#pragma once
#include <std/dsp/fft.h>
#include <std/dsp/complex_dsp.h>
#include <std/math.h>
#include <std/mem.h>
#include <std/atomic.h>
#include <std/thread.h>
#include <std/sync/parallel.h>
#include <std/simd/simd.h>

#define SAFEC_DSP_TWO_PI 6.28318530717958647692

namespace std {

inline int dsp_is_pow2(unsigned long n) {
    if (n == 0UL) { return 0; }
    return (n & (n - 1UL)) == 0UL ? 1 : 0;
}

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

static long long g_fft_tw_ptr = 0LL;   // (double*) table, interleaved re/im per entry
static long long g_fft_tw_n = 0LL;     // the table covers lengths <= this
static int g_fft_tw_lock = 0;

static unsigned long fft_tw_len_() {
#if FFT_HAS_THREADS
    return (unsigned long)atomic_load_ll(&g_fft_tw_n);
#else
    return (unsigned long)g_fft_tw_n;
#endif
}

static double* fft_tw_ptr_() {
#if FFT_HAS_THREADS
    long long p = atomic_load_ll(&g_fft_tw_ptr);
#else
    long long p = g_fft_tw_ptr;
#endif
    unsafe { return (double*)p; }
}

static void fft_tw_build_(double* t, unsigned long N) {
    unsafe {
        unsigned long h = N / 2UL;
        // Top level, h = N/2: exact cos/sin every 32 entries, complex
        // rotation by exp(-2*pi*i/N) in between (error stays ~1e-15).
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
                t[2UL * (h + k)] = wr;
                t[2UL * (h + k) + 1UL] = wi;
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
static const double* fft_twiddles_(unsigned long n) {
    if (fft_tw_len_() >= n) return (const double*)fft_tw_ptr_();
#if FFT_HAS_THREADS
    while (atomic_exchange_int(&g_fft_tw_lock, 1) != 0) { thread_yield(); }
#endif
    const double* r = (const double*)0;
    if (fft_tw_len_() >= n) {
        r = (const double*)fft_tw_ptr_();
    } else {
        unsigned long N = n < 64UL ? 64UL : n;
        unsafe {
            double* t = (double*)alloc_raw_(sizeof(double) * 2UL * N);
            if (t != (double*)0) {
                fft_tw_build_(t, N);
#if FFT_HAS_THREADS
                atomic_store_ll(&g_fft_tw_ptr, (long long)t);
                atomic_store_ll(&g_fft_tw_n, (long long)N);
#else
                g_fft_tw_ptr = (long long)t;
                g_fft_tw_n = (long long)N;
#endif
                r = (const double*)t;
            }
        }
    }
#if FFT_HAS_THREADS
    atomic_store_int(&g_fft_tw_lock, 0);
#endif
    return r;
}

int fft_prepare(unsigned long n) {
    if (dsp_is_pow2(n) == 0) return 0;
    return fft_twiddles_(n) != (const double*)0 ? 1 : 0;
}

// ── Radix-2 core on interleaved (re, im) doubles ─────────────────────────────

// Stages up to this length run block by block, so a block (2^11 complex =
// 32 KB) stays in L1/L2 through all of them.
#define FFT_BLOCK   2048UL
// Single transforms at least this long use the thread pool.
#define FFT_PAR_MIN 32768UL

// Bit-reversal swaps for i in [lo, hi). Each pair (i, rev i) is swapped by
// the range holding the smaller index, so disjoint ranges never touch the
// same element.
static void fft_bitrev_(double* d, unsigned long n, unsigned long lo, unsigned long hi) {
    int bits = dsp_log2_(n);
    unsigned long j = dsp_bit_reverse_(lo, bits);
    unsigned long i = lo;
    unsafe {
        while (i < hi) {
            if (i < j) {
                double tr = d[2UL * i];
                double ti = d[2UL * i + 1UL];
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
static void fft_pass_(double* d, const double* t, unsigned long len, unsigned long g0, unsigned long g1, int inv) {
    unsigned long half = len / 2UL;
    unsafe {
        if (half == 1UL) {
            unsigned long g = g0;
            while (g < g1) {
                double* a = &d[4UL * g];
                double ur = a[0]; double ui = a[1];
                double vr = a[2]; double vi = a[3];
                a[0] = ur + vr; a[1] = ui + vi;
                a[2] = ur - vr; a[3] = ui - vi;
                g = g + 1UL;
            }
            return;
        }
        double sgn = inv ? -1.0 : 1.0;
        unsigned long g = g0;
        while (g < g1) {
            unsigned long blk = g / half;
            unsigned long k = g % half;
            unsigned long kend = half - k < g1 - g ? half : k + (g1 - g);
            double* a = &d[2UL * (blk * len)];
            double* b = &a[2UL * half];
            const double* w = &t[2UL * half];
            unsigned long run = kend - k;
            while (k < kend) {
                // One complex value per f64x2: v = b * w as
                // (br, bi) * wr + (bi, br) * (-wi, wi).
                f64x2 bv = *(f64x2*)&b[2UL * k];
                f64x2 bs;
                bs[0] = bv[1];
                bs[1] = bv[0];
                double wr = w[2UL * k];
                double wi = sgn * w[2UL * k + 1UL];
                f64x2 vr; vr[0] = wr; vr[1] = wr;
                f64x2 vi; vi[0] = -wi; vi[1] = wi;
                f64x2 v = bv * vr + bs * vi;
                f64x2 u = *(f64x2*)&a[2UL * k];
                *(f64x2*)&a[2UL * k] = u + v;
                *(f64x2*)&b[2UL * k] = u - v;
                k = k + 1UL;
            }
            g = g + run;
        }
    }
}

struct FftJob {
    double*       d;
    const double* t;
    unsigned long n;
    unsigned long len;     // stage, for fft_stage_job_
    unsigned long block;   // block size, for fft_block_job_
    int           inv;
    double        scale;   // for fft_scale_job_
};

static void fft_bitrev_job_(&FftJob j, unsigned long lo, unsigned long hi) { fft_bitrev_(j.d, j.n, lo, hi); }

// All stages of length <= j.block within blocks [lo, hi).
static void fft_block_job_(&FftJob j, unsigned long lo, unsigned long hi) {
    unsigned long per = j.block / 2UL;   // butterflies per block and stage
    unsigned long len = 2UL;
    while (len <= j.block) {
        fft_pass_(j.d, j.t, len, lo * per, hi * per, j.inv);
        len = len * 2UL;
    }
}

static void fft_stage_job_(&FftJob j, unsigned long lo, unsigned long hi) { fft_pass_(j.d, j.t, j.len, lo, hi, j.inv); }

static void fft_scale_job_(&FftJob j, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long i = 2UL * lo;
        while (i < 2UL * hi) { j.d[i] = j.d[i] * j.scale; i = i + 1UL; }
    }
}

// In-place transform of n (a power of two, >= 2) complex values; the
// inverse includes 1/n. 0 if the twiddle table couldn't be allocated.
static int fft_run_(double* d, unsigned long n, int inv, int threaded) {
    const double* t = fft_twiddles_(n);
    if (t == (const double*)0) return 0;
    struct FftJob j;
    j.d = d;
    j.t = t;
    j.n = n;
    j.inv = inv;
    j.block = n < FFT_BLOCK ? n : FFT_BLOCK;
    j.len = 0UL;
    j.scale = 1.0 / (double)n;
    int par = threaded && n >= FFT_PAR_MIN;
    if (par) parallel_for(n, 8192UL, fft_bitrev_job_, &j);
    else fft_bitrev_(d, n, 0UL, n);
    if (par) parallel_for(n / j.block, 1UL, fft_block_job_, &j);
    else fft_block_job_(&j, 0UL, n / j.block);
    unsigned long len = j.block * 2UL;
    while (len <= n) {
        j.len = len;
        if (par) parallel_for(n / 2UL, 4096UL, fft_stage_job_, &j);
        else fft_pass_(d, t, len, 0UL, n / 2UL, inv);
        len = len * 2UL;
    }
    if (inv) {
        if (par) parallel_for(n, 16384UL, fft_scale_job_, &j);
        else fft_scale_job_(&j, 0UL, n);
    }
    return 1;
}

static int fft_dispatch_(&Complex data, unsigned long n, int inv) {
    if (dsp_is_pow2(n) == 0) return 0;
    if (n <= 1UL) return 1;
    unsafe { return fft_run_((double*)data, n, inv, 1); }
}

int fft(&Complex data, unsigned long n) { return fft_dispatch_(data, n, 0); }
int ifft(&Complex data, unsigned long n) { return fft_dispatch_(data, n, 1); }

int fft_serial(&Complex data, unsigned long n) {
    if (dsp_is_pow2(n) == 0) return 0;
    if (n <= 1UL) return 1;
    unsafe { return fft_run_((double*)data, n, 0, 0); }
}

int ifft_serial(&Complex data, unsigned long n) {
    if (dsp_is_pow2(n) == 0) return 0;
    if (n <= 1UL) return 1;
    unsafe { return fft_run_((double*)data, n, 1, 0); }
}

// ── Real input ───────────────────────────────────────────────────────────────
// A length-n real transform from one length-n/2 complex one: pack
// z[k] = x[2k] + i*x[2k+1], transform, then split Z into the spectra of the
// even and odd samples, E and O, and combine X[k] = E[k] + W^k O[k]
// (W = exp(-2*pi*i/n)). The upper half follows from X[n-k] = conj(X[k]).
static int fft_real_run_(const double* in, double* out, unsigned long n, int threaded) {
    unsafe {
        if (n == 1UL) { out[0] = in[0]; out[1] = 0.0; return 1; }
        unsigned long m = n / 2UL;
        unsigned long k = 0UL;
        while (k < m) {
            out[2UL * k] = in[2UL * k];
            out[2UL * k + 1UL] = in[2UL * k + 1UL];
            k = k + 1UL;
        }
        if (m >= 2UL && fft_run_(out, m, 0, threaded) == 0) return 0;
        const double* t = fft_twiddles_(n);
        if (t == (const double*)0) return 0;
        const double* w = &t[2UL * m];   // w[k] = W^k, k < m
        double z0r = out[0];
        double z0i = out[1];
        k = 1UL;
        while (k <= m / 2UL) {
            unsigned long jj = m - k;
            double zkr = out[2UL * k];   double zki = out[2UL * k + 1UL];
            double zjr = out[2UL * jj];  double zji = out[2UL * jj + 1UL];
            double er = 0.5 * (zkr + zjr);
            double ei = 0.5 * (zki - zji);
            double or_ = 0.5 * (zki + zji);
            double oi = -0.5 * (zkr - zjr);
            double wr = w[2UL * k];
            double wi = w[2UL * k + 1UL];
            double pr = wr * or_ - wi * oi;   // W^k * O[k]
            double pi = wr * oi + wi * or_;
            out[2UL * k] = er + pr;
            out[2UL * k + 1UL] = ei + pi;
            if (jj != k) {                  // X[m-k] = conj(E[k] - W^k O[k])
                out[2UL * jj] = er - pr;
                out[2UL * jj + 1UL] = -(ei - pi);
            }
            k = k + 1UL;
        }
        out[0] = z0r + z0i;
        out[1] = 0.0;
        out[2UL * m] = z0r - z0i;
        out[2UL * m + 1UL] = 0.0;
        k = 1UL;
        while (k < m) {
            out[2UL * (n - k)] = out[2UL * k];
            out[2UL * (n - k) + 1UL] = -out[2UL * k + 1UL];
            k = k + 1UL;
        }
    }
    return 1;
}

int fft_real(const &double in, &Complex out, unsigned long n) {
    if (dsp_is_pow2(n) == 0) return 0;
    unsafe { return fft_real_run_((const double*)in, (double*)out, n, 1); }
}

// ── Batches ──────────────────────────────────────────────────────────────────

struct FftBatchJob {
    const double* in;      // fft_real_batch only
    double*       d;
    unsigned long n;
    int           inv;
    int           ok;      // cleared (atomically) if a transform failed
};

static void fft_batch_job_(&FftBatchJob j, unsigned long lo, unsigned long hi) {
    unsigned long r = lo;
    while (r < hi) {
        unsafe {
            if (fft_run_(&j.d[2UL * r * j.n], j.n, j.inv, 0) == 0) atomic_store_int(&j.ok, 0);
        }
        r = r + 1UL;
    }
}

static void fft_real_batch_job_(&FftBatchJob j, unsigned long lo, unsigned long hi) {
    unsigned long r = lo;
    while (r < hi) {
        unsafe {
            if (fft_real_run_(&j.in[r * j.n], &j.d[2UL * r * j.n], j.n, 0) == 0) atomic_store_int(&j.ok, 0);
        }
        r = r + 1UL;
    }
}

int fft_batch(&Complex data, unsigned long n, unsigned long batch, int inverse) {
    if (dsp_is_pow2(n) == 0) return 0;
    if (n <= 1UL || batch == 0UL) return 1;
    if (fft_twiddles_(n) == (const double*)0) return 0;
    struct FftBatchJob j;
    unsafe {
        j.in = (const double*)0;
        j.d = (double*)data;
    }
    j.n = n;
    j.inv = inverse ? 1 : 0;
    j.ok = 1;
    if (batch == 1UL) return fft_run_(j.d, n, j.inv, 1);
    // Rows go to threads whole; ~16K complex values per chunk.
    parallel_for(batch, 16384UL / n + 1UL, fft_batch_job_, &j);
    return j.ok;
}

int fft_real_batch(const &double in, &Complex out, unsigned long n, unsigned long batch) {
    if (dsp_is_pow2(n) == 0) return 0;
    if (batch == 0UL) return 1;
    if (fft_twiddles_(n) == (const double*)0) return 0;
    struct FftBatchJob j;
    unsafe {
        j.in = (const double*)in;
        j.d = (double*)out;
    }
    j.n = n;
    j.inv = 0;
    j.ok = 1;
    if (batch == 1UL) { unsafe { return fft_real_run_(j.in, j.d, n, 1); } }
    parallel_for(batch, 16384UL / n + 1UL, fft_real_batch_job_, &j);
    return j.ok;
}

int ffft(&FComplex data, unsigned long n) {
    if (dsp_is_pow2(n) == 0) {
        return 0;
    }
    if (n <= 1UL) {
        return 1;
    }
    int numBits = dsp_log2_(n);

    unsigned long i = 0UL;
    while (i < n) {
        unsigned long j = dsp_bit_reverse_(i, numBits);
        if (i < j) {
            unsafe {
                struct FComplex t = data[i];
                data[i] = data[j];
                data[j] = t;
            }
        }
        i = i + 1UL;
    }

    unsigned long len = 2UL;
    while (len <= n) {
        unsigned long half = len / 2UL;
        unsigned long block = 0UL;
        while (block < n) {
            unsigned long k = 0UL;
            while (k < half) {
                double theta = -SAFEC_DSP_TWO_PI * (double)k / (double)len;
                struct FComplex w = complex_to_fixed(complex_from_polar(1.0, theta));
                unsafe {
                    struct FComplex u = data[block + k];
                    struct FComplex v = data[block + k + half] * w;
                    data[block + k]        = u + v;
                    data[block + k + half] = u - v;
                }
                k = k + 1UL;
            }
            block = block + len;
        }
        len = len * 2UL;
    }
    return 1;
}

int fifft(&FComplex data, unsigned long n) {
    if (dsp_is_pow2(n) == 0) {
        return 0;
    }
    unsigned long i = 0UL;
    while (i < n) {
        unsafe { data[i] = data[i].conj(); }
        i = i + 1UL;
    }
    ffft(data, n);
    if (n == 0UL) {
        return 1;
    }
    Fixed invN = fixed_div((Fixed)FIXED_ONE, fixed_from_int((int)n));
    i = 0UL;
    while (i < n) {
        unsafe {
            struct FComplex c = data[i].conj();
            data[i] = fcomplex_new(fixed_mul(c.re, invN), fixed_mul(c.im, invN));
        }
        i = i + 1UL;
    }
    return 1;
}

} // namespace std
