// SafeC Standard Library — packed SGEMM (see gemm.h).
//
// C is cut into MC x NC tiles, each one task for the thread pool. A task
// walks k in KC-deep slices: it packs its A slice into MR-row strips
// (alpha folded in) and its B slice into NR-column strips, both zero-padded
// to whole strips, then runs the MR x NR kernel over every strip pair.
// A packed A slice (MC x KC) stays in L2 across the B strips; a B strip
// (KC x NR) stays in L1 across the A strips.
#pragma once
#include <std/simd/gemm.h>
#include <std/simd/simd.h>
#include <std/sync/parallel.h>
#include <std/mem.h>
#include <std/panic.h>

namespace std {

#define GEMM_MR 4UL
#define GEMM_NR 16UL
#define GEMM_MC 64UL
#define GEMM_NC 256UL
#define GEMM_KC 256UL
// Below this many multiply-adds a product runs on one thread: waking the
// pool costs a few microseconds.
#define GEMM_PAR_MIN 1048576UL

// Packing buffers, kept per thread and reused: allocating them per call
// would mmap and page-fault ~320 KB every time. Freestanding targets have no
// TLS, so there they are shared, and sgemm must not run on two threads at once.
#if defined(__SAFEC_FREESTANDING__) || defined(__wasm__)
static float* g_gemm_buf = (float*)0;
#else
static thread_local float* g_gemm_buf = (float*)0;
#endif

static float* gemm_scratch_() {
    if (g_gemm_buf == (float*)0) {
        unsafe { g_gemm_buf = (float*)alloc_raw_(sizeof(float) * (GEMM_MC * GEMM_KC + GEMM_KC * GEMM_NC)); }
        if (g_gemm_buf == (float*)0) panic_at("sgemm: out of memory", "gemm.sc", __LINE__);
    }
    return g_gemm_buf;
}

static f32x8 gemm_splat_(float x) {
    f32x8 v;
    v[0] = x; v[1] = x; v[2] = x; v[3] = x; v[4] = x; v[5] = x; v[6] = x; v[7] = x;
    return v;
}

// cp[0..4)[0..16) += ap (kc x 4 strip) * bp (kc x 16 strip).
static void gemm_kernel_(const float* ap, const float* bp, unsigned long kc, float* cp, unsigned long ldc) {
    f32x8 c00; f32x8 c01; f32x8 c10; f32x8 c11;
    f32x8 c20; f32x8 c21; f32x8 c30; f32x8 c31;
    unsafe {
        unsigned long p = 0UL;
        while (p < kc) {
            f32x8 b0 = *(const f32x8*)&bp[0];
            f32x8 b1 = *(const f32x8*)&bp[8];
            f32x8 a = gemm_splat_(ap[0]);
            c00 = __builtin_fmuladd(a, b0, c00); c01 = __builtin_fmuladd(a, b1, c01);
            a = gemm_splat_(ap[1]);
            c10 = __builtin_fmuladd(a, b0, c10); c11 = __builtin_fmuladd(a, b1, c11);
            a = gemm_splat_(ap[2]);
            c20 = __builtin_fmuladd(a, b0, c20); c21 = __builtin_fmuladd(a, b1, c21);
            a = gemm_splat_(ap[3]);
            c30 = __builtin_fmuladd(a, b0, c30); c31 = __builtin_fmuladd(a, b1, c31);
            ap = ap + 4;
            bp = bp + 16;
            p = p + 1UL;
        }
        float* r = cp;
        *(f32x8*)&r[0] = *(f32x8*)&r[0] + c00; *(f32x8*)&r[8] = *(f32x8*)&r[8] + c01;
        r = r + ldc;
        *(f32x8*)&r[0] = *(f32x8*)&r[0] + c10; *(f32x8*)&r[8] = *(f32x8*)&r[8] + c11;
        r = r + ldc;
        *(f32x8*)&r[0] = *(f32x8*)&r[0] + c20; *(f32x8*)&r[8] = *(f32x8*)&r[8] + c21;
        r = r + ldc;
        *(f32x8*)&r[0] = *(f32x8*)&r[0] + c30; *(f32x8*)&r[8] = *(f32x8*)&r[8] + c31;
    }
}

struct GemmJob {
    int transA;
    int transB;
    unsigned long m;
    unsigned long n;
    unsigned long k;
    float alpha;
    float beta;
    const float* a;
    unsigned long lda;
    const float* b;
    unsigned long ldb;
    float* c;
    unsigned long ldc;
    unsigned long mc;       // tile size (<= GEMM_MC, a multiple of GEMM_MR)
    unsigned long nc;       // tile size (<= GEMM_NC, a multiple of GEMM_NR)
    unsigned long tilesN;   // tiles per row of C
};

// Packs op(A)[i0 .. i0+mc, p0 .. p0+kc) * alpha into MR-row strips.
static void gemm_pack_a_(const &GemmJob j, float* dst, unsigned long i0, unsigned long mc,
                         unsigned long p0, unsigned long kc) {
    unsafe {
        unsigned long s = 0UL;
        while (s < mc) {
            unsigned long p = 0UL;
            while (p < kc) {
                unsigned long r = 0UL;
                while (r < GEMM_MR) {
                    float v = (float)0.0;
                    if (s + r < mc) {
                        unsigned long i = i0 + s + r;
                        unsigned long q = p0 + p;
                        v = j.transA ? j.a[q * j.lda + i] : j.a[i * j.lda + q];
                    }
                    dst[r] = v * j.alpha;
                    r = r + 1UL;
                }
                dst = dst + GEMM_MR;
                p = p + 1UL;
            }
            s = s + GEMM_MR;
        }
    }
}

// Packs op(B)[p0 .. p0+kc, j0 .. j0+nc) into NR-column strips.
static void gemm_pack_b_(const &GemmJob j, float* dst, unsigned long p0, unsigned long kc,
                         unsigned long j0, unsigned long nc) {
    unsafe {
        unsigned long s = 0UL;
        while (s < nc) {
            unsigned long w = nc - s < GEMM_NR ? nc - s : GEMM_NR;
            unsigned long p = 0UL;
            while (p < kc) {
                unsigned long q = p0 + p;
                if (!j.transB && w == GEMM_NR) {
                    const float* src = &j.b[q * j.ldb + j0 + s];
                    *(f32x8*)&dst[0] = *(const f32x8*)&src[0];
                    *(f32x8*)&dst[8] = *(const f32x8*)&src[8];
                } else {
                    unsigned long t = 0UL;
                    while (t < GEMM_NR) {
                        float v = (float)0.0;
                        if (t < w) {
                            unsigned long col = j0 + s + t;
                            v = j.transB ? j.b[col * j.ldb + q] : j.b[q * j.ldb + col];
                        }
                        dst[t] = v;
                        t = t + 1UL;
                    }
                }
                dst = dst + GEMM_NR;
                p = p + 1UL;
            }
            s = s + GEMM_NR;
        }
    }
}

static void gemm_tiles_(&GemmJob j, unsigned long lo, unsigned long hi) {
    unsafe {
        float* apack = gemm_scratch_();
        float* bpack = &apack[GEMM_MC * GEMM_KC];
        float edge[64];                      // GEMM_MR x GEMM_NR partial tile
        unsigned long t = lo;
        while (t < hi) {
            unsigned long i0 = (t / j.tilesN) * j.mc;
            unsigned long j0 = (t % j.tilesN) * j.nc;
            unsigned long mc = j.m - i0 < j.mc ? j.m - i0 : j.mc;
            unsigned long nc = j.n - j0 < j.nc ? j.n - j0 : j.nc;

            // C tile *= beta (beta == 0: clear, never read the old values).
            unsigned long r = 0UL;
            while (r < mc) {
                float* row = &j.c[(i0 + r) * j.ldc + j0];
                unsigned long q = 0UL;
                if (j.beta == (float)0.0) {
                    while (q < nc) { row[q] = (float)0.0; q = q + 1UL; }
                } else if (j.beta != (float)1.0) {
                    while (q < nc) { row[q] = row[q] * j.beta; q = q + 1UL; }
                }
                r = r + 1UL;
            }

            unsigned long p0 = 0UL;
            while (p0 < j.k) {
                unsigned long kc = j.k - p0 < GEMM_KC ? j.k - p0 : GEMM_KC;
                gemm_pack_a_(j, apack, i0, mc, p0, kc);
                gemm_pack_b_(j, bpack, p0, kc, j0, nc);
                unsigned long sb = 0UL;
                while (sb < nc) {
                    unsigned long w = nc - sb < GEMM_NR ? nc - sb : GEMM_NR;
                    const float* bp = &bpack[sb * kc];
                    unsigned long sa = 0UL;
                    while (sa < mc) {
                        unsigned long h = mc - sa < GEMM_MR ? mc - sa : GEMM_MR;
                        const float* ap = &apack[sa * kc];
                        float* cp = &j.c[(i0 + sa) * j.ldc + j0 + sb];
                        if (h == GEMM_MR && w == GEMM_NR) {
                            gemm_kernel_(ap, bp, kc, cp, j.ldc);
                        } else {
                            unsigned long e = 0UL;
                            while (e < 64UL) { edge[e] = (float)0.0; e = e + 1UL; }
                            gemm_kernel_(ap, bp, kc, &edge[0], GEMM_NR);
                            unsigned long y = 0UL;
                            while (y < h) {
                                unsigned long x = 0UL;
                                while (x < w) {
                                    cp[y * j.ldc + x] = cp[y * j.ldc + x] + edge[y * GEMM_NR + x];
                                    x = x + 1UL;
                                }
                                y = y + 1UL;
                            }
                        }
                        sa = sa + GEMM_MR;
                    }
                    sb = sb + GEMM_NR;
                }
                p0 = p0 + kc;
            }
            t = t + 1UL;
        }
    }
}

static void gemm_run_(int transA, int transB, unsigned long m, unsigned long n, unsigned long k,
                      float alpha, const &float a, unsigned long lda,
                      const &float b, unsigned long ldb,
                      float beta, &float c, unsigned long ldc, int threaded) {
    if (m == 0UL || n == 0UL) return;
    struct GemmJob j;
    j.transA = transA;
    j.transB = transB;
    j.m = m;
    j.n = n;
    j.k = k;
    j.alpha = alpha;
    j.beta = beta;
    unsafe {
        j.a = (const float*)a;
        j.b = (const float*)b;
        j.c = (float*)c;
    }
    j.lda = lda;
    j.ldb = ldb;
    j.ldc = ldc;
    j.mc = GEMM_MC;
    j.nc = GEMM_NC;
    int par = threaded && m * n * k >= GEMM_PAR_MIN;
    if (par) {
        // Shrink the tiles until every thread gets one, so a mid-sized
        // product doesn't leave most of the pool idle.
        unsigned long want = parallel_threads();
        while (((m + j.mc - 1UL) / j.mc) * ((n + j.nc - 1UL) / j.nc) < want) {
            if (j.nc > 64UL && j.nc >= j.mc) j.nc = j.nc / 2UL;
            else if (j.mc > 16UL) j.mc = j.mc / 2UL;
            else break;
        }
    }
    j.tilesN = (n + j.nc - 1UL) / j.nc;
    unsigned long tiles = ((m + j.mc - 1UL) / j.mc) * j.tilesN;
    if (par && tiles > 1UL)
        parallel_for(tiles, 1UL, gemm_tiles_, &j);
    else
        gemm_tiles_(&j, 0UL, tiles);
}

void sgemm(int transA, int transB, unsigned long m, unsigned long n, unsigned long k,
           float alpha, const &float a, unsigned long lda,
           const &float b, unsigned long ldb,
           float beta, &float c, unsigned long ldc) {
    gemm_run_(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 1);
}

void sgemm_serial(int transA, int transB, unsigned long m, unsigned long n, unsigned long k,
                  float alpha, const &float a, unsigned long lda,
                  const &float b, unsigned long ldb,
                  float beta, &float c, unsigned long ldc) {
    gemm_run_(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 0);
}

} // namespace std
