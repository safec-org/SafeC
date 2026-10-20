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
#include <std/simd/cpu.h>
#include <std/sync/parallel.h>
#include <std/mem.h>
#include <std/panic.h>
#include <std/atomic.h>
#include <std/thread.h>

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

static f32x8 gemm_splat_(float x) __attribute__((always_inline)) {
    f32x8 v;
    v[0] = x; v[1] = x; v[2] = x; v[3] = x; v[4] = x; v[5] = x; v[6] = x; v[7] = x;
    return v;
}

// cp[0..4)[0..16) += ap (kc x 4 strip) * bp (kc x 16 strip), on f32x8.
// One body, compiled once per ISA level by the target() wrappers below.
static void gemm_kernel4_(const float* ap, const float* bp, unsigned long kc, float* cp, unsigned long ldc) __attribute__((always_inline)) {
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

static void gemm_k_sse_(const float* ap, const float* bp, unsigned long kc, float* cp, unsigned long ldc) {
    gemm_kernel4_(ap, bp, kc, cp, ldc);
}

#if defined(__x86_64__)
typedef vec<float, 16> gemm_f32x16_;

static gemm_f32x16_ gemm_splat16_(float x) __attribute__((always_inline)) {
    gemm_f32x16_ v;
    v[0] = x; v[1] = x; v[2] = x; v[3] = x; v[4] = x; v[5] = x; v[6] = x; v[7] = x;
    v[8] = x; v[9] = x; v[10] = x; v[11] = x; v[12] = x; v[13] = x; v[14] = x; v[15] = x;
    return v;
}

static void gemm_k_avx_(const float* ap, const float* bp, unsigned long kc, float* cp, unsigned long ldc)
    __attribute__((target("avx"))) {
    gemm_kernel4_(ap, bp, kc, cp, ldc);
}

static void gemm_k_avx2_(const float* ap, const float* bp, unsigned long kc, float* cp, unsigned long ldc)
    __attribute__((target("avx2,fma"))) {
    gemm_kernel4_(ap, bp, kc, cp, ldc);
}

// AVX-512: cp[0..8)[0..16) += ap (kc x 8 strip) * bp (kc x 16 strip), one
// zmm per row, eight accumulators.
static void gemm_k_avx512_(const float* ap, const float* bp, unsigned long kc, float* cp, unsigned long ldc)
    __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx2,fma"))) {
    gemm_f32x16_ c0; gemm_f32x16_ c1; gemm_f32x16_ c2; gemm_f32x16_ c3;
    gemm_f32x16_ c4; gemm_f32x16_ c5; gemm_f32x16_ c6; gemm_f32x16_ c7;
    unsafe {
        unsigned long p = 0UL;
        while (p < kc) {
            gemm_f32x16_ b = *(const gemm_f32x16_*)&bp[0];
            c0 = __builtin_fmuladd(gemm_splat16_(ap[0]), b, c0);
            c1 = __builtin_fmuladd(gemm_splat16_(ap[1]), b, c1);
            c2 = __builtin_fmuladd(gemm_splat16_(ap[2]), b, c2);
            c3 = __builtin_fmuladd(gemm_splat16_(ap[3]), b, c3);
            c4 = __builtin_fmuladd(gemm_splat16_(ap[4]), b, c4);
            c5 = __builtin_fmuladd(gemm_splat16_(ap[5]), b, c5);
            c6 = __builtin_fmuladd(gemm_splat16_(ap[6]), b, c6);
            c7 = __builtin_fmuladd(gemm_splat16_(ap[7]), b, c7);
            ap = ap + 8;
            bp = bp + 16;
            p = p + 1UL;
        }
        float* r = cp;
        *(gemm_f32x16_*)&r[0] = *(gemm_f32x16_*)&r[0] + c0; r = r + ldc;
        *(gemm_f32x16_*)&r[0] = *(gemm_f32x16_*)&r[0] + c1; r = r + ldc;
        *(gemm_f32x16_*)&r[0] = *(gemm_f32x16_*)&r[0] + c2; r = r + ldc;
        *(gemm_f32x16_*)&r[0] = *(gemm_f32x16_*)&r[0] + c3; r = r + ldc;
        *(gemm_f32x16_*)&r[0] = *(gemm_f32x16_*)&r[0] + c4; r = r + ldc;
        *(gemm_f32x16_*)&r[0] = *(gemm_f32x16_*)&r[0] + c5; r = r + ldc;
        *(gemm_f32x16_*)&r[0] = *(gemm_f32x16_*)&r[0] + c6; r = r + ldc;
        *(gemm_f32x16_*)&r[0] = *(gemm_f32x16_*)&r[0] + c7;
    }
}
#endif

#define GEMM_K_SSE    0
#define GEMM_K_AVX    1
#define GEMM_K_AVX2   2
#define GEMM_K_AVX512 3

// The kernel for this CPU (std/simd/cpu.h), and its strip height.
static int gemm_pick_kernel_() {
#if defined(__x86_64__)
    int l = simd_level();
    if (l >= SIMD_AVX512) return GEMM_K_AVX512;
    if (l >= SIMD_AVX2) return GEMM_K_AVX2;
    if (l >= SIMD_AVX) return GEMM_K_AVX;
#endif
    return GEMM_K_SSE;
}

static void gemm_kernel_(int kern, const float* ap, const float* bp, unsigned long kc, float* cp, unsigned long ldc) {
#if defined(__x86_64__)
    if (kern == GEMM_K_AVX512) { gemm_k_avx512_(ap, bp, kc, cp, ldc); return; }
    if (kern == GEMM_K_AVX2) { gemm_k_avx2_(ap, bp, kc, cp, ldc); return; }
    if (kern == GEMM_K_AVX) { gemm_k_avx_(ap, bp, kc, cp, ldc); return; }
#endif
    gemm_k_sse_(ap, bp, kc, cp, ldc);
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
    int           vendor;   // tiles go to the vendor BLAS
    int           kern;     // GEMM_K_*: built-in kernel for this CPU
    unsigned long mr;       // its strip height: 4, or 8 for AVX-512
};

// ── Vendor BLAS ──────────────────────────────────────────────────────────────
// On hosted targets, sgemm hands large products to the CPU vendor's BLAS
// when the program links one: AOCL-BLAS on AMD, oneMKL on Intel, Accelerate
// on Apple (safeguard's "blas" feature links the right one; any other CBLAS
// works too). All export the same cblas_sgemm. The reference is weak, so
// without a BLAS it is null and the built-in kernel runs; which one is used
// is fixed when the program is linked.
#if defined(__SAFEC_FREESTANDING__) || defined(__wasm__) || defined(_WIN32)
#define GEMM_VENDOR 0
#else
#define GEMM_VENDOR 1
#define GEMM_ROW_MAJOR 101   // CblasRowMajor
#define GEMM_NOTRANS   111   // CblasNoTrans
#define GEMM_TRANSP    112   // CblasTrans
extern void cblas_sgemm(int order, int transA, int transB, int m, int n, int k, float alpha,
                        const float* a, int lda, const float* b, int ldb, float beta,
                        float* c, int ldc) __attribute__((weak));
// Set to one thread on first use (see gemm_vendor_ready_).
extern void MKL_Set_Num_Threads(int n) __attribute__((weak));     // oneMKL C API (mkl_set_num_threads is a macro for it)
extern void bli_thread_set_num_threads(long n) __attribute__((weak));
extern void openblas_set_num_threads(int n) __attribute__((weak));
// One symbol only each library exports, to name it.
extern const char* bli_info_get_version_str() __attribute__((weak));     // BLIS / AOCL-BLAS
extern void MKL_Get_Version_String(char* buf, int len) __attribute__((weak));  // oneMKL
extern char* openblas_get_config() __attribute__((weak));                 // OpenBLAS
#endif
// Below this many multiply-adds the built-in kernel is as fast as a call
// into the library.
#define GEMM_VENDOR_MIN 32768UL
// Starting tile size when the library computes the tiles.
#define GEMM_VENDOR_TILE 512UL

// The libraries thread through OpenMP, whose idle workers busy-wait for a
// while after each call -- and then compete with this thread pool for the
// same cores (a U-Net forward pass ran 6x slower that way). So outside
// Apple, std runs the library on one thread and does the threading itself:
// sgemm splits C into tiles on std/sync/parallel.h's pool, and each tile is
// one single-threaded library call. Accelerate (no OpenMP) gets whole calls.
// Note this also makes the library single-threaded for the program's own
// direct calls into it.
static int g_gemm_vendor_state = 0;   // 0 = not yet, 1 = initializing, 2 = ready

static int gemm_vendor_ready_() {
#if GEMM_VENDOR
    unsafe {
        if ((void*)cblas_sgemm == (void*)0) return 0;
        if (atomic_load_int(&g_gemm_vendor_state) == 2) return 1;
        int expected = 0;
        if (atomic_cas_int(&g_gemm_vendor_state, &expected, 1)) {
#ifndef __APPLE__
            if ((void*)MKL_Set_Num_Threads != (void*)0) MKL_Set_Num_Threads(1);
            if ((void*)bli_thread_set_num_threads != (void*)0) bli_thread_set_num_threads(1L);
            if ((void*)openblas_set_num_threads != (void*)0) openblas_set_num_threads(1);
#endif
            atomic_store_int(&g_gemm_vendor_state, 2);
        } else {
            while (atomic_load_int(&g_gemm_vendor_state) != 2) { thread_yield(); }
        }
        return 1;
    }
#else
    return 0;
#endif
}

static int gemm_vendor_linked_() {
#if GEMM_VENDOR
    unsafe { return (void*)cblas_sgemm != (void*)0; }
#else
    return 0;
#endif
}

int sgemm_vendor() { return gemm_vendor_linked_(); }

// One library call for the C tile [i0, i0+mc) x [j0, j0+nc).
static void gemm_vendor_tile_(const &GemmJob j, unsigned long i0, unsigned long mc,
                              unsigned long j0, unsigned long nc) {
#if GEMM_VENDOR
    unsafe {
        const float* a = j.transA ? &j.a[i0] : &j.a[i0 * j.lda];
        const float* b = j.transB ? &j.b[j0 * j.ldb] : &j.b[j0];
        cblas_sgemm(GEMM_ROW_MAJOR, j.transA ? GEMM_TRANSP : GEMM_NOTRANS, j.transB ? GEMM_TRANSP : GEMM_NOTRANS,
                    (int)mc, (int)nc, (int)j.k, j.alpha, a, (int)j.lda, b, (int)j.ldb,
                    j.beta, &j.c[i0 * j.ldc + j0], (int)j.ldc);
    }
#endif
}

const char* sgemm_backend() {
#if GEMM_VENDOR
    if (!gemm_vendor_linked_()) return "built-in";
    unsafe {
#ifdef __APPLE__
        return "Accelerate";
#else
        if ((void*)MKL_Get_Version_String != (void*)0) return "oneMKL";
        if ((void*)bli_info_get_version_str != (void*)0) return "AOCL-BLAS (BLIS)";
        if ((void*)openblas_get_config != (void*)0) return "OpenBLAS";
        return "CBLAS";
#endif
    }
#else
    return "built-in";
#endif
}

// Packs op(A)[i0 .. i0+mc, p0 .. p0+kc) * alpha into MR-row strips.
static void gemm_pack_a_(const &GemmJob j, float* dst, unsigned long i0, unsigned long mc,
                         unsigned long p0, unsigned long kc) {
    unsafe {
        unsigned long s = 0UL;
        while (s < mc) {
            unsigned long p = 0UL;
            while (p < kc) {
                unsigned long r = 0UL;
                while (r < j.mr) {
                    float v = (float)0.0;
                    if (s + r < mc) {
                        unsigned long i = i0 + s + r;
                        unsigned long q = p0 + p;
                        v = j.transA ? j.a[q * j.lda + i] : j.a[i * j.lda + q];
                    }
                    dst[r] = v * j.alpha;
                    r = r + 1UL;
                }
                dst = dst + j.mr;
                p = p + 1UL;
            }
            s = s + j.mr;
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
        float edge[128];                     // up to 8 x GEMM_NR partial tile
        unsigned long t = lo;
        while (t < hi) {
            unsigned long i0 = (t / j.tilesN) * j.mc;
            unsigned long j0 = (t % j.tilesN) * j.nc;
            unsigned long mc = j.m - i0 < j.mc ? j.m - i0 : j.mc;
            unsigned long nc = j.n - j0 < j.nc ? j.n - j0 : j.nc;
            if (j.vendor) {
                gemm_vendor_tile_(j, i0, mc, j0, nc);
                t = t + 1UL;
                continue;
            }

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
                        unsigned long h = mc - sa < j.mr ? mc - sa : j.mr;
                        const float* ap = &apack[sa * kc];
                        float* cp = &j.c[(i0 + sa) * j.ldc + j0 + sb];
                        if (h == j.mr && w == GEMM_NR) {
                            gemm_kernel_(j.kern, ap, bp, kc, cp, j.ldc);
                        } else {
                            unsigned long e = 0UL;
                            while (e < 128UL) { edge[e] = (float)0.0; e = e + 1UL; }
                            gemm_kernel_(j.kern, ap, bp, kc, &edge[0], GEMM_NR);
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
                        sa = sa + j.mr;
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
    int vendor = m * n * k >= GEMM_VENDOR_MIN && k > 0UL &&
                 m <= 2147483647UL && n <= 2147483647UL && k <= 2147483647UL &&
                 lda <= 2147483647UL && ldb <= 2147483647UL && ldc <= 2147483647UL &&
                 gemm_vendor_ready_();
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
    j.vendor = vendor;
    j.kern = gemm_pick_kernel_();
    j.mr = j.kern == GEMM_K_AVX512 ? 8UL : GEMM_MR;
    j.mc = vendor ? GEMM_VENDOR_TILE : GEMM_MC;
    j.nc = vendor ? GEMM_VENDOR_TILE : GEMM_NC;
    int par = threaded && m * n * k >= GEMM_PAR_MIN;
#ifdef __APPLE__
    if (vendor) par = 0;   // Accelerate threads (and uses the AMX units) itself
#endif
    if (vendor && !par) { j.mc = m; j.nc = n; }
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
