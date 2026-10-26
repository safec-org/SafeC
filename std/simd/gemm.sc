// SafeC Standard Library — packed SGEMM (see gemm.h).
//
// C is cut into MC x NC tiles, each one task for the thread pool. A task
// walks k in KC-deep slices and runs the MR x NR kernel over every pair of
// an MR-row strip of op(A) (alpha folded in) and an NR-column strip of
// op(B), both packed contiguously and zero-padded to whole strips. A packed
// A slice (MC x KC) stays in L2 across the B strips; a B strip (KC x NR)
// stays in L1 across the A strips.
//
// When both packed operands fit in GEMM_PREPACK_MAX floats, they are packed
// once per call, strips split across the pool, and every tile then reads
// them; otherwise each tile packs its own slices. (Packing per tile packs B
// once per tile row and A once per tile column, which for the mid-sized
// products of training -- 128 x 256 x 784 split into 16 tiles -- cost about
// as much as the multiply.)
// safeguard: link hosted blas            (cblas_sgemm, weak: the [build] blas choice)
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
// Up to this many floats (A and B packed together, 16 MB) are packed once
// per call instead of per tile.
#define GEMM_PREPACK_MAX 4194304UL

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
    float*        apk;      // packed op(A), all of it (prepacked path), else null
    float*        bpk;      // packed op(B), all of it (prepacked path), else null
    unsigned long mp;       // m rounded up to whole MR strips
    unsigned long np;       // n rounded up to whole NR strips
    unsigned long slices;   // KC-deep slices of k
    unsigned long aUnits;   // A strips per slice
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
        if ((unsigned char*)cblas_sgemm == (unsigned char*)0) return 0;
        if (atomic_load_int(&g_gemm_vendor_state) == 2) return 1;
        int expected = 0;
        if (atomic_cas_int(&g_gemm_vendor_state, &expected, 1)) {
#ifndef __APPLE__
            if ((unsigned char*)MKL_Set_Num_Threads != (unsigned char*)0) MKL_Set_Num_Threads(1);
            if ((unsigned char*)bli_thread_set_num_threads != (unsigned char*)0) bli_thread_set_num_threads(1L);
            if ((unsigned char*)openblas_set_num_threads != (unsigned char*)0) openblas_set_num_threads(1);
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
    unsafe { return (unsigned char*)cblas_sgemm != (unsigned char*)0; }
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

const &char sgemm_backend() {
#if GEMM_VENDOR
    if (!gemm_vendor_linked_()) return "built-in";
    unsafe {
#ifdef __APPLE__
        return "Accelerate";
#else
        if ((unsigned char*)MKL_Get_Version_String != (unsigned char*)0) return "oneMKL";
        if ((unsigned char*)bli_info_get_version_str != (unsigned char*)0) return "AOCL-BLAS (BLIS)";
        if ((unsigned char*)openblas_get_config != (unsigned char*)0) return "OpenBLAS";
        return "CBLAS";
#endif
    }
#else
    return "built-in";
#endif
}

// Packs one MR-row strip of op(A) -- rows i .. i+h (h <= mr, the rest
// zero), columns p0 .. p0+kc -- times alpha: dst[p * mr + r]. Each loop
// reads memory contiguously.
static void gemm_pack_a_strip_(const &GemmJob j, float* dst, unsigned long i, unsigned long h,
                               unsigned long p0, unsigned long kc) {
    unsafe {
        unsigned long mr = j.mr;
        float al = j.alpha;
        if (j.transA) {
            // op(A)[i + r][p0 + p] = a[(p0 + p) * lda + i + r]: rows of a are contiguous in r
            unsigned long p = 0UL;
            while (p < kc) {
                const float* src = &j.a[(p0 + p) * j.lda + i];
                float* d = &dst[p * mr];
                unsigned long r = 0UL;
                while (r < h) { d[r] = src[r] * al; r = r + 1UL; }
                while (r < mr) { d[r] = (float)0.0; r = r + 1UL; }
                p = p + 1UL;
            }
        } else {
            unsigned long r = 0UL;
            while (r < h) {
                const float* src = &j.a[(i + r) * j.lda + p0];
                unsigned long p = 0UL;
                while (p < kc) { dst[p * mr + r] = src[p] * al; p = p + 1UL; }
                r = r + 1UL;
            }
            while (r < mr) {
                unsigned long p = 0UL;
                while (p < kc) { dst[p * mr + r] = (float)0.0; p = p + 1UL; }
                r = r + 1UL;
            }
        }
    }
}

// Packs one NR-column strip of op(B) -- columns c0 .. c0+w (w <= NR, the
// rest zero), rows p0 .. p0+kc: dst[p * NR + t].
static void gemm_pack_b_strip_(const &GemmJob j, float* dst, unsigned long c0, unsigned long w,
                               unsigned long p0, unsigned long kc) {
    unsafe {
        if (!j.transB) {
            unsigned long p = 0UL;
            while (p < kc) {
                const float* src = &j.b[(p0 + p) * j.ldb + c0];
                float* d = &dst[p * GEMM_NR];
                if (w == GEMM_NR) {
                    *(f32x8*)&d[0] = *(const f32x8*)&src[0];
                    *(f32x8*)&d[8] = *(const f32x8*)&src[8];
                } else {
                    unsigned long t = 0UL;
                    while (t < w) { d[t] = src[t]; t = t + 1UL; }
                    while (t < GEMM_NR) { d[t] = (float)0.0; t = t + 1UL; }
                }
                p = p + 1UL;
            }
        } else {
            // op(B)[p0 + p][c0 + t] = b[(c0 + t) * ldb + p0 + p]: rows of b are contiguous in p
            unsigned long t = 0UL;
            while (t < w) {
                const float* src = &j.b[(c0 + t) * j.ldb + p0];
                unsigned long p = 0UL;
                while (p < kc) { dst[p * GEMM_NR + t] = src[p]; p = p + 1UL; }
                t = t + 1UL;
            }
            while (t < GEMM_NR) {
                unsigned long p = 0UL;
                while (p < kc) { dst[p * GEMM_NR + t] = (float)0.0; p = p + 1UL; }
                t = t + 1UL;
            }
        }
    }
}

// Packs op(A)[i0 .. i0+mc, p0 .. p0+kc) * alpha into MR-row strips.
static void gemm_pack_a_(const &GemmJob j, float* dst, unsigned long i0, unsigned long mc,
                         unsigned long p0, unsigned long kc) {
    unsigned long s = 0UL;
    while (s < mc) {
        unsigned long h = mc - s < j.mr ? mc - s : j.mr;
        unsafe { gemm_pack_a_strip_(j, &dst[s * kc], i0 + s, h, p0, kc); }
        s = s + j.mr;
    }
}

// Packs op(B)[p0 .. p0+kc, j0 .. j0+nc) into NR-column strips.
static void gemm_pack_b_(const &GemmJob j, float* dst, unsigned long p0, unsigned long kc,
                         unsigned long j0, unsigned long nc) {
    unsigned long s = 0UL;
    while (s < nc) {
        unsigned long w = nc - s < GEMM_NR ? nc - s : GEMM_NR;
        unsafe { gemm_pack_b_strip_(j, &dst[s * kc], j0 + s, w, p0, kc); }
        s = s + GEMM_NR;
    }
}

// Prepacked path: unit u < slices * aUnits is A strip (u % aUnits) of slice
// (u / aUnits); the rest are B strips. Slice q starts at element mp * q * KC
// of apk (np * q * KC of bpk), and a strip there is kc * mr (kc * NR) long.
static void gemm_pack_units_(&GemmJob j, unsigned long lo, unsigned long hi) {
    unsigned long u = lo;
    unsigned long bStrips = j.np / GEMM_NR;
    while (u < hi) {
        unsigned long aTotal = j.slices * j.aUnits;
        if (u < aTotal) {
            unsigned long q = u / j.aUnits;
            unsigned long st = u % j.aUnits;
            unsigned long p0 = q * GEMM_KC;
            unsigned long kc = j.k - p0 < GEMM_KC ? j.k - p0 : GEMM_KC;
            unsigned long i = st * j.mr;
            unsigned long h = j.m - i < j.mr ? j.m - i : j.mr;
            unsafe { gemm_pack_a_strip_(j, &j.apk[j.mp * p0 + st * j.mr * kc], i, h, p0, kc); }
        } else {
            unsigned long v = u - aTotal;
            unsigned long q = v / bStrips;
            unsigned long st = v % bStrips;
            unsigned long p0 = q * GEMM_KC;
            unsigned long kc = j.k - p0 < GEMM_KC ? j.k - p0 : GEMM_KC;
            unsigned long c0 = st * GEMM_NR;
            unsigned long w = j.n - c0 < GEMM_NR ? j.n - c0 : GEMM_NR;
            unsafe { gemm_pack_b_strip_(j, &j.bpk[j.np * p0 + st * GEMM_NR * kc], c0, w, p0, kc); }
        }
        u = u + 1UL;
    }
}

// The packed buffer for the prepacked path, kept per thread (the caller's)
// and grown as needed.
#if defined(__SAFEC_FREESTANDING__) || defined(__wasm__)
static float* g_gemm_pack = (float*)0;
static unsigned long g_gemm_pack_cap = 0UL;
#else
static thread_local float* g_gemm_pack = (float*)0;
static thread_local unsigned long g_gemm_pack_cap = 0UL;
#endif

static float* gemm_pack_buf_(unsigned long floats) {
    if (floats > g_gemm_pack_cap) {
        unsafe {
            if (g_gemm_pack != (float*)0) dealloc(g_gemm_pack);
            g_gemm_pack = (float*)alloc_raw_(sizeof(float) * floats);
        }
        if (g_gemm_pack == (float*)0) panic_at("sgemm: out of memory", "gemm.sc", __LINE__);
        g_gemm_pack_cap = floats;
    }
    return g_gemm_pack;
}

static void gemm_tiles_(&GemmJob j, unsigned long lo, unsigned long hi) {
    unsafe {
        float* apack = (float*)0;
        float* bpack = (float*)0;
        if (j.apk == (float*)0) {
            apack = gemm_scratch_();
            bpack = &apack[GEMM_MC * GEMM_KC];
        }
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
                const float* abase;
                const float* bbase;
                if (j.apk != (float*)0) {
                    // strips of this tile inside the call's packed operands
                    abase = (const float*)&j.apk[j.mp * p0 + i0 * kc];
                    bbase = (const float*)&j.bpk[j.np * p0 + j0 * kc];
                } else {
                    gemm_pack_a_(j, apack, i0, mc, p0, kc);
                    gemm_pack_b_(j, bpack, p0, kc, j0, nc);
                    abase = (const float*)apack;
                    bbase = (const float*)bpack;
                }
                unsigned long sb = 0UL;
                while (sb < nc) {
                    unsigned long w = nc - sb < GEMM_NR ? nc - sb : GEMM_NR;
                    const float* bp = &bbase[sb * kc];
                    unsigned long sa = 0UL;
                    while (sa < mc) {
                        unsigned long h = mc - sa < j.mr ? mc - sa : j.mr;
                        const float* ap = &abase[sa * kc];
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
    j.mp = (m + j.mr - 1UL) / j.mr * j.mr;
    j.np = (n + GEMM_NR - 1UL) / GEMM_NR * GEMM_NR;
    j.slices = (k + GEMM_KC - 1UL) / GEMM_KC;
    j.aUnits = j.mp / j.mr;
    unsafe { j.apk = (float*)0; j.bpk = (float*)0; }
    // Pack once per call when more than one tile would repack the same strips.
    if (!vendor && k > 0UL && tiles > 1UL && (j.mp + j.np) * k <= GEMM_PREPACK_MAX) {
        float* buf = gemm_pack_buf_((j.mp + j.np) * k);
        unsafe { j.apk = buf; j.bpk = &buf[j.mp * k]; }
        unsigned long units = j.slices * (j.aUnits + j.np / GEMM_NR);
        if (par && units > 1UL) parallel_for(units, 4UL, gemm_pack_units_, &j);
        else gemm_pack_units_(&j, 0UL, units);
    }
    if (par && tiles > 1UL)
        parallel_for(tiles, 1UL, gemm_tiles_, &j);
    else
        gemm_tiles_(&j, 0UL, tiles);
}

void sgemm_n_(int transA, int transB, unsigned long m, unsigned long n, unsigned long k,
             float alpha, const &float a, unsigned long lda,
             const &float b, unsigned long ldb,
             float beta, &float c, unsigned long ldc) {
    gemm_run_(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 1);
}

void sgemm_serial_n_(int transA, int transB, unsigned long m, unsigned long n, unsigned long k,
                     float alpha, const &float a, unsigned long lda,
                     const &float b, unsigned long ldb,
                     float beta, &float c, unsigned long ldc) {
    gemm_run_(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 0);
}

// Elements a rows x cols matrix with leading dimension ld spans (0 if empty).
static unsigned long gemm_span_(unsigned long rows, unsigned long cols, unsigned long ld) {
    if (rows == 0UL || cols == 0UL) { return 0UL; }
    return (rows - 1UL) * ld + cols;
}

static int gemm_fits_(int transA, int transB, unsigned long m, unsigned long n, unsigned long k,
                      unsigned long aLen, unsigned long lda, unsigned long bLen, unsigned long ldb,
                      unsigned long cLen, unsigned long ldc) {
    unsigned long aRows = transA == SGEMM_TRANS ? k : m;
    unsigned long aCols = transA == SGEMM_TRANS ? m : k;
    unsigned long bRows = transB == SGEMM_TRANS ? n : k;
    unsigned long bCols = transB == SGEMM_TRANS ? k : n;
    if (lda < aCols || ldb < bCols || ldc < n) { return 0; }
    return aLen >= gemm_span_(aRows, aCols, lda) && bLen >= gemm_span_(bRows, bCols, ldb)
        && cLen >= gemm_span_(m, n, ldc);
}

int sgemm(int transA, int transB, unsigned long m, unsigned long n, unsigned long k,
          float alpha, []const float a, unsigned long lda,
          []const float b, unsigned long ldb,
          float beta, []float c, unsigned long ldc) {
    if (!gemm_fits_(transA, transB, m, n, k, a.len, lda, b.len, ldb, c.len, ldc)) { return 0; }
    if (m == 0UL || n == 0UL) { return 1; }
    float zero = (float)0;
    unsafe {
        const &float ar = k == 0UL ? (const &float)&zero : (const &float)a.ptr;
        const &float br = k == 0UL ? (const &float)&zero : (const &float)b.ptr;
        sgemm_n_(transA, transB, m, n, k, alpha, ar, lda, br, ldb, beta, (&float)c.ptr, ldc);
    }
    return 1;
}

int sgemm_serial(int transA, int transB, unsigned long m, unsigned long n, unsigned long k,
                 float alpha, []const float a, unsigned long lda,
                 []const float b, unsigned long ldb,
                 float beta, []float c, unsigned long ldc) {
    if (!gemm_fits_(transA, transB, m, n, k, a.len, lda, b.len, ldb, c.len, ldc)) { return 0; }
    if (m == 0UL || n == 0UL) { return 1; }
    float zero = (float)0;
    unsafe {
        const &float ar = k == 0UL ? (const &float)&zero : (const &float)a.ptr;
        const &float br = k == 0UL ? (const &float)&zero : (const &float)b.ptr;
        sgemm_serial_n_(transA, transB, m, n, k, alpha, ar, lda, br, ldb, beta, (&float)c.ptr, ldc);
    }
    return 1;
}

} // namespace std
