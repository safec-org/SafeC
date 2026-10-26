// SafeC Standard Library — fp16/bf16 conversion implementation (see
// float16.h). Bit-level float32 access goes through memcpy into/out of an
// unsigned int, the same reinterpret-via-memcpy pattern gpu_webgpu.sc
// already uses for packing scalars into a push-constant/uniform buffer --
// avoids relying on pointer-punning through mismatched types, which
// safec (like C) doesn't guarantee is well-defined.
#pragma once
#include <std/stddef.h>
#include <std/ml/float16.h>
#include <std/sync/parallel.h>
#include <std/sys.h>
#include <std/str.h>
#include <std/atomic.h>

namespace std {

extern unsigned char* memcpy(unsigned char* dst, const unsigned char* src, size_t n);

static unsigned int __f32_bits(float v) {
    unsigned int bits = 0U;
    unsafe { memcpy((unsigned char*)&bits, (const unsigned char*)&v, 4UL); }
    return bits;
}

static float __bits_f32(unsigned int bits) {
    float v = 0.0f;
    unsafe { memcpy((unsigned char*)&v, (const unsigned char*)&bits, 4UL); }
    return v;
}

unsigned short f32_to_fp16(float v) {
    unsigned int bits = __f32_bits(v);
    unsigned int sign = (bits >> 16) & 0x8000U;
    unsigned int expField = (bits >> 23) & 0xFFU;
    unsigned int mant = bits & 0x7FFFFFU;
    int exp = (int)expField - 127 + 15; // rebias float32's exponent into fp16's

    if (expField == 0xFFU) {
        // Infinity (mant == 0) or NaN (mant != 0) -- force a nonzero
        // mantissa bit for NaN so it can't collapse into infinity.
        unsigned int m = (mant != 0U) ? 0x200U : 0U;
        return (unsigned short)(sign | 0x7C00U | m);
    }
    if (exp >= 0x1F) {
        // Overflow -- magnitude too large for fp16, rounds to infinity.
        return (unsigned short)(sign | 0x7C00U);
    }
    if (exp <= 0) {
        // Subnormal (or underflows to zero) in fp16.
        if (exp < -10) {
            return (unsigned short)sign; // too small to round up to even the smallest subnormal
        }
        unsigned int sig = mant | 0x800000U; // restore the implicit leading 1 -> 24-bit significand
        unsigned int shift = (unsigned int)(14 - exp);
        unsigned int roundedMant = sig >> shift;
        unsigned int remainder = sig & ((1U << shift) - 1U);
        unsigned int halfway = 1U << (shift - 1U);
        if (remainder > halfway || (remainder == halfway && (roundedMant & 1U) == 1U)) {
            roundedMant = roundedMant + 1U;
        }
        return (unsigned short)(sign | roundedMant);
    }

    // Normal range: round the 23-bit mantissa down to 10 bits,
    // round-to-nearest-even, with carry-out bumping the exponent (and
    // possibly overflowing to infinity) on a mantissa round-up to 0x400.
    unsigned int roundedMant = mant >> 13;
    unsigned int remainder = mant & 0x1FFFU;
    unsigned int halfway = 0x1000U;
    if (remainder > halfway || (remainder == halfway && (roundedMant & 1U) == 1U)) {
        roundedMant = roundedMant + 1U;
        if (roundedMant == 0x400U) {
            roundedMant = 0U;
            exp = exp + 1;
            if (exp >= 0x1F) {
                return (unsigned short)(sign | 0x7C00U);
            }
        }
    }
    return (unsigned short)(sign | ((unsigned int)exp << 10) | roundedMant);
}

float fp16_to_f32(unsigned short h) {
    unsigned int sign = ((unsigned int)h & 0x8000U) << 16;
    unsigned int exp = ((unsigned int)h >> 10) & 0x1FU;
    unsigned int mant = (unsigned int)h & 0x3FFU;

    if (exp == 0U) {
        if (mant == 0U) {
            return __bits_f32(sign); // signed zero
        }
        // Subnormal: normalize by left-shifting until the implicit
        // leading bit (bit 10) is set, tracking the shift count so the
        // float32 exponent can be rebased to match.
        unsigned int m = mant;
        unsigned int s = 0U;
        while ((m & 0x400U) == 0U) {
            m = m << 1;
            s = s + 1U;
        }
        m = m & 0x3FFU; // drop the now-explicit leading bit
        unsigned int outExp = 113U - s; // = (-14 - 1 - s) rebiased to float32's 127 bias -- see float16.sc's derivation
        unsigned int outMant = m << 13;
        return __bits_f32(sign | (outExp << 23) | outMant);
    }
    if (exp == 0x1FU) {
        return __bits_f32(sign | 0x7F800000U | (mant << 13)); // infinity or NaN
    }
    unsigned int outExp = exp - 15U + 127U;
    unsigned int outMant = mant << 13;
    return __bits_f32(sign | (outExp << 23) | outMant);
}

unsigned short f32_to_bf16(float v) {
    unsigned int bits = __f32_bits(v);
    unsigned int expField = (bits >> 23) & 0xFFU;
    if (expField == 0xFFU && (bits & 0x7FFFFFU) != 0U) {
        // NaN -- truncating could zero out the mantissa entirely (turning
        // it into infinity), so force one mantissa bit set.
        unsigned int top = (bits >> 16) & 0xFFFFU;
        return (unsigned short)(top | 0x0040U);
    }
    unsigned int roundBit = (bits >> 15) & 1U;
    unsigned int sticky = ((bits & 0x7FFFU) != 0U) ? 1U : 0U;
    unsigned int lsb = (bits >> 16) & 1U;
    unsigned int roundUp = (roundBit != 0U && (sticky != 0U || lsb != 0U)) ? 1U : 0U;
    unsigned int result = (bits >> 16) + roundUp; // plain integer add -- a mantissa carry correctly ripples into the (shared-width) exponent field
    return (unsigned short)(result & 0xFFFFU);
}

float bf16_to_f32(unsigned short b) {
    unsigned int bits = ((unsigned int)b) << 16;
    return __bits_f32(bits);
}

// ══ CPU features ═════════════════════════════════════════════════════════════

static int g_half_caps = -1;   // -1 = not yet detected (benign race: same answer)

#if defined(__x86_64__) && !defined(__SAFEC_FREESTANDING__)
static void hc_cpuid_(unsigned int leaf, unsigned int sub, &unsigned int a, &unsigned int b,
                      &unsigned int c, &unsigned int d) {
    unsigned int ra = 0U; unsigned int rb = 0U; unsigned int rc = 0U; unsigned int rd = 0U;
    unsafe { asm volatile ("cpuid" : "=a"(ra), "=b"(rb), "=c"(rc), "=d"(rd) : "a"(leaf), "c"(sub)); }
    *a = ra; *b = rb; *c = rc; *d = rd;
}

static int half_detect_() {
    unsigned int a = 0U; unsigned int b = 0U; unsigned int c = 0U; unsigned int d = 0U;
    hc_cpuid_(0U, 0U, &a, &b, &c, &d);
    unsigned int maxLeaf = a;
    hc_cpuid_(1U, 0U, &a, &b, &c, &d);
    // F16C runs on YMM state: needs AVX, OSXSAVE and the OS saving YMM (XCR0)
    if ((c & (1U << 28)) == 0U || (c & (1U << 27)) == 0U) return 0;
    unsigned int f16c = c & (1U << 29);
    unsigned int xlo = 0U; unsigned int xhi = 0U; unsigned int zero = 0U;
    unsafe { asm volatile ("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(zero)); }
    if ((xlo & 6U) != 6U) return 0;
    int caps = f16c != 0U ? HALF_CPU_F16_CONVERT : 0;
    if (maxLeaf < 7U) return caps;
    hc_cpuid_(7U, 0U, &a, &b, &c, &d);
    unsigned int need512 = (1U << 16) | (1U << 17) | (1U << 30) | (1U << 31);   // F, DQ, BW, VL
    int avx512 = (b & need512) == need512 && (xlo & 0xE6U) == 0xE6U;
    if (!avx512) return caps;
    if ((d & (1U << 23)) != 0U) caps = caps | HALF_CPU_F16_ARITH;               // AVX512-FP16
    hc_cpuid_(7U, 1U, &a, &b, &c, &d);
    if ((a & (1U << 5)) != 0U) caps = caps | HALF_CPU_BF16_CONVERT | HALF_CPU_BF16_DOT;   // AVX512-BF16
    return caps;
}
#elif defined(__aarch64__) && defined(__linux__) && !defined(__SAFEC_FREESTANDING__)
extern unsigned long getauxval(unsigned long type);
static int half_detect_() {
    unsigned long hw = 0UL;
    unsigned long hw2 = 0UL;
    unsafe { hw = getauxval(16UL); hw2 = getauxval(26UL); }          // AT_HWCAP, AT_HWCAP2
    int caps = HALF_CPU_F16_CONVERT;                                  // FCVTL / FCVTN: base AArch64
    if ((hw & (1UL << 9)) != 0UL && (hw & (1UL << 10)) != 0UL) caps = caps | HALF_CPU_F16_ARITH;   // FPHP, ASIMDHP
    if ((hw2 & (1UL << 14)) != 0UL) caps = caps | HALF_CPU_BF16_CONVERT | HALF_CPU_BF16_DOT;      // HWCAP2_BF16
    return caps;
}
#elif defined(__aarch64__) && defined(__APPLE__) && !defined(__SAFEC_FREESTANDING__)
extern int sysctlbyname(const char* name, int* oldp, size_t* oldlenp, int* newp, size_t newlen);   // an int value
static int half_sysctl_(const char* name) {
    int v = 0;
    size_t len = sizeof(int);
    unsafe { if (sysctlbyname(name, &v, &len, (int*)0, 0UL) != 0) return 0; }
    return v != 0;
}
static int half_detect_() {
    int caps = HALF_CPU_F16_CONVERT | HALF_CPU_F16_ARITH;           // every Apple core has FEAT_FP16
    if (half_sysctl_("hw.optional.arm.FEAT_BF16")) caps = caps | HALF_CPU_BF16_CONVERT | HALF_CPU_BF16_DOT;
    return caps;
}
#else
static int half_detect_() {
    int caps = 0;
#if defined(__aarch64__)
    caps = HALF_CPU_F16_CONVERT;
#endif
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    caps = caps | HALF_CPU_F16_ARITH;
#endif
#if defined(__ARM_FEATURE_BF16)
    caps = caps | HALF_CPU_BF16_CONVERT | HALF_CPU_BF16_DOT;
#endif
    return caps;
}
#endif

int half_cpu_features() {
    int c = atomic_load_int(&g_half_caps);
    if (c >= 0) return c;
    c = half_detect_();
#ifndef __SAFEC_FREESTANDING__
    match (sys_getenv("SAFEC_HALF")) {
        case some(e): { if (str_eq(e, "soft")) c = 0; }
        case null: { }
    }
#endif
    atomic_store_int(&g_half_caps, c);
    return c;
}

const &char half_cpu_features_name() {
    int c = half_cpu_features();
#if defined(__x86_64__)
    if ((c & HALF_CPU_F16_ARITH) != 0 && (c & HALF_CPU_BF16_CONVERT) != 0) return "F16C, AVX512-FP16, AVX512-BF16";
    if ((c & HALF_CPU_BF16_CONVERT) != 0) return (c & HALF_CPU_F16_CONVERT) != 0 ? "F16C, AVX512-BF16" : "AVX512-BF16";
    if ((c & HALF_CPU_F16_ARITH) != 0) return "F16C, AVX512-FP16";
    if ((c & HALF_CPU_F16_CONVERT) != 0) return "F16C";
#else
    if ((c & HALF_CPU_F16_ARITH) != 0 && (c & HALF_CPU_BF16_CONVERT) != 0) return "FP16 conversion, FEAT_FP16, FEAT_BF16";
    if ((c & HALF_CPU_BF16_CONVERT) != 0) return "FP16 conversion, FEAT_BF16";
    if ((c & HALF_CPU_F16_ARITH) != 0) return "FP16 conversion, FEAT_FP16";
    if ((c & HALF_CPU_F16_CONVERT) != 0) return "FP16 conversion";
#endif
    return "none";
}

// ══ Bulk conversion kernels ══════════════════════════════════════════════════
// Each converts [lo, hi) of its arrays; the public functions below split
// large arrays across the thread pool. Inline assembly reads and writes
// memory directly (SafeC has no vector types to bind registers to).

#define HC_F32_FP16  0
#define HC_FP16_F32  1
#define HC_F32_BF16  2
#define HC_BF16_F32  3
#define HC_FP16_BF16 4
#define HC_BF16_FP16 5
#define HC_PAR_MIN   262144UL
#define HC_GRAIN     65536UL

struct HcJob_ {
    const unsigned char* src;   // the op says the element types
    unsigned char*       dst;
    int         op;
    int         caps;
};

// float32 -> bf16, round to nearest even, NaN kept a NaN: branch-free, so
// the loop vectorizes.
static void hc_f32_bf16_soft_(const unsigned int* s, unsigned short* d, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long i = lo;
        while (i < hi) {
            unsigned int u = s[i];
            unsigned int r = (u + 0x7FFFU + ((u >> 16) & 1U)) >> 16;
            unsigned int q = (u >> 16) | 0x40U;
            d[i] = (unsigned short)((u & 0x7FFFFFFFU) > 0x7F800000U ? q : r);
            i = i + 1UL;
        }
    }
}

static void hc_bf16_f32_soft_(const unsigned short* s, unsigned int* d, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long i = lo;
        while (i < hi) { d[i] = (unsigned int)s[i] << 16; i = i + 1UL; }
    }
}

static void hc_f32_fp16_soft_(const float* s, unsigned short* d, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long i = lo;
        while (i < hi) { d[i] = f32_to_fp16(s[i]); i = i + 1UL; }
    }
}

static void hc_fp16_f32_soft_(const unsigned short* s, float* d, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long i = lo;
        while (i < hi) { d[i] = fp16_to_f32(s[i]); i = i + 1UL; }
    }
}

// The fp16 / bf16 hardware loops; each returns the first index it did not
// convert (the caller finishes the tail in software).
#if defined(__x86_64__) && !defined(__SAFEC_FREESTANDING__)
static unsigned long hc_f32_fp16_hw_(const float* s, unsigned short* d, unsigned long lo, unsigned long hi) {
    unsigned long i = lo;
    unsafe {
        while (i + 8UL <= hi) {
            asm volatile ("vmovups (%0), %%ymm0\n\tvcvtps2ph $$0, %%ymm0, (%1)"
                          : : "r"(&s[i]), "r"(&d[i]) : "xmm0", "memory");
            i = i + 8UL;
        }
        asm volatile ("vzeroupper" : : : "memory");
    }
    return i;
}

static unsigned long hc_fp16_f32_hw_(const unsigned short* s, float* d, unsigned long lo, unsigned long hi) {
    unsigned long i = lo;
    unsafe {
        while (i + 8UL <= hi) {
            asm volatile ("vcvtph2ps (%0), %%ymm0\n\tvmovups %%ymm0, (%1)"
                          : : "r"(&s[i]), "r"(&d[i]) : "xmm0", "memory");
            i = i + 8UL;
        }
        asm volatile ("vzeroupper" : : : "memory");
    }
    return i;
}

// VCVTNEPS2BF16 flushes float32 subnormals to zero; VFPCLASSPS spots
// blocks holding one, which are redone in software.
static unsigned long hc_f32_bf16_hw_(const float* s, unsigned short* d, unsigned long lo, unsigned long hi) {
    unsigned long i = lo;
    unsafe {
        while (i + 16UL <= hi) {
            unsigned int sub = 0U;
            asm volatile ("vmovups (%1), %%zmm0\n\t"
                          "vfpclassps $$0x20, %%zmm0, %%k1\n\t"
                          "kmovw %%k1, %0\n\t"
                          "vcvtneps2bf16 %%zmm0, %%ymm1\n\t"
                          "vmovdqu %%ymm1, (%2)"
                          : "=r"(sub) : "r"(&s[i]), "r"(&d[i]) : "xmm0", "xmm1", "k1", "memory");
            if (sub != 0U) hc_f32_bf16_soft_((const unsigned int*)s, d, i, i + 16UL);
            i = i + 16UL;
        }
        asm volatile ("vzeroupper" : : : "memory");
    }
    return i;
}
#elif defined(__aarch64__) && !defined(__SAFEC_FREESTANDING__)
static unsigned long hc_f32_fp16_hw_(const float* s, unsigned short* d, unsigned long lo, unsigned long hi) {
    unsigned long i = lo;
    unsafe {
        while (i + 4UL <= hi) {
            asm volatile ("ldr q0, [%0]\n\tfcvtn v0.4h, v0.4s\n\tstr d0, [%1]"
                          : : "r"(&s[i]), "r"(&d[i]) : "v0", "memory");
            i = i + 4UL;
        }
    }
    return i;
}

static unsigned long hc_fp16_f32_hw_(const unsigned short* s, float* d, unsigned long lo, unsigned long hi) {
    unsigned long i = lo;
    unsafe {
        while (i + 4UL <= hi) {
            asm volatile ("ldr d0, [%0]\n\tfcvtl v0.4s, v0.4h\n\tstr q0, [%1]"
                          : : "r"(&s[i]), "r"(&d[i]) : "v0", "memory");
            i = i + 4UL;
        }
    }
    return i;
}

// No hardware float32 -> bf16 here: BFCVTN's handling of subnormals
// depends on FPCR and FEAT_EBF16, and the vectorized integer rounding
// already runs at memory speed.
static unsigned long hc_f32_bf16_hw_(const float* s, unsigned short* d, unsigned long lo, unsigned long hi) { return lo; }
#else
static unsigned long hc_f32_fp16_hw_(const float* s, unsigned short* d, unsigned long lo, unsigned long hi) { return lo; }
static unsigned long hc_fp16_f32_hw_(const unsigned short* s, float* d, unsigned long lo, unsigned long hi) { return lo; }
static unsigned long hc_f32_bf16_hw_(const float* s, unsigned short* d, unsigned long lo, unsigned long hi) { return lo; }
#endif

static void hc_f32_to_fp16_(const float* s, unsigned short* d, unsigned long lo, unsigned long hi, int caps) {
    unsigned long i = lo;
    if ((caps & HALF_CPU_F16_CONVERT) != 0) i = hc_f32_fp16_hw_(s, d, lo, hi);
    hc_f32_fp16_soft_(s, d, i, hi);
}

static void hc_fp16_to_f32_(const unsigned short* s, float* d, unsigned long lo, unsigned long hi, int caps) {
    unsigned long i = lo;
    if ((caps & HALF_CPU_F16_CONVERT) != 0) i = hc_fp16_f32_hw_(s, d, lo, hi);
    hc_fp16_f32_soft_(s, d, i, hi);
}

static void hc_f32_to_bf16_(const float* s, unsigned short* d, unsigned long lo, unsigned long hi, int caps) {
    unsigned long i = lo;
    if ((caps & HALF_CPU_BF16_CONVERT) != 0) i = hc_f32_bf16_hw_(s, d, lo, hi);
    unsafe { hc_f32_bf16_soft_((const unsigned int*)s, d, i, hi); }
}

// fp16 <-> bf16 through a float32 block on the stack.
#define HC_BLOCK 512UL
static void hc_half_to_half_(const unsigned short* s, unsigned short* d, unsigned long lo, unsigned long hi,
                             int toBf, int caps) {
    float buf[512];
    unsafe {
        unsigned long i = lo;
        while (i < hi) {
            unsigned long e = i + HC_BLOCK < hi ? i + HC_BLOCK : hi;
            float* b = (float*)&buf[0];
            if (toBf) {
                hc_fp16_to_f32_(&s[i], b, 0UL, e - i, caps);
                hc_f32_to_bf16_((const float*)b, &d[i], 0UL, e - i, caps);
            } else {
                hc_bf16_f32_soft_(&s[i], (unsigned int*)b, 0UL, e - i);
                hc_f32_to_fp16_((const float*)b, &d[i], 0UL, e - i, caps);
            }
            i = e;
        }
    }
}

static void hc_range_(&HcJob_ j, unsigned long lo, unsigned long hi) {
    unsafe {
        int op = j.op;
        if (op == HC_F32_FP16) hc_f32_to_fp16_((const float*)j.src, (unsigned short*)j.dst, lo, hi, j.caps);
        else if (op == HC_FP16_F32) hc_fp16_to_f32_((const unsigned short*)j.src, (float*)j.dst, lo, hi, j.caps);
        else if (op == HC_F32_BF16) hc_f32_to_bf16_((const float*)j.src, (unsigned short*)j.dst, lo, hi, j.caps);
        else if (op == HC_BF16_F32) hc_bf16_f32_soft_((const unsigned short*)j.src, (unsigned int*)j.dst, lo, hi);
        else hc_half_to_half_((const unsigned short*)j.src, (unsigned short*)j.dst, lo, hi, op == HC_FP16_BF16, j.caps);
    }
}

static void hc_run_(const unsigned char* src, unsigned char* dst, unsigned long n, int op) {
    struct HcJob_ j;
    j.src = src;
    j.dst = dst;
    j.op = op;
    j.caps = half_cpu_features();
    if (n >= HC_PAR_MIN) parallel_for(n, HC_GRAIN, hc_range_, &j);
    else hc_range_(&j, 0UL, n);
}

void f32_to_fp16_bulk_n_(const &float src, &unsigned short dst, unsigned long n) {
    unsafe { hc_run_((const unsigned char*)src, (unsigned char*)dst, n, HC_F32_FP16); }
}

void f32_to_fp16_bulk([]const float src_s, []unsigned short dst_s) {
    unsigned long n = src_s.len < dst_s.len ? src_s.len : dst_s.len;
    float src_d_[1];
    const &float src = &src_d_[0];
    if (src_s.len > 0UL) { unsafe { src = (const &float)src_s.ptr; } }
    unsigned short dst_d_[1];
    &unsigned short dst = &dst_d_[0];
    if (dst_s.len > 0UL) { unsafe { dst = (&unsigned short)dst_s.ptr; } }
    f32_to_fp16_bulk_n_(src, dst, n);
}

void fp16_to_f32_bulk_n_(const &unsigned short src, &float dst, unsigned long n) {
    unsafe { hc_run_((const unsigned char*)src, (unsigned char*)dst, n, HC_FP16_F32); }
}

void fp16_to_f32_bulk([]const unsigned short src_s, []float dst_s) {
    unsigned long n = src_s.len < dst_s.len ? src_s.len : dst_s.len;
    unsigned short src_d_[1];
    const &unsigned short src = &src_d_[0];
    if (src_s.len > 0UL) { unsafe { src = (const &unsigned short)src_s.ptr; } }
    float dst_d_[1];
    &float dst = &dst_d_[0];
    if (dst_s.len > 0UL) { unsafe { dst = (&float)dst_s.ptr; } }
    fp16_to_f32_bulk_n_(src, dst, n);
}

void f32_to_bf16_bulk_n_(const &float src, &unsigned short dst, unsigned long n) {
    unsafe { hc_run_((const unsigned char*)src, (unsigned char*)dst, n, HC_F32_BF16); }
}

void f32_to_bf16_bulk([]const float src_s, []unsigned short dst_s) {
    unsigned long n = src_s.len < dst_s.len ? src_s.len : dst_s.len;
    float src_d_[1];
    const &float src = &src_d_[0];
    if (src_s.len > 0UL) { unsafe { src = (const &float)src_s.ptr; } }
    unsigned short dst_d_[1];
    &unsigned short dst = &dst_d_[0];
    if (dst_s.len > 0UL) { unsafe { dst = (&unsigned short)dst_s.ptr; } }
    f32_to_bf16_bulk_n_(src, dst, n);
}

void bf16_to_f32_bulk_n_(const &unsigned short src, &float dst, unsigned long n) {
    unsafe { hc_run_((const unsigned char*)src, (unsigned char*)dst, n, HC_BF16_F32); }
}

void bf16_to_f32_bulk([]const unsigned short src_s, []float dst_s) {
    unsigned long n = src_s.len < dst_s.len ? src_s.len : dst_s.len;
    unsigned short src_d_[1];
    const &unsigned short src = &src_d_[0];
    if (src_s.len > 0UL) { unsafe { src = (const &unsigned short)src_s.ptr; } }
    float dst_d_[1];
    &float dst = &dst_d_[0];
    if (dst_s.len > 0UL) { unsafe { dst = (&float)dst_s.ptr; } }
    bf16_to_f32_bulk_n_(src, dst, n);
}

void fp16_to_bf16_bulk_n_(const &unsigned short src, &unsigned short dst, unsigned long n) {
    unsafe { hc_run_((const unsigned char*)src, (unsigned char*)dst, n, HC_FP16_BF16); }
}

void fp16_to_bf16_bulk([]const unsigned short src_s, []unsigned short dst_s) {
    unsigned long n = src_s.len < dst_s.len ? src_s.len : dst_s.len;
    unsigned short src_d_[1];
    const &unsigned short src = &src_d_[0];
    if (src_s.len > 0UL) { unsafe { src = (const &unsigned short)src_s.ptr; } }
    unsigned short dst_d_[1];
    &unsigned short dst = &dst_d_[0];
    if (dst_s.len > 0UL) { unsafe { dst = (&unsigned short)dst_s.ptr; } }
    fp16_to_bf16_bulk_n_(src, dst, n);
}

void bf16_to_fp16_bulk_n_(const &unsigned short src, &unsigned short dst, unsigned long n) {
    unsafe { hc_run_((const unsigned char*)src, (unsigned char*)dst, n, HC_BF16_FP16); }
}

void bf16_to_fp16_bulk([]const unsigned short src_s, []unsigned short dst_s) {
    unsigned long n = src_s.len < dst_s.len ? src_s.len : dst_s.len;
    unsigned short src_d_[1];
    const &unsigned short src = &src_d_[0];
    if (src_s.len > 0UL) { unsafe { src = (const &unsigned short)src_s.ptr; } }
    unsigned short dst_d_[1];
    &unsigned short dst = &dst_d_[0];
    if (dst_s.len > 0UL) { unsafe { dst = (&unsigned short)dst_s.ptr; } }
    bf16_to_fp16_bulk_n_(src, dst, n);
}

} // namespace std
