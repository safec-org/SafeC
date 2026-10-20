// SafeC Standard Library — SIMD level detection (see cpu.h).
#pragma once
#include <std/simd/cpu.h>
#include <std/sys.h>
#include <std/str.h>
#include <std/atomic.h>

namespace std {

static int g_simd_level = -1;   // -1 = not yet detected (benign race: same answer)

// The level this build already guarantees.
static int simd_compiled_() {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VL__)
    return SIMD_AVX512;
#elif defined(__AVX2__) && defined(__FMA__)
    return SIMD_AVX2;
#elif defined(__AVX__)
    return SIMD_AVX;
#elif defined(__SSE4_1__)
    return SIMD_SSE41;
#elif defined(__SSE2__) || defined(__x86_64__)
    return SIMD_SSE2;
#elif defined(__ARM_FEATURE_SVE)
    return SIMD_SVE;
#elif defined(__aarch64__) || defined(__ARM_NEON)
    return SIMD_NEON;
#elif defined(__ARM_FEATURE_MVE)
    return SIMD_HELIUM;
#elif defined(__ARM_FEATURE_DSP)
    return SIMD_DSP;
#elif defined(__wasm_simd128__)
    return SIMD_WASM128;
#else
    return SIMD_SCALAR;
#endif
}

#if defined(__x86_64__) && !defined(__SAFEC_FREESTANDING__)
static void simd_cpuid_(unsigned int leaf, unsigned int sub, &unsigned int a, &unsigned int b,
                        &unsigned int c, &unsigned int d) {
    unsigned int ra = 0U; unsigned int rb = 0U; unsigned int rc = 0U; unsigned int rd = 0U;
    unsafe { asm volatile ("cpuid" : "=a"(ra), "=b"(rb), "=c"(rc), "=d"(rd) : "a"(leaf), "c"(sub)); }
    *a = ra; *b = rb; *c = rc; *d = rd;
}

static int simd_detect_() {
    unsigned int a = 0U; unsigned int b = 0U; unsigned int c = 0U; unsigned int d = 0U;
    simd_cpuid_(0U, 0U, &a, &b, &c, &d);
    unsigned int maxLeaf = a;
    simd_cpuid_(1U, 0U, &a, &b, &c, &d);
    int level = (d & (1U << 26)) != 0U ? SIMD_SSE2 : SIMD_SCALAR;
    if (level == SIMD_SCALAR) return level;
    if ((c & (1U << 19)) != 0U) level = SIMD_SSE41;
    int fma = (c & (1U << 12)) != 0U;
    // AVX needs the CPU bit and the OS saving YMM state (OSXSAVE + XCR0).
    if ((c & (1U << 28)) == 0U || (c & (1U << 27)) == 0U) return level;
    unsigned int xlo = 0U; unsigned int xhi = 0U; unsigned int zero = 0U;
    unsafe { asm volatile ("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(zero)); }
    if ((xlo & 6U) != 6U) return level;
    level = SIMD_AVX;
    if (maxLeaf < 7U) return level;
    simd_cpuid_(7U, 0U, &a, &b, &c, &d);
    if ((b & (1U << 5)) != 0U && fma) level = SIMD_AVX2;
    unsigned int need512 = (1U << 16) | (1U << 17) | (1U << 30) | (1U << 31);   // F, DQ, BW, VL
    if (level == SIMD_AVX2 && (b & need512) == need512 && (xlo & 0xE6U) == 0xE6U) level = SIMD_AVX512;
    return level;
}
#elif defined(__aarch64__) && defined(__linux__) && !defined(__SAFEC_FREESTANDING__)
extern unsigned long getauxval(unsigned long type);
static int simd_detect_() {
    unsigned long hw = 0UL;
    unsafe { hw = getauxval(16UL); }                  // AT_HWCAP
    return (hw & (1UL << 22)) != 0UL ? SIMD_SVE : SIMD_NEON;   // HWCAP_SVE
}
#else
static int simd_detect_() { return simd_compiled_(); }
#endif

static int simd_cap_(int level) {
#ifdef __SAFEC_FREESTANDING__
    return level;
#else
    int cap = -1;
    match (sys_getenv("SAFEC_SIMD")) {
        case some(e): {
            if (str_eq(e, "sse2")) cap = SIMD_SSE2;
            else if (str_eq(e, "sse4.1") || str_eq(e, "sse41")) cap = SIMD_SSE41;
            else if (str_eq(e, "avx")) cap = SIMD_AVX;
            else if (str_eq(e, "avx2")) cap = SIMD_AVX2;
            else if (str_eq(e, "avx512")) cap = SIMD_AVX512;
            else if (str_eq(e, "neon")) cap = SIMD_NEON;
            else if (str_eq(e, "sve")) cap = SIMD_SVE;
            else if (str_eq(e, "scalar")) cap = SIMD_SCALAR;
        }
        case null: { }
    }
    // A cap from another family, or above what the CPU has, changes nothing.
    if (cap < 0 || cap > level || (cap >= SIMD_NEON) != (level >= SIMD_NEON)) return level;
    // Never below what the build itself already requires.
    int floor = simd_compiled_();
    return cap < floor ? floor : cap;
#endif
}

int simd_level() {
    int l = atomic_load_int(&g_simd_level);
    if (l >= 0) return l;
    int d = simd_detect_();
    int c = simd_compiled_();
    l = d > c ? d : c;
    l = simd_cap_(l);
    atomic_store_int(&g_simd_level, l);
    return l;
}

const char* simd_level_name(int level) {
    if (level == SIMD_AVX512) return "AVX-512";
    if (level == SIMD_AVX2) return "AVX2";
    if (level == SIMD_AVX) return "AVX";
    if (level == SIMD_SSE41) return "SSE4.1";
    if (level == SIMD_SSE2) return "SSE2";
    if (level == SIMD_NEON) return "Neon";
    if (level == SIMD_SVE) return "SVE";
    if (level == SIMD_DSP) return "Cortex-M DSP";
    if (level == SIMD_HELIUM) return "Helium";
    if (level == SIMD_WASM128) return "WebAssembly SIMD128";
    return "scalar";
}

} // namespace std
