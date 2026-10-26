// fp16 / bf16 on the CPU: hardware-feature detection, the bulk conversions
// (hardware paths against the scalar reference, bit for bit), and the
// mixed-precision batch helpers in tensor_half.h.
#include <std/str.h>
#include <std/test/test.h>
#include <std/ml/float16.h>
#include <std/ml/tensor_half.h>
#include <std/ml/tensor.h>
#include <std/collections/vec.h>

extern void* malloc(unsigned long size);
extern void free(void* ptr);
extern void* memcpy(void* dst, const void* src, unsigned long n);

static unsigned int bits_of_(float v) {
    unsigned int u = 0U;
    unsafe { memcpy((void*)&u, (const void*)&v, 4UL); }
    return u;
}

static float float_of_(unsigned int u) {
    float v = (float)0.0;
    unsafe { memcpy((void*)&v, (const void*)&u, 4UL); }
    return v;
}

static int f32_nan_(unsigned int u) { return (u & 0x7FFFFFFFU) > 0x7F800000U; }
static int fp16_nan_(unsigned short h) { return ((unsigned int)h & 0x7FFFU) > 0x7C00U; }
static int bf16_nan_(unsigned short h) { return ((unsigned int)h & 0x7FFFU) > 0x7F80U; }

static unsigned int lcg_(&unsigned long s) {
    *s = *s * 6364136223846793005UL + 1442695040888963407UL;
    return (unsigned int)(*s >> 32);
}

// n float32 bit patterns of every class: random bits, plus subnormals,
// zeros, infinities, NaNs and values near the fp16 range limits.
static float* test_values_(unsigned long n, unsigned long seed) {
    unsigned long s = seed;
    float* v;
    unsafe {
        v = (float*)malloc(sizeof(float) * n);
        unsigned long i = 0UL;
        while (i < n) {
            unsigned int u = lcg_(&s);
            unsigned long kind = i % 8UL;
            if (kind == 1UL) u = u & 0x807FFFFFU;                    // float32 subnormal
            else if (kind == 2UL) u = (u & 0x80FFFFFFU) | 0x38000000U;  // near fp16's subnormal range
            else if (kind == 3UL) u = (u & 0x80FFFFFFU) | 0x47000000U;  // near fp16's max
            else if (kind == 4UL) u = (u & 0x803FFFFFU) | 0x3F800000U;  // ordinary magnitudes
            v[i] = float_of_(u);
            i = i + 1UL;
        }
        v[0] = (float)0.0; v[8] = float_of_(0x80000000U); v[16] = float_of_(0x7F800000U);
        v[24] = float_of_(0xFF800000U); v[32] = float_of_(0x7FC00001U); v[40] = float_of_(0x7F800001U);
        v[48] = (float)65504.0; v[56] = (float)65520.0; v[64] = float_of_(0x33000000U);
        // one subnormal alone in a block of normals (the AVX512-BF16 path's fallback)
        unsigned long k = 96UL;
        while (k < 112UL) { v[k] = (float)1.5; k = k + 1UL; }
        v[101] = float_of_(0x00000123U);
    }
    return v;
}

void t_half_features() {
    int c = half_cpu_features();
    ASSERT_TRUE(c >= 0 && c < 16);
    ASSERT_EQ(c & HALF_CPU_BF16_CONVERT ? 1 : 0, c & HALF_CPU_BF16_DOT ? 1 : 0);
    ASSERT_EQ(half_cpu_features(), c);          // cached
    ASSERT_TRUE(str_len(half_cpu_features_name()) > 0UL);
#if defined(__aarch64__)
    ASSERT_TRUE((c & HALF_CPU_F16_CONVERT) != 0);
#endif
}

// Every one of the 65536 fp16 patterns widens exactly as fp16_to_f32 does.
void t_fp16_to_f32_all() {
    unsafe {
        unsigned short* h = (unsigned short*)malloc(2UL * 65536UL);
        float* f = (float*)malloc(4UL * 65536UL);
        unsigned long i = 0UL;
        while (i < 65536UL) { h[i] = (unsigned short)i; i = i + 1UL; }
        fp16_to_f32_bulk(((const unsigned short*)((const &unsigned short)h))[0..65536UL], ((float*)((&float)f))[0..65536UL]);
        unsigned long bad = 0UL;
        i = 0UL;
        while (i < 65536UL) {
            unsigned int got = bits_of_(f[i]);
            unsigned int want = bits_of_(fp16_to_f32(h[i]));
            if (fp16_nan_(h[i])) { if (!f32_nan_(got)) bad = bad + 1UL; }
            else if (got != want) bad = bad + 1UL;
            i = i + 1UL;
        }
        ASSERT_EQ(bad, 0UL);
        free((void*)h); free((void*)f);
    }
}

// float32 -> fp16 / bf16 and bf16 -> float32 in bulk against the scalar
// functions, at a small size (one thread, odd tail) and a large one (the
// thread pool).
static unsigned long check_narrowing_(unsigned long n, unsigned long seed) {
    unsigned long bad = 0UL;
    unsafe {
        float* v = test_values_(n, seed);
        unsigned short* h = (unsigned short*)malloc(2UL * n);
        unsigned short* b = (unsigned short*)malloc(2UL * n);
        float* back = (float*)malloc(4UL * n);
        f32_to_fp16_bulk(((const float*)((const &float)v))[0..n], ((unsigned short*)((&unsigned short)h))[0..n]);
        f32_to_bf16_bulk(((const float*)((const &float)v))[0..n], ((unsigned short*)((&unsigned short)b))[0..n]);
        bf16_to_f32_bulk(((const unsigned short*)((const &unsigned short)b))[0..n], ((float*)((&float)back))[0..n]);
        unsigned long i = 0UL;
        while (i < n) {
            unsigned int u = bits_of_(v[i]);
            unsigned short wh = f32_to_fp16(v[i]);
            unsigned short wb = f32_to_bf16(v[i]);
            if (f32_nan_(u)) {
                if (!fp16_nan_(h[i]) || !bf16_nan_(b[i])) bad = bad + 1UL;
            } else {
                if (h[i] != wh || b[i] != wb) bad = bad + 1UL;
            }
            if (bits_of_(back[i]) != ((unsigned int)b[i] << 16)) bad = bad + 1UL;
            i = i + 1UL;
        }
        free((void*)v); free((void*)h); free((void*)b); free((void*)back);
    }
    return bad;
}

void t_narrowing_bulk() {
    ASSERT_EQ(check_narrowing_(1003UL, 7UL), 0UL);
    ASSERT_EQ(check_narrowing_(300007UL, 11UL), 0UL);    // >= 256K: split across the pool
}

// fp16 <-> bf16 equals going through float32 with the scalar functions.
void t_half_to_half_bulk() {
    unsafe {
        unsigned long n = 65536UL;
        unsigned short* src = (unsigned short*)malloc(2UL * n);
        unsigned short* d = (unsigned short*)malloc(2UL * n);
        unsigned long i = 0UL;
        while (i < n) { src[i] = (unsigned short)i; i = i + 1UL; }
        fp16_to_bf16_bulk(((const unsigned short*)((const &unsigned short)src))[0..n], ((unsigned short*)((&unsigned short)d))[0..n]);
        unsigned long bad = 0UL;
        i = 0UL;
        while (i < n) {
            unsigned short want = f32_to_bf16(fp16_to_f32(src[i]));
            if (fp16_nan_(src[i])) { if (!bf16_nan_(d[i])) bad = bad + 1UL; }
            else if (d[i] != want) bad = bad + 1UL;
            i = i + 1UL;
        }
        bf16_to_fp16_bulk(((const unsigned short*)((const &unsigned short)src))[0..n], ((unsigned short*)((&unsigned short)d))[0..n]);
        i = 0UL;
        while (i < n) {
            unsigned short want = f32_to_fp16(bf16_to_f32(src[i]));
            if (bf16_nan_(src[i])) { if (!fp16_nan_(d[i])) bad = bad + 1UL; }
            else if (d[i] != want) bad = bad + 1UL;
            i = i + 1UL;
        }
        ASSERT_EQ(bad, 0UL);
        free((void*)src); free((void*)d);
    }
}

generic<E>
static void tensor_drop_(&DenseTensor<E> t) {
    t->free();
    unsafe { free((void*)t); }
}

// Masters -> 16-bit copies -> back; fp16 <-> bf16; gradients with the loss
// scale divided out, and overflow detection.
void t_mixed_precision_batch() {
    struct Vec<&FloatTensor> ps = Vec::new();
    ps.push(FloatTensor::new_2d(3UL, 5UL, 1));
    ps.push(FloatTensor::new_1d(7UL, 1));
    unsafe {
        unsigned long t = 0UL;
        while (t < 2UL) {
            &FloatTensor p = *ps.at(t);
            unsigned long i = 0UL;
            while (i < p->size) { p->data[i] = (float)i * (float)0.37 - (float)1.1 + (float)t; i = i + 1UL; }
            t = t + 1UL;
        }
    }
    struct Vec<&BF16Tensor> w16 = Vec::new();
    tensors_convert(&ps, &w16);
    ASSERT_EQ(w16.length(), 2UL);
    &BF16Tensor h0 = *w16.at(0UL);
    ASSERT_EQ(h0.size, 15UL);
    ASSERT_EQ(h0.ndim, 2UL);
    ASSERT_EQ(h0.requiresGrad, 0);
    unsafe { ASSERT_TRUE(h0.at1(4UL) == bf16_to_f32(f32_to_bf16((*ps.at(0UL))->data[4]))); }

    // refresh after the masters change
    unsafe { (*ps.at(1UL))->data[3] = (float)42.0; }
    ASSERT_TRUE(tensors_convert_into(&ps, &w16));
    &BF16Tensor h1 = *w16.at(1UL);
    ASSERT_TRUE(h1.at1(3UL) == (float)42.0);

    // back into fresh tensors
    struct Vec<&FloatTensor> copies = Vec::new();
    copies.push(FloatTensor::new_2d(3UL, 5UL, 0));
    copies.push(FloatTensor::new_1d(7UL, 0));
    ASSERT_TRUE(tensors_convert_into(&w16, &copies));
    unsafe { ASSERT_TRUE((*copies.at(1UL))->data[3] == (float)42.0); }

    // bf16 -> fp16
    &HalfTensor f16 = HalfTensor::new_1d(7UL, 0);
    ASSERT_TRUE(tensor_convert_into(h1, f16));
    ASSERT_TRUE(f16.at1(3UL) == (float)42.0);
    &HalfTensor small = HalfTensor::new_1d(3UL, 0);
    ASSERT_TRUE(!tensor_convert_into(h1, small));

    // gradients: fp16 values scaled by 1024, divided back out
    struct Vec<&HalfTensor> g16 = Vec::new();
    g16.push(HalfTensor::new_2d(3UL, 5UL, 0));
    g16.push(HalfTensor::new_1d(7UL, 0));
    (*g16.at(0UL)).fill((float)1024.0 * (float)0.25);
    (*g16.at(1UL)).fill((float)-2048.0);
    ASSERT_TRUE(half_grads_accumulate(&g16, &ps, (float)1.0 / (float)1024.0));
    unsafe {
        ASSERT_TRUE((*ps.at(0UL))->grad[7] == (float)0.25);
        ASSERT_TRUE((*ps.at(1UL))->grad[2] == (float)-2.0);
    }
    // an overflowed gradient (fp16 infinity) is reported
    (*g16.at(1UL)).set1(5UL, (float)1.0e9);   // rounds to +inf in fp16
    ASSERT_TRUE(!half_grads_accumulate(&g16, &ps, (float)1.0));

    // LossScaler::unscale_all over the list
    struct LossScaler sc = LossScaler::new();
    sc.scale = (float)4.0;
    (*ps.at(0UL)).zero_grad();
    (*ps.at(1UL)).zero_grad();
    unsafe { (*ps.at(0UL))->grad[0] = (float)8.0; }
    ASSERT_TRUE(sc.unscale_all(&ps));
    unsafe { ASSERT_TRUE((*ps.at(0UL))->grad[0] == (float)2.0); }

    tensors_free(&g16);
    tensors_free(&w16);
    tensor_drop_(f16);
    tensor_drop_(small);
    tensor_drop_(*copies.at(0UL)); tensor_drop_(*copies.at(1UL));
    tensor_drop_(*ps.at(0UL)); tensor_drop_(*ps.at(1UL));
    copies.free();
    ps.free();
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("half-precision CPU features", t_half_features);
        t.add("fp16 -> f32 bulk, all 65536 patterns", t_fp16_to_f32_all);
        t.add("f32 -> fp16 / bf16 bulk = scalar", t_narrowing_bulk);
        t.add("fp16 <-> bf16 bulk", t_half_to_half_bulk);
        t.add("mixed-precision batch conversion", t_mixed_precision_batch);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
