// SafeC Standard Library — fp16 / bf16 tensors and mixed-precision ops
// (see tensor_half.h).
#include <std/ml/tensor_half.h>
#include <std/sync/parallel.h>
#include <std/simd/cpu.h>
#include <std/simd/simd.h>   // f32x4 / u32x4

namespace std {

extern void* memcpy(void* dst, const void* src, unsigned long n);

// ── Precision helpers ────────────────────────────────────────────────────────

unsigned short half_encode(float v, int dtype) {
    if (dtype == TENSOR_BF16) return f32_to_bf16(v);
    return f32_to_fp16(v);
}

float half_decode(unsigned short bits, int dtype) {
    if (dtype == TENSOR_BF16) return bf16_to_f32(bits);
    return fp16_to_f32(bits);
}

float half_round(float v, int dtype) {
    if (dtype == TENSOR_FP16) return fp16_to_f32(f32_to_fp16(v));
    if (dtype == TENSOR_BF16) return bf16_to_f32(f32_to_bf16(v));
    return v;
}

// ── HalfTensor ───────────────────────────────────────────────────────────────

static struct HalfTensor __half_alloc(unsigned long d0, unsigned long d1, unsigned long ndim, int dtype) {
    struct HalfTensor h;
    h.shape[0] = d0;
    h.shape[1] = d1;
    h.ndim = ndim;
    h.size = ndim == 1UL ? d0 : d0 * d1;
    h.dtype = dtype;
    unsafe {
        unsigned long bytes = h.size > 0UL ? sizeof(unsigned short) * h.size : sizeof(unsigned short);
        unsigned short* buf = (unsigned short*)malloc(bytes);
        unsigned long i = 0UL;
        while (i < h.size) { buf[i] = (unsigned short)0; i = i + 1UL; }
        h.data = (&heap unsigned short)buf;
    }
    return h;
}

// An empty result for shape/dtype mismatches: size 0, still safe to free().
static struct HalfTensor __half_empty(int dtype) {
    return __half_alloc(0UL, 0UL, 1UL, dtype);
}

static int __half_same_shape(const &HalfTensor a, const &HalfTensor b) {
    if (a.dtype != b.dtype || a.ndim != b.ndim) return 0;
    if (a.shape[0] != b.shape[0]) return 0;
    if (a.ndim == 2UL && a.shape[1] != b.shape[1]) return 0;
    return 1;
}

static struct HalfTensor __half_like(const &HalfTensor a) {
    return __half_alloc(a.shape[0], a.shape[1], a.ndim, a.dtype);
}

float HalfTensor::at1(unsigned long i) const {
    float v = (float)0.0;
    unsafe { v = half_decode(self.data[i], self.dtype); }
    return v;
}

float HalfTensor::at2(unsigned long r, unsigned long c) const {
    float v = (float)0.0;
    unsafe { v = half_decode(self.data[r * self.shape[1] + c], self.dtype); }
    return v;
}

void HalfTensor::set1(unsigned long i, float v) {
    unsafe { self.data[i] = half_encode(v, self.dtype); }
}

void HalfTensor::set2(unsigned long r, unsigned long c, float v) {
    unsafe { self.data[r * self.shape[1] + c] = half_encode(v, self.dtype); }
}

void HalfTensor::free() {
    unsafe { std::free((void*)self.data); }
    self.size = 0UL;
}

struct HalfTensor HalfTensor::new_1d(unsigned long n, int dtype) {
    return __half_alloc(n, 1UL, 1UL, dtype);
}

struct HalfTensor HalfTensor::new_2d(unsigned long rows, unsigned long cols, int dtype) {
    return __half_alloc(rows, cols, 2UL, dtype);
}

struct HalfTensor Tensor::to_half(int dtype) const {
    struct HalfTensor h;
    unsafe {
        unsigned long d1 = self->ndim == 2UL ? self->shape[1] : 1UL;
        h = __half_alloc(self->shape[0], d1, self->ndim, dtype);
        if (dtype == TENSOR_BF16) f32_to_bf16_bulk((const float*)self->data, (unsigned short*)h.data, self->size);
        else f32_to_fp16_bulk((const float*)self->data, (unsigned short*)h.data, self->size);
    }
    return h;
}

&Tensor HalfTensor::to_tensor(int requiresGrad) const {
    &Tensor out = self.ndim == 2UL ? Tensor::new_2d(self.shape[0], self.shape[1], requiresGrad)
                                : Tensor::new_1d(self.shape[0], requiresGrad);
    unsafe {
        if (self.dtype == TENSOR_BF16) bf16_to_f32_bulk((const unsigned short*)self.data, (float*)out->data, self.size);
        else fp16_to_f32_bulk((const unsigned short*)self.data, (float*)out->data, self.size);
    }
    return out;
}

// ── Batch conversion ─────────────────────────────────────────────────────────

static void __half_encode_bulk(const float* src, unsigned short* dst, unsigned long n, int dtype) {
    unsafe {
        if (dtype == TENSOR_BF16) f32_to_bf16_bulk((const &float)src, (&unsigned short)dst, n);
        else f32_to_fp16_bulk((const &float)src, (&unsigned short)dst, n);
    }
}

static void __half_decode_bulk(const unsigned short* src, float* dst, unsigned long n, int dtype) {
    unsafe {
        if (dtype == TENSOR_BF16) bf16_to_f32_bulk((const &unsigned short)src, (&float)dst, n);
        else fp16_to_f32_bulk((const &unsigned short)src, (&float)dst, n);
    }
}

int tensor_to_half_into(const &Tensor t, &HalfTensor dst) {
    unsafe {
        if (t->size != dst.size) return 0;
        __half_encode_bulk((const float*)t->data, (unsigned short*)dst.data, t->size, dst.dtype);
    }
    return 1;
}

int half_to_tensor_into(const &HalfTensor h, &Tensor dst) {
    unsafe {
        if (dst->size != h.size) return 0;
        __half_decode_bulk((const unsigned short*)h.data, (float*)dst->data, h.size, h.dtype);
    }
    return 1;
}

int half_convert(const &HalfTensor src, &HalfTensor dst) {
    if (src.size != dst.size) return 0;
    unsafe {
        const unsigned short* s = (const unsigned short*)src.data;
        unsigned short* d = (unsigned short*)dst.data;
        if (src.dtype == dst.dtype) memcpy((void*)d, (const void*)s, src.size * 2UL);
        else if (dst.dtype == TENSOR_BF16) fp16_to_bf16_bulk((const &unsigned short)s, (&unsigned short)d, src.size);
        else bf16_to_fp16_bulk((const &unsigned short)s, (&unsigned short)d, src.size);
    }
    return 1;
}

struct Vec<struct HalfTensor> tensors_to_half(const &Vec<&Tensor> ts, int dtype) {
    struct Vec<struct HalfTensor> out = vec_new();
    unsigned long i = 0UL;
    while (i < ts.length()) { out.push((*ts.at(i)).to_half(dtype)); i = i + 1UL; }
    return out;
}

int tensors_to_half_into(const &Vec<&Tensor> ts, &Vec<struct HalfTensor> dst) {
    if (ts.length() != dst.length()) return 0;
    int ok = 1;
    unsigned long i = 0UL;
    while (i < ts.length()) {
        if (!tensor_to_half_into(*ts.at(i), dst.at(i))) ok = 0;
        i = i + 1UL;
    }
    return ok;
}

int halves_to_tensors(const &Vec<struct HalfTensor> hs, &Vec<&Tensor> dst) {
    if (hs.length() != dst.length()) return 0;
    int ok = 1;
    unsigned long i = 0UL;
    while (i < hs.length()) {
        if (!half_to_tensor_into(hs.at(i), *dst.at(i))) ok = 0;
        i = i + 1UL;
    }
    return ok;
}

void half_tensors_free(&Vec<struct HalfTensor> hs) {
    unsigned long i = 0UL;
    while (i < hs.length()) { &HalfTensor h = hs.at(i); h.free(); i = i + 1UL; }
    hs.free();
}

#define HALF_GRAD_CHUNK 4096UL

int half_grads_accumulate(const &Vec<struct HalfTensor> grads, &Vec<&Tensor> params, float invScale) {
    if (grads.length() != params.length()) return 0;
    int finite = 1;
    float buf[4096];
    unsigned long t = 0UL;
    while (t < grads.length()) {
        &HalfTensor g = grads.at(t);
        &Tensor p = *params.at(t);
        unsafe {
            if (g.size != p->size) return 0;
            if (p->grad == (float*)0) {
                float* z = (float*)malloc(sizeof(float) * p->size);
                unsigned long k = 0UL;
                while (k < p->size) { z[k] = (float)0.0; k = k + 1UL; }
                p->grad = (&heap float)z;
            }
            unsigned long i = 0UL;
            while (i < g.size) {
                unsigned long n = g.size - i < HALF_GRAD_CHUNK ? g.size - i : HALF_GRAD_CHUNK;
                float* b = (float*)&buf[0];
                __half_decode_bulk((const unsigned short*)&g.data[i], b, n, g.dtype);
                unsigned long k = 0UL;
                while (k < n) {
                    float v = p->grad[i + k] + b[k] * invScale;
                    if (v - v != (float)0.0) finite = 0;    // inf or NaN
                    p->grad[i + k] = v;
                    k = k + 1UL;
                }
                i = i + n;
            }
        }
        t = t + 1UL;
    }
    return finite;
}

// ── HalfTensor ops ───────────────────────────────────────────────────────────

#define HALF_ADD 0
#define HALF_SUB 1
#define HALF_MUL 2

static struct HalfTensor __half_binary(const &HalfTensor a, const &HalfTensor b, int op) {
    if (!__half_same_shape(a, b)) return __half_empty(a.dtype);
    struct HalfTensor out = __half_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < a.size) {
            float x = half_decode(a.data[i], a.dtype);
            float y = half_decode(b.data[i], a.dtype);
            float r = op == HALF_ADD ? x + y : (op == HALF_SUB ? x - y : x * y);
            out.data[i] = half_encode(r, a.dtype);
            i = i + 1UL;
        }
    }
    return out;
}

struct HalfTensor half_add(const &HalfTensor a, const &HalfTensor b) { return __half_binary(a, b, HALF_ADD); }
struct HalfTensor half_sub(const &HalfTensor a, const &HalfTensor b) { return __half_binary(a, b, HALF_SUB); }
struct HalfTensor half_mul(const &HalfTensor a, const &HalfTensor b) { return __half_binary(a, b, HALF_MUL); }

struct HalfTensor HalfTensor::scale(float k) const {
    struct HalfTensor out = __half_like(self);
    unsafe {
        unsigned long i = 0UL;
        while (i < self.size) {
            out.data[i] = half_encode(half_decode(self.data[i], self.dtype) * k, self.dtype);
            i = i + 1UL;
        }
    }
    return out;
}

struct HalfTensor HalfTensor::relu() const {
    struct HalfTensor out = __half_like(self);
    unsafe {
        unsigned long i = 0UL;
        while (i < self.size) {
            float x = half_decode(self.data[i], self.dtype);
            out.data[i] = x > (float)0.0 ? self.data[i] : (unsigned short)0;
            i = i + 1UL;
        }
    }
    return out;
}

struct HalfTensor half_matmul(const &HalfTensor a, const &HalfTensor b) {
    if (a.ndim != 2UL || b.ndim != 2UL || a.dtype != b.dtype || a.shape[1] != b.shape[0])
        return __half_empty(a.dtype);
    unsigned long m = a.shape[0];
    unsigned long k = a.shape[1];
    unsigned long n = b.shape[1];
    struct HalfTensor out = __half_alloc(m, n, 2UL, a.dtype);
    unsafe {
        // Widen both operands once, multiply in float32 with sgemm (SIMD,
        // threaded, or the vendor BLAS), round each output element once.
        float* fa = (float*)malloc(sizeof(float) * (m * k > 0UL ? m * k : 1UL));
        float* fb = (float*)malloc(sizeof(float) * (k * n > 0UL ? k * n : 1UL));
        float* acc = (float*)malloc(sizeof(float) * (m * n > 0UL ? m * n : 1UL));
        if (a.dtype == TENSOR_BF16) {
            bf16_to_f32_bulk((const unsigned short*)a.data, fa, m * k);
            bf16_to_f32_bulk((const unsigned short*)b.data, fb, k * n);
        } else {
            fp16_to_f32_bulk((const unsigned short*)a.data, fa, m * k);
            fp16_to_f32_bulk((const unsigned short*)b.data, fb, k * n);
        }
        if (k == 0UL) {
            unsigned long z = 0UL;
            while (z < m * n) { acc[z] = (float)0.0; z = z + 1UL; }
        } else {
            sgemm(SGEMM_NO_TRANS, SGEMM_NO_TRANS, m, n, k, (float)1.0, (const &float)fa, k,
                  (const &float)fb, n, (float)0.0, (&float)acc, n);
        }
        if (a.dtype == TENSOR_BF16) f32_to_bf16_bulk((const float*)acc, (unsigned short*)out.data, m * n);
        else f32_to_fp16_bulk((const float*)acc, (unsigned short*)out.data, m * n);
        free((void*)fa);
        free((void*)fb);
        free((void*)acc);
    }
    return out;
}

float HalfTensor::sum() const {
    float acc = (float)0.0;
    unsafe {
        unsigned long i = 0UL;
        while (i < self.size) { acc = acc + half_decode(self.data[i], self.dtype); i = i + 1UL; }
    }
    return acc;
}

// ── Mixed precision with Autograd::get ────────────────────────────────────────────

// Straight-through: d(cast(a))/da is treated as the identity.
static void __cast_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (a->requiresGrad) __tensor_accumulate(a, (const float*)selfT->grad);
    }
}

&Tensor Tensor::cast(int dtype) const {
    &Tensor out = __tensor_alloc_uninit_like(self);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) { out->data[i] = half_round(self->data[i], dtype); i = i + 1UL; }
    }
    __tensor_link1(out, self, (void*)__cast_backward);
    return out;
}

&Tensor tensor_matmul_lowp(const &Tensor a, const &Tensor b, int dtype) {
    &Tensor ca = a.cast(dtype);
    &Tensor cb = b.cast(dtype);
    return tensor_matmul(ca, cb).cast(dtype);
}

// ── Dynamic loss scaling ─────────────────────────────────────────────────────

struct LossScaler LossScaler::new() {
    struct LossScaler s;
    s.scale = (float)65536.0;
    s.growthFactor = (float)2.0;
    s.backoffFactor = (float)0.5;
    s.growthInterval = 2000;
    s.goodSteps = 0;
    return s;
}

// x is finite iff x - x == 0 (inf - inf and NaN - NaN are both NaN, and
// NaN never compares equal).
static int __is_finite(float x) {
    float d = x - x;
    return d == (float)0.0;
}

int LossScaler::unscale(&Tensor p) {
    int finite = 1;
    unsafe {
        if (p->grad == (float*)0) return 1;
        float inv = (float)1.0 / self.scale;
        unsigned long i = 0UL;
        while (i < p->size) {
            float g = p->grad[i] * inv;
            if (!__is_finite(g)) finite = 0;
            p->grad[i] = g;
            i = i + 1UL;
        }
    }
    return finite;
}

int LossScaler::unscale_all(&Vec<&Tensor> params) {
    int finite = 1;
    unsigned long i = 0UL;
    while (i < params.length()) {
        if (!self.unscale(*params.at(i))) finite = 0;
        i = i + 1UL;
    }
    return finite;
}

void LossScaler::update(int allFinite) {
    if (!allFinite) {
        self.scale = self.scale * self.backoffFactor;
        if (self.scale < (float)1.0) self.scale = (float)1.0;
        self.goodSteps = 0;
        return;
    }
    self.goodSteps = self.goodSteps + 1;
    if (self.goodSteps >= self.growthInterval) {
        self.scale = self.scale * self.growthFactor;
        self.goodSteps = 0;
    }
}


// ── HalfTensor::gemv ────────────────────────────────────────────────────────────────
// Rows [lo, hi) of y = w x. The widening is written so clang vectorizes it:
// bf16 is the top half of a float32; fp16 is (bits & 0x7fff) << 13 read as
// a float32 times 2^112 (exact for normal and subnormal values), with
// inf/NaN patched and the sign put back. Eight accumulators per row keep
// the adds independent.
struct HalfGemvJob {
    const unsigned short* w;
    const float* x;
    float* y;
    unsigned long k;
    int bf;
    const float* xe;   // AArch64 kernel: x[0], x[2], ... and x[1], x[3], ... (see below)
    const float* xo;
};

static float half_gemv_w_(unsigned short h, int bf) __attribute__((always_inline)) {
    unsigned int u = 0U;
    float v = (float)0.0;
    if (bf) {
        u = ((unsigned int)h) << 16;
        unsafe { memcpy((void*)&v, (const void*)&u, 4UL); }
        return v;
    }
    u = ((unsigned int)h & 0x7fffU) << 13;
    unsafe { memcpy((void*)&v, (const void*)&u, 4UL); }
    v = v * (float)5.192296858534828e33;                      // 2^112
    unsigned int r = 0U;
    unsafe { memcpy((void*)&r, (const void*)&v, 4UL); }
    if (((unsigned int)h & 0x7c00U) == 0x7c00U) r = u | 0x7f800000U;   // inf / NaN
    r = r | (((unsigned int)h & 0x8000U) << 16);
    unsafe { memcpy((void*)&v, (const void*)&r, 4UL); }
    return v;
}

static void half_gemv_rows_body_(&HalfGemvJob j, unsigned long lo, unsigned long hi) __attribute__((always_inline)) {
    unsigned long k = j.k;
    unsafe {
        unsigned long r = lo;
        while (r < hi) {
            const unsigned short* w = j.w + r * k;
            float a0 = (float)0.0; float a1 = (float)0.0; float a2 = (float)0.0; float a3 = (float)0.0;
            float a4 = (float)0.0; float a5 = (float)0.0; float a6 = (float)0.0; float a7 = (float)0.0;
            unsigned long c = 0UL;
            if (j.bf) {
                while (c + 8UL <= k) {
                    a0 = a0 + half_gemv_w_(w[c], 1) * j.x[c];
                    a1 = a1 + half_gemv_w_(w[c + 1UL], 1) * j.x[c + 1UL];
                    a2 = a2 + half_gemv_w_(w[c + 2UL], 1) * j.x[c + 2UL];
                    a3 = a3 + half_gemv_w_(w[c + 3UL], 1) * j.x[c + 3UL];
                    a4 = a4 + half_gemv_w_(w[c + 4UL], 1) * j.x[c + 4UL];
                    a5 = a5 + half_gemv_w_(w[c + 5UL], 1) * j.x[c + 5UL];
                    a6 = a6 + half_gemv_w_(w[c + 6UL], 1) * j.x[c + 6UL];
                    a7 = a7 + half_gemv_w_(w[c + 7UL], 1) * j.x[c + 7UL];
                    c = c + 8UL;
                }
            } else {
                while (c + 8UL <= k) {
                    a0 = a0 + half_gemv_w_(w[c], 0) * j.x[c];
                    a1 = a1 + half_gemv_w_(w[c + 1UL], 0) * j.x[c + 1UL];
                    a2 = a2 + half_gemv_w_(w[c + 2UL], 0) * j.x[c + 2UL];
                    a3 = a3 + half_gemv_w_(w[c + 3UL], 0) * j.x[c + 3UL];
                    a4 = a4 + half_gemv_w_(w[c + 4UL], 0) * j.x[c + 4UL];
                    a5 = a5 + half_gemv_w_(w[c + 5UL], 0) * j.x[c + 5UL];
                    a6 = a6 + half_gemv_w_(w[c + 6UL], 0) * j.x[c + 6UL];
                    a7 = a7 + half_gemv_w_(w[c + 7UL], 0) * j.x[c + 7UL];
                    c = c + 8UL;
                }
            }
            float a = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
            while (c < k) { a = a + half_gemv_w_(w[c], j.bf) * j.x[c]; c = c + 1UL; }
            j.y[r] = a;
            r = r + 1UL;
        }
    }
}

static void half_gemv_rows_base_(&HalfGemvJob j, unsigned long lo, unsigned long hi) { half_gemv_rows_body_(j, lo, hi); }

#if defined(__aarch64__)
// AArch64: the scalar-written body above doesn't vectorize well on NEON (an
// M1 Pro core reached 8.5 GB/s of bf16 weights, 3.7 for fp16, against ~50
// GB/s of single-core read bandwidth). This kernel never widens 16-bit
// lanes: eight weights load as four 32-bit lanes, each holding w[2i] in its
// low half and w[2i+1] in its high half. For bf16, (lane << 16) and
// (lane & 0xffff0000) already are the two float32 values; for fp16 each half
// goes through the same exponent-rebias trick as half_gemv_w_, on whole
// vectors. x is split once per call into its even and odd elements (xe, xo)
// to match, and four rows share each xe/xo load. Measured on one M1 Pro core:
// 30.8 GB/s bf16 (3.6x), 8.9 GB/s fp16 (2.4x).
// Local splats: std/simd's simd_splat_* live in another translation unit, so
// calling them here would cost a call per use and hide the constant
// divisors below from the optimizer.
static u32x4 half_gemv_u4_(unsigned int v) __attribute__((always_inline)) {
    u32x4 r;
    r[0] = v; r[1] = v; r[2] = v; r[3] = v;
    return r;
}

static f32x4 half_gemv_f4_(float v) __attribute__((always_inline)) {
    f32x4 r;
    r[0] = v; r[1] = v; r[2] = v; r[3] = v;
    return r;
}

static f32x4 half_gemv_as_f32x4_(u32x4 u) __attribute__((always_inline)) {
    f32x4 f;
    unsafe { memcpy((void*)&f, (const void*)&u, 16UL); }
    return f;
}

static u32x4 half_gemv_as_u32x4_(f32x4 f) __attribute__((always_inline)) {
    u32x4 u;
    unsafe { memcpy((void*)&u, (const void*)&f, 16UL); }
    return u;
}

// SafeC's vec<> has no bitwise operators, so the masks and shifts below are
// written as multiplies and divides by powers of two (and disjoint bit
// fields are joined with +); LLVM folds them back into shifts and ANDs.
//
// fp16 values in the low 16 bits of each lane -> float32 (exact, incl.
// subnormals, signed zeros, inf and NaN).
static f32x4 half_gemv_fp16x4_(u32x4 lane) __attribute__((always_inline)) {
    u32x4 k64 = half_gemv_u4_(65536U);
    u32x4 k32 = half_gemv_u4_(32768U);
    u32x4 h = lane - (lane / k64) * k64;                          // lane & 0xffff
    u32x4 sgn = h / k32;                                         // 0 or 1
    u32x4 m = (h - sgn * k32) * half_gemv_u4_(8192U);          // (h & 0x7fff) << 13
    f32x4 v = half_gemv_as_f32x4_(m) * half_gemv_f4_((float)5.192296858534828e33);   // 2^112
    u32x4 r = half_gemv_as_u32x4_(v);
    // inf / NaN (exponent all ones): (h & 0x7fff) / 0x7c00 is 1 exactly
    // then, 0 otherwise -- a blend without a per-lane branch.
    u32x4 one = half_gemv_u4_(1U);
    u32x4 special = (h - sgn * k32) / half_gemv_u4_(0x7c00U);
    r = r * (one - special) + (m + half_gemv_u4_(0x7f800000U)) * special;
    return half_gemv_as_f32x4_(r + sgn * half_gemv_u4_(0x80000000U));
}

static void half_gemv_pair_(u32x4 lane, int bf, &f32x4 even, &f32x4 odd) __attribute__((always_inline)) {
    u32x4 k64 = half_gemv_u4_(65536U);
    u32x4 hi = lane / k64;                                       // lane >> 16
    if (bf) {
        *even = half_gemv_as_f32x4_(lane * k64);                 // lane << 16
        *odd = half_gemv_as_f32x4_(hi * k64);                    // lane & 0xffff0000
    } else {
        *even = half_gemv_fp16x4_(lane);
        *odd = half_gemv_fp16x4_(hi);
    }
}

static float half_gemv_hsum_(f32x4 a, f32x4 b) __attribute__((always_inline)) {
    f32x4 s = a + b;
    return (s[0] + s[1]) + (s[2] + s[3]);
}

static void half_gemv_rows_neon_(&HalfGemvJob j, unsigned long lo, unsigned long hi) {
    unsigned long k = j.k;
    unsigned long k8 = k - k % 8UL;
    int bf = j.bf;
    unsafe {
        unsigned long r = lo;
        while (r < hi) {
            unsigned long rows = hi - r < 4UL ? hi - r : 4UL;
            const unsigned short* w0 = j.w + r * k;
            const unsigned short* w1 = rows > 1UL ? w0 + k : w0;
            const unsigned short* w2 = rows > 2UL ? w0 + 2UL * k : w0;
            const unsigned short* w3 = rows > 3UL ? w0 + 3UL * k : w0;
            f32x4 z = half_gemv_f4_((float)0.0);
            f32x4 a0 = z; f32x4 b0 = z; f32x4 a1 = z; f32x4 b1 = z;
            f32x4 a2 = z; f32x4 b2 = z; f32x4 a3 = z; f32x4 b3 = z;
            unsigned long c = 0UL;
            while (c < k8) {
                f32x4 xe = *(const f32x4*)&j.xe[c / 2UL];
                f32x4 xo = *(const f32x4*)&j.xo[c / 2UL];
                f32x4 we; f32x4 wo;
                half_gemv_pair_(*(const u32x4*)&w0[c], bf, &we, &wo);
                a0 = __builtin_fmuladd(we, xe, a0); b0 = __builtin_fmuladd(wo, xo, b0);
                half_gemv_pair_(*(const u32x4*)&w1[c], bf, &we, &wo);
                a1 = __builtin_fmuladd(we, xe, a1); b1 = __builtin_fmuladd(wo, xo, b1);
                half_gemv_pair_(*(const u32x4*)&w2[c], bf, &we, &wo);
                a2 = __builtin_fmuladd(we, xe, a2); b2 = __builtin_fmuladd(wo, xo, b2);
                half_gemv_pair_(*(const u32x4*)&w3[c], bf, &we, &wo);
                a3 = __builtin_fmuladd(we, xe, a3); b3 = __builtin_fmuladd(wo, xo, b3);
                c = c + 8UL;
            }
            float acc[4];
            acc[0] = half_gemv_hsum_(a0, b0); acc[1] = half_gemv_hsum_(a1, b1);
            acc[2] = half_gemv_hsum_(a2, b2); acc[3] = half_gemv_hsum_(a3, b3);
            unsigned long i = 0UL;
            while (i < rows) {
                const unsigned short* w = j.w + (r + i) * k;
                float a = acc[i];
                unsigned long t = k8;
                while (t < k) { a = a + half_gemv_w_(w[t], bf) * j.x[t]; t = t + 1UL; }
                j.y[r + i] = a;
                i = i + 1UL;
            }
            r = r + rows;
        }
    }
}
#endif
#if defined(__x86_64__)
static void half_gemv_rows_avx2_(&HalfGemvJob j, unsigned long lo, unsigned long hi)
    __attribute__((target("avx2,fma"))) { half_gemv_rows_body_(j, lo, hi); }
static void half_gemv_rows_avx512_(&HalfGemvJob j, unsigned long lo, unsigned long hi)
    __attribute__((target("avx512f,avx512bw,avx512vl,avx2,fma"))) { half_gemv_rows_body_(j, lo, hi); }
#endif

static void half_gemv_rows_(&HalfGemvJob j, unsigned long lo, unsigned long hi) {
#if defined(__aarch64__)
    half_gemv_rows_neon_(j, lo, hi);
    return;
#endif
#if defined(__x86_64__)
    int l = simd_level();
    if (l >= SIMD_AVX512) { half_gemv_rows_avx512_(j, lo, hi); return; }
    if (l >= SIMD_AVX2) { half_gemv_rows_avx2_(j, lo, hi); return; }
#endif
    half_gemv_rows_base_(j, lo, hi);
}

// m vectors at once: each chunk of rows is walked in blocks small enough to
// stay in L1/L2, and every vector runs over a block before the next one, so
// the weights come from memory once per call instead of once per vector.
struct HalfGemvBatch {
    struct HalfGemvJob j;
    unsigned long m;
    unsigned long n;
    unsigned long half;   // per-vector stride of the xe/xo split (AArch64)
};

#if defined(__aarch64__)
// Two rows against up to four vectors (v0 .. v0+nv-1): each weight is widened
// once and used for every vector, which the per-vector kernel can't do.
static void half_gemv_multi_neon_(&HalfGemvBatch b, unsigned long lo, unsigned long hi,
                                  unsigned long v0, unsigned long nv) {
    unsigned long k = b.j.k;
    unsigned long k8 = k - k % 8UL;
    unsigned long st = 2UL * b.half;
    int bf = b.j.bf;
    unsafe {
        // vectors past nv alias the first one; their sums are dropped
        const float* e0 = b.j.xe + v0 * st;
        const float* o0 = b.j.xo + v0 * st;
        const float* e1 = nv > 1UL ? e0 + st : e0;  const float* o1 = nv > 1UL ? o0 + st : o0;
        const float* e2 = nv > 2UL ? e0 + 2UL * st : e0;  const float* o2 = nv > 2UL ? o0 + 2UL * st : o0;
        const float* e3 = nv > 3UL ? e0 + 3UL * st : e0;  const float* o3 = nv > 3UL ? o0 + 3UL * st : o0;
        unsigned long r = lo;
        while (r < hi) {
            unsigned long rows = hi - r < 2UL ? 1UL : 2UL;
            const unsigned short* w0 = b.j.w + r * k;
            const unsigned short* w1 = rows > 1UL ? w0 + k : w0;
            f32x4 z = half_gemv_f4_((float)0.0);
            // row 0 / row 1 against vectors 0..3; even and odd halves apart so
            // no accumulator waits on two FMAs per step
            f32x4 p0 = z; f32x4 p1 = z; f32x4 p2 = z; f32x4 p3 = z;
            f32x4 q0 = z; f32x4 q1 = z; f32x4 q2 = z; f32x4 q3 = z;
            f32x4 s0 = z; f32x4 s1 = z; f32x4 s2 = z; f32x4 s3 = z;
            f32x4 t0 = z; f32x4 t1 = z; f32x4 t2 = z; f32x4 t3 = z;
            unsigned long c = 0UL;
            while (c < k8) {
                unsigned long h = c / 2UL;
                f32x4 we; f32x4 wo; f32x4 ue; f32x4 uo;
                half_gemv_pair_(*(const u32x4*)&w0[c], bf, &we, &wo);
                half_gemv_pair_(*(const u32x4*)&w1[c], bf, &ue, &uo);
                f32x4 xe = *(const f32x4*)&e0[h]; f32x4 xo = *(const f32x4*)&o0[h];
                p0 = __builtin_fmuladd(we, xe, p0); s0 = __builtin_fmuladd(wo, xo, s0);
                q0 = __builtin_fmuladd(ue, xe, q0); t0 = __builtin_fmuladd(uo, xo, t0);
                xe = *(const f32x4*)&e1[h]; xo = *(const f32x4*)&o1[h];
                p1 = __builtin_fmuladd(we, xe, p1); s1 = __builtin_fmuladd(wo, xo, s1);
                q1 = __builtin_fmuladd(ue, xe, q1); t1 = __builtin_fmuladd(uo, xo, t1);
                xe = *(const f32x4*)&e2[h]; xo = *(const f32x4*)&o2[h];
                p2 = __builtin_fmuladd(we, xe, p2); s2 = __builtin_fmuladd(wo, xo, s2);
                q2 = __builtin_fmuladd(ue, xe, q2); t2 = __builtin_fmuladd(uo, xo, t2);
                xe = *(const f32x4*)&e3[h]; xo = *(const f32x4*)&o3[h];
                p3 = __builtin_fmuladd(we, xe, p3); s3 = __builtin_fmuladd(wo, xo, s3);
                q3 = __builtin_fmuladd(ue, xe, q3); t3 = __builtin_fmuladd(uo, xo, t3);
                c = c + 8UL;
            }
            float acc[8];
            acc[0] = half_gemv_hsum_(p0, s0); acc[1] = half_gemv_hsum_(p1, s1);
            acc[2] = half_gemv_hsum_(p2, s2); acc[3] = half_gemv_hsum_(p3, s3);
            acc[4] = half_gemv_hsum_(q0, t0); acc[5] = half_gemv_hsum_(q1, t1);
            acc[6] = half_gemv_hsum_(q2, t2); acc[7] = half_gemv_hsum_(q3, t3);
            unsigned long i = 0UL;
            while (i < rows) {
                const unsigned short* w = b.j.w + (r + i) * k;
                unsigned long v = 0UL;
                while (v < nv) {
                    const float* x = b.j.x + (v0 + v) * k;
                    float a = acc[i * 4UL + v];
                    unsigned long t = k8;
                    while (t < k) { a = a + half_gemv_w_(w[t], bf) * x[t]; t = t + 1UL; }
                    b.j.y[(v0 + v) * b.n + r + i] = a;
                    v = v + 1UL;
                }
                i = i + 1UL;
            }
            r = r + rows;
        }
    }
}
#endif

static void half_gemv_batch_rows_(&HalfGemvBatch b, unsigned long lo, unsigned long hi) {
#if defined(__aarch64__)
    if (b.m > 1UL) {
        unsigned long block = 16384UL / b.j.k;   // ~32 KB of weights
        if (block < 2UL) block = 2UL;
        unsigned long r = lo;
        while (r < hi) {
            unsigned long e = hi - r < block ? hi : r + block;
            unsigned long v = 0UL;
            while (v < b.m) {
                unsigned long nv = b.m - v < 4UL ? b.m - v : 4UL;
                half_gemv_multi_neon_(b, r, e, v, nv);
                v = v + nv;
            }
            r = e;
        }
        return;
    }
#endif
    unsigned long block = 16384UL / b.j.k;   // ~32 KB of weights
    if (block < 4UL) block = 4UL;
    struct HalfGemvJob v = b.j;
    unsigned long r = lo;
    while (r < hi) {
        unsigned long e = hi - r < block ? hi : r + block;
        unsigned long i = 0UL;
        while (i < b.m) {
            unsafe {
                v.x = b.j.x + i * b.j.k;
                v.y = b.j.y + i * b.n;
                v.xe = b.j.xe + i * 2UL * b.half;
                v.xo = b.j.xo + i * 2UL * b.half;
            }
            half_gemv_rows_(&v, r, e);
            i = i + 1UL;
        }
        r = e;
    }
}

int HalfTensor::gemv_batch(const &float x, unsigned long m, &float y) const {
    if (self.ndim != 2UL || self.shape[0] == 0UL || self.shape[1] == 0UL) return 0;
    if (m == 0UL) return 1;
    struct HalfGemvBatch b;
    unsafe {
        b.j.w = (const unsigned short*)self.data;
        b.j.x = (const float*)x;
        b.j.y = (float*)y;
        b.j.xe = (const float*)0;
        b.j.xo = (const float*)0;
    }
    b.j.k = self.shape[1];
    b.j.bf = self.dtype == TENSOR_BF16 ? 1 : 0;
    b.m = m;
    b.n = self.shape[0];
    b.half = 0UL;
    unsigned long n = b.n;
    float* split = (float*)0;
#if defined(__aarch64__)
    unsafe {
        unsigned long k = b.j.k;
        unsigned long half = k / 2UL + 4UL;
        split = (float*)malloc(sizeof(float) * 2UL * half * m);
        if (split == (float*)0) return 0;
        unsigned long i = 0UL;
        while (i < m) {
            const float* xi = b.j.x + i * k;
            float* s = &split[i * 2UL * half];
            unsigned long c = 0UL;
            while (c + 1UL < k) { s[c / 2UL] = xi[c]; s[half + c / 2UL] = xi[c + 1UL]; c = c + 2UL; }
            i = i + 1UL;
        }
        b.half = half;
        b.j.xe = (const float*)split;
        b.j.xo = (const float*)&split[half];
    }
#endif
    // at least ~256 KB of weights per chunk, so small layers stay on one thread
    unsigned long grain = 131072UL / b.j.k;
    if (grain < 1UL) grain = 1UL;
    if (n * b.j.k >= 262144UL) parallel_for(n, grain, half_gemv_batch_rows_, &b);
    else half_gemv_batch_rows_(&b, 0UL, n);
    unsafe { if (split != (float*)0) free((void*)split); }
    return 1;
}

int HalfTensor::gemv(const &float x, &float y) const { return self.gemv_batch(x, 1UL, y); }

} // namespace std
