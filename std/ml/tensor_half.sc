// SafeC Standard Library — fp16 / bf16 tensors and mixed-precision ops
// (see tensor_half.h).
#include <std/ml/tensor_half.h>

namespace std {

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

struct HalfTensor half_tensor_new_1d(unsigned long n, int dtype) {
    return __half_alloc(n, 1UL, 1UL, dtype);
}

struct HalfTensor half_tensor_new_2d(unsigned long rows, unsigned long cols, int dtype) {
    return __half_alloc(rows, cols, 2UL, dtype);
}

struct HalfTensor tensor_to_half(const &Tensor t, int dtype) {
    struct HalfTensor h;
    unsafe {
        unsigned long d1 = t->ndim == 2UL ? t->shape[1] : 1UL;
        h = __half_alloc(t->shape[0], d1, t->ndim, dtype);
        if (dtype == TENSOR_BF16) f32_to_bf16_bulk((const float*)t->data, (unsigned short*)h.data, t->size);
        else f32_to_fp16_bulk((const float*)t->data, (unsigned short*)h.data, t->size);
    }
    return h;
}

&Tensor half_to_tensor(const &HalfTensor h, int requiresGrad) {
    &Tensor out = h.ndim == 2UL ? tensor_new_2d(h.shape[0], h.shape[1], requiresGrad)
                                : tensor_new_1d(h.shape[0], requiresGrad);
    unsafe {
        if (h.dtype == TENSOR_BF16) bf16_to_f32_bulk((const unsigned short*)h.data, (float*)out->data, h.size);
        else fp16_to_f32_bulk((const unsigned short*)h.data, (float*)out->data, h.size);
    }
    return out;
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

struct HalfTensor half_scale(const &HalfTensor a, float k) {
    struct HalfTensor out = __half_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < a.size) {
            out.data[i] = half_encode(half_decode(a.data[i], a.dtype) * k, a.dtype);
            i = i + 1UL;
        }
    }
    return out;
}

struct HalfTensor half_relu(const &HalfTensor a) {
    struct HalfTensor out = __half_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < a.size) {
            float x = half_decode(a.data[i], a.dtype);
            out.data[i] = x > (float)0.0 ? a.data[i] : (unsigned short)0;
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

float half_sum(const &HalfTensor a) {
    float acc = (float)0.0;
    unsafe {
        unsigned long i = 0UL;
        while (i < a.size) { acc = acc + half_decode(a.data[i], a.dtype); i = i + 1UL; }
    }
    return acc;
}

// ── Mixed precision with autograd ────────────────────────────────────────────

// Straight-through: d(cast(a))/da is treated as the identity.
static void __cast_backward(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (a->requiresGrad) __tensor_accumulate(a, (const float*)selfT->grad);
    }
}

&Tensor tensor_cast(const &Tensor a, int dtype) {
    &Tensor out = __tensor_alloc_uninit_like(a);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) { out->data[i] = half_round(a->data[i], dtype); i = i + 1UL; }
    }
    __tensor_link1(out, a, (void*)__cast_backward);
    return out;
}

&Tensor tensor_matmul_lowp(const &Tensor a, const &Tensor b, int dtype) {
    &Tensor ca = tensor_cast(a, dtype);
    &Tensor cb = tensor_cast(b, dtype);
    return tensor_cast(tensor_matmul(ca, cb), dtype);
}

// ── Dynamic loss scaling ─────────────────────────────────────────────────────

struct LossScaler loss_scaler_new() {
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

int loss_scaler_unscale(&LossScaler s, &Tensor p) {
    int finite = 1;
    unsafe {
        if (p->grad == (float*)0) return 1;
        float inv = (float)1.0 / s.scale;
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

void loss_scaler_update(&LossScaler s, int allFinite) {
    if (!allFinite) {
        s.scale = s.scale * s.backoffFactor;
        if (s.scale < (float)1.0) s.scale = (float)1.0;
        s.goodSteps = 0;
        return;
    }
    s.goodSteps = s.goodSteps + 1;
    if (s.goodSteps >= s.growthInterval) {
        s.scale = s.scale * s.growthFactor;
        s.goodSteps = 0;
    }
}

} // namespace std
