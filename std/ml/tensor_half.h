#pragma once
// SafeC Standard Library — fp16 / bf16 tensors and mixed-precision ops.
//
// SafeC has no native 16-bit float scalar (see float16.h's header
// comment), so half-precision tensors are carried the way float16.h
// carries single values: raw IEEE binary16 / bfloat16 bits in 'unsigned
// short', converted with float16.h's round-to-nearest-even routines.
//
// Two layers:
//
//  1. HalfTensor (fp16) and BF16Tensor (bf16) — tensor.h's DenseTensor
//     stored in 16 bits (half the memory of a FloatTensor). Every op, its
//     autograd and the backward pass work on them the same as on a
//     FloatTensor: the op loads its operands to float32, runs the float32
//     kernel and rounds its result once (tensor_matmul accumulates every
//     dot product in float32, like tensor cores / Apple's simdgroup
//     matrices do); gradients are float32. This header adds conversions
//     between element types, the batch helpers mixed-precision training
//     uses, and gemv for decoding with 16-bit weights.
//
//  2. Mixed precision on float32 tensors: FloatTensor::cast() rounds a
//     tensor's values to fp16/bf16 precision in the forward pass and passes
//     gradients through unchanged in the backward pass (the straight-
//     through estimator), so a graph can compute in reduced precision while
//     its master weights and gradients stay float32. tensor_matmul_lowp()
//     is the common case built from it. LossScaler is fp16 training's
//     dynamic loss scaling: scale the loss up so small gradients survive
//     fp16's narrow range, then unscale and skip the step when any gradient
//     overflowed. (bf16 shares float32's exponent range and doesn't need it.)
//
// GPU half-precision compute lives in gpu_mps.h / gpu_cuda.h /
// gpu_webgpu.h (MpsBuf::upload_bits / upload_f16 etc.); a HalfTensor's or
// BF16Tensor's 'data' buffer is exactly the bit layout those functions read
// and write.
#include <std/ml/tensor.h>
#include <std/ml/float16.h>
#include <std/collections/vec.h>

#define TENSOR_FP16 1
#define TENSOR_BF16 2

namespace std {

// ── Precision helpers ────────────────────────────────────────────────────────
// Round 'v' to the nearest value representable in 'dtype' (returned as
// float32). Any dtype other than TENSOR_FP16/TENSOR_BF16 returns 'v'.
float half_round(float v, int dtype);
unsigned short half_encode(float v, int dtype);
float          half_decode(unsigned short bits, int dtype);

// ── Conversion between element types ────────────────────────────────────────
// dst's values = src's, rounded to dst's element type (any pair of
// FloatTensor / HalfTensor / BF16Tensor, including fp16 <-> bf16). Shapes
// may differ; sizes must match (0 otherwise). Not a graph node -- for that,
// see DenseTensor::to_float / from_float.
generic<S, D> int tensor_convert_into(const &DenseTensor<S> src, &DenseTensor<D> dst);

// ── Batch conversion for mixed precision ────────────────────────────────────
// The usual layout: float32 master weights (what the optimizer updates), a
// 16-bit copy refreshed after every step for the low-precision forward pass
// or for uploading to a GPU, and 16-bit gradients brought back into the
// masters' float32 gradients with the loss scale divided out. All of these
// go through float16.h's bulk conversions (hardware where the CPU has it,
// split across the thread pool on large tensors).
//
//   struct Vec<&BF16Tensor> w16 = vec_new();
//   tensors_convert(&net.params, &w16);
//   ... opt.step();
//   tensors_convert_into(&net.params, &w16);           // refresh the copy
//   ... after a GPU backward that produced fp16 gradients g16:
//   int ok = half_grads_accumulate(&g16, &net.params, (float)1.0 / scaler.scale);
//   scaler.update(ok);

// Appends one converted copy of each tensor in 'src' to 'out' (heap,
// requiresGrad 0; free with tensors_free).
generic<S, D> void tensors_convert(const &Vec<&DenseTensor<S>> src, &Vec<&DenseTensor<D>> out);
// Refresh copies made by tensors_convert, or write them back; 0 if the
// lists differ in length or any pair in size.
generic<S, D> int tensors_convert_into(const &Vec<&DenseTensor<S>> src, &Vec<&DenseTensor<D>> dst);
// Frees every tensor in the list (DenseTensor::free and its own memory)
// and the list.
generic<E> void tensors_free(&Vec<&DenseTensor<E>> ts);
// params[i].grad += grads[i] * invScale (allocating the gradient if
// needed). Returns 1 if every resulting gradient value is finite -- the
// LossScaler::update argument -- and 0 on an overflow or a size mismatch.
generic<E> int half_grads_accumulate(const &Vec<&DenseTensor<E>> grads, &Vec<&FloatTensor> params, float invScale);

// ── Decoding (one token at a time) ───────────────────────────────────────────
// y = w x for a linear layer's weight 'w' (2D, [n][k]) and float32 vectors
// x (k values) and y (n values): the matrix-vector product of decoding one
// token, which is bound by reading the weights. Each weight is widened in
// registers (no float32 copy of the matrix), accumulated in float32,
// vectorized for the CPU's SIMD tier and split over rows on
// std/sync/parallel.h's pool. 1 on success, 0 if 'w' isn't 2D.
int HalfTensor::gemv(const &float x, &float y) const;
int BF16Tensor::gemv(const &float x, &float y) const;
// The same for m vectors: x is [m][k] and y is [m][n], row-major. The
// weights are read from memory once for all m (a short prompt's prefill,
// speculative verification), not once per vector.
int HalfTensor::gemv_batch(const &float x, unsigned long m, &float y) const;
int BF16Tensor::gemv_batch(const &float x, unsigned long m, &float y) const;

// ── Mixed precision with autograd (float32 Tensors) ─────────────────────────
// Forward: each element rounded to 'dtype' precision. Backward: gradient
// passed through unchanged (straight-through estimator).
&FloatTensor FloatTensor::cast(int dtype) const;
// cast(cast(a) @ cast(b)): operands and result rounded to 'dtype', products
// accumulated in float32. Gradients flow to a and b in float32.
&FloatTensor tensor_matmul_lowp(const &FloatTensor a, const &FloatTensor b, int dtype);

// ── Dynamic loss scaling (fp16 training) ────────────────────────────────────
struct LossScaler {
    float scale;           // multiply the loss by this before Tensor::backward()
    float growthFactor;    // scale *= growthFactor after growthInterval clean steps
    float backoffFactor;   // scale *= backoffFactor after an overflow
    int   growthInterval;
    int   goodSteps;       // clean steps since the last change

    // Call after Tensor::backward() on (loss * scale): divides p.grad by the
    // scale in place and returns 1 if every gradient is finite, or 0 if any
    // gradient overflowed (inf/NaN) — skip the optimizer step in that case.
    int unscale(&FloatTensor p);
    // LossScaler::unscale over every tensor in a list; 1 if all are finite.
    int unscale_all(&Vec<&FloatTensor> params);
    // Call once per step with the combined result of LossScaler::unscale over
    // all parameters (1 = all finite). Grows or backs off the scale.
    void update(int allFinite);

    // scale 65536, growth 2x every 2000 clean steps, backoff 0.5x — the
    // defaults PyTorch's GradScaler uses.
    static struct LossScaler new();
};


// ══ Generic implementation ═══════════════════════════════════════════════════

// n values of t from element 'off' on, as float32 / from float32.
generic<E>
static void __half_load(const &DenseTensor<E> t, unsigned long off, &float dst, unsigned long n) {
    if const (sizeof(E) == 4UL) {
        unsafe { __tensor_ew(EW_COPY, (const float*)&t->data[off], (const float*)0, (float)0.0, (float*)dst, n); }
    } else {
        unsafe { E::load_n((const &E)&t->data[off], dst, n); }
    }
}

generic<E>
static void __half_store(&DenseTensor<E> t, unsigned long off, const &float src, unsigned long n) {
    if const (sizeof(E) == 4UL) {
        unsafe { __tensor_ew(EW_COPY, (const float*)src, (const float*)0, (float)0.0, (float*)&t->data[off], n); }
    } else {
        unsafe { E::store_n(src, (&E)&t->data[off], n); }
    }
}

#define HALF_CHUNK 4096UL

generic<S, D>
int tensor_convert_into(const &DenseTensor<S> src, &DenseTensor<D> dst) {
    if (src.size != dst.size) return 0;
    if const (sizeof(S) == 4UL) {
        unsafe { __half_store(dst, 0UL, (const &float)src->data, src->size); }
        return 1;
    }
    if const (sizeof(D) == 4UL) {
        unsafe { __half_load(src, 0UL, (&float)dst->data, src->size); }
        return 1;
    }
    // 16 bits to 16 bits through a float32 chunk (exact: every fp16 and bf16
    // value is a float32 value)
    float buf[4096];
    unsigned long i = 0UL;
    while (i < src.size) {
        unsigned long n = src.size - i < HALF_CHUNK ? src.size - i : HALF_CHUNK;
        unsafe {
            __half_load(src, i, (&float)&buf[0], n);
            __half_store(dst, i, (const &float)&buf[0], n);
        }
        i = i + n;
    }
    return 1;
}

generic<S, D>
void tensors_convert(const &Vec<&DenseTensor<S>> src, &Vec<&DenseTensor<D>> out) {
    unsigned long i = 0UL;
    while (i < src.length()) {
        &DenseTensor<S> s = *src.at(i);
        &DenseTensor<D> c;
        unsafe { c = __dt_alloc((const unsigned long*)s->shape, s->ndim, 0); }
        tensor_convert_into(s, c);
        out.push(c);
        i = i + 1UL;
    }
}

generic<S, D>
int tensors_convert_into(const &Vec<&DenseTensor<S>> src, &Vec<&DenseTensor<D>> dst) {
    if (src.length() != dst.length()) return 0;
    int ok = 1;
    unsigned long i = 0UL;
    while (i < src.length()) {
        if (!tensor_convert_into(*src.at(i), *dst.at(i))) ok = 0;
        i = i + 1UL;
    }
    return ok;
}

generic<E>
void tensors_free(&Vec<&DenseTensor<E>> ts) {
    unsigned long i = 0UL;
    while (i < ts.length()) {
        &DenseTensor<E> t = *ts.at(i);
        int onHeap = 1;
        match (t.owner) { case some(o): { onHeap = 0; } case null: { } }
        unsafe { t.free(); }   // a heap tensor: destroyed in place, its memory freed below
        if (onHeap) { unsafe { std::free((void*)t); } }
        i = i + 1UL;
    }
    ts.free();
}

generic<E>
int half_grads_accumulate(const &Vec<&DenseTensor<E>> grads, &Vec<&FloatTensor> params, float invScale) {
    if (grads.length() != params.length()) return 0;
    int finite = 1;
    float buf[4096];
    unsigned long t = 0UL;
    while (t < grads.length()) {
        &DenseTensor<E> g = *grads.at(t);
        &FloatTensor p = *params.at(t);
        if (g.size != p.size) return 0;
        unsafe {
            __tensor_ensure_grad((FloatTensor*)p);
            unsigned long i = 0UL;
            while (i < g.size) {
                unsigned long n = g.size - i < HALF_CHUNK ? g.size - i : HALF_CHUNK;
                __half_load(g, i, (&float)&buf[0], n);
                unsigned long k = 0UL;
                while (k < n) {
                    float v = p->grad[i + k] + buf[k] * invScale;
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

} // namespace std
