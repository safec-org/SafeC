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
//  1. HalfTensor — a 1D/2D tensor *stored* in fp16 or bf16 (half the
//     memory of a float32 Tensor), for weights and activations you want to
//     keep small. Its ops widen each operand to float32, compute, and
//     round the result back to the tensor's storage format — the same
//     arithmetic a GPU's half-precision path performs. half_matmul
//     accumulates every dot product in float32 and rounds once at the end,
//     like tensor cores / Apple's simdgroup matrices do, rather than
//     rounding after every multiply-add. No autograd on this layer.
//
//  2. Mixed precision on ordinary float32 Tensors, with autograd:
//     tensor_cast() rounds a Tensor's values to fp16/bf16 precision in the
//     forward pass and passes gradients through unchanged in the backward
//     pass (the straight-through estimator every mixed-precision trainer
//     uses), so a graph can compute in reduced precision while its master
//     weights and gradients stay float32. tensor_matmul_lowp() is the
//     common case built from it. LossScaler is fp16 training's dynamic loss
//     scaling: scale the loss up so small gradients survive fp16's narrow
//     range, then unscale and skip the step when any gradient overflowed.
//     (bf16 shares float32's exponent range and doesn't need it.)
//
// GPU half-precision compute lives in gpu_mps.h / gpu_cuda.h /
// gpu_webgpu.h (mps_upload_f16_persistent etc.); a HalfTensor's 'data'
// buffer is exactly the bit layout those functions read and write.
#include <std/ml/tensor.h>
#include <std/ml/float16.h>
#include <std/collections/vec.h>

#define TENSOR_FP16 1
#define TENSOR_BF16 2

namespace std {

struct HalfTensor {
    &heap unsigned short data;     // flat row-major fp16 or bf16 bits, length == size
    unsigned long        shape[2]; // shape[0..ndim)
    unsigned long        ndim;     // 1 or 2
    unsigned long        size;
    int                  dtype;    // TENSOR_FP16 or TENSOR_BF16

    // Element access, widened to / rounded from float32.
    float at1(unsigned long i) const;
    float at2(unsigned long r, unsigned long c) const;
    void  set1(unsigned long i, float v);
    void  set2(unsigned long r, unsigned long c, float v);

    void  free();
};

// ── Precision helpers ────────────────────────────────────────────────────────
// Round 'v' to the nearest value representable in 'dtype' (returned as
// float32). Any dtype other than TENSOR_FP16/TENSOR_BF16 returns 'v'.
float half_round(float v, int dtype);
unsigned short half_encode(float v, int dtype);
float          half_decode(unsigned short bits, int dtype);

// ── HalfTensor construction and conversion ──────────────────────────────────
struct HalfTensor half_tensor_new_1d(unsigned long n, int dtype);          // zero-filled
struct HalfTensor half_tensor_new_2d(unsigned long rows, unsigned long cols, int dtype);
struct HalfTensor tensor_to_half(const &Tensor t, int dtype);               // round-to-nearest-even
&Tensor           half_to_tensor(const &HalfTensor h, int requiresGrad);    // exact widening

// ── Batch conversion for mixed precision ────────────────────────────────────
// The usual layout: float32 master weights (what the optimizer updates), a
// 16-bit copy refreshed after every step for the low-precision forward pass
// or for uploading to a GPU, and 16-bit gradients brought back into the
// masters' float32 gradients with the loss scale divided out. All of these
// go through float16.h's bulk conversions (hardware where the CPU has it,
// split across the thread pool on large tensors).
//
//   struct Vec<struct HalfTensor> w16 = tensors_to_half(&net.params, TENSOR_BF16);
//   ... optimizer_step(&opt);
//   tensors_to_half_into(&net.params, &w16);            // refresh the copy
//   ... after a GPU backward that produced fp16 gradients g16:
//   int ok = half_grads_accumulate(&g16, &net.params, (float)1.0 / scaler.scale);
//   loss_scaler_update(&scaler, ok);

// Into an existing tensor of the same size (dst keeps its dtype / shape);
// 0 on a size mismatch.
int tensor_to_half_into(const &Tensor t, &HalfTensor dst);
int half_to_tensor_into(const &HalfTensor h, &Tensor dst);
// dst = src in dst's dtype (fp16 <-> bf16, or a copy when they match).
int half_convert(const &HalfTensor src, &HalfTensor dst);
// One 16-bit copy per tensor (heap; free with half_tensors_free).
struct Vec<struct HalfTensor> tensors_to_half(const &Vec<&Tensor> ts, int dtype);
// Refresh copies made by tensors_to_half / write them back; 0 if the lists
// differ in length or any pair in size.
int  tensors_to_half_into(const &Vec<&Tensor> ts, &Vec<struct HalfTensor> dst);
int  halves_to_tensors(const &Vec<struct HalfTensor> hs, &Vec<&Tensor> dst);
void half_tensors_free(&Vec<struct HalfTensor> hs);
// params[i].grad += decode(grads[i]) * invScale (allocating the gradient
// if needed). Returns 1 if every resulting gradient value is finite -- the
// loss_scaler_update argument -- and 0 on an overflow or a size mismatch.
int  half_grads_accumulate(const &Vec<struct HalfTensor> grads, &Vec<&Tensor> params, float invScale);

// ── HalfTensor ops (float32 compute, result rounded to a's dtype) ───────────
// Binary ops require matching shapes and dtypes; on a mismatch the result
// is empty (size 0, data null-equivalent) — check '.size'.
struct HalfTensor half_add(const &HalfTensor a, const &HalfTensor b);
struct HalfTensor half_sub(const &HalfTensor a, const &HalfTensor b);
struct HalfTensor half_mul(const &HalfTensor a, const &HalfTensor b);     // elementwise
struct HalfTensor half_scale(const &HalfTensor a, float k);
struct HalfTensor half_relu(const &HalfTensor a);
struct HalfTensor half_matmul(const &HalfTensor a, const &HalfTensor b);  // 2D x 2D, f32 accumulate
float             half_sum(const &HalfTensor a);                          // f32 accumulate, not rounded

// ── Decoding (one token at a time) ───────────────────────────────────────────
// y = w x for a linear layer's weight 'w' (2D, [n][k], fp16 or bf16) and
// float32 vectors x (k values) and y (n values): the matrix-vector product
// of decoding one token, which is bound by reading the weights. Each weight
// is widened in registers (no float32 copy of the matrix), accumulated in
// float32, vectorized for the CPU's SIMD tier and split over rows on
// std/sync/parallel.h's pool. 1 on success, 0 if 'w' isn't 2D.
int half_gemv(const &HalfTensor w, const &float x, &float y);
// The same for m vectors: x is [m][k] and y is [m][n], row-major. The
// weights are read from memory once for all m (a short prompt's prefill,
// speculative verification), not once per vector.
int half_gemv_batch(const &HalfTensor w, const &float x, unsigned long m, &float y);

// ── Mixed precision with autograd (float32 Tensors) ─────────────────────────
// Forward: each element rounded to 'dtype' precision. Backward: gradient
// passed through unchanged (straight-through estimator).
&Tensor tensor_cast(const &Tensor a, int dtype);
// cast(cast(a) @ cast(b)): operands and result rounded to 'dtype', products
// accumulated in float32. Gradients flow to a and b in float32.
&Tensor tensor_matmul_lowp(const &Tensor a, const &Tensor b, int dtype);

// ── Dynamic loss scaling (fp16 training) ────────────────────────────────────
struct LossScaler {
    float scale;           // multiply the loss by this before tensor_backward()
    float growthFactor;    // scale *= growthFactor after growthInterval clean steps
    float backoffFactor;   // scale *= backoffFactor after an overflow
    int   growthInterval;
    int   goodSteps;       // clean steps since the last change
};

// scale 65536, growth 2x every 2000 clean steps, backoff 0.5x — the
// defaults PyTorch's GradScaler uses.
struct LossScaler loss_scaler_new();

// Call after tensor_backward() on (loss * scale): divides p.grad by the
// scale in place and returns 1 if every gradient is finite, or 0 if any
// gradient overflowed (inf/NaN) — skip the optimizer step in that case.
int  loss_scaler_unscale(&LossScaler s, &Tensor p);
// loss_scaler_unscale over every tensor in a list; 1 if all are finite.
int  loss_scaler_unscale_all(&LossScaler s, &Vec<&Tensor> params);
// Call once per step with the combined result of loss_scaler_unscale over
// all parameters (1 = all finite). Grows or backs off the scale.
void loss_scaler_update(&LossScaler s, int allFinite);

} // namespace std
