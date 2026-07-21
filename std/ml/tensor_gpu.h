#pragma once
// SafeC Standard Library — GPU-backed Tensor ops (Metal/MPS).
//
// Separate from tensor.h/tensor.sc on purpose: tensor.sc's CPU path is
// portable (no framework dependency), while this pulls in gpu_mps.h/.sc,
// which only links/runs on macOS with Metal (see gpu_mps.h's header
// comment) — the same "gated behind a feature, not baked into the
// portable core" convention gpu_mps.h itself already follows. Include this
// file in addition to tensor.sc/tensor.h, not instead of them.
//
// Every op below is the GPU-dispatched twin of the same-named tensor.sc
// function (tensor_add_gpu <-> tensor_add, etc): same forward math, linked
// into the same autograd graph via the same backward function (backward
// doesn't care which backend computed forward, so nothing autograd-side
// needed to change), same fallback behavior. Tensor is float32 (see
// tensor.h), the same precision Metal itself uses (Metal has no 'double'
// at all) — every op here now reads/writes a Tensor's 'data'/'grad'
// buffers directly with mps_*_f32, no per-call CPU-side conversion pass
// the way this file needed when Tensor was double. What's left is GPU
// kernel dispatch's own real fixed overhead (command buffer setup,
// submission, waitUntilCompleted), which a small op's actual compute time
// won't hide — see the benchmarks page's "GPU is slower than CPU here...
// and that's correct" finding for PyTorch-MPS/TensorFlow-GPU, and the
// follow-up "GPU wins here" finding once the shape is big enough to hide
// that overhead behind — the same physics applies to every op here, not
// just matmul.
// Every op falls back to its ordinary CPU tensor.sc twin if no Metal
// device is available (mps_available() == 0) or the GPU dispatch itself
// fails for any other reason, so all of these are always safe to call —
// just not always actually running on the GPU.
#include <std/ml/tensor.h>

namespace std {

&Tensor tensor_add_gpu(const &Tensor a, const &Tensor b);
// tensor_sub_gpu is chain-aware (see tensor_gpu.sc's chain-tracking
// comment): under an open batch, both its forward and its backward
// (__sub_backward_gpu) participate in the fused forward+loss+backward
// training-step path -- e.g. diff = tensor_sub_gpu(pred, target) reading
// 'pred' straight from a still-pending matmul/relu output, no CPU round
// trip, and diff's own gradient propagating onward the same way.
&Tensor tensor_sub_gpu(const &Tensor a, const &Tensor b);
&Tensor tensor_mul_gpu(const &Tensor a, const &Tensor b);
&Tensor tensor_scale_gpu(const &Tensor a, float k);
&Tensor tensor_relu_gpu(const &Tensor a);
// out = a * a -- chain-aware self-multiply, used for the loss computation's
// sq = diff * diff. Not just tensor_mul_gpu(a, a): when 'a' is itself still
// GPU-pending (the usual case here, diff coming straight from a batched
// tensor_sub_gpu), tensor_mul_gpu's generic two-CPU-array signature can't
// express "both operands are the same still-pending buffer" -- see
// gpu_mps.h's mps_square_f32_chained comment.
&Tensor tensor_square_gpu(const &Tensor a);
// tensor_sum_gpu's GPU reduction (std::mps_sum_f32) is a two-stage
// parallel reduction (grid-stride accumulate + threadgroup tree
// reduction) -- see gpu_mps.sc's comment on mps_sum_f32. Chain-aware like
// tensor_sub_gpu/tensor_square_gpu above.
&Tensor tensor_sum_gpu(const &Tensor a);
&Tensor tensor_matmul_gpu(const &Tensor a, const &Tensor b);

// ── Batched dispatch (chains of tensor_*_gpu ops with no CPU step
// between them, e.g. matmul -> relu -> matmul) ──────────────────────────────
// Thin wrappers around gpu_mps.h's mps_batch_begin/mps_batch_end that
// also reset tensor_gpu.sc's own "which Tensor*s are still-pending
// GPU-only results" chain-tracking table. That table has to be cleared
// at the exact moment a NEW batch starts, not lazily whenever the
// tracking code next notices no batch is active — by the time any
// tensor_*_gpu call runs, mps_batch_begin() has already flipped
// mps_batch_is_active() to true, so "reset if idle" checked from inside
// an op is always too late, and stale entries from a PREVIOUS batch
// silently accumulate (found via direct testing across a real 100-step
// training loop: the table overflowing its fixed size after enough
// iterations caused later chained ops to silently fall back to reading
// stale CPU data again — the exact bug mps_batch_begin's own doc
// comment describes, just delayed rather than immediate). Call these,
// not gpu_mps.h's mps_batch_begin/end directly, whenever the batch
// contains any tensor_*_gpu call.
void tensor_gpu_batch_begin();
void tensor_gpu_batch_end();

// ── Cross-batch persistent tensors ──────────────────────────────────────────
// Marks 't' so every tensor_matmul_gpu/tensor_relu_gpu call (forward AND
// the corresponding backward's data lookups) that reads it finds its
// device buffer directly instead of re-uploading (memcpy) t->data fresh on
// every single call — see tensor_gpu.sc's comment on the implementation.
// Only correct for a tensor whose data genuinely never changes for as long
// as it stays marked: a training loop's fixed input X is the case this
// was built for (uploaded once, read by every step's forward AND backward
// matmul instead of BATCH*IN_DIM*sizeof(float) bytes of real memcpy per
// step) — NOT a weight matrix, which SGD rewrites every step. Call
// tensor_gpu_release_all_persistent() once, when truly done with every
// marked tensor (e.g. at the very end of a training run).
void tensor_gpu_mark_persistent(const &Tensor t);
void tensor_gpu_release_all_persistent();

} // namespace std
