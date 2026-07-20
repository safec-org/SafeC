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
// needed to change), same fallback behavior, and the same two real costs
// traded for GPU execution on every call, not something to reach for by
// default:
//   1. Metal has no 'double' type at all — Tensor's 'double' data gets
//      converted to float32 on the way in and back to 'double' on the way
//      out, on *every* call (not once, cached) since a Tensor doesn't carry
//      a persistent GPU-resident copy of itself between ops. This loses
//      precision (~7 decimal digits instead of ~15-17) and costs a real
//      O(size) conversion pass beyond the GPU dispatch itself.
//   2. GPU kernel dispatch has real fixed overhead (command buffer setup,
//      submission, waitUntilCompleted) that a small op's actual compute
//      time won't hide — see the benchmarks page's "GPU is slower than CPU
//      here... and that's correct" finding for PyTorch-MPS/TensorFlow-GPU,
//      and the follow-up "GPU wins here" finding once the shape is big
//      enough to hide that overhead behind — the same physics applies to
//      every op here, not just matmul.
// Every op falls back to its ordinary CPU tensor.sc twin if no Metal
// device is available (mps_available() == 0) or the GPU dispatch itself
// fails for any other reason, so all of these are always safe to call —
// just not always actually running on the GPU.
#include <std/ml/tensor.h>

namespace std {

&Tensor tensor_add_gpu(const &Tensor a, const &Tensor b);
&Tensor tensor_sub_gpu(const &Tensor a, const &Tensor b);
&Tensor tensor_mul_gpu(const &Tensor a, const &Tensor b);
&Tensor tensor_scale_gpu(const &Tensor a, double k);
&Tensor tensor_relu_gpu(const &Tensor a);
// tensor_sum_gpu's GPU reduction (std::mps_sum_f32) is a single-thread
// serial sum, not a real parallel tree reduction -- see gpu_mps.sc's
// comment on mps_sum_f32. Included for completeness of the op set (every
// other op here is real, dispatched, parallel GPU work); this one mostly
// just isn't slower in a way that matters, since tensor_sum's whole input
// is already read once regardless of backend.
&Tensor tensor_sum_gpu(const &Tensor a);
&Tensor tensor_matmul_gpu(const &Tensor a, const &Tensor b);

} // namespace std
