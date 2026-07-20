#pragma once
// SafeC Standard Library — size-based automatic backend selection for
// Tensor ops (mixing SIMD/vDSP/BLAS the way tensor_dispatch.h mixes
// explicit devices, but chosen FOR the caller instead of BY the caller).
//
// tensor_dispatch.h's tensor_matmul_on(a, b, device) already lets a
// caller pick a backend explicitly. What it doesn't do is pick the RIGHT
// one automatically — every backend has a size range where it's the
// wrong choice (BLAS's own per-call setup overhead costs more than a
// tiny naive loop saves; vDSP has the same shape of overhead; GPU's
// command-buffer submission cost dominates below a problem size that,
// in this codebase's own benchmarks, no shape tried so far has reached).
// The functions here measure that instead of guessing, and pick per
// call — see tensor_auto.sc for the actual thresholds and how they were
// obtained.
//
// Deliberately CPU-only for now: matmul benchmarked from 2x2x2 up to
// 128x128x128 never found a size where the naive loop beat BLAS (BLAS
// won or tied at every point tested — see tensor_auto.sc), and this
// project's own from-scratch GPU benchmarks (see benchmarks.md's "GPU is
// slower than CPU here" and "GPU backward was the real bottleneck"
// findings) never found a shape where MPS beat BLAS either, up to and
// including a 128x512x1024x256 training step. Auto-selecting GPU would
// mean picking it on the strength of zero measured wins — not done here.
// A caller who wants GPU regardless (e.g. because their real workload is
// larger than anything benchmarked so far, or to free up the CPU for
// other work) should keep using tensor_matmul_on(..., DEVICE_MPS)
// explicitly; that path doesn't disappear, this file just doesn't add
// GPU to what "auto" means until there's a measured case for it.
//
// Apple's Neural Engine (ANE/"NPU") isn't offered as an option at all,
// automatic or explicit: unlike Metal, there is no public low-level API
// to submit an arbitrary compute kernel to the ANE the way gpu_mps.sc
// submits one to the GPU. The only supported path is CoreML — hand it a
// whole compiled model graph (an .mlmodel/.mlpackage) via MLModel and let
// CoreML's own scheduler decide ANE vs GPU vs CPU internally, not "call a
// function, get a kernel to run." That's a fundamentally different
// integration (author/compile a model graph, not link a kernel function)
// and a substantially larger undertaking than everything else in std/ml —
// not attempted here, and not something this file's Device-enum-style
// dispatch could grow into incrementally.
namespace std {

&Tensor tensor_add_auto(const &Tensor a, const &Tensor b);
&Tensor tensor_sub_auto(const &Tensor a, const &Tensor b);
&Tensor tensor_mul_auto(const &Tensor a, const &Tensor b);
&Tensor tensor_sum_auto(const &Tensor a);
&Tensor tensor_matmul_auto(const &Tensor a, const &Tensor b);

} // namespace std
