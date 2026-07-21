#pragma once
// SafeC Standard Library — GPU tensor ops via Metal (Apple Silicon/macOS).
//
// The "unified memory" piece of the ML story (MLX's actual selling point):
// a Metal buffer created with the default storage mode is CPU+GPU shared
// memory on Apple Silicon — no explicit host<->device copy step the way
// CUDA/ROCm need (see gpu_cuda.h/gpu_rocm.h's discrete-memory model) —
// 'contents' on an MTLBuffer just IS a pointer the CPU can read straight
// out of after the GPU finishes writing it.
//
// Same "no real Objective-C support in safec" situation as
// std/gui/gui_cocoa.sc — every call here goes through the plain-C
// Objective-C runtime (objc_msgSend et al.) and the Metal framework's own
// C entry point (MTLCreateSystemDefaultDevice). See that file's header
// comment for the full technique (typedef fn-cast objc_msgSend per
// message shape). Only linkable/runnable on macOS with a Metal-capable
// GPU (every Apple Silicon Mac, and most Intel Macs from the last
// decade) — link with '-framework Metal -framework Foundation'.
//
// Gated behind the 'mps' feature (see Package.toml's [features] — same
// convention as every other backend-specific piece of std/ this session
// added): only compile/link this on a build that opted in, since a
// non-Apple target obviously can't provide these frameworks.
namespace std {

// Elementwise 'out[i] = a[i] + b[i]' for i in [0, n), computed on the GPU
// via a small inline Metal Shading Language compute kernel compiled at
// runtime. Returns 1 on success, 0 if no Metal device is available or any
// step of the pipeline setup fails (compile error in the embedded MSL
// source, no GPU present, etc.) — 'out' is left untouched on failure.
//
// Verified working end-to-end on real Apple Silicon hardware (correct
// output for a real GPU dispatch). Previously segfaulted at the
// dispatchThreadgroups:threadsPerThreadgroup: call specifically — root
// cause was a genuine SafeC compiler codegen bug (fixed in CodeGen.cpp's
// genCall), not anything specific to this file: any indirect call passing
// two or more consecutive struct-by-value arguments over 16 bytes (and not
// a Homogeneous Floating-point Aggregate) through a function-pointer cast
// hit it, because this compiler represented such arguments as raw LLVM
// aggregate values instead of the AAPCS64-correct "caller copies to a
// stack temporary, passes a plain pointer" form — self-consistent for
// SafeC-to-SafeC calls (which is why that never showed up before), but not
// what a real, externally-compiled function like objc_msgSend's target IMP
// expects to receive in its argument registers.
int mps_add_f32(const float* a, const float* b, float* out, unsigned long n);

// Same shape/verification status as mps_add_f32 (elementwise, GPU-executed,
// unified-memory readback) — sub/mul/div/pow are two-input, log/exp/sqrt
// are one-input. mul/div/pow/log/exp/sqrt share mps_add_f32's setup/
// dispatch/readback path (see __mps_run_binary_kernel/__mps_run_unary_kernel
// in gpu_mps.sc), so the fix that made mps_add_f32's GPU dispatch work
// applies to all of them identically — verified directly for
// mps_sub_f32/mps_mul_f32 as representative binary/no-special-case cases;
// div/pow/log/exp/sqrt follow the exact same code path with only the
// one-line kernel body differing. mps_sub_f32 itself is NOT routed through
// __mps_run_binary_kernel (unlike mul/div/pow) — it's batch-aware, same
// reasoning as mps_relu_f32/mps_matmul_f32 (see mps_relu_f32's comment):
// tensor_gpu.sc's tensor_sub_gpu needs to encode it onto a shared batch's
// command buffer for the fused forward+loss+backward training-step path,
// not force its own commit+wait mid-batch.
int mps_sub_f32(const float* a, const float* b, float* out, unsigned long n);
int mps_mul_f32(const float* a, const float* b, float* out, unsigned long n);
int mps_div_f32(const float* a, const float* b, float* out, unsigned long n);
int mps_pow_f32(const float* a, const float* b, float* out, unsigned long n);
int mps_log_f32(const float* a, float* out, unsigned long n);
int mps_exp_f32(const float* a, float* out, unsigned long n);
int mps_sqrt_f32(const float* a, float* out, unsigned long n);
int mps_relu_f32(const float* a, float* out, unsigned long n);
int mps_scale_f32(const float* a, float k, float* out, unsigned long n);

// out[0] = sum(a[0..n)) -- two-stage parallel reduction (grid-stride
// accumulate + threadgroup tree reduction, see gpu_mps_kernels.metal's
// sum_kernel comment), with the small per-threadgroup partial-sums array
// combined on the CPU after one readback. Batch-aware like mps_sub_f32/
// mps_relu_f32 above: under an open batch, the CPU-side partial-sum
// combine can't run until mps_batch_end()'s single wait, so it's deferred
// into a registered finalize callback instead of happening inline here —
// see gpu_mps.sc's implementation comment.
int mps_sum_f32(const float* a, float* out, unsigned long n);

// out[M,N] = a[M,K] . b[K,N], computed on the GPU. float32-only (Metal has
// no 'double' type at all). Internally picks between three kernels
// depending on shape, most-specialized first:
//   - M, K, N all exact multiples of 32: matmul_kernel_smma_multi --
//     threadgroup-memory tiling (data reuse, like matmul_kernel) AND
//     simdgroup_matrix hardware matrix units (like matmul_kernel_smma)
//     together, the same combination PyTorch's MPSMatrixMultiplication and
//     MLX's fastest GEMM paths use.
//   - otherwise, M, K, N all exact multiples of 8: matmul_kernel_smma --
//     simdgroup_matrix alone, no threadgroup-memory tiling.
//   - otherwise: matmul_kernel's ordinary threadgroup-memory tiling, no
//     alignment requirement at all.
// The two simdgroup_matrix kernels require their alignment because
// simdgroup_load/store past a matrix's real bounds is undefined behavior,
// unlike matmul_kernel's explicit boundary checks. See gpu_mps.sc's
// comment on the implementation, and gpu_mps_kernels.metal's comments on
// each kernel, for detail on all three paths.
int mps_matmul_f32(const float* a, const float* b, float* out,
                    unsigned long M, unsigned long K, unsigned long N);

// GPU-backed matmul backward: dA = dC . B^T, dB = A^T . dC (same math
// tensor_blas.sc's __matmul_backward_blas passes to cblas_dgemm via a
// transpose flag; here each is its own kernel instead). Used by
// tensor_gpu.sc's __matmul_backward_gpu so a matmul_gpu forward pass gets
// a matmul_gpu backward pass too, instead of falling back to the CPU
// backward the way it used to — see that function's comment for why that
// fallback was the actual dominant cost in a training loop, not dispatch
// overhead.
//   mps_matmul_abt_f32: out[M,K] = a[M,N] . b^T, b stored [K,N] (dA's shape)
//   mps_matmul_atb_f32: out[K,N] = a^T . b, a stored [M,K] (dB's shape)
// Both are batch-aware (see "Batched dispatch" below) the same way
// mps_matmul_f32/mps_relu_f32 are -- a training step's backward pass is
// its own sequential chain (matmul2-backward -> relu-backward ->
// matmul1-backward, each waiting on the previous), so it pays the same
// per-op command-buffer round-trip cost forward used to before batching.
// Same three-tier smma-multi/smma/tiled shape-based kernel choice as
// mps_matmul_f32 (see its comment) internally, too.
int mps_matmul_abt_f32(const float* a, const float* b, float* out,
                        unsigned long M, unsigned long N, unsigned long K);
int mps_matmul_atb_f32(const float* a, const float* b, float* out,
                        unsigned long M, unsigned long K, unsigned long N);

// GPU-backed relu backward: out[i] = a[i] > 0 ? selfGrad[i] : 0. Same
// shape/verification path as mps_add_f32 (see __mps_run_binary_kernel).
// Batch-aware, same as the two matmul-backward entry points above.
int mps_relu_backward_f32(const float* a, const float* selfGrad, float* out, unsigned long n);

// True if a Metal device is available on this machine at all (checks
// MTLCreateSystemDefaultDevice() != nil) — lets callers fall back to the
// CPU Tensor ops (tensor.h) when there's no GPU, without needing to
// attempt an mps_* call first just to find out.
int mps_available();

// ── Batched dispatch ──────────────────────────────────────────────────────────
// Every mps_*_f32 call above independently creates its own command
// buffer, dispatches, and blocks on waitUntilCompleted before returning
// -- correct, but for a chain of several ops (e.g. matmul -> relu ->
// matmul in a forward pass), paying one full command-buffer-submission +
// host/device sync round trip PER OP is the dominant cost once shader
// compilation is already cached (see mps_matmul_f32's implementation
// comment), especially for small workloads where dispatch latency swamps
// actual compute time.
//
// mps_batch_begin()/mps_batch_end() let a caller wrap several mps_*_f32
// calls so they share ONE command buffer and ONE sync point:
//
//   mps_batch_begin();
//   mps_matmul_f32(x, w1, h, ...);    // encoded onto the shared buffer,
//                                      // NOT dispatched yet -- 'h' is not
//                                      // valid at this point
//   mps_relu_f32(h, r, ...);          // reads 'h' — safe: this dispatch
//                                      // is encoded AFTER matmul's on the
//                                      // same buffer, so Metal orders it
//                                      // after even though the CPU
//                                      // hasn't waited yet
//   mps_matmul_f32(r, w2, y, ...);
//   mps_batch_end();                   // commits once, waits once, THEN
//                                       // every op's 'out' pointer above
//                                       // becomes valid
//
// Every batched op's 'out' buffer is invalid to read until
// mps_batch_end() returns — the same contract as reading a
// cudaMemcpyAsync destination before synchronizing.
//
// IMPORTANT: that example is only safe because none of mps_matmul_f32's
// or mps_relu_f32's *inputs* above are themselves another batched op's
// still-pending output — 'x'/'w1'/'w2' are all already-populated,
// ordinary CPU float arrays. mps_matmul_f32/mps_relu_f32 always upload
// their input from a CPU float* into a fresh GPU buffer, so feeding one
// batched op's 'out' pointer into a later op's 'a'/'b' *within the same
// unflushed batch* silently reads whatever was in that CPU memory before
// the batch started (found and fixed via direct testing, not something
// caught by inspection: a naive matmul -> relu chain like this computed
// 0 instead of the real value). Use mps_matmul_f32_chained/
// mps_relu_f32_chained (below) instead for that case — they take the
// *GPU buffer* a previous batched op already wrote to directly, skipping
// the CPU round trip entirely rather than reading stale CPU data.
void mps_batch_begin();
void mps_batch_end();
int  mps_batch_is_active();

typedef fn void(void* ctx) MpsFinalizeFn;

// Register a callback to run once, in registration order, after
// mps_batch_end()'s single wait and raw GPU->CPU readback of every
// batched op's output buffer. Exists for callers (tensor_gpu.sc) that
// need to do their own CPU-side post-processing on a batched op's result
// (e.g. float32->float64 conversion) — that processing has to wait for
// the same sync point the raw readback does, so it can't just run
// immediately after the mps_*_f32 call returns the way it does outside
// a batch.
void mps_batch_register_finalize(MpsFinalizeFn finalizeFn, void* ctx);

// The GPU buffer handle mps_matmul_f32/mps_relu_f32 most recently wrote
// their (still-pending) output to, while a batch is active — NULL if no
// batch is active or nothing's been dispatched into this batch yet. Call
// right after a batched op returns to capture its output buffer for
// chaining into the *_chained entry points below, instead of waiting for
// mps_batch_end() and reading CPU floats.
void* mps_batch_last_output_buffer();

// Same computation as mps_matmul_f32/mps_relu_f32, but the primary input
// ('bufA') is a GPU buffer handle — typically mps_batch_last_output_
// buffer()'s return from a previous batched op in the same still-open
// batch — instead of a CPU float array, so no CPU<->GPU round trip
// happens for that input at all. Only usable while a batch is active
// (mps_batch_is_active()); the secondary input 'b' (matmul's second
// operand) is still an ordinary CPU float array — realistic chains in
// this codebase (matmul -> relu -> matmul) only ever have ONE chained
// input at a time (the running activation; weight matrices are always
// freshly uploaded), so that's the only shape these support.
int mps_matmul_f32_chained(void* bufA, const float* b, float* out,
                            unsigned long M, unsigned long K, unsigned long N);
int mps_relu_f32_chained(void* bufA, float* out, unsigned long n);

// out[i] = bufA[i] - b[i] -- same chaining shape as mps_matmul_f32_chained/
// mps_relu_f32_chained above, used by tensor_sub_gpu for the loss
// computation's diff = pred - target (pred, e.g. a matmul's output, is
// still GPU-pending; target is an ordinary already-populated Tensor).
int mps_sub_f32_chained(void* bufA, const float* b, float* out, unsigned long n);

// out[i] = bufA[i] * b[i] -- generic chained elementwise multiply (one
// pending GPU input, one plain CPU array), used by tensor_square_gpu's
// backward: 2*diff (still GPU-pending from the forward pass) times the
// incoming gradient (an ordinary CPU array -- see __sum_backward, which
// writes it eagerly with no GPU involvement at all).
int mps_mul_f32_chained(void* bufA, const float* b, float* out, unsigned long n);

// out[i] = buf[i] * buf[i] -- self-multiply of a single still-pending GPU
// buffer (both mul_kernel operand slots bound to the same buffer; reading
// the same device memory twice is safe, there's no write). Used by
// tensor_square_gpu's forward pass for the loss computation's
// sq = diff * diff, where 'diff' is itself still GPU-pending -- the
// generic mps_mul_f32_chained above can't express this shape since its
// second operand has to already be a plain CPU array.
int mps_square_f32_chained(void* buf, float* out, unsigned long n);

// out[0] = sum(bufA[0..n)) -- chained version of mps_sum_f32 above, input
// already a still-pending GPU buffer. Same deferred-partial-sum-combine
// finalize as the non-chained batched path.
int mps_sum_f32_chained(void* bufA, float* out, unsigned long n);

// out[i] = buf[i] * k -- chained version of mps_scale_f32, input already a
// still-pending GPU buffer. Used by tensor_sub_gpu's backward for
// dTarget = -1 * (incoming gradient), when that incoming gradient is
// itself still GPU-pending.
int mps_scale_f32_chained(void* buf, float k, float* out, unsigned long n);

// ── Backward chained variants ─────────────────────────────────────────────────
// Same idea as mps_matmul_f32_chained/mps_relu_f32_chained above, for the
// backward chain (matmul2-backward's dA=dH output feeds relu-backward's
// selfGrad input; relu-backward's output feeds matmul1-backward's dC
// input) -- each takes the *upstream op's pending GPU buffer* for
// whichever argument is the flowing gradient, so no CPU round trip
// happens for it. The other argument (always an ordinary already-
// populated Tensor -- a forward activation or a weight matrix) stays a
// plain CPU float array, same restriction as the forward chained pair.
//   mps_matmul_abt_f32_chained: 'bufDC' replaces 'a' (the incoming
//     gradient) -- for a deeper graph where dC is itself still-pending.
//   mps_matmul_atb_f32_chained: 'bufDC' replaces 'b' (matmul_atb_f32's
//     second argument is the incoming gradient) -- this is the one this
//     library's own 2-layer MLP training loop actually needs, for
//     dW1 = X^T . dH_pre where dH_pre is relu-backward's pending output.
//   mps_relu_backward_f32_chained: 'bufSelfGrad' replaces 'selfGrad' --
//     this library's own MLP needs this one too, for dH_pre = relu'(H) *
//     dH where dH is matmul2-backward's pending dA output.
int mps_matmul_abt_f32_chained(void* bufDC, const float* b, float* out,
                                unsigned long M, unsigned long N, unsigned long K);
int mps_matmul_atb_f32_chained(const float* a, void* bufDC, float* out,
                                unsigned long M, unsigned long K, unsigned long N);
int mps_relu_backward_f32_chained(const float* a, void* bufSelfGrad, float* out, unsigned long n);

// Same as mps_relu_backward_f32_chained, but 'a' is ALSO a still-pending
// GPU buffer -- needed when relu's own input is itself an activation
// computed earlier in the SAME still-open batch (the fused
// forward+loss+backward training step). See gpu_mps.sc's comment.
int mps_relu_backward_f32_chained_ab(void* bufA, void* bufSelfGrad, float* out, unsigned long n);

// Same as mps_matmul_atb_f32_chained, but 'a' is ALSO a still-pending GPU
// buffer (not a plain CPU array) -- needed when the matmul's first operand
// is itself an activation computed earlier in the SAME still-open batch
// (e.g. dW2 = H^T . dY in the fused forward+loss+backward training step,
// where H hasn't been read back to CPU yet either). See gpu_mps.sc's
// comment on the implementation.
int mps_matmul_atb_f32_chained_ab(void* bufA, void* bufDC, float* out,
                                   unsigned long M, unsigned long K, unsigned long N);

} // namespace std
