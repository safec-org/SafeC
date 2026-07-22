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
// operand) is still an ordinary CPU float array here — see
// mps_matmul_f32_persistent below for the case where BOTH operands are
// already device buffers (e.g. a weight matrix uploaded once via
// mps_upload_persistent instead of fresh every call).
int mps_matmul_f32_chained(void* bufA, const float* b, float* out,
                            unsigned long M, unsigned long K, unsigned long N);
int mps_relu_f32_chained(void* bufA, float* out, unsigned long n);

// out[M,N] = bufA[M,K] . bufB[K,N] -- like mps_matmul_f32_chained, but
// 'bufB' is ALSO a pre-existing device buffer instead of a plain CPU
// array to upload fresh. For a matmul run repeatedly against the same
// weight matrix (e.g. an inference loop doing the same forward pass
// hundreds of times), mps_matmul_f32_chained would re-upload (memcpy)
// that weight matrix from CPU on every single call even though it never
// changes — this skips that entirely for both operands.
int mps_matmul_f32_persistent(void* bufA, void* bufB, float* out,
                               unsigned long M, unsigned long K, unsigned long N);

// Uploads 'data' to a device buffer ONCE; the caller keeps the returned
// handle and reuses it directly (as an argument to mps_matmul_f32_
// persistent/_chained above) across as many dispatches and batches as it
// wants, instead of re-uploading fresh every call. Safe to call whether
// or not a batch is open, but only once no GPU work that might still be
// reading the reused pool slot's previous contents is in flight (i.e.
// after any earlier batch's mps_batch_end() has already returned). The
// caller owns the returned buffer's lifetime and must release it with
// mps_release_persistent below exactly once, when truly done with it.
void* mps_upload_persistent(const float* data, unsigned long bytes);
void  mps_release_persistent(void* buf, unsigned long bytes);

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

// Fused gated-activation (GeGLU/tanh-GELU) step: 'bufRaw' is a still-
// pending GPU buffer holding one row-major [rows, 2*intermediateSize]
// matmul output (gate half then up half per row -- e.g. the result of a
// matmul against a row-concatenated [gateW; upW] weight); computes
// out[row,col] = gelu(bufRaw[row,col]) * bufRaw[row, intermediateSize+col]
// for col in [0, intermediateSize), same tanh-approx GELU formula as
// std::tensor_gelu_fwd (activations.sc), in one dispatch -- see
// gpu_mps_kernels.metal's gated_gelu_kernel. 'out' has rows*
// intermediateSize elements, same throwaway-CPU-scratch-until-batch-end
// contract as every other *_chained function here.
int mps_gated_gelu_f32_chained(void* bufRaw, float* out, unsigned long rows, unsigned long intermediateSize);

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

// Same as mps_matmul_abt_f32_chained, but 'b' is ALSO a pre-existing
// device buffer (typically mps_upload_persistent, e.g. a weight matrix
// that stays GPU-resident across a whole training run) instead of a
// plain CPU array. Needed for dH = dY . W2^T when W2 is never read back
// to a CPU array between steps (see mps_sgd_update_f32_chained below).
int mps_matmul_abt_f32_persistent(void* bufDC, void* bufB, float* out,
                                   unsigned long M, unsigned long N, unsigned long K);

// In-place SGD step against a persistent device buffer: w[i] -= lr*
// grad[i]. Both 'bufW' and 'bufGrad' are already device buffers (bufW
// typically from mps_upload_persistent, bufGrad typically another op's
// still-pending output from the same batch's backward pass) -- no 'out'
// parameter, since the whole point is that the updated weight never
// leaves the GPU. See gpu_mps.sc's comment and gpu_mps_kernels.metal's
// sgd_update_kernel.
int mps_sgd_update_f32_chained(void* bufW, void* bufGrad, float lr, unsigned long n);

// ── fp16 / bf16 tier ─────────────────────────────────────────────────────
// Real half/bfloat compute on the GPU, not just storage: gpu_mps_kernels.metal's
// matmul_kernel_f16/matmul_kernel_bf16/relu_kernel_f16/relu_kernel_bf16
// hold/multiply in the reduced-precision type and accumulate in float
// (standard mixed-precision-GEMM practice) — verified to compile through
// this machine's real Metal toolchain (both 'half' and 'bfloat' are
// natively supported MSL types on this SDK). The functions below take
// and return ordinary 'float*' host arrays, same shape as mps_matmul_f32/
// mps_relu_f32, so a caller doesn't need to touch fp16/bf16 bit patterns
// directly: conversion (see float16.h's f32_to_fp16_bulk/f32_to_bf16_bulk
// and their inverses) happens on the CPU side of the upload/readback,
// invisibly, inside these functions.
//
// This is the naive (TILE_SIZE=16, threadgroup-memory-tiled) kernel tier
// only — the smma/multi-simdgroup tiers (see mps_matmul_f32's comment)
// use simdgroup_float8x8, a float-only type on this Metal SDK; porting
// those to half/bfloat accumulation is future work, not attempted here.
// Every call here does its own upload/dispatch/readback (no persistent-
// buffer or batching integration) — the simplest, most directly testable
// shape, matching mps_matmul_f32's non-batched fallback path.
//
// Expected precision loss is real and by design: fp16's ~3-4 significant
// decimal digits (10 mantissa bits) and bf16's ~2-3 (7 mantissa bits) are
// both coarser than float32's ~7 — see float16.h's precision notes.
// Returns 1 on success, 0 on failure (no Metal device, pipeline
// compilation failure, or allocation failure) — 'out' is left untouched
// on failure.
int mps_matmul_f16(const float* a, const float* b, float* out,
                    unsigned long M, unsigned long K, unsigned long N);
int mps_matmul_bf16(const float* a, const float* b, float* out,
                     unsigned long M, unsigned long K, unsigned long N);
int mps_relu_f16(const float* a, float* out, unsigned long n);
int mps_relu_bf16(const float* a, float* out, unsigned long n);

// ── fp16 / bf16 persistent-buffer / batched tier ────────────────────────
// Mirrors mps_upload_persistent/mps_matmul_f32_persistent/mps_relu_f32_
// chained/mps_sub_f32_chained/mps_scale_f32_chained/mps_matmul_atb_f32_
// chained_ab/mps_matmul_abt_f32_persistent/mps_relu_backward_f32_chained_
// ab/mps_sgd_update_f32_chained's whole family, in fp16/bf16 — enough ops
// to run a full GPU-resident forward+backward+SGD step (see
// gpu_mps_kernels.metal's matching kernel-tier comment) with weights that
// never leave the GPU as float32 at all, the same shape as the float32
// GPU-resident training rewrite this session's MPS work already
// delivered, just in half the bytes per weight/activation.
//
// Key difference from mps_matmul_f16/mps_relu_f16 above: those take/
// return 'float*' and do a full convert-upload-dispatch-convert round
// trip on every call, meant for one-shot use. Everything below instead
// passes 'unsigned short*' scratch buffers (raw fp16/bf16 bits) for any
// CPU-side throwaway output, exactly mirroring how mps_matmul_f32_
// persistent's chained scratch arrays (scratchZ1/scratchH/etc. in
// train_gpu.sc) are never meaningfully read except the one value that
// matters -- so intermediate activations/gradients here never get
// converted back to float32 at all, only read once as raw bits into a
// scratch array mps_batch_end() overwrites into, matching the float32
// version's "throwaway CPU scratch, GPU buffer is what actually chains"
// design. A caller converts host-side f32 data to fp16/bf16 bits with
// float16.h's f32_to_fp16_bulk/f32_to_bf16_bulk before calling
// mps_upload_f16_persistent/mps_upload_bf16_persistent, and converts
// back with fp16_to_f32_bulk/bf16_to_f32_bulk only for values it
// actually needs on the CPU (e.g. a loss scalar for monitoring).
//
// Same naive (TILE_SIZE=16 tiled) kernel tier as mps_matmul_f16/
// mps_matmul_bf16 above for every matmul-shaped op here -- no smma/
// multi-simdgroup fast path (see that comment).

void* mps_upload_f16_persistent(const float* data, unsigned long n);
void* mps_upload_bf16_persistent(const float* data, unsigned long n);
// Both release through mps_release_persistent(buf, n * 2UL) above --
// storage is 2 bytes/element regardless of fp16 vs bf16, and
// mps_release_persistent only ever needed the byte count, not the dtype.

// out = bufA[M,K] . bufB[K,N] -- both operands already device buffers
// (e.g. from mps_upload_f16_persistent). Must be called within an open
// batch (mps_batch_begin()); 'out' is throwaway scratch (raw fp16/bf16
// bits, not valid until mps_batch_end()) -- read the real result via
// mps_batch_last_output_buffer() and chain it into the next op, same
// pattern as mps_matmul_f32_persistent.
int mps_matmul_f16_persistent(void* bufA, void* bufB, unsigned short* out,
                               unsigned long M, unsigned long K, unsigned long N);
int mps_matmul_bf16_persistent(void* bufA, void* bufB, unsigned short* out,
                                unsigned long M, unsigned long K, unsigned long N);

int mps_relu_f16_chained(void* bufA, unsigned short* out, unsigned long n);
int mps_relu_bf16_chained(void* bufA, unsigned short* out, unsigned long n);

// out[i] = bufA[i] - bHalf[i]/bBf[i] -- 'bHalf'/'bBf' is a plain CPU
// array of raw fp16/bf16 bits (e.g. a training target, pre-converted
// once with f32_to_fp16_bulk/f32_to_bf16_bulk before the loop), uploaded
// fresh each call, same shape as mps_sub_f32_chained.
int mps_sub_f16_chained(void* bufA, const unsigned short* bHalf, unsigned short* out, unsigned long n);
int mps_sub_bf16_chained(void* bufA, const unsigned short* bBf, unsigned short* out, unsigned long n);

// out[i] = buf[i] * k -- 'k' is a plain float scalar (see gpu_mps_
// kernels.metal's scale_kernel_f16/bf16 comment on why the scalar itself
// stays float-precision).
int mps_scale_f16_chained(void* buf, float k, unsigned short* out, unsigned long n);
int mps_scale_bf16_chained(void* buf, float k, unsigned short* out, unsigned long n);

// dB[K,N] = A^T[K,M] . dC[M,N], A stored [M,K] -- both operands already
// device buffers, same shape as mps_matmul_atb_f32_chained_ab.
int mps_matmul_atb_f16_chained_ab(void* bufA, void* bufDC, unsigned short* out,
                                   unsigned long M, unsigned long K, unsigned long N);
int mps_matmul_atb_bf16_chained_ab(void* bufA, void* bufDC, unsigned short* out,
                                    unsigned long M, unsigned long K, unsigned long N);

// dA[M,K] = dC[M,N] . B^T[N,K], B stored [K,N] -- same shape as
// mps_matmul_abt_f32_persistent.
int mps_matmul_abt_f16_persistent(void* bufDC, void* bufB, unsigned short* out,
                                   unsigned long M, unsigned long N, unsigned long K);
int mps_matmul_abt_bf16_persistent(void* bufDC, void* bufB, unsigned short* out,
                                    unsigned long M, unsigned long N, unsigned long K);

int mps_relu_backward_f16_chained_ab(void* bufA, void* bufSelfGrad, unsigned short* out, unsigned long n);
int mps_relu_backward_bf16_chained_ab(void* bufA, void* bufSelfGrad, unsigned short* out, unsigned long n);

// In-place SGD step against a persistent fp16/bf16 device buffer --
// same shape as mps_sgd_update_f32_chained, weights never leave the GPU.
int mps_sgd_update_f16_chained(void* bufW, void* bufGrad, float lr, unsigned long n);
int mps_sgd_update_bf16_chained(void* bufW, void* bufGrad, float lr, unsigned long n);

} // namespace std
