#pragma once
#include <std/stddef.h>
// SafeC Standard Library — Tensor: CPU tensors with reverse-mode Autograd::get.
//
// PyTorch-style ergonomics (build a graph implicitly by calling ops;
// Tensor::backward() walks it) over a plain flat f32 buffer — no unified-
// memory story here (that's MPS's job; see gpu_mps.h), just row-major
// single-precision arrays with a recorded computation graph. Scope: 1D
// and 2D tensors (vectors and matrices — everything a small MLP/attention
// block needs), add/sub/mul/relu/sum/matmul, and reverse-mode autodiff
// over that op set. Not a general n-dimensional array library — no
// broadcasting beyond 2D, no int/complex dtypes, no in-place ops.
//
// float32, not float64: PyTorch defaults to float32, and this codebase's
// training-loop shapes measured ~30% faster on Accelerate's cblas_sgemm
// than cblas_dgemm for identical work, on top of halving memory
// bandwidth for every elementwise pass -- see tensor_blas.h. Also the
// same precision Metal itself supports (Metal has no double at all), so
// this removes the double<->float conversion every GPU-backed op used to
// pay at its CPU/GPU boundary (see gpu_mps.h's history) instead of just
// moving where the precision loss happens.
//
// Internal helpers (__tensor_alloc, __tensor_link1/2, every __X_backward
// function, etc.) are defined right here in the header as 'static'
// functions, not in tensor.sc — this is what lets tensor.sc, tensor_nn.sc,
// and the tensor_<gpu-backend>.sc family compile as genuinely separate
// translation units (each #includes this header, gets its own private
// copy of these helpers, and is linked against the others' *public*
// symbols — tensor_add, tensor_matmul, etc. — at build time) instead of
// every file needing to textually #include tensor.sc's entire body just
// to reach a handful of internal-linkage helpers it happens to reuse.
//
// Every op here is its own small function with one loop -- not just
// style. A hand-rolled training loop with everything (BLAS calls,
// elementwise passes) inlined into one large function was measured to
// make this compiler's -O2 optimizer hang (40+ minutes, not just slow)
// on exactly this kind of training-loop shape; splitting into small
// single-purpose functions avoided it entirely and let -O2's
// auto-vectorization actually run. Keeping every op as its own function
// isn't just for consistency -- it's load-bearing.
#include <std/collections/vec.h>
#include <std/simd/gemm.h>
#include <std/sync/parallel.h>
#include <std/simd/cpu.h>
#include <std/ml/element.h>
#include <std/ml/device.h>
#include <std/ml/device_kernels.h>

namespace std {

// The static helpers below call libc malloc/free directly (not
// std::alloc/dealloc, so no dependency on mem.sc's allocator machinery
// for this hot path) -- declared here rather than pulled in via
// std/mem.sc, matching every other .sc file in std/ that touches libc
// memory functions directly (see mem.sc, heap.sc, result.sc, etc., each
// of which declares its own copy of these same externs rather than
// including one another's .sc file for them).
extern void* malloc(size_t size);
extern void  free(void* ptr);

typedef fn void(void* selfTensor) TensorBackwardFn;

// ── Tensor arenas ─────────────────────────────────────────────────────────────
// Every op allocates its output. In a training loop that is a malloc (and,
// for fresh memory, page faults) per op per step: on a small MLP that cost
// more than the arithmetic. A TensorArena is a chunked bump allocator for a
// step's temporaries: between a.begin() and tensor_arena_end()
// every tensor created on this thread (data, shape, gradient) comes from
// it, and a.reset() drops them all at once while keeping the
// memory -- already faulted in -- for the next step.
//
//   struct TensorArena a = TensorArena::new(0UL);
//   while (step < steps) {
//       a.begin();
//       ... forward, Tensor::backward, optimizer update ...
//       tensor_arena_end();
//       a.reset();     // this step's tensors are gone
//   }
//   a.free();
//
// Tensors created outside the arena (parameters) stay on the heap, and so do
// their gradients, so they survive resets. Tensor::free() on an arena tensor
// does nothing. A tensor must not be used after its arena is reset.
struct TensorArena {
    unsigned long first;       // chunk list (internal)
    unsigned long current;
    unsigned long chunkBytes;
    unsigned long tracked;     // tensors to release on reset (internal)
    unsigned long trackedCount;
    unsigned long trackedCap;

    void begin();   // this thread's new tensors go to this arena
    void reset();   // drop every tensor in it, keep the memory
    void free();
    // Bytes handed out since the last reset.
    unsigned long used() const;

    // chunkBytes 0 = 16 MiB chunks (larger requests get their own chunk).
    static struct TensorArena new(unsigned long chunkBytes);
};

void tensor_arena_end();                   // back to the heap

// Internal (tensor.h's allocation helpers): the arena new tensors use on this
// thread, or null; 64-byte-aligned memory from it; and registering a tensor
// so a reset releases what it holds outside the arena.
?&TensorArena tensor_arena_current_();
void* tensor_arena_alloc_raw_(&TensorArena a, unsigned long bytes);
void tensor_arena_track_(&TensorArena a, void* t);

// ── Tensors ──────────────────────────────────────────────────────────────────
// A DenseTensor<E> stores its values as E (element.h) and computes in
// float32; its gradient is always float32. FloatTensor (E = float) is the
// float32 tensor every op has always used; HalfTensor (F16) and BF16Tensor
// (BF16) keep half the memory per value and run every op and its backward
// pass through float32. Generic code is written against trait Tensor and
// works with all three.
//
// The graph fields below have the same layout for every E (data is one
// pointer whatever E is), so the backward pass walks a graph that mixes
// element types: to_float() / from_float() are graph nodes between them.
generic<E>
struct DenseTensor {
    &heap E       data;        // flat row-major buffer, length == size
    &heap unsigned long shape; // shape[0..ndim)
    unsigned long ndim;
    unsigned long size;        // product of shape
    int           requiresGrad;
    &heap float   grad;        // float32, length == size; only allocated if requiresGrad
    // Autograd graph edges. 'gradFn' is the node's backward function,
    // stored untyped (void*) so a graph can mix element types; each
    // function knows its own node's type.
    void*         gradFn;
    struct Vec<&DenseTensor<E>> parents; // inputs this tensor was computed from
    int           visited;     // topological-sort scratch flag, used and cleared by backward()
    float         extraScalar; // op-specific constant (e.g. scale's k) the gradFn needs
    ?&TensorArena owner;       // the TensorArena that owns this tensor's memory, or null (heap)
    // Where the values live (device.h). 'data' always has host storage; a
    // tensor on a GPU also has 'dbuf' there, and 'coh' says which copies of
    // the values and of the gradient are current (TS_* below), so each side
    // is copied only when the other one changed. 16-bit tensors stay on the
    // CPU (place() refuses), their ops computing through float32 tensors.
    int           dev;         // DEVICE_*; DEVICE_CPU: no device copy
    unsigned long dbuf;        // the values on 'dev' (backend handle), 0 if none
    unsigned long dgrad;       // the gradient on 'dev', 0 if none
    int           coh;         // TS_* bits

    // Element access as float32 (row-major; 'i' for a rank-1 tensor,
    // 'r, c' for a rank-2 one).
    float   at1(unsigned long i) const;
    float   at2(unsigned long r, unsigned long c) const;
    void    set1(unsigned long i, float v);
    void    set2(unsigned long r, unsigned long c, float v);

    void    free();

    // Must be a scalar (size == 1, e.g. a loss). Seeds grad = 1, walks the
    // graph in reverse topological order, and accumulates gradients into
    // every requiresGrad ancestor's 'grad' buffer (allocating it on first
    // write). Gradients accumulate across calls, matching PyTorch's default
    // (zero_grad() between optimizer steps if that's not what you want).
    void backward();
    void zero_grad();

    static &DenseTensor<E> new_1d(unsigned long n, int requiresGrad);
    static &DenseTensor<E> new_2d(unsigned long rows, unsigned long cols, int requiresGrad);
    // Copies float32 'values' (length must equal the tensor's size) into a
    // freshly allocated tensor of the given shape, rounded to E.
    static &DenseTensor<E> from_1d(const &float values, unsigned long n, int requiresGrad);
    static &DenseTensor<E> from_2d(const &float values, unsigned long rows, unsigned long cols, int requiresGrad);
    &DenseTensor<E> zeros_like() const;
    void    fill(float v);   // in place
    &DenseTensor<E> scale(float k) const;   // a * scalar k
    &DenseTensor<E> sum() const;   // -> scalar (1-element) tensor

    // The values as a float32 tensor, and a float32 tensor's values rounded
    // to E: graph nodes (gradients pass through unchanged), the way an op on
    // a 16-bit tensor computes in float32.
    &DenseTensor<float> to_float() const;
    static &DenseTensor<E> from_float(const &DenseTensor<float> x);

    // ── Devices (device.h) ──
    int  device() const;
    // Moves this tensor itself -- values and gradient -- to 'device', keeping
    // it the same object (parameters, other leaves). 1 on success; 0 for an
    // unavailable device or a 16-bit tensor (they stay on the CPU).
    int  place(int device);
    // Makes the host copy of the values current (waits for the device). 1 on success.
    int  sync();
    // A copy on 'device', or this tensor if it is there already: a graph
    // node, the gradient flows back to this tensor's device.
    &DenseTensor<E> to(int device) const;
    // The values' device buffer handle (0 when the tensor is on the CPU).
    unsigned long device_buffer() const;
};

typedef struct DenseTensor<float> FloatTensor;
typedef struct DenseTensor<struct F16> HalfTensor;
typedef struct DenseTensor<struct BF16> BF16Tensor;

// What generic ML code asks of a tensor: FloatTensor, HalfTensor and
// BF16Tensor all satisfy it ('generic<T: Tensor>').
trait Tensor {
    float at1(unsigned long i) const;
    float at2(unsigned long r, unsigned long c) const;
    void  set1(unsigned long i, float v);
    void  set2(unsigned long r, unsigned long c, float v);
    void  fill(float v);
    void  backward();
    void  zero_grad();
    void  free();
}

// What a tensor that can live on a device adds (device.h): DenseTensor and
// SparseTensor (sparse.h) conform.
//   device()   where the values live (DEVICE_CPU, DEVICE_ROCM, DEVICE_CUDA, DEVICE_MPS)
//   place(d)   moves the tensor itself to d (parameters, leaves); 1 on success
//   sync()     makes the host copy current; 1 on success
trait DeviceTensor : Tensor {
    int  device() const;
    int  place(int device);
    int  sync();
}

// One trait per GPU backend, each inheriting DeviceTensor: code written for
// one backend's buffers (a hand-written kernel, a vendor library call) asks
// for its tensor. device_buffer() is the values' buffer on that backend --
// a hipDeviceptr, a CUdeviceptr, an MTLBuffer's GPU address, a Vulkan buffer
// device address -- and 0 while the tensor is elsewhere: check device()
// first, or place() it.
trait RocmTensor : DeviceTensor {
    unsigned long device_buffer() const;
}
trait CudaTensor : DeviceTensor {
    unsigned long device_buffer() const;
}
trait MpsTensor : DeviceTensor {
    unsigned long device_buffer() const;
}
trait VulkanTensor : DeviceTensor {
    unsigned long device_buffer() const;
}

// Coherence bits ('coh'): which copies are current. Both the values' bits
// may be set (the copies agree); a gradient that doesn't exist yet counts as
// current zeros on both sides.
#define TS_HOST   1    // the values on the host
#define TS_DEV    2    // the values on the device
#define TS_GHOST  4    // the gradient on the host
#define TS_GDEV   8    // the gradient on the device
#define TS_DEVOP  16   // made by a device op: its backward runs on the device
#define TS_GZERO  32   // the device gradient is all zeros, not yet written (its buffer may hold anything)

// Internal (tensor_device.sc): keeping the copies coherent, for ops on any
// element type (the graph fields have one layout).
void tensor_host_(void* t);              // values current on the host
void tensor_host_write_(void* t);        // ... and about to be written there
void tensor_grad_host_(void* t);         // gradient current on the host
void tensor_grad_host_write_(void* t);   // ... and about to be written there
int  tensor_dev_(void* t, int device);   // values current on 'device' (uploads); 1 on success
int  tensor_grad_dev_(void* t, int device);   // gradient current on 'device'
// As tensor_grad_dev_ for an op about to add to the device gradient: 2 when
// the gradient is still all zeros (the op must write every value, '=' instead
// of '+='), 1 to add, 0 on failure.
int  tensor_grad_dev_acc_(void* t, int device);
// fp16: unscales cnt gradients {dgrad, n, start} in one launch (the table
// cached on the device while it repeats); 0 when the backend can't.
// A device tensor whose values are stored as 'tag' (device.h's DEVICE_FP32 /
// FP16 / BF16).
?&DenseTensor<float> tensor_dev_new_t_(const &unsigned long shape, unsigned long ndim, int device, int tag);
// Where atomics add into t's device gradient (current on 'device'): the
// gradient itself, or for a 16-bit one a zeroed fp32 scratch that
// tensor_grad_atomic_end_ adds into it and frees.
unsigned long tensor_grad_atomic_begin_(void* t, int device);
void tensor_grad_atomic_end_(void* t, int device, unsigned long target);
int  tensor_amp_unscale_table_(int device, unsigned long amp, const &unsigned long tab, unsigned long cnt,
                               unsigned long total);
void tensor_dev_wrote_(void* t);         // the device copy of the values changed
void tensor_grad_dev_wrote_(void* t);    // the device copy of the gradient changed
void tensor_release_device_(void* t);    // brings values and gradient home, returns the device buffers
void tensor_free_device_(void* t);       // returns the device buffers, dropping their contents
int  tensor_place_(void* t, int device);
// The device an op on these operands runs on: a GPU if any operand is on one
// (and none on another), else DEVICE_CPU.
int  tensor_op_device_(void* a, void* b, void* c);
// A fresh float32 tensor whose values will be written on 'device'.
?&DenseTensor<float> tensor_dev_new_(const &unsigned long shape, unsigned long ndim, int device);

// Internal: the ops on a device (tensor_device.sc). Each returns null when the
// operands are on the CPU or the device can't run it (the caller computes on
// the CPU). Unary: forward EW op 'fwd' with k, k2; backward EW op 'bop' with
// y from the input (ysrc 1) or output (2) and constant bk, z from the output
// (zsrc 2). Binary kind: 0 add, 1 sub, 2 mul, 3 minimum. Loss kind: 0
// cross-entropy on class indices, 1 BCE on logits, 2 Huber (k = delta).
?&DenseTensor<float> tdev_unary_(const &DenseTensor<float> a, int fwd, float k, float k2, int bop, int ysrc, float bk, int zsrc);
?&DenseTensor<float> tdev_binary_(const &DenseTensor<float> a, const &DenseTensor<float> b, int kind);
?&DenseTensor<float> tdev_sum_(const &DenseTensor<float> a, float scale);
?&DenseTensor<float> tdev_matmul_(const &DenseTensor<float> a, const &DenseTensor<float> b);
?&DenseTensor<float> tdev_add_bias_(const &DenseTensor<float> x, const &DenseTensor<float> b);
?&DenseTensor<float> tdev_mul_row_(const &DenseTensor<float> x, const &DenseTensor<float> g);
?&DenseTensor<float> tdev_sum_rows_(const &DenseTensor<float> a);
?&DenseTensor<float> tdev_concat_cols_(const &DenseTensor<float> a, const &DenseTensor<float> b);
?&DenseTensor<float> tdev_slice_cols_(const &DenseTensor<float> a, unsigned long start, unsigned long len);
?&DenseTensor<float> tdev_gather_rows_(const &DenseTensor<float> table, const &DenseTensor<float> idx);
?&DenseTensor<float> tdev_softmax_(const &DenseTensor<float> a, int logMode);
// Row softmax with attention.h's mask (causal, sliding window: ATTN_* flags).
?&DenseTensor<float> tdev_softmax_mask_(const &DenseTensor<float> a, int causal, unsigned long window);
?&DenseTensor<float> tdev_transpose_(const &DenseTensor<float> a);
// Fused multi-head attention (attention.h's mha layout, every head in one
// launch) with its backward: on the operands' device, or on the CPU through
// device_ref.sc's versions of the same kernels. Null when a head is wider
// than 128 or the device can't run it.
?&DenseTensor<float> tensor_attention_fused_(const &DenseTensor<float> q, const &DenseTensor<float> k,
                                             const &DenseTensor<float> v, unsigned long heads, int causal,
                                             unsigned long window, float scale);
?&DenseTensor<float> tdev_loss_(const &DenseTensor<float> x, const &DenseTensor<float> t, int kind, float k);
?&DenseTensor<float> tdev_layer_norm_(const &DenseTensor<float> x, float eps);
?&DenseTensor<float> tdev_detach_(const &DenseTensor<float> a);
// Convolution (transpose 0) or transposed convolution (1), in convolution
// terms: image side C x H x W, column side oC x oH x oW, weight [oC, C*kH*kW].
?&DenseTensor<float> tdev_conv_(const &DenseTensor<float> x, const &DenseTensor<float> w, const &DenseTensor<float> b,
                                int transpose, unsigned long C, unsigned long H, unsigned long W,
                                unsigned long oC, unsigned long oH, unsigned long oW,
                                unsigned long kH, unsigned long kW, unsigned long sH, unsigned long sW,
                                unsigned long pH, unsigned long pW, unsigned long dH, unsigned long dW);
// Normalization over [batch, C*plane]: batch norm (running statistics
// updated in training) and group norm (G groups), followed by act (nn.h's
// ACT_NONE, ACT_RELU or ACT_LEAKY) in the same kernels.
?&DenseTensor<float> tdev_batch_norm_(const &DenseTensor<float> x, const &DenseTensor<float> gamma,
                                      const &DenseTensor<float> beta, &DenseTensor<float> runMean,
                                      &DenseTensor<float> runVar, unsigned long C, unsigned long P,
                                      int training, float momentum, float eps, int act);
?&DenseTensor<float> tdev_group_norm_(const &DenseTensor<float> x, const &DenseTensor<float> gamma,
                                      const &DenseTensor<float> beta, unsigned long C, unsigned long P,
                                      unsigned long G, float eps, int act);
// Spatial ops over [batch, C*H*W] planes (convnet.h's layout).
#define TDEV_POOL     0   // window pooling: mode 0 max, 1 avg (padded divisor), 2 avg (inside)
#define TDEV_GPOOL    1   // global pooling: mode 0 max, 1 avg
#define TDEV_UPSAMPLE 2   // nearest by the integer factor kH
#define TDEV_RESIZE   3   // to oH x oW, mode RESIZE_*, align
struct TdevSpatial_ {
    int kind;
    unsigned long C; unsigned long H; unsigned long W; unsigned long oH; unsigned long oW;
    unsigned long kH; unsigned long kW; unsigned long sH; unsigned long sW; unsigned long pH; unsigned long pW;
    int mode;
    int align;
};
?&DenseTensor<float> tdev_spatial_(const &DenseTensor<float> x, const &TdevSpatial_ g);
// The EW codes past tensor.h's own, shared with device_kernels.cu.
#define EW_EXP        20
#define EW_LOG        21
#define EW_ABS        22
#define EW_LEAKY      23
#define EW_CLAMP      24
#define EW_SIGMOID    25
#define EW_TANH       26
#define EW_GELU       27
#define EW_SILU       28
#define EW_ADDK       29
#define EW_MIN        30
#define EW_ELU        31
#define EW_LERP       32   // out += k * (a - out), in place
#define EW_MULS       33   // out = a * b[0], b a device scalar
#define EW_ACC_PK     40
#define EW_ACC_EXPB   41
#define EW_ACC_LOGB   42
#define EW_ACC_ABSB   43
#define EW_ACC_LEAKYB 44
#define EW_ACC_CLAMPB 45
#define EW_ACC_GELUB  46
#define EW_ACC_SILUB  47
#define EW_ACC_MINA   48
#define EW_ACC_MINB   49
#define EW_ACC_ELUB   50
#define EW_ACC_PSCALE 51

// Every function below takes/returns 'Tensor' by a region-less reference
// ('&Tensor'/'const &Tensor' — see README's "Outliving references"
// section): a Tensor is never null anywhere in this library, and nothing
// here cares whether the caller's tensor is heap/stack/static/arena-
// backed, so pinning one specific region would only narrow who can call
// these for no safety benefit. Internally, tensors are still allocated
// with a plain 'malloc' (see '__tensor_alloc' below) and freed with
// 'free()' via 'Tensor::free()' below — 'free()' takes no argument beyond
// 'self' precisely because there's no separate heap-region bookkeeping to
// release; call it, then let the reference go out of scope.

// ── Construction ─────────────────────────────────────────────────────────────

// ── Forward ops (each records a backward edge when either operand
// requires grad) ────────────────────────────────────────────────────────────
// Generic over the element type (defined at the end of this header); a
// HalfTensor / BF16Tensor op computes in float32 and rounds its result.
generic<E> &DenseTensor<E> tensor_add(const &DenseTensor<E> a, const &DenseTensor<E> b);
generic<E> &DenseTensor<E> tensor_sub(const &DenseTensor<E> a, const &DenseTensor<E> b);
generic<E> &DenseTensor<E> tensor_mul(const &DenseTensor<E> a, const &DenseTensor<E> b);      // elementwise
generic<E> &DenseTensor<E> tensor_matmul(const &DenseTensor<E> a, const &DenseTensor<E> b);   // 2D x 2D

// Internal: the float32 implementations (tensor.sc).
&FloatTensor tensor_add_f32_(const &FloatTensor a, const &FloatTensor b);
&FloatTensor tensor_sub_f32_(const &FloatTensor a, const &FloatTensor b);
&FloatTensor tensor_mul_f32_(const &FloatTensor a, const &FloatTensor b);
&FloatTensor tensor_matmul_f32_(const &FloatTensor a, const &FloatTensor b);
&FloatTensor tensor_scale_f32_(const &FloatTensor a, float k);
&FloatTensor tensor_sum_f32_(const &FloatTensor a);
// Tensor::relu lives in tensor_nn.h alongside the rest of the activation
// functions (sigmoid, tanh, elu, gelu, silu, glu, swiglu) -- include that
// header too wherever relu is needed.

// ── Autograd ─────────────────────────────────────────────────────────────────

// ── Gradient mode (no_grad) ──────────────────────────────────────────────────
// Global, not per-TU: every tensor_<op> across every separately-compiled
// file linked into the same program shares one flag (that's why these
// are ordinary externally-linked functions, defined once in tensor.sc,
// not 'static' — unlike the internal helpers below, this state must be
// visible and consistent everywhere, not private per including file).
//
// Mirrors PyTorch's torch.no_grad(): while disabled, every tensor
// produced by tensor_add/matmul/etc. gets requiresGrad=0 regardless of
// its inputs' — __tensor_link1/__tensor_link2 (below) skip pushing to
// 'parents' and setting 'gradFn' entirely, which is the real cost this
// exists to avoid (Vec::push's first call allocates; skipping it removes
// a real std::alloc from every op call during inference). Doesn't affect
// LEAF tensors' own requiresGrad (Tensor::new_2d's explicit argument is
// unaffected) — only what happens when ops COMPOSE tensors.
//
// The flag lives in the program's one Autograd state, reached through
// Autograd::get():
//   struct GradGuard g = Autograd::get().no_grad();   // inference from here
//   ... forward passes record nothing ...
//   g.restore();                                 // back to the previous mode
struct GradGuard {
    int was;                  // the mode no_grad() found
    // Puts the mode back as no_grad() found it (guards nest).
    void restore();
};

struct Autograd {
    int gradEnabled;          // 1: ops record their backward graph

    int  enabled() const;
    void set_enabled(int enabled);
    // Disables recording and returns a guard that restores the previous mode.
    struct GradGuard no_grad();

    // The program's Autograd::get state (one for all threads).
    static &Autograd get();
};

// ══ Internal helpers (static — each including file gets its own copy) ═══════
// Not part of the public API; used by tensor.sc's own op implementations
// and reused directly (as raw function-pointer values, e.g.
// '(void*)__add_backward') by tensor_nn.sc and the tensor_<backend>.sc
// family so a GPU-computed forward result can link into the exact same
// backward math as the CPU path, without duplicating that math.

// ── Elementwise kernels ───────────────────────────────────────────────────────
// One loop per op over raw arrays, which the compiler vectorizes; arrays of
// TENSOR_EW_PAR or more elements are split across std/sync/parallel.h's
// thread pool. Used by the forward ops in tensor.sc / tensor_nn.sc and by
// the backward functions below.
#define TENSOR_EW_PAR 65536UL

#define EW_ADD        0   // out = a + b
#define EW_SUB        1   // out = a - b
#define EW_MUL        2   // out = a * b
#define EW_SCALE      3   // out = a * k
#define EW_RELU       4   // out = max(a, 0)
#define EW_FILL       5   // out = k
#define EW_COPY       6   // out = a
#define EW_ACC        7   // out += a
#define EW_DEC        8   // out -= a
#define EW_ACC_MUL    9   // out += a * b
#define EW_ACC_SCALE  10  // out += a * k
#define EW_ACC_RELU   11  // out += b > 0 ? a : 0
#define EW_ACC_K      12  // out += k
#define EW_ACC_SIGB   13  // out += a * b * (1 - b)   (sigmoid backward, b = output)
#define EW_ACC_TANHB  14  // out += a * (1 - b * b)   (tanh backward, b = output)

struct __TensorEwJob {
    const float* a;
    const float* b;
    float*       out;
    float        k;
    int          op;
};

static void __tensor_ew_body(const &__TensorEwJob j, unsigned long lo, unsigned long hi) __attribute__((always_inline)) {
    unsafe {
        const float* a = j.a;
        const float* b = j.b;
        float* o = j.out;
        float k = j.k;
        unsigned long i = lo;
        int op = j.op;
        if (op == EW_ADD)            { while (i < hi) { o[i] = a[i] + b[i]; i = i + 1UL; } }
        else if (op == EW_SUB)       { while (i < hi) { o[i] = a[i] - b[i]; i = i + 1UL; } }
        else if (op == EW_MUL)       { while (i < hi) { o[i] = a[i] * b[i]; i = i + 1UL; } }
        else if (op == EW_SCALE)     { while (i < hi) { o[i] = a[i] * k; i = i + 1UL; } }
        else if (op == EW_RELU)      { while (i < hi) { float v = a[i]; o[i] = v > (float)0.0 ? v : (float)0.0; i = i + 1UL; } }
        else if (op == EW_FILL)      { while (i < hi) { o[i] = k; i = i + 1UL; } }
        else if (op == EW_COPY)      { while (i < hi) { o[i] = a[i]; i = i + 1UL; } }
        else if (op == EW_ACC)       { while (i < hi) { o[i] = o[i] + a[i]; i = i + 1UL; } }
        else if (op == EW_DEC)       { while (i < hi) { o[i] = o[i] - a[i]; i = i + 1UL; } }
        else if (op == EW_ACC_MUL)   { while (i < hi) { o[i] = __builtin_fmuladd(a[i], b[i], o[i]); i = i + 1UL; } }
        else if (op == EW_ACC_SCALE) { while (i < hi) { o[i] = __builtin_fmuladd(a[i], k, o[i]); i = i + 1UL; } }
        else if (op == EW_ACC_RELU)  { while (i < hi) { o[i] = o[i] + (b[i] > (float)0.0 ? a[i] : (float)0.0); i = i + 1UL; } }
        else if (op == EW_ACC_K)     { while (i < hi) { o[i] = o[i] + k; i = i + 1UL; } }
        else if (op == EW_ACC_SIGB)  { while (i < hi) { float y = b[i]; o[i] = __builtin_fmuladd(a[i] * y, (float)1.0 - y, o[i]); i = i + 1UL; } }
        else if (op == EW_ACC_TANHB) { while (i < hi) { float y = b[i]; o[i] = __builtin_fmuladd(a[i], __builtin_fmuladd(-y, y, (float)1.0), o[i]); i = i + 1UL; } }
    }
}

// The loops per ISA level (std/simd/cpu.h): wider vectors and FMA.
static void __tensor_ew_base(const &__TensorEwJob j, unsigned long lo, unsigned long hi) { __tensor_ew_body(j, lo, hi); }
#if defined(__x86_64__)
static void __tensor_ew_avx2(const &__TensorEwJob j, unsigned long lo, unsigned long hi)
    __attribute__((target("avx2,fma"))) { __tensor_ew_body(j, lo, hi); }
static void __tensor_ew_avx512(const &__TensorEwJob j, unsigned long lo, unsigned long hi)
    __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx2,fma"))) { __tensor_ew_body(j, lo, hi); }
#endif

static void __tensor_ew_range(&__TensorEwJob j, unsigned long lo, unsigned long hi) {
#if defined(__x86_64__)
    int l = simd_level();
    if (l >= SIMD_AVX512) { __tensor_ew_avx512(j, lo, hi); return; }
    if (l >= SIMD_AVX2) { __tensor_ew_avx2(j, lo, hi); return; }
#endif
    __tensor_ew_base(j, lo, hi);
}

static void __tensor_ew(int op, const float* a, const float* b, float k, float* out, unsigned long n) {
    struct __TensorEwJob j;
    j.a = a;
    j.b = b;
    j.out = out;
    j.k = k;
    j.op = op;
    if (n >= TENSOR_EW_PAR) parallel_for(n, 16384UL, __tensor_ew_range, &j);
    else __tensor_ew_range(&j, 0UL, n);
}

static &FloatTensor __tensor_alloc(const unsigned long* shape, unsigned long ndim, int requiresGrad) {
    unsafe {
        ?&TensorArena ar = tensor_arena_current_();
        unsigned long size = 1UL;
        unsigned long i = 0UL;
        while (i < ndim) { size = size * shape[i]; i = i + 1UL; }
        FloatTensor* t;
        unsigned long* shapeCopy;
        float* buf;
        match (ar) {
            case some(a): {
                t = (FloatTensor*)tensor_arena_alloc_raw_(a, sizeof(FloatTensor));
                shapeCopy = (unsigned long*)tensor_arena_alloc_raw_(a, sizeof(unsigned long) * ndim);
                buf = (float*)tensor_arena_alloc_raw_(a, sizeof(float) * size);
            }
            case null: {
                t = (FloatTensor*)malloc(sizeof(FloatTensor));
                shapeCopy = (unsigned long*)malloc(sizeof(unsigned long) * ndim);
                buf = (float*)malloc(sizeof(float) * size);
            }
        }
        i = 0UL;
        while (i < ndim) { shapeCopy[i] = shape[i]; i = i + 1UL; }

        t->data = (&heap float)buf;
        t->shape = (&heap unsigned long)shapeCopy;
        t->ndim = ndim;
        t->size = size;
        t->requiresGrad = requiresGrad;
        t->grad = (&heap float)0;
        t->gradFn = (void*)0;
        t->parents = vec_new();
        t->visited = 0;
        t->extraScalar = (float)0.0;
        t->owner = ar;
        t->dev = DEVICE_CPU;
        t->dbuf = 0UL;
        t->dgrad = 0UL;
        t->coh = TS_HOST | TS_GHOST | TS_GDEV;
        match (ar) {
            case some(a): { tensor_arena_track_(a, (void*)t); }
            case null: { }
        }
        return (&FloatTensor)t;
    }
}

static void __tensor_ensure_grad(FloatTensor* t) {
    unsafe {
        // a gradient on a device comes back first: the caller writes it here
        if (t->dev != DEVICE_CPU) { tensor_grad_host_write_((void*)t); return; }
        if (t->grad == (float*)0) {
            float* g;
            match (t->owner) {
                case some(a): { g = (float*)tensor_arena_alloc_raw_(a, sizeof(float) * t->size); }
                case null: { g = (float*)malloc(sizeof(float) * t->size); }
            }
            __tensor_ew(EW_FILL, (const float*)0, (const float*)0, (float)0.0, g, t->size);
            t->grad = (&heap float)g;
        }
    }
}

// Same shape/alloc as Tensor::zeros_like, but skips the zero-fill --
// profiled (via 'sample' on a real training run): tensor_add/sub/mul/
// scale/relu were all allocating their output through Tensor::zeros_like
// and then immediately overwriting every single element in the very next
// loop, making the zero-fill (a real, measured cost -- __bzero showed up
// as its own line in the profile) pure wasted memory-bandwidth work. Any
// caller using this MUST write every element before returning; it is not
// a safe drop-in replacement for Tensor::zeros_like everywhere.
static &FloatTensor __tensor_alloc_uninit_like(FloatTensor* t) {
    unsafe { return __tensor_alloc((const unsigned long*)t->shape, t->ndim, 0); }
}

static &FloatTensor __tensor_parent(FloatTensor* t, unsigned long idx) {
    unsafe {
        return (&FloatTensor)*t->parents.at(idx);
    }
}

static void __tensor_accumulate(FloatTensor* dst, const float* delta) {
    __tensor_ensure_grad(dst);
    unsafe {
        __tensor_ew(EW_ACC, delta, (const float*)0, (float)0.0, (float*)dst->grad, dst->size);
    }
}

static void __tensor_link2(FloatTensor* out, FloatTensor* a, FloatTensor* b, void* backwardFn) {
    unsafe {
        if (!Autograd::get().enabled()) { out->requiresGrad = 0; return; }
        out->requiresGrad = a->requiresGrad || b->requiresGrad;
        if (!out->requiresGrad) return;
        out->parents.push(a);
        out->parents.push(b);
        out->gradFn = backwardFn;
    }
}

static void __tensor_link1(FloatTensor* out, FloatTensor* a, void* backwardFn) {
    unsafe {
        if (!Autograd::get().enabled()) { out->requiresGrad = 0; return; }
        out->requiresGrad = a->requiresGrad;
        if (!out->requiresGrad) return;
        out->parents.push(a);
        out->gradFn = backwardFn;
    }
}

// ── Backward functions (one per forward op in tensor.sc) ─────────────────────
// __sub_backward through __sum_backward accumulate straight into the
// destination's grad (x->grad[i] +=) instead of building a throwaway
// 'delta' buffer and calling __tensor_accumulate -- profiled (via 'sample'
// on a real training run): the old throwaway-buffer version cost a full
// extra malloc/free and a full extra pass over the data on every single
// backward call, for no benefit at this size. __matmul_backward does the
// same through sgemm's beta = 1.

static void __add_backward(void* selfPtr) {
    unsafe {
        FloatTensor* selfT = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(selfT, 0UL);
        FloatTensor* b = __tensor_parent(selfT, 1UL);
        if (a->requiresGrad) __tensor_accumulate(a, (const float*)selfT->grad);
        if (b->requiresGrad) __tensor_accumulate(b, (const float*)selfT->grad);
    }
}

static void __sub_backward(void* selfPtr) {
    unsafe {
        FloatTensor* selfT = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(selfT, 0UL);
        FloatTensor* b = __tensor_parent(selfT, 1UL);
        if (a->requiresGrad) __tensor_accumulate(a, (const float*)selfT->grad);
        if (b->requiresGrad) {
            __tensor_ensure_grad(b);
            __tensor_ew(EW_DEC, (const float*)selfT->grad, (const float*)0, (float)0.0, (float*)b->grad, selfT->size);
        }
    }
}

static void __mul_backward(void* selfPtr) {
    unsafe {
        FloatTensor* selfT = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(selfT, 0UL);
        FloatTensor* b = __tensor_parent(selfT, 1UL);
        if (a->requiresGrad) {
            __tensor_ensure_grad(a);
            __tensor_ew(EW_ACC_MUL, (const float*)selfT->grad, (const float*)b->data, (float)0.0, (float*)a->grad, selfT->size);
        }
        if (b->requiresGrad) {
            __tensor_ensure_grad(b);
            __tensor_ew(EW_ACC_MUL, (const float*)selfT->grad, (const float*)a->data, (float)0.0, (float*)b->grad, selfT->size);
        }
    }
}

static void __scale_backward(void* selfPtr) {
    unsafe {
        FloatTensor* selfT = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(selfT, 0UL);
        if (!a->requiresGrad) return;
        __tensor_ensure_grad(a);
        __tensor_ew(EW_ACC_SCALE, (const float*)selfT->grad, (const float*)0, selfT->extraScalar, (float*)a->grad, selfT->size);
    }
}

static void __sum_backward(void* selfPtr) {
    unsafe {
        FloatTensor* selfT = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(selfT, 0UL);
        if (!a->requiresGrad) return;
        __tensor_ensure_grad(a);
        float seed = selfT->grad[0];
        __tensor_ew(EW_ACC_K, (const float*)0, (const float*)0, seed, (float*)a->grad, a->size);
    }
}

static void __matmul_backward(void* selfPtr) {
    unsafe {
        FloatTensor* selfT = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(selfT, 0UL);
        FloatTensor* b = __tensor_parent(selfT, 1UL);
        unsigned long m = a->shape[0];
        unsigned long k = a->shape[1];
        unsigned long n = b->shape[1];

        // dA[m,k] += dC[m,n] . B^T[n,k]
        // dB[k,n] += A^T[k,m] . dC[m,n]
        // Both accumulate straight into the existing gradient (beta = 1).
        if (a->requiresGrad) {
            __tensor_ensure_grad(a);
            sgemm(SGEMM_NO_TRANS, SGEMM_TRANS, m, k, n, (float)1.0,
                  (const &float)selfT->grad, n, (const &float)b->data, n, (float)1.0, (&float)a->grad, k);
        }
        if (b->requiresGrad) {
            __tensor_ensure_grad(b);
            sgemm(SGEMM_TRANS, SGEMM_NO_TRANS, k, n, m, (float)1.0,
                  (const &float)a->data, k, (const &float)selfT->grad, n, (float)1.0, (&float)b->grad, n);
        }
    }
}


// ══ Generic implementation ═══════════════════════════════════════════════════

// A tensor of E with the given shape (uninitialized values), in the active
// arena if any. Same layout and bookkeeping as __tensor_alloc.
generic<E>
static &DenseTensor<E> __dt_alloc(const unsigned long* shape, unsigned long ndim, int requiresGrad) {
    unsafe {
        ?&TensorArena ar = tensor_arena_current_();
        unsigned long size = 1UL;
        unsigned long i = 0UL;
        while (i < ndim) { size = size * shape[i]; i = i + 1UL; }
        struct DenseTensor<E>* t;
        unsigned long* shapeCopy;
        E* buf;
        match (ar) {
            case some(a): {
                t = (struct DenseTensor<E>*)tensor_arena_alloc_raw_(a, sizeof(struct DenseTensor<E>));
                shapeCopy = (unsigned long*)tensor_arena_alloc_raw_(a, sizeof(unsigned long) * ndim);
                buf = (E*)tensor_arena_alloc_raw_(a, sizeof(E) * size);
            }
            case null: {
                t = (struct DenseTensor<E>*)malloc(sizeof(struct DenseTensor<E>));
                shapeCopy = (unsigned long*)malloc(sizeof(unsigned long) * ndim);
                buf = (E*)malloc(sizeof(E) * size);
            }
        }
        i = 0UL;
        while (i < ndim) { shapeCopy[i] = shape[i]; i = i + 1UL; }
        t->data = (&heap E)buf;
        t->shape = (&heap unsigned long)shapeCopy;
        t->ndim = ndim;
        t->size = size;
        t->requiresGrad = requiresGrad;
        t->grad = (&heap float)0;
        t->gradFn = (void*)0;
        t->parents = vec_new();
        t->visited = 0;
        t->extraScalar = (float)0.0;
        t->owner = ar;
        t->dev = DEVICE_CPU;
        t->dbuf = 0UL;
        t->dgrad = 0UL;
        t->coh = TS_HOST | TS_GHOST | TS_GDEV;
        match (ar) {
            case some(a): { tensor_arena_track_(a, (void*)t); }
            case null: { }
        }
        return (&DenseTensor<E>)t;
    }
}

// A to_float / from_float node: the gradient passes to the input unchanged.
// Every tensor's gradient is float32 and the graph fields have one layout,
// so the same function serves every pair of element types.
static void __dt_cast_backward(void* selfPtr) {
    unsafe {
        FloatTensor* selfT = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(selfT, 0UL);
        if (a->requiresGrad) __tensor_accumulate(a, (const float*)selfT->grad);
    }
}

// Links 'out' to an input of another element type through __dt_cast_backward.
static void __dt_link_cast(void* out, void* in) {
    unsafe {
        FloatTensor* o = (FloatTensor*)out;
        FloatTensor* a = (FloatTensor*)in;
        if (!Autograd::get().enabled()) { o->requiresGrad = 0; return; }
        o->requiresGrad = a->requiresGrad;
        if (!o->requiresGrad) return;
        o->parents.push((&FloatTensor)a);
        o->gradFn = (void*)__dt_cast_backward;
    }
}

generic<E>
float DenseTensor::at1(unsigned long i) const {
    unsafe { if (self.dev != DEVICE_CPU) tensor_host_((void*)self); }
    float v = (float)0.0;
    if const (sizeof(E) == 4UL) { unsafe { v = self.data[i]; } }
    else { unsafe { E::load_n((const &E)&self.data[i], (&float)&v, 1UL); } }
    return v;
}

generic<E>
float DenseTensor::at2(unsigned long r, unsigned long c) const {
    unsigned long cols = 0UL;
    unsafe { cols = self.shape[1]; }
    return self.at1(r * cols + c);
}

generic<E>
void DenseTensor::set1(unsigned long i, float v) {
    unsafe { if (self.dev != DEVICE_CPU) tensor_host_write_((void*)self); }
    if const (sizeof(E) == 4UL) { unsafe { self.data[i] = v; } }
    else { unsafe { E::store_n((const &float)&v, (&E)&self.data[i], 1UL); } }
}

generic<E>
void DenseTensor::set2(unsigned long r, unsigned long c, float v) {
    unsigned long cols = 0UL;
    unsafe { cols = self.shape[1]; }
    self.set1(r * cols + c, v);
}

generic<E>
void DenseTensor::free() {
    unsafe { tensor_free_device_((void*)self); }
    match (self.owner) { case some(o): { return; } case null: { } }   // its TensorArena owns it
    unsafe {
        if ((void*)self.data != (void*)0) { std::free((void*)self.data); self.data = (E*)0; }
        if (self.shape != (unsigned long*)0) { std::free((void*)self.shape); self.shape = (unsigned long*)0; }
        if (self.grad != (float*)0) { std::free((void*)self.grad); self.grad = (float*)0; }
        self.parents.free();
    }
}

generic<E>
void DenseTensor::fill(float v) {
    unsafe { if (self.dev != DEVICE_CPU) tensor_host_write_((void*)self); }
    if const (sizeof(E) == 4UL) {
        unsafe { __tensor_ew(EW_FILL, (const float*)0, (const float*)0, v, (float*)self.data, self.size); }
    } else {
        E one;
        unsafe {
            E::store_n((const &float)&v, (&E)&one, 1UL);
            unsigned long i = 0UL;
            while (i < self.size) { self.data[i] = one; i = i + 1UL; }
        }
    }
}

generic<E>
&DenseTensor<E> DenseTensor::new_1d(unsigned long n, int requiresGrad) {
    unsigned long shape[1];
    unsafe { shape[0] = n; }
    &DenseTensor<E> t = __dt_alloc(shape, 1UL, requiresGrad);
    t.fill((float)0.0);
    return t;
}

generic<E>
&DenseTensor<E> DenseTensor::new_2d(unsigned long rows, unsigned long cols, int requiresGrad) {
    unsigned long shape[2];
    unsafe { shape[0] = rows; shape[1] = cols; }
    &DenseTensor<E> t = __dt_alloc(shape, 2UL, requiresGrad);
    t.fill((float)0.0);
    return t;
}

// Internal: n float32 values into a fresh tensor's storage.
generic<E>
static void __dt_store(&DenseTensor<E> t, const &float values, unsigned long n) {
    if const (sizeof(E) == 4UL) {
        unsafe { __tensor_ew(EW_COPY, (const float*)values, (const float*)0, (float)0.0, (float*)t->data, n); }
    } else {
        unsafe { E::store_n(values, (&E)t->data, n); }
    }
}

generic<E>
&DenseTensor<E> DenseTensor::from_1d(const &float values, unsigned long n, int requiresGrad) {
    unsigned long shape[1];
    unsafe { shape[0] = n; }
    &DenseTensor<E> t = __dt_alloc(shape, 1UL, requiresGrad);
    __dt_store(t, values, n);
    return t;
}

generic<E>
&DenseTensor<E> DenseTensor::from_2d(const &float values, unsigned long rows, unsigned long cols, int requiresGrad) {
    unsigned long shape[2];
    unsafe { shape[0] = rows; shape[1] = cols; }
    &DenseTensor<E> t = __dt_alloc(shape, 2UL, requiresGrad);
    __dt_store(t, values, rows * cols);
    return t;
}

generic<E>
&DenseTensor<E> DenseTensor::zeros_like() const {
    &DenseTensor<E> out;
    unsafe { out = __dt_alloc((const unsigned long*)self.shape, self.ndim, 0); }
    out.fill((float)0.0);
    return out;
}

generic<E>
&DenseTensor<float> DenseTensor::to_float() const {
    if const (sizeof(E) == 4UL) {
        unsafe { return (&DenseTensor<float>)(const DenseTensor<float>*)self; }   // already float32: itself
    } else {
        &FloatTensor out;
        unsafe {
            out = __tensor_alloc((const unsigned long*)self.shape, self.ndim, 0);
            E::load_n((const &E)self.data, (&float)out->data, self.size);
            __dt_link_cast((void*)out, (void*)self);
        }
        return out;
    }
}

generic<E>
&DenseTensor<E> DenseTensor::from_float(const &DenseTensor<float> x) {
    &DenseTensor<E> out;
    unsafe { out = __dt_alloc((const unsigned long*)x->shape, x->ndim, 0); }
    if const (sizeof(E) == 4UL) {
        unsafe { __tensor_ew(EW_COPY, (const float*)x->data, (const float*)0, (float)0.0, (float*)out->data, x->size); }
    } else {
        unsafe { E::store_n((const &float)x->data, (&E)out->data, x->size); }
    }
    unsafe { __dt_link_cast((void*)out, (void*)x); }
    return out;
}

generic<E>
&DenseTensor<E> DenseTensor::scale(float k) const {
    if const (sizeof(E) == 4UL) { return tensor_scale_f32_(self, k); }
    else { return DenseTensor::from_float(tensor_scale_f32_(self.to_float(), k)); }
}

generic<E>
&DenseTensor<E> DenseTensor::sum() const {
    if const (sizeof(E) == 4UL) { return tensor_sum_f32_(self); }
    else { return DenseTensor::from_float(tensor_sum_f32_(self.to_float())); }
}

generic<E>
&DenseTensor<E> tensor_add(const &DenseTensor<E> a, const &DenseTensor<E> b) {
    if const (sizeof(E) == 4UL) { return tensor_add_f32_(a, b); }
    else { return DenseTensor::from_float(tensor_add_f32_(a.to_float(), b.to_float())); }
}

generic<E>
&DenseTensor<E> tensor_sub(const &DenseTensor<E> a, const &DenseTensor<E> b) {
    if const (sizeof(E) == 4UL) { return tensor_sub_f32_(a, b); }
    else { return DenseTensor::from_float(tensor_sub_f32_(a.to_float(), b.to_float())); }
}

generic<E>
&DenseTensor<E> tensor_mul(const &DenseTensor<E> a, const &DenseTensor<E> b) {
    if const (sizeof(E) == 4UL) { return tensor_mul_f32_(a, b); }
    else { return DenseTensor::from_float(tensor_mul_f32_(a.to_float(), b.to_float())); }
}

generic<E>
&DenseTensor<E> tensor_matmul(const &DenseTensor<E> a, const &DenseTensor<E> b) {
    if const (sizeof(E) == 4UL) { return tensor_matmul_f32_(a, b); }
    else { return DenseTensor::from_float(tensor_matmul_f32_(a.to_float(), b.to_float())); }
}

// ── Backward pass ────────────────────────────────────────────────────────────
// A node's parents may be of another element type (a cast node's): the walk
// only reads the graph fields, whose layout is the same for every E.
generic<E>
static void __dt_toposort(&DenseTensor<E> t, &Vec<&DenseTensor<E>> order) {
    unsafe { if (t->visited) return; }
    unsafe { t->visited = 1; }
    unsigned long n;
    unsafe { n = t->parents.length(); }
    unsigned long i = 0UL;
    while (i < n) {
        unsafe { __dt_toposort(*t->parents.at(i), order); }
        i = i + 1UL;
    }
    order.push(t);
}

// fp16 on a device (device_set_precision): the loss is scaled by the device's
// loss scale for the backward pass and every leaf's gradient unscaled after
// it (the overflow flag raised if one isn't finite); gradients the leaves
// already held are scaled first so they come out unchanged.
#define __DT_AMP_TABLE 512UL
generic<E>
static void __dt_amp_leaves_(&Vec<&DenseTensor<E>> order, int device, unsigned long amp, int unscale) {
    unsafe {
        unsigned long j = 0UL;
        if (unscale) {
            // one launch for all of them when the table fits
            unsigned long tab[__DT_AMP_TABLE * 3UL];
            unsigned long cnt = 0UL;
            unsigned long total = 0UL;
            int fits = 1;
            while (fits && j < order.length()) {
                &DenseTensor<E> t = *order.at(j);
                if (t->gradFn == (void*)0 && t->requiresGrad && t->dev == device && t->dgrad != 0UL &&
                    (t->coh & TS_GZERO) == 0 && (t->coh & TS_GDEV) != 0) {
                    if (cnt >= __DT_AMP_TABLE || total + t->size >= 4294967295UL) fits = 0;
                    else {
                        tab[cnt * 3UL] = t->dgrad; tab[cnt * 3UL + 1UL] = t->size; tab[cnt * 3UL + 2UL] = total;
                        total = total + t->size;
                        cnt = cnt + 1UL;
                    }
                }
                j = j + 1UL;
            }
            if (fits && (cnt == 0UL || tensor_amp_unscale_table_(device, amp, (const &unsigned long)&tab[0], cnt, total))) {
                j = 0UL;
                while (j < order.length()) {
                    &DenseTensor<E> t = *order.at(j);
                    if (t->gradFn == (void*)0 && t->requiresGrad && t->dev == device && t->dgrad != 0UL &&
                        (t->coh & TS_GZERO) == 0 && (t->coh & TS_GDEV) != 0) tensor_grad_dev_wrote_((void*)t);
                    j = j + 1UL;
                }
                return;
            }
            j = 0UL;
        }
        while (j < order.length()) {
            &DenseTensor<E> t = *order.at(j);
            if (t->gradFn == (void*)0 && t->requiresGrad && t->dev == device) {
                int has = (t->dgrad != 0UL && (t->coh & TS_GZERO) == 0) || (t->grad != (float*)0 && (t->coh & TS_GHOST) != 0);
                if (unscale ? t->dgrad != 0UL && (t->coh & TS_GZERO) == 0 && tensor_grad_dev_((void*)t, device)
                            : has && tensor_grad_dev_((void*)t, device)) {
                    struct DevArgs a = dev_args_();
                    a.p[0] = t->dgrad;
                    if (unscale) {
                        a.p[1] = amp;
                        device_launch_(device, DK_AMP_UNSCALE, t->size, &a);
                    } else {
                        a.p[1] = amp; a.p[2] = t->dgrad; a.u[0] = (unsigned int)EW_MULS;
                        device_launch_(device, DK_EW, t->size, &a);
                    }
                    tensor_grad_dev_wrote_((void*)t);
                }
            }
            j = j + 1UL;
        }
    }
}

generic<E>
void DenseTensor::backward() {
    unsigned long amp = 0UL;
    unsafe { if (self.dev != DEVICE_CPU && (self.coh & TS_DEVOP) != 0) amp = device_amp_state_(self.dev); }
    struct Vec<&DenseTensor<E>> order = vec_new();
    __dt_toposort(self, &order);
    unsafe {
        if (amp != 0UL) {
            __dt_amp_leaves_(&order, self.dev, amp, 0);
            // the seed: the loss scale, on the device
            int m = tensor_grad_dev_acc_((void*)self, self.dev);
            struct DevArgs a = dev_args_();
            a.p[1] = amp; a.p[2] = self.dgrad; a.u[0] = (unsigned int)EW_ACC_PK; a.u[1] = 1U;
            if (m != 0) { device_launch_(self.dev, DK_EW, self.size, &a); tensor_grad_dev_wrote_((void*)self); }
        } else {
            __tensor_ensure_grad((FloatTensor*)self);
            unsigned long i = 0UL;
            while (i < self.size) { self.grad[i] = (float)1.0; i = i + 1UL; }
        }
        unsigned long n = order.length();
        long long idx = (long long)n - 1LL;
        while (idx >= 0LL) {
            &DenseTensor<E> node = *order.at((unsigned long)idx);
            if (node->gradFn != (void*)0) {
                // the node's gradient where its backward function reads it
                int run = 1;
                if ((node->coh & TS_DEVOP) != 0) {
                    // nothing reached a device node's gradient: its backward would add zeros
                    if ((node->dgrad == 0UL && node->grad == (float*)0) ||
                        (node->dgrad != 0UL && (node->coh & (TS_GZERO | TS_GDEV)) == (TS_GZERO | TS_GDEV))) run = 0;
                    else tensor_grad_dev_((void*)node, node->dev);
                }
                else if (node->dev != DEVICE_CPU) tensor_grad_host_((void*)node);
                if (run) {
                    TensorBackwardFn backwardFn = (TensorBackwardFn)node->gradFn;
                    backwardFn((void*)node);
                }
            }
            idx = idx - 1LL;
        }
        if (amp != 0UL) __dt_amp_leaves_(&order, self.dev, amp, 1);
        unsigned long j = 0UL;
        while (j < n) {
            &DenseTensor<E> node2 = *order.at(j);
            node2->visited = 0;
            j = j + 1UL;
        }
        order.free();
    }
}

generic<E>
void DenseTensor::zero_grad() {
    unsafe {
        if (self.dev != DEVICE_CPU) {
            // zeros on both sides: no copy needed later (the device's are
            // written by whichever op first adds to them)
            if (self.dgrad != 0UL) self.coh = self.coh | TS_GZERO;
        }
        self.coh = self.coh | TS_GHOST | TS_GDEV;
        if (self.grad != (float*)0) {
            unsigned long i = 0UL;
            while (i < self.size) { self.grad[i] = (float)0.0; i = i + 1UL; }
        }
    }
}


// ── Devices ──────────────────────────────────────────────────────────────────

generic<E>
int DenseTensor::device() const { return self.dev; }

generic<E>
int DenseTensor::place(int device) {
    if const (sizeof(E) != 4UL) { return device == DEVICE_CPU; }
    else { unsafe { return tensor_place_((void*)self, device); } }
}

generic<E>
int DenseTensor::sync() {
    unsafe { tensor_host_((void*)self); return (self.coh & TS_HOST) != 0; }
}

generic<E>
unsigned long DenseTensor::device_buffer() const { return self.dbuf; }

// A copy on another device: a cast-like graph node (the gradient passes
// back unchanged to wherever the input lives).
generic<E>
&DenseTensor<E> DenseTensor::to(int device) const {
    if (device == self.dev) { unsafe { return (&DenseTensor<E>)(const DenseTensor<E>*)self; } }
    &DenseTensor<E> out;
    unsafe { out = __dt_alloc((const unsigned long*)self.shape, self.ndim, 0); }
    unsafe {
        tensor_host_((void*)self);
        unsigned long i = 0UL;
        while (i < self.size) { out->data[i] = self.data[i]; i = i + 1UL; }
        __dt_link_cast((void*)out, (void*)self);
    }
    if const (sizeof(E) == 4UL) {
        if (device != DEVICE_CPU) { unsafe { tensor_place_((void*)out, device); } }
    }
    return out;
}

} // namespace std
