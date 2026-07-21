// SafeC Standard Library — GPU-backed Tensor ops implementation (see
// tensor_gpu.h).
#pragma once
#include <std/ml/tensor_gpu.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/gpu_mps.h>

namespace std {

extern void* malloc(unsigned long size);
extern void  free(void* ptr);

// ── Batched-dispatch support ──────────────────────────────────────────────────
// When mps_batch_is_active() (see gpu_mps.h's mps_batch_begin comment), an
// mps_*_f32 call returns immediately without waiting for the GPU or
// populating its 'out' buffer -- both happen later, inside mps_batch_end().
// Tensor being float32 (same as every mps_*_f32 buffer) means 'out->data'
// itself can be passed straight through as that buffer -- no separate
// malloc'd scratch + deferred float->double conversion step the way this
// file needed when Tensor was double (that used a
// '__MpsF32ToF64Ctx'/finalize-callback pair registered via
// mps_batch_register_finalize; both are gone, since there's no conversion
// left to defer). 'out->data' is heap-allocated and stays valid until the
// Tensor itself is freed, well past batch_end()'s GPU readback into it.

// ── Chain tracking ───────────────────────────────────────────────────────────
// Which Tensor*s currently hold a batched-but-not-yet-synced GPU result,
// and which GPU buffer that result actually lives in. tensor_matmul_gpu/
// tensor_relu_gpu/tensor_sub_gpu/tensor_square_gpu/tensor_sum_gpu consult
// this before uploading an input: if the input tensor is in this table,
// its real data is already sitting in a GPU buffer from an earlier op in
// the SAME open batch, so the mps_*_f32_chained entry points (gpu_mps.h)
// read that buffer directly instead of re-uploading the tensor's
// (not-yet-populated) 'data' — which is the bug this table exists to
// prevent (see gpu_mps.h's mps_batch_begin comment for how it was found).
//
// TWO separate tables, not one: forward ops mark a Tensor's pending
// *->data* (this is the DATA table); backward ops mark a Tensor's pending
// *->grad* (the GRAD table) for the NEXT backward call in the walk to pick
// up. Before the fused forward+loss+backward training step existed, a
// batch was always either all-forward or all-backward (never both), so
// reusing ONE table for both purposes was safe — tensor_gpu_batch_begin()
// resetting it once per batch was enough. Now that loss ops (tensor_sub_
// gpu/tensor_square_gpu/tensor_sum_gpu) let forward and backward run
// inside the SAME open batch (see the training loop's single
// tensor_gpu_batch_begin/end pair), a backward step's grad-table lookup
// on a Tensor* would otherwise collide with that SAME Tensor*'s still-live
// DATA entry from forward — e.g. __matmul_backward_gpu looking up Y's
// pending *gradient* buffer would instead find Y's pending *data* buffer
// from tensor_matmul_gpu's own forward marking, silently computing
// garbage. Two independent tables, both reset only at
// tensor_gpu_batch_begin(), avoid that collision entirely.
//
// Sized to match gpu_mps.sc's own MPS_BATCH_MAX_PENDING; entries are
// appended while a batch is open and never removed mid-batch (stale
// lookups just miss, which only costs a redundant upload, never wrong
// results) — both tables are logically invalidated by mps_batch_end()
// returning, since every reader only consults them while their own count
// reflects the CURRENTLY open batch (reset at the top of both, whenever
// no batch is active).
#define MPS_CHAIN_MAX 512
static struct Tensor* __mps_chain_data_tensor[MPS_CHAIN_MAX];
static void*          __mps_chain_data_buf[MPS_CHAIN_MAX];
static unsigned long  __mps_chain_data_count = 0UL;
static struct Tensor* __mps_chain_grad_tensor[MPS_CHAIN_MAX];
static void*          __mps_chain_grad_buf[MPS_CHAIN_MAX];
static unsigned long  __mps_chain_grad_count = 0UL;

// The correct place to clear these tables is tensor_gpu_batch_begin()
// (right when a NEW batch starts, before any op runs) — see that
// function and tensor_gpu.h's own comment on why "reset if idle" checked
// from inside an op is always one call too late.
void tensor_gpu_batch_begin() {
    __mps_chain_data_count = 0UL;
    __mps_chain_grad_count = 0UL;
    mps_batch_begin();
}

void tensor_gpu_batch_end() {
    mps_batch_end();
}

static void __mps_chain_data_mark(struct Tensor* t, void* buf) {
    if (__mps_chain_data_count < (unsigned long)MPS_CHAIN_MAX) {
        __mps_chain_data_tensor[__mps_chain_data_count] = t;
        __mps_chain_data_buf[__mps_chain_data_count] = buf;
        __mps_chain_data_count = __mps_chain_data_count + 1UL;
    }
}

// ── Cross-batch persistent tensors ──────────────────────────────────────────
// A second, separate table from the per-batch chain tables above: entries
// here are NOT reset by tensor_gpu_batch_begin() and stay valid across as
// many batches as the caller wants (typically an entire training run).
// tensor_gpu_mark_persistent(t) uploads t's CURRENT ->data to a device
// buffer once (mps_upload_persistent, see gpu_mps.h) and registers it here;
// __mps_chain_data_lookup below then finds it automatically for every
// caller that already consults that function (tensor_matmul_gpu/
// tensor_relu_gpu's forward chaining, __matmul_backward_gpu/__relu_
// backward_gpu's dB/aDataBuf checks) — no other code needs to change to
// benefit. Only correct for a tensor whose ->data genuinely never changes
// for as long as it stays marked (e.g. a training loop's fixed input X,
// not a weight matrix SGD rewrites every step) — nothing here re-uploads
// on a data change, since nothing here is told one happened.
#define MPS_PERSISTENT_MAX 16
static struct Tensor* __mps_persistent_tensor[MPS_PERSISTENT_MAX];
static void*          __mps_persistent_buf[MPS_PERSISTENT_MAX];
static unsigned long  __mps_persistent_count = 0UL;

static void* __mps_persistent_lookup(struct Tensor* t) {
    unsigned long i = 0UL;
    while (i < __mps_persistent_count) {
        if (__mps_persistent_tensor[i] == t) return __mps_persistent_buf[i];
        i = i + 1UL;
    }
    return (void*)0;
}

void tensor_gpu_mark_persistent(const &Tensor t) {
    unsafe {
        if (__mps_persistent_count >= (unsigned long)MPS_PERSISTENT_MAX) return;
        if (!mps_available()) return;
        struct Tensor* tp = (struct Tensor*)t;
        void* buf = mps_upload_persistent((const float*)tp->data, tp->size * sizeof(float));
        if (buf == (void*)0) return;
        __mps_persistent_tensor[__mps_persistent_count] = tp;
        __mps_persistent_buf[__mps_persistent_count] = buf;
        __mps_persistent_count = __mps_persistent_count + 1UL;
    }
}

// Releases every tensor mps_release_persistent above ever marked, in
// registration order, and empties the table. Call once, when truly done
// (e.g. at the very end of a training run) — not safe to call while a
// batch that might still be reading one of these buffers is open.
void tensor_gpu_release_all_persistent() {
    unsafe {
        unsigned long i = 0UL;
        while (i < __mps_persistent_count) {
            mps_release_persistent(__mps_persistent_buf[i], __mps_persistent_tensor[i]->size * sizeof(float));
            i = i + 1UL;
        }
        __mps_persistent_count = 0UL;
    }
}

// Returns the pending GPU data buffer for 't', or NULL if 't' isn't
// batched-pending (an ordinary, already-populated tensor — the common
// case, e.g. a persistent weight matrix) AND isn't cross-batch-persistent
// either (see tensor_gpu_mark_persistent above).
static void* __mps_chain_data_lookup(struct Tensor* t) {
    unsigned long i = __mps_chain_data_count;
    // Walk backwards: if the same Tensor* were ever marked more than
    // once (not expected in practice — a fresh Tensor* per op), the most
    // recent entry is the correct one.
    while (i > 0UL) {
        i = i - 1UL;
        if (__mps_chain_data_tensor[i] == t) return __mps_chain_data_buf[i];
    }
    return __mps_persistent_lookup(t);
}

static void __mps_chain_grad_mark(struct Tensor* t, void* buf) {
    if (__mps_chain_grad_count < (unsigned long)MPS_CHAIN_MAX) {
        __mps_chain_grad_tensor[__mps_chain_grad_count] = t;
        __mps_chain_grad_buf[__mps_chain_grad_count] = buf;
        __mps_chain_grad_count = __mps_chain_grad_count + 1UL;
    }
}

// Returns the pending GPU gradient buffer for 't', or NULL if 't'*s
// gradient isn't batched-pending (e.g. a leaf tensor like a weight matrix,
// whose ->grad is only ever accumulated into via a deferred finalize, not
// chained further).
static void* __mps_chain_grad_lookup(struct Tensor* t) {
    unsigned long i = __mps_chain_grad_count;
    while (i > 0UL) {
        i = i - 1UL;
        if (__mps_chain_grad_tensor[i] == t) return __mps_chain_grad_buf[i];
    }
    return (void*)0;
}

&Tensor tensor_add_gpu(const &Tensor a, const &Tensor b) {
    if (!mps_available()) { return tensor_add(a, b); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        int ok = mps_add_f32((const float*)a->data, (const float*)b->data, (float*)out->data, out->size);
        if (!ok) {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] + b->data[i]; i = i + 1UL; }
        }
    }
    __tensor_link2(out, a, b, (void*)__add_backward);
    return out;
}

&Tensor tensor_sub_gpu(const &Tensor a, const &Tensor b) {
    if (!mps_available()) { return tensor_sub(a, b); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        if (mps_batch_is_active()) {
            void* chainedBuf = __mps_chain_data_lookup((struct Tensor*)a);
            int ok;
            if (chainedBuf != (void*)0) {
                // 'a' is itself still-pending output from an earlier op in
                // this same batch (e.g. a matmul's result feeding straight
                // into the loss computation) -- read its real GPU buffer
                // directly instead of uploading a->data (not populated yet).
                ok = mps_sub_f32_chained(chainedBuf, (const float*)b->data, (float*)out->data, out->size);
            } else {
                ok = mps_sub_f32((const float*)a->data, (const float*)b->data, (float*)out->data, out->size);
            }
            if (ok) {
                __mps_chain_data_mark(out, mps_batch_last_output_buffer());
            } else if (chainedBuf == (void*)0) {
                unsigned long i = 0UL;
                while (i < out->size) { out->data[i] = a->data[i] - b->data[i]; i = i + 1UL; }
            }
            // Chained dispatch failing with no CPU-side fallback available
            // is the one case this codebase's "always safe to call" GPU-op
            // guarantee can't uphold -- see tensor_relu_gpu's comment.
        } else {
            int ok = mps_sub_f32((const float*)a->data, (const float*)b->data, (float*)out->data, out->size);
            if (!ok) {
                unsigned long i = 0UL;
                while (i < out->size) { out->data[i] = a->data[i] - b->data[i]; i = i + 1UL; }
            }
        }
    }
    __tensor_link2(out, a, b, (void*)__sub_backward_gpu);
    return out;
}

// out = a * a -- forward twin of tensor_mul_gpu(a, a), but chain-aware for
// the case where 'a' is itself still GPU-pending (see tensor_gpu.h's
// comment on why tensor_mul_gpu's generic two-CPU-array signature can't
// express that).
&Tensor tensor_square_gpu(const &Tensor a) {
    if (!mps_available()) { return tensor_mul(a, a); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        if (mps_batch_is_active()) {
            void* chainedBuf = __mps_chain_data_lookup((struct Tensor*)a);
            int ok;
            if (chainedBuf != (void*)0) {
                ok = mps_square_f32_chained(chainedBuf, (float*)out->data, out->size);
            } else {
                ok = mps_mul_f32((const float*)a->data, (const float*)a->data, (float*)out->data, out->size);
            }
            if (ok) {
                __mps_chain_data_mark(out, mps_batch_last_output_buffer());
            } else if (chainedBuf == (void*)0) {
                unsigned long i = 0UL;
                while (i < out->size) { float v = a->data[i]; out->data[i] = v * v; i = i + 1UL; }
            }
        } else {
            int ok = mps_mul_f32((const float*)a->data, (const float*)a->data, (float*)out->data, out->size);
            if (!ok) {
                unsigned long i = 0UL;
                while (i < out->size) { float v = a->data[i]; out->data[i] = v * v; i = i + 1UL; }
            }
        }
    }
    __tensor_link1(out, a, (void*)__square_backward_gpu);
    return out;
}

&Tensor tensor_mul_gpu(const &Tensor a, const &Tensor b) {
    if (!mps_available()) { return tensor_mul(a, b); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        int ok = mps_mul_f32((const float*)a->data, (const float*)b->data, (float*)out->data, out->size);
        if (!ok) {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] * b->data[i]; i = i + 1UL; }
        }
    }
    __tensor_link2(out, a, b, (void*)__mul_backward);
    return out;
}

&Tensor tensor_scale_gpu(const &Tensor a, float k) {
    if (!mps_available()) { return tensor_scale(a, k); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        int ok = mps_scale_f32((const float*)a->data, k, (float*)out->data, out->size);
        if (!ok) {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] * k; i = i + 1UL; }
        }
    }
    unsafe { out->extraScalar = k; }
    __tensor_link1(out, a, (void*)__scale_backward);
    return out;
}

// ── GPU-backed backward passes ────────────────────────────────────────────────
// Profiling train_gpu.sc's actual training loop (not guessing) found the
// real bottleneck: tensor_matmul_gpu/tensor_relu_gpu only ran their FORWARD
// pass on the GPU — both were still linked to tensor.h/tensor_nn.h's plain
// CPU __matmul_backward/__relu_backward for backward, the same functions
// the naive CPU-only path uses. Measured directly: backward accounted for
// ~2400ms out of a ~2650ms training step (roughly 90%) while GPU forward
// was ~215ms — every earlier fix to the GPU dispatch path itself (pipeline
// caching, compile-time kernel loading, buffer pooling) could only ever
// move that ~215ms, never the ~2400ms actually dominating the number. This
// is exactly the gap PyTorch and MLX don't have: moving a tensor to a GPU
// device there puts the *entire* graph — forward and backward alike — on
// that device; there's no per-op forward/backward device split the way
// this file used to have. __matmul_backward_gpu/__relu_backward_gpu close
// it the same way tensor_blas.sc's __matmul_backward_blas already closes
// the equivalent gap for the BLAS path: dedicated backward kernels
// (mps_matmul_abt_f32/mps_matmul_atb_f32/mps_relu_backward_f32, see
// gpu_mps.h) instead of falling through to the CPU functions.
//
// Batched backward (mps_batch_is_active()): a training step's backward
// pass is its own sequential chain (matmul2-backward -> relu-backward ->
// matmul1-backward, each op waiting on the previous), same shape as
// forward's matmul->relu->matmul, and pays the same per-op command-buffer
// round-trip cost forward used to before tensor_gpu_batch_begin/end
// existed. Uses the GRAD chain table (__mps_chain_grad_mark/__mps_chain_
// grad_lookup, see the "Chain tracking" comment above): marking a Tensor*
// right after computing ITS OWN gradient contribution works because that
// Tensor* is exactly the 'selfT' the NEXT backward call in the walk
// receives (e.g. matmul2-backward computes into 'a' = H, and H's own
// gradFn -- relu-backward -- is called next with selfT=H).
//
// Gradients accumulate (a Tensor used more than once sums every
// contribution into ->grad), but a batched dispatch's result isn't
// materialized in CPU memory until mps_batch_end()'s single readback --
// so the accumulate step (target[i] += result[i]) has to be deferred too,
// via mps_batch_register_finalize, into a scratch buffer rather than
// straight into ->grad. Scope limit: this correctly sums multiple
// finalize-deferred contributions to the same Tensor (each one's own
// finalize runs in registration order and accumulates), but
// __mps_chain_grad_lookup only returns the MOST RECENT producer for a
// given Tensor* -- a Tensor with two GPU-batched consumers reading its
// (not-yet-finalized) grad within the same batch would only chain to the
// last one. Not reachable by this library's own MLP/transformer graphs
// (every intermediate tensor here is consumed exactly once on the way
// back), but a real limit for a future branching graph under batching.
struct __MpsGradAccumCtx {
    float* scratch;
    float* target;
    unsigned long count;
};

static void __mps_grad_accum_finalize(void* ctxPtr) {
    unsafe {
        struct __MpsGradAccumCtx* ctx = (struct __MpsGradAccumCtx*)ctxPtr;
        unsigned long i = 0UL;
        while (i < ctx->count) { ctx->target[i] = ctx->target[i] + ctx->scratch[i]; i = i + 1UL; }
        free((void*)ctx->scratch);
        free((void*)ctx);
    }
}

// dSub/dA = identity, dSub/dB = negate -- same math as __sub_backward, but
// chain-aware: if selfT's OWN gradient is itself still GPU-pending (the
// usual case in the fused forward+loss+backward training step), 'a' just
// takes over that SAME pending buffer (no GPU dispatch needed at all,
// since the gradient value doesn't change), and 'b' -- if it even requires
// grad, which target/label tensors normally don't -- gets a chained
// negate dispatch instead of reading selfT->grad from the CPU (not valid
// yet). Falls back to plain __sub_backward whenever selfT->grad isn't
// chain-pending (not batched at all, or 'a' was an ordinary tensor).
static void __sub_backward_gpu(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        struct Tensor* b = __tensor_parent(selfT, 1UL);

        if (!mps_batch_is_active()) { __sub_backward(selfPtr); return; }

        void* selfGradBuf = __mps_chain_grad_lookup(selfT);
        if (selfGradBuf == (void*)0) {
            if (a->requiresGrad) { __tensor_accumulate(a, (const float*)selfT->grad); }
            if (b->requiresGrad) {
                __tensor_ensure_grad(b);
                unsigned long i = 0UL;
                while (i < selfT->size) { b->grad[i] = b->grad[i] - selfT->grad[i]; i = i + 1UL; }
            }
            return;
        }

        if (a->requiresGrad) { __mps_chain_grad_mark(a, selfGradBuf); }

        if (b->requiresGrad) {
            __tensor_ensure_grad(b);
            float* scratch = (float*)malloc(sizeof(float) * selfT->size);
            int ok = mps_scale_f32_chained(selfGradBuf, (float)-1.0, scratch, selfT->size);
            if (ok) {
                struct __MpsGradAccumCtx* ctx = (struct __MpsGradAccumCtx*)malloc(sizeof(struct __MpsGradAccumCtx));
                ctx->scratch = scratch; ctx->target = (float*)b->grad; ctx->count = selfT->size;
                mps_batch_register_finalize((MpsFinalizeFn)__mps_grad_accum_finalize, (void*)ctx);
            } else {
                free((void*)scratch);
                // Chained dispatch failed with no CPU-side fallback
                // available -- same documented gap as tensor_relu_gpu's
                // forward chain.
            }
        }
    }
}

// d(a*a)/da = 2*a*upstreamGrad -- chain-aware backward for tensor_square_
// gpu. upstreamGrad (selfT->grad) is always an ordinary CPU array by the
// time this runs in the training loop's actual loss chain (see
// __sum_backward: it writes a plain CPU broadcast, no GPU involvement at
// all), so only 'a's DATA needs the chained path here.
static void __square_backward_gpu(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (!a->requiresGrad) return;
        __tensor_ensure_grad(a);

        if (mps_batch_is_active()) {
            void* aDataBuf = __mps_chain_data_lookup(a);
            if (aDataBuf != (void*)0) {
                float* twoGrad = (float*)malloc(sizeof(float) * selfT->size);
                unsigned long i = 0UL;
                while (i < selfT->size) { twoGrad[i] = (float)2.0 * selfT->grad[i]; i = i + 1UL; }

                float* scratch = (float*)malloc(sizeof(float) * selfT->size);
                int ok = mps_mul_f32_chained(aDataBuf, (const float*)twoGrad, scratch, selfT->size);
                free((void*)twoGrad);
                if (ok) {
                    struct __MpsGradAccumCtx* ctx = (struct __MpsGradAccumCtx*)malloc(sizeof(struct __MpsGradAccumCtx));
                    ctx->scratch = scratch; ctx->target = (float*)a->grad; ctx->count = selfT->size;
                    mps_batch_register_finalize((MpsFinalizeFn)__mps_grad_accum_finalize, (void*)ctx);
                    __mps_chain_grad_mark(a, mps_batch_last_output_buffer());
                } else {
                    free((void*)scratch);
                    // Chained dispatch failed with no CPU-side fallback
                    // available -- same documented gap as tensor_relu_gpu's
                    // forward chain.
                }
                return;
            }
        }

        // 'a' isn't chain-pending -- its data is an ordinary CPU array,
        // safe to read directly even mid-batch (or there's no batch at all).
        unsigned long i = 0UL;
        while (i < selfT->size) { a->grad[i] = a->grad[i] + (float)2.0 * a->data[i] * selfT->grad[i]; i = i + 1UL; }
    }
}

static void __relu_backward_gpu(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (!a->requiresGrad) return;
        __tensor_ensure_grad(a);
        if (!mps_available()) { __relu_backward(selfPtr); return; }

        if (mps_batch_is_active()) {
            // 'a' (Z1, relu's own input) can itself be still-pending here
            // too, same reasoning as __matmul_backward_gpu's dB branch --
            // in the fused forward+loss+backward training step, relu's
            // input was computed by tensor_matmul_gpu earlier in this SAME
            // open batch and hasn't been read back to CPU yet.
            void* selfGradBuf = __mps_chain_grad_lookup(selfT);
            void* aDataBuf = __mps_chain_data_lookup(a);
            float* scratch = (float*)malloc(sizeof(float) * selfT->size);
            int ok;
            if (selfGradBuf != (void*)0 && aDataBuf != (void*)0) {
                ok = mps_relu_backward_f32_chained_ab(aDataBuf, selfGradBuf, scratch, selfT->size);
            } else if (selfGradBuf != (void*)0) {
                ok = mps_relu_backward_f32_chained((const float*)a->data, selfGradBuf, scratch, selfT->size);
            } else {
                ok = mps_relu_backward_f32((const float*)a->data, (const float*)selfT->grad, scratch, selfT->size);
            }
            if (ok) {
                struct __MpsGradAccumCtx* ctx = (struct __MpsGradAccumCtx*)malloc(sizeof(struct __MpsGradAccumCtx));
                ctx->scratch = scratch; ctx->target = (float*)a->grad; ctx->count = selfT->size;
                mps_batch_register_finalize((MpsFinalizeFn)__mps_grad_accum_finalize, (void*)ctx);
                __mps_chain_grad_mark(a, mps_batch_last_output_buffer());
            } else if (selfGradBuf == (void*)0 && aDataBuf == (void*)0) {
                free((void*)scratch);
                unsigned long i = 0UL;
                while (i < selfT->size) {
                    if (a->data[i] > (float)0.0) { a->grad[i] = a->grad[i] + selfT->grad[i]; }
                    i = i + 1UL;
                }
            } else {
                // Chained dispatch failed with a chain-pending operand and
                // no safe CPU fallback available -- same documented gap as
                // tensor_relu_gpu's forward chain.
                free((void*)scratch);
            }
            return;
        }

        float* fout = (float*)alloc(checked_mul_size(sizeof(float), selfT->size));
        int ok = mps_relu_backward_f32((const float*)a->data, (const float*)selfT->grad, fout, selfT->size);
        if (ok) {
            unsigned long i = 0UL;
            while (i < selfT->size) { a->grad[i] = a->grad[i] + fout[i]; i = i + 1UL; }
        } else {
            // Device present but dispatch failed -- same rare fallback
            // shape as tensor_matmul_gpu's __tensor_matmul_cpu_into.
            unsigned long i = 0UL;
            while (i < selfT->size) {
                if (a->data[i] > (float)0.0) { a->grad[i] = a->grad[i] + selfT->grad[i]; }
                i = i + 1UL;
            }
        }
        dealloc((void*)fout);
    }
}

static void __matmul_backward_gpu(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        struct Tensor* b = __tensor_parent(selfT, 1UL);
        unsigned long m = a->shape[0];
        unsigned long k = a->shape[1];
        unsigned long n = b->shape[1];
        if (!mps_available()) { __matmul_backward(selfPtr); return; }

        int batched = mps_batch_is_active();
        void* dcBuf = batched ? __mps_chain_grad_lookup(selfT) : (void*)0;

        if (a->requiresGrad) {
            __tensor_ensure_grad(a);
            // dA[m,k] = dC[m,n] . B^T[n,k]
            if (batched) {
                float* scratch = (float*)malloc(sizeof(float) * m * k);
                int ok = (dcBuf != (void*)0)
                    ? mps_matmul_abt_f32_chained(dcBuf, (const float*)b->data, scratch, m, n, k)
                    : mps_matmul_abt_f32((const float*)selfT->grad, (const float*)b->data, scratch, m, n, k);
                if (ok) {
                    struct __MpsGradAccumCtx* ctx = (struct __MpsGradAccumCtx*)malloc(sizeof(struct __MpsGradAccumCtx));
                    ctx->scratch = scratch; ctx->target = (float*)a->grad; ctx->count = m * k;
                    mps_batch_register_finalize((MpsFinalizeFn)__mps_grad_accum_finalize, (void*)ctx);
                    __mps_chain_grad_mark(a, mps_batch_last_output_buffer());
                } else if (dcBuf == (void*)0) {
                    free((void*)scratch);
                    unsigned long ii = 0UL;
                    while (ii < m) {
                        unsigned long jj = 0UL;
                        while (jj < k) {
                            float acc = (float)0.0;
                            unsigned long p = 0UL;
                            while (p < n) { acc = acc + selfT->grad[ii * n + p] * b->data[jj * n + p]; p = p + 1UL; }
                            a->grad[ii * k + jj] = a->grad[ii * k + jj] + acc;
                            jj = jj + 1UL;
                        }
                        ii = ii + 1UL;
                    }
                } else {
                    free((void*)scratch);
                }
            } else {
                float* fdA = (float*)alloc(checked_mul_size(sizeof(float), m * k));
                int ok = mps_matmul_abt_f32((const float*)selfT->grad, (const float*)b->data, fdA, m, n, k);
                if (ok) {
                    unsigned long i = 0UL;
                    while (i < m * k) { a->grad[i] = a->grad[i] + fdA[i]; i = i + 1UL; }
                } else {
                    unsigned long ii = 0UL;
                    while (ii < m) {
                        unsigned long jj = 0UL;
                        while (jj < k) {
                            float acc = (float)0.0;
                            unsigned long p = 0UL;
                            while (p < n) { acc = acc + selfT->grad[ii * n + p] * b->data[jj * n + p]; p = p + 1UL; }
                            a->grad[ii * k + jj] = a->grad[ii * k + jj] + acc;
                            jj = jj + 1UL;
                        }
                        ii = ii + 1UL;
                    }
                }
                dealloc((void*)fdA);
            }
        }
        if (b->requiresGrad) {
            __tensor_ensure_grad(b);
            // dB[k,n] = A^T[k,m] . dC[m,n]
            if (batched) {
                // 'a' itself can be still-pending here too -- unlike
                // matmul1's backward (whose 'a' is always X, a real input),
                // matmul2's 'a' is H, an activation this SAME open batch
                // computed via tensor_relu_gpu and hasn't read back yet in
                // the fused forward+loss+backward training step. Reading
                // a->data directly in that case would silently read
                // garbage -- see mps_matmul_atb_f32_chained_ab's comment.
                void* aDataBuf = __mps_chain_data_lookup(a);
                float* scratch = (float*)malloc(sizeof(float) * k * n);
                int ok;
                if (dcBuf != (void*)0 && aDataBuf != (void*)0) {
                    ok = mps_matmul_atb_f32_chained_ab(aDataBuf, dcBuf, scratch, m, k, n);
                } else if (dcBuf != (void*)0) {
                    ok = mps_matmul_atb_f32_chained((const float*)a->data, dcBuf, scratch, m, k, n);
                } else {
                    ok = mps_matmul_atb_f32((const float*)a->data, (const float*)selfT->grad, scratch, m, k, n);
                }
                if (ok) {
                    struct __MpsGradAccumCtx* ctx = (struct __MpsGradAccumCtx*)malloc(sizeof(struct __MpsGradAccumCtx));
                    ctx->scratch = scratch; ctx->target = (float*)b->grad; ctx->count = k * n;
                    mps_batch_register_finalize((MpsFinalizeFn)__mps_grad_accum_finalize, (void*)ctx);
                    // 'b' (a weight matrix) isn't chain-marked -- nothing
                    // downstream reads a parameter's own gradient as an
                    // upstream input the way an intermediate activation's is.
                } else if (dcBuf == (void*)0 && aDataBuf == (void*)0) {
                    free((void*)scratch);
                    unsigned long p = 0UL;
                    while (p < m) {
                        unsigned long ii = 0UL;
                        while (ii < k) {
                            float aVal = a->data[p * k + ii];
                            unsigned long jj = 0UL;
                            while (jj < n) {
                                b->grad[ii * n + jj] = b->grad[ii * n + jj] + aVal * selfT->grad[p * n + jj];
                                jj = jj + 1UL;
                            }
                            ii = ii + 1UL;
                        }
                        p = p + 1UL;
                    }
                } else {
                    free((void*)scratch);
                    // Chained dispatch failed with a chain-pending operand
                    // and no safe CPU fallback available -- same documented
                    // gap as tensor_relu_gpu's forward chain.
                }
            } else {
                float* fdB = (float*)alloc(checked_mul_size(sizeof(float), k * n));
                int ok = mps_matmul_atb_f32((const float*)a->data, (const float*)selfT->grad, fdB, m, k, n);
                if (ok) {
                    unsigned long i = 0UL;
                    while (i < k * n) { b->grad[i] = b->grad[i] + fdB[i]; i = i + 1UL; }
                } else {
                    unsigned long p = 0UL;
                    while (p < m) {
                        unsigned long ii = 0UL;
                        while (ii < k) {
                            float aVal = a->data[p * k + ii];
                            unsigned long jj = 0UL;
                            while (jj < n) {
                                b->grad[ii * n + jj] = b->grad[ii * n + jj] + aVal * selfT->grad[p * n + jj];
                                jj = jj + 1UL;
                            }
                            ii = ii + 1UL;
                        }
                        p = p + 1UL;
                    }
                }
                dealloc((void*)fdB);
            }
        }
    }
}

&Tensor tensor_relu_gpu(const &Tensor a) {
    if (!mps_available()) { return tensor_relu(a); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        if (mps_batch_is_active()) {
            void* chainedBuf = __mps_chain_data_lookup((struct Tensor*)a);
            int ok;
            if (chainedBuf != (void*)0) {
                // 'a' is itself still-pending output from an earlier op
                // in this same batch -- read its real GPU buffer directly
                // instead of uploading a->data (not populated yet).
                ok = mps_relu_f32_chained(chainedBuf, (float*)out->data, out->size);
            } else {
                ok = mps_relu_f32((const float*)a->data, (float*)out->data, out->size);
            }
            if (ok) {
                __mps_chain_data_mark(out, mps_batch_last_output_buffer());
            } else if (chainedBuf == (void*)0) {
                // Dispatch failed even with a device present -- can't defer
                // a CPU fallback into a GPU batch's finalize step, so
                // compute it immediately instead. Only possible on the
                // non-chained path: 'a' itself came from CPU data, so a
                // CPU fallback can read it.
                unsigned long i = 0UL;
                while (i < out->size) {
                    float v = a->data[i];
                    out->data[i] = (v > (float)0.0) ? v : (float)0.0;
                    i = i + 1UL;
                }
            }
            // Chained dispatch failing with no CPU-side fallback available
            // (chainedBuf != 0 and ok == 0) is the one case this codebase's
            // "always safe to call, just not always actually on the GPU"
            // GPU-op guarantee (see tensor_gpu.h) can't uphold -- 'out->data'
            // is left uninitialized rather than silently wrong, since
            // recovering a CPU value here would need synchronously
            // flushing mid-batch.
        } else {
            int ok = mps_relu_f32((const float*)a->data, (float*)out->data, out->size);
            if (!ok) {
                unsigned long i = 0UL;
                while (i < out->size) {
                    float v = a->data[i];
                    out->data[i] = (v > (float)0.0) ? v : (float)0.0;
                    i = i + 1UL;
                }
            }
        }
    }
    __tensor_link1(out, a, (void*)__relu_backward_gpu);
    return out;
}

&Tensor tensor_sum_gpu(const &Tensor a) {
    if (!mps_available()) { return tensor_sum(a); }
    unsigned long one[1];
    unsafe { one[0] = 1UL; }
    struct Tensor* out = __tensor_alloc(one, 1UL, 0);
    unsafe {
        if (mps_batch_is_active()) {
            void* chainedBuf = __mps_chain_data_lookup((struct Tensor*)a);
            int ok;
            if (chainedBuf != (void*)0) {
                // out->data[0] isn't valid until mps_batch_end()'s deferred
                // finalize runs (see mps_sum_f32's comment) -- fine here,
                // since nothing reads a loss tensor's own DATA during the
                // backward walk that follows in the same open batch, only
                // its seeded ->grad.
                ok = mps_sum_f32_chained(chainedBuf, (float*)&out->data[0], a->size);
            } else {
                ok = mps_sum_f32((const float*)a->data, (float*)&out->data[0], a->size);
            }
            if (!ok && chainedBuf == (void*)0) {
                float acc = (float)0.0;
                unsigned long i = 0UL;
                while (i < a->size) { acc = acc + a->data[i]; i = i + 1UL; }
                out->data[0] = acc;
            }
        } else {
            float fout[1];
            int ok = mps_sum_f32((const float*)a->data, (float*)fout, a->size);
            if (ok) {
                out->data[0] = fout[0];
            } else {
                float acc = (float)0.0;
                unsigned long i = 0UL;
                while (i < a->size) { acc = acc + a->data[i]; i = i + 1UL; }
                out->data[0] = acc;
            }
        }
    }
    __tensor_link1(out, a, (void*)__sum_backward);
    return out;
}

// Shared by tensor_matmul_gpu's batched and non-batched CPU-fallback
// paths (dispatch failed despite a device being present) -- same i,p,j
// loop as tensor_matmul's CPU path (see its comment for why that order).
static void __tensor_matmul_cpu_into(struct Tensor* out, struct Tensor* a, struct Tensor* b,
                                      unsigned long m, unsigned long k, unsigned long n) {
    unsafe {
        unsigned long i = 0UL;
        while (i < m) {
            unsigned long j0 = 0UL;
            while (j0 < n) { out->data[i * n + j0] = (float)0.0; j0 = j0 + 1UL; }
            i = i + 1UL;
        }
        i = 0UL;
        while (i < m) {
            unsigned long p = 0UL;
            while (p < k) {
                float aVal = a->data[i * k + p];
                unsigned long j = 0UL;
                while (j < n) {
                    out->data[i * n + j] = out->data[i * n + j] + aVal * b->data[p * n + j];
                    j = j + 1UL;
                }
                p = p + 1UL;
            }
            i = i + 1UL;
        }
    }
}

&Tensor tensor_matmul_gpu(const &Tensor a, const &Tensor b) {
    if (!mps_available()) { return tensor_matmul(a, b); }

    unsigned long m; unsigned long k; unsigned long n;
    unsafe { m = a->shape[0]; k = a->shape[1]; n = b->shape[1]; }
    struct Tensor* out = tensor_new_2d(m, n, 0);
    unsafe {
        if (mps_batch_is_active()) {
            void* chainedBuf = __mps_chain_data_lookup((struct Tensor*)a);
            int ok;
            if (chainedBuf != (void*)0) {
                ok = mps_matmul_f32_chained(chainedBuf, (const float*)b->data, (float*)out->data, m, k, n);
            } else {
                ok = mps_matmul_f32((const float*)a->data, (const float*)b->data, (float*)out->data, m, k, n);
            }
            if (ok) {
                __mps_chain_data_mark(out, mps_batch_last_output_buffer());
            } else if (chainedBuf == (void*)0) {
                __tensor_matmul_cpu_into(out, a, b, m, k, n);
            }
            // Chained-dispatch-failed-with-no-fallback case: see
            // tensor_relu_gpu's comment above.
        } else {
            int ok = mps_matmul_f32((const float*)a->data, (const float*)b->data, (float*)out->data, m, k, n);
            if (!ok) {
                __tensor_matmul_cpu_into(out, a, b, m, k, n);
            }
        }
    }
    __tensor_link2(out, a, b, (void*)__matmul_backward_gpu);
    return out;
}

} // namespace std
