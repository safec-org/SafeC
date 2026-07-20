// SafeC Standard Library — GPU-backed Tensor ops implementation (see
// tensor_gpu.h).
#pragma once
#include <std/ml/tensor_gpu.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/gpu_mps.h>

namespace std {

extern void* malloc(unsigned long size);
extern void  free(void* ptr);

// double -> float32 and back, shared by every op below. Metal has no
// 'double' type at all (see gpu_mps.sc), so this conversion is unavoidable
// anywhere a Tensor's 'double' data crosses into an mps_* call.
static float* __tensor_gpu_to_f32(const double* src, unsigned long n) {
    unsafe {
        float* out = (float*)alloc(checked_mul_size(sizeof(float), n));
        unsigned long i = 0UL;
        while (i < n) { out[i] = (float)src[i]; i = i + 1UL; }
        return out;
    }
}

static void __tensor_gpu_from_f32(double* dst, const float* src, unsigned long n) {
    unsafe {
        unsigned long i = 0UL;
        while (i < n) { dst[i] = (double)src[i]; i = i + 1UL; }
    }
}

// ── Batched-dispatch support ──────────────────────────────────────────────────
// When mps_batch_is_active() (see gpu_mps.h's mps_batch_begin comment),
// an mps_*_f32 call returns immediately without waiting for the GPU or
// populating its 'fout' scratch buffer -- both happen later, inside
// mps_batch_end(). So the usual "convert fout to out->data, then free
// fout" step each op below does right after its mps_*_f32 call can't run
// yet either; it has to become a finalize callback registered via
// mps_batch_register_finalize instead, which mps_batch_end() runs once
// fout is actually populated. 'fout' itself must be malloc'd (not
// alloc/dealloc'd within the same call, like every non-batched path
// here) since it has to outlive this function returning -- the finalize
// callback below is what eventually frees it.
struct __MpsF32ToF64Ctx {
    float* fbuf;
    double* dbuf;
    unsigned long count;
};

static void __mps_f32_to_f64_finalize(void* ctxPtr) {
    unsafe {
        struct __MpsF32ToF64Ctx* ctx = (struct __MpsF32ToF64Ctx*)ctxPtr;
        unsigned long i = 0UL;
        while (i < ctx->count) { ctx->dbuf[i] = (double)ctx->fbuf[i]; i = i + 1UL; }
        free((void*)ctx->fbuf);
        free((void*)ctx);
    }
}

// ── Chain tracking ───────────────────────────────────────────────────────────
// Which Tensor*s currently hold a batched-but-not-yet-synced GPU result,
// and which GPU buffer that result actually lives in. tensor_matmul_gpu/
// tensor_relu_gpu consult this before uploading an input from CPU floats:
// if the input tensor is in this table, its real data is already sitting
// in a GPU buffer from an earlier op in the SAME open batch, so the
// mps_*_f32_chained entry points (gpu_mps.h) read that buffer directly
// instead of re-uploading the tensor's (not-yet-populated) CPU 'data' --
// which is the bug this table exists to prevent (see gpu_mps.h's
// mps_batch_begin comment for how it was found). Sized to match
// gpu_mps.sc's own MPS_BATCH_MAX_PENDING; entries are appended while a
// batch is open and never removed mid-batch (stale lookups just miss,
// which only costs a redundant upload, never wrong results) — the whole
// table is logically invalidated by mps_batch_end() returning, since
// nothing but tensor_matmul_gpu/tensor_relu_gpu ever reads it and both
// only do so while __mps_chain_count reflects the CURRENTLY open batch
// (reset at the top of both, whenever no batch is active).
#define MPS_CHAIN_MAX 64
static struct Tensor* __mps_chain_tensor[MPS_CHAIN_MAX];
static void*          __mps_chain_buf[MPS_CHAIN_MAX];
static unsigned long  __mps_chain_count = 0UL;

// The correct place to clear this table is tensor_gpu_batch_begin()
// (right when a NEW batch starts, before any op runs) — see that
// function and tensor_gpu.h's own comment on why "reset if idle" checked
// from inside an op is always one call too late.
void tensor_gpu_batch_begin() {
    __mps_chain_count = 0UL;
    mps_batch_begin();
}

void tensor_gpu_batch_end() {
    mps_batch_end();
}

static void __mps_chain_mark(struct Tensor* t, void* buf) {
    if (__mps_chain_count < (unsigned long)MPS_CHAIN_MAX) {
        __mps_chain_tensor[__mps_chain_count] = t;
        __mps_chain_buf[__mps_chain_count] = buf;
        __mps_chain_count = __mps_chain_count + 1UL;
    }
}

// Returns the pending GPU buffer for 't', or NULL if 't' isn't
// batched-pending (an ordinary, already-populated tensor — the common
// case, e.g. a persistent weight matrix).
static void* __mps_chain_lookup(struct Tensor* t) {
    unsigned long i = __mps_chain_count;
    // Walk backwards: if the same Tensor* were ever marked more than
    // once (not expected in practice — a fresh Tensor* per op), the most
    // recent entry is the correct one.
    while (i > 0UL) {
        i = i - 1UL;
        if (__mps_chain_tensor[i] == t) return __mps_chain_buf[i];
    }
    return (void*)0;
}

&Tensor tensor_add_gpu(const &Tensor a, const &Tensor b) {
    if (!mps_available()) { return tensor_add(a, b); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        float* fa = __tensor_gpu_to_f32(a->data, out->size);
        float* fb = __tensor_gpu_to_f32(b->data, out->size);
        float* fout = (float*)alloc(checked_mul_size(sizeof(float), out->size));
        int ok = mps_add_f32((const float*)fa, (const float*)fb, fout, out->size);
        if (ok) {
            __tensor_gpu_from_f32(out->data, (const float*)fout, out->size);
        } else {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] + b->data[i]; i = i + 1UL; }
        }
        dealloc((void*)fa); dealloc((void*)fb); dealloc((void*)fout);
    }
    __tensor_link2(out, a, b, (void*)__add_backward);
    return out;
}

&Tensor tensor_sub_gpu(const &Tensor a, const &Tensor b) {
    if (!mps_available()) { return tensor_sub(a, b); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        float* fa = __tensor_gpu_to_f32(a->data, out->size);
        float* fb = __tensor_gpu_to_f32(b->data, out->size);
        float* fout = (float*)alloc(checked_mul_size(sizeof(float), out->size));
        int ok = mps_sub_f32((const float*)fa, (const float*)fb, fout, out->size);
        if (ok) {
            __tensor_gpu_from_f32(out->data, (const float*)fout, out->size);
        } else {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] - b->data[i]; i = i + 1UL; }
        }
        dealloc((void*)fa); dealloc((void*)fb); dealloc((void*)fout);
    }
    __tensor_link2(out, a, b, (void*)__sub_backward);
    return out;
}

&Tensor tensor_mul_gpu(const &Tensor a, const &Tensor b) {
    if (!mps_available()) { return tensor_mul(a, b); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        float* fa = __tensor_gpu_to_f32(a->data, out->size);
        float* fb = __tensor_gpu_to_f32(b->data, out->size);
        float* fout = (float*)alloc(checked_mul_size(sizeof(float), out->size));
        int ok = mps_mul_f32((const float*)fa, (const float*)fb, fout, out->size);
        if (ok) {
            __tensor_gpu_from_f32(out->data, (const float*)fout, out->size);
        } else {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] * b->data[i]; i = i + 1UL; }
        }
        dealloc((void*)fa); dealloc((void*)fb); dealloc((void*)fout);
    }
    __tensor_link2(out, a, b, (void*)__mul_backward);
    return out;
}

&Tensor tensor_scale_gpu(const &Tensor a, double k) {
    if (!mps_available()) { return tensor_scale(a, k); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        float* fa = __tensor_gpu_to_f32(a->data, out->size);
        float* fout = (float*)alloc(checked_mul_size(sizeof(float), out->size));
        int ok = mps_scale_f32((const float*)fa, (float)k, fout, out->size);
        if (ok) {
            __tensor_gpu_from_f32(out->data, (const float*)fout, out->size);
        } else {
            unsigned long i = 0UL;
            while (i < out->size) { out->data[i] = a->data[i] * k; i = i + 1UL; }
        }
        dealloc((void*)fa); dealloc((void*)fout);
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
static void __relu_backward_gpu(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        if (!a->requiresGrad) return;
        __tensor_ensure_grad(a);
        if (!mps_available()) { __relu_backward(selfPtr); return; }

        float* fa = __tensor_gpu_to_f32(a->data, selfT->size);
        float* fgrad = __tensor_gpu_to_f32(selfT->grad, selfT->size);
        float* fout = (float*)alloc(checked_mul_size(sizeof(float), selfT->size));
        int ok = mps_relu_backward_f32((const float*)fa, (const float*)fgrad, fout, selfT->size);
        if (ok) {
            unsigned long i = 0UL;
            while (i < selfT->size) { a->grad[i] = a->grad[i] + (double)fout[i]; i = i + 1UL; }
        } else {
            // Device present but dispatch failed -- same rare fallback
            // shape as tensor_matmul_gpu's __tensor_matmul_cpu_into.
            unsigned long i = 0UL;
            while (i < selfT->size) {
                if (a->data[i] > 0.0) { a->grad[i] = a->grad[i] + selfT->grad[i]; }
                i = i + 1UL;
            }
        }
        dealloc((void*)fa); dealloc((void*)fgrad); dealloc((void*)fout);
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

        if (a->requiresGrad) {
            __tensor_ensure_grad(a);
            // dA[m,k] = dC[m,n] . B^T[n,k]
            float* fgrad = __tensor_gpu_to_f32(selfT->grad, m * n);
            float* fb = __tensor_gpu_to_f32(b->data, k * n);
            float* fdA = (float*)alloc(checked_mul_size(sizeof(float), m * k));
            int ok = mps_matmul_abt_f32((const float*)fgrad, (const float*)fb, fdA, m, n, k);
            if (ok) {
                unsigned long i = 0UL;
                while (i < m * k) { a->grad[i] = a->grad[i] + (double)fdA[i]; i = i + 1UL; }
            } else {
                unsigned long ii = 0UL;
                while (ii < m) {
                    unsigned long jj = 0UL;
                    while (jj < k) {
                        double acc = 0.0;
                        unsigned long p = 0UL;
                        while (p < n) { acc = acc + selfT->grad[ii * n + p] * b->data[jj * n + p]; p = p + 1UL; }
                        a->grad[ii * k + jj] = a->grad[ii * k + jj] + acc;
                        jj = jj + 1UL;
                    }
                    ii = ii + 1UL;
                }
            }
            dealloc((void*)fgrad); dealloc((void*)fb); dealloc((void*)fdA);
        }
        if (b->requiresGrad) {
            __tensor_ensure_grad(b);
            // dB[k,n] = A^T[k,m] . dC[m,n]
            float* fa = __tensor_gpu_to_f32(a->data, m * k);
            float* fgrad2 = __tensor_gpu_to_f32(selfT->grad, m * n);
            float* fdB = (float*)alloc(checked_mul_size(sizeof(float), k * n));
            int ok = mps_matmul_atb_f32((const float*)fa, (const float*)fgrad2, fdB, m, k, n);
            if (ok) {
                unsigned long i = 0UL;
                while (i < k * n) { b->grad[i] = b->grad[i] + (double)fdB[i]; i = i + 1UL; }
            } else {
                unsigned long p = 0UL;
                while (p < m) {
                    unsigned long ii = 0UL;
                    while (ii < k) {
                        double aVal = a->data[p * k + ii];
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
            dealloc((void*)fa); dealloc((void*)fgrad2); dealloc((void*)fdB);
        }
    }
}

&Tensor tensor_relu_gpu(const &Tensor a) {
    if (!mps_available()) { return tensor_relu(a); }
    struct Tensor* out = __tensor_alloc_uninit_like(a);
    unsafe {
        if (mps_batch_is_active()) {
            void* chainedBuf = __mps_chain_lookup((struct Tensor*)a);
            // Must outlive this call -- freed by __mps_f32_to_f64_finalize
            // once mps_batch_end() actually runs it (see that function's
            // comment above).
            float* fout = (float*)malloc(sizeof(float) * out->size);
            int ok;
            if (chainedBuf != (void*)0) {
                // 'a' is itself still-pending output from an earlier op
                // in this same batch -- read its real GPU buffer directly
                // instead of uploading a->data (not populated yet).
                ok = mps_relu_f32_chained(chainedBuf, fout, out->size);
            } else {
                float* fa = __tensor_gpu_to_f32(a->data, out->size);
                ok = mps_relu_f32((const float*)fa, fout, out->size);
                dealloc((void*)fa);
            }
            if (ok) {
                struct __MpsF32ToF64Ctx* ctx =
                    (struct __MpsF32ToF64Ctx*)malloc(sizeof(struct __MpsF32ToF64Ctx));
                ctx->fbuf = fout;
                ctx->dbuf = (double*)out->data;
                ctx->count = out->size;
                mps_batch_register_finalize((MpsFinalizeFn)__mps_f32_to_f64_finalize, (void*)ctx);
                __mps_chain_mark(out, mps_batch_last_output_buffer());
            } else if (chainedBuf == (void*)0) {
                // Dispatch failed even with a device present -- can't defer
                // a CPU fallback into a GPU batch's finalize step, so
                // compute it immediately instead (harmless: nothing
                // references 'out' via a pending finalize since none got
                // registered). Only possible on the non-chained path: 'a'
                // itself came from CPU data, so a CPU fallback can read it.
                free((void*)fout);
                unsigned long i = 0UL;
                while (i < out->size) {
                    double v = a->data[i];
                    out->data[i] = (v > 0.0) ? v : 0.0;
                    i = i + 1UL;
                }
            } else {
                // Chained dispatch failed and 'a's real data only exists
                // in a GPU buffer we have no CPU-side fallback for. This
                // is the one case this codebase's "always safe to call,
                // just not always actually on the GPU" GPU-op guarantee
                // (see tensor_gpu.h) can't uphold -- documented rather
                // than silently wrong, since it would need synchronously
                // flushing mid-batch to recover a CPU value at all.
                free((void*)fout);
            }
        } else {
            float* fa = __tensor_gpu_to_f32(a->data, out->size);
            float* fout = (float*)alloc(checked_mul_size(sizeof(float), out->size));
            int ok = mps_relu_f32((const float*)fa, fout, out->size);
            if (ok) {
                __tensor_gpu_from_f32(out->data, (const float*)fout, out->size);
            } else {
                unsigned long i = 0UL;
                while (i < out->size) {
                    double v = a->data[i];
                    out->data[i] = (v > 0.0) ? v : 0.0;
                    i = i + 1UL;
                }
            }
            dealloc((void*)fa); dealloc((void*)fout);
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
        float* fa = __tensor_gpu_to_f32(a->data, a->size);
        float fout[1];
        int ok = mps_sum_f32((const float*)fa, (float*)fout, a->size);
        if (ok) {
            out->data[0] = (double)fout[0];
        } else {
            double acc = 0.0;
            unsigned long i = 0UL;
            while (i < a->size) { acc = acc + a->data[i]; i = i + 1UL; }
            out->data[0] = acc;
        }
        dealloc((void*)fa);
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
            while (j0 < n) { out->data[i * n + j0] = 0.0; j0 = j0 + 1UL; }
            i = i + 1UL;
        }
        i = 0UL;
        while (i < m) {
            unsigned long p = 0UL;
            while (p < k) {
                double aVal = a->data[i * k + p];
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
            void* chainedBuf = __mps_chain_lookup((struct Tensor*)a);
            float* fb = (float*)alloc(checked_mul_size(sizeof(float), k * n));
            unsigned long z = 0UL;
            while (z < k * n) { fb[z] = (float)b->data[z]; z = z + 1UL; }
            // Must outlive this call -- freed by __mps_f32_to_f64_finalize
            // once mps_batch_end() actually runs it.
            float* fout = (float*)malloc(sizeof(float) * m * n);
            int ok;
            if (chainedBuf != (void*)0) {
                ok = mps_matmul_f32_chained(chainedBuf, (const float*)fb, fout, m, k, n);
            } else {
                float* fa = (float*)alloc(checked_mul_size(sizeof(float), m * k));
                z = 0UL;
                while (z < m * k) { fa[z] = (float)a->data[z]; z = z + 1UL; }
                ok = mps_matmul_f32((const float*)fa, (const float*)fb, fout, m, k, n);
                dealloc((void*)fa);
            }
            dealloc((void*)fb);
            if (ok) {
                struct __MpsF32ToF64Ctx* ctx =
                    (struct __MpsF32ToF64Ctx*)malloc(sizeof(struct __MpsF32ToF64Ctx));
                ctx->fbuf = fout;
                ctx->dbuf = (double*)out->data;
                ctx->count = m * n;
                mps_batch_register_finalize((MpsFinalizeFn)__mps_f32_to_f64_finalize, (void*)ctx);
                __mps_chain_mark(out, mps_batch_last_output_buffer());
            } else if (chainedBuf == (void*)0) {
                free((void*)fout);
                __tensor_matmul_cpu_into(out, a, b, m, k, n);
            } else {
                // Same undocumented-fallback-impossible case as
                // tensor_relu_gpu's chained path -- see its comment.
                free((void*)fout);
            }
        } else {
            float* fa = (float*)alloc(checked_mul_size(sizeof(float), m * k));
            float* fb = (float*)alloc(checked_mul_size(sizeof(float), k * n));
            unsigned long z = 0UL;
            while (z < m * k) { fa[z] = (float)a->data[z]; z = z + 1UL; }
            z = 0UL;
            while (z < k * n) { fb[z] = (float)b->data[z]; z = z + 1UL; }
            float* fout = (float*)alloc(checked_mul_size(sizeof(float), m * n));
            int ok = mps_matmul_f32((const float*)fa, (const float*)fb, fout, m, k, n);
            if (ok) {
                z = 0UL;
                while (z < m * n) { out->data[z] = (double)fout[z]; z = z + 1UL; }
            } else {
                __tensor_matmul_cpu_into(out, a, b, m, k, n);
            }
            dealloc((void*)fa); dealloc((void*)fb); dealloc((void*)fout);
        }
    }
    __tensor_link2(out, a, b, (void*)__matmul_backward_gpu);
    return out;
}

} // namespace std
