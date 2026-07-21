// SafeC Standard Library — BLAS-accelerated Tensor matmul implementation
// (see tensor_blas.h).
#pragma once
#include <std/ml/tensor_blas.h>

namespace std {

extern void* malloc(unsigned long size);
extern void  free(void* ptr);

// Standard CBLAS C ABI (Accelerate, OpenBLAS, and MKL's cblas_* entry
// points all share this exact signature) — declared directly rather than
// through a vendor header since SafeC has no C-header-interop story yet
// (same convention as gpu_mps.sc declaring objc_msgSend itself, or
// gpu_spirv.sc declaring the Vulkan entry points itself).
// order: 101=RowMajor, 102=ColMajor. trans: 111=NoTrans, 112=Trans.
extern void cblas_sgemm(int order, int transA, int transB,
                         int M, int N, int K,
                         float alpha, const float* A, int lda,
                         const float* B, int ldb,
                         float beta, float* C, int ldc);

// vDSP (also Accelerate.framework, alongside cblas_sgemm above): a
// vectorized float32 elementwise add, C[i] = A[i] + B[i] for i in [0,N),
// strides IA/IB/IC (1 for contiguous, this file's only use). Verified
// directly (not assumed) that C may alias B for an in-place accumulate
// (C = A + C) before relying on that below (same check as tensor_f32's
// predecessor did for the double-precision vDSP_vaddD).
extern void vDSP_vadd(const float* A, long IA, const float* B, long IB,
                       float* C, long IC, unsigned long N);

// ── Backward (BLAS-accelerated) ───────────────────────────────────────────────
// Same math as tensor.h's __matmul_backward (dA = dC . B^T, dB = A^T . dC):
// each a single cblas_sgemm call with a transpose flag instead of a hand-
// rolled loop — BLAS takes the transpose directly (CblasTrans), no need
// to physically materialize B^T/A^T first.
//
// Profiling a real training step found backward taking ~50% of total
// train time. Two things tried against that, each measured directly
// (controlled, alternating comparisons) rather than assumed:
//   - cblas_sgemm with beta=1.0 straight into a->grad/b->grad (BLAS
//     natively computes C = alpha*A*B + beta*C, which would skip the
//     scratch buffer and accumulate step entirely) — consistently
//     *slower* (measured on the double-precision predecessor of this
//     file; Accelerate's gemm apparently has a less-optimized path for
//     beta!=0, a real read-modify-write accumulate, than for the far-
//     more-common beta=0 pure-write case). Reverted.
//   - Replacing malloc+free-every-call with a grow-on-demand scratch
//     buffer reused across calls, and the manual accumulate loop with
//     vDSP_vadd (vectorized, same Accelerate framework as cblas_sgemm):
//     a real win, kept below.
// __blas_scratch_ is safe to share across both branches and across calls:
// dA is fully consumed (accumulated into a->grad) before dB's computation
// begins, and tensor_backward()'s toposort walk finishes one node's
// backward function before the next, so no two live users ever overlap.
static float*         __blas_scratch_ = (float*)0;
static unsigned long __blas_scratch_cap_ = 0UL;

static float* __blas_scratch_get_(unsigned long n) {
    unsafe {
        if (n > __blas_scratch_cap_) {
            if (__blas_scratch_ != (float*)0) { free((void*)__blas_scratch_); }
            __blas_scratch_ = (float*)malloc(sizeof(float) * n);
            __blas_scratch_cap_ = n;
        }
        return __blas_scratch_;
    }
}

static void __matmul_backward_blas(void* selfPtr) {
    unsafe {
        struct Tensor* selfT = (struct Tensor*)selfPtr;
        struct Tensor* a = __tensor_parent(selfT, 0UL);
        struct Tensor* b = __tensor_parent(selfT, 1UL);
        unsigned long m = a->shape[0];
        unsigned long k = a->shape[1];
        unsigned long n = b->shape[1];

        if (a->requiresGrad) {
            __tensor_ensure_grad(a);
            // dA[m,k] = dC[m,n] . B^T[n,k]
            float* dA = __blas_scratch_get_(m * k);
            cblas_sgemm(101, 111, 112, (int)m, (int)k, (int)n,
                (float)1.0, (const float*)selfT->grad, (int)n,
                (const float*)b->data, (int)n,
                (float)0.0, dA, (int)k);
            vDSP_vadd((const float*)dA, 1L, (const float*)a->grad, 1L, (float*)a->grad, 1L, m * k);
        }
        if (b->requiresGrad) {
            __tensor_ensure_grad(b);
            // dB[k,n] = A^T[k,m] . dC[m,n]
            float* dB = __blas_scratch_get_(k * n);
            cblas_sgemm(101, 112, 111, (int)k, (int)n, (int)m,
                (float)1.0, (const float*)a->data, (int)k,
                (const float*)selfT->grad, (int)n,
                (float)0.0, dB, (int)n);
            vDSP_vadd((const float*)dB, 1L, (const float*)b->grad, 1L, (float*)b->grad, 1L, k * n);
        }
    }
}

// ── Forward ──────────────────────────────────────────────────────────────────
&Tensor tensor_matmul_blas(const &Tensor a, const &Tensor b) {
    unsigned long m; unsigned long k; unsigned long n;
    unsafe { m = a->shape[0]; k = a->shape[1]; n = b->shape[1]; }
    struct Tensor* out = tensor_new_2d(m, n, 0);
    unsafe {
        cblas_sgemm(101, 111, 111, (int)m, (int)n, (int)k,
            (float)1.0, (const float*)a->data, (int)k,
            (const float*)b->data, (int)n,
            (float)0.0, (float*)out->data, (int)n);
    }
    __tensor_link2(out, a, b, (void*)__matmul_backward_blas);
    return out;
}

} // namespace std
