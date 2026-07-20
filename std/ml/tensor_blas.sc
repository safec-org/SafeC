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
extern void cblas_dgemm(int order, int transA, int transB,
                         int M, int N, int K,
                         double alpha, const double* A, int lda,
                         const double* B, int ldb,
                         double beta, double* C, int ldc);

// vDSP (also Accelerate.framework, alongside cblas_dgemm above): a
// vectorized double-precision elementwise add, C[i] = A[i] + B[i] for
// i in [0,N), strides IA/IB/IC (1 for contiguous, this file's only use).
// Verified directly (not assumed) that C may alias B for an in-place
// accumulate (C = A + C) before relying on that below.
extern void vDSP_vaddD(const double* A, long IA, const double* B, long IB,
                        double* C, long IC, unsigned long N);

// ── Backward (BLAS-accelerated) ───────────────────────────────────────────────
// Same math as tensor.h's __matmul_backward (dA = dC . B^T, dB = A^T . dC):
// each a single cblas_dgemm call with a transpose flag instead of a hand-
// rolled loop — BLAS takes the transpose directly (CblasTrans), no need
// to physically materialize B^T/A^T first.
//
// Profiling a real training step found backward taking ~50% of total
// train time. Two things tried against that, each measured directly
// (controlled, alternating comparisons) rather than assumed:
//   - cblas_dgemm with beta=1.0 straight into a->grad/b->grad (BLAS
//     natively computes C = alpha*A*B + beta*C, which would skip the
//     scratch buffer and accumulate step entirely) — consistently
//     *slower* (4/4 alternating runs), not faster. Reverted; Accelerate's
//     dgemm apparently has a less-optimized path for beta!=0 (a real
//     read-modify-write accumulate) than for the far-more-common beta=0
//     pure-write case.
//   - Replacing malloc+free-every-call with a grow-on-demand scratch
//     buffer reused across calls: real but small (~1-2%, close to noise).
//     The manual accumulate loop turned out to matter more than the
//     allocation — replaced with vDSP_vaddD (vectorized, same Accelerate
//     framework as cblas_dgemm) below: backward dropped ~119-124ms to
//     ~106-113ms per 50 steps.
// __blas_scratch_ is safe to share across both branches and across calls:
// dA is fully consumed (accumulated into a->grad) before dB's computation
// begins, and tensor_backward()'s toposort walk finishes one node's
// backward function before the next, so no two live users ever overlap.
static double*       __blas_scratch_ = (double*)0;
static unsigned long __blas_scratch_cap_ = 0UL;

static double* __blas_scratch_get_(unsigned long n) {
    unsafe {
        if (n > __blas_scratch_cap_) {
            if (__blas_scratch_ != (double*)0) { free((void*)__blas_scratch_); }
            __blas_scratch_ = (double*)malloc(sizeof(double) * n);
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
            double* dA = __blas_scratch_get_(m * k);
            cblas_dgemm(101, 111, 112, (int)m, (int)k, (int)n,
                1.0, (const double*)selfT->grad, (int)n,
                (const double*)b->data, (int)n,
                0.0, dA, (int)k);
            vDSP_vaddD((const double*)dA, 1L, (const double*)a->grad, 1L, (double*)a->grad, 1L, m * k);
        }
        if (b->requiresGrad) {
            __tensor_ensure_grad(b);
            // dB[k,n] = A^T[k,m] . dC[m,n]
            double* dB = __blas_scratch_get_(k * n);
            cblas_dgemm(101, 112, 111, (int)k, (int)n, (int)m,
                1.0, (const double*)a->data, (int)k,
                (const double*)selfT->grad, (int)n,
                0.0, dB, (int)n);
            vDSP_vaddD((const double*)dB, 1L, (const double*)b->grad, 1L, (double*)b->grad, 1L, k * n);
        }
    }
}

// ── Forward ──────────────────────────────────────────────────────────────────
&Tensor tensor_matmul_blas(const &Tensor a, const &Tensor b) {
    unsigned long m; unsigned long k; unsigned long n;
    unsafe { m = a->shape[0]; k = a->shape[1]; n = b->shape[1]; }
    struct Tensor* out = tensor_new_2d(m, n, 0);
    unsafe {
        cblas_dgemm(101, 111, 111, (int)m, (int)n, (int)k,
            1.0, (const double*)a->data, (int)k,
            (const double*)b->data, (int)n,
            0.0, (double*)out->data, (int)n);
    }
    __tensor_link2(out, a, b, (void*)__matmul_backward_blas);
    return out;
}

} // namespace std
