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

// ── Backward (BLAS-accelerated) ───────────────────────────────────────────────
// Same math as tensor.h's __matmul_backward (dA = dC . B^T, dB = A^T . dC)
// but each is a single cblas_dgemm call with a transpose flag instead of
// a hand-rolled loop — BLAS takes the transpose directly (CblasTrans),
// so there's no need to physically materialize B^T/A^T into a scratch
// buffer first.
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
            double* dA = (double*)malloc(sizeof(double) * m * k);
            cblas_dgemm(101, 111, 112, (int)m, (int)k, (int)n,
                1.0, (const double*)selfT->grad, (int)n,
                (const double*)b->data, (int)n,
                0.0, dA, (int)k);
            unsigned long i = 0UL;
            while (i < m * k) { a->grad[i] = a->grad[i] + dA[i]; i = i + 1UL; }
            free((void*)dA);
        }
        if (b->requiresGrad) {
            __tensor_ensure_grad(b);
            // dB[k,n] = A^T[k,m] . dC[m,n]
            double* dB = (double*)malloc(sizeof(double) * k * n);
            cblas_dgemm(101, 112, 111, (int)k, (int)n, (int)m,
                1.0, (const double*)a->data, (int)k,
                (const double*)selfT->grad, (int)n,
                0.0, dB, (int)n);
            unsigned long i = 0UL;
            while (i < k * n) { b->grad[i] = b->grad[i] + dB[i]; i = i + 1UL; }
            free((void*)dB);
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
