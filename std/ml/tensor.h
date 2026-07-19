#pragma once
// SafeC Standard Library — Tensor: CPU tensors with reverse-mode autograd.
//
// PyTorch-style ergonomics (build a graph implicitly by calling ops;
// tensor_backward() walks it) over a plain flat f64 buffer — no unified-
// memory story here (that's MPS's job; see gpu_mps.h), just row-major
// double-precision arrays with a recorded computation graph. Scope: 1D
// and 2D tensors (vectors and matrices — everything a small MLP/attention
// block needs), add/sub/mul/relu/sum/matmul, and reverse-mode autodiff
// over that op set. Not a general n-dimensional array library — no
// broadcasting beyond 2D, no int/complex dtypes, no in-place ops.
#include <std/collections/vec.h>

namespace std {

typedef fn void(void* selfTensor) TensorBackwardFn;

struct Tensor {
    &heap double  data;       // flat row-major buffer, length == size
    &heap unsigned long shape; // shape[0..ndim)
    unsigned long ndim;
    unsigned long size;        // product of shape
    int           requiresGrad;
    &heap double  grad;        // same length as data; only allocated if requiresGrad
    // Autograd graph edges. 'gradFn' is void* (not TensorBackwardFn)
    // purely to dodge the typedef-ordering constraint every other
    // callback-carrying struct in std/ hits (see std/gui/gui_widget.h's
    // header comment for the exact same workaround) — cast at the handful
    // of call sites that read/write it.
    void*         gradFn;
    struct Vec    parents;     // Vec<struct Tensor*> — inputs this tensor was computed from
    int           visited;     // topological-sort scratch flag, used and cleared by tensor_backward()
    double        extraScalar; // op-specific constant (e.g. tensor_scale's k) the gradFn needs

    // Element access (row-major; 'idx1D' for a rank-1 tensor,
    // 'row,col' for a rank-2 one).
    double  at1(unsigned long i) const;
    double  at2(unsigned long r, unsigned long c) const;
    void    set1(unsigned long i, double v);
    void    set2(unsigned long r, unsigned long c, double v);

    void    free();
};

// Every function below takes/returns 'Tensor' by a region-less reference
// ('&Tensor'/'const &Tensor' — see README's "Outliving references"
// section): a Tensor is never null anywhere in this library, and nothing
// here cares whether the caller's tensor is heap/stack/static/arena-
// backed, so pinning one specific region would only narrow who can call
// these for no safety benefit. Internally, tensors are still allocated
// with a plain 'malloc' (see tensor.sc's '__tensor_alloc') and freed
// with 'free()' via 'Tensor::free()' below — 'free()' takes no argument
// beyond 'self' precisely because there's no separate heap-region
// bookkeeping to release; call it, then let the reference go out of
// scope.

// ── Construction ─────────────────────────────────────────────────────────────
&Tensor tensor_new_1d(unsigned long n, int requiresGrad);
&Tensor tensor_new_2d(unsigned long rows, unsigned long cols, int requiresGrad);
// Copies 'values' (length must equal the tensor's size) into a freshly
// allocated tensor of the given shape.
&Tensor tensor_from_1d(const double* values, unsigned long n, int requiresGrad);
&Tensor tensor_from_2d(const double* values, unsigned long rows, unsigned long cols, int requiresGrad);
&Tensor tensor_zeros_like(const &Tensor t);
&Tensor tensor_fill(&Tensor t, double v);

// ── Forward ops (each records a backward edge when either operand
// requires grad) ────────────────────────────────────────────────────────────
&Tensor tensor_add(const &Tensor a, const &Tensor b);
&Tensor tensor_sub(const &Tensor a, const &Tensor b);
&Tensor tensor_mul(const &Tensor a, const &Tensor b);      // elementwise
&Tensor tensor_scale(const &Tensor a, double k);            // a * scalar k
&Tensor tensor_matmul(const &Tensor a, const &Tensor b);   // 2D x 2D
&Tensor tensor_relu(const &Tensor a);
&Tensor tensor_sum(const &Tensor a);                        // -> scalar (1-element) tensor

// ── Autograd ─────────────────────────────────────────────────────────────────
// 't' must be a scalar (size == 1, e.g. a loss). Seeds t->grad = 1, walks
// the graph in reverse topological order, and accumulates gradients into
// every requiresGrad ancestor's 'grad' buffer (allocating it on first
// write). Safe to call multiple times — gradients accumulate across
// calls, matching PyTorch's default (call tensor_zero_grad() between
// optimizer steps if that's not what you want).
void tensor_backward(&Tensor t);
void tensor_zero_grad(&Tensor t);

} // namespace std
