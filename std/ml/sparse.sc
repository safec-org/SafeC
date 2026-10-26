// SafeC Standard Library — sparse tensors (see sparse.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/sparse.h>
#include <std/ml/tensor.h>
#include <std/ml/device.h>
#include <std/ml/device_kernels.h>
#include <std/panic.h>
#include <std/ml/nn.h>

namespace std {

static ?&TensorArena sp_heap_begin_() {
    ?&TensorArena cur = tensor_arena_current_();
    tensor_arena_end();
    return cur;
}

static void sp_heap_end_(?&TensorArena cur) {
    match (cur) {
        case some(a): { a.begin(); }
        case null: { }
    }
}

static void sp_drop_(&FloatTensor t) {
    t.free();
    unsafe { match (t->owner) { case some(a): { } case null: { free((void*)t); } } }
}

// The pattern arrays as tensors (on the heap: they outlive any step).
static struct SparseTensor sp_alloc_(unsigned long rows, unsigned long cols, unsigned long nnz, int requiresGrad) {
    ?&TensorArena cur = sp_heap_begin_();
    struct SparseTensor s;
    s.rows = rows;
    s.cols = cols;
    s.nnz = nnz;
    s.ownsPattern = 1;
    s.values = FloatTensor::new_2d(nnz == 0UL ? 1UL : nnz, 1UL, requiresGrad);   // [nnz, 1]: a gather table
    s.colIdx = FloatTensor::new_1d(nnz == 0UL ? 1UL : nnz, 0);
    s.rowPtr = FloatTensor::new_1d(rows + 1UL, 0);
    s.rowOf = FloatTensor::new_1d(nnz == 0UL ? 1UL : nnz, 0);
    sp_heap_end_(cur);
    return s;
}

static void sp_fill_rows_(&SparseTensor s) {
    unsafe {
        unsigned long r = 0UL;
        while (r < s.rows) {
            unsigned long e = (unsigned long)s.rowPtr->data[r];
            unsigned long hi = (unsigned long)s.rowPtr->data[r + 1UL];
            while (e < hi) { s.rowOf->data[e] = (float)r; e = e + 1UL; }
            r = r + 1UL;
        }
    }
}

struct SparseTensor SparseTensor::new_csr(unsigned long rows, unsigned long cols, []const unsigned long rowPtr_s,
                                          []const unsigned long colIdx_s, []const float values_s, int requiresGrad) {
    if (rowPtr_s.len < rows + 1UL) { panic_at("SparseTensor::new_csr: rowPtr needs rows + 1 entries", "sparse.sc", __LINE__); }
    unsigned long nnz = rowPtr_s[rows];
    if (colIdx_s.len < nnz || values_s.len < nnz) { panic_at("SparseTensor::new_csr: colIdx / values shorter than rowPtr[rows]", "sparse.sc", __LINE__); }
    const &unsigned long rowPtr;
    unsafe { rowPtr = (const &unsigned long)rowPtr_s.ptr; }
    unsigned long colIdx_d_[1];
    const &unsigned long colIdx = &colIdx_d_[0];
    if (nnz > 0UL) { unsafe { colIdx = (const &unsigned long)colIdx_s.ptr; } }
    float values_d_[1];
    const &float values = &values_d_[0];
    if (nnz > 0UL) { unsafe { values = (const &float)values_s.ptr; } }
    struct SparseTensor s = sp_alloc_(rows, cols, nnz, requiresGrad);
    unsafe {
        unsigned long r = 0UL;
        while (r <= rows) { s.rowPtr->data[r] = (float)rowPtr[r]; r = r + 1UL; }
        unsigned long e = 0UL;
        while (e < nnz) {
            if (colIdx[e] >= cols) panic_at("SparseTensor::new_csr: a column index is out of range", "sparse.sc", __LINE__);
            s.colIdx->data[e] = (float)colIdx[e];
            s.values->data[e] = values[e];
            e = e + 1UL;
        }
    }
    sp_fill_rows_(&s);
    return s;
}

struct SparseTensor SparseTensor::from_dense(const &FloatTensor dense, float threshold, int requiresGrad) {
    unsafe { tensor_host_raw_((void*)dense); }
    unsigned long rows; unsigned long cols;
    unsafe { rows = dense->shape[0]; cols = dense->size / rows; }
    unsigned long nnz = 0UL;
    unsafe {
        unsigned long i = 0UL;
        while (i < dense->size) {
            float v = dense->data[i];
            if (v > threshold || v < -threshold) nnz = nnz + 1UL;
            i = i + 1UL;
        }
    }
    struct SparseTensor s = sp_alloc_(rows, cols, nnz, requiresGrad);
    unsafe {
        unsigned long e = 0UL;
        unsigned long r = 0UL;
        while (r < rows) {
            s.rowPtr->data[r] = (float)e;
            unsigned long c = 0UL;
            while (c < cols) {
                float v = dense->data[r * cols + c];
                if (v > threshold || v < -threshold) {
                    s.colIdx->data[e] = (float)c;
                    s.values->data[e] = v;
                    e = e + 1UL;
                }
                c = c + 1UL;
            }
            r = r + 1UL;
        }
        s.rowPtr->data[rows] = (float)e;
    }
    sp_fill_rows_(&s);
    return s;
}

struct SparseTensor SparseTensor::identity(unsigned long n, int requiresGrad) {
    struct SparseTensor s = sp_alloc_(n, n, n, requiresGrad);
    unsafe {
        unsigned long i = 0UL;
        while (i < n) {
            s.rowPtr->data[i] = (float)i;
            s.colIdx->data[i] = (float)i;
            s.values->data[i] = (float)1.0;
            s.rowOf->data[i] = (float)i;
            i = i + 1UL;
        }
        s.rowPtr->data[n] = (float)n;
    }
    return s;
}

// The storage position of (r, c), or nnz when it isn't stored.
static unsigned long sp_find_(const &SparseTensor s, unsigned long r, unsigned long c) {
    unsafe {
        tensor_host_raw_((void*)s.rowPtr);
        tensor_host_raw_((void*)s.colIdx);
        if (r >= s.rows) return s.nnz;
        unsigned long e = (unsigned long)s.rowPtr->data[r];
        unsigned long hi = (unsigned long)s.rowPtr->data[r + 1UL];
        while (e < hi) {
            if ((unsigned long)s.colIdx->data[e] == c) return e;
            e = e + 1UL;
        }
    }
    return s.nnz;
}

float SparseTensor::at2(unsigned long r, unsigned long c) const {
    unsigned long e = sp_find_(self, r, c);
    if (e >= self.nnz) return (float)0.0;
    return self.values.at1(e);
}

float SparseTensor::at1(unsigned long i) const { return self.at2(i / self.cols, i % self.cols); }

void SparseTensor::set2(unsigned long r, unsigned long c, float v) {
    unsigned long e = sp_find_(self, r, c);
    if (e < self.nnz) self.values.set1(e, v);
}

void SparseTensor::set1(unsigned long i, float v) { self.set2(i / self.cols, i % self.cols, v); }

void SparseTensor::fill(float v) { self.values.fill(v); }

void SparseTensor::backward() { self.values.backward(); }

void SparseTensor::zero_grad() { self.values.zero_grad(); }

void SparseTensor::free() {
    sp_drop_(self.values);
    if (!self.ownsPattern) return;
    sp_drop_(self.colIdx);
    sp_drop_(self.rowPtr);
    sp_drop_(self.rowOf);
}

int SparseTensor::device() const { return self.values.device(); }

int SparseTensor::place(int device) {
    return self.values.place(device) && self.colIdx.place(device) && self.rowPtr.place(device) &&
           self.rowOf.place(device);
}

int SparseTensor::sync() { return self.values.sync(); }

unsigned long SparseTensor::device_buffer() const { return self.values.device_buffer(); }

// ── Sparse x dense ───────────────────────────────────────────────────────────
// Parents: values, X, then the pattern (colIdx, rowPtr, rowOf), which never
// needs a gradient.
static void sp_spmm_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* out = (FloatTensor*)selfPtr;
        FloatTensor* vals = __tensor_parent(out, 0UL);
        FloatTensor* x = __tensor_parent(out, 1UL);
        FloatTensor* col = __tensor_parent(out, 2UL);
        FloatTensor* rowOf = __tensor_parent(out, 4UL);
        unsigned long n = out->shape[1];
        unsigned long nnz = (unsigned long)out->extraScalar;
        int d = out->dev;
        if ((out->coh & TS_DEVOP) != 0) {
            if (!tensor_dev_raw_((void*)vals, d) || !tensor_dev_raw_((void*)x, d) || !tensor_dev_raw_((void*)col, d) ||
                !tensor_dev_raw_((void*)rowOf, d)) return;
            if (x->requiresGrad && tensor_grad_dev_raw_((void*)x, d)) {
                struct DevArgs a = dev_args_();
                unsigned long tgt = tensor_grad_atomic_begin_raw_((void*)x, d);
                a.p[1] = col->dbuf; a.p[2] = vals->dbuf; a.p[3] = out->dgrad; a.p[4] = tgt; a.p[5] = rowOf->dbuf;
                a.u[0] = (unsigned int)n;
                if (tgt != 0UL) device_launch_(d, DK_SPMM_TGRAD, nnz * n, &a);
                tensor_grad_atomic_end_raw_((void*)x, d, tgt);
                tensor_grad_dev_wrote_raw_((void*)x);
            }
            if (vals->requiresGrad && tensor_grad_dev_raw_((void*)vals, d)) {
                struct DevArgs a = dev_args_();
                a.p[1] = col->dbuf; a.p[3] = x->dbuf; a.p[4] = out->dgrad; a.p[5] = rowOf->dbuf; a.p[6] = vals->dgrad;
                a.u[0] = (unsigned int)n;
                device_launch_(d, DK_SPMM_VGRAD, nnz, &a);
                tensor_grad_dev_wrote_raw_((void*)vals);
            }
            return;
        }
        tensor_host_raw_((void*)vals); tensor_host_raw_((void*)x); tensor_host_raw_((void*)col); tensor_host_raw_((void*)rowOf);
        if (x->requiresGrad) {
            __tensor_ensure_grad(x);
            unsigned long e = 0UL;
            while (e < nnz) {
                unsigned long r = (unsigned long)rowOf->data[e];
                unsigned long c = (unsigned long)col->data[e];
                float v = vals->data[e];
                unsigned long j = 0UL;
                while (j < n) { x->grad[c * n + j] = x->grad[c * n + j] + v * out->grad[r * n + j]; j = j + 1UL; }
                e = e + 1UL;
            }
        }
        if (vals->requiresGrad) {
            __tensor_ensure_grad(vals);
            unsigned long e = 0UL;
            while (e < nnz) {
                unsigned long r = (unsigned long)rowOf->data[e];
                unsigned long c = (unsigned long)col->data[e];
                float s = (float)0.0;
                unsigned long j = 0UL;
                while (j < n) { s = s + out->grad[r * n + j] * x->data[c * n + j]; j = j + 1UL; }
                vals->grad[e] = vals->grad[e] + s;
                e = e + 1UL;
            }
        }
    }
}

static void sp_link_(FloatTensor* out, const &SparseTensor s, FloatTensor* x) {
    unsafe {
        if (!Autograd::get().enabled() || !(s.values->requiresGrad || x->requiresGrad)) { out->requiresGrad = 0; return; }
        out->requiresGrad = 1;
        out->parents.push(s.values);
        out->parents.push(x);
        out->parents.push(s.colIdx);
        out->parents.push(s.rowPtr);
        out->parents.push(s.rowOf);
        out->gradFn = (void*)sp_spmm_backward_;
    }
}

&FloatTensor tensor_spmm(const &SparseTensor s, const &FloatTensor x) {
    unsigned long n;
    unsafe {
        if (x->shape[0] != s.cols) panic_at("tensor_spmm: X is not [cols, n]", "sparse.sc", __LINE__);
        n = x->size / s.cols;
    }
    int d;
    unsafe { d = tensor_op_device_raw_((void*)s.values, (void*)x, (void*)0); }
    if (d != DEVICE_CPU) {
        unsafe {
            if (tensor_dev_raw_((void*)s.values, d) && tensor_dev_raw_((void*)x, d) && tensor_dev_raw_((void*)s.colIdx, d) &&
                tensor_dev_raw_((void*)s.rowPtr, d)) {
                unsigned long shape[2];
                shape[0] = s.rows; shape[1] = n;
                match (tensor_dev_new_t_((const &unsigned long)&shape[0], 2UL, d, dev_tag_(x->dbuf) != 0UL ? device_store_tag_(d) : DEVICE_FP32)) {
                    case some(o): {
                        struct DevArgs a = dev_args_();
                        a.p[0] = s.rowPtr->dbuf; a.p[1] = s.colIdx->dbuf; a.p[2] = s.values->dbuf; a.p[3] = x->dbuf;
                        a.p[4] = o->dbuf; a.u[0] = (unsigned int)n; a.u[1] = 0U;
                        if (device_launch_(d, DK_SPMM, s.rows * n, &a)) {
                            o->extraScalar = (float)s.nnz;
                            sp_link_((FloatTensor*)o, s, (FloatTensor*)x);
                            return o;
                        }
                        tensor_free_device_raw_((void*)o);
                    }
                    case null: { }
                }
            }
        }
    }
    unsafe {
        tensor_host_raw_((void*)s.values); tensor_host_raw_((void*)x); tensor_host_raw_((void*)s.colIdx); tensor_host_raw_((void*)s.rowPtr);
    }
    unsigned long shape[2];
    unsafe { shape[0] = s.rows; shape[1] = n; }
    &FloatTensor out = __tensor_alloc(shape, 2UL, 0);
    unsafe {
        unsigned long r = 0UL;
        while (r < s.rows) {
            unsigned long lo = (unsigned long)s.rowPtr->data[r];
            unsigned long hi = (unsigned long)s.rowPtr->data[r + 1UL];
            unsigned long j = 0UL;
            while (j < n) {
                float acc = (float)0.0;
                unsigned long e = lo;
                while (e < hi) { acc = acc + s.values->data[e] * x->data[(unsigned long)s.colIdx->data[e] * n + j]; e = e + 1UL; }
                out->data[r * n + j] = acc;
                j = j + 1UL;
            }
            r = r + 1UL;
        }
        out->extraScalar = (float)s.nnz;
    }
    unsafe { sp_link_((FloatTensor*)out, s, (FloatTensor*)x); }
    return out;
}

// to_dense: spmm with the identity on the right would cost cols^2; instead a
// direct scatter, its backward a gather (values grad = out grad at each nonzero).
static void sp_dense_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* out = (FloatTensor*)selfPtr;
        FloatTensor* vals = __tensor_parent(out, 0UL);
        FloatTensor* col = __tensor_parent(out, 1UL);
        FloatTensor* rowOf = __tensor_parent(out, 2UL);
        if (!vals->requiresGrad) return;
        tensor_grad_host_raw_((void*)out);
        tensor_host_raw_((void*)col); tensor_host_raw_((void*)rowOf);
        __tensor_ensure_grad(vals);
        unsigned long cols = out->shape[1];
        unsigned long e = 0UL;
        while (e < vals->size) {
            unsigned long k = (unsigned long)rowOf->data[e] * cols + (unsigned long)col->data[e];
            vals->grad[e] = vals->grad[e] + out->grad[k];
            e = e + 1UL;
        }
    }
}

&FloatTensor SparseTensor::to_dense() const {
    unsafe { tensor_host_raw_((void*)self.values); tensor_host_raw_((void*)self.colIdx); tensor_host_raw_((void*)self.rowOf); }
    &FloatTensor out = FloatTensor::new_2d(self.rows, self.cols, 0);
    unsafe {
        unsigned long e = 0UL;
        while (e < self.nnz) {
            out->data[(unsigned long)self.rowOf->data[e] * self.cols + (unsigned long)self.colIdx->data[e]] = self.values->data[e];
            e = e + 1UL;
        }
        if (Autograd::get().enabled() && self.values->requiresGrad && self.nnz > 0UL) {
            out->requiresGrad = 1;
            out->parents.push(self.values);
            out->parents.push(self.colIdx);
            out->parents.push(self.rowOf);
            out->gradFn = (void*)sp_dense_backward_;
        }
    }
    return out;
}

struct SparseTensor SparseTensor::scale(float k) const {
    struct SparseTensor s;
    s.rows = self.rows; s.cols = self.cols; s.nnz = self.nnz;
    s.colIdx = self.colIdx; s.rowPtr = self.rowPtr; s.rowOf = self.rowOf;   // the pattern is shared
    s.ownsPattern = 0;
    s.values = self.values.scale(k);
    return s;
}

struct SparseTensor SparseTensor::transpose() const {
    unsafe { tensor_host_raw_((void*)self.colIdx); tensor_host_raw_((void*)self.rowOf); }
    // counting sort of the nonzeros by column
    struct SparseTensor t = sp_alloc_(self.cols, self.rows, self.nnz, 0);
    &FloatTensor perm = FloatTensor::new_1d(self.nnz == 0UL ? 1UL : self.nnz, 0);
    unsafe {
        unsigned long c = 0UL;
        while (c <= self.cols) { t.rowPtr->data[c] = (float)0.0; c = c + 1UL; }
        unsigned long e = 0UL;
        while (e < self.nnz) {
            unsigned long cc = (unsigned long)self.colIdx->data[e];
            t.rowPtr->data[cc + 1UL] = t.rowPtr->data[cc + 1UL] + (float)1.0;
            e = e + 1UL;
        }
        c = 0UL;
        while (c < self.cols) { t.rowPtr->data[c + 1UL] = t.rowPtr->data[c + 1UL] + t.rowPtr->data[c]; c = c + 1UL; }
        unsigned long* next = (unsigned long*)malloc(sizeof(unsigned long) * (self.cols + 1UL));
        c = 0UL;
        while (c < self.cols) { next[c] = (unsigned long)t.rowPtr->data[c]; c = c + 1UL; }
        e = 0UL;
        while (e < self.nnz) {
            unsigned long cc = (unsigned long)self.colIdx->data[e];
            unsigned long dst = next[cc];
            next[cc] = dst + 1UL;
            t.colIdx->data[dst] = self.rowOf->data[e];
            t.rowOf->data[dst] = (float)cc;
            perm->data[dst] = (float)e;
            e = e + 1UL;
        }
        free((void*)next);
    }
    // the values: a gather of this matrix's, so gradients flow back to them
    sp_drop_(t.values);
    t.values = tensor_gather_rows(self.values, perm);
    sp_drop_(perm);
    return t;
}

} // namespace std
