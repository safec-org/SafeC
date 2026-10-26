#pragma once
// SafeC Standard Library — sparse tensors (CSR) with autograd and devices.
//
// A SparseTensor is a rows x cols matrix that stores only its nonzeros, in
// compressed sparse row form: for row r, entries rowPtr[r] .. rowPtr[r+1]-1
// of colIdx / values. The values are an ordinary FloatTensor of length nnz,
// so a sparse matrix can be a trained parameter (a sparse layer, a graph's
// learned edge weights): gradients reach the values and nothing else -- the
// sparsity pattern is fixed. The index arrays are float32 tensors too
// (indices exact up to 2^24), so the whole matrix moves to a device like any
// tensor and its ops run there.
//
//   struct SparseTensor A = SparseTensor::from_dense(adj, (float)0.0, 1);   // learned edge weights
//   &FloatTensor H = tensor_spmm(&A, X);           // [rows, n] = A [rows, cols] . X [cols, n]
//   ... loss ... loss.backward();                   // A.values.grad, X.grad
//   A.place(DEVICE_ROCM);                          // the values and indices on the GPU
//
// Conforms to trait Tensor (element access, fill, backward, zero_grad, free)
// and trait DeviceTensor (device, place, sync), so generic code written
// against those traits takes it next to a DenseTensor.
#include <std/ml/tensor.h>

namespace std {

struct SparseTensor {
    &FloatTensor values;    // [nnz, 1], trainable
    &FloatTensor colIdx;    // [nnz] column of each nonzero
    &FloatTensor rowPtr;    // [rows + 1] start of each row in values / colIdx
    &FloatTensor rowOf;     // [nnz] row of each nonzero (for the backward pass)
    unsigned long rows;
    unsigned long cols;
    unsigned long nnz;
    int ownsPattern;        // 0 for a scale() view, which shares the original's pattern

    // Element (r, c), 0 where there is no nonzero; at1 takes r * cols + c.
    float at1(unsigned long i) const;
    float at2(unsigned long r, unsigned long c) const;
    // Sets an existing nonzero; a position outside the pattern is left alone.
    void  set1(unsigned long i, float v);
    void  set2(unsigned long r, unsigned long c, float v);
    void  fill(float v);          // every stored value
    void  backward();             // the values must be a scalar loss's graph: forwards to values.backward()
    void  zero_grad();
    void  free();

    int  device() const;
    int  place(int device);       // values and indices together; 1 on success
    int  sync();
    unsigned long device_buffer() const;   // the values' device buffer

    // A dense [rows, cols] copy: a graph node (gradients reach the values).
    &FloatTensor to_dense() const;
    // The same pattern with every value scaled: a graph node. It shares the
    // pattern arrays (its free() releases only its values).
    struct SparseTensor scale(float k) const;
    // The transpose (cols x rows), its values a graph node of these.
    struct SparseTensor transpose() const;

    // From CSR arrays (host): rowPtr [rows + 1], colIdx and values [nnz]
    // (shorter slices panic).
    static struct SparseTensor new_csr(unsigned long rows, unsigned long cols, []const unsigned long rowPtr,
                                       []const unsigned long colIdx, []const float values, int requiresGrad);
    // The entries of a dense [rows, cols] tensor whose magnitude exceeds 'threshold'.
    static struct SparseTensor from_dense(const &FloatTensor dense, float threshold, int requiresGrad);
    // The n x n identity.
    static struct SparseTensor identity(unsigned long n, int requiresGrad);
};

// out [rows, n] = S [rows, cols] . X [cols, n], with gradients to S's values
// and to X. Runs on the device of S's values / X when either is on one.
&FloatTensor tensor_spmm(const &SparseTensor s, const &FloatTensor x);

} // namespace std
