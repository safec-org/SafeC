#pragma once
// SafeC Standard Library — Recurrent cells: RNN, GRU, LSTM, xLSTM.
//
// All vectors are row tensors: x_t is [1, inputSize], h_t/c_t are
// [1, hiddenSize]. Weight matrices are laid out so a plain
// tensor_matmul(x_t, W) gives the right shape directly (i.e. W is
// [inputSize, hiddenSize] or [hiddenSize, hiddenSize], the transpose of
// PyTorch's own [hiddenSize, inputSize] convention — chosen so this
// library's existing row-vector Tensor shape needs no extra transpose
// per step). Forward-only, like the rest of std/ml (see attention.h).
#include <std/ml/tensor.h>

#include <std/collections/vec.h>

namespace std {

// ── Plain RNN ────────────────────────────────────────────────────────────────
// h_t = tanh(x_t @ W_ih + h_{t-1} @ W_hh + b_ih + b_hh)
struct RNNCell {
    &FloatTensor W_ih; // [inputSize, hiddenSize]
    &FloatTensor W_hh; // [hiddenSize, hiddenSize]
    &FloatTensor b_ih; // [1, hiddenSize]
    &FloatTensor b_hh; // [1, hiddenSize]

    &FloatTensor forward(const &FloatTensor x_t, const &FloatTensor h_prev) const;
    // Parameterized (module.h): every weight tensor.
    void collect_params(&Vec<&FloatTensor> out) const;
    void collect_state(&Vec<&FloatTensor> out) const;
    void drop();
    void free() consume;   // = drop()

    static struct RNNCell new(unsigned long inputSize, unsigned long hiddenSize);
};

// ── GRU ──────────────────────────────────────────────────────────────────────
// r_t = sigmoid(x_t@W_ir + h@W_hr + b_ir + b_hr)          reset gate
// z_t = sigmoid(x_t@W_iz + h@W_hz + b_iz + b_hz)          update gate
// n_t = tanh(x_t@W_in + b_in + r_t * (h@W_hn + b_hn))     candidate
// h_t = (1 - z_t) * n_t + z_t * h_{t-1}
struct GRUCell {
    &FloatTensor W_ir; &FloatTensor W_hr; &FloatTensor b_ir; &FloatTensor b_hr;
    &FloatTensor W_iz; &FloatTensor W_hz; &FloatTensor b_iz; &FloatTensor b_hz;
    &FloatTensor W_in; &FloatTensor W_hn; &FloatTensor b_in; &FloatTensor b_hn;

    &FloatTensor forward(const &FloatTensor x_t, const &FloatTensor h_prev) const;
    // Parameterized (module.h): every weight tensor.
    void collect_params(&Vec<&FloatTensor> out) const;
    void collect_state(&Vec<&FloatTensor> out) const;
    void drop();
    void free() consume;   // = drop()

    static struct GRUCell new(unsigned long inputSize, unsigned long hiddenSize);
};

// ── LSTM ─────────────────────────────────────────────────────────────────────
// i_t = sigmoid(x_t@W_ii + h@W_hi + b_ii + b_hi)   input gate
// f_t = sigmoid(x_t@W_if + h@W_hf + b_if + b_hf)   forget gate
// g_t = tanh(x_t@W_ig + h@W_hg + b_ig + b_hg)      cell candidate
// o_t = sigmoid(x_t@W_io + h@W_ho + b_io + b_ho)   output gate
// c_t = f_t * c_{t-1} + i_t * g_t
// h_t = o_t * tanh(c_t)
struct LSTMCell {
    &FloatTensor W_ii; &FloatTensor W_hi; &FloatTensor b_ii; &FloatTensor b_hi;
    &FloatTensor W_if; &FloatTensor W_hf; &FloatTensor b_if; &FloatTensor b_hf;
    &FloatTensor W_ig; &FloatTensor W_hg; &FloatTensor b_ig; &FloatTensor b_hg;
    &FloatTensor W_io; &FloatTensor W_ho; &FloatTensor b_io; &FloatTensor b_ho;

    // Writes the new hidden state's *tensor* as the return value; '*c_out'
    // receives the new cell state (caller-owned, like the return value).
    &FloatTensor forward(const &FloatTensor x_t, const &FloatTensor h_prev, const &FloatTensor c_prev, &&FloatTensor c_out) const;
    // Parameterized (module.h): every weight tensor.
    void collect_params(&Vec<&FloatTensor> out) const;
    void collect_state(&Vec<&FloatTensor> out) const;
    void drop();
    void free() consume;   // = drop()

    static struct LSTMCell new(unsigned long inputSize, unsigned long hiddenSize);
};

// ── xLSTM (sLSTM variant, Beck et al. 2024) ─────────────────────────────────
// The scalar-memory sLSTM cell — xLSTM's headline idea is *exponential*
// gating (replacing LSTM's sigmoid input/forget gates with unbounded
// exp(), which can represent much larger gate values) made numerically
// stable with a running log-domain stabilizer state 'm_t', plus a
// normalizer state 'n_t' that keeps the exponential gates' magnitude in
// check without clipping. This is the scalar-memory (sLSTM) member of
// the xLSTM family specifically — mLSTM (matrix-memory, fully
// parallelizable) is a separate, larger cell not implemented here.
//   z_t = tanh(x_t@Wz + h@Rz + bz)                    cell input
//   i_raw = x_t@Wi + h@Ri + bi;  f_raw = x_t@Wf + h@Rf + bf   (pre-activation logits)
//   o_t = sigmoid(x_t@Wo + h@Ro + bo)
//   m_t = max(f_raw + m_{t-1}, i_raw)                 stabilizer (log-domain)
//   i_t = exp(i_raw - m_t);  f_t = exp(f_raw + m_{t-1} - m_t)
//   c_t = f_t * c_{t-1} + i_t * z_t
//   n_t = f_t * n_{t-1} + i_t
//   h_t = o_t * (c_t / n_t)
struct XLSTMCell {
    &FloatTensor Wz; &FloatTensor Rz; &FloatTensor bz;
    &FloatTensor Wi; &FloatTensor Ri; &FloatTensor bi;
    &FloatTensor Wf; &FloatTensor Rf; &FloatTensor bf;
    &FloatTensor Wo; &FloatTensor Ro; &FloatTensor bo;

    struct XLSTMState forward(const &FloatTensor x_t, const &XLSTMState prev) const;
    // Parameterized (module.h): every weight tensor.
    void collect_params(&Vec<&FloatTensor> out) const;
    void collect_state(&Vec<&FloatTensor> out) const;
    void drop();
    void free() consume;   // = drop()

    static struct XLSTMCell new(unsigned long inputSize, unsigned long hiddenSize);
};

struct XLSTMState {
    &FloatTensor h;
    &FloatTensor c;
    &FloatTensor n;
    &FloatTensor m; // [1,1] scalar stabilizer (broadcast across hidden dim)
    void drop();

    void free() consume;   // = drop()
};

} // namespace std
