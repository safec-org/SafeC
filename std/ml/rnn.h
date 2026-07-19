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

namespace std {

// ── Plain RNN ────────────────────────────────────────────────────────────────
// h_t = tanh(x_t @ W_ih + h_{t-1} @ W_hh + b_ih + b_hh)
struct RNNCell {
    &Tensor W_ih; // [inputSize, hiddenSize]
    &Tensor W_hh; // [hiddenSize, hiddenSize]
    &Tensor b_ih; // [1, hiddenSize]
    &Tensor b_hh; // [1, hiddenSize]
};

struct RNNCell rnn_cell_new(unsigned long inputSize, unsigned long hiddenSize);
&Tensor rnn_cell_forward(const &RNNCell cell, const &Tensor x_t, const &Tensor h_prev);
void rnn_cell_free(&RNNCell cell);

// ── GRU ──────────────────────────────────────────────────────────────────────
// r_t = sigmoid(x_t@W_ir + h@W_hr + b_ir + b_hr)          reset gate
// z_t = sigmoid(x_t@W_iz + h@W_hz + b_iz + b_hz)          update gate
// n_t = tanh(x_t@W_in + b_in + r_t * (h@W_hn + b_hn))     candidate
// h_t = (1 - z_t) * n_t + z_t * h_{t-1}
struct GRUCell {
    &Tensor W_ir; &Tensor W_hr; &Tensor b_ir; &Tensor b_hr;
    &Tensor W_iz; &Tensor W_hz; &Tensor b_iz; &Tensor b_hz;
    &Tensor W_in; &Tensor W_hn; &Tensor b_in; &Tensor b_hn;
};

struct GRUCell gru_cell_new(unsigned long inputSize, unsigned long hiddenSize);
&Tensor gru_cell_forward(const &GRUCell cell, const &Tensor x_t, const &Tensor h_prev);
void gru_cell_free(&GRUCell cell);

// ── LSTM ─────────────────────────────────────────────────────────────────────
// i_t = sigmoid(x_t@W_ii + h@W_hi + b_ii + b_hi)   input gate
// f_t = sigmoid(x_t@W_if + h@W_hf + b_if + b_hf)   forget gate
// g_t = tanh(x_t@W_ig + h@W_hg + b_ig + b_hg)      cell candidate
// o_t = sigmoid(x_t@W_io + h@W_ho + b_io + b_ho)   output gate
// c_t = f_t * c_{t-1} + i_t * g_t
// h_t = o_t * tanh(c_t)
struct LSTMCell {
    &Tensor W_ii; &Tensor W_hi; &Tensor b_ii; &Tensor b_hi;
    &Tensor W_if; &Tensor W_hf; &Tensor b_if; &Tensor b_hf;
    &Tensor W_ig; &Tensor W_hg; &Tensor b_ig; &Tensor b_hg;
    &Tensor W_io; &Tensor W_ho; &Tensor b_io; &Tensor b_ho;
};

struct LSTMCell lstm_cell_new(unsigned long inputSize, unsigned long hiddenSize);
// Writes the new hidden state's *tensor* as the return value; '*c_out'
// receives the new cell state (caller-owned, like the return value).
// 'c_out' stays a raw pointer-to-pointer out-param (not a clean single-
// object '&T' case — it's an output slot for a whole other reference).
&Tensor lstm_cell_forward(const &LSTMCell cell, const &Tensor x_t,
                           const &Tensor h_prev, const &Tensor c_prev,
                           struct Tensor** c_out);
void lstm_cell_free(&LSTMCell cell);

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
    &Tensor Wz; &Tensor Rz; &Tensor bz;
    &Tensor Wi; &Tensor Ri; &Tensor bi;
    &Tensor Wf; &Tensor Rf; &Tensor bf;
    &Tensor Wo; &Tensor Ro; &Tensor bo;
};

struct XLSTMState {
    &Tensor h;
    &Tensor c;
    &Tensor n;
    &Tensor m; // [1,1] scalar stabilizer (broadcast across hidden dim)
};

struct XLSTMCell xlstm_cell_new(unsigned long inputSize, unsigned long hiddenSize);
struct XLSTMState xlstm_cell_forward(const &XLSTMCell cell, const &Tensor x_t,
                                      const &XLSTMState prev);
void xlstm_cell_free(&XLSTMCell cell);
void xlstm_state_free(&XLSTMState s);

} // namespace std
