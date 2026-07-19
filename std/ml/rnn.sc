// SafeC Standard Library — Recurrent cell implementations (see rnn.h).
#pragma once
#include <std/ml/rnn.h>
#include <std/ml/tensor.h>
#include <std/ml/tensor.sc>
#include <std/ml/activations.h>
#include <std/ml/activations.sc>
#include <std/math.h>
#include <std/math.sc>
#include <std/mem.sc>

namespace std {

// Small deterministic pseudo-random fill (not a real RNG — this library's
// point is correct *math*, not trained weights; callers that need real
// initialization should overwrite '.data' themselves, e.g. by loading
// trained weights). Values stay in a small range to keep gate pre-
// activations well-behaved.
static struct Tensor* __rnn_init_weight(unsigned long rows, unsigned long cols, unsigned long seed) {
    struct Tensor* t = tensor_new_2d(rows, cols, 0);
    unsafe {
        unsigned long i = 0UL;
        while (i < t->size) {
            unsigned long h = (i + 1UL) * 2654435761UL + seed * 40503UL;
            h = (h ^ (h >> 13UL)) * 2246822519UL;
            double v = ((double)(h % 2000UL) / 1000.0) - 1.0; // [-1, 1)
            t->data[i] = v * 0.3;
            i = i + 1UL;
        }
    }
    return t;
}

static struct Tensor* __rnn_zeros(unsigned long rows, unsigned long cols) {
    return tensor_new_2d(rows, cols, 0);
}

static struct Tensor* __row_add4(struct Tensor* a, struct Tensor* b,
                                  struct Tensor* c, struct Tensor* d) {
    struct Tensor* ab = tensor_add(a, b);
    struct Tensor* cd = tensor_add(c, d);
    struct Tensor* out = tensor_add(ab, cd);
    ab->free(); cd->free();
    unsafe { free((void*)ab); free((void*)cd); }
    return out;
}

static struct Tensor* __one_minus(struct Tensor* z) {
    struct Tensor* out = tensor_zeros_like(z);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) { out->data[i] = 1.0 - z->data[i]; i = i + 1UL; }
    }
    return out;
}

// ── Plain RNN ────────────────────────────────────────────────────────────────

struct RNNCell rnn_cell_new(unsigned long inputSize, unsigned long hiddenSize) {
    struct RNNCell c;
    c.W_ih = __rnn_init_weight(inputSize, hiddenSize, 1UL);
    c.W_hh = __rnn_init_weight(hiddenSize, hiddenSize, 2UL);
    c.b_ih = __rnn_zeros(1UL, hiddenSize);
    c.b_hh = __rnn_zeros(1UL, hiddenSize);
    return c;
}

struct Tensor* rnn_cell_forward(struct RNNCell* cell, struct Tensor* x_t, struct Tensor* h_prev) {
    unsafe {
        struct Tensor* xi = tensor_matmul(x_t, cell->W_ih);
        struct Tensor* hh = tensor_matmul(h_prev, cell->W_hh);
        struct Tensor* sum = __row_add4(xi, hh, cell->b_ih, cell->b_hh);
        struct Tensor* h_t = tensor_tanh(sum);
        xi->free(); hh->free(); sum->free();
        free((void*)xi); free((void*)hh); free((void*)sum);
        return h_t;
    }
}

void rnn_cell_free(struct RNNCell* cell) {
    unsafe {
        cell->W_ih->free(); cell->W_hh->free(); cell->b_ih->free(); cell->b_hh->free();
        free((void*)cell->W_ih); free((void*)cell->W_hh);
        free((void*)cell->b_ih); free((void*)cell->b_hh);
    }
}

// ── GRU ──────────────────────────────────────────────────────────────────────

struct GRUCell gru_cell_new(unsigned long inputSize, unsigned long hiddenSize) {
    struct GRUCell c;
    c.W_ir = __rnn_init_weight(inputSize, hiddenSize, 10UL); c.W_hr = __rnn_init_weight(hiddenSize, hiddenSize, 11UL);
    c.b_ir = __rnn_zeros(1UL, hiddenSize); c.b_hr = __rnn_zeros(1UL, hiddenSize);
    c.W_iz = __rnn_init_weight(inputSize, hiddenSize, 12UL); c.W_hz = __rnn_init_weight(hiddenSize, hiddenSize, 13UL);
    c.b_iz = __rnn_zeros(1UL, hiddenSize); c.b_hz = __rnn_zeros(1UL, hiddenSize);
    c.W_in = __rnn_init_weight(inputSize, hiddenSize, 14UL); c.W_hn = __rnn_init_weight(hiddenSize, hiddenSize, 15UL);
    c.b_in = __rnn_zeros(1UL, hiddenSize); c.b_hn = __rnn_zeros(1UL, hiddenSize);
    return c;
}

struct Tensor* gru_cell_forward(struct GRUCell* cell, struct Tensor* x_t, struct Tensor* h_prev) {
    unsafe {
        struct Tensor* r_sum = __row_add4(tensor_matmul(x_t, cell->W_ir), tensor_matmul(h_prev, cell->W_hr),
                                           cell->b_ir, cell->b_hr);
        struct Tensor* r_t = tensor_sigmoid(r_sum);

        struct Tensor* z_sum = __row_add4(tensor_matmul(x_t, cell->W_iz), tensor_matmul(h_prev, cell->W_hz),
                                           cell->b_iz, cell->b_hz);
        struct Tensor* z_t = tensor_sigmoid(z_sum);

        struct Tensor* hn = tensor_matmul(h_prev, cell->W_hn);
        struct Tensor* hn_b = tensor_add(hn, cell->b_hn);
        struct Tensor* r_hn = tensor_mul(r_t, hn_b);
        struct Tensor* in_ = tensor_matmul(x_t, cell->W_in);
        struct Tensor* in_b = tensor_add(in_, cell->b_in);
        struct Tensor* n_sum = tensor_add(in_b, r_hn);
        struct Tensor* n_t = tensor_tanh(n_sum);

        struct Tensor* one_minus_z = __one_minus(z_t);
        struct Tensor* term1 = tensor_mul(one_minus_z, n_t);
        struct Tensor* term2 = tensor_mul(z_t, h_prev);
        struct Tensor* h_t = tensor_add(term1, term2);

        struct Tensor* garbage[13];
        garbage[0]=r_sum; garbage[1]=r_t; garbage[2]=z_sum; garbage[3]=z_t;
        garbage[4]=hn; garbage[5]=hn_b; garbage[6]=r_hn; garbage[7]=in_;
        garbage[8]=in_b; garbage[9]=n_sum; garbage[10]=n_t;
        garbage[11]=one_minus_z; garbage[12]=term1;
        int gi = 0;
        while (gi < 13) { garbage[gi]->free(); free((void*)garbage[gi]); gi = gi + 1; }
        term2->free(); free((void*)term2);
        return h_t;
    }
}

void gru_cell_free(struct GRUCell* cell) {
    unsafe {
        struct Tensor* all[12];
        all[0]=cell->W_ir; all[1]=cell->W_hr; all[2]=cell->b_ir; all[3]=cell->b_hr;
        all[4]=cell->W_iz; all[5]=cell->W_hz; all[6]=cell->b_iz; all[7]=cell->b_hz;
        all[8]=cell->W_in; all[9]=cell->W_hn; all[10]=cell->b_in; all[11]=cell->b_hn;
        int i = 0;
        while (i < 12) { all[i]->free(); free((void*)all[i]); i = i + 1; }
    }
}

// ── LSTM ─────────────────────────────────────────────────────────────────────

struct LSTMCell lstm_cell_new(unsigned long inputSize, unsigned long hiddenSize) {
    struct LSTMCell c;
    c.W_ii = __rnn_init_weight(inputSize, hiddenSize, 20UL); c.W_hi = __rnn_init_weight(hiddenSize, hiddenSize, 21UL);
    c.b_ii = __rnn_zeros(1UL, hiddenSize); c.b_hi = __rnn_zeros(1UL, hiddenSize);
    c.W_if = __rnn_init_weight(inputSize, hiddenSize, 22UL); c.W_hf = __rnn_init_weight(hiddenSize, hiddenSize, 23UL);
    c.b_if = __rnn_zeros(1UL, hiddenSize); c.b_hf = __rnn_zeros(1UL, hiddenSize);
    c.W_ig = __rnn_init_weight(inputSize, hiddenSize, 24UL); c.W_hg = __rnn_init_weight(hiddenSize, hiddenSize, 25UL);
    c.b_ig = __rnn_zeros(1UL, hiddenSize); c.b_hg = __rnn_zeros(1UL, hiddenSize);
    c.W_io = __rnn_init_weight(inputSize, hiddenSize, 26UL); c.W_ho = __rnn_init_weight(hiddenSize, hiddenSize, 27UL);
    c.b_io = __rnn_zeros(1UL, hiddenSize); c.b_ho = __rnn_zeros(1UL, hiddenSize);
    return c;
}

struct Tensor* lstm_cell_forward(struct LSTMCell* cell, struct Tensor* x_t,
                                  struct Tensor* h_prev, struct Tensor* c_prev,
                                  struct Tensor** c_out) {
    unsafe {
        struct Tensor* i_sum = __row_add4(tensor_matmul(x_t, cell->W_ii), tensor_matmul(h_prev, cell->W_hi),
                                           cell->b_ii, cell->b_hi);
        struct Tensor* i_t = tensor_sigmoid(i_sum);
        struct Tensor* f_sum = __row_add4(tensor_matmul(x_t, cell->W_if), tensor_matmul(h_prev, cell->W_hf),
                                           cell->b_if, cell->b_hf);
        struct Tensor* f_t = tensor_sigmoid(f_sum);
        struct Tensor* g_sum = __row_add4(tensor_matmul(x_t, cell->W_ig), tensor_matmul(h_prev, cell->W_hg),
                                           cell->b_ig, cell->b_hg);
        struct Tensor* g_t = tensor_tanh(g_sum);
        struct Tensor* o_sum = __row_add4(tensor_matmul(x_t, cell->W_io), tensor_matmul(h_prev, cell->W_ho),
                                           cell->b_io, cell->b_ho);
        struct Tensor* o_t = tensor_sigmoid(o_sum);

        struct Tensor* fc = tensor_mul(f_t, c_prev);
        struct Tensor* ig = tensor_mul(i_t, g_t);
        struct Tensor* c_t = tensor_add(fc, ig);
        struct Tensor* tanh_c = tensor_tanh(c_t);
        struct Tensor* h_t = tensor_mul(o_t, tanh_c);

        struct Tensor* garbage[9];
        garbage[0]=i_sum; garbage[1]=i_t; garbage[2]=f_sum; garbage[3]=f_t;
        garbage[4]=g_sum; garbage[5]=g_t; garbage[6]=o_sum; garbage[7]=o_t;
        garbage[8]=tanh_c;
        int gi = 0;
        while (gi < 9) { garbage[gi]->free(); free((void*)garbage[gi]); gi = gi + 1; }
        fc->free(); free((void*)fc);
        ig->free(); free((void*)ig);

        *c_out = c_t;
        return h_t;
    }
}

void lstm_cell_free(struct LSTMCell* cell) {
    unsafe {
        struct Tensor* all[16];
        all[0]=cell->W_ii; all[1]=cell->W_hi; all[2]=cell->b_ii; all[3]=cell->b_hi;
        all[4]=cell->W_if; all[5]=cell->W_hf; all[6]=cell->b_if; all[7]=cell->b_hf;
        all[8]=cell->W_ig; all[9]=cell->W_hg; all[10]=cell->b_ig; all[11]=cell->b_hg;
        all[12]=cell->W_io; all[13]=cell->W_ho; all[14]=cell->b_io; all[15]=cell->b_ho;
        int i = 0;
        while (i < 16) { all[i]->free(); free((void*)all[i]); i = i + 1; }
    }
}

// ── xLSTM (sLSTM) ─────────────────────────────────────────────────────────────

struct XLSTMCell xlstm_cell_new(unsigned long inputSize, unsigned long hiddenSize) {
    struct XLSTMCell c;
    c.Wz = __rnn_init_weight(inputSize, hiddenSize, 30UL); c.Rz = __rnn_init_weight(hiddenSize, hiddenSize, 31UL);
    c.bz = __rnn_zeros(1UL, hiddenSize);
    c.Wi = __rnn_init_weight(inputSize, hiddenSize, 32UL); c.Ri = __rnn_init_weight(hiddenSize, hiddenSize, 33UL);
    c.bi = __rnn_zeros(1UL, hiddenSize);
    c.Wf = __rnn_init_weight(inputSize, hiddenSize, 34UL); c.Rf = __rnn_init_weight(hiddenSize, hiddenSize, 35UL);
    c.bf = __rnn_zeros(1UL, hiddenSize);
    c.Wo = __rnn_init_weight(inputSize, hiddenSize, 36UL); c.Ro = __rnn_init_weight(hiddenSize, hiddenSize, 37UL);
    c.bo = __rnn_zeros(1UL, hiddenSize);
    return c;
}

struct XLSTMState xlstm_cell_forward(struct XLSTMCell* cell, struct Tensor* x_t,
                                      struct XLSTMState* prev) {
    unsafe {
        struct Tensor* z_sum = __row_add4(tensor_matmul(x_t, cell->Wz), tensor_matmul(prev->h, cell->Rz),
                                           cell->bz, tensor_zeros_like(cell->bz));
        struct Tensor* z_t = tensor_tanh(z_sum);

        struct Tensor* zerob = tensor_zeros_like(cell->bi);
        struct Tensor* i_raw = __row_add4(tensor_matmul(x_t, cell->Wi), tensor_matmul(prev->h, cell->Ri),
                                           cell->bi, zerob);
        struct Tensor* zerob2 = tensor_zeros_like(cell->bf);
        struct Tensor* f_raw = __row_add4(tensor_matmul(x_t, cell->Wf), tensor_matmul(prev->h, cell->Rf),
                                           cell->bf, zerob2);
        struct Tensor* zerob3 = tensor_zeros_like(cell->bo);
        struct Tensor* o_sum = __row_add4(tensor_matmul(x_t, cell->Wo), tensor_matmul(prev->h, cell->Ro),
                                           cell->bo, zerob3);
        struct Tensor* o_t = tensor_sigmoid(o_sum);

        // Stabilizer: m_t = max(f_raw + m_prev, i_raw), elementwise
        // (broadcasting prev->m's single scalar across the hidden dim).
        unsigned long hiddenSize;
        hiddenSize = i_raw->shape[1];
        struct Tensor* m_t = tensor_new_2d(1UL, hiddenSize, 0);
        struct Tensor* i_t = tensor_new_2d(1UL, hiddenSize, 0);
        struct Tensor* f_t = tensor_new_2d(1UL, hiddenSize, 0);
        double mPrevScalar = prev->m->data[0];
        unsigned long k = 0UL;
        while (k < hiddenSize) {
            double fRaw = f_raw->data[k];
            double iRaw = i_raw->data[k];
            double candidateF = fRaw + mPrevScalar;
            double m = (candidateF > iRaw) ? candidateF : iRaw;
            m_t->data[k] = m;
            i_t->data[k] = exp_d(iRaw - m);
            f_t->data[k] = exp_d(candidateF - m);
            k = k + 1UL;
        }

        struct Tensor* fc = tensor_mul(f_t, prev->c);
        struct Tensor* iz = tensor_mul(i_t, z_t);
        struct Tensor* c_t = tensor_add(fc, iz);

        struct Tensor* fnT = tensor_mul(f_t, prev->n);
        struct Tensor* n_t = tensor_add(fnT, i_t);

        // h_t = o_t * (c_t / n_t), elementwise divide
        struct Tensor* ratio = tensor_new_2d(1UL, hiddenSize, 0);
        k = 0UL;
        while (k < hiddenSize) {
            double denom = n_t->data[k];
            ratio->data[k] = (denom != 0.0) ? (c_t->data[k] / denom) : 0.0;
            k = k + 1UL;
        }
        struct Tensor* h_t = tensor_mul(o_t, ratio);

        // m_t is stored as a [1,1] scalar (the *maximum* over the hidden
        // dim, matching the paper's scalar-per-timestep stabilizer) —
        // reduce the elementwise m_t computed above.
        struct Tensor* m_scalar = tensor_new_2d(1UL, 1UL, 0);
        double mMax = m_t->data[0];
        k = 1UL;
        while (k < hiddenSize) { if (m_t->data[k] > mMax) mMax = m_t->data[k]; k = k + 1UL; }
        m_scalar->data[0] = mMax;

        struct Tensor* garbage[15];
        garbage[0]=z_sum; garbage[1]=zerob; garbage[2]=i_raw; garbage[3]=zerob2;
        garbage[4]=f_raw; garbage[5]=zerob3; garbage[6]=o_sum; garbage[7]=m_t;
        garbage[8]=i_t; garbage[9]=f_t; garbage[10]=fc; garbage[11]=iz;
        garbage[12]=fnT; garbage[13]=ratio; garbage[14]=z_t;
        int gi = 0;
        while (gi < 15) { garbage[gi]->free(); free((void*)garbage[gi]); gi = gi + 1; }

        struct XLSTMState next;
        next.h = h_t;
        next.c = c_t;
        next.n = n_t;
        next.m = m_scalar;
        return next;
    }
}

void xlstm_cell_free(struct XLSTMCell* cell) {
    unsafe {
        struct Tensor* all[12];
        all[0]=cell->Wz; all[1]=cell->Rz; all[2]=cell->bz;
        all[3]=cell->Wi; all[4]=cell->Ri; all[5]=cell->bi;
        all[6]=cell->Wf; all[7]=cell->Rf; all[8]=cell->bf;
        all[9]=cell->Wo; all[10]=cell->Ro; all[11]=cell->bo;
        int i = 0;
        while (i < 12) { all[i]->free(); free((void*)all[i]); i = i + 1; }
    }
}

void xlstm_state_free(struct XLSTMState* s) {
    unsafe {
        s->h->free(); s->c->free(); s->n->free(); s->m->free();
        free((void*)s->h); free((void*)s->c); free((void*)s->n); free((void*)s->m);
    }
}

} // namespace std
