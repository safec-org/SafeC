// SafeC Standard Library — Recurrent cell implementations (see rnn.h).
#pragma once
#include <std/ml/rnn.h>
#include <std/ml/tensor.h>
#include <std/ml/tensor.h>
#include <std/ml/activations.h>
#include <std/ml/activations.h>
#include <std/math.h>
#include <std/math.h>
#include <std/mem.h>

namespace std {

// Small deterministic pseudo-random fill (not a real RNG — this library's
// point is correct *math*, not trained weights; callers that need real
// initialization should overwrite '.data' themselves, e.g. by loading
// trained weights). Values stay in a small range to keep gate pre-
// activations well-behaved.
static &FloatTensor __rnn_init_weight(unsigned long rows, unsigned long cols, unsigned long seed) {
    &FloatTensor t = FloatTensor::new_2d(rows, cols, 0);
    unsafe {
        unsigned long i = 0UL;
        while (i < t->size) {
            unsigned long h = (i + 1UL) * 2654435761UL + seed * 40503UL;
            h = (h ^ (h >> 13UL)) * 2246822519UL;
            float v = ((float)(h % 2000UL) / (float)1000.0) - (float)1.0; // [-1, 1)
            t->data[i] = v * (float)0.3;
            i = i + 1UL;
        }
    }
    return t;
}

static &FloatTensor __rnn_zeros(unsigned long rows, unsigned long cols) {
    return FloatTensor::new_2d(rows, cols, 0);
}

static &FloatTensor __row_add4(&FloatTensor a, &FloatTensor b,
                                  &FloatTensor c, &FloatTensor d) {
    &FloatTensor ab = tensor_add(a, b);
    &FloatTensor cd = tensor_add(c, d);
    &FloatTensor out = tensor_add(ab, cd);
    ab->free(); cd->free();
    unsafe { free((void*)ab); free((void*)cd); }
    return out;
}

static &FloatTensor __one_minus(&FloatTensor z) {
    &FloatTensor out = z.zeros_like();
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) { out->data[i] = (float)1.0 - z->data[i]; i = i + 1UL; }
    }
    return out;
}

// ── Plain RNN ────────────────────────────────────────────────────────────────

struct RNNCell RNNCell::new(unsigned long inputSize, unsigned long hiddenSize) {
    struct RNNCell c;
    c.W_ih = __rnn_init_weight(inputSize, hiddenSize, 1UL);
    c.W_hh = __rnn_init_weight(hiddenSize, hiddenSize, 2UL);
    c.b_ih = __rnn_zeros(1UL, hiddenSize);
    c.b_hh = __rnn_zeros(1UL, hiddenSize);
    return c;
}

&FloatTensor RNNCell::forward(const &FloatTensor x_t, const &FloatTensor h_prev) const {
    unsafe {
        &FloatTensor xi = tensor_matmul(x_t, self->W_ih);
        &FloatTensor hh = tensor_matmul(h_prev, self->W_hh);
        &FloatTensor sum = __row_add4(xi, hh, self->b_ih, self->b_hh);
        &FloatTensor h_t = sum.tanh_fwd();
        xi->free(); hh->free(); sum->free();
        free((void*)xi); free((void*)hh); free((void*)sum);
        return h_t;
    }
}

void RNNCell::free() {
    unsafe {
        self->W_ih->free(); self->W_hh->free(); self->b_ih->free(); self->b_hh->free();
        free((void*)self->W_ih); free((void*)self->W_hh);
        free((void*)self->b_ih); free((void*)self->b_hh);
    }
}

// ── GRU ──────────────────────────────────────────────────────────────────────

struct GRUCell GRUCell::new(unsigned long inputSize, unsigned long hiddenSize) {
    struct GRUCell c;
    c.W_ir = __rnn_init_weight(inputSize, hiddenSize, 10UL); c.W_hr = __rnn_init_weight(hiddenSize, hiddenSize, 11UL);
    c.b_ir = __rnn_zeros(1UL, hiddenSize); c.b_hr = __rnn_zeros(1UL, hiddenSize);
    c.W_iz = __rnn_init_weight(inputSize, hiddenSize, 12UL); c.W_hz = __rnn_init_weight(hiddenSize, hiddenSize, 13UL);
    c.b_iz = __rnn_zeros(1UL, hiddenSize); c.b_hz = __rnn_zeros(1UL, hiddenSize);
    c.W_in = __rnn_init_weight(inputSize, hiddenSize, 14UL); c.W_hn = __rnn_init_weight(hiddenSize, hiddenSize, 15UL);
    c.b_in = __rnn_zeros(1UL, hiddenSize); c.b_hn = __rnn_zeros(1UL, hiddenSize);
    return c;
}

&FloatTensor GRUCell::forward(const &FloatTensor x_t, const &FloatTensor h_prev) const {
    unsafe {
        &FloatTensor r_sum = __row_add4(tensor_matmul(x_t, self->W_ir), tensor_matmul(h_prev, self->W_hr),
                                           self->b_ir, self->b_hr);
        &FloatTensor r_t = r_sum.sigmoid_fwd();

        &FloatTensor z_sum = __row_add4(tensor_matmul(x_t, self->W_iz), tensor_matmul(h_prev, self->W_hz),
                                           self->b_iz, self->b_hz);
        &FloatTensor z_t = z_sum.sigmoid_fwd();

        &FloatTensor hn = tensor_matmul(h_prev, self->W_hn);
        &FloatTensor hn_b = tensor_add(hn, self->b_hn);
        &FloatTensor r_hn = tensor_mul(r_t, hn_b);
        &FloatTensor in_ = tensor_matmul(x_t, self->W_in);
        &FloatTensor in_b = tensor_add(in_, self->b_in);
        &FloatTensor n_sum = tensor_add(in_b, r_hn);
        &FloatTensor n_t = n_sum.tanh_fwd();

        &FloatTensor one_minus_z = __one_minus(z_t);
        &FloatTensor term1 = tensor_mul(one_minus_z, n_t);
        &FloatTensor term2 = tensor_mul(z_t, h_prev);
        &FloatTensor h_t = tensor_add(term1, term2);

        &FloatTensor garbage[13];
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

void GRUCell::free() {
    unsafe {
        &FloatTensor all[12];
        all[0]=self->W_ir; all[1]=self->W_hr; all[2]=self->b_ir; all[3]=self->b_hr;
        all[4]=self->W_iz; all[5]=self->W_hz; all[6]=self->b_iz; all[7]=self->b_hz;
        all[8]=self->W_in; all[9]=self->W_hn; all[10]=self->b_in; all[11]=self->b_hn;
        int i = 0;
        while (i < 12) { all[i]->free(); free((void*)all[i]); i = i + 1; }
    }
}

// ── LSTM ─────────────────────────────────────────────────────────────────────

struct LSTMCell LSTMCell::new(unsigned long inputSize, unsigned long hiddenSize) {
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

&FloatTensor LSTMCell::forward(const &FloatTensor x_t, const &FloatTensor h_prev, const &FloatTensor c_prev, &&FloatTensor c_out) const {
    unsafe {
        &FloatTensor i_sum = __row_add4(tensor_matmul(x_t, self->W_ii), tensor_matmul(h_prev, self->W_hi),
                                           self->b_ii, self->b_hi);
        &FloatTensor i_t = i_sum.sigmoid_fwd();
        &FloatTensor f_sum = __row_add4(tensor_matmul(x_t, self->W_if), tensor_matmul(h_prev, self->W_hf),
                                           self->b_if, self->b_hf);
        &FloatTensor f_t = f_sum.sigmoid_fwd();
        &FloatTensor g_sum = __row_add4(tensor_matmul(x_t, self->W_ig), tensor_matmul(h_prev, self->W_hg),
                                           self->b_ig, self->b_hg);
        &FloatTensor g_t = g_sum.tanh_fwd();
        &FloatTensor o_sum = __row_add4(tensor_matmul(x_t, self->W_io), tensor_matmul(h_prev, self->W_ho),
                                           self->b_io, self->b_ho);
        &FloatTensor o_t = o_sum.sigmoid_fwd();

        &FloatTensor fc = tensor_mul(f_t, c_prev);
        &FloatTensor ig = tensor_mul(i_t, g_t);
        &FloatTensor c_t = tensor_add(fc, ig);
        &FloatTensor tanh_c = c_t.tanh_fwd();
        &FloatTensor h_t = tensor_mul(o_t, tanh_c);

        &FloatTensor garbage[9];
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

void LSTMCell::free() {
    unsafe {
        &FloatTensor all[16];
        all[0]=self->W_ii; all[1]=self->W_hi; all[2]=self->b_ii; all[3]=self->b_hi;
        all[4]=self->W_if; all[5]=self->W_hf; all[6]=self->b_if; all[7]=self->b_hf;
        all[8]=self->W_ig; all[9]=self->W_hg; all[10]=self->b_ig; all[11]=self->b_hg;
        all[12]=self->W_io; all[13]=self->W_ho; all[14]=self->b_io; all[15]=self->b_ho;
        int i = 0;
        while (i < 16) { all[i]->free(); free((void*)all[i]); i = i + 1; }
    }
}

// ── xLSTM (sLSTM) ─────────────────────────────────────────────────────────────

struct XLSTMCell XLSTMCell::new(unsigned long inputSize, unsigned long hiddenSize) {
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

struct XLSTMState XLSTMCell::forward(const &FloatTensor x_t, const &XLSTMState prev) const {
    unsafe {
        &FloatTensor z_sum = __row_add4(tensor_matmul(x_t, self->Wz), tensor_matmul(prev->h, self->Rz),
                                           self->bz, self->bz.zeros_like());
        &FloatTensor z_t = z_sum.tanh_fwd();

        &FloatTensor zerob = self->bi.zeros_like();
        &FloatTensor i_raw = __row_add4(tensor_matmul(x_t, self->Wi), tensor_matmul(prev->h, self->Ri),
                                           self->bi, zerob);
        &FloatTensor zerob2 = self->bf.zeros_like();
        &FloatTensor f_raw = __row_add4(tensor_matmul(x_t, self->Wf), tensor_matmul(prev->h, self->Rf),
                                           self->bf, zerob2);
        &FloatTensor zerob3 = self->bo.zeros_like();
        &FloatTensor o_sum = __row_add4(tensor_matmul(x_t, self->Wo), tensor_matmul(prev->h, self->Ro),
                                           self->bo, zerob3);
        &FloatTensor o_t = o_sum.sigmoid_fwd();

        // Stabilizer: m_t = max(f_raw + m_prev, i_raw), elementwise
        // (broadcasting prev->m's single scalar across the hidden dim).
        unsigned long hiddenSize;
        hiddenSize = i_raw->shape[1];
        &FloatTensor m_t = FloatTensor::new_2d(1UL, hiddenSize, 0);
        &FloatTensor i_t = FloatTensor::new_2d(1UL, hiddenSize, 0);
        &FloatTensor f_t = FloatTensor::new_2d(1UL, hiddenSize, 0);
        float mPrevScalar = prev->m->data[0];
        unsigned long k = 0UL;
        while (k < hiddenSize) {
            float fRaw = f_raw->data[k];
            float iRaw = i_raw->data[k];
            float candidateF = fRaw + mPrevScalar;
            float m = (candidateF > iRaw) ? candidateF : iRaw;
            m_t->data[k] = m;
            i_t->data[k] = exp_f(iRaw - m);
            f_t->data[k] = exp_f(candidateF - m);
            k = k + 1UL;
        }

        &FloatTensor fc = tensor_mul(f_t, prev->c);
        &FloatTensor iz = tensor_mul(i_t, z_t);
        &FloatTensor c_t = tensor_add(fc, iz);

        &FloatTensor fnT = tensor_mul(f_t, prev->n);
        &FloatTensor n_t = tensor_add(fnT, i_t);

        // h_t = o_t * (c_t / n_t), elementwise divide
        &FloatTensor ratio = FloatTensor::new_2d(1UL, hiddenSize, 0);
        k = 0UL;
        while (k < hiddenSize) {
            float denom = n_t->data[k];
            ratio->data[k] = (denom != (float)0.0) ? (c_t->data[k] / denom) : (float)0.0;
            k = k + 1UL;
        }
        &FloatTensor h_t = tensor_mul(o_t, ratio);

        // m_t is stored as a [1,1] scalar (the *maximum* over the hidden
        // dim, matching the paper's scalar-per-timestep stabilizer) —
        // reduce the elementwise m_t computed above.
        &FloatTensor m_scalar = FloatTensor::new_2d(1UL, 1UL, 0);
        float mMax = m_t->data[0];
        k = 1UL;
        while (k < hiddenSize) { if (m_t->data[k] > mMax) mMax = m_t->data[k]; k = k + 1UL; }
        m_scalar->data[0] = mMax;

        &FloatTensor garbage[15];
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

void XLSTMCell::free() {
    unsafe {
        &FloatTensor all[12];
        all[0]=self->Wz; all[1]=self->Rz; all[2]=self->bz;
        all[3]=self->Wi; all[4]=self->Ri; all[5]=self->bi;
        all[6]=self->Wf; all[7]=self->Rf; all[8]=self->bf;
        all[9]=self->Wo; all[10]=self->Ro; all[11]=self->bo;
        int i = 0;
        while (i < 12) { all[i]->free(); free((void*)all[i]); i = i + 1; }
    }
}

void XLSTMState::free() {
    unsafe {
        self->h->free(); self->c->free(); self->n->free(); self->m->free();
        free((void*)self->h); free((void*)self->c); free((void*)self->n); free((void*)self->m);
    }
}

void RNNCell::collect_params(&Vec<&FloatTensor> out) const {
    out.push(self.W_ih);
    out.push(self.W_hh);
    out.push(self.b_ih);
    out.push(self.b_hh);
}

void RNNCell::collect_state(&Vec<&FloatTensor> out) const { self.collect_params(out); }

void GRUCell::collect_params(&Vec<&FloatTensor> out) const {
    out.push(self.W_ir);
    out.push(self.W_hr);
    out.push(self.b_ir);
    out.push(self.b_hr);
    out.push(self.W_iz);
    out.push(self.W_hz);
    out.push(self.b_iz);
    out.push(self.b_hz);
    out.push(self.W_in);
    out.push(self.W_hn);
    out.push(self.b_in);
    out.push(self.b_hn);
}

void GRUCell::collect_state(&Vec<&FloatTensor> out) const { self.collect_params(out); }

void LSTMCell::collect_params(&Vec<&FloatTensor> out) const {
    out.push(self.W_ii);
    out.push(self.W_hi);
    out.push(self.b_ii);
    out.push(self.b_hi);
    out.push(self.W_if);
    out.push(self.W_hf);
    out.push(self.b_if);
    out.push(self.b_hf);
    out.push(self.W_ig);
    out.push(self.W_hg);
    out.push(self.b_ig);
    out.push(self.b_hg);
    out.push(self.W_io);
    out.push(self.W_ho);
    out.push(self.b_io);
    out.push(self.b_ho);
}

void LSTMCell::collect_state(&Vec<&FloatTensor> out) const { self.collect_params(out); }

void XLSTMCell::collect_params(&Vec<&FloatTensor> out) const {
    out.push(self.Wz);
    out.push(self.Rz);
    out.push(self.bz);
    out.push(self.Wi);
    out.push(self.Ri);
    out.push(self.bi);
    out.push(self.Wf);
    out.push(self.Rf);
    out.push(self.bf);
    out.push(self.Wo);
    out.push(self.Ro);
    out.push(self.bo);
}

void XLSTMCell::collect_state(&Vec<&FloatTensor> out) const { self.collect_params(out); }

} // namespace std
