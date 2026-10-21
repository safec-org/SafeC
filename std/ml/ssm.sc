// SafeC Standard Library — state space models (see ssm.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/ssm.h>
#include <std/ml/tensor.h>
#include <std/math.h>
#include <std/sync/parallel.h>

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);

static void ssm_release_(&Tensor t) {
    t->free();
    unsafe { free((void*)t); }
}

static float ssm_silu_(float v) { return v / ((float)1.0 + exp_f(-v)); }

static float ssm_softplus_(float v) {
    if (v > (float)20.0) return v;   // torch.nn.functional.softplus's threshold
    return log_f((float)1.0 + exp_f(v));
}

// ── Selective scan, parallel over channels ──────────────────────────────────
// Rows of x / delta / z / y are 'ld' floats apart (x and z can be column
// ranges of one wider matrix); B and C are rows 'ldBC' apart. When 'dtBias'
// is set, delta holds raw projections and softplus(delta + dtBias) is
// applied here; when 'z' is set, y is gated by silu(z).
struct ScanJob {
    const float* x;      unsigned long ldx;
    const float* delta;  unsigned long ldd;
    const float* z;      unsigned long ldz;
    const float* B;
    const float* C;      unsigned long ldBC;
    const float* A;      // [D][N]; negated exp(Alog) when 'expA'
    int          expA;
    const float* Dskip;
    const float* dtBias;
    float*       h;      // [D][N] state, read and updated
    float*       y;      unsigned long ldy;
    unsigned long T;
    unsigned long N;
};

static void scan_channels_(&ScanJob j, unsigned long lo, unsigned long hi) {
    unsafe {
        float* a = (float*)malloc(sizeof(float) * j.N);
        unsigned long d = lo;
        while (d < hi) {
            float* h = &j.h[d * j.N];
            unsigned long n = 0UL;
            while (n < j.N) {
                float av = j.A[d * j.N + n];
                a[n] = j.expA ? -exp_f(av) : av;
                n = n + 1UL;
            }
            float skip = j.Dskip[d];
            unsigned long t = 0UL;
            while (t < j.T) {
                float xv = j.x[t * j.ldx + d];
                float dt = j.delta[t * j.ldd + d];
                if (j.dtBias != (const float*)0) dt = ssm_softplus_(dt + j.dtBias[d]);
                const float* Bt = &j.B[t * j.ldBC];
                const float* Ct = &j.C[t * j.ldBC];
                float dx = dt * xv;
                float acc = (float)0.0;
                n = 0UL;
                while (n < j.N) {
                    float hv = exp_f(dt * a[n]) * h[n] + dx * Bt[n];
                    h[n] = hv;
                    acc = acc + Ct[n] * hv;
                    n = n + 1UL;
                }
                float yv = acc + skip * xv;
                if (j.z != (const float*)0) yv = yv * ssm_silu_(j.z[t * j.ldz + d]);
                j.y[t * j.ldy + d] = yv;
                t = t + 1UL;
            }
            d = d + 1UL;
        }
        free((void*)a);
    }
}

static void scan_run_(&ScanJob j, unsigned long D) {
    unsigned long per = j.T * j.N + 1UL;
    unsigned long grain = 32768UL / per + 1UL;
    if (D * per >= 65536UL) parallel_for(D, grain, scan_channels_, j);
    else scan_channels_(j, 0UL, D);
}

&Tensor ssm_selective_scan(const &Tensor x, const &Tensor delta, const &Tensor A,
                           const &Tensor B, const &Tensor C, const &Tensor Dskip) {
    unsigned long T; unsigned long D; unsigned long N;
    unsafe { T = x->shape[0]; D = x->shape[1]; N = A->shape[1]; }
    &Tensor y = Tensor::new_2d(T, D, 0);
    struct ScanJob j;
    unsafe {
        float* h = (float*)malloc(sizeof(float) * D * N);
        unsigned long i = 0UL;
        while (i < D * N) { h[i] = (float)0.0; i = i + 1UL; }
        j.x = (const float*)x->data; j.ldx = D;
        j.delta = (const float*)delta->data; j.ldd = D;
        j.z = (const float*)0; j.ldz = 0UL;
        j.B = (const float*)B->data; j.C = (const float*)C->data; j.ldBC = N;
        j.A = (const float*)A->data; j.expA = 0;
        j.Dskip = (const float*)Dskip->data;
        j.dtBias = (const float*)0;
        j.h = h;
        j.y = (float*)y->data; j.ldy = D;
        j.T = T; j.N = N;
        scan_run_(&j, D);
        free((void*)h);
    }
    return y;
}

// ── Causal depthwise convolution + SiLU, parallel over channels ─────────────
// Input row t of channel d is in[t * ldin + d]; positions before the
// sequence come from 'state' ([D][K-1], oldest first), which is then
// replaced by the last K-1 inputs.
struct ConvJob {
    const float* in;  unsigned long ldin;
    const float* w;   // [D][K]
    const float* b;   // [D]
    float*       state;
    float*       out; // [T][D]
    unsigned long T;
    unsigned long D;
    unsigned long K;
};

static void conv_channels_(&ConvJob j, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long K1 = j.K - 1UL;
        float* hist = (float*)malloc(sizeof(float) * (K1 + 1UL));
        unsigned long d = lo;
        while (d < hi) {
            const float* w = &j.w[d * j.K];
            float* st = &j.state[d * K1];
            unsigned long t = 0UL;
            while (t < j.T) {
                float acc = j.b[d];
                unsigned long k = 0UL;
                while (k < j.K) {
                    // input index t - K1 + k
                    float v = (float)0.0;
                    if (t + k >= K1) v = j.in[(t + k - K1) * j.ldin + d];
                    else v = st[t + k];
                    acc = acc + w[k] * v;
                    k = k + 1UL;
                }
                j.out[t * j.D + d] = ssm_silu_(acc);
                t = t + 1UL;
            }
            // new state: the last K1 inputs of (old state ++ this sequence)
            unsigned long k = 0UL;
            while (k < K1) {
                unsigned long idx = j.T + k;   // index into old state ++ inputs
                if (idx >= K1) hist[k] = j.in[(idx - K1) * j.ldin + d];
                else hist[k] = st[idx];
                k = k + 1UL;
            }
            k = 0UL;
            while (k < K1) { st[k] = hist[k]; k = k + 1UL; }
            d = d + 1UL;
        }
        free((void*)hist);
    }
}

// ── Mamba block ──────────────────────────────────────────────────────────────

static float ssm_init_(unsigned long i, unsigned long seed) {
    unsigned long h = (i + 1UL) * 2654435761UL + seed * 40503UL;
    h = (h ^ (h >> 13UL)) * 2246822519UL;
    h = h ^ (h >> 16UL);
    return ((float)(h % 2000UL) / (float)1000.0) - (float)1.0;   // [-1, 1)
}

static &Tensor ssm_rand_(unsigned long rows, unsigned long cols, unsigned long seed, float scale) {
    &Tensor t = Tensor::new_2d(rows, cols, 0);
    unsafe {
        unsigned long i = 0UL;
        while (i < rows * cols) { t->data[i] = ssm_init_(i, seed) * scale; i = i + 1UL; }
    }
    return t;
}

struct Mamba Mamba::new(unsigned long dModel, unsigned long dState, unsigned long dConv, unsigned long expand) {
    struct Mamba m;
    unsigned long dInner = expand * dModel;
    unsigned long dtRank = (dModel + 15UL) / 16UL;
    m.dModel = dModel; m.dInner = dInner; m.dState = dState; m.dConv = dConv; m.dtRank = dtRank;
    m.inProj  = ssm_rand_(dModel, 2UL * dInner, 31UL, (float)1.0 / sqrt_f((float)dModel));
    m.convW   = ssm_rand_(dInner, dConv, 32UL, (float)1.0 / sqrt_f((float)dConv));
    m.convB   = ssm_rand_(1UL, dInner, 33UL, (float)1.0 / sqrt_f((float)dConv));
    m.xProj   = ssm_rand_(dInner, dtRank + 2UL * dState, 34UL, (float)1.0 / sqrt_f((float)dInner));
    m.dtProj  = ssm_rand_(dtRank, dInner, 35UL, (float)1.0 / sqrt_f((float)dtRank));
    m.outProj = ssm_rand_(dInner, dModel, 36UL, (float)1.0 / sqrt_f((float)dInner));
    m.dtBias  = Tensor::new_2d(1UL, dInner, 0);
    m.Alog    = Tensor::new_2d(dInner, dState, 0);
    m.Dskip   = Tensor::new_2d(1UL, dInner, 0);
    unsafe {
        float lo = log_f((float)0.001);
        float hi = log_f((float)0.1);
        unsigned long d = 0UL;
        while (d < dInner) {
            // dt log-uniform in [0.001, 0.1]; bias = softplus^-1(dt) = dt + log(1 - exp(-dt))
            float u = (ssm_init_(d, 37UL) + (float)1.0) * (float)0.5;
            float dt = exp_f(lo + u * (hi - lo));
            m.dtBias->data[d] = dt + log_f((float)1.0 - exp_f(-dt));
            m.Dskip->data[d] = (float)1.0;
            unsigned long n = 0UL;
            while (n < dState) {
                m.Alog->data[d * dState + n] = log_f((float)(n + 1UL));
                n = n + 1UL;
            }
            d = d + 1UL;
        }
    }
    return m;
}

void Mamba::free() {
    ssm_release_(self->inProj); ssm_release_(self->convW); ssm_release_(self->convB);
    ssm_release_(self->xProj); ssm_release_(self->dtProj); ssm_release_(self->dtBias);
    ssm_release_(self->Alog); ssm_release_(self->Dskip); ssm_release_(self->outProj);
}

// Runs T tokens from the given state (conv [dInner][dConv-1], h
// [dInner][dState]) and leaves the state after the last token there.
static &Tensor mamba_run_(const &Mamba m, const &Tensor X, &Tensor convState, &Tensor hState) {
    unsigned long T; unsigned long E; unsigned long N; unsigned long R;
    unsafe { T = X->size / m->dModel; }
    E = m->dInner; N = m->dState; R = m->dtRank;

    &Tensor X2 = Tensor::new_2d(T, m->dModel, 0);
    unsafe {
        unsigned long i = 0UL;
        while (i < T * m->dModel) { X2->data[i] = X->data[i]; i = i + 1UL; }
    }
    &Tensor xz = tensor_matmul(X2, m->inProj);          // [T, 2E]: x | z
    ssm_release_(X2);

    &Tensor xc = Tensor::new_2d(T, E, 0);
    struct ConvJob cj;
    unsafe {
        cj.in = (const float*)xz->data; cj.ldin = 2UL * E;
        cj.w = (const float*)m->convW->data; cj.b = (const float*)m->convB->data;
        cj.state = (float*)convState->data;
        cj.out = (float*)xc->data;
    }
    cj.T = T; cj.D = E; cj.K = m->dConv;
    if (T * E * m->dConv >= 65536UL) parallel_for(E, 64UL, conv_channels_, &cj);
    else conv_channels_(&cj, 0UL, E);

    &Tensor xdbl = tensor_matmul(xc, m->xProj);         // [T, R + 2N]: dt | B | C
    unsigned long W = R + 2UL * N;
    &Tensor dtLow = Tensor::new_2d(T, R, 0);
    unsafe {
        unsigned long t = 0UL;
        while (t < T) {
            unsigned long r = 0UL;
            while (r < R) { dtLow->data[t * R + r] = xdbl->data[t * W + r]; r = r + 1UL; }
            t = t + 1UL;
        }
    }
    &Tensor dt = tensor_matmul(dtLow, m->dtProj);       // [T, E], before bias + softplus
    ssm_release_(dtLow);

    &Tensor y = Tensor::new_2d(T, E, 0);
    struct ScanJob j;
    unsafe {
        j.x = (const float*)xc->data; j.ldx = E;
        j.delta = (const float*)dt->data; j.ldd = E;
        j.z = (const float*)&xz->data[E]; j.ldz = 2UL * E;
        j.B = (const float*)&xdbl->data[R];
        j.C = (const float*)&xdbl->data[R + N]; j.ldBC = W;
        j.A = (const float*)m->Alog->data; j.expA = 1;
        j.Dskip = (const float*)m->Dskip->data;
        j.dtBias = (const float*)m->dtBias->data;
        j.h = (float*)hState->data;
        j.y = (float*)y->data; j.ldy = E;
    }
    j.T = T; j.N = N;
    scan_run_(&j, E);

    &Tensor out = tensor_matmul(y, m->outProj);         // [T, dModel]
    ssm_release_(xz); ssm_release_(xc); ssm_release_(xdbl); ssm_release_(dt); ssm_release_(y);
    return out;
}

struct MambaState MambaState::new(const &Mamba layer) {
    struct MambaState s;
    unsigned long k1 = layer->dConv > 1UL ? layer->dConv - 1UL : 1UL;
    s.conv = Tensor::new_2d(layer->dInner, k1, 0);
    s.h = Tensor::new_2d(layer->dInner, layer->dState, 0);
    return s;
}

void MambaState::reset() {
    unsafe {
        unsigned long i = 0UL;
        while (i < self->conv->size) { self->conv->data[i] = (float)0.0; i = i + 1UL; }
        i = 0UL;
        while (i < self->h->size) { self->h->data[i] = (float)0.0; i = i + 1UL; }
    }
}

void MambaState::free() {
    ssm_release_(self->conv);
    ssm_release_(self->h);
}

&Tensor Mamba::forward(const &Tensor X) const {
    struct MambaState s = MambaState::new(self);
    s.reset();
    &Tensor out = mamba_run_(self, X, s.conv, s.h);
    s.free();
    return out;
}

&Tensor Mamba::step(&MambaState state, const &Tensor x) const {
    return mamba_run_(self, x, state->conv, state->h);
}

void Mamba::collect_params(&Vec<&Tensor> out) const {
    out.push(self.inProj);
    out.push(self.convW);
    out.push(self.convB);
    out.push(self.xProj);
    out.push(self.dtProj);
    out.push(self.dtBias);
    out.push(self.Alog);
    out.push(self.Dskip);
    out.push(self.outProj);
}

void Mamba::collect_state(&Vec<&Tensor> out) const { self.collect_params(out); }

} // namespace std
