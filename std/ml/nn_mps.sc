// SafeC Standard Library — device-resident training on Apple GPUs (see nn_mps.h).
//
// The kernels are in gpu_mps_kernels.metal's "Training" section (the
// library's one embedded metallib) and run through gpu_mps.h's
// mps_dispatch; each wrapper below fills the kernel's constant struct (the
// Nn* structs there, field for field). Matmuls are mps_gemm.
#pragma once
#include <std/ml/nn_mps.h>
#include <std/ml/gpu_mps.h>
#include <std/ml/nn.h>
#include <std/ml/datasets.h>
#include <std/ml/trainer.h>
#include <std/math.h>

namespace std {

extern void* malloc(unsigned long size);
extern void* calloc(unsigned long count, unsigned long size);
extern void  free(void* ptr);

#define NM_GROUP 256UL   // threads per threadgroup for elementwise kernels and reductions

// ── Buffers ──────────────────────────────────────────────────────────────────

static struct MpsBuf nm_none_() {
    struct MpsBuf b;
    unsafe { b.handle = (void*)0; }
    b.bytes = 0UL;
    b.dtype = 0;
    return b;
}

// b holds at least n floats (4-byte values)
static int nm_fits_(struct MpsBuf b, unsigned long n) { return b.valid() && b.bytes >= n * 4UL; }

static int nm_alloc_(&MpsBuf b, unsigned long n) {
    *b = MpsBuf::alloc(n, 0);
    return b.valid();
}

// n zeroed floats
static int nm_alloc_zero_(&MpsBuf b, unsigned long n) {
    if (!nm_alloc_(b, n)) return 0;
    int ok = 0;
    unsafe {
        void* z = calloc(n == 0UL ? 1UL : n, 4UL);
        if (z == (void*)0) return 0;
        ok = b.write((const &void)(const void*)z, n * 4UL);
        free(z);
    }
    return ok;
}

static int nm_tensor_ok_(const &FloatTensor t, unsigned long size) {
    unsafe { return t->size == size; }
}

static int nm_upload_(&MpsBuf b, const &FloatTensor t, unsigned long size) {
    if (!nm_tensor_ok_(t, size)) return 0;
    unsafe { *b = MpsBuf::upload((const &float)(const float*)t->data, size); }
    return b.valid();
}

// A batch of its own when none is open: 'own' says to end it (nm_end_).
static int nm_begin_(&int own) {
    *own = !mps_batch_is_active();
    if (*own) mps_batch_begin();
    return mps_batch_is_active();
}

static void nm_end_(int own) {
    if (own) mps_batch_end();
}

// ── Kernel wrappers ──────────────────────────────────────────────────────────

struct NmBiasAct_ {
    unsigned int n;
    unsigned int cols;
    int act;
};

struct NmActGrad_ {
    unsigned int rows;
    unsigned int cols;
    int act;
};

struct NmLoss_ {
    unsigned int n;
    unsigned int cols;
    float scale;
};

struct NmSumSq_ {
    unsigned int n;
    unsigned int slot;
};

struct NmOpt_ {
    unsigned int n;
    int kind;
    unsigned int nTensors;
    float lr;
    float mu;
    float b1;
    float b2;
    float eps;
    float wd;
    float ibc1;
    float isbc2;
    float clip;
    float clampW;
};

struct NmGather_ {
    unsigned int base;
    unsigned int rows;
    unsigned int fx;
    unsigned int fy;
};

static unsigned long nm_groups_(unsigned long n) { return (n + NM_GROUP - 1UL) / NM_GROUP; }

// h = act(z = z + b[column]) over n values of rows of 'cols'
static int nm_bias_act_(struct MpsBuf z, struct MpsBuf h, struct MpsBuf b, unsigned long n, unsigned long cols,
                        int act) {
    struct MpsBuf bufs[3];
    bufs[0] = z; bufs[1] = h; bufs[2] = b;
    struct NmBiasAct_ p;
    p.n = (unsigned int)n; p.cols = (unsigned int)cols; p.act = act;
    unsafe {
        return mps_dispatch("nn_bias_act", &bufs[0], 3UL, (const &void)(const void*)&p, sizeof(struct NmBiasAct_),
                            nm_groups_(n), NM_GROUP);
    }
}

// d *= act'(src) in place, gb = d's column sums ([rows][cols])
static int nm_act_grad_rows_(struct MpsBuf d, struct MpsBuf src, struct MpsBuf gb, unsigned long rows,
                             unsigned long cols, int act) {
    struct MpsBuf bufs[3];
    bufs[0] = d; bufs[1] = src; bufs[2] = gb;
    struct NmActGrad_ p;
    p.rows = (unsigned int)rows; p.cols = (unsigned int)cols; p.act = act;
    unsafe {
        return mps_dispatch("nn_act_grad_rows", &bufs[0], 3UL, (const &void)(const void*)&p,
                            sizeof(struct NmActGrad_), (cols + 15UL) / 16UL, 256UL);
    }
}

// nn_ce_loss / nn_bce_loss / nn_mse_loss: one threadgroup
static int nm_loss_(const &char kernel, struct MpsBuf x, struct MpsBuf t, struct MpsBuf d, struct MpsBuf loss1,
                    unsigned long n, unsigned long cols, float scale) {
    struct MpsBuf bufs[4];
    bufs[0] = x; bufs[1] = t; bufs[2] = d; bufs[3] = loss1;
    struct NmLoss_ p;
    p.n = (unsigned int)n; p.cols = (unsigned int)cols; p.scale = scale;
    unsafe {
        return mps_dispatch(kernel, &bufs[0], 4UL, (const &void)(const void*)&p, sizeof(struct NmLoss_), 1UL, NM_GROUP);
    }
}

// ── Losses ───────────────────────────────────────────────────────────────────

int mps_cross_entropy(struct MpsBuf logits, struct MpsBuf labels, struct MpsBuf d, struct MpsBuf loss1,
                      unsigned long rows, unsigned long classes) {
    if (rows == 0UL || classes == 0UL || !nm_fits_(logits, rows * classes) || !nm_fits_(labels, rows) ||
        !nm_fits_(d, rows * classes) || !nm_fits_(loss1, 1UL)) return 0;
    return nm_loss_("nn_ce_loss", logits, labels, d, loss1, rows, classes, (float)1.0);
}

int mps_bce_logits(struct MpsBuf logits, struct MpsBuf target, struct MpsBuf d, struct MpsBuf loss1,
                   unsigned long n, float scale) {
    if (n == 0UL || !nm_fits_(logits, n) || !nm_fits_(target, n) || !nm_fits_(d, n) || !nm_fits_(loss1, 1UL))
        return 0;
    return nm_loss_("nn_bce_loss", logits, target, d, loss1, n, 1UL, scale);
}

int mps_mse(struct MpsBuf pred, struct MpsBuf target, struct MpsBuf d, struct MpsBuf loss1,
            unsigned long n, float scale) {
    if (n == 0UL || !nm_fits_(pred, n) || !nm_fits_(target, n) || !nm_fits_(d, n) || !nm_fits_(loss1, 1UL))
        return 0;
    return nm_loss_("nn_mse_loss", pred, target, d, loss1, n, 1UL, scale);
}

// ── Layers ───────────────────────────────────────────────────────────────────

static int nm_needs_z_(int act) { return act == ACT_GELU || act == ACT_SILU; }

static void nm_layer_free_(&MpsLayer L) {
    L.w.free(); L.b.free(); L.gw.free(); L.gb.free();
    L.z.free(); L.h.free(); L.dh.free();
}

struct MpsNet MpsNet::new(unsigned long maxBatch) {
    struct MpsNet n;
    n.layers = vec_new();
    n.maxBatch = maxBatch == 0UL ? 1UL : maxBatch;
    n.ok = mps_available();
    return n;
}

int MpsNet::add_linear(const &FloatTensor w, const &FloatTensor b, int act) {
    unsigned long in;
    unsigned long out;
    unsafe {
        if (w->ndim != 2UL) { self.ok = 0; return 0; }
        in = w->shape[0];
        out = w->shape[1];
    }
    struct MpsLayer L;
    L.kind = MPS_LAYER_LINEAR;
    L.act = act;
    L.inC = in;
    L.outC = out;
    L.w = nm_none_(); L.b = nm_none_(); L.gw = nm_none_(); L.gb = nm_none_();
    L.z = nm_none_(); L.h = nm_none_(); L.dh = nm_none_();
    unsigned long count = self.layers.length();
    unsigned long outN = self.maxBatch * out;
    int ok = self.ok && in != 0UL && out != 0UL && (count == 0UL || self.layers.at(count - 1UL).outC == in) &&
             nm_upload_(&L.w, w, in * out) && nm_upload_(&L.b, b, out) && nm_alloc_(&L.gw, in * out) &&
             nm_alloc_(&L.gb, out) && nm_alloc_(&L.h, outN) && nm_alloc_(&L.dh, outN) &&
             (!nm_needs_z_(act) || nm_alloc_(&L.z, outN));
    if (!ok) { nm_layer_free_(&L); self.ok = 0; return 0; }
    self.layers.push(L);
    return 1;
}

struct MpsNet MpsNet::from_mlp(const &Mlp m, unsigned long maxBatch) {
    struct MpsNet n = MpsNet::new(maxBatch);
    unsigned long l = 0UL;
    while (n.ok && l < m.nLayers) {
        int act = l + 1UL == m.nLayers ? m.outAct : m.act;
        n.add_linear(*m.params.at(2UL * l), *m.params.at(2UL * l + 1UL), act);
        l = l + 1UL;
    }
    return n;
}

unsigned long MpsNet::in_dim() const {
    if (self.layers.length() == 0UL) return 0UL;
    return self.layers.at(0UL).inC;
}

unsigned long MpsNet::out_dim() const {
    unsigned long c = self.layers.length();
    if (c == 0UL) return 0UL;
    return self.layers.at(c - 1UL).outC;
}

// Copies every (w, b) pair between the net and host tensors.
static int nm_transfer_(const &MpsNet n, &Vec<&FloatTensor> ts, int download) {
    if (!n.ok || ts.length() != 2UL * n.layers.length()) return 0;
    unsigned long i = 0UL;
    while (i < n.layers.length()) {
        &MpsLayer L = n.layers.at(i);
        &FloatTensor tw = *ts.at(2UL * i);
        &FloatTensor tb = *ts.at(2UL * i + 1UL);
        if (!nm_tensor_ok_(tw, L.inC * L.outC) || !nm_tensor_ok_(tb, L.outC)) return 0;
        unsafe {
            if (download) {
                if (!L.w.read((&void)(void*)tw->data, L.inC * L.outC * 4UL) ||
                    !L.b.read((&void)(void*)tb->data, L.outC * 4UL)) return 0;
            } else {
                if (!L.w.write((const &void)(const void*)tw->data, L.inC * L.outC * 4UL) ||
                    !L.b.write((const &void)(const void*)tb->data, L.outC * 4UL)) return 0;
            }
        }
        i = i + 1UL;
    }
    return 1;
}

int MpsNet::download(&Vec<&FloatTensor> params) const { return nm_transfer_(self, params, 1); }

int MpsNet::upload(const &Vec<&FloatTensor> params) {
    unsafe { return nm_transfer_(self, (&Vec<&FloatTensor>)params, 0); }
}

int MpsNet::download_mlp(&Mlp m) const { return self.download(&m.params); }

void MpsNet::free() {
    unsigned long i = 0UL;
    while (i < self.layers.length()) { nm_layer_free_(self.layers.at(i)); i = i + 1UL; }
    self.layers.free();
    self.ok = 0;
}

// ── Forward / backward ───────────────────────────────────────────────────────

static int nm_forward_(&MpsNet n, struct MpsBuf x, unsigned long batch) {
    struct MpsBuf cur = x;
    unsigned long i = 0UL;
    while (i < n.layers.length()) {
        &MpsLayer L = n.layers.at(i);
        struct MpsBuf zt = L.z.valid() ? L.z : L.h;
        if (!mps_gemm(cur, 0, L.w, 0, zt, batch, L.outC, L.inC)) return 0;
        if (!nm_bias_act_(zt, L.h, L.b, batch * L.outC, L.outC, L.act)) return 0;
        cur = L.h;
        i = i + 1UL;
    }
    return 1;
}

struct MpsBuf MpsNet::forward(struct MpsBuf x, unsigned long batch) {
    unsigned long c = self.layers.length();
    if (!self.ok || c == 0UL || batch == 0UL || batch > self.maxBatch || !nm_fits_(x, batch * self.in_dim()))
        return nm_none_();
    int own = 0;
    int ok = nm_begin_(&own) && nm_forward_(self, x, batch);
    nm_end_(own);
    if (!ok) return nm_none_();
    return self.layers.at(c - 1UL).h;
}

struct MpsBuf MpsNet::grad_output() const {
    unsigned long c = self.layers.length();
    if (c == 0UL) return nm_none_();
    return self.layers.at(c - 1UL).dh;
}

// Each layer from the last: L.dh becomes the pre-activation gradient (and
// gb its column sums), gw = in^T dh, and the input's gradient dh W^T goes
// to the previous layer's dh (or dx).
static int nm_backward_(&MpsNet n, struct MpsBuf x, unsigned long batch, struct MpsBuf dx) {
    unsigned long i = n.layers.length();
    while (i > 0UL) {
        i = i - 1UL;
        &MpsLayer L = n.layers.at(i);
        struct MpsBuf in = i > 0UL ? n.layers.at(i - 1UL).h : x;
        struct MpsBuf src = L.z.valid() ? L.z : L.h;
        if (!nm_act_grad_rows_(L.dh, src, L.gb, batch, L.outC, L.act)) return 0;
        if (!mps_gemm(in, 1, L.dh, 0, L.gw, L.inC, L.outC, batch)) return 0;
        if (i > 0UL) {
            if (!mps_gemm(L.dh, 0, L.w, 1, n.layers.at(i - 1UL).dh, batch, L.inC, L.outC)) return 0;
        } else if (dx.valid()) {
            if (!mps_gemm(L.dh, 0, L.w, 1, dx, batch, L.inC, L.outC)) return 0;
        }
    }
    return 1;
}

int MpsNet::backward(struct MpsBuf x, unsigned long batch, struct MpsBuf dx) {
    unsigned long c = self.layers.length();
    if (!self.ok || c == 0UL || batch == 0UL || batch > self.maxBatch || !nm_fits_(x, batch * self.in_dim()) ||
        (dx.valid() && !nm_fits_(dx, batch * self.in_dim()))) return 0;
    int own = 0;
    int ok = nm_begin_(&own) && nm_backward_(self, x, batch, dx);
    nm_end_(own);
    return ok;
}

// ── Training steps ───────────────────────────────────────────────────────────

// kind: 0 cross-entropy, 1 BCE on logits, 2 MSE
static int nm_step_(&MpsNet n, struct MpsBuf x, struct MpsBuf t, unsigned long rows, struct MpsBuf loss1, int kind) {
    int own = 0;
    if (!nm_begin_(&own)) return 0;
    struct MpsBuf out = n.forward(x, rows);
    unsigned long od = n.out_dim();
    int ok = out.valid();
    if (ok && kind == 0) ok = mps_cross_entropy(out, t, n.grad_output(), loss1, rows, od);
    if (ok && kind == 1) ok = mps_bce_logits(out, t, n.grad_output(), loss1, rows * od, (float)1.0);
    if (ok && kind == 2) ok = mps_mse(out, t, n.grad_output(), loss1, rows * od, (float)1.0);
    ok = ok && n.backward(x, rows, nm_none_());
    nm_end_(own);
    return ok;
}

int mps_net_ce_step(&MpsNet n, struct MpsBuf x, struct MpsBuf y, unsigned long rows, struct MpsBuf loss1) {
    return nm_step_(n, x, y, rows, loss1, 0);
}

int mps_net_bce_step(&MpsNet n, struct MpsBuf x, struct MpsBuf y, unsigned long rows, struct MpsBuf loss1) {
    return nm_step_(n, x, y, rows, loss1, 1);
}

int mps_net_mse_step(&MpsNet n, struct MpsBuf x, struct MpsBuf y, unsigned long rows, struct MpsBuf loss1) {
    return nm_step_(n, x, y, rows, loss1, 2);
}

int mps_net_recon_bce_step(&MpsNet n, struct MpsBuf x, struct MpsBuf y, unsigned long rows, struct MpsBuf loss1) {
    return nm_step_(n, x, x, rows, loss1, 1);
}

int mps_net_recon_mse_step(&MpsNet n, struct MpsBuf x, struct MpsBuf y, unsigned long rows, struct MpsBuf loss1) {
    return nm_step_(n, x, x, rows, loss1, 2);
}

// ── Optimizer ────────────────────────────────────────────────────────────────

struct MpsOptimizer MpsOptimizer::new(int kind, float lr) {
    struct MpsOptimizer o;
    o.w = vec_new(); o.g = vec_new(); o.m = vec_new(); o.v = vec_new();
    o.kind = kind;
    o.lr = lr;
    o.momentum = (float)0.9;
    o.beta1 = (float)0.9;
    o.beta2 = kind == OPT_RMSPROP || kind == OPT_LION ? (float)0.99 : (float)0.999;
    o.eps = kind == OPT_ADAGRAD ? (float)0.0000000001 : (float)0.00000001;
    o.weightDecay = kind == OPT_ADAMW ? (float)0.01 : (float)0.0;
    o.clipNorm = (float)0.0;
    o.clampWeights = (float)0.0;
    o.t = 0UL;
    o.sumsq = nm_none_();
    o.ok = mps_available();
    return o;
}

struct MpsOptimizer MpsOptimizer::from(const &Optimizer o) {
    struct MpsOptimizer d = MpsOptimizer::new(o.kind, o.lr);
    d.momentum = o.momentum;
    d.beta1 = o.beta1;
    d.beta2 = o.beta2;
    d.eps = o.eps;
    d.weightDecay = o.weightDecay;
    return d;
}

int MpsOptimizer::add(struct MpsBuf w, struct MpsBuf g) {
    if (!self.ok || !w.valid() || !g.valid() || g.bytes < w.bytes) return 0;
    unsigned long n = w.bytes / 4UL;
    struct MpsBuf m = nm_none_();
    struct MpsBuf v = nm_none_();
    if (!nm_alloc_zero_(&m, n) || !nm_alloc_zero_(&v, n)) {
        m.free(); v.free();
        self.ok = 0;
        return 0;
    }
    self.w.push(w); self.g.push(g); self.m.push(m); self.v.push(v);
    // one squared sum per parameter for clipping
    self.sumsq.free();
    if (!nm_alloc_zero_(&self.sumsq, self.w.length())) { self.ok = 0; return 0; }
    return 1;
}

int MpsNet::add_params(&MpsOptimizer o) const {
    if (!self.ok) { o.ok = 0; return 0; }
    unsigned long i = 0UL;
    while (i < self.layers.length()) {
        &MpsLayer L = self.layers.at(i);
        if (!o.add(L.w, L.gw) || !o.add(L.b, L.gb)) return 0;
        i = i + 1UL;
    }
    return 1;
}

int MpsOptimizer::step() {
    if (!self.ok) return 0;
    unsigned long count = self.w.length();
    if (count == 0UL) return 1;
    int own = 0;
    if (!nm_begin_(&own)) return 0;
    int clip = self.clipNorm > (float)0.0;
    int ok = 1;
    unsigned long i = 0UL;
    // clipping: every gradient's squared sum first (the norm needs them all)
    while (ok && clip && i < count) {
        struct MpsBuf bufs[2];
        bufs[0] = *self.g.at(i); bufs[1] = self.sumsq;
        struct NmSumSq_ q;
        q.n = (unsigned int)(self.w.at(i).bytes / 4UL); q.slot = (unsigned int)i;
        unsafe {
            ok = mps_dispatch("nn_sumsq", &bufs[0], 2UL, (const &void)(const void*)&q, sizeof(struct NmSumSq_), 1UL,
                              NM_GROUP);
        }
        i = i + 1UL;
    }
    // Optimizer::step's bias corrections, computed the same way
    unsigned long t = self.t + 1UL;
    float bc1 = (float)1.0 - pow_f(self.beta1, (float)t);
    float bc2 = (float)1.0 - pow_f(self.beta2, (float)t);
    struct NmOpt_ p;
    p.kind = self.kind;
    p.nTensors = (unsigned int)count;
    p.lr = self.lr;
    p.mu = self.momentum;
    p.b1 = self.beta1;
    p.b2 = self.beta2;
    p.eps = self.eps;
    p.wd = self.weightDecay;
    p.ibc1 = (float)1.0 / bc1;
    p.isbc2 = (float)1.0 / sqrt_f(bc2);
    p.clip = clip ? self.clipNorm : (float)0.0;
    p.clampW = self.clampWeights;
    i = 0UL;
    while (ok && i < count) {
        struct MpsBuf bufs[5];
        bufs[0] = *self.w.at(i); bufs[1] = *self.g.at(i); bufs[2] = *self.m.at(i); bufs[3] = *self.v.at(i);
        bufs[4] = self.sumsq;
        unsigned long n = bufs[0].bytes / 4UL;
        p.n = (unsigned int)n;
        unsafe {
            ok = mps_dispatch("nn_opt_step", &bufs[0], 5UL, (const &void)(const void*)&p, sizeof(struct NmOpt_),
                              nm_groups_(n), NM_GROUP);
        }
        i = i + 1UL;
    }
    nm_end_(own);
    if (ok) self.t = t;
    return ok;
}

void MpsOptimizer::free() {
    unsigned long i = 0UL;
    while (i < self.m.length()) { self.m.at(i).free(); self.v.at(i).free(); i = i + 1UL; }
    self.w.free(); self.g.free(); self.m.free(); self.v.free();
    self.sumsq.free();
    self.ok = 0;
}

// ── Batches ──────────────────────────────────────────────────────────────────

// DataLoader's shuffle (datasets.sc's dl_shuffle_), for the same row order.
static void nm_shuffle_(&MpsBatches b) {
    unsafe {
        unsigned long i = b.n;
        while (i > 1UL) {
            unsigned long j = b.rng.below(i);
            unsigned long t = b.perm[i - 1UL]; b.perm[i - 1UL] = b.perm[j]; b.perm[j] = t;
            i = i - 1UL;
        }
    }
}

struct MpsBatches MpsBatches::new(const &Dataset d, unsigned long batchSize, int shuffle, unsigned long seed) {
    struct MpsBatches b;
    b.n = d.n;
    b.batchSize = batchSize == 0UL ? 1UL : batchSize;
    if (b.batchSize > b.n && b.n > 0UL) b.batchSize = b.n;
    b.pos = 0UL;
    b.shuffle = shuffle;
    b.rng = Rng::new(seed);
    unsafe {
        b.features = d.x->shape[1];
        b.targets = d.y->shape[1];
        unsigned long* perm = (unsigned long*)malloc(sizeof(unsigned long) * (d.n + 1UL));
        unsigned int* permu = (unsigned int*)malloc(sizeof(unsigned int) * (d.n + 1UL));
        unsigned long i = 0UL;
        while (i < d.n) { perm[i] = i; i = i + 1UL; }
        b.perm = (&heap unsigned long)perm;
        b.permu = (&heap unsigned int)permu;
    }
    if (shuffle) nm_shuffle_(&b);
    b.x = nm_none_(); b.y = nm_none_(); b.order = nm_none_();
    b.xb = nm_none_(); b.yb = nm_none_(); b.loss = nm_none_();
    b.ok = mps_available() && b.n > 0UL && b.n < 4294967295UL && nm_alloc_(&b.order, b.n) &&
           nm_alloc_zero_(&b.loss, 1UL) && nm_alloc_(&b.xb, b.batchSize * b.features) &&
           nm_alloc_(&b.yb, b.batchSize * b.targets);
    unsafe {
        b.ok = b.ok && nm_upload_(&b.x, d.x, b.n * b.features) && nm_upload_(&b.y, d.y, b.n * b.targets);
    }
    return b;
}

int MpsBatches::epoch() {
    if (!self.ok || mps_batch_is_active()) return 0;
    float zero = (float)0.0;
    unsafe {
        unsigned long i = 0UL;
        while (i < self.n) { self.permu[i] = (unsigned int)self.perm[i]; i = i + 1UL; }
        if (!self.order.write((const &void)(const void*)self.permu, self.n * 4UL)) return 0;
        if (!self.loss.write((const &void)(const void*)&zero, 4UL)) return 0;
    }
    // DataLoader reshuffles at the end of each epoch for the next
    if (self.shuffle) nm_shuffle_(self);
    self.pos = 0UL;
    return 1;
}

int MpsBatches::gather(unsigned long rows) {
    if (!self.ok || rows == 0UL || rows > self.batchSize || self.pos + rows > self.n) return 0;
    struct MpsBuf bufs[5];
    bufs[0] = self.x; bufs[1] = self.xb; bufs[2] = self.y; bufs[3] = self.yb; bufs[4] = self.order;
    struct NmGather_ p;
    p.base = (unsigned int)self.pos; p.rows = (unsigned int)rows;
    p.fx = (unsigned int)self.features; p.fy = (unsigned int)self.targets;
    int ok = 0;
    unsafe {
        ok = mps_dispatch("nn_gather", &bufs[0], 5UL, (const &void)(const void*)&p, sizeof(struct NmGather_),
                          nm_groups_(rows * (self.features + self.targets)), NM_GROUP);
    }
    if (ok) self.pos = self.pos + rows;
    return ok;
}

float MpsBatches::loss_sum() {
    float v = (float)0.0;
    unsafe { if (!self.ok || !self.loss.read((&void)(void*)&v, 4UL)) return (float)0.0; }
    return v;
}

void MpsBatches::free() {
    self.x.free(); self.y.free(); self.order.free();
    self.xb.free(); self.yb.free(); self.loss.free();
    unsafe { free((void*)self.perm); free((void*)self.permu); }
    self.ok = 0;
}

// ── The training traits ──────────────────────────────────────────────────────

int mps_trainer_reserve_(&MpsBuf b, unsigned long n) {
    if (nm_fits_(*b, n)) return 1;
    b.free();
    return nm_alloc_(b, n);
}

int mps_trainer_write_(struct MpsBuf b, const &FloatTensor t, unsigned long n) {
    unsafe { return b.write((const &void)(const void*)t->data, n * 4UL); }
}

float mps_trainer_read1_(struct MpsBuf b) {
    float v = (float)-1.0;
    unsafe { if (!b.read((&void)(void*)&v, 4UL)) return (float)-1.0; }
    return v;
}

static fn int(&MpsNet, struct MpsBuf, struct MpsBuf, unsigned long, struct MpsBuf) nm_loss_step_(int loss) {
    if (loss == TRAIN_LOSS_MSE) return mps_net_mse_step;
    if (loss == TRAIN_LOSS_BCE) return mps_net_bce_step;
    return mps_net_ce_step;
}

// MpsTrainer calls these through its fn fields.
static int  nm_download_mlp_(const &MpsNet n, &Mlp m) { return n.download_mlp(m); }
static void nm_free_net_(&MpsNet n) { n.free(); }

struct MpsTrainer<struct Mlp, struct MpsNet> mps_mlp_trainer_new(struct Mlp m, int loss, const &Optimizer opt,
                                                                 const &TrainerOptions o) {
    struct MpsTrainer<struct Mlp, struct MpsNet> t;
    t.host = m;
    t.dev = MpsNet::from_mlp(&t.host, o.maxBatch);
    t.opt = MpsOptimizer::from(opt);
    t.maxBatch = o.maxBatch;
    t.xs = nm_none_();
    t.ys = nm_none_();
    t.loss1 = nm_none_();
    t.active = 0;
    t.ok = t.dev.ok && o.precision == 0 && t.dev.add_params(&t.opt) && t.opt.ok;
    t.stepFn = nm_loss_step_(loss);
    t.downloadFn = nm_download_mlp_;
    t.freeFn = nm_free_net_;
    return t;
}

} // namespace std
