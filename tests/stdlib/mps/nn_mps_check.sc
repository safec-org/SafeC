// Correctness check for std/ml/nn_mps.h's training stack against nn.h's CPU
// training: mps_gemm in every operand form and kernel tier, every layer
// activation and loss forward and backward (one Autograd step on the CPU vs
// one MpsNet pass), every optimizer kind's steps vs Optimizer::step
// (weight decay, clipping, lr changed inside a batch), and full fits whose
// per-epoch losses must track CpuTrainer's (the same batches: MpsBatches
// shuffles as DataLoader does). Needs an Apple GPU; run with
// tests/stdlib/mps/run.sh.
#include <std/ml/nn_mps.h>
#include <std/ml/gpu_mps.h>
#include <std/ml/nn.h>
#include <std/ml/module.h>
#include <std/ml/datasets.h>
#include <std/ml/trainer.h>
#include <std/ml/train_cpu.h>
#include <std/collections/vec.h>
extern int printf(const char* fmt, ...);
extern int snprintf(char* buf, unsigned long n, const char* fmt, ...);
extern void free(void* ptr);

static int g_checks = 0;
static int g_fails = 0;

static void check(int ok, const char* what) {
    g_checks = g_checks + 1;
    if (!ok) {
        g_fails = g_fails + 1;
        unsafe { printf("  FAIL %s\n", what); }
    }
}

static void tensor_drop(&FloatTensor t) {
    t->free();
    unsafe { free((void*)t); }
}

// Max |gpu - cpu| over n values, relative to max(1, max |cpu|); NaN -> 1e9.
static double err_vals(const float* gpu, const float* cpu, unsigned long n) {
    double m = 0.0;
    double big = 1.0;
    unsigned long i = 0UL;
    unsafe {
        while (i < n) {
            double d = (double)gpu[i] - (double)cpu[i];
            if (d < 0.0) d = -d;
            if (d != d) d = 1.0e9;
            if (d > m) m = d;
            double r = (double)cpu[i];
            if (r < 0.0) r = -r;
            if (r > big) big = r;
            i = i + 1UL;
        }
    }
    return m / big;
}

// A device buffer's first n floats against a host array.
static double err_buf(struct MpsBuf b, const float* cpu, unsigned long n) {
    struct Vec<float> v = std::Vec::with_cap(n);
    unsigned long i = 0UL;
    while (i < n) { v.push((float)0.0); i = i + 1UL; }
    double e = 1.0e9;
    unsafe { if (b.read((&void)(void*)v.at(0UL), n * 4UL)) e = err_vals((const float*)v.at(0UL), cpu, n); }
    v.free();
    return e;
}

static void report(const char* what, double e, double tol) {
    unsafe { printf("  %-46s err %.2e (tol %.0e)\n", what, e, tol); }
    check(e <= tol, what);
}

static struct MpsBuf upload_t(const &FloatTensor t) {
    unsafe { return MpsBuf::upload((const &float)(const float*)t->data, t->size); }
}

static struct MpsBuf zeros(unsigned long n) {
    struct MpsBuf b = MpsBuf::alloc(n, 0);
    struct Vec<float> z = std::Vec::with_cap(n);
    unsigned long i = 0UL;
    while (i < n) { z.push((float)0.0); i = i + 1UL; }
    unsafe { b.write((const &void)(const void*)z.at(0UL), n * 4UL); }
    z.free();
    return b;
}

// ── mps_gemm ─────────────────────────────────────────────────────────────────

// c = op(a) op(b) for one (ta, tb) and shape, against a double loop.
static void gemm_case(int ta, int tb, unsigned long m, unsigned long n, unsigned long k) {
    struct Rng r = Rng::new(m * 131UL + n * 17UL + k);
    &FloatTensor a = FloatTensor::new_2d(ta ? k : m, ta ? m : k, 0);
    &FloatTensor b = FloatTensor::new_2d(tb ? n : k, tb ? k : n, 0);
    &FloatTensor c = FloatTensor::new_2d(m, n, 0);
    r.fill_normal(a, (float)0.0, (float)1.0);
    r.fill_normal(b, (float)0.0, (float)1.0);
    unsafe {
        unsigned long i = 0UL;
        while (i < m) {
            unsigned long j = 0UL;
            while (j < n) {
                double s = 0.0;
                unsigned long p = 0UL;
                while (p < k) {
                    double av = (double)(ta ? a->data[p * m + i] : a->data[i * k + p]);
                    double bv = (double)(tb ? b->data[j * k + p] : b->data[p * n + j]);
                    s = s + av * bv;
                    p = p + 1UL;
                }
                c->data[i * n + j] = (float)s;
                j = j + 1UL;
            }
            i = i + 1UL;
        }
    }
    struct MpsBuf da = upload_t(a);
    struct MpsBuf db = upload_t(b);
    struct MpsBuf dc = MpsBuf::alloc(m * n, 0);
    int ok = mps_gemm(da, ta, db, tb, dc, m, n, k);
    double e = 1.0e9;
    unsafe { if (ok) e = err_buf(dc, (const float*)c->data, m * n); }
    char name[96];
    unsafe { snprintf(&name[0], 96UL, "gemm ta %d tb %d  %lu x %lu x %lu", ta, tb, m, n, k); }
    report(&name[0], e, 0.00002);
    da.free(); db.free(); dc.free();
    tensor_drop(a); tensor_drop(b); tensor_drop(c);
}

static void gemm_checks() {
    // odd shapes (tiled kernels), multiples of 8 (simdgroup matrix) and of 32 (multi-simdgroup)
    int f = 0;
    while (f < 3) {
        int ta = f == 2 ? 1 : 0;
        int tb = f == 1 ? 1 : 0;
        gemm_case(ta, tb, 7UL, 13UL, 5UL);
        gemm_case(ta, tb, 16UL, 24UL, 40UL);
        gemm_case(ta, tb, 64UL, 32UL, 96UL);
        f = f + 1;
    }
    struct MpsBuf x = MpsBuf::alloc(16UL, 0);
    check(!mps_gemm(x, 1, x, 1, x, 4UL, 4UL, 4UL), "gemm rejects ta and tb both set");
    check(!mps_gemm(x, 0, x, 0, x, 8UL, 8UL, 8UL), "gemm rejects buffers too small");
    x.free();
}

// ── Layers and losses: one forward / backward vs Autograd ────────────────────

static const char* act_name(int a) {
    if (a == ACT_RELU) return "relu";
    if (a == ACT_LEAKY) return "leaky";
    if (a == ACT_TANH) return "tanh";
    if (a == ACT_SIGMOID) return "sigmoid";
    if (a == ACT_GELU) return "gelu";
    if (a == ACT_SILU) return "silu";
    return "none";
}

static const char* loss_name(int l) {
    if (l == TRAIN_LOSS_MSE) return "mse";
    if (l == TRAIN_LOSS_BCE) return "bce";
    return "ce";
}

// module.h's built-in loss of an output (module_loss without the forward pass).
static &FloatTensor module_loss_of_(const &FloatTensor out, int loss, const &FloatTensor y) {
    if (loss == TRAIN_LOSS_MSE) return tensor_mse_loss(out, y);
    if (loss == TRAIN_LOSS_BCE) return tensor_bce_logits_loss(out, y);
    return tensor_cross_entropy_loss(out, y);
}

// An Mlp in -> hidden -> hidden -> out ('act' inside; also after the last
// layer for MSE), a batch of 'rows': the output, loss, every parameter
// gradient and the input gradient, CPU vs GPU.
static void layer_case(int act, int loss, unsigned long in, unsigned long hidden, unsigned long out,
                       unsigned long rows) {
    struct Rng r = Rng::new(7UL + (unsigned long)act * 11UL + (unsigned long)loss * 3UL + in);
    struct Mlp m = Mlp::new(in, hidden, 2UL, out, act, loss == TRAIN_LOSS_MSE ? act : ACT_NONE, &r);
    &FloatTensor x = FloatTensor::new_2d(rows, in, 1);
    r.fill_normal(x, (float)0.0, (float)1.0);
    &FloatTensor y = FloatTensor::new_2d(rows, loss == TRAIN_LOSS_CE ? 1UL : out, 0);
    unsafe {
        unsigned long i = 0UL;
        while (i < y->size) {
            if (loss == TRAIN_LOSS_CE) y->data[i] = (float)r.below(out);
            else if (loss == TRAIN_LOSS_BCE) y->data[i] = r.uniform();
            else y->data[i] = r.normal();
            i = i + 1UL;
        }
    }
    struct MpsNet dn = MpsNet::from_mlp(&m, rows);

    // CPU: one Autograd pass
    &FloatTensor o = m.forward(x);
    &FloatTensor l = module_loss_of_(o, loss, y);
    l.backward();

    // GPU: forward, loss, backward with the input gradient
    struct MpsBuf dx = upload_t(x);
    struct MpsBuf dy = upload_t(y);
    struct MpsBuf dxg = MpsBuf::alloc(rows * in, 0);
    struct MpsBuf l1 = zeros(1UL);
    mps_batch_begin();
    struct MpsBuf dout = dn.forward(dx, rows);
    int ok = dout.valid();
    if (ok && loss == TRAIN_LOSS_CE) ok = mps_cross_entropy(dout, dy, dn.grad_output(), l1, rows, out);
    if (ok && loss == TRAIN_LOSS_BCE) ok = mps_bce_logits(dout, dy, dn.grad_output(), l1, rows * out, (float)1.0);
    if (ok && loss == TRAIN_LOSS_MSE) ok = mps_mse(dout, dy, dn.grad_output(), l1, rows * out, (float)1.0);
    ok = ok && dn.backward(dx, rows, dxg);
    mps_batch_end();

    double ef = 1.0e9;
    double eb = 1.0e9;
    if (ok) {
        unsafe {
            ef = err_buf(dout, (const float*)o->data, rows * out);
            double el = err_buf(l1, (const float*)l->data, 1UL);
            if (el > ef) ef = el;
            eb = err_buf(dxg, (const float*)x->grad, rows * in);
            unsigned long li = 0UL;
            while (li < dn.layers.length()) {
                &MpsLayer L = dn.layers.at(li);
                &FloatTensor W = *m.params.at(2UL * li);
                &FloatTensor B = *m.params.at(2UL * li + 1UL);
                double e1 = err_buf(L.gw, (const float*)W->grad, W->size);
                double e2 = err_buf(L.gb, (const float*)B->grad, B->size);
                if (e1 > eb) eb = e1;
                if (e2 > eb) eb = e2;
                li = li + 1UL;
            }
        }
    }
    char name[96];
    unsafe {
        snprintf(&name[0], 96UL, "fwd+loss %s %s %lux%lu-%lu-%lu", act_name(act), loss_name(loss), rows, in, hidden, out);
        report(&name[0], ef, 0.00001);
        snprintf(&name[0], 96UL, "backward %s %s %lux%lu-%lu-%lu", act_name(act), loss_name(loss), rows, in, hidden, out);
        report(&name[0], eb, 0.00002);
    }
    dx.free(); dy.free(); dxg.free(); l1.free();
    dn.free();
    m.free();
    tensor_drop(x); tensor_drop(y);
}

static void layer_checks() {
    int act = 0;
    while (act <= ACT_SILU) {
        int loss = 0;
        while (loss <= TRAIN_LOSS_BCE) {
            layer_case(act, loss, 5UL, 13UL, 3UL, 7UL);   // tiled kernels
            loss = loss + 1;
        }
        act = act + 1;
    }
    layer_case(ACT_RELU, TRAIN_LOSS_CE, 8UL, 24UL, 8UL, 16UL);     // simdgroup matrix
    layer_case(ACT_GELU, TRAIN_LOSS_MSE, 8UL, 24UL, 8UL, 16UL);
    layer_case(ACT_RELU, TRAIN_LOSS_CE, 32UL, 64UL, 32UL, 64UL);   // multi-simdgroup
    layer_case(ACT_TANH, TRAIN_LOSS_BCE, 32UL, 64UL, 32UL, 64UL);
}

// ── Optimizers: steps vs Optimizer::step ─────────────────────────────────────

static const char* opt_name(int k) {
    if (k == OPT_SGD) return "sgd";
    if (k == OPT_MOMENTUM) return "momentum";
    if (k == OPT_NESTEROV) return "nesterov";
    if (k == OPT_RMSPROP) return "rmsprop";
    if (k == OPT_ADAM) return "adam";
    if (k == OPT_ADAMW) return "adamw";
    if (k == OPT_ADAGRAD) return "adagrad";
    return "lion";
}

// Fills every parameter's gradient (CPU) and copies it to the device's.
static void set_grads(&Vec<&FloatTensor> ps, &Vec<struct MpsBuf> gs, &Rng r, float scale) {
    unsigned long i = 0UL;
    while (i < ps.length()) {
        &FloatTensor p = *ps.at(i);
        unsafe {
            __tensor_ensure_grad((FloatTensor*)p);
            unsigned long j = 0UL;
            while (j < p->size) { p->grad[j] = scale * r.normal(); j = j + 1UL; }
            gs.at(i).write((const &void)(const void*)p->grad, p->size * 4UL);
        }
        i = i + 1UL;
    }
}

// 'kind' with weight decay wd: three steps in one batch (the lr changed
// between them, the gradients fixed), then two more one at a time with new
// gradients; clip > 0 clips the global gradient norm first.
static void opt_case(int kind, float wd, float clip) {
    struct Rng r = Rng::new(100UL + (unsigned long)kind);
    struct Vec<&FloatTensor> ps = std::Vec::new();
    struct Vec<struct MpsBuf> ws = std::Vec::new();
    struct Vec<struct MpsBuf> gs = std::Vec::new();
    unsigned long sizes[3];
    sizes[0] = 1000UL; sizes[1] = 37UL; sizes[2] = 5UL;
    float lr = kind == OPT_LION ? (float)0.003 : (float)0.01;
    struct Optimizer cpu = Optimizer::new(kind, lr);
    cpu.weightDecay = wd;
    struct MpsOptimizer gpu = MpsOptimizer::from(&cpu);
    gpu.clipNorm = clip;
    unsigned long i = 0UL;
    while (i < 3UL) {
        &FloatTensor p = FloatTensor::new_1d(sizes[i], 1);
        r.fill_normal(p, (float)0.0, (float)1.0);
        ps.push(p);
        ws.push(upload_t(p));
        gs.push(MpsBuf::alloc(sizes[i], 0));
        cpu.add(p);
        gpu.add(*ws.at(i), *gs.at(i));
        i = i + 1UL;
    }
    // gradients large enough that clipping (to 1) applies
    set_grads(&ps, &gs, &r, clip > (float)0.0 ? (float)1.0 : (float)0.5);
    mps_batch_begin();
    int s = 0;
    int ok = 1;
    while (s < 3) {
        if (clip > (float)0.0) cpu.clip_grad_norm(clip);
        cpu.step();
        ok = ok && gpu.step();
        if (clip > (float)0.0) {
            // the CPU scaled its gradients in place; the GPU scales on the fly
            // from the same unscaled ones, so put them back
            unsigned long q = 0UL;
            while (q < ps.length()) {
                &FloatTensor p = *ps.at(q);
                unsafe { gs.at(q).read((&void)(void*)p->grad, p->size * 4UL); }
                q = q + 1UL;
            }
        }
        cpu.lr = cpu.lr * (float)0.5;
        gpu.lr = gpu.lr * (float)0.5;
        s = s + 1;
    }
    mps_batch_end();
    while (s < 5) {
        set_grads(&ps, &gs, &r, (float)0.5);
        if (clip > (float)0.0) cpu.clip_grad_norm(clip);
        cpu.step();
        ok = ok && gpu.step();
        s = s + 1;
    }
    double e = ok && gpu.t == cpu.t ? 0.0 : 1.0e9;
    i = 0UL;
    while (i < ps.length()) {
        &FloatTensor p = *ps.at(i);
        double ei = 0.0;
        unsafe { ei = err_buf(*ws.at(i), (const float*)p->data, p->size); }
        if (ei > e) e = ei;
        i = i + 1UL;
    }
    char name[96];
    unsafe {
        snprintf(&name[0], 96UL, "optimizer %s wd %g%s", opt_name(kind), (double)wd, clip > (float)0.0 ? " clip" : "");
    }
    report(&name[0], e, 0.000002);
    gpu.free();
    cpu.free();
    i = 0UL;
    while (i < ps.length()) { tensor_drop(*ps.at(i)); ws.at(i).free(); gs.at(i).free(); i = i + 1UL; }
    ps.free(); ws.free(); gs.free();
}

static void opt_checks() {
    int k = OPT_SGD;
    while (k <= OPT_LION) {
        opt_case(k, (float)0.0, (float)0.0);
        opt_case(k, (float)0.05, (float)0.0);
        k = k + 1;
    }
    opt_case(OPT_ADAM, (float)0.0, (float)1.0);
    opt_case(OPT_MOMENTUM, (float)0.01, (float)1.0);

    // weight clamping (RocmOptimizer's clampWeights): SGD, then clamp on the host
    struct Rng r = Rng::new(5UL);
    &FloatTensor p = FloatTensor::new_1d(300UL, 1);
    r.fill_normal(p, (float)0.0, (float)0.2);
    struct MpsBuf w = upload_t(p);
    struct MpsBuf g = MpsBuf::alloc(300UL, 0);
    struct Vec<&FloatTensor> ps = std::Vec::new();
    struct Vec<struct MpsBuf> gs = std::Vec::new();
    ps.push(p);
    gs.push(g);
    set_grads(&ps, &gs, &r, (float)1.0);
    struct Optimizer cpu = Optimizer::new(OPT_SGD, (float)0.1);
    cpu.add(p);
    struct MpsOptimizer gpu = MpsOptimizer::new(OPT_SGD, (float)0.1);
    gpu.clampWeights = (float)0.05;
    gpu.add(w, g);
    cpu.step();
    int ok = gpu.step();
    unsafe {
        unsigned long i = 0UL;
        while (i < p->size) {
            float v = p->data[i];
            p->data[i] = v > (float)0.05 ? (float)0.05 : (v < (float)-0.05 ? (float)-0.05 : v);
            i = i + 1UL;
        }
        report("optimizer sgd clampWeights", ok ? err_buf(w, (const float*)p->data, p->size) : 1.0e9, 0.000001);
    }
    gpu.free();
    cpu.free();
    w.free(); g.free();
    gs.free(); ps.free();
    tensor_drop(p);
}

// ── Full fits: per-epoch losses vs CpuTrainer ────────────────────────────────

// The same Mlp (same seed) trained through train_traits.h's interface on the
// CPU and on the GPU, epoch by epoch; every epoch's loss and the final
// weights must agree.
static void fit_case(const char* name, const &Dataset data, int loss, unsigned long in, unsigned long hidden,
                     unsigned long out, int act, int optKind, float lr, const &TrainConfig tc, double tol) {
    struct Rng r1 = Rng::new(42UL);
    struct Rng r2 = Rng::new(42UL);
    struct Optimizer cfgOpt = Optimizer::new(optKind, lr);
    struct TrainerOptions to = TrainerOptions::default();
    to.maxBatch = 64UL;
    struct CpuTrainer<struct Mlp> c = CpuTrainer::new(Mlp::new(in, hidden, 2UL, out, act, ACT_NONE, &r1), loss, &cfgOpt);
    struct MpsTrainer<struct Mlp, struct MpsNet> g = mps_mlp_trainer_new(Mlp::new(in, hidden, 2UL, out, act, ACT_NONE, &r2),
                                                                         loss, &cfgOpt, &to);
    int ok = g.ok && c.fit_begin(data, tc) && g.fit_begin(data, tc);
    double e = 0.0;
    float first = (float)0.0;
    float last = (float)0.0;
    unsigned long ep = 0UL;
    while (ok && ep < tc.epochs) {
        struct EpochLoss lc = c.fit_epoch();
        struct EpochLoss lg = g.fit_epoch();
        ok = lc.ok && lg.ok && lc.steps == lg.steps;
        float a = lc.loss;
        double ei = err_vals(&lg.loss, &a, 1UL);
        if (ei > e) e = ei;
        if (ep == 0UL) first = lg.loss;
        last = lg.loss;
        c.set_learning_rate(c.learning_rate() * tc.lrDecay);
        g.set_learning_rate(g.learning_rate() * tc.lrDecay);
        ep = ep + 1UL;
    }
    c.fit_end();
    g.fit_end();
    ok = ok && g.sync();
    double ew = 1.0e9;
    if (ok) {
        ew = 0.0;
        unsigned long i = 0UL;
        while (i < c.host.params.length()) {
            &FloatTensor a = *c.host.params.at(i);
            &FloatTensor b = *g.host.params.at(i);
            double ei = 0.0;
            unsafe { ei = err_vals((const float*)b->data, (const float*)a->data, a->size); }
            if (ei > ew) ew = ei;
            i = i + 1UL;
        }
    } else {
        e = 1.0e9;
    }
    char buf[96];
    unsafe {
        printf("  %-12s %lu epochs, loss %.5f -> %.5f\n", name, tc.epochs, (double)first, (double)last);
        snprintf(&buf[0], 96UL, "fit %s: every epoch's loss", name);
        report(&buf[0], e, tol);
        snprintf(&buf[0], 96UL, "fit %s: trained weights", name);
        report(&buf[0], ew, tol * 4.0);
    }
    check(ok && last < first, name);
    c.free();
    c.host.free();
    g.free();
    g.host.free();
    cfgOpt.free();
}

// train_step on host batches, CPU vs GPU.
static void step_case() {
    struct Rng drng = Rng::new(8UL);
    struct Dataset sine = Dataset::sine(48UL, (float)0.0, &drng);
    struct Rng r1 = Rng::new(9UL);
    struct Rng r2 = Rng::new(9UL);
    struct Optimizer cfgOpt = Optimizer::new(OPT_ADAM, (float)0.01);
    struct TrainerOptions to = TrainerOptions::default();
    struct CpuTrainer<struct Mlp> c = CpuTrainer::new(Mlp::new(1UL, 16UL, 1UL, 1UL, ACT_TANH, ACT_NONE, &r1),
                                                      TRAIN_LOSS_MSE, &cfgOpt);
    struct MpsTrainer<struct Mlp, struct MpsNet> g = mps_mlp_trainer_new(Mlp::new(1UL, 16UL, 1UL, 1UL, ACT_TANH, ACT_NONE, &r2),
                                                                         TRAIN_LOSS_MSE, &cfgOpt, &to);
    double e = 0.0;
    int i = 0;
    while (i < 10) {
        float a = c.train_step(sine.x, sine.y);
        float b = g.train_step(sine.x, sine.y);
        double ei = b < (float)0.0 ? 1.0e9 : err_vals(&b, &a, 1UL);
        if (ei > e) e = ei;
        i = i + 1;
    }
    report("train_step: 10 losses", e, 0.00002);
    c.free(); c.host.free();
    g.free(); g.host.free();

    // what MpsTrainer refuses
    struct Rng r3 = Rng::new(9UL);
    to.precision = 1;
    struct MpsTrainer<struct Mlp, struct MpsNet> h = mps_mlp_trainer_new(Mlp::new(1UL, 16UL, 1UL, 1UL, ACT_TANH, ACT_NONE, &r3),
                                                                         TRAIN_LOSS_MSE, &cfgOpt, &to);
    struct TrainConfig tc = TrainConfig::default();
    check(!h.ok && !h.fit_begin(&sine, &tc), "precision set: the trainer doesn't build");
    h.free(); h.host.free();
    struct Rng r4 = Rng::new(9UL);
    to.precision = 0;
    to.maxBatch = 16UL;
    struct MpsTrainer<struct Mlp, struct MpsNet> k = mps_mlp_trainer_new(Mlp::new(1UL, 16UL, 1UL, 1UL, ACT_TANH, ACT_NONE, &r4),
                                                                         TRAIN_LOSS_MSE, &cfgOpt, &to);
    check(k.ok && !k.fit_begin(&sine, &tc), "batch larger than maxBatch: fit_begin fails");
    k.free(); k.host.free();
    cfgOpt.free();
    sine.free();
}

static void fit_checks() {
    struct Rng drng = Rng::new(3UL);
    struct Dataset spir = Dataset::spirals(300UL, 3UL, (float)0.1, &drng);   // 9 full batches + a 12-row tail
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 8UL;
    tc.clipNorm = (float)1.0;
    tc.lrDecay = (float)0.9;
    fit_case("ce adam", &spir, TRAIN_LOSS_CE, 2UL, 32UL, 3UL, ACT_TANH, OPT_ADAM, (float)0.01, &tc, 0.0001);

    struct Dataset sine = Dataset::sine(200UL, (float)0.05, &drng);
    tc.clipNorm = (float)0.0;
    tc.lrDecay = (float)1.0;
    tc.batchSize = 16UL;
    fit_case("mse adamw", &sine, TRAIN_LOSS_MSE, 1UL, 24UL, 1UL, ACT_GELU, OPT_ADAMW, (float)0.01, &tc, 0.0001);

    struct Dataset circ = Dataset::circles(256UL, (float)0.05, (float)0.5, &drng);
    tc.batchSize = 64UL;
    tc.epochs = 10UL;
    fit_case("bce nesterov", &circ, TRAIN_LOSS_BCE, 2UL, 16UL, 1UL, ACT_SILU, OPT_NESTEROV, (float)0.3, &tc, 0.0001);

    // the generic loop on an MpsNet directly: mps_trainer_fit vs trainer_fit
    struct Rng r1 = Rng::new(5UL);
    struct Rng r2 = Rng::new(5UL);
    struct Mlp a = Mlp::new(2UL, 32UL, 2UL, 3UL, ACT_RELU, ACT_NONE, &r1);
    struct Mlp b = Mlp::new(2UL, 32UL, 2UL, 3UL, ACT_RELU, ACT_NONE, &r2);
    struct Optimizer opt = Optimizer::new(OPT_RMSPROP, (float)0.003);
    model_add_params(&a, &opt);
    tc.batchSize = 32UL;
    tc.epochs = 5UL;
    struct TrainReport rc = module_fit(&a, TRAIN_LOSS_CE, &opt, &spir, &tc);
    struct MpsNet dn = MpsNet::from_mlp(&b, 32UL);
    struct MpsOptimizer mo = MpsOptimizer::new(OPT_RMSPROP, (float)0.003);
    dn.add_params(&mo);
    struct TrainReport rm = mps_trainer_fit(&dn, mps_net_ce_step, &mo, &spir, &tc);
    float pa[2];
    float pb[2];
    pa[0] = rc.firstEpochLoss; pa[1] = rc.lastEpochLoss;
    pb[0] = rm.firstEpochLoss; pb[1] = rm.lastEpochLoss;
    report("mps_trainer_fit vs trainer_fit (rmsprop)", rm.steps == rc.steps ? err_vals(&pb[0], &pa[0], 2UL) : 1.0e9,
           0.0001);
    dn.free(); mo.free(); opt.free();
    a.free(); b.free();
    spir.free(); sine.free(); circ.free();
}

int main() {
    if (!mps_available()) {
        unsafe { printf("nn_mps_check: no Metal device, skipped\n"); }
        return 0;
    }
    unsafe { printf("mps_gemm:\n"); }
    gemm_checks();
    unsafe { printf("layers and losses (one step vs Autograd):\n"); }
    layer_checks();
    unsafe { printf("optimizers (steps vs Optimizer::step):\n"); }
    opt_checks();
    unsafe { printf("fits (per-epoch loss vs CpuTrainer):\n"); }
    fit_checks();
    step_case();
    unsafe { printf("nn_mps: %d checks, %d failures\n", g_checks, g_fails); }
    return g_fails == 0 ? 0 : 1;
}
