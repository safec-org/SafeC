// Correctness check for std/ml/nn_rocm (device-resident training) against
// std::ml's CPU Autograd::get. Needs a ROCm GPU; run with tests/stdlib/rocm/run.sh.
#include <std/ml/nn_rocm.h>
#include <std/ml/module.h>
#include <std/ml/convnet.h>
#include <std/collections/vec.h>
#include <std/math.h>
extern int printf(const char* fmt, ...);

static int g_checks = 0;
static int g_fails = 0;
static int   g_prec = 0;                   // the nets' activations (RocmNet::set_precision)
static float g_tol = (float)0.0005;        // the gradient checks' tolerance at that precision
static float g_scale = (float)1.0;         // the fp16 loss scale (in the parameter gradients)
static float g_div = (float)1.0;           // rel_diff_ divides the GPU's values by this

static void check(int ok, const char* what) {
    g_checks = g_checks + 1;
    if (!ok) {
        g_fails = g_fails + 1;
        unsafe { printf("  FAIL %s\n", what); }
    }
}

static float absf_(float x) { return x < (float)0.0 ? -x : x; }

// max |gpu - host| / (1 + max |host|) over n values
static float rel_diff_(struct RocmBuf d, const float* host, unsigned long n) {
    struct Vec<float> g = std::vec_with_cap(n);
    unsigned long i = 0UL;
    while (i < n) { g.push((float)0.0); i = i + 1UL; }
    float m = (float)0.0;
    float scale = (float)1.0;
    if (!std::rocm_read(d, g.at(0UL), n)) { g.free(); return (float)1.0e30; }
    unsafe {
        i = 0UL;
        while (i < n) {
            float df = absf_(*g.at(i) / g_div - host[i]);
            if (df > m) m = df;
            if (absf_(host[i]) > scale) scale = absf_(host[i]);
            i = i + 1UL;
        }
    }
    g.free();
    return m / scale;
}

static &Tensor param_(unsigned long r, unsigned long c, float scale, &Rng rng) {
    &Tensor t = Tensor::new_2d(r, c, 1);
    rng.fill_uniform(t, -scale, scale);
    return t;
}

static struct RocmBuf upload_(const &Tensor t) {
    unsafe { return RocmBuf::upload((const &float)(const float*)t->data, t->size); }
}

static float read1_(struct RocmBuf b) {
    float v = (float)0.0;
    std::rocm_read(b, &v, 1UL);
    return v;
}

// ── Layer specs: one network built both on the CPU (Autograd::get) and the GPU ───
struct Spec {
    int kind;
    unsigned long inC; unsigned long inH; unsigned long inW;
    unsigned long k; unsigned long s; unsigned long p;
    int act;
    &Tensor w; &Tensor b;
};

static struct Spec lin_(unsigned long in, unsigned long out, int act, &Rng rng) {
    struct Spec s;
    s.kind = ROCM_LAYER_LINEAR; s.inC = in; s.inH = 1UL; s.inW = 1UL; s.k = 1UL; s.s = 1UL; s.p = 0UL; s.act = act;
    float sc = (float)1.0 / sqrt_f((float)in);
    s.w = param_(in, out, sc, rng);
    s.b = param_(1UL, out, sc, rng);
    return s;
}

static struct Spec conv_(int kind, unsigned long inC, unsigned long hw, unsigned long outC, unsigned long k,
                         unsigned long st, unsigned long p, int act, &Rng rng) {
    struct Spec s;
    s.kind = kind; s.inC = inC; s.inH = hw; s.inW = hw; s.k = k; s.s = st; s.p = p; s.act = act;
    float sc = (float)1.0 / sqrt_f((float)(inC * k * k));
    if (kind == ROCM_LAYER_CONV) s.w = param_(outC, inC * k * k, sc, rng);
    else s.w = param_(inC, outC * k * k, sc, rng);
    s.b = param_(1UL, outC, sc, rng);
    return s;
}

static &Tensor cpu_layer_(const &Spec s, &Tensor x) {
    &Tensor y = x;
    if (s.kind == ROCM_LAYER_LINEAR) y = tensor_add_bias(tensor_matmul(x, s.w), s.b);
    else if (s.kind == ROCM_LAYER_CONV) y = tensor_conv2d(x, s.w, s.b, s.inC, s.inH, s.inW, s.k, s.k, s.s, s.p);
    else y = tensor_conv_transpose2d(x, s.w, s.b, s.inC, s.inH, s.inW, s.k, s.k, s.s, s.p);
    return y.activate(s.act);
}

static int gpu_add_(&RocmNet n, const &Spec s) {
    if (s.kind == ROCM_LAYER_LINEAR) return n.add_linear(s.w, s.b, s.act);
    if (s.kind == ROCM_LAYER_CONV) return n.add_conv2d(s.w, s.b, s.inC, s.inH, s.inW, s.k, s.s, s.p, s.act);
    return n.add_conv_transpose2d(s.w, s.b, s.inC, s.inH, s.inW, s.k, s.s, s.p, s.act);
}

#define LOSS_CE  0
#define LOSS_MSE 1
#define LOSS_BCE 2

// Builds the net from specs, runs one forward/loss/backward on both sides
// and compares the loss, every parameter gradient and d(loss)/dx. With
// 'step' the GPU side runs the rocm_net_*_step function instead (one fused
// kernel for small fully connected nets) and dx is not compared.
static void grad_case_(const char* name, &Vec<struct Spec> specs, unsigned long rows, unsigned long inDim,
                       unsigned long outDim, int loss, unsigned long seed, int step) {
    struct Rng rng = Rng::new(seed);
    &Tensor x = Tensor::new_2d(rows, inDim, 1);
    rng.fill_normal(x, (float)0.0, (float)1.0);
    unsigned long tcols = loss == LOSS_CE ? 1UL : outDim;
    &Tensor y = Tensor::new_2d(rows, tcols, 0);
    if (loss == LOSS_CE) {
        unsafe { unsigned long r = 0UL; while (r < rows) { y->data[r] = (float)((r * 7UL) % outDim); r = r + 1UL; } }
    } else if (loss == LOSS_BCE) {
        rng.fill_uniform(y, (float)0.0, (float)1.0);
    } else {
        rng.fill_normal(y, (float)0.0, (float)1.0);
    }
    // CPU
    unsigned long i = 0UL;
    while (i < specs.length()) { specs.at(i).w.zero_grad(); specs.at(i).b.zero_grad(); i = i + 1UL; }
    &Tensor h = x;
    i = 0UL;
    while (i < specs.length()) { h = cpu_layer_(specs.at(i), h); i = i + 1UL; }
    &Tensor l = loss == LOSS_CE ? tensor_cross_entropy_loss(h, y)
              : (loss == LOSS_MSE ? tensor_mse_loss(h, y) : tensor_bce_logits_loss(h, y));
    l.backward();
    // GPU
    struct RocmNet n = RocmNet::new(rows + 3UL);
    i = 0UL;
    while (i < specs.length()) { gpu_add_(&n, specs.at(i)); i = i + 1UL; }
    check(n.ok && n.in_dim() == inDim && n.out_dim() == outDim, name);
    if (g_prec != 0) check(n.set_precision(g_prec) && n.set_loss_scale(g_scale), name);
    struct RocmBuf dx = upload_(x);
    struct RocmBuf dy = upload_(y);
    struct RocmBuf ddx = RocmBuf::alloc(rows * inDim);
    struct RocmBuf lb = RocmBuf::alloc(1UL);
    std::rocm_zero(lb);
    int ok = 1;
    struct RocmBuf out = g_prec != 0 ? n.out32 : n.layers.at(n.layers.length() - 1UL).h;
    if (step) {
        if (loss == LOSS_CE) ok = rocm_net_ce_step(&n, dx, dy, rows, lb);
        else if (loss == LOSS_MSE) ok = rocm_net_mse_step(&n, dx, dy, rows, lb);
        else ok = rocm_net_bce_step(&n, dx, dy, rows, lb);
    } else {
        out = n.forward(dx, rows);
        ok = out.dev != 0UL;
        if (loss == LOSS_CE) ok = ok && rocm_cross_entropy(out, dy, n.grad_output(), lb, rows, outDim);
        else if (loss == LOSS_MSE) ok = ok && rocm_mse(out, dy, n.grad_output(), lb, rows * outDim, (float)1.0);
        else ok = ok && rocm_bce_logits(out, dy, n.grad_output(), lb, rows * outDim, (float)1.0);
        ok = ok && n.backward(dx, rows, ddx);
    }
    check(ok, name);
    float worst = (float)0.0;
    unsafe {
        float lossErr = absf_(read1_(lb) - l->data[0]) / ((float)1.0 + absf_(l->data[0]));
        if (lossErr > worst) worst = lossErr;
        float e = rel_diff_(out, (const float*)h->data, rows * outDim);
        if (e > worst) worst = e;
        g_div = g_scale;
        i = 0UL;
        while (i < specs.length()) {
            &RocmLayer L = n.layers.at(i);
            &Spec s = specs.at(i);
            e = rel_diff_(L.gw, (const float*)s.w->grad, s.w->size);
            if (e > worst) worst = e;
            e = rel_diff_(L.gb, (const float*)s.b->grad, s.b->size);
            if (e > worst) worst = e;
            i = i + 1UL;
        }
        g_div = (float)1.0;
        if (!step) {
            e = rel_diff_(ddx, (const float*)x->grad, rows * inDim);
            if (e > worst) worst = e;
        }
        printf("  %-38s%s max rel err %.2e\n", name, step ? " (step)" : "       ", (double)worst);
    }
    check(worst < g_tol, name);
    n.free();
    dx.free(); dy.free(); ddx.free(); lb.free();
}

static void grad_tests_() {
    struct Rng rng = Rng::new(11UL);
    struct Vec<struct Spec> a = std::vec_new();
    a.push(lin_(6UL, 16UL, ACT_TANH, &rng));
    a.push(lin_(16UL, 16UL, ACT_RELU, &rng));
    a.push(lin_(16UL, 4UL, ACT_NONE, &rng));
    grad_case_("mlp tanh/relu, cross-entropy", &a, 9UL, 6UL, 4UL, LOSS_CE, 1UL, 0);
    grad_case_("mlp tanh/relu, cross-entropy", &a, 9UL, 6UL, 4UL, LOSS_CE, 1UL, 1);
    a.free();

    struct Vec<struct Spec> b = std::vec_new();
    b.push(lin_(5UL, 12UL, ACT_GELU, &rng));
    b.push(lin_(12UL, 12UL, ACT_SILU, &rng));
    b.push(lin_(12UL, 7UL, ACT_SIGMOID, &rng));
    grad_case_("mlp gelu/silu/sigmoid, mse", &b, 13UL, 5UL, 7UL, LOSS_MSE, 2UL, 0);
    grad_case_("mlp gelu/silu/sigmoid, mse", &b, 13UL, 5UL, 7UL, LOSS_MSE, 2UL, 1);
    b.free();

    // multi-channel input and output: both layout conversions at the edges
    struct Vec<struct Spec> c = std::vec_new();
    c.push(conv_(ROCM_LAYER_CONV, 2UL, 6UL, 4UL, 3UL, 2UL, 1UL, ACT_RELU, &rng));            // 6x6 -> 3x3
    c.push(conv_(ROCM_LAYER_CONVT, 4UL, 3UL, 3UL, 4UL, 2UL, 1UL, ACT_LEAKY, &rng));          // 3x3 -> 6x6
    grad_case_("conv -> convT, mse", &c, 5UL, 72UL, 108UL, LOSS_MSE, 3UL, 0);
    grad_case_("conv -> convT, mse", &c, 5UL, 72UL, 108UL, LOSS_MSE, 3UL, 1);
    c.free();

    // fully connected -> convolution -> transposed convolution
    struct Vec<struct Spec> d = std::vec_new();
    d.push(lin_(10UL, 32UL, ACT_RELU, &rng));
    d.push(conv_(ROCM_LAYER_CONV, 2UL, 4UL, 3UL, 3UL, 1UL, 1UL, ACT_GELU, &rng));            // 4x4 -> 4x4
    d.push(conv_(ROCM_LAYER_CONVT, 3UL, 4UL, 2UL, 2UL, 2UL, 0UL, ACT_NONE, &rng));           // 4x4 -> 8x8
    grad_case_("linear -> conv -> convT, bce", &d, 6UL, 10UL, 128UL, LOSS_BCE, 4UL, 0);
    grad_case_("linear -> conv -> convT, bce", &d, 6UL, 10UL, 128UL, LOSS_BCE, 4UL, 1);
    d.free();

    // convolution -> fully connected (a classifier)
    struct Vec<struct Spec> e = std::vec_new();
    e.push(conv_(ROCM_LAYER_CONV, 3UL, 7UL, 4UL, 3UL, 2UL, 1UL, ACT_TANH, &rng));            // 7x7 -> 4x4
    e.push(conv_(ROCM_LAYER_CONV, 4UL, 4UL, 5UL, 3UL, 1UL, 0UL, ACT_SILU, &rng));            // 4x4 -> 2x2
    e.push(lin_(20UL, 3UL, ACT_NONE, &rng));
    grad_case_("conv -> conv -> linear, cross-entropy", &e, 7UL, 147UL, 3UL, LOSS_CE, 5UL, 0);
    grad_case_("conv -> conv -> linear, cross-entropy", &e, 7UL, 147UL, 3UL, LOSS_CE, 5UL, 1);
    e.free();

    // single-channel convolutions (no layout conversion): conv_ae's shape
    struct Vec<struct Spec> f = std::vec_new();
    f.push(conv_(ROCM_LAYER_CONV, 1UL, 8UL, 4UL, 3UL, 2UL, 1UL, ACT_RELU, &rng));            // 8x8 -> 4x4
    f.push(conv_(ROCM_LAYER_CONVT, 4UL, 4UL, 1UL, 4UL, 2UL, 1UL, ACT_NONE, &rng));           // 4x4 -> 8x8
    grad_case_("1-channel conv -> convT, bce", &f, 40UL, 64UL, 64UL, LOSS_BCE, 6UL, 0);
    grad_case_("1-channel conv -> convT, bce", &f, 40UL, 64UL, 64UL, LOSS_BCE, 6UL, 1);
    f.free();

    // a weight gradient over batch x plane = 20 x 1024 values: split across blocks (split-K)
    struct Vec<struct Spec> g2 = std::vec_new();
    g2.push(conv_(ROCM_LAYER_CONV, 1UL, 32UL, 4UL, 3UL, 1UL, 1UL, ACT_RELU, &rng));           // 32x32 -> 32x32
    g2.push(conv_(ROCM_LAYER_CONV, 4UL, 32UL, 2UL, 3UL, 2UL, 1UL, ACT_NONE, &rng));           // -> 16x16
    grad_case_("conv 32x32, split-K weight gradient", &g2, 20UL, 1024UL, 512UL, LOSS_MSE, 7UL, 0);
    g2.free();
}

// ── ConvNets (every layer kind RocmNet::from_convnet takes) ─────────────────
// One forward / loss / backward of a convnet.h ConvNet on the CPU and of its
// RocmNet::from_convnet copy; compares the output, every parameter
// gradient, d(loss)/dx and the batch norms' running statistics. With
// 'twice' the CPU runs two batches through separate passes (summing their
// losses) and the GPU both stacked as one batch with segments = 2, as
// gan.h's discriminator step does.
static float max_rel_tensor_(struct RocmBuf d, const &Tensor t) {
    unsafe { return rel_diff_(d, (const float*)t->data, t->size); }
}

// The net's buffers (RocmNet::download_stats) against a ConvNet's.
static float buffers_diff_(&RocmNet n, const &Vec<&Tensor> cpu) {
    struct Vec<&Tensor> copy = std::vec_new();
    unsigned long i = 0UL;
    while (i < cpu.length()) {
        unsigned long sz;
        unsafe { sz = (*cpu.at(i))->size; }
        copy.push(Tensor::new_1d(sz, 0));
        i = i + 1UL;
    }
    float e = (float)1.0e30;
    if (n.download_stats(&copy)) {
        e = (float)0.0;
        i = 0UL;
        while (i < cpu.length()) {
            &Tensor a = *copy.at(i);
            &Tensor b = *cpu.at(i);
            unsafe {
                unsigned long j = 0UL;
                while (j < a->size) {
                    float d = absf_(a->data[j] - b->data[j]) / ((float)1.0 + absf_(b->data[j]));
                    if (d > e) e = d;
                    j = j + 1UL;
                }
            }
            i = i + 1UL;
        }
    }
    copy.free();
    return e;
}

static void convnet_case_(const char* name, &ConvNet c, unsigned long rows, int loss, unsigned long seed, int twice) {
    struct Rng rng = Rng::new(seed);
    unsigned long inDim = c.in_dim();
    unsigned long outDim = c.out_dim();
    unsigned long total = twice ? 2UL * rows : rows;
    &Tensor x = Tensor::new_2d(total, inDim, 1);
    rng.fill_normal(x, (float)0.0, (float)1.0);
    unsigned long tcols = loss == LOSS_CE ? 1UL : outDim;
    &Tensor y = Tensor::new_2d(total, tcols, 0);
    if (loss == LOSS_CE) {
        unsafe { unsigned long r = 0UL; while (r < total) { y->data[r] = (float)((r * 7UL) % outDim); r = r + 1UL; } }
    } else {
        rng.fill_normal(y, (float)0.0, (float)1.0);
    }
    // the GPU copy first: the CPU pass moves the running statistics
    struct RocmNet n = RocmNet::from_convnet(c, total);
    check(n.ok && n.in_dim() == inDim && n.out_dim() == outDim, name);
    if (g_prec != 0) check(n.set_precision(g_prec) && n.set_loss_scale(g_scale), name);
    unsigned long i = 0UL;
    while (i < c.params.length()) { (*c.params.at(i)).zero_grad(); i = i + 1UL; }
    &Tensor out = Tensor::new_2d(total, outDim, 0);  // the CPU output
    &Tensor gx = Tensor::new_2d(total, inDim, 0);    // d(loss)/dx on the CPU
    float lval = (float)0.0;
    if (twice) {
        // rows [0, m) and [m, 2m) as two passes
        &Tensor x1 = Tensor::new_2d(rows, inDim, 1);
        &Tensor x2 = Tensor::new_2d(rows, inDim, 1);
        &Tensor y1 = Tensor::new_2d(rows, tcols, 0);
        &Tensor y2 = Tensor::new_2d(rows, tcols, 0);
        unsafe {
            unsigned long k = 0UL;
            while (k < rows * inDim) { x1->data[k] = x->data[k]; x2->data[k] = x->data[rows * inDim + k]; k = k + 1UL; }
            k = 0UL;
            while (k < rows * tcols) { y1->data[k] = y->data[k]; y2->data[k] = y->data[rows * tcols + k]; k = k + 1UL; }
        }
        &Tensor o1 = c.forward(x1);
        &Tensor o2 = c.forward(x2);
        &Tensor l2 = tensor_add(tensor_mse_loss(o1, y1), tensor_mse_loss(o2, y2));
        l2.backward();
        unsafe {
            lval = l2->data[0];
            unsigned long k = 0UL;
            while (k < rows * inDim) { gx->data[k] = x1->grad[k]; gx->data[rows * inDim + k] = x2->grad[k]; k = k + 1UL; }
            k = 0UL;
            while (k < rows * outDim) { out->data[k] = o1->data[k]; out->data[rows * outDim + k] = o2->data[k]; k = k + 1UL; }
        }
        n.segments = 2UL;
    } else {
        &Tensor oc = c.forward(x);
        &Tensor l1 = loss == LOSS_CE ? tensor_cross_entropy_loss(oc, y) : tensor_mse_loss(oc, y);
        l1.backward();
        unsafe {
            lval = l1->data[0];
            unsigned long k = 0UL;
            while (k < total * inDim) { gx->data[k] = x->grad[k]; k = k + 1UL; }
            k = 0UL;
            while (k < total * outDim) { out->data[k] = oc->data[k]; k = k + 1UL; }
        }
    }
    struct RocmBuf dx = upload_(x);
    struct RocmBuf dy = upload_(y);
    struct RocmBuf ddx = RocmBuf::alloc(total * inDim);
    struct RocmBuf lb = RocmBuf::alloc(1UL);
    std::rocm_zero(lb);
    struct RocmBuf o = n.forward(dx, total);
    int ok = o.dev != 0UL;
    if (loss == LOSS_CE) ok = ok && rocm_cross_entropy(o, dy, n.grad_output(), lb, total, outDim);
    else ok = ok && rocm_mse(o, dy, n.grad_output(), lb, total * outDim, twice ? (float)2.0 : (float)1.0);
    ok = ok && n.backward(dx, total, ddx);
    check(ok, name);
    float worst = (float)0.0;
    unsafe {
        float e = absf_(read1_(lb) - lval) / ((float)1.0 + absf_(lval));
        if (e > worst) worst = e;
        e = rel_diff_(o, (const float*)out->data, total * outDim);
        if (e > worst) worst = e;
        unsigned long k = 0UL;
        g_div = g_scale;
        i = 0UL;
        while (i < n.layers.length()) {
            &RocmLayer L = n.layers.at(i);
            if (L.w.dev != 0UL) {
                e = rel_diff_(L.gw, (const float*)(*c.params.at(k))->grad, L.gw.len);
                if (e > worst) worst = e;
                e = rel_diff_(L.gb, (const float*)(*c.params.at(k + 1UL))->grad, L.gb.len);
                if (e > worst) worst = e;
                k = k + 2UL;
            }
            i = i + 1UL;
        }
        g_div = (float)1.0;
        // every buffer (batch norm statistics, spectral norms' u), in order
        e = buffers_diff_(&n, &c.buffers);
        if (e > worst) worst = e;
        e = rel_diff_(ddx, (const float*)gx->data, total * inDim);
        if (e > worst) worst = e;
        printf("  %-45s max rel err %.2e\n", name, (double)worst);
    }
    check(worst < g_tol, name);
    n.free();
    dx.free(); dy.free(); ddx.free(); lb.free();
}

static int nr_fill_test_(struct RocmBuf b, unsigned long n) {
    struct Vec<float> ones = std::vec_with_cap(n);
    unsigned long i = 0UL;
    while (i < n) { ones.push((float)1.0); i = i + 1UL; }
    int ok = std::rocm_write(b, ones.at(0UL), n) && std::rocm_sync();
    ones.free();
    return ok;
}

// Dropout in training: every value 0 or x / (1 - p), about p of the units
// dropped (whole planes for dropout2d), the gradient the same mask.
static void dropout_props_(int byChannel) {
    unsigned long rows = 64UL;
    unsigned long C = 4UL;
    unsigned long P = 25UL;
    float p = (float)0.3;
    struct RocmNet n = RocmNet::new(rows);
    n.set_input(C, 5UL, 5UL);
    n.add_dropout(p, byChannel, 1234UL);
    struct Rng rng = Rng::new(50UL);
    &Tensor x = Tensor::new_2d(rows, C * P, 0);
    rng.fill_uniform(x, (float)0.5, (float)1.5);
    struct RocmBuf dx = upload_(x);
    struct RocmBuf ddx = RocmBuf::alloc(rows * C * P);
    struct RocmBuf o = n.forward(dx, rows);
    int ok = o.dev != 0UL && nr_fill_test_(n.grad_output(), rows * C * P) &&
             n.backward(dx, rows, ddx);
    struct Vec<float> out = std::vec_with_cap(rows * C * P);
    struct Vec<float> grad = std::vec_with_cap(rows * C * P);
    unsigned long i = 0UL;
    while (i < rows * C * P) { out.push((float)0.0); grad.push((float)0.0); i = i + 1UL; }
    ok = ok && std::rocm_read(o, out.at(0UL), rows * C * P) && std::rocm_read(ddx, grad.at(0UL), rows * C * P);
    float keep = (float)1.0 / ((float)1.0 - p);
    unsigned long zeros = 0UL;
    int good = 1;
    i = 0UL;
    unsafe {
        while (i < rows * C * P) {
            float xv = x->data[i];
            float ov = *out.at(i);
            float m = *grad.at(i);
            if (ov == (float)0.0) {
                zeros = zeros + 1UL;
                if (m != (float)0.0) good = 0;
            } else if (absf_(ov - xv * keep) > (float)0.00001 || m != keep) {
                good = 0;
            }
            // dropout2d: a plane is all dropped or all kept
            if (byChannel && i % P != 0UL && ((*out.at(i) == (float)0.0) != (*out.at(i - i % P) == (float)0.0))) good = 0;
            i = i + 1UL;
        }
    }
    float frac = (float)zeros / (float)(rows * C * P);
    unsafe { printf("  dropout%s (training): %.3f dropped (p %.2f)\n", byChannel ? "2d" : "", (double)frac, (double)p); }
    check(ok && good && frac > p - (float)0.06 && frac < p + (float)0.06, byChannel ? "dropout2d properties" : "dropout properties");
    out.free(); grad.free();
    n.free();
    dx.free(); ddx.free();
}

static void convnet_tests_() {
    struct Rng rng = Rng::new(31UL);
    // grouped rectangular dilated conv, batch norm, ceil-mode max pool, group
    // norm, grouped transposed conv with output padding, avg pool without the
    // padding, global avg pool, dense
    struct ConvNet a = ConvNet::new(4UL, 9UL, 8UL);
    struct Conv2dSpec s1 = Conv2dSpec::new(3UL, 1UL, 1UL);
    s1.kW = 2UL; s1.strideH = 2UL; s1.padW = 0UL; s1.dilationW = 2UL; s1.groups = 2UL;
    a.conv_ex(6UL, &s1, ACT_NONE, &rng);
    a.batchnorm(ACT_RELU);
    struct Pool2dSpec p1 = Pool2dSpec::new(3UL, 2UL, 1UL);
    p1.ceilMode = 1;
    a.maxpool_ex(&p1);
    a.groupnorm(2UL, ACT_SILU);
    struct Conv2dSpec s2 = Conv2dSpec::new(3UL, 2UL, 1UL);
    s2.groups = 3UL; s2.outPadH = 1UL;
    a.deconv_ex(6UL, &s2, ACT_TANH, &rng);
    struct Pool2dSpec p2 = Pool2dSpec::new(2UL, 2UL, 1UL);
    p2.countIncludePad = 0;
    a.avgpool_ex(&p2);
    a.global_avgpool();
    a.dense(5UL, ACT_NONE, &rng);
    convnet_case_("convnet: groups/dilation/pools/norms", &a, 6UL, LOSS_CE, 41UL, 0);
    a.free();

    // dense -> reshape -> upsample -> conv -> layer norm -> global max pool -> dense
    struct ConvNet b = ConvNet::new(1UL, 1UL, 10UL);
    b.dense(3UL * 4UL * 4UL, ACT_LEAKY, &rng);
    b.reshape(3UL, 4UL, 4UL);
    b.upsample(2UL);
    b.conv(5UL, 3UL, 1UL, 1UL, ACT_NONE, &rng);
    b.layernorm(ACT_GELU);
    b.instancenorm(ACT_SIGMOID);
    // (at 16 bits a max over near-equal values can pick another element)
    if (g_prec == 0) b.global_maxpool();
    else b.global_avgpool();
    b.dense(3UL, ACT_NONE, &rng);
    convnet_case_("convnet: dense/reshape/upsample/norms", &b, 7UL, LOSS_MSE, 42UL, 0);
    b.free();

    // a DCGAN-style discriminator with batch norm: two stacked batches
    struct ConvNet d = ConvNet::new(1UL, 8UL, 8UL);
    d.conv(4UL, 4UL, 2UL, 1UL, ACT_LEAKY, &rng);
    d.conv(8UL, 4UL, 2UL, 1UL, ACT_NONE, &rng);
    d.batchnorm(ACT_LEAKY);
    d.conv(1UL, 2UL, 1UL, 0UL, ACT_NONE, &rng);
    convnet_case_("convnet: batch norm, two stacked batches", &d, 5UL, LOSS_MSE, 43UL, 1);
    // ... and in inference mode (running statistics)
    d.set_training(0);
    convnet_case_("convnet: batch norm, inference mode", &d, 5UL, LOSS_MSE, 44UL, 0);
    d.free();
    // residual blocks (identity and projection shortcuts, batch norm) -> global pool -> dense
    struct ConvNet r = ConvNet::new(3UL, 8UL, 8UL);
    r.conv(4UL, 3UL, 1UL, 1UL, ACT_RELU, &rng);
    r.residual_block(4UL, 1UL, 1, ACT_RELU, &rng);     // identity shortcut
    r.residual_block(8UL, 2UL, 1, ACT_RELU, &rng);     // projection, stride 2
    r.residual_block(8UL, 1UL, 0, ACT_LEAKY, &rng);    // no batch norm
    r.global_avgpool();
    r.dense(4UL, ACT_NONE, &rng);
    convnet_case_("convnet: residual blocks", &r, 6UL, LOSS_CE, 45UL, 0);
    r.free();

    // U-Net-style: saves reused twice, concat skips, bilinear / bicubic / nearest resizing
    struct ConvNet u = ConvNet::new(2UL, 8UL, 8UL);
    u.conv(4UL, 3UL, 1UL, 1UL, ACT_RELU, &rng);
    unsigned long sv0 = u.save();                          // 4 x 8 x 8
    u.maxpool(2UL, 2UL, 0UL);                           // 4 x 4 x 4
    u.conv(6UL, 3UL, 1UL, 1UL, ACT_TANH, &rng);
    unsigned long sv1 = u.save();                          // 6 x 4 x 4
    u.resize(3UL, 3UL, RESIZE_BICUBIC, 1);              // 6 x 3 x 3 (down)
    u.resize(4UL, 4UL, RESIZE_NEAREST, 0);              // 6 x 4 x 4
    u.concat_skip(sv1);                                  // 12 x 4 x 4
    u.upsample_bilinear(2UL);                           // 12 x 8 x 8
    u.concat_skip(sv0);                                  // 16 x 8 x 8
    u.add_skip(sv0, ACT_NONE, &rng);                     // + projection 4 -> 16
    u.resize(11UL, 7UL, RESIZE_BICUBIC, 0);             // 16 x 11 x 7
    u.conv(1UL, 3UL, 1UL, 1UL, ACT_NONE, &rng);
    convnet_case_("convnet: concat skips, resizes", &u, 4UL, LOSS_MSE, 46UL, 0);
    u.free();

    // spectral norm on conv, deconv and dense weights (training, then inference)
    struct ConvNet sn = ConvNet::new(2UL, 6UL, 6UL);
    sn.conv(5UL, 3UL, 2UL, 1UL, ACT_LEAKY, &rng);
    sn.spectral_norm(&rng);
    sn.deconv(3UL, 2UL, 2UL, 0UL, ACT_LEAKY, &rng);
    sn.spectral_norm(&rng);
    sn.dense(4UL, ACT_NONE, &rng);
    sn.spectral_norm(&rng);
    convnet_case_("convnet: spectral norm (training)", &sn, 5UL, LOSS_CE, 47UL, 0);
    sn.set_training(0);
    convnet_case_("convnet: spectral norm (inference)", &sn, 5UL, LOSS_CE, 48UL, 0);
    sn.free();

    // dropout in inference passes values through (the CPU's masks come from its own Rng)
    struct ConvNet dr = ConvNet::new(3UL, 4UL, 4UL);
    dr.conv(4UL, 3UL, 1UL, 1UL, ACT_RELU, &rng);
    dr.dropout((float)0.3, 1, &rng);
    dr.dense(3UL, ACT_NONE, &rng);
    dr.set_training(0);
    convnet_case_("convnet: dropout (inference)", &dr, 5UL, LOSS_CE, 49UL, 0);
    dr.free();
    dropout_props_(0);
    dropout_props_(1);
}

// ── Mixed precision: a small ResNet trained in fp32 and in 16 bits ──────────
static struct ConvNet half_net_(unsigned long seed) {
    struct Rng rng = Rng::new(seed);
    struct ConvNet c = ConvNet::new(1UL, 16UL, 16UL);
    c.conv(8UL, 3UL, 1UL, 1UL, ACT_NONE, &rng);
    c.batchnorm(ACT_RELU);
    c.residual_block(8UL, 1UL, 1, ACT_RELU, &rng);
    c.residual_block(16UL, 2UL, 1, ACT_RELU, &rng);
    c.global_avgpool();
    c.dense(4UL, ACT_NONE, &rng);
    return c;
}

// The 16-bit run's losses track the fp32 run's. scale0 > 0: an fp16 loss
// scale that overflows at first (steps skipped, the scale backs off).
static void half_train_case_(const char* name, int prec, float scale0) {
    struct Rng drng = Rng::new(5UL);
    struct Dataset data = Dataset::shapes(512UL, 16UL, &drng);
    struct ConvNet c1 = half_net_(9UL);
    struct ConvNet c2 = half_net_(9UL);
    struct RocmNet a = RocmNet::from_convnet(&c1, 64UL);
    struct RocmNet b = RocmNet::from_convnet(&c2, 64UL);
    int ok = b.set_precision(prec);
    if (scale0 > (float)0.0) ok = ok && b.set_loss_scale(scale0);
    struct RocmOptimizer oa = RocmOptimizer::new(OPT_ADAM, (float)0.003);
    struct RocmOptimizer ob = RocmOptimizer::new(OPT_ADAM, (float)0.003);
    ok = ok && a.add_params(&oa) && b.add_params(&ob);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 5UL;
    tc.batchSize = 64UL;
    struct TrainReport ra = rocm_trainer_fit(&a, rocm_net_ce_step, &oa, &data, &tc);
    struct TrainReport rb = rocm_trainer_fit(&b, rocm_net_ce_step, &ob, &data, &tc);
    float s = b.loss_scale();
    unsafe { printf("  %-38s loss fp32 %.4f -> %.4f, 16-bit %.4f -> %.4f, loss scale %g\n", name,
                    (double)ra.firstEpochLoss, (double)ra.lastEpochLoss, (double)rb.firstEpochLoss,
                    (double)rb.lastEpochLoss, (double)s); }
    ok = ok && rb.steps == ra.steps && rb.lastEpochLoss < (float)0.5 * rb.firstEpochLoss;
    // (the overflowing run skips its first steps)
    if (scale0 == (float)0.0) ok = ok && absf_(rb.lastEpochLoss - ra.lastEpochLoss) < (float)0.1 * ra.firstEpochLoss;
    if (prec == ROCM_FP16) ok = ok && (scale0 > (float)0.0 ? s < scale0 : s == (float)65536.0);
    check(ok, name);
    oa.free(); ob.free();
    a.free(); b.free();
    c1.free(); c2.free();
    data.free();
}

static void half_tests_(int prec) {
    g_prec = prec;
    g_tol = prec == ROCM_FP16 ? (float)0.02 : (float)0.1;
    g_scale = prec == ROCM_FP16 ? (float)1024.0 : (float)1.0;   // fp16: keeps small gradients out of the subnormals
    unsafe { printf("activations: %s\n", prec == ROCM_FP16 ? "fp16" : "bf16"); }
    grad_tests_();
    convnet_tests_();
    g_prec = 0;
    g_tol = (float)0.0005;
    g_scale = (float)1.0;
    half_train_case_(prec == ROCM_FP16 ? "train fp16" : "train bf16", prec, (float)0.0);
    if (prec == ROCM_FP16) half_train_case_("train fp16, overflowing loss scale", prec, (float)16777216.0);
}

// ── Optimizers: every kind against Optimizer::step, eager and replayed ───────
static void opt_case_(int kind, float clip, int graph) {
    unsigned long n = 1000UL;
    struct Rng rng = Rng::new(20UL + (unsigned long)kind);
    &Tensor w = Tensor::new_1d(n, 1);
    rng.fill_normal(w, (float)0.0, (float)1.0);
    w.sum().backward();   // allocates w's gradient, overwritten below
    &Tensor g = Tensor::new_1d(n, 0);
    rng.fill_normal(g, (float)0.0, (float)1.0);
    struct RocmBuf dw = upload_(w);
    struct RocmBuf dg = upload_(g);
    struct Optimizer o = Optimizer::new(kind, (float)0.01);
    o.weightDecay = (float)0.05;
    o.add(w);
    struct RocmOptimizer d = RocmOptimizer::new(kind, (float)0.01);
    d.weightDecay = (float)0.05;
    d.clipNorm = clip;
    d.add(dw, dg);
    struct RocmGraph gr;
    gr.exec = 0UL;
    if (graph) {
        d.sync();
        std::rocm_capture_begin();
        d.step();
        gr = std::rocm_capture_end();
        check(gr.exec != 0UL, "optimizer graph capture");
    }
    int s = 0;
    while (s < 4) {
        unsafe {
            unsigned long i = 0UL;
            while (i < n) { w->grad[i] = g->data[i]; i = i + 1UL; }
        }
        if (clip > (float)0.0) o.clip_grad_norm(clip);
        o.step();
        if (graph) std::rocm_graph_launch(gr);
        else d.step();
        s = s + 1;
    }
    float e;
    unsafe { e = rel_diff_(dw, (const float*)w->data, n); }
    unsafe { printf("  optimizer kind %d%s%s  max rel err %.2e\n", kind, clip > (float)0.0 ? " clip" : "",
                    graph ? " graph" : "", (double)e); }
    check(d.ok && e < (float)0.00001, "optimizer step");
    gr.free();
    d.free();
    o.free();
    dw.free(); dg.free();
}

static void opt_tests_() {
    int k = OPT_SGD;
    while (k <= OPT_LION) { opt_case_(k, (float)0.0, 0); k = k + 1; }
    opt_case_(OPT_ADAM, (float)5.0, 0);
    opt_case_(OPT_ADAM, (float)0.0, 1);
    opt_case_(OPT_ADAMW, (float)5.0, 1);
}

// ── rocm_trainer_fit against trainer_fit ─────────────────────────────────────
static &Tensor clf_loss_(&Mlp m, const &Tensor x, const &Tensor y) {
    return tensor_cross_entropy_loss(m.forward(x), y);
}

static void trainer_test_() {
    struct Rng drng = Rng::new(1UL);
    struct Dataset data = Dataset::xor(250UL, (float)0.0, &drng);   // 7 full batches + a 26-row tail
    struct Rng r1 = Rng::new(3UL);
    struct Mlp cpu = Mlp::new(2UL, 16UL, 2UL, 2UL, ACT_RELU, ACT_NONE, &r1);
    struct Rng r2 = Rng::new(3UL);
    struct Mlp host = Mlp::new(2UL, 16UL, 2UL, 2UL, ACT_RELU, ACT_NONE, &r2);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 6UL;
    tc.clipNorm = (float)1.0;
    tc.lrDecay = (float)0.9;
    struct Optimizer o = Optimizer::new(OPT_ADAM, (float)0.01);
    model_add_params(&cpu, &o);
    struct TrainReport rc = trainer_fit(&cpu, clf_loss_, &o, &data, &tc);
    struct RocmNet net = RocmNet::from_mlp(&host, 32UL);
    struct RocmOptimizer d = RocmOptimizer::new(OPT_ADAM, (float)0.01);
    net.add_params(&d);
    struct TrainReport rg = rocm_trainer_fit(&net, rocm_net_ce_step, &d, &data, &tc);
    check(rg.steps == rc.steps && rg.epochs == rc.epochs && d.t == rc.steps, "trainer step count");
    check(net.download(&host.params), "trainer download");
    float worst = (float)0.0;
    unsigned long i = 0UL;
    while (i < host.params.length()) {
        &Tensor a = *host.params.at(i);
        &Tensor b = *cpu.params.at(i);
        unsafe {
            unsigned long j = 0UL;
            while (j < a->size) { float df = absf_(a->data[j] - b->data[j]); if (df > worst) worst = df; j = j + 1UL; }
        }
        i = i + 1UL;
    }
    float le = absf_(rg.lastEpochLoss - rc.lastEpochLoss);
    float fe = absf_(rg.firstEpochLoss - rc.firstEpochLoss);
    unsafe { printf("  trainer: loss cpu %.5f / %.5f gpu %.5f / %.5f, max weight diff %.2e\n",
                    (double)rc.firstEpochLoss, (double)rc.lastEpochLoss, (double)rg.firstEpochLoss,
                    (double)rg.lastEpochLoss, (double)worst); }
    check(fe < (float)0.0005 && le < (float)0.001 && worst < (float)0.001, "trainer matches trainer_fit");
    d.free();
    o.free();
    net.free();
    cpu.free();
    host.free();
    data.free();
}

// ── VAE: trains, and agrees with the CPU VAE's loss level ───────────────────
static void vae_case_(unsigned long classes) {
    struct Rng drng = Rng::new(9UL);
    struct Dataset data = Dataset::shapes(256UL, 8UL, &drng);   // 4 shape classes
    struct VaeConfig cfg = VaeConfig::default(4UL);
    cfg.hidden = 64UL;
    cfg.classes = classes;
    struct Vae cpu = Vae::new(64UL, &cfg);
    struct Vae host = Vae::new(64UL, &cfg);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 15UL;
    struct Optimizer o = Optimizer::new(OPT_ADAM, (float)0.003);
    model_add_params(&cpu, &o);
    struct TrainReport rc = model_fit(&cpu, &o, &data, &tc);
    struct RocmVae dv = RocmVae::new(&host, 32UL);
    check(dv.ok, "RocmVae::new");
    struct RocmOptimizer d = RocmOptimizer::new(OPT_ADAM, (float)0.003);
    dv.add_params(&d);
    struct TrainReport rg = rocm_trainer_fit(&dv, rocm_vae_step, &d, &data, &tc);
    unsafe { printf("  %s: loss cpu %.3f -> %.3f, gpu %.3f -> %.3f\n", classes > 0UL ? "cvae" : "vae", (double)rc.firstEpochLoss,
                    (double)rc.lastEpochLoss, (double)rg.firstEpochLoss, (double)rg.lastEpochLoss); }
    check(rg.steps == rc.steps, "vae steps");
    check(rg.lastEpochLoss < rg.firstEpochLoss * (float)0.8, "vae loss falls");
    check(absf_(rg.firstEpochLoss - rc.firstEpochLoss) < (float)0.03 * rc.firstEpochLoss, "vae first epoch ~ cpu");
    check(absf_(rg.lastEpochLoss - rc.lastEpochLoss) < (float)0.1 * rc.lastEpochLoss, "vae last epoch ~ cpu");
    check(dv.download(&host), "vae download");
    d.free();
    o.free();
    dv.free();
    cpu.free();
    host.free();
    data.free();
}

// ── Convolutional autoencoders ───────────────────────────────────────────────
static &Tensor cae_loss_(&ConvAutoencoder a, const &Tensor x, const &Tensor y) { return a.loss(x, y); }

static struct ConvAutoencoder cae_build_() {
    struct Rng rng = Rng::new(61UL);
    struct ConvNet enc = ConvNet::new(1UL, 8UL, 8UL);
    enc.conv(4UL, 3UL, 2UL, 1UL, ACT_RELU, &rng);     // 4 x 4 x 4
    enc.conv(8UL, 3UL, 2UL, 1UL, ACT_RELU, &rng);     // 8 x 2 x 2
    struct ConvNet dec = ConvNet::new(8UL, 2UL, 2UL);
    dec.deconv(4UL, 4UL, 2UL, 1UL, ACT_RELU, &rng);   // 4 x 4 x 4
    dec.upsample_bilinear(2UL);                       // 4 x 8 x 8
    dec.conv(1UL, 3UL, 1UL, 1UL, ACT_NONE, &rng);     // 1 x 8 x 8
    return ConvAutoencoder::new(enc, dec, RECON_BCE);
}

static void conv_ae_test_() {
    struct Rng drng = Rng::new(9UL);
    struct Dataset data = Dataset::shapes(250UL, 8UL, &drng);
    struct ConvAutoencoder cpu = cae_build_();
    struct ConvAutoencoder host = cae_build_();
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 4UL;
    struct Optimizer o = Optimizer::new(OPT_ADAM, (float)0.003);
    model_add_params(&cpu, &o);
    struct TrainReport rc = trainer_fit(&cpu, cae_loss_, &o, &data, &tc);
    struct RocmConvAe dev = RocmConvAe::new(&host, 32UL);
    check(dev.ok, "RocmConvAe::new");
    struct RocmOptimizer d = RocmOptimizer::new(OPT_ADAM, (float)0.003);
    dev.add_params(&d);
    struct TrainReport rg = rocm_trainer_fit(&dev, rocm_conv_autoencoder_step, &d, &data, &tc);
    check(dev.download(&host), "conv autoencoder download");
    float e = params_diff_(&host.enc.params, &cpu.enc.params);
    float e2 = params_diff_(&host.dec.params, &cpu.dec.params);
    if (e2 > e) e = e2;
    unsafe { printf("  conv autoencoder: loss cpu %.5f -> %.5f gpu %.5f -> %.5f, weights %.2e\n",
                    (double)rc.firstEpochLoss, (double)rc.lastEpochLoss, (double)rg.firstEpochLoss,
                    (double)rg.lastEpochLoss, (double)e); }
    check(rg.steps == rc.steps && absf_(rg.lastEpochLoss - rc.lastEpochLoss) < (float)0.0005 && e < (float)0.001,
          "conv autoencoder matches trainer_fit");
    d.free();
    o.free();
    dev.free();
    cpu.free();
    host.free();
    data.free();
}

static struct ConvVae cvae_build_(const &VaeConfig cfg) {
    struct Rng rng = Rng::new(62UL);
    unsigned long c = cfg.classes;
    struct ConvNet enc = ConvNet::new(1UL + c, 8UL, 8UL);
    enc.conv(8UL, 3UL, 2UL, 1UL, ACT_RELU, &rng);     // 8 x 4 x 4
    enc.conv(16UL, 3UL, 2UL, 1UL, ACT_RELU, &rng);    // 16 x 2 x 2
    struct ConvNet dec = ConvNet::new(cfg.latent + c, 1UL, 1UL);
    dec.deconv(16UL, 2UL, 1UL, 0UL, ACT_RELU, &rng);  // 16 x 2 x 2
    dec.deconv(8UL, 4UL, 2UL, 1UL, ACT_RELU, &rng);   // 8 x 4 x 4
    dec.deconv(1UL, 4UL, 2UL, 1UL, ACT_NONE, &rng);   // 1 x 8 x 8
    return ConvVae::new(enc, dec, cfg);
}

static void conv_vae_case_(unsigned long classes) {
    struct Rng drng = Rng::new(9UL);
    struct Dataset data = Dataset::shapes(256UL, 8UL, &drng);
    struct VaeConfig cfg = VaeConfig::default(4UL);
    cfg.classes = classes;
    struct ConvVae cpu = cvae_build_(&cfg);
    struct ConvVae host = cvae_build_(&cfg);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 15UL;
    struct Optimizer o = Optimizer::new(OPT_ADAM, (float)0.003);
    model_add_params(&cpu, &o);
    struct TrainReport rc = model_fit(&cpu, &o, &data, &tc);
    struct RocmVae dv = RocmVae::new_conv(&host, 32UL);
    check(dv.ok, "RocmVae::new_conv");
    struct RocmOptimizer d = RocmOptimizer::new(OPT_ADAM, (float)0.003);
    dv.add_params(&d);
    struct TrainReport rg = rocm_trainer_fit(&dv, rocm_vae_step, &d, &data, &tc);
    unsafe { printf("  conv %s: loss cpu %.3f -> %.3f, gpu %.3f -> %.3f\n", classes > 0UL ? "cvae" : "vae",
                    (double)rc.firstEpochLoss, (double)rc.lastEpochLoss, (double)rg.firstEpochLoss,
                    (double)rg.lastEpochLoss); }
    check(rg.steps == rc.steps && rg.lastEpochLoss < rg.firstEpochLoss * (float)0.8, "conv vae loss falls");
    check(absf_(rg.firstEpochLoss - rc.firstEpochLoss) < (float)0.03 * rc.firstEpochLoss, "conv vae first epoch ~ cpu");
    check(absf_(rg.lastEpochLoss - rc.lastEpochLoss) < (float)0.1 * rc.lastEpochLoss, "conv vae last epoch ~ cpu");
    check(dv.download_conv_vae(&host), "conv vae download");
    d.free();
    o.free();
    dv.free();
    cpu.free();
    host.free();
    data.free();
}

static void vae_test_() {
    vae_case_(0UL);
    vae_case_(4UL);
    conv_ae_test_();
    conv_vae_case_(0UL);
    conv_vae_case_(4UL);
}

// ── GANs: steps against Gan::train_step / Dcgan::train_step with the same noise ─
static float params_diff_(const &Vec<&Tensor> a, const &Vec<&Tensor> b) {
    float worst = (float)0.0;
    unsigned long i = 0UL;
    while (i < a.length()) {
        &Tensor x = *a.at(i);
        &Tensor y = *b.at(i);
        unsafe {
            float scale = (float)1.0;
            float m = (float)0.0;
            unsigned long j = 0UL;
            while (j < x->size) {
                float d = absf_(x->data[j] - y->data[j]);
                if (d > m) m = d;
                if (absf_(y->data[j]) > scale) scale = absf_(y->data[j]);
                j = j + 1UL;
            }
            if (m / scale > worst) worst = m / scale;
        }
        i = i + 1UL;
    }
    return worst;
}

static struct RocmBuf view_(struct RocmBuf b, unsigned long off, unsigned long n) {
    struct RocmBuf v;
    v.dev = b.dev + off * 4UL;
    v.len = n;
    return v;
}

static void upload_at_(struct RocmBuf b, unsigned long off, const &Tensor t) {
    unsafe { std::rocm_write(view_(b, off, t->size), (const &float)(const float*)t->data, t->size); }
}

// The noise Gan::train_step will draw from rng (a copy), into dev.noise.
static void gan_noise_(&RocmGan dev, struct Rng rng, unsigned long m) {
    unsigned long nc = dev.cfg.nCritic == 0UL ? 1UL : dev.cfg.nCritic;
    unsigned long Z = dev.cfg.latent;
    unsigned long c = 0UL;
    while (c <= nc) {
        &Tensor z = rng.normal_tensor(m, Z);
        upload_at_(dev.noise, c * dev.maxBatch * Z, z);
        if (c < nc && dev.cfg.loss == GAN_WGAN_GP) {
            &Tensor t = Tensor::new_1d(m, 0);
            unsafe { unsigned long r = 0UL; while (r < m) { t->data[r] = rng.uniform(); r = r + 1UL; } }
            upload_at_(dev.noise, (nc + 1UL) * dev.maxBatch * Z + c * dev.maxBatch, t);
        }
        c = c + 1UL;
    }
    std::rocm_sync();
}

// sgd: both models' optimizers become momentum SGD. Adam (and RMSProp)
// normalize each weight's step, which turns gradients that are zero up to
// rounding (a bias before a batch norm, a flat hinge, the penalty's central
// difference) into full-size steps of either sign: the comparison then
// measures rounding, not the step.
static void gan_sgd_(&Optimizer a, &Optimizer b) {
    a.kind = OPT_MOMENTUM; b.kind = OPT_MOMENTUM;
    a.lr = (float)0.01; b.lr = (float)0.01;
}

static void gan_case_(const char* name, int loss, unsigned long classes, unsigned long nCritic, int sgd) {
    unsigned long m = 6UL;
    unsigned long D = 3UL;
    struct GanConfig cfg = GanConfig::default(4UL);
    cfg.hidden = 16UL;
    cfg.loss = loss;
    cfg.classes = classes;
    cfg.nCritic = nCritic;
    cfg.emaDecay = (float)0.9;
    cfg.lrG = (float)0.002;
    cfg.lrD = (float)0.002;
    cfg.clip = (float)0.05;
    struct Gan cpu = Gan::new(D, &cfg);
    struct Gan host = Gan::new(D, &cfg);
    if (sgd) { gan_sgd_(&cpu.optG, &host.optG); gan_sgd_(&cpu.optD, &host.optD); }
    struct RocmGan dev = RocmGan::new(&host, m);
    check(dev.ok, name);
    struct Rng rng = Rng::new(77UL);
    &Tensor real = Tensor::new_2d(m, D, 0);
    rng.fill_normal(real, (float)1.0, (float)0.5);
    &Tensor lab = Tensor::new_1d(m, 0);
    unsafe { unsigned long r = 0UL; while (r < m) { lab->data[r] = (float)(r % (classes == 0UL ? 1UL : classes)); r = r + 1UL; } }
    struct RocmBuf dr = upload_(real);
    struct RocmBuf dl = upload_(lab);
    struct GanLosses lc;
    int s = 0;
    int ok = 1;
    while (s < 3) {
        gan_noise_(&dev, cpu.rng, m);
        if (classes > 0UL) lc = cpu.train_step(real, (?&Tensor)lab);
        else lc = cpu.train_step(real, null);
        ok = ok && rocm_gan_step_(&dev, dr, dl, m, 0);
        s = s + 1;
    }
    struct GanLosses lg = dev.losses();
    ok = ok && dev.download(&host);
    float e = params_diff_(&host.gen.params, &cpu.gen.params);
    float e2 = params_diff_(&host.disc.params, &cpu.disc.params);
    float e3 = params_diff_(&host.genEma.params, &cpu.genEma.params);
    if (e2 > e) e = e2;
    if (e3 > e) e = e3;
    float le = absf_(lg.d - lc.d) / ((float)1.0 + absf_(lc.d));
    float lge = absf_(lg.g - lc.g) / ((float)1.0 + absf_(lc.g));
    if (lge > le) le = lge;
    unsafe { printf("  %-38s weights %.2e  losses %.2e (d %.4f / %.4f)\n", name, (double)e, (double)le,
                    (double)lc.d, (double)lg.d); }
    check(ok && host.steps == cpu.steps && e < (float)0.001 && le < (float)0.001, name);
    dev.free();
    cpu.free();
    host.free();
    dr.free(); dl.free();
}

static struct Dcgan dcgan_build_(const &GanConfig cfg) {
    struct Rng rng = Rng::new(5UL);
    unsigned long c = cfg.classes;
    struct ConvNet gen = ConvNet::new(cfg.latent + c, 1UL, 1UL);
    gen.deconv(8UL, 4UL, 1UL, 0UL, ACT_NONE, &rng);   // 8 x 4 x 4
    gen.batchnorm(ACT_RELU);
    gen.deconv(1UL, 4UL, 2UL, 1UL, ACT_TANH, &rng);   // 1 x 8 x 8
    struct ConvNet disc = ConvNet::new(1UL + c, 8UL, 8UL);
    disc.conv(4UL, 4UL, 2UL, 1UL, ACT_LEAKY, &rng);   // 4 x 4 x 4
    disc.conv(8UL, 4UL, 2UL, 1UL, ACT_NONE, &rng);    // 8 x 2 x 2
    disc.batchnorm(ACT_LEAKY);
    disc.conv(1UL, 2UL, 1UL, 0UL, ACT_NONE, &rng);    // 1 x 1 x 1
    return Dcgan::new(gen, disc, cfg);
}

static void dcgan_case_(const char* name, unsigned long classes) {
    unsigned long m = 5UL;
    struct GanConfig cfg = GanConfig::default(3UL);
    cfg.classes = classes;
    cfg.emaDecay = (float)0.9;
    cfg.lrG = (float)0.002;
    cfg.lrD = (float)0.002;
    struct Dcgan cpu = dcgan_build_(&cfg);
    struct Dcgan host = dcgan_build_(&cfg);
    gan_sgd_(&cpu.optG, &host.optG);
    gan_sgd_(&cpu.optD, &host.optD);
    struct RocmGan dev = RocmGan::new_dcgan(&host, m);
    check(dev.ok, name);
    struct Rng rng = Rng::new(78UL);
    &Tensor real = Tensor::new_2d(m, 64UL, 0);
    rng.fill_uniform(real, (float)-1.0, (float)1.0);
    &Tensor lab = Tensor::new_1d(m, 0);
    unsafe { unsigned long r = 0UL; while (r < m) { lab->data[r] = (float)(r % (classes == 0UL ? 1UL : classes)); r = r + 1UL; } }
    struct RocmBuf dr = upload_(real);
    struct RocmBuf dl = upload_(lab);
    struct GanLosses lc;
    int ok = 1;
    int s = 0;
    while (s < 3) {
        gan_noise_(&dev, cpu.rng, m);
        if (classes > 0UL) lc = cpu.train_step(real, (?&Tensor)lab);
        else lc = cpu.train_step(real, null);
        ok = ok && rocm_gan_step_(&dev, dr, dl, m, 0);
        s = s + 1;
    }
    struct GanLosses lg = dev.losses();
    ok = ok && dev.download_dcgan(&host);
    float e = params_diff_(&host.gen.params, &cpu.gen.params);
    float e2 = params_diff_(&host.disc.params, &cpu.disc.params);
    float e3 = params_diff_(&host.genEma.params, &cpu.genEma.params);
    float e4 = params_diff_(&host.gen.buffers, &cpu.gen.buffers);
    float e5 = params_diff_(&host.disc.buffers, &cpu.disc.buffers);
    float e6 = params_diff_(&host.genEma.buffers, &cpu.genEma.buffers);
    if (e2 > e) e = e2;
    if (e3 > e) e = e3;
    if (e4 > e) e = e4;
    if (e5 > e) e = e5;
    if (e6 > e) e = e6;
    float le = absf_(lg.d - lc.d) / ((float)1.0 + absf_(lc.d));
    float lge = absf_(lg.g - lc.g) / ((float)1.0 + absf_(lc.g));
    if (lge > le) le = lge;
    unsafe { printf("  %-38s weights+stats %.2e  losses %.2e\n", name, (double)e, (double)le); }
    check(ok && e < (float)0.001 && le < (float)0.001, name);
    dev.free();
    cpu.free();
    host.free();
    dr.free(); dl.free();
}

static void gan_tests_() {
    gan_case_("gan standard (adam)", GAN_STANDARD, 0UL, 1UL, 0);
    gan_case_("gan lsgan, conditional (adam)", GAN_LSGAN, 3UL, 1UL, 0);
    gan_case_("gan wgan (clip), 2 critic steps", GAN_WGAN, 0UL, 2UL, 1);
    gan_case_("gan hinge, conditional", GAN_HINGE, 2UL, 1UL, 1);
    gan_case_("gan wgan-gp, 2 critic steps", GAN_WGAN_GP, 0UL, 2UL, 1);
    gan_case_("gan wgan-gp, conditional", GAN_WGAN_GP, 3UL, 1UL, 1);
    dcgan_case_("dcgan (batch norm)", 0UL);
    dcgan_case_("dcgan conditional", 2UL);
    // the training loop runs and its graphs replay
    struct Rng drng = Rng::new(3UL);
    struct Dataset data = Dataset::blobs(200UL, 1UL, 2UL, (float)0.5, &drng);   // 3 full batches of 64 + a tail
    struct GanConfig cfg = GanConfig::default(4UL);
    cfg.hidden = 32UL;
    struct Gan g = Gan::new(2UL, &cfg);
    struct RocmGan dev = RocmGan::new(&g, 64UL);
    struct GanLosses l = dev.train(&data, 5UL, 64UL);
    check(dev.steps == 20UL && dev.download(&g) && g.steps == 20UL && l.d > (float)0.0 && l.d < (float)5.0,
          "RocmGan::train");
    unsafe { printf("  RocmGan::train: 20 steps, last losses d %.4f g %.4f\n", (double)l.d, (double)l.g); }
    dev.free();
    g.free();
    data.free();
}

// ── RL: device agents follow the host agents step for step ──────────────────
static void dqn_case_(const char* name, unsigned long targetEvery, int doubleDqn) {
    struct DqnConfig cfg = DqnConfig::default();
    cfg.hidden = 32UL;
    cfg.batchSize = 32UL;
    cfg.bufferSize = 500UL;
    cfg.learningStarts = 64UL;
    cfg.epsDecaySteps = 100UL;
    cfg.targetEvery = targetEvery;
    cfg.doubleDqn = doubleDqn;
    struct Dqn cpu = Dqn::new(4UL, 3UL, &cfg);
    struct Dqn host = Dqn::new(4UL, 3UL, &cfg);
    struct RocmDqn dev = RocmDqn::new(&host);
    check(dev.ok, name);
    struct Rng env = Rng::new(91UL);
    &Tensor o = Tensor::new_1d(4UL, 0);
    &Tensor n = Tensor::new_1d(4UL, 0);
    int same = 1;
    int ok = 1;
    float lc = (float)-1.0;
    unsigned long i = 0UL;
    while (i < 160UL) {
        env.fill_normal(o, (float)0.0, (float)1.0);
        env.fill_normal(n, (float)0.0, (float)1.0);
        unsigned long a1 = cpu.act(o, 1);
        unsigned long a2 = dev.act(o, 1);
        if (a1 != a2) same = 0;
        float r = env.uniform();
        int t = env.below(10UL) == 0UL;
        float l = cpu.observe(o, a1, r, n, t);
        if (l >= (float)0.0) lc = l;
        if (dev.observe(o, a1, r, n, t) < 0) ok = 0;
        i = i + 1UL;
    }
    // greedy actions agree too
    unsigned long g1 = cpu.act(o, 0);
    unsigned long g2 = dev.act(o, 0);
    float lg = dev.loss();
    ok = ok && dev.download(&host);
    float e = params_diff_(&host.q.params, &cpu.q.params);
    float e2 = params_diff_(&host.target.params, &cpu.target.params);
    if (e2 > e) e = e2;
    float le = absf_(lg - lc) / ((float)1.0 + absf_(lc));
    unsafe { printf("  %-38s %lu updates, weights %.2e, loss %.2e\n", name, host.updates, (double)e, (double)le); }
    check(ok && same && g1 == g2 && host.updates == cpu.updates && host.steps == cpu.steps && e < (float)0.001 &&
          le < (float)0.001, name);
    dev.free();
    cpu.free();
    host.free();
}

static void pg_case_(const char* name, int algo, unsigned long rollout) {
    struct PgConfig cfg = PgConfig::default(algo);
    cfg.hidden = 32UL;
    cfg.minibatch = 24UL;      // with rollout 100: 4 full minibatches and a short one
    cfg.epochs = 3UL;
    struct PolicyAgent cpu = PolicyAgent::new(4UL, 3UL, &cfg);
    struct PolicyAgent host = PolicyAgent::new(4UL, 3UL, &cfg);
    struct RocmPolicy dev = RocmPolicy::new(&host, 256UL);
    check(dev.ok, name);
    struct Rng env = Rng::new(92UL);
    &Tensor o = Tensor::new_1d(4UL, 0);
    &Tensor n = Tensor::new_1d(4UL, 0);
    int same = 1;
    int ok = 1;
    float lc = (float)0.0;
    float lg = (float)0.0;
    int u = 0;
    while (u < 2) {
        unsigned long i = 0UL;
        while (i < rollout) {
            env.fill_normal(o, (float)0.0, (float)1.0);
            env.fill_normal(n, (float)0.0, (float)1.0);
            unsigned long a1 = cpu.act(o, 1);
            unsigned long a2 = dev.act(o, 1);
            if (a1 != a2) same = 0;
            float r = env.uniform();
            int t = env.below(20UL) == 0UL;
            int tr = !t && env.below(30UL) == 0UL;
            cpu.record(o, a1, r, n, t, tr);
            dev.record(o, a1, r, n, t, tr);
            i = i + 1UL;
        }
        lc = cpu.update();
        lg = dev.update();
        if (lg == (float)-1.0) ok = 0;
        u = u + 1;
    }
    ok = ok && dev.download(&host);
    float e = params_diff_(&host.policy.params, &cpu.policy.params);
    float e2 = params_diff_(&host.value.params, &cpu.value.params);
    if (e2 > e) e = e2;
    float le = absf_(lg - lc) / ((float)1.0 + absf_(lc));
    unsafe { printf("  %-38s weights %.2e, loss %.2e (%.5f / %.5f)\n", name, (double)e, (double)le, (double)lc,
                    (double)lg); }
    check(ok && same && e < (float)0.001 && le < (float)0.001, name);
    dev.free();
    cpu.free();
    host.free();
}

static void rl_tests_() {
    dqn_case_("dqn (double, soft target)", 0UL, 1);
    dqn_case_("dqn (hard target every 5)", 5UL, 0);
    pg_case_("ppo", PG_PPO, 100UL);
    pg_case_("a2c", PG_A2C, 64UL);
    pg_case_("reinforce", PG_REINFORCE, 64UL);
}

int main() {
    if (!std::rocm_available()) { unsafe { printf("no ROCm GPU -- skipping\n"); } return 0; }
    unsafe { printf("nn_rocm on %s\n", std::rocm_device_name()); }
    // the convolution layers on MIOpen (when linked) and on the im2col engine
    int mio = rocm_conv_engine(ROCM_CONV_AUTO);
    unsafe { printf("conv engine: %s\n", mio ? "MIOpen" : "im2col"); }
    grad_tests_();
    convnet_tests_();
    if (mio) {
        rocm_conv_engine(ROCM_CONV_IM2COL);
        unsafe { printf("conv engine: im2col\n"); }
        grad_tests_();
        convnet_tests_();
        rocm_conv_engine(ROCM_CONV_AUTO);
        // 16-bit activations (convolutions on MIOpen only)
        half_tests_(ROCM_FP16);
        half_tests_(ROCM_BF16);
    }
    opt_tests_();
    trainer_test_();
    vae_test_();
    gan_tests_();
    rl_tests_();
    unsafe { printf("%d checks, %d failures\n", g_checks, g_fails); }
    return g_fails == 0 ? 0 : 1;
}
