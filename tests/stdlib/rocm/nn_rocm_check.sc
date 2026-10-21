// Correctness check for std/ml/nn_rocm (device-resident training) against
// std::ml's CPU autograd. Needs a ROCm GPU; run with tests/stdlib/rocm/run.sh.
#include <std/ml/nn_rocm.h>
#include <std/ml/convnet.h>
#include <std/collections/vec.h>
#include <std/math.h>
extern int printf(const char* fmt, ...);

static int g_checks = 0;
static int g_fails = 0;
static int   g_prec = 0;                   // the nets' activations (rocm_net_set_precision)
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
    &Tensor t = tensor_new_2d(r, c, 1);
    rng_fill_uniform(rng, t, -scale, scale);
    return t;
}

static struct RocmBuf upload_(const &Tensor t) {
    unsafe { return std::rocm_upload((const &float)(const float*)t->data, t->size); }
}

static float read1_(struct RocmBuf b) {
    float v = (float)0.0;
    std::rocm_read(b, &v, 1UL);
    return v;
}

// ── Layer specs: one network built both on the CPU (autograd) and the GPU ───
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
    return tensor_activate(y, s.act);
}

static int gpu_add_(&RocmNet n, const &Spec s) {
    if (s.kind == ROCM_LAYER_LINEAR) return rocm_net_add_linear(n, s.w, s.b, s.act);
    if (s.kind == ROCM_LAYER_CONV) return rocm_net_add_conv2d(n, s.w, s.b, s.inC, s.inH, s.inW, s.k, s.s, s.p, s.act);
    return rocm_net_add_conv_transpose2d(n, s.w, s.b, s.inC, s.inH, s.inW, s.k, s.s, s.p, s.act);
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
    struct Rng rng = rng_new(seed);
    &Tensor x = tensor_new_2d(rows, inDim, 1);
    rng_fill_normal(&rng, x, (float)0.0, (float)1.0);
    unsigned long tcols = loss == LOSS_CE ? 1UL : outDim;
    &Tensor y = tensor_new_2d(rows, tcols, 0);
    if (loss == LOSS_CE) {
        unsafe { unsigned long r = 0UL; while (r < rows) { y->data[r] = (float)((r * 7UL) % outDim); r = r + 1UL; } }
    } else if (loss == LOSS_BCE) {
        rng_fill_uniform(&rng, y, (float)0.0, (float)1.0);
    } else {
        rng_fill_normal(&rng, y, (float)0.0, (float)1.0);
    }
    // CPU
    unsigned long i = 0UL;
    while (i < specs.length()) { tensor_zero_grad(specs.at(i).w); tensor_zero_grad(specs.at(i).b); i = i + 1UL; }
    &Tensor h = x;
    i = 0UL;
    while (i < specs.length()) { h = cpu_layer_(specs.at(i), h); i = i + 1UL; }
    &Tensor l = loss == LOSS_CE ? tensor_cross_entropy_loss(h, y)
              : (loss == LOSS_MSE ? tensor_mse_loss(h, y) : tensor_bce_logits_loss(h, y));
    tensor_backward(l);
    // GPU
    struct RocmNet n = rocm_net_new(rows + 3UL);
    i = 0UL;
    while (i < specs.length()) { gpu_add_(&n, specs.at(i)); i = i + 1UL; }
    check(n.ok && rocm_net_in_dim(&n) == inDim && rocm_net_out_dim(&n) == outDim, name);
    if (g_prec != 0) check(rocm_net_set_precision(&n, g_prec) && rocm_net_set_loss_scale(&n, g_scale), name);
    struct RocmBuf dx = upload_(x);
    struct RocmBuf dy = upload_(y);
    struct RocmBuf ddx = std::rocm_alloc(rows * inDim);
    struct RocmBuf lb = std::rocm_alloc(1UL);
    std::rocm_zero(lb);
    int ok = 1;
    struct RocmBuf out = g_prec != 0 ? n.out32 : n.layers.at(n.layers.length() - 1UL).h;
    if (step) {
        if (loss == LOSS_CE) ok = rocm_net_ce_step(&n, dx, dy, rows, lb);
        else if (loss == LOSS_MSE) ok = rocm_net_mse_step(&n, dx, dy, rows, lb);
        else ok = rocm_net_bce_step(&n, dx, dy, rows, lb);
    } else {
        out = rocm_net_forward(&n, dx, rows);
        ok = out.dev != 0UL;
        if (loss == LOSS_CE) ok = ok && rocm_cross_entropy(out, dy, rocm_net_grad_output(&n), lb, rows, outDim);
        else if (loss == LOSS_MSE) ok = ok && rocm_mse(out, dy, rocm_net_grad_output(&n), lb, rows * outDim, (float)1.0);
        else ok = ok && rocm_bce_logits(out, dy, rocm_net_grad_output(&n), lb, rows * outDim, (float)1.0);
        ok = ok && rocm_net_backward(&n, dx, rows, ddx);
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
    rocm_net_free(&n);
    std::rocm_free(&dx); std::rocm_free(&dy); std::rocm_free(&ddx); std::rocm_free(&lb);
}

static void grad_tests_() {
    struct Rng rng = rng_new(11UL);
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

// ── ConvNets (every layer kind rocm_net_from_convnet takes) ─────────────────
// One forward / loss / backward of a convnet.h ConvNet on the CPU and of its
// rocm_net_from_convnet copy; compares the output, every parameter
// gradient, d(loss)/dx and the batch norms' running statistics. With
// 'twice' the CPU runs two batches through separate passes (summing their
// losses) and the GPU both stacked as one batch with segments = 2, as
// gan.h's discriminator step does.
static float max_rel_tensor_(struct RocmBuf d, const &Tensor t) {
    unsafe { return rel_diff_(d, (const float*)t->data, t->size); }
}

// The net's buffers (rocm_net_download_stats) against a ConvNet's.
static float buffers_diff_(&RocmNet n, const &Vec<&Tensor> cpu) {
    struct Vec<&Tensor> copy = std::vec_new();
    unsigned long i = 0UL;
    while (i < cpu.length()) {
        unsigned long sz;
        unsafe { sz = (*cpu.at(i))->size; }
        copy.push(tensor_new_1d(sz, 0));
        i = i + 1UL;
    }
    float e = (float)1.0e30;
    if (rocm_net_download_stats(n, &copy)) {
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
    struct Rng rng = rng_new(seed);
    unsigned long inDim = convnet_in_dim(c);
    unsigned long outDim = convnet_out_dim(c);
    unsigned long total = twice ? 2UL * rows : rows;
    &Tensor x = tensor_new_2d(total, inDim, 1);
    rng_fill_normal(&rng, x, (float)0.0, (float)1.0);
    unsigned long tcols = loss == LOSS_CE ? 1UL : outDim;
    &Tensor y = tensor_new_2d(total, tcols, 0);
    if (loss == LOSS_CE) {
        unsafe { unsigned long r = 0UL; while (r < total) { y->data[r] = (float)((r * 7UL) % outDim); r = r + 1UL; } }
    } else {
        rng_fill_normal(&rng, y, (float)0.0, (float)1.0);
    }
    // the GPU copy first: the CPU pass moves the running statistics
    struct RocmNet n = rocm_net_from_convnet(c, total);
    check(n.ok && rocm_net_in_dim(&n) == inDim && rocm_net_out_dim(&n) == outDim, name);
    if (g_prec != 0) check(rocm_net_set_precision(&n, g_prec) && rocm_net_set_loss_scale(&n, g_scale), name);
    unsigned long i = 0UL;
    while (i < c.params.length()) { tensor_zero_grad(*c.params.at(i)); i = i + 1UL; }
    &Tensor out = tensor_new_2d(total, outDim, 0);  // the CPU output
    &Tensor gx = tensor_new_2d(total, inDim, 0);    // d(loss)/dx on the CPU
    float lval = (float)0.0;
    if (twice) {
        // rows [0, m) and [m, 2m) as two passes
        &Tensor x1 = tensor_new_2d(rows, inDim, 1);
        &Tensor x2 = tensor_new_2d(rows, inDim, 1);
        &Tensor y1 = tensor_new_2d(rows, tcols, 0);
        &Tensor y2 = tensor_new_2d(rows, tcols, 0);
        unsafe {
            unsigned long k = 0UL;
            while (k < rows * inDim) { x1->data[k] = x->data[k]; x2->data[k] = x->data[rows * inDim + k]; k = k + 1UL; }
            k = 0UL;
            while (k < rows * tcols) { y1->data[k] = y->data[k]; y2->data[k] = y->data[rows * tcols + k]; k = k + 1UL; }
        }
        &Tensor o1 = convnet_forward(c, x1);
        &Tensor o2 = convnet_forward(c, x2);
        &Tensor l2 = tensor_add(tensor_mse_loss(o1, y1), tensor_mse_loss(o2, y2));
        tensor_backward(l2);
        unsafe {
            lval = l2->data[0];
            unsigned long k = 0UL;
            while (k < rows * inDim) { gx->data[k] = x1->grad[k]; gx->data[rows * inDim + k] = x2->grad[k]; k = k + 1UL; }
            k = 0UL;
            while (k < rows * outDim) { out->data[k] = o1->data[k]; out->data[rows * outDim + k] = o2->data[k]; k = k + 1UL; }
        }
        n.segments = 2UL;
    } else {
        &Tensor oc = convnet_forward(c, x);
        &Tensor l1 = loss == LOSS_CE ? tensor_cross_entropy_loss(oc, y) : tensor_mse_loss(oc, y);
        tensor_backward(l1);
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
    struct RocmBuf ddx = std::rocm_alloc(total * inDim);
    struct RocmBuf lb = std::rocm_alloc(1UL);
    std::rocm_zero(lb);
    struct RocmBuf o = rocm_net_forward(&n, dx, total);
    int ok = o.dev != 0UL;
    if (loss == LOSS_CE) ok = ok && rocm_cross_entropy(o, dy, rocm_net_grad_output(&n), lb, total, outDim);
    else ok = ok && rocm_mse(o, dy, rocm_net_grad_output(&n), lb, total * outDim, twice ? (float)2.0 : (float)1.0);
    ok = ok && rocm_net_backward(&n, dx, total, ddx);
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
    rocm_net_free(&n);
    std::rocm_free(&dx); std::rocm_free(&dy); std::rocm_free(&ddx); std::rocm_free(&lb);
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
    struct RocmNet n = rocm_net_new(rows);
    rocm_net_set_input(&n, C, 5UL, 5UL);
    rocm_net_add_dropout(&n, p, byChannel, 1234UL);
    struct Rng rng = rng_new(50UL);
    &Tensor x = tensor_new_2d(rows, C * P, 0);
    rng_fill_uniform(&rng, x, (float)0.5, (float)1.5);
    struct RocmBuf dx = upload_(x);
    struct RocmBuf ddx = std::rocm_alloc(rows * C * P);
    struct RocmBuf o = rocm_net_forward(&n, dx, rows);
    int ok = o.dev != 0UL && nr_fill_test_(rocm_net_grad_output(&n), rows * C * P) &&
             rocm_net_backward(&n, dx, rows, ddx);
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
    rocm_net_free(&n);
    std::rocm_free(&dx); std::rocm_free(&ddx);
}

static void convnet_tests_() {
    struct Rng rng = rng_new(31UL);
    // grouped rectangular dilated conv, batch norm, ceil-mode max pool, group
    // norm, grouped transposed conv with output padding, avg pool without the
    // padding, global avg pool, dense
    struct ConvNet a = convnet_new(4UL, 9UL, 8UL);
    struct Conv2dSpec s1 = conv2d_spec(3UL, 1UL, 1UL);
    s1.kW = 2UL; s1.strideH = 2UL; s1.padW = 0UL; s1.dilationW = 2UL; s1.groups = 2UL;
    convnet_conv_ex(&a, 6UL, &s1, ACT_NONE, &rng);
    convnet_batchnorm(&a, ACT_RELU);
    struct Pool2dSpec p1 = pool2d_spec(3UL, 2UL, 1UL);
    p1.ceilMode = 1;
    convnet_maxpool_ex(&a, &p1);
    convnet_groupnorm(&a, 2UL, ACT_SILU);
    struct Conv2dSpec s2 = conv2d_spec(3UL, 2UL, 1UL);
    s2.groups = 3UL; s2.outPadH = 1UL;
    convnet_deconv_ex(&a, 6UL, &s2, ACT_TANH, &rng);
    struct Pool2dSpec p2 = pool2d_spec(2UL, 2UL, 1UL);
    p2.countIncludePad = 0;
    convnet_avgpool_ex(&a, &p2);
    convnet_global_avgpool(&a);
    convnet_dense(&a, 5UL, ACT_NONE, &rng);
    convnet_case_("convnet: groups/dilation/pools/norms", &a, 6UL, LOSS_CE, 41UL, 0);
    convnet_free(&a);

    // dense -> reshape -> upsample -> conv -> layer norm -> global max pool -> dense
    struct ConvNet b = convnet_new(1UL, 1UL, 10UL);
    convnet_dense(&b, 3UL * 4UL * 4UL, ACT_LEAKY, &rng);
    convnet_reshape(&b, 3UL, 4UL, 4UL);
    convnet_upsample(&b, 2UL);
    convnet_conv(&b, 5UL, 3UL, 1UL, 1UL, ACT_NONE, &rng);
    convnet_layernorm(&b, ACT_GELU);
    convnet_instancenorm(&b, ACT_SIGMOID);
    // (at 16 bits a max over near-equal values can pick another element)
    if (g_prec == 0) convnet_global_maxpool(&b);
    else convnet_global_avgpool(&b);
    convnet_dense(&b, 3UL, ACT_NONE, &rng);
    convnet_case_("convnet: dense/reshape/upsample/norms", &b, 7UL, LOSS_MSE, 42UL, 0);
    convnet_free(&b);

    // a DCGAN-style discriminator with batch norm: two stacked batches
    struct ConvNet d = convnet_new(1UL, 8UL, 8UL);
    convnet_conv(&d, 4UL, 4UL, 2UL, 1UL, ACT_LEAKY, &rng);
    convnet_conv(&d, 8UL, 4UL, 2UL, 1UL, ACT_NONE, &rng);
    convnet_batchnorm(&d, ACT_LEAKY);
    convnet_conv(&d, 1UL, 2UL, 1UL, 0UL, ACT_NONE, &rng);
    convnet_case_("convnet: batch norm, two stacked batches", &d, 5UL, LOSS_MSE, 43UL, 1);
    // ... and in inference mode (running statistics)
    convnet_set_training(&d, 0);
    convnet_case_("convnet: batch norm, inference mode", &d, 5UL, LOSS_MSE, 44UL, 0);
    convnet_free(&d);
    // residual blocks (identity and projection shortcuts, batch norm) -> global pool -> dense
    struct ConvNet r = convnet_new(3UL, 8UL, 8UL);
    convnet_conv(&r, 4UL, 3UL, 1UL, 1UL, ACT_RELU, &rng);
    convnet_residual_block(&r, 4UL, 1UL, 1, ACT_RELU, &rng);     // identity shortcut
    convnet_residual_block(&r, 8UL, 2UL, 1, ACT_RELU, &rng);     // projection, stride 2
    convnet_residual_block(&r, 8UL, 1UL, 0, ACT_LEAKY, &rng);    // no batch norm
    convnet_global_avgpool(&r);
    convnet_dense(&r, 4UL, ACT_NONE, &rng);
    convnet_case_("convnet: residual blocks", &r, 6UL, LOSS_CE, 45UL, 0);
    convnet_free(&r);

    // U-Net-style: saves reused twice, concat skips, bilinear / bicubic / nearest resizing
    struct ConvNet u = convnet_new(2UL, 8UL, 8UL);
    convnet_conv(&u, 4UL, 3UL, 1UL, 1UL, ACT_RELU, &rng);
    unsigned long sv0 = convnet_save(&u);                          // 4 x 8 x 8
    convnet_maxpool(&u, 2UL, 2UL, 0UL);                           // 4 x 4 x 4
    convnet_conv(&u, 6UL, 3UL, 1UL, 1UL, ACT_TANH, &rng);
    unsigned long sv1 = convnet_save(&u);                          // 6 x 4 x 4
    convnet_resize(&u, 3UL, 3UL, RESIZE_BICUBIC, 1);              // 6 x 3 x 3 (down)
    convnet_resize(&u, 4UL, 4UL, RESIZE_NEAREST, 0);              // 6 x 4 x 4
    convnet_concat_skip(&u, sv1);                                  // 12 x 4 x 4
    convnet_upsample_bilinear(&u, 2UL);                           // 12 x 8 x 8
    convnet_concat_skip(&u, sv0);                                  // 16 x 8 x 8
    convnet_add_skip(&u, sv0, ACT_NONE, &rng);                     // + projection 4 -> 16
    convnet_resize(&u, 11UL, 7UL, RESIZE_BICUBIC, 0);             // 16 x 11 x 7
    convnet_conv(&u, 1UL, 3UL, 1UL, 1UL, ACT_NONE, &rng);
    convnet_case_("convnet: concat skips, resizes", &u, 4UL, LOSS_MSE, 46UL, 0);
    convnet_free(&u);

    // spectral norm on conv, deconv and dense weights (training, then inference)
    struct ConvNet sn = convnet_new(2UL, 6UL, 6UL);
    convnet_conv(&sn, 5UL, 3UL, 2UL, 1UL, ACT_LEAKY, &rng);
    convnet_spectral_norm(&sn, &rng);
    convnet_deconv(&sn, 3UL, 2UL, 2UL, 0UL, ACT_LEAKY, &rng);
    convnet_spectral_norm(&sn, &rng);
    convnet_dense(&sn, 4UL, ACT_NONE, &rng);
    convnet_spectral_norm(&sn, &rng);
    convnet_case_("convnet: spectral norm (training)", &sn, 5UL, LOSS_CE, 47UL, 0);
    convnet_set_training(&sn, 0);
    convnet_case_("convnet: spectral norm (inference)", &sn, 5UL, LOSS_CE, 48UL, 0);
    convnet_free(&sn);

    // dropout in inference passes values through (the CPU's masks come from its own Rng)
    struct ConvNet dr = convnet_new(3UL, 4UL, 4UL);
    convnet_conv(&dr, 4UL, 3UL, 1UL, 1UL, ACT_RELU, &rng);
    convnet_dropout(&dr, (float)0.3, 1, &rng);
    convnet_dense(&dr, 3UL, ACT_NONE, &rng);
    convnet_set_training(&dr, 0);
    convnet_case_("convnet: dropout (inference)", &dr, 5UL, LOSS_CE, 49UL, 0);
    convnet_free(&dr);
    dropout_props_(0);
    dropout_props_(1);
}

// ── Mixed precision: a small ResNet trained in fp32 and in 16 bits ──────────
static struct ConvNet half_net_(unsigned long seed) {
    struct Rng rng = rng_new(seed);
    struct ConvNet c = convnet_new(1UL, 16UL, 16UL);
    convnet_conv(&c, 8UL, 3UL, 1UL, 1UL, ACT_NONE, &rng);
    convnet_batchnorm(&c, ACT_RELU);
    convnet_residual_block(&c, 8UL, 1UL, 1, ACT_RELU, &rng);
    convnet_residual_block(&c, 16UL, 2UL, 1, ACT_RELU, &rng);
    convnet_global_avgpool(&c);
    convnet_dense(&c, 4UL, ACT_NONE, &rng);
    return c;
}

// The 16-bit run's losses track the fp32 run's. scale0 > 0: an fp16 loss
// scale that overflows at first (steps skipped, the scale backs off).
static void half_train_case_(const char* name, int prec, float scale0) {
    struct Rng drng = rng_new(5UL);
    struct Dataset data = dataset_shapes(512UL, 16UL, &drng);
    struct ConvNet c1 = half_net_(9UL);
    struct ConvNet c2 = half_net_(9UL);
    struct RocmNet a = rocm_net_from_convnet(&c1, 64UL);
    struct RocmNet b = rocm_net_from_convnet(&c2, 64UL);
    int ok = rocm_net_set_precision(&b, prec);
    if (scale0 > (float)0.0) ok = ok && rocm_net_set_loss_scale(&b, scale0);
    struct RocmOptimizer oa = rocm_optimizer_new(OPT_ADAM, (float)0.003);
    struct RocmOptimizer ob = rocm_optimizer_new(OPT_ADAM, (float)0.003);
    ok = ok && rocm_optimizer_add_net(&oa, &a) && rocm_optimizer_add_net(&ob, &b);
    struct TrainConfig tc = train_config_default();
    tc.epochs = 5UL;
    tc.batchSize = 64UL;
    struct TrainReport ra = rocm_trainer_fit(&a, rocm_net_ce_step, &oa, &data, &tc);
    struct TrainReport rb = rocm_trainer_fit(&b, rocm_net_ce_step, &ob, &data, &tc);
    float s = rocm_net_loss_scale(&b);
    unsafe { printf("  %-38s loss fp32 %.4f -> %.4f, 16-bit %.4f -> %.4f, loss scale %g\n", name,
                    (double)ra.firstEpochLoss, (double)ra.lastEpochLoss, (double)rb.firstEpochLoss,
                    (double)rb.lastEpochLoss, (double)s); }
    ok = ok && rb.steps == ra.steps && rb.lastEpochLoss < (float)0.5 * rb.firstEpochLoss;
    // (the overflowing run skips its first steps)
    if (scale0 == (float)0.0) ok = ok && absf_(rb.lastEpochLoss - ra.lastEpochLoss) < (float)0.1 * ra.firstEpochLoss;
    if (prec == ROCM_FP16) ok = ok && (scale0 > (float)0.0 ? s < scale0 : s == (float)65536.0);
    check(ok, name);
    rocm_optimizer_free(&oa); rocm_optimizer_free(&ob);
    rocm_net_free(&a); rocm_net_free(&b);
    convnet_free(&c1); convnet_free(&c2);
    dataset_free(&data);
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

// ── Optimizers: every kind against optimizer_step, eager and replayed ───────
static void opt_case_(int kind, float clip, int graph) {
    unsigned long n = 1000UL;
    struct Rng rng = rng_new(20UL + (unsigned long)kind);
    &Tensor w = tensor_new_1d(n, 1);
    rng_fill_normal(&rng, w, (float)0.0, (float)1.0);
    tensor_backward(tensor_sum(w));   // allocates w's gradient, overwritten below
    &Tensor g = tensor_new_1d(n, 0);
    rng_fill_normal(&rng, g, (float)0.0, (float)1.0);
    struct RocmBuf dw = upload_(w);
    struct RocmBuf dg = upload_(g);
    struct Optimizer o = optimizer_new(kind, (float)0.01);
    o.weightDecay = (float)0.05;
    optimizer_add(&o, w);
    struct RocmOptimizer d = rocm_optimizer_new(kind, (float)0.01);
    d.weightDecay = (float)0.05;
    d.clipNorm = clip;
    rocm_optimizer_add(&d, dw, dg);
    struct RocmGraph gr;
    gr.exec = 0UL;
    if (graph) {
        rocm_optimizer_sync(&d);
        std::rocm_capture_begin();
        rocm_optimizer_step(&d);
        gr = std::rocm_capture_end();
        check(gr.exec != 0UL, "optimizer graph capture");
    }
    int s = 0;
    while (s < 4) {
        unsafe {
            unsigned long i = 0UL;
            while (i < n) { w->grad[i] = g->data[i]; i = i + 1UL; }
        }
        if (clip > (float)0.0) optimizer_clip_grad_norm(&o, clip);
        optimizer_step(&o);
        if (graph) std::rocm_graph_launch(gr);
        else rocm_optimizer_step(&d);
        s = s + 1;
    }
    float e;
    unsafe { e = rel_diff_(dw, (const float*)w->data, n); }
    unsafe { printf("  optimizer kind %d%s%s  max rel err %.2e\n", kind, clip > (float)0.0 ? " clip" : "",
                    graph ? " graph" : "", (double)e); }
    check(d.ok && e < (float)0.00001, "optimizer step");
    std::rocm_graph_free(&gr);
    rocm_optimizer_free(&d);
    optimizer_free(&o);
    std::rocm_free(&dw); std::rocm_free(&dg);
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
    return tensor_cross_entropy_loss(mlp_forward(m, x), y);
}

static void trainer_test_() {
    struct Rng drng = rng_new(1UL);
    struct Dataset data = dataset_xor(250UL, (float)0.0, &drng);   // 7 full batches + a 26-row tail
    struct Rng r1 = rng_new(3UL);
    struct Mlp cpu = mlp_new(2UL, 16UL, 2UL, 2UL, ACT_RELU, ACT_NONE, &r1);
    struct Rng r2 = rng_new(3UL);
    struct Mlp host = mlp_new(2UL, 16UL, 2UL, 2UL, ACT_RELU, ACT_NONE, &r2);
    struct TrainConfig tc = train_config_default();
    tc.epochs = 6UL;
    tc.clipNorm = (float)1.0;
    tc.lrDecay = (float)0.9;
    struct Optimizer o = optimizer_new(OPT_ADAM, (float)0.01);
    optimizer_add_mlp(&o, &cpu);
    struct TrainReport rc = trainer_fit(&cpu, clf_loss_, &o, &data, &tc);
    struct RocmNet net = rocm_net_from_mlp(&host, 32UL);
    struct RocmOptimizer d = rocm_optimizer_new(OPT_ADAM, (float)0.01);
    rocm_optimizer_add_net(&d, &net);
    struct TrainReport rg = rocm_trainer_fit(&net, rocm_net_ce_step, &d, &data, &tc);
    check(rg.steps == rc.steps && rg.epochs == rc.epochs && d.t == rc.steps, "trainer step count");
    check(rocm_net_download(&net, &host.params), "trainer download");
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
    rocm_optimizer_free(&d);
    optimizer_free(&o);
    rocm_net_free(&net);
    mlp_free(&cpu);
    mlp_free(&host);
    dataset_free(&data);
}

// ── VAE: trains, and agrees with the CPU VAE's loss level ───────────────────
static void vae_case_(unsigned long classes) {
    struct Rng drng = rng_new(9UL);
    struct Dataset data = dataset_shapes(256UL, 8UL, &drng);   // 4 shape classes
    struct VaeConfig cfg = vae_config_default(4UL);
    cfg.hidden = 64UL;
    cfg.classes = classes;
    struct Vae cpu = vae_new(64UL, &cfg);
    struct Vae host = vae_new(64UL, &cfg);
    struct TrainConfig tc = train_config_default();
    tc.epochs = 15UL;
    struct Optimizer o = optimizer_new(OPT_ADAM, (float)0.003);
    vae_add_params(&o, &cpu);
    struct TrainReport rc = trainer_fit(&cpu, vae_loss, &o, &data, &tc);
    struct RocmVae dv = rocm_vae_new(&host, 32UL);
    check(dv.ok, "rocm_vae_new");
    struct RocmOptimizer d = rocm_optimizer_new(OPT_ADAM, (float)0.003);
    rocm_optimizer_add_vae(&d, &dv);
    struct TrainReport rg = rocm_trainer_fit(&dv, rocm_vae_step, &d, &data, &tc);
    unsafe { printf("  %s: loss cpu %.3f -> %.3f, gpu %.3f -> %.3f\n", classes > 0UL ? "cvae" : "vae", (double)rc.firstEpochLoss,
                    (double)rc.lastEpochLoss, (double)rg.firstEpochLoss, (double)rg.lastEpochLoss); }
    check(rg.steps == rc.steps, "vae steps");
    check(rg.lastEpochLoss < rg.firstEpochLoss * (float)0.8, "vae loss falls");
    check(absf_(rg.firstEpochLoss - rc.firstEpochLoss) < (float)0.03 * rc.firstEpochLoss, "vae first epoch ~ cpu");
    check(absf_(rg.lastEpochLoss - rc.lastEpochLoss) < (float)0.1 * rc.lastEpochLoss, "vae last epoch ~ cpu");
    check(rocm_vae_download(&dv, &host), "vae download");
    rocm_optimizer_free(&d);
    optimizer_free(&o);
    rocm_vae_free(&dv);
    vae_free(&cpu);
    vae_free(&host);
    dataset_free(&data);
}

// ── Convolutional autoencoders ───────────────────────────────────────────────
static &Tensor cae_loss_(&ConvAutoencoder a, const &Tensor x, const &Tensor y) { return conv_autoencoder_loss(a, x, y); }

static struct ConvAutoencoder cae_build_() {
    struct Rng rng = rng_new(61UL);
    struct ConvNet enc = convnet_new(1UL, 8UL, 8UL);
    convnet_conv(&enc, 4UL, 3UL, 2UL, 1UL, ACT_RELU, &rng);     // 4 x 4 x 4
    convnet_conv(&enc, 8UL, 3UL, 2UL, 1UL, ACT_RELU, &rng);     // 8 x 2 x 2
    struct ConvNet dec = convnet_new(8UL, 2UL, 2UL);
    convnet_deconv(&dec, 4UL, 4UL, 2UL, 1UL, ACT_RELU, &rng);   // 4 x 4 x 4
    convnet_upsample_bilinear(&dec, 2UL);                       // 4 x 8 x 8
    convnet_conv(&dec, 1UL, 3UL, 1UL, 1UL, ACT_NONE, &rng);     // 1 x 8 x 8
    return conv_autoencoder_new(enc, dec, RECON_BCE);
}

static void conv_ae_test_() {
    struct Rng drng = rng_new(9UL);
    struct Dataset data = dataset_shapes(250UL, 8UL, &drng);
    struct ConvAutoencoder cpu = cae_build_();
    struct ConvAutoencoder host = cae_build_();
    struct TrainConfig tc = train_config_default();
    tc.epochs = 4UL;
    struct Optimizer o = optimizer_new(OPT_ADAM, (float)0.003);
    conv_autoencoder_add_params(&o, &cpu);
    struct TrainReport rc = trainer_fit(&cpu, cae_loss_, &o, &data, &tc);
    struct RocmConvAe dev = rocm_conv_autoencoder_new(&host, 32UL);
    check(dev.ok, "rocm_conv_autoencoder_new");
    struct RocmOptimizer d = rocm_optimizer_new(OPT_ADAM, (float)0.003);
    rocm_optimizer_add_conv_autoencoder(&d, &dev);
    struct TrainReport rg = rocm_trainer_fit(&dev, rocm_conv_autoencoder_step, &d, &data, &tc);
    check(rocm_conv_autoencoder_download(&dev, &host), "conv autoencoder download");
    float e = params_diff_(&host.enc.params, &cpu.enc.params);
    float e2 = params_diff_(&host.dec.params, &cpu.dec.params);
    if (e2 > e) e = e2;
    unsafe { printf("  conv autoencoder: loss cpu %.5f -> %.5f gpu %.5f -> %.5f, weights %.2e\n",
                    (double)rc.firstEpochLoss, (double)rc.lastEpochLoss, (double)rg.firstEpochLoss,
                    (double)rg.lastEpochLoss, (double)e); }
    check(rg.steps == rc.steps && absf_(rg.lastEpochLoss - rc.lastEpochLoss) < (float)0.0005 && e < (float)0.001,
          "conv autoencoder matches trainer_fit");
    rocm_optimizer_free(&d);
    optimizer_free(&o);
    rocm_conv_autoencoder_free(&dev);
    conv_autoencoder_free(&cpu);
    conv_autoencoder_free(&host);
    dataset_free(&data);
}

static struct ConvVae cvae_build_(const &VaeConfig cfg) {
    struct Rng rng = rng_new(62UL);
    unsigned long c = cfg.classes;
    struct ConvNet enc = convnet_new(1UL + c, 8UL, 8UL);
    convnet_conv(&enc, 8UL, 3UL, 2UL, 1UL, ACT_RELU, &rng);     // 8 x 4 x 4
    convnet_conv(&enc, 16UL, 3UL, 2UL, 1UL, ACT_RELU, &rng);    // 16 x 2 x 2
    struct ConvNet dec = convnet_new(cfg.latent + c, 1UL, 1UL);
    convnet_deconv(&dec, 16UL, 2UL, 1UL, 0UL, ACT_RELU, &rng);  // 16 x 2 x 2
    convnet_deconv(&dec, 8UL, 4UL, 2UL, 1UL, ACT_RELU, &rng);   // 8 x 4 x 4
    convnet_deconv(&dec, 1UL, 4UL, 2UL, 1UL, ACT_NONE, &rng);   // 1 x 8 x 8
    return conv_vae_new(enc, dec, cfg);
}

static void conv_vae_case_(unsigned long classes) {
    struct Rng drng = rng_new(9UL);
    struct Dataset data = dataset_shapes(256UL, 8UL, &drng);
    struct VaeConfig cfg = vae_config_default(4UL);
    cfg.classes = classes;
    struct ConvVae cpu = cvae_build_(&cfg);
    struct ConvVae host = cvae_build_(&cfg);
    struct TrainConfig tc = train_config_default();
    tc.epochs = 15UL;
    struct Optimizer o = optimizer_new(OPT_ADAM, (float)0.003);
    conv_vae_add_params(&o, &cpu);
    struct TrainReport rc = trainer_fit(&cpu, conv_vae_loss, &o, &data, &tc);
    struct RocmVae dv = rocm_conv_vae_new(&host, 32UL);
    check(dv.ok, "rocm_conv_vae_new");
    struct RocmOptimizer d = rocm_optimizer_new(OPT_ADAM, (float)0.003);
    rocm_optimizer_add_vae(&d, &dv);
    struct TrainReport rg = rocm_trainer_fit(&dv, rocm_vae_step, &d, &data, &tc);
    unsafe { printf("  conv %s: loss cpu %.3f -> %.3f, gpu %.3f -> %.3f\n", classes > 0UL ? "cvae" : "vae",
                    (double)rc.firstEpochLoss, (double)rc.lastEpochLoss, (double)rg.firstEpochLoss,
                    (double)rg.lastEpochLoss); }
    check(rg.steps == rc.steps && rg.lastEpochLoss < rg.firstEpochLoss * (float)0.8, "conv vae loss falls");
    check(absf_(rg.firstEpochLoss - rc.firstEpochLoss) < (float)0.03 * rc.firstEpochLoss, "conv vae first epoch ~ cpu");
    check(absf_(rg.lastEpochLoss - rc.lastEpochLoss) < (float)0.1 * rc.lastEpochLoss, "conv vae last epoch ~ cpu");
    check(rocm_conv_vae_download(&dv, &host), "conv vae download");
    rocm_optimizer_free(&d);
    optimizer_free(&o);
    rocm_vae_free(&dv);
    conv_vae_free(&cpu);
    conv_vae_free(&host);
    dataset_free(&data);
}

static void vae_test_() {
    vae_case_(0UL);
    vae_case_(4UL);
    conv_ae_test_();
    conv_vae_case_(0UL);
    conv_vae_case_(4UL);
}

// ── GANs: steps against gan_train_step / dcgan_train_step with the same noise ─
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

// The noise gan_train_step will draw from rng (a copy), into dev.noise.
static void gan_noise_(&RocmGan dev, struct Rng rng, unsigned long m) {
    unsigned long nc = dev.cfg.nCritic == 0UL ? 1UL : dev.cfg.nCritic;
    unsigned long Z = dev.cfg.latent;
    unsigned long c = 0UL;
    while (c <= nc) {
        &Tensor z = rng_normal_tensor(&rng, m, Z);
        upload_at_(dev.noise, c * dev.maxBatch * Z, z);
        if (c < nc && dev.cfg.loss == GAN_WGAN_GP) {
            &Tensor t = tensor_new_1d(m, 0);
            unsafe { unsigned long r = 0UL; while (r < m) { t->data[r] = rng_uniform(&rng); r = r + 1UL; } }
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
    struct GanConfig cfg = gan_config_default(4UL);
    cfg.hidden = 16UL;
    cfg.loss = loss;
    cfg.classes = classes;
    cfg.nCritic = nCritic;
    cfg.emaDecay = (float)0.9;
    cfg.lrG = (float)0.002;
    cfg.lrD = (float)0.002;
    cfg.clip = (float)0.05;
    struct Gan cpu = gan_new(D, &cfg);
    struct Gan host = gan_new(D, &cfg);
    if (sgd) { gan_sgd_(&cpu.optG, &host.optG); gan_sgd_(&cpu.optD, &host.optD); }
    struct RocmGan dev = rocm_gan_new(&host, m);
    check(dev.ok, name);
    struct Rng rng = rng_new(77UL);
    &Tensor real = tensor_new_2d(m, D, 0);
    rng_fill_normal(&rng, real, (float)1.0, (float)0.5);
    &Tensor lab = tensor_new_1d(m, 0);
    unsafe { unsigned long r = 0UL; while (r < m) { lab->data[r] = (float)(r % (classes == 0UL ? 1UL : classes)); r = r + 1UL; } }
    struct RocmBuf dr = upload_(real);
    struct RocmBuf dl = upload_(lab);
    struct GanLosses lc;
    int s = 0;
    int ok = 1;
    while (s < 3) {
        gan_noise_(&dev, cpu.rng, m);
        if (classes > 0UL) lc = gan_train_step(&cpu, real, (?&Tensor)lab);
        else lc = gan_train_step(&cpu, real, null);
        ok = ok && rocm_gan_step_(&dev, dr, dl, m, 0);
        s = s + 1;
    }
    struct GanLosses lg = rocm_gan_losses(&dev);
    ok = ok && rocm_gan_download(&dev, &host);
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
    rocm_gan_free(&dev);
    gan_free(&cpu);
    gan_free(&host);
    std::rocm_free(&dr); std::rocm_free(&dl);
}

static struct Dcgan dcgan_build_(const &GanConfig cfg) {
    struct Rng rng = rng_new(5UL);
    unsigned long c = cfg.classes;
    struct ConvNet gen = convnet_new(cfg.latent + c, 1UL, 1UL);
    convnet_deconv(&gen, 8UL, 4UL, 1UL, 0UL, ACT_NONE, &rng);   // 8 x 4 x 4
    convnet_batchnorm(&gen, ACT_RELU);
    convnet_deconv(&gen, 1UL, 4UL, 2UL, 1UL, ACT_TANH, &rng);   // 1 x 8 x 8
    struct ConvNet disc = convnet_new(1UL + c, 8UL, 8UL);
    convnet_conv(&disc, 4UL, 4UL, 2UL, 1UL, ACT_LEAKY, &rng);   // 4 x 4 x 4
    convnet_conv(&disc, 8UL, 4UL, 2UL, 1UL, ACT_NONE, &rng);    // 8 x 2 x 2
    convnet_batchnorm(&disc, ACT_LEAKY);
    convnet_conv(&disc, 1UL, 2UL, 1UL, 0UL, ACT_NONE, &rng);    // 1 x 1 x 1
    return dcgan_new(gen, disc, cfg);
}

static void dcgan_case_(const char* name, unsigned long classes) {
    unsigned long m = 5UL;
    struct GanConfig cfg = gan_config_default(3UL);
    cfg.classes = classes;
    cfg.emaDecay = (float)0.9;
    cfg.lrG = (float)0.002;
    cfg.lrD = (float)0.002;
    struct Dcgan cpu = dcgan_build_(&cfg);
    struct Dcgan host = dcgan_build_(&cfg);
    gan_sgd_(&cpu.optG, &host.optG);
    gan_sgd_(&cpu.optD, &host.optD);
    struct RocmGan dev = rocm_dcgan_new(&host, m);
    check(dev.ok, name);
    struct Rng rng = rng_new(78UL);
    &Tensor real = tensor_new_2d(m, 64UL, 0);
    rng_fill_uniform(&rng, real, (float)-1.0, (float)1.0);
    &Tensor lab = tensor_new_1d(m, 0);
    unsafe { unsigned long r = 0UL; while (r < m) { lab->data[r] = (float)(r % (classes == 0UL ? 1UL : classes)); r = r + 1UL; } }
    struct RocmBuf dr = upload_(real);
    struct RocmBuf dl = upload_(lab);
    struct GanLosses lc;
    int ok = 1;
    int s = 0;
    while (s < 3) {
        gan_noise_(&dev, cpu.rng, m);
        if (classes > 0UL) lc = dcgan_train_step(&cpu, real, (?&Tensor)lab);
        else lc = dcgan_train_step(&cpu, real, null);
        ok = ok && rocm_gan_step_(&dev, dr, dl, m, 0);
        s = s + 1;
    }
    struct GanLosses lg = rocm_gan_losses(&dev);
    ok = ok && rocm_dcgan_download(&dev, &host);
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
    rocm_gan_free(&dev);
    dcgan_free(&cpu);
    dcgan_free(&host);
    std::rocm_free(&dr); std::rocm_free(&dl);
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
    struct Rng drng = rng_new(3UL);
    struct Dataset data = dataset_blobs(200UL, 1UL, 2UL, (float)0.5, &drng);   // 3 full batches of 64 + a tail
    struct GanConfig cfg = gan_config_default(4UL);
    cfg.hidden = 32UL;
    struct Gan g = gan_new(2UL, &cfg);
    struct RocmGan dev = rocm_gan_new(&g, 64UL);
    struct GanLosses l = rocm_gan_train(&dev, &data, 5UL, 64UL);
    check(dev.steps == 20UL && rocm_gan_download(&dev, &g) && g.steps == 20UL && l.d > (float)0.0 && l.d < (float)5.0,
          "rocm_gan_train");
    unsafe { printf("  rocm_gan_train: 20 steps, last losses d %.4f g %.4f\n", (double)l.d, (double)l.g); }
    rocm_gan_free(&dev);
    gan_free(&g);
    dataset_free(&data);
}

// ── RL: device agents follow the host agents step for step ──────────────────
static void dqn_case_(const char* name, unsigned long targetEvery, int doubleDqn) {
    struct DqnConfig cfg = dqn_config_default();
    cfg.hidden = 32UL;
    cfg.batchSize = 32UL;
    cfg.bufferSize = 500UL;
    cfg.learningStarts = 64UL;
    cfg.epsDecaySteps = 100UL;
    cfg.targetEvery = targetEvery;
    cfg.doubleDqn = doubleDqn;
    struct Dqn cpu = dqn_new(4UL, 3UL, &cfg);
    struct Dqn host = dqn_new(4UL, 3UL, &cfg);
    struct RocmDqn dev = rocm_dqn_new(&host);
    check(dev.ok, name);
    struct Rng env = rng_new(91UL);
    &Tensor o = tensor_new_1d(4UL, 0);
    &Tensor n = tensor_new_1d(4UL, 0);
    int same = 1;
    int ok = 1;
    float lc = (float)-1.0;
    unsigned long i = 0UL;
    while (i < 160UL) {
        rng_fill_normal(&env, o, (float)0.0, (float)1.0);
        rng_fill_normal(&env, n, (float)0.0, (float)1.0);
        unsigned long a1 = dqn_act(&cpu, o, 1);
        unsigned long a2 = rocm_dqn_act(&dev, o, 1);
        if (a1 != a2) same = 0;
        float r = rng_uniform(&env);
        int t = rng_below(&env, 10UL) == 0UL;
        float l = dqn_observe(&cpu, o, a1, r, n, t);
        if (l >= (float)0.0) lc = l;
        if (rocm_dqn_observe(&dev, o, a1, r, n, t) < 0) ok = 0;
        i = i + 1UL;
    }
    // greedy actions agree too
    unsigned long g1 = dqn_act(&cpu, o, 0);
    unsigned long g2 = rocm_dqn_act(&dev, o, 0);
    float lg = rocm_dqn_loss(&dev);
    ok = ok && rocm_dqn_download(&dev, &host);
    float e = params_diff_(&host.q.params, &cpu.q.params);
    float e2 = params_diff_(&host.target.params, &cpu.target.params);
    if (e2 > e) e = e2;
    float le = absf_(lg - lc) / ((float)1.0 + absf_(lc));
    unsafe { printf("  %-38s %lu updates, weights %.2e, loss %.2e\n", name, host.updates, (double)e, (double)le); }
    check(ok && same && g1 == g2 && host.updates == cpu.updates && host.steps == cpu.steps && e < (float)0.001 &&
          le < (float)0.001, name);
    rocm_dqn_free(&dev);
    dqn_free(&cpu);
    dqn_free(&host);
}

static void pg_case_(const char* name, int algo, unsigned long rollout) {
    struct PgConfig cfg = pg_config_default(algo);
    cfg.hidden = 32UL;
    cfg.minibatch = 24UL;      // with rollout 100: 4 full minibatches and a short one
    cfg.epochs = 3UL;
    struct PolicyAgent cpu = pg_new(4UL, 3UL, &cfg);
    struct PolicyAgent host = pg_new(4UL, 3UL, &cfg);
    struct RocmPolicy dev = rocm_pg_new(&host, 256UL);
    check(dev.ok, name);
    struct Rng env = rng_new(92UL);
    &Tensor o = tensor_new_1d(4UL, 0);
    &Tensor n = tensor_new_1d(4UL, 0);
    int same = 1;
    int ok = 1;
    float lc = (float)0.0;
    float lg = (float)0.0;
    int u = 0;
    while (u < 2) {
        unsigned long i = 0UL;
        while (i < rollout) {
            rng_fill_normal(&env, o, (float)0.0, (float)1.0);
            rng_fill_normal(&env, n, (float)0.0, (float)1.0);
            unsigned long a1 = pg_act(&cpu, o, 1);
            unsigned long a2 = rocm_pg_act(&dev, o, 1);
            if (a1 != a2) same = 0;
            float r = rng_uniform(&env);
            int t = rng_below(&env, 20UL) == 0UL;
            int tr = !t && rng_below(&env, 30UL) == 0UL;
            pg_record(&cpu, o, a1, r, n, t, tr);
            rocm_pg_record(&dev, o, a1, r, n, t, tr);
            i = i + 1UL;
        }
        lc = pg_update(&cpu);
        lg = rocm_pg_update(&dev);
        if (lg == (float)-1.0) ok = 0;
        u = u + 1;
    }
    ok = ok && rocm_pg_download(&dev, &host);
    float e = params_diff_(&host.policy.params, &cpu.policy.params);
    float e2 = params_diff_(&host.value.params, &cpu.value.params);
    if (e2 > e) e = e2;
    float le = absf_(lg - lc) / ((float)1.0 + absf_(lc));
    unsafe { printf("  %-38s weights %.2e, loss %.2e (%.5f / %.5f)\n", name, (double)e, (double)le, (double)lc,
                    (double)lg); }
    check(ok && same && e < (float)0.001 && le < (float)0.001, name);
    rocm_pg_free(&dev);
    pg_free(&cpu);
    pg_free(&host);
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
