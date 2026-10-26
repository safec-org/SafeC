// Correctness check for std/ml/gpu_spirv.h (array ops on a Vulkan GPU)
// and tensor_spirv.h against CPU loops and tensors: every host-array op, a chain of buffer ops with no
// read until its end, sums and products of awkward sizes, a write queued
// behind ops that read the old contents, dependent dispatches in order, and the precise math the kernels
// need (fast math stays off under MoltenVK; a non-finite value is caught).
// Needs a Vulkan GPU; run with tests/stdlib/vulkan/run.sh.
#include <std/ml/gpu_spirv.h>
#include <std/ml/tensor_spirv.h>
#include <std/ml/tensor.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/device.h>
#include <std/ml/device_kernels.h>
#include <std/collections/vec.h>
#include <std/math.h>
extern int printf(const char* fmt, ...);

static int g_checks = 0;
static int g_fails = 0;

static void check(int ok, const char* what) {
    g_checks = g_checks + 1;
    if (!ok) {
        g_fails = g_fails + 1;
        unsafe { printf("  FAIL %s\n", what); }
    }
}

// Max |a - b| relative to max(1, max |b|) over n values.
static double relerr(&Vec<float> a, &Vec<float> b, unsigned long n) {
    double m = 0.0;
    double big = 1.0;
    unsigned long i = 0UL;
    while (i < n) {
        double d = (double)*a.at(i) - (double)*b.at(i);
        if (d < 0.0) d = -d;
        if (d > m) m = d;
        if (d != d) m = 1.0e9;   // NaN
        double r = (double)*b.at(i);
        if (r < 0.0) r = -r;
        if (r > big) big = r;
        i = i + 1UL;
    }
    return m / big;
}

// Deterministic values in [lo, lo + span).
static struct Vec<float> data(unsigned long n, unsigned long seed, float lo, float span) {
    struct Vec<float> v = std::Vec::with_cap(n);
    unsigned long i = 0UL;
    while (i < n) {
        v.push(lo + span * (float)((i * 7919UL + seed * 104729UL) % 2003UL) / (float)2003.0);
        i = i + 1UL;
    }
    return v;
}

static struct Vec<float> zeros(unsigned long n) {
    struct Vec<float> v = std::Vec::with_cap(n);
    unsigned long i = 0UL;
    while (i < n) { v.push((float)0.0); i = i + 1UL; }
    return v;
}

#define OP_ADD 0
#define OP_SUB 1
#define OP_MUL 2
#define OP_DIV 3
#define OP_POW 4
#define OP_RELU 5
#define OP_LOG 6
#define OP_EXP 7
#define OP_SQRT 8
#define OP_SCALE 9

static float ref_(int op, float x, float y) {
    if (op == OP_ADD) return x + y;
    if (op == OP_SUB) return x - y;
    if (op == OP_MUL) return x * y;
    if (op == OP_DIV) return x / y;
    if (op == OP_POW) return (float)std::pow_d((double)x, (double)y);
    if (op == OP_RELU) return x > (float)0.0 ? x : (float)0.0;
    if (op == OP_LOG) return (float)std::log_d((double)x);
    if (op == OP_EXP) return (float)std::exp_d((double)x);
    if (op == OP_SQRT) return (float)std::sqrt_d((double)x);
    return x * (float)1.75;
}

static int host_op_(int op, &Vec<float> a, &Vec<float> b, &Vec<float> o, unsigned long n) {
    unsafe {
        []const float pa = a.as_slice()[0..n];
        []const float pb = b.as_slice()[0..n];
        []float po = o.as_slice()[0..n];
        if (op == OP_ADD) return spirv_add_f32(pa, pb, po);
        if (op == OP_SUB) return spirv_sub_f32(pa, pb, po);
        if (op == OP_MUL) return spirv_mul_f32(pa, pb, po);
        if (op == OP_DIV) return spirv_div_f32(pa, pb, po);
        if (op == OP_POW) return spirv_pow_f32(pa, pb, po);
        if (op == OP_RELU) return spirv_relu_f32(pa, po);
        if (op == OP_LOG) return spirv_log_f32(pa, po);
        if (op == OP_EXP) return spirv_exp_f32(pa, po);
        if (op == OP_SQRT) return spirv_sqrt_f32(pa, po);
        return spirv_scale_f32(pa, (float)1.75, po);
    }
}

// Every host-array op on n values (positive inputs where the op needs them).
static void check_ops_(unsigned long n) {
    const char* names[10];
    unsafe {
        names[0] = "add"; names[1] = "sub"; names[2] = "mul"; names[3] = "div"; names[4] = "pow";
        names[5] = "relu"; names[6] = "log"; names[7] = "exp"; names[8] = "sqrt"; names[9] = "scale";
    }
    int op = 0;
    while (op < 10) {
        int pos = op == OP_DIV || op == OP_POW || op == OP_LOG || op == OP_SQRT;
        struct Vec<float> a = pos ? data(n, 1UL, (float)0.1, (float)4.0) : data(n, 1UL, (float)-3.0, (float)6.0);
        struct Vec<float> b = data(n, 2UL, (float)0.5, (float)2.0);
        struct Vec<float> o = zeros(n);
        struct Vec<float> r = zeros(n);
        unsigned long i = 0UL;
        while (i < n) { *r.at(i) = ref_(op, *a.at(i), *b.at(i)); i = i + 1UL; }
        int ok = host_op_(op, &a, &b, &o, n);
        double e = relerr(&o, &r, n);
        unsafe {
            if (!ok || e > 2.0e-5) printf("    %s n=%lu: ran %d, error %g\n", names[op], n, ok, e);
            check(ok && e <= 2.0e-5, names[op]);
        }
        a.free(); b.free(); o.free(); r.free();
        op = op + 1;
    }
}

// sum and matmul against double-precision loops
static void check_reductions_(unsigned long n, unsigned long M, unsigned long K, unsigned long N) {
    struct Vec<float> a = data(n, 3UL, (float)-1.0, (float)2.0);
    double s = 0.0;
    unsigned long i = 0UL;
    while (i < n) { s = s + (double)*a.at(i); i = i + 1UL; }
    float got[1];
    int ok = 0;
    unsafe { got[0] = (float)0.0; ok = spirv_sum_f32(a.as_slice()[0..n], got[0..1]); }
    double d = (double)got[0] - s;
    if (d < 0.0) d = -d;
    unsafe {
        if (!ok || d > 1.0e-4 * (1.0 + (double)n / 100.0)) printf("    sum n=%lu: %g, want %g\n", n, (double)got[0], s);
    }
    check(ok && d <= 1.0e-4 * (1.0 + (double)n / 100.0), "sum");
    a.free();

    struct Vec<float> x = data(M * K, 4UL, (float)-1.0, (float)2.0);
    struct Vec<float> w = data(K * N, 5UL, (float)-1.0, (float)2.0);
    struct Vec<float> o = zeros(M * N);
    struct Vec<float> r = zeros(M * N);
    unsigned long p = 0UL;
    while (p < M) {
        unsigned long q = 0UL;
        while (q < N) {
            double acc = 0.0;
            unsigned long k = 0UL;
            while (k < K) { acc = acc + (double)*x.at(p * K + k) * (double)*w.at(k * N + q); k = k + 1UL; }
            *r.at(p * N + q) = (float)acc;
            q = q + 1UL;
        }
        p = p + 1UL;
    }
    unsafe { ok = spirv_matmul_f32((const &float)x.at(0UL), (const &float)w.at(0UL), (&float)o.at(0UL), M, K, N); }
    double e = relerr(&o, &r, M * N);
    unsafe { if (!ok || e > 1.0e-5) printf("    matmul %lux%lux%lu: ran %d, error %g\n", M, K, N, ok, e); }
    check(ok && e <= 1.0e-5, "matmul");
    x.free(); w.free(); o.free(); r.free();
}

// relu(a * b + a) * 2 on buffers, read once at the end; then a write queued
// behind an op that reads the old contents.
static void check_chain_() {
    unsigned long n = 10000UL;
    struct Vec<float> a = data(n, 6UL, (float)-2.0, (float)4.0);
    struct Vec<float> b = data(n, 7UL, (float)-2.0, (float)4.0);
    struct Vec<float> r = zeros(n);
    unsigned long i = 0UL;
    while (i < n) {
        float v = *a.at(i) * *b.at(i) + *a.at(i);
        *r.at(i) = (v > (float)0.0 ? v : (float)0.0) * (float)2.0;
        i = i + 1UL;
    }
    struct SpirvBuf ba = SpirvBuf::upload(a.as_slice()[0..n]);
    struct SpirvBuf bb = SpirvBuf::upload(b.as_slice()[0..n]);
    struct SpirvBuf t = SpirvBuf::alloc(n, 0);
    check(ba.valid() && bb.valid() && t.valid(), "buffers");
    int ok = spirv_mul(ba, bb, t, n) && spirv_add(t, ba, t, n) && spirv_relu(t, t, n) && spirv_scale(t, (float)2.0, t, n);
    // queued behind the chain: b := a; the chain read b first
    struct SpirvBuf copy = SpirvBuf::alloc(n, 0);
    ok = ok && spirv_add(bb, ba, copy, n);   // copy = a + b(old)
    unsafe { ok = ok && bb.write(((const unsigned char*)a.at(0UL))[0..4UL * n]); }
    struct Vec<float> o = zeros(n);
    unsafe { ok = ok && t.read(((unsigned char*)o.at(0UL))[0..4UL * n]); }
    double e = relerr(&o, &r, n);
    unsafe { if (!ok || e > 1.0e-6) printf("    chain: ran %d, error %g\n", ok, e); }
    check(ok && e <= 1.0e-6, "buffer chain");
    struct Vec<float> c = zeros(n);
    struct Vec<float> cr = zeros(n);
    i = 0UL;
    while (i < n) { *cr.at(i) = *a.at(i) + *b.at(i); i = i + 1UL; }
    unsafe { ok = copy.read(((unsigned char*)(void*)(c.at(0UL)))[0..4UL * n]); }
    check(ok && relerr(&c, &cr, n) <= 1.0e-6, "ops queued before a write read the old contents");
    struct Vec<float> nb = zeros(n);
    unsafe { ok = bb.read(((unsigned char*)(void*)(nb.at(0UL)))[0..4UL * n]); }
    check(ok && relerr(&nb, &a, n) == 0.0, "write lands");
    ba.free(); bb.free(); t.free(); copy.free();
    check(!ba.valid() && !t.valid(), "free");
    a.free(); b.free(); r.free(); o.free(); c.free(); cr.free(); nb.free();
}

// exp to a few ulp over its range: MoltenVK's fast math (off: the layer
// setting device_vulkan.sc passes) makes it 60 ulp.
static void check_precise_() {
    unsigned long n = 200000UL;
    struct Vec<float> a = zeros(n);
    struct Vec<float> o = zeros(n);
    unsigned long i = 0UL;
    while (i < n) { *a.at(i) = (float)-80.0 + (float)160.0 * (float)i / (float)n; i = i + 1UL; }
    int ok = 0;
    unsafe { ok = spirv_exp_f32(a.as_slice()[0..n], o.as_slice()[0..n]); }
    double worst = 0.0;
    i = 0UL;
    while (i < n) {
        double r = std::exp_d((double)*a.at(i));
        double d = ((double)*o.at(i) - r) / r;
        if (d < 0.0) d = -d;
        if (d > worst) worst = d;
        i = i + 1UL;
    }
    unsafe { if (!ok || worst > 4.0 * 5.96e-8) printf("    exp: ran %d, worst relative error %g\n", ok, worst); }
    check(ok && worst <= 4.0 * 5.96e-8, "exp within 4 ulp (no fast math)");
    a.free(); o.free();
}

// The loss scaler's overflow test (isfinite) sees an infinity.
static void check_ieee_() {
    float g[4];
    float st[4];
    unsafe {
        g[0] = (float)1.0; g[1] = (float)3.0e38; g[2] = (float)2.0; g[3] = (float)-1.0;
        st[0] = (float)1.0; st[1] = (float)4.0; st[2] = (float)0.0; st[3] = (float)0.0;   // 1/scale = 4: 3e38 * 4 = inf
    }
    struct SpirvBuf bg = SpirvBuf::alloc(4UL, 0);
    struct SpirvBuf bs = SpirvBuf::alloc(4UL, 0);
    int ok = 0;
    unsafe { ok = bg.write(((const unsigned char*)(const void*)(&g[0]))[0..16UL]) && bs.write(((const unsigned char*)(const void*)(&st[0]))[0..16UL]); }
    struct DevArgs a = dev_args_();
    unsafe { a.p[0] = bg.handle; a.p[1] = bs.handle; }
    ok = ok && device_launch_(DEVICE_VULKAN, DK_AMP_UNSCALE, 4UL, &a);
    unsafe { ok = ok && bs.read(((unsigned char*)(void*)(&st[0]))[0..16UL]); }
    unsafe { check(ok && st[2] == (float)1.0, "a non-finite gradient raises the overflow flag"); }
    bg.free(); bs.free();
}

// 400 dependent dispatches over 4M values, queued back to back: each must
// see the one before it (on MoltenVK the backend leaves out the barriers
// between dispatches, its serial encoders ordering them).
static void check_order_() {
    unsigned long n = 4000000UL;
    struct Vec<float> z = zeros(n);
    struct Vec<float> o = zeros(n);
    unsigned long i = 0UL;
    while (i < n) { *o.at(i) = (float)1.0; i = i + 1UL; }
    struct SpirvBuf x = SpirvBuf::upload(z.as_slice()[0..n]);
    struct SpirvBuf one = SpirvBuf::upload(o.as_slice()[0..n]);
    struct SpirvBuf y = SpirvBuf::alloc(n, 0);
    int ok = x.valid() && one.valid() && y.valid();
    int k = 0;
    while (k < 200 && ok) { ok = spirv_add(x, one, y, n) && spirv_add(y, one, x, n); k = k + 1; }
    unsafe { ok = ok && x.read(((unsigned char*)(void*)(z.at(0UL)))[0..4UL * n]); }
    unsigned long wrong = 0UL;
    i = 0UL;
    while (i < n) { if (*z.at(i) != (float)400.0) wrong = wrong + 1UL; i = i + 1UL; }
    unsafe { if (!ok || wrong > 0UL) printf("    order: ran %d, %lu values wrong\n", ok, wrong); }
    check(ok && wrong == 0UL, "dependent dispatches run in order");
    x.free(); one.free(); y.free();
    z.free(); o.free();
}

// sum(relu(a . b)) and its gradients through tensor_spirv.h = through the CPU ops
static void check_tensors_() {
    &FloatTensor a = FloatTensor::new_2d(5UL, 7UL, 1);
    &FloatTensor b = FloatTensor::new_2d(7UL, 3UL, 1);
    &FloatTensor a2 = FloatTensor::new_2d(5UL, 7UL, 1);
    &FloatTensor b2 = FloatTensor::new_2d(7UL, 3UL, 1);
    struct Vec<float> va = data(35UL, 8UL, (float)-1.0, (float)2.0);
    struct Vec<float> vb = data(21UL, 9UL, (float)-1.0, (float)2.0);
    unsigned long i = 0UL;
    while (i < 35UL) { a.set1(i, *va.at(i)); a2.set1(i, *va.at(i)); i = i + 1UL; }
    i = 0UL;
    while (i < 21UL) { b.set1(i, *vb.at(i)); b2.set1(i, *vb.at(i)); i = i + 1UL; }
    &FloatTensor y = tensor_matmul_spirv(a, b).relu_spirv().scale_spirv((float)0.5).sum_spirv();
    &FloatTensor y2 = tensor_matmul(a2, b2).relu().scale((float)0.5).sum();
    y.backward();
    y2.backward();
    float worst = (float)0.0;
    unsafe {
        i = 0UL;
        while (i < 35UL) { float d = a->grad[i] - a2->grad[i]; if (d < (float)0.0) d = -d; if (d > worst) worst = d; i = i + 1UL; }
        i = 0UL;
        while (i < 21UL) { float d = b->grad[i] - b2->grad[i]; if (d < (float)0.0) d = -d; if (d > worst) worst = d; i = i + 1UL; }
    }
    float dy = y.at1(0UL) - y2.at1(0UL);
    if (dy < (float)0.0) dy = -dy;
    unsafe { if (dy > (float)1e-5 || worst > (float)1e-5) printf("    tensors: loss %g vs %g, worst gradient difference %g\n", (double)y.at1(0UL), (double)y2.at1(0UL), (double)worst); }
    check(dy <= (float)1e-5 && worst <= (float)1e-5, "tensor_spirv ops and gradients = the CPU's");
    va.free(); vb.free();
}

int main() {
    int avail = spirv_available();
    unsafe { printf("gpu_spirv: available %d\n", avail); }
    if (!avail) { unsafe { printf("  FAIL no Vulkan GPU\n"); } return 1; }
    check_ops_(1UL);
    check_ops_(1000UL);
    check_ops_(300001UL);
    check_reductions_(1UL, 1UL, 1UL, 1UL);
    check_reductions_(1000003UL, 37UL, 65UL, 129UL);
    check_reductions_(4096UL, 128UL, 256UL, 64UL);
    check_chain_();
    check_order_();
    check_tensors_();
    check_precise_();
    check_ieee_();
    unsafe { printf("vulkan spirv: %d checks, %d failures\n", g_checks, g_fails); }
    return g_fails == 0 ? 0 : 1;
}
