// Correctness check for std/ml/gpu_webgpu.h's array API against CPU
// results: the host-array ops, WebGpuBuf's constructors, reads and writes in
// each element type, matrix products (fp32 and 16-bit operands, edge tiles),
// the SGD update, and buffers reused after free. Needs a WebGPU adapter;
// run with tests/stdlib/webgpu/run.sh. Built with the webgpu feature, it also
// reads a DEVICE_WEBGPU tensor's buffer (trait WebGpuTensor) through the
// array API (the two share one device), and checks the convolutions large
// enough for the backend's larger tiles, which device_check's don't reach.
#include <std/ml/gpu_webgpu.h>
#include <std/ml/device.h>
#include <std/ml/tensor.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/convnet.h>
#include <std/ml/float16.h>
#include <std/math.h>
extern int printf(const char* fmt, ...);
extern void* malloc(unsigned long n);
extern void free(void* p);

static int g_checks = 0;
static int g_fails = 0;

static void check(int ok, const char* what) {
    g_checks = g_checks + 1;
    if (!ok) {
        g_fails = g_fails + 1;
        unsafe { printf("  FAIL %s\n", what); }
    }
}

// Max |a - b| relative to max(1, max |b|); NaN counts as a large error.
static double relerr(const float* a, const float* b, unsigned long n) {
    double m = 0.0;
    double big = 1.0;
    unsafe {
        unsigned long i = 0UL;
        while (i < n) {
            double d = (double)a[i] - (double)b[i];
            if (d < 0.0) d = -d;
            if (d != d) d = 1.0e9;
            if (d > m) m = d;
            double r = (double)b[i];
            if (r < 0.0) r = -r;
            if (r > big) big = r;
            i = i + 1UL;
        }
    }
    return m / big;
}

// Deterministic values in [lo, lo + span).
static float* data(unsigned long n, unsigned long seed, float lo, float span) {
    unsafe {
        float* v = (float*)malloc(4UL * n + 4UL);
        unsigned long i = 0UL;
        while (i < n) {
            v[i] = lo + span * (float)((i * 7919UL + seed * 104729UL) % 2003UL) / (float)2003.0;
            i = i + 1UL;
        }
        return v;
    }
}

static float* zeros(unsigned long n) {
    unsafe {
        float* v = (float*)malloc(4UL * n + 4UL);
        unsigned long i = 0UL;
        while (i < n) { v[i] = (float)0.0; i = i + 1UL; }
        return v;
    }
}

// The elementwise host-array ops over n values.
static void check_elementwise(unsigned long n) {
    float* a = data(n, 1UL, (float)0.1, (float)2.0);
    float* b = data(n, 2UL, (float)0.5, (float)1.5);
    float* s = data(n, 3UL, (float)-2.0, (float)4.0);
    float* got = zeros(n);
    float* want = zeros(n);
    unsafe {
        int op = 0;
        while (op < 10) {
            int ok = 0;
            unsigned long i = 0UL;
            if (op == 0) { ok = webgpu_add_f32(a, b, got, n); while (i < n) { want[i] = a[i] + b[i]; i = i + 1UL; } }
            if (op == 1) { ok = webgpu_sub_f32(a, b, got, n); while (i < n) { want[i] = a[i] - b[i]; i = i + 1UL; } }
            if (op == 2) { ok = webgpu_mul_f32(a, b, got, n); while (i < n) { want[i] = a[i] * b[i]; i = i + 1UL; } }
            if (op == 3) { ok = webgpu_div_f32(a, b, got, n); while (i < n) { want[i] = a[i] / b[i]; i = i + 1UL; } }
            if (op == 4) { ok = webgpu_pow_f32(a, b, got, n); while (i < n) { want[i] = (float)std::pow_d((double)a[i], (double)b[i]); i = i + 1UL; } }
            if (op == 5) { ok = webgpu_relu_f32(s, got, n); while (i < n) { want[i] = s[i] > (float)0.0 ? s[i] : (float)0.0; i = i + 1UL; } }
            if (op == 6) { ok = webgpu_log_f32(a, got, n); while (i < n) { want[i] = (float)std::log_d((double)a[i]); i = i + 1UL; } }
            if (op == 7) { ok = webgpu_exp_f32(s, got, n); while (i < n) { want[i] = (float)std::exp_d((double)s[i]); i = i + 1UL; } }
            if (op == 8) { ok = webgpu_sqrt_f32(a, got, n); while (i < n) { want[i] = (float)std::sqrt_d((double)a[i]); i = i + 1UL; } }
            if (op == 9) { ok = webgpu_scale_f32(s, (float)-1.5, got, n); while (i < n) { want[i] = s[i] * (float)-1.5; i = i + 1UL; } }
            const char* names[10];
            names[0] = "add"; names[1] = "sub"; names[2] = "mul"; names[3] = "div"; names[4] = "pow";
            names[5] = "relu"; names[6] = "log"; names[7] = "exp"; names[8] = "sqrt"; names[9] = "scale";
            double e = relerr(got, want, n);
            printf("  %-6s n %-8lu err %.2e\n", names[op], n, e);
            check(ok && e < 2.0e-6, names[op]);
            op = op + 1;
        }
        free((void*)a); free((void*)b); free((void*)s); free((void*)got); free((void*)want);
    }
}

// The sum over n values: per-workgroup sums added atomically.
static void check_sum(unsigned long n) {
    float* a = data(n, 4UL, (float)-1.0, (float)2.5);
    float got = (float)0.0;
    double want = 0.0;
    unsafe {
        unsigned long i = 0UL;
        while (i < n) { want = want + (double)a[i]; i = i + 1UL; }
        int ok = webgpu_sum_f32(a, &got, n);
        double e = (double)got - want;
        if (e < 0.0) e = -e;
        printf("  sum    n %-8lu got %.6g want %.6g\n", n, (double)got, want);
        check(ok && e <= 1.0e-5 * (double)n, "sum");
        free((void*)a);
    }
}

// out = a b on the CPU in double.
static void ref_matmul(const float* a, const float* b, float* out, unsigned long M, unsigned long K, unsigned long N) {
    unsafe {
        unsigned long i = 0UL;
        while (i < M) {
            unsigned long j = 0UL;
            while (j < N) {
                double s = 0.0;
                unsigned long k = 0UL;
                while (k < K) { s = s + (double)a[i * K + k] * (double)b[k * N + j]; k = k + 1UL; }
                out[i * N + j] = (float)s;
                j = j + 1UL;
            }
            i = i + 1UL;
        }
    }
}

// Rounds n values to dtype in place (what a 16-bit buffer holds).
static void round_to(float* v, unsigned long n, int dtype) {
    unsafe {
        unsigned long i = 0UL;
        while (i < n) {
            if (dtype == WEBGPU_FP16) v[i] = std::fp16_to_f32(std::f32_to_fp16(v[i]));
            if (dtype == WEBGPU_BF16) v[i] = std::bf16_to_f32(std::f32_to_bf16(v[i]));
            i = i + 1UL;
        }
    }
}

// A matrix product with operands of 'dtype' and an fp32 or 16-bit result.
static void check_matmul(unsigned long M, unsigned long K, unsigned long N, int dtype, int outType) {
    float* a = data(M * K, 5UL, (float)-1.0, (float)2.0);
    float* b = data(K * N, 6UL, (float)-1.0, (float)2.0);
    float* got = zeros(M * N);
    float* want = zeros(M * N);
    unsafe {
        round_to(a, M * K, dtype);
        round_to(b, K * N, dtype);
        int ok = 0;
        if (dtype == 0 && outType == 0) ok = webgpu_matmul_f32(a, b, got, M, K, N);
        else {
            struct WebGpuBuf ba = dtype == WEBGPU_FP16 ? WebGpuBuf::upload_f16(a, M * K)
                                : (dtype == WEBGPU_BF16 ? WebGpuBuf::upload_bf16(a, M * K) : WebGpuBuf::upload(a, M * K));
            struct WebGpuBuf bb = dtype == WEBGPU_FP16 ? WebGpuBuf::upload_f16(b, K * N)
                                : (dtype == WEBGPU_BF16 ? WebGpuBuf::upload_bf16(b, K * N) : WebGpuBuf::upload(b, K * N));
            struct WebGpuBuf bo = WebGpuBuf::alloc(M * N, outType);
            ok = webgpu_matmul(ba, bb, bo, M, K, N) && bo.read_f32(got, M * N);
            ba.free(); bb.free(); bo.free();
        }
        ref_matmul(a, b, want, M, K, N);
        round_to(want, M * N, outType);
        double e = relerr(got, want, M * N);
        double tol = outType == 0 ? 2.0e-5 : (outType == WEBGPU_FP16 ? 2.0e-3 : 1.6e-2);
        const char* tn = dtype == 0 ? "fp32" : (dtype == WEBGPU_FP16 ? "fp16" : "bf16");
        const char* on = outType == 0 ? "fp32" : (outType == WEBGPU_FP16 ? "fp16" : "bf16");
        printf("  matmul %s -> %s %lux%lux%lu err %.2e\n", tn, on, M, K, N, e);
        check(ok && e < tol, "matmul");
        free((void*)a); free((void*)b); free((void*)got); free((void*)want);
    }
}

// WebGpuBuf: round trips in each type, partial reads, odd 16-bit lengths,
// writes after uploads, sgd updates queued between them.
static void check_buffers() {
    unsigned long n = 1001UL;
    float* a = data(n, 7UL, (float)-3.0, (float)6.0);
    float* g = data(n, 8UL, (float)-1.0, (float)2.0);
    float* got = zeros(n);
    float* want = zeros(n);
    unsafe {
        struct WebGpuBuf f = WebGpuBuf::upload(a, n);
        check(f.valid() && f.bytes == 4UL * n && f.dtype == 0, "upload: valid fp32 buffer");
        check(f.read_f32(got, n) && relerr(got, a, n) == 0.0, "upload: fp32 round trip");
        check(!f.read((&void)(void*)got, 4UL * n + 4UL), "read past the end refused");
        int dt = WEBGPU_FP16;
        while (dt <= WEBGPU_BF16) {
            struct WebGpuBuf h = dt == WEBGPU_FP16 ? WebGpuBuf::upload_f16(a, n) : WebGpuBuf::upload_bf16(a, n);
            unsigned long i = 0UL;
            while (i < n) { want[i] = a[i]; i = i + 1UL; }
            round_to(want, n, dt);
            check(h.valid() && h.bytes == 2UL * n && h.read_f32(got, n) && relerr(got, want, n) == 0.0,
                  dt == WEBGPU_FP16 ? "upload_f16: round trip (odd length)" : "upload_bf16: round trip (odd length)");
            // w -= lr g in the buffer's type, then a write over it and a read
            struct WebGpuBuf gb = WebGpuBuf::upload(g, n);
            check(webgpu_sgd_update(h, gb, (float)0.25, n) && h.read_f32(got, n), "sgd update on a 16-bit buffer");
            i = 0UL;
            while (i < n) { want[i] = want[i] - (float)0.25 * g[i]; i = i + 1UL; }
            round_to(want, n, dt);
            check(relerr(got, want, n) < 1.0e-2, "sgd update: values");
            unsigned short bits[3];
            bits[0] = dt == WEBGPU_FP16 ? std::f32_to_fp16((float)1.5) : std::f32_to_bf16((float)1.5);
            bits[1] = dt == WEBGPU_FP16 ? std::f32_to_fp16((float)-2.0) : std::f32_to_bf16((float)-2.0);
            bits[2] = dt == WEBGPU_FP16 ? std::f32_to_fp16((float)0.25) : std::f32_to_bf16((float)0.25);
            check(h.write((const &void)(const void*)&bits[0], 6UL) && h.read_f32(got, 4UL), "write 3 halves");
            check(got[0] == (float)1.5 && got[1] == (float)-2.0 && got[2] == (float)0.25, "write 3 halves: values");
            h.free(); gb.free();
            check(!h.valid() && h.bytes == 0UL, "free invalidates");
            dt = dt + 1;
        }
        // fp32 sgd and buffers reused after free
        struct WebGpuBuf gb = WebGpuBuf::upload(g, n);
        check(webgpu_sgd_update(f, gb, (float)0.5, n) && f.read_f32(got, n), "sgd update fp32");
        unsigned long i = 0UL;
        while (i < n) { want[i] = a[i] - (float)0.5 * g[i]; i = i + 1UL; }
        check(relerr(got, want, n) < 1.0e-6, "sgd update fp32: values");
        f.free(); gb.free();
        int round = 0;
        int same = 1;
        while (round < 50) {
            struct WebGpuBuf t = WebGpuBuf::upload(g, n);
            if (!t.read_f32(got, n) || relerr(got, g, n) != 0.0) same = 0;
            t.free();
            round = round + 1;
        }
        check(same, "50 upload / read / free rounds");
        struct WebGpuBuf bad = WebGpuBuf::alloc(4UL, 7);
        check(!bad.valid(), "alloc of an unknown dtype is invalid");
        check(webgpu_sync(), "sync");
        free((void*)a); free((void*)g); free((void*)got); free((void*)want);
    }
}

// A tensor's buffer on the WebGPU device, for any WebGpuTensor.
generic<T: WebGpuTensor>
unsigned long webgpu_buffer_of(const &T t) {
    if (t.device() != DEVICE_WEBGPU) return 0UL;
    return t.device_buffer();
}

static void check_tensor_interop() {
    &FloatTensor t = FloatTensor::new_2d(3UL, 5UL, 0);
    unsigned long i = 0UL;
    while (i < 15UL) { t.set1(i, (float)i * (float)0.5 - (float)2.0); i = i + 1UL; }
    check(webgpu_buffer_of(t) == 0UL, "WebGpuTensor: no buffer on the CPU");
    check(t.place(DEVICE_WEBGPU), "tensor placed on DEVICE_WEBGPU");
    struct WebGpuBuf b;
    b.handle = webgpu_buffer_of(t);
    b.bytes = 60UL;
    b.dtype = 0;
    float got[15];
    unsafe {
        check(b.valid() && b.read_f32((&float)&got[0], 15UL), "WebGpuTensor: the tensor's buffer reads back");
        int same = 1;
        i = 0UL;
        while (i < 15UL) { if (got[i] != (float)i * (float)0.5 - (float)2.0) same = 0; i = i + 1UL; }
        check(same, "WebGpuTensor: the tensor's values");
        // a tensor op queued on the device, its result read through the array API
        &FloatTensor u = t.scale((float)2.0);
        check(u.device() == DEVICE_WEBGPU, "the tensor op ran on the device");
        b.handle = webgpu_buffer_of(u);
        check(b.valid() && b.read_f32((&float)&got[0], 15UL) && got[3] == (float)-1.0 && got[14] == (float)10.0,
              "a tensor op's result through the array API");
        u.free();
    }
    t.free();
}

// The backend's GEMM paths through device.h (c = alpha op(a) op(b) + beta c):
// the per-element kernel (small products), 64 x 64 tiles, split-K (few
// tiles, long k: slices added atomically into c scaled by beta) and
// 128 x 128 tiles, with transposes and 16-bit operands and results.
static void check_gemm_path(int ta, int tb, unsigned long m, unsigned long n, unsigned long k, int dt, int ct, const char* what) {
    int d = DEVICE_WEBGPU;
    float* a = data(m * k, 11UL, (float)-1.0, (float)2.0);
    float* b = data(k * n, 12UL, (float)-1.0, (float)2.0);
    float* c0 = data(m * n, 13UL, (float)-1.0, (float)2.0);
    float* got = zeros(m * n);
    float* want = zeros(m * n);
    unsafe {
        round_to(a, m * k, dt); round_to(b, k * n, dt); round_to(c0, m * n, ct);
        float alpha = (float)1.5;
        float beta = (float)-0.5;
        unsigned long lda = ta ? m : k;
        unsigned long ldb = tb ? k : n;
        unsigned long i = 0UL;
        while (i < m) {
            unsigned long j = 0UL;
            while (j < n) {
                double acc = 0.0;
                unsigned long q = 0UL;
                while (q < k) {
                    double av = (double)(ta ? a[q * lda + i] : a[i * lda + q]);
                    double bv = (double)(tb ? b[j * ldb + q] : b[q * ldb + j]);
                    acc = acc + av * bv;
                    q = q + 1UL;
                }
                want[i * n + j] = (float)((double)alpha * acc + (double)beta * (double)c0[i * n + j]);
                j = j + 1UL;
            }
            i = i + 1UL;
        }
        round_to(want, m * n, ct);
        // fp32 copies on the device, converted to the operand types there
        unsigned long fa = device_mem_alloc_(d, 4UL * m * k), fb = device_mem_alloc_(d, 4UL * k * n), fc = device_mem_alloc_(d, 4UL * m * n);
        unsigned long ha = dt == 0 ? fa : dev_tagged_(device_mem_alloc_(d, 2UL * m * k + 4UL), (unsigned long)dt);
        unsigned long hb = dt == 0 ? fb : dev_tagged_(device_mem_alloc_(d, 2UL * k * n + 4UL), (unsigned long)dt);
        unsigned long hc = ct == 0 ? fc : dev_tagged_(device_mem_alloc_(d, 2UL * m * n + 4UL), (unsigned long)ct);
        int ok = device_upload_(d, fa, (const &float)a, m * k) && device_upload_(d, fb, (const &float)b, k * n) &&
                 device_upload_(d, fc, (const &float)c0, m * n);
        if (ok && dt != 0) ok = device_copy_(d, ha, fa, m * k) && device_copy_(d, hb, fb, k * n);
        if (ok && ct != 0) ok = device_copy_(d, hc, fc, m * n);
        struct DevGemm g;
        g.dtype = dt; g.ctype = ct; g.ta = ta; g.tb = tb; g.m = m; g.n = n; g.k = k; g.alpha = alpha;
        g.a = ha; g.lda = lda; g.sa = 0UL; g.b = hb; g.ldb = ldb; g.sb = 0UL; g.beta = beta; g.c = hc; g.ldc = n; g.sc = 0UL;
        g.batch = 1UL;
        if (ok) ok = device_gemm_batched_(d, &g);
        if (ok && ct != 0) ok = device_copy_(d, fc, hc, m * n);
        if (ok) ok = device_download_(d, (&float)got, fc, m * n);
        double e = relerr(got, want, m * n);
        double tol = (dt == 0 && ct == 0) ? 1.0e-5 : (ct == WEBGPU_BF16 || dt == WEBGPU_BF16 ? 1.6e-2 : 2.0e-3);
        printf("  gemm %-22s %c%c %4lux%4lux%4lu err %.2e\n", what, ta ? 't' : 'n', tb ? 't' : 'n', m, n, k, e);
        check(ok && e < tol, "gemm path");
        device_mem_free_(d, fa, 4UL * m * k); device_mem_free_(d, fb, 4UL * k * n); device_mem_free_(d, fc, 4UL * m * n);
        if (dt != 0) { device_mem_free_(d, ha, 2UL * m * k + 4UL); device_mem_free_(d, hb, 2UL * k * n + 4UL); }
        if (ct != 0) device_mem_free_(d, hc, 2UL * m * n + 4UL);
        free((void*)a); free((void*)b); free((void*)c0); free((void*)got); free((void*)want);
    }
}

// Convolutions large enough for the backend's 8 x 8-block tiles (the
// forward pass in 64 x 256 and 128 x 128 tiles, the weight gradient in
// 128 x 128): output and gradients on DEVICE_WEBGPU against the CPU.
static float conv_err(const float* a, const float* b, unsigned long n) {
    float worst = (float)0.0;
    float scale = (float)1.0;
    unsafe {
        unsigned long i = 0UL;
        while (i < n) {
            float d = a[i] - b[i];
            if (d < (float)0.0) d = -d;
            if (d != d) d = (float)1e9;
            if (d > worst) worst = d;
            float m = a[i] < (float)0.0 ? -a[i] : a[i];
            if (m + (float)1.0 > scale) scale = m + (float)1.0;
            i = i + 1UL;
        }
    }
    return worst / scale;
}

static void conv_values(&FloatTensor t, float* out, int grad) {
    unsafe {
        if (grad) tensor_grad_host_((void*)t); else tensor_host_((void*)t);
        unsigned long i = 0UL;
        while (i < t->size) { out[i] = grad ? t->grad[i] : t->data[i]; i = i + 1UL; }
    }
}

static void check_conv_tiles(unsigned long batch, unsigned long ch, unsigned long side) {
    struct Rng rng = Rng::new(21UL);
    &FloatTensor x = FloatTensor::new_2d(batch, ch * side * side, 1);
    &FloatTensor w = FloatTensor::new_2d(ch, ch * 9UL, 1);
    &FloatTensor b = FloatTensor::new_1d(ch, 1);
    &FloatTensor R = FloatTensor::new_2d(batch, ch * side * side, 0);
    rng.fill_normal(x, (float)0.0, (float)1.0);
    rng.fill_uniform(w, (float)-0.1, (float)0.1);
    rng.fill_normal(b, (float)0.0, (float)0.1);
    rng.fill_normal(R, (float)0.0, (float)1.0);
    unsigned long no = batch * ch * side * side;
    unsigned long nw = ch * ch * 9UL;
    unsafe {
        float* want = (float*)malloc(4UL * (2UL * no + nw));
        float* got = (float*)malloc(4UL * (2UL * no + nw));
        int pass = 0;
        while (pass < 2) {
            float* v = pass == 0 ? want : got;
            if (pass == 1) { x.place(DEVICE_WEBGPU); w.place(DEVICE_WEBGPU); b.place(DEVICE_WEBGPU); }
            x.zero_grad(); w.zero_grad(); b.zero_grad();
            &FloatTensor o = tensor_conv2d(x, w, b, ch, side, side, 3UL, 3UL, 1UL, 1UL);
            if (pass == 1) check(o.device() == DEVICE_WEBGPU, "convolution on the device");
            tensor_mul(o, R).sum().backward();
            conv_values(o, v, 0);
            conv_values(x, &v[no], 1);
            conv_values(w, &v[2UL * no], 1);
            pass = pass + 1;
        }
        x.place(DEVICE_CPU); w.place(DEVICE_CPU); b.place(DEVICE_CPU);
        float eo = conv_err(want, got, no);
        float ex = conv_err(&want[no], &got[no], no);
        float ew = conv_err(&want[2UL * no], &got[2UL * no], nw);
        printf("  conv %lu->%lu side %lu batch %lu: output %.2e, dx %.2e, dw %.2e\n", ch, ch, side, batch, (double)eo, (double)ex, (double)ew);
        check(eo < (float)2e-4 && ex < (float)2e-4 && ew < (float)2e-4, "convolution in 8 x 8-block tiles");
        free((void*)want); free((void*)got);
    }
    x.free(); w.free(); b.free(); R.free();
}

int main() {
    unsafe { printf("webgpu available: %d\n", webgpu_available()); }
    if (!webgpu_available()) { unsafe { printf("webgpu arrays: no WebGPU adapter\n"); } return 1; }
    check_elementwise(1000UL);
    check_elementwise(1000003UL);   // more than 4096 workgroups' worth: grid-stride loops
    check_sum(1UL);
    check_sum(100000UL);
    check_sum(3000000UL);
    check_matmul(64UL, 64UL, 64UL, 0, 0);
    check_matmul(1UL, 1UL, 1UL, 0, 0);
    check_matmul(37UL, 129UL, 70UL, 0, 0);    // edge tiles on every side
    check_matmul(200UL, 300UL, 150UL, 0, 0);
    check_matmul(37UL, 129UL, 70UL, WEBGPU_FP16, 0);
    check_matmul(37UL, 129UL, 70UL, WEBGPU_BF16, 0);
    check_matmul(33UL, 64UL, 65UL, WEBGPU_FP16, WEBGPU_FP16);   // 16-bit results across odd rows
    check_matmul(33UL, 64UL, 65UL, WEBGPU_BF16, WEBGPU_BF16);
    check_buffers();
    if (device_available(DEVICE_WEBGPU)) {
        check_tensor_interop();
        check_gemm_path(0, 0, 33UL, 65UL, 70UL, 0, 0, "per element");
        check_gemm_path(1, 1, 90UL, 40UL, 300UL, 1, 1, "per element, 16-bit");
        check_gemm_path(0, 1, 200UL, 300UL, 150UL, 0, 0, "64 x 64 tiles");
        check_gemm_path(1, 0, 130UL, 260UL, 100UL, 2, 2, "64 x 64 tiles, 16-bit");
        check_gemm_path(0, 0, 100UL, 130UL, 1000UL, 0, 0, "split k");
        check_gemm_path(1, 1, 70UL, 150UL, 777UL, 0, 0, "split k");
        check_gemm_path(0, 0, 768UL, 770UL, 64UL, 0, 0, "128 x 128 tiles");
        check_gemm_path(1, 0, 768UL, 768UL, 40UL, 1, 1, "128 x 128 tiles, 16-bit");
        check_conv_tiles(32UL, 64UL, 16UL);    // forward: 64 x 256 tiles
        check_conv_tiles(64UL, 128UL, 8UL);    // forward and weight gradient: 128 x 128
    }
    unsafe { printf("webgpu arrays: %d checks, %d failures\n", g_checks, g_fails); }
    return g_fails == 0 ? 0 : 1;
}
