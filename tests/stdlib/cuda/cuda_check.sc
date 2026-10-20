// Correctness check for std/ml/gpu_cuda against CPU reference results.
// Needs an NVIDIA GPU; run with tests/stdlib/cuda/run.sh.
#include <std/ml/gpu_cuda.h>
#include <std/ml/float16.h>
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

static float maxdiff(&Vec<float> a, &Vec<float> b) {
    float m = 0.0f;
    unsigned long i = 0UL;
    while (i < a.length()) {
        float d = *a.at(i) - *b.at(i);
        if (d < 0.0f) d = -d;
        if (d > m) m = d;
        i = i + 1UL;
    }
    return m;
}

static struct Vec<float> seq(unsigned long n, float scale, float offset) {
    struct Vec<float> v = std::vec_with_cap(n);
    unsigned long i = 0UL;
    while (i < n) { v.push((float)((i * 37UL) % 101UL) * scale + offset); i = i + 1UL; }
    return v;
}

// CPU reference: c[m,n] = op(a) op(b), row-major, ta/tb = stored transposed.
static struct Vec<float> cpu_gemm(&Vec<float> a, int ta, &Vec<float> b, int tb,
                                  unsigned long m, unsigned long n, unsigned long k) {
    struct Vec<float> c = std::vec_with_cap(m * n);
    unsigned long i = 0UL;
    while (i < m) {
        unsigned long j = 0UL;
        while (j < n) {
            float s = 0.0f;
            unsigned long p = 0UL;
            while (p < k) {
                float x = ta ? *a.at(p * m + i) : *a.at(i * k + p);
                float y = tb ? *b.at(j * k + p) : *b.at(p * n + j);
                s = s + x * y;
                p = p + 1UL;
            }
            c.push(s);
            j = j + 1UL;
        }
        i = i + 1UL;
    }
    return c;
}

static struct Vec<float> download(struct CudaBuf b) {
    struct Vec<float> v = std::vec_with_cap(b.len);
    unsigned long i = 0UL;
    while (i < b.len) { v.push(0.0f); i = i + 1UL; }
    std::cuda_read(b, v.at(0UL), b.len);
    return v;
}

int main() {
    if (!std::cuda_available()) { unsafe { printf("no CUDA GPU -- skipped\n"); } return 0; }
    unsafe { printf("device: %s\n", std::cuda_device_name()); }

    unsigned long n = 10007UL;   // not a multiple of the block size
    struct Vec<float> a = seq(n, 0.01f, 0.1f);
    struct Vec<float> b = seq(n, 0.02f, 0.5f);
    struct CudaBuf da = std::cuda_upload(a.at(0UL), n);
    struct CudaBuf db = std::cuda_upload(b.at(0UL), n);
    struct CudaBuf dout = std::cuda_alloc(n);
    check(da.dev != 0UL && db.dev != 0UL && dout.dev != 0UL, "upload/alloc");

    // elementwise vs CPU
    struct Vec<float> want = std::vec_with_cap(n);
    unsigned long i = 0UL;
    while (i < n) { want.push(*a.at(i) + *b.at(i)); i = i + 1UL; }
    check(std::cuda_add(da, db, dout), "add enqueued");
    struct Vec<float> got = download(dout);
    check(maxdiff(&got, &want) < 1e-5f, "add values");
    got.free();

    std::cuda_mul(da, db, dout);
    got = download(dout);
    i = 0UL;
    while (i < n) { *want.at(i) = *a.at(i) * *b.at(i); i = i + 1UL; }
    check(maxdiff(&got, &want) < 1e-4f, "mul values");
    got.free();

    std::cuda_scale(da, -3.0f, dout);
    std::cuda_relu(dout, dout);   // in place: all zero
    got = download(dout);
    i = 0UL;
    while (i < n) { *want.at(i) = 0.0f; i = i + 1UL; }
    check(maxdiff(&got, &want) == 0.0f, "scale + relu in place");
    got.free();

    std::cuda_exp(da, dout);
    got = download(dout);
    i = 0UL;
    while (i < n) { *want.at(i) = std::exp_f(*a.at(i)); i = i + 1UL; }
    check(maxdiff(&got, &want) < 1e-3f, "exp values");
    got.free();

    // sum
    struct CudaBuf d1 = std::cuda_alloc(1UL);
    std::cuda_sum(da, d1);
    got = download(d1);
    double cpuSum = 0.0;
    i = 0UL;
    while (i < n) { cpuSum = cpuSum + (double)*a.at(i); i = i + 1UL; }
    double rel = ((double)*got.at(0UL) - cpuSum) / cpuSum;
    if (rel < 0.0) rel = -rel;
    check(rel < 1e-5, "sum");
    got.free();

    // fused MSE gradient
    std::cuda_mse_grad(da, db, dout, d1);
    got = download(dout);
    struct Vec<float> lossv = download(d1);
    double cpuLoss = 0.0;
    i = 0UL;
    while (i < n) {
        float d = *a.at(i) - *b.at(i);
        *want.at(i) = (float)2.0 * d;
        cpuLoss = cpuLoss + (double)(d * d);
        i = i + 1UL;
    }
    check(maxdiff(&got, &want) < 1e-5f, "mse grad values");
    rel = ((double)*lossv.at(0UL) - cpuLoss) / cpuLoss;
    if (rel < 0.0) rel = -rel;
    check(rel < 1e-4, "mse loss");
    got.free();
    lossv.free();

    // sgd in place
    std::cuda_sgd(da, db, 0.5f);
    got = download(da);
    i = 0UL;
    while (i < n) { *want.at(i) = *a.at(i) - (float)0.5 * *b.at(i); i = i + 1UL; }
    check(maxdiff(&got, &want) < 1e-5f, "sgd");
    got.free();

    // size checks
    struct CudaBuf small = std::cuda_alloc(10UL);
    check(!std::cuda_add(small, db, dout), "length mismatch rejected");

    // gemm, every transpose combination, odd sizes
    unsigned long m = 67UL;
    unsigned long nn = 45UL;
    unsigned long k = 129UL;
    struct Vec<float> ga = seq(m * k, 0.003f, -0.1f);
    struct Vec<float> gb = seq(k * nn, 0.002f, -0.07f);
    struct CudaBuf dga = std::cuda_upload(ga.at(0UL), m * k);
    struct CudaBuf dgb = std::cuda_upload(gb.at(0UL), k * nn);
    struct CudaBuf dgc = std::cuda_alloc(m * nn);
    int t = 0;
    while (t < 4) {
        int ta = t & 1;
        int tb = (t >> 1) & 1;
        struct Vec<float> ref = cpu_gemm(&ga, ta, &gb, tb, m, nn, k);
        check(std::cuda_gemm(dga, ta, dgb, tb, dgc, m, nn, k), "gemm enqueued");
        struct Vec<float> gc = download(dgc);
        check(maxdiff(&gc, &ref) < 1e-4f, ta ? (tb ? "gemm tt" : "gemm tn") : (tb ? "gemm nt" : "gemm nn"));
        gc.free();
        ref.free();
        t = t + 1;
    }
    check(!std::cuda_gemm(dga, 0, dgb, 0, dgc, m, nn, k + 1UL), "gemm size check");

    // TF32 tensor cores: same product to ~1e-3 relative
    check(std::cuda_set_tf32(1), "tf32 on");
    struct Vec<float> r32 = cpu_gemm(&ga, 0, &gb, 0, m, nn, k);
    check(std::cuda_gemm(dga, 0, dgb, 0, dgc, m, nn, k), "tf32 gemm enqueued");
    struct Vec<float> g32 = download(dgc);
    check(maxdiff(&g32, &r32) < 1e-2f, "tf32 gemm values");
    g32.free();
    check(std::cuda_gemm_relu(dga, 0, dgb, 0, dgc, m, nn, k), "tf32 gemm_relu enqueued");
    check(std::cuda_set_tf32(0), "tf32 off");
    r32.free();

    // host-array API: custom kernel and cuBLAS agree with CPU
    struct Vec<float> ref = cpu_gemm(&ga, 0, &gb, 0, m, nn, k);
    struct Vec<float> hc = std::vec_with_cap(m * nn);
    i = 0UL;
    while (i < m * nn) { hc.push(0.0f); i = i + 1UL; }
    unsafe {
        check(std::cuda_matmul_f32((const float*)ga.at(0UL), (const float*)gb.at(0UL), (float*)hc.at(0UL), m, k, nn), "custom matmul ran");
    }
    check(maxdiff(&hc, &ref) < 1e-4f, "custom matmul values");
    unsafe {
        check(std::cuda_matmul_f32_blas((const float*)ga.at(0UL), (const float*)gb.at(0UL), (float*)hc.at(0UL), m, k, nn), "blas matmul ran");
    }
    check(maxdiff(&hc, &ref) < 1e-4f, "blas matmul values");

    // fused gemm + relu (cuBLASLt epilogue), every transpose combination
    t = 0;
    while (t < 4) {
        int ta = t & 1;
        int tb = (t >> 1) & 1;
        struct Vec<float> rr = cpu_gemm(&ga, ta, &gb, tb, m, nn, k);
        i = 0UL;
        while (i < m * nn) { if (*rr.at(i) < 0.0f) *rr.at(i) = 0.0f; i = i + 1UL; }
        check(std::cuda_gemm_relu(dga, ta, dgb, tb, dgc, m, nn, k), "gemm_relu enqueued");
        struct Vec<float> gc = download(dgc);
        check(maxdiff(&gc, &rr) < 1e-4f, ta ? (tb ? "gemm_relu tt" : "gemm_relu tn") : (tb ? "gemm_relu nt" : "gemm_relu nn"));
        gc.free();
        rr.free();
        t = t + 1;
    }

    // relu_grad against relu output and sgd2 over two buffers
    std::cuda_sub(da, db, dout);                 // mixed signs
    struct CudaBuf dg = std::cuda_alloc(n);
    std::cuda_relu_grad(dout, da, dg);
    got = download(dg);
    i = 0UL;
    while (i < n) { float z = *a.at(i) - *b.at(i); *want.at(i) = 0.0f; if (z > 0.0f) *want.at(i) = *a.at(i); i = i + 1UL; }
    check(maxdiff(&got, &want) == 0.0f, "relu_grad values");
    got.free();
    struct CudaBuf w1 = std::cuda_upload(a.at(0UL), n);
    struct CudaBuf w2 = std::cuda_upload(ga.at(0UL), m * k);
    check(std::cuda_sgd2(w1, db, w2, dga, 0.25f), "sgd2 enqueued");
    got = download(w1);
    i = 0UL;
    while (i < n) { *want.at(i) = *a.at(i) - (float)0.25 * *b.at(i); i = i + 1UL; }
    check(maxdiff(&got, &want) < 1e-6f, "sgd2 first tensor");
    got.free();
    got = download(w2);
    float w2err = 0.0f;
    i = 0UL;
    while (i < m * k) {
        float d = *got.at(i) - (*ga.at(i) - (float)0.25 * *ga.at(i));
        if (d < 0.0f) d = -d;
        if (d > w2err) w2err = d;
        i = i + 1UL;
    }
    check(w2err < 1e-6f, "sgd2 second tensor");
    got.free();

    // older host / raw-pointer API on the same backend
    struct Vec<float> hv = std::vec_with_cap(n);
    i = 0UL;
    while (i < n) { hv.push(0.0f); i = i + 1UL; }
    unsafe {
        check(std::cuda_add_f32((const float*)a.at(0UL), (const float*)b.at(0UL), (float*)hv.at(0UL), n), "add_f32 ran");
    }
    i = 0UL;
    while (i < n) { *want.at(i) = *a.at(i) + *b.at(i); i = i + 1UL; }
    check(maxdiff(&hv, &want) < 1e-5f, "add_f32 values");
    float hsum = 0.0f;
    unsafe { check(std::cuda_sum_f32((const float*)a.at(0UL), &hsum, n), "sum_f32 ran"); }
    rel = ((double)hsum - cpuSum) / cpuSum;
    if (rel < 0.0) rel = -rel;
    check(rel < 1e-5, "sum_f32 value");
    unsafe {
        void* pa = std::cuda_upload_persistent((const float*)ga.at(0UL), m * k * 4UL);
        void* pb = std::cuda_upload_persistent((const float*)gb.at(0UL), k * nn * 4UL);
        check(pa != (void*)0 && pb != (void*)0, "persistent upload");
        check(std::cuda_matmul_f32_persistent(pa, pb, (float*)hc.at(0UL), m, k, nn), "persistent matmul ran");
        check(maxdiff(&hc, &ref) < 1e-4f, "persistent matmul values");
        check(std::cuda_sgd_update_f32(pa, pa, 1.0f, m * k), "persistent sgd");   // pa -= pa -> 0
        check(std::cuda_matmul_f32_persistent(pa, pb, (float*)hc.at(0UL), m, k, nn), "persistent matmul after sgd");
        float zmax = 0.0f;
        i = 0UL;
        while (i < m * nn) { float v = *hc.at(i); if (v < 0.0f) v = -v; if (v > zmax) zmax = v; i = i + 1UL; }
        check(zmax == 0.0f, "persistent sgd values");
        std::cuda_release_persistent(pa);
        std::cuda_release_persistent(pb);

        // fp16 / bf16: compare with the fp32 product at 16-bit tolerance
        unsigned short* raw = (unsigned short*)std::alloc(m * nn * 2UL);
        float* back = (float*)std::alloc(m * nn * 4UL);
        int bf = 0;
        while (bf < 2) {
            void* ha = bf ? std::cuda_upload_bf16_persistent((const float*)ga.at(0UL), m * k)
                          : std::cuda_upload_f16_persistent((const float*)ga.at(0UL), m * k);
            void* hb = bf ? std::cuda_upload_bf16_persistent((const float*)gb.at(0UL), k * nn)
                          : std::cuda_upload_f16_persistent((const float*)gb.at(0UL), k * nn);
            int ran = bf ? std::cuda_matmul_bf16_persistent(ha, hb, raw, m, k, nn)
                         : std::cuda_matmul_f16_persistent(ha, hb, raw, m, k, nn);
            check(ran, bf ? "bf16 matmul ran" : "f16 matmul ran");
            if (bf) std::bf16_to_f32_bulk((const unsigned short*)raw, back, m * nn);
            else std::fp16_to_f32_bulk((const unsigned short*)raw, back, m * nn);
            float err = 0.0f;
            i = 0UL;
            while (i < m * nn) { float d = back[i] - *ref.at(i); if (d < 0.0f) d = -d; if (d > err) err = d; i = i + 1UL; }
            check(err < (bf ? 0.05f : 0.01f), bf ? "bf16 matmul values" : "f16 matmul values");
            check(bf ? std::cuda_sgd_update_bf16(ha, ha, 1.0f, m * k) : std::cuda_sgd_update_f16(ha, ha, 1.0f, m * k),
                  bf ? "bf16 sgd" : "f16 sgd");
            int ran2 = bf ? std::cuda_matmul_bf16_persistent(ha, hb, raw, m, k, nn)
                          : std::cuda_matmul_f16_persistent(ha, hb, raw, m, k, nn);
            int allZero = ran2;
            i = 0UL;
            while (i < m * nn) { if ((raw[i] & (unsigned short)32767) != (unsigned short)0) allZero = 0; i = i + 1UL; }
            check(allZero, bf ? "bf16 sgd values" : "f16 sgd values");
            std::cuda_release_persistent(ha);
            std::cuda_release_persistent(hb);
            bf = bf + 1;
        }
        std::dealloc((void*)raw);
        std::dealloc((void*)back);
    }
    std::cuda_free(&dg); std::cuda_free(&w1); std::cuda_free(&w2);
    hv.free();

    // graph: record add + scale, replay twice
    check(std::cuda_capture_begin(), "capture begin");
    std::cuda_add(db, db, dout);
    std::cuda_scale(dout, 0.25f, dout);
    struct CudaGraph g = std::cuda_capture_end();
    check(g.exec != 0UL, "capture end");
    check(std::cuda_graph_launch(g) && std::cuda_graph_launch(g), "graph launch");
    got = download(dout);
    i = 0UL;
    while (i < n) { *want.at(i) = *b.at(i) * (float)0.5; i = i + 1UL; }
    check(maxdiff(&got, &want) < 1e-5f, "graph replay values");
    got.free();
    std::cuda_graph_free(&g);

    std::cuda_free(&da); std::cuda_free(&db); std::cuda_free(&dout); std::cuda_free(&d1);
    std::cuda_free(&small); std::cuda_free(&dga); std::cuda_free(&dgb); std::cuda_free(&dgc);
    check(da.dev == 0UL, "free clears handle");
    a.free(); b.free(); want.free(); ga.free(); gb.free(); ref.free(); hc.free();

    unsafe { printf("cuda: %d checks, %d failures\n", g_checks, g_fails); }
    return g_fails == 0 ? 0 : 1;
}
