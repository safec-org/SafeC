// Correctness check for std/ml/gpu_cuda_attention against CPU reference
// results (double-precision loops here, and attention.h's own CPU flash /
// linear attention for single heads). Needs an NVIDIA GPU; run with
// tests/stdlib/cuda/run.sh.
#include <std/ml/gpu_cuda.h>
#include <std/ml/gpu_cuda_attention.h>
#include <std/ml/attention.h>
#include <std/ml/tensor.h>
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

// Max |a - b| relative to max(1, max |b|).
static double relerr(&Vec<float> a, &Vec<float> b) {
    double m = 0.0;
    double big = 1.0;
    unsigned long i = 0UL;
    while (i < b.length()) {
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

// Deterministic values in [-amp, amp), rounded to 'dtype' (0 = fp32) so the
// CPU reference sees exactly what the GPU reads.
static struct Vec<float> data(unsigned long n, unsigned long seed, float amp, int dtype) {
    struct Vec<float> v = std::vec_with_cap(n);
    unsigned long i = 0UL;
    while (i < n) {
        float x = ((float)((i * 7919UL + seed * 104729UL) % 2003UL) / (float)1001.5 - (float)1.0) * amp;
        if (dtype == CUDA_FP16) x = std::fp16_to_f32(std::f32_to_fp16(x));
        if (dtype == CUDA_BF16) x = std::bf16_to_f32(std::f32_to_bf16(x));
        v.push(x);
        i = i + 1UL;
    }
    return v;
}

static double phi_d(double x) {
    if (x > 0.0) return x + 1.0;
    return std::exp_d(x);
}

// Softmax (lin = 0) or linear (lin = 1) attention, batched multi-head,
// in double, with the header's layout and bottom-right causal mask.
static struct Vec<float> ref_attn(&Vec<float> q, &Vec<float> k, &Vec<float> v, struct CudaAttnShape s,
                                  int causal, int lin) {
    unsigned long D = s.headDim;
    struct Vec<float> out = std::vec_with_cap(s.batch * s.heads * s.seqLen * D);
    struct Vec<double> w = std::vec_with_cap(s.kvLen);
    unsigned long i = 0UL;
    while (i < s.kvLen) { w.push(0.0); i = i + 1UL; }
    double scale = 1.0 / std::sqrt_d((double)D);
    long off = (long)s.kvLen - (long)s.seqLen;
    unsigned long b = 0UL;
    while (b < s.batch) {
        unsigned long h = 0UL;
        while (h < s.heads) {
            unsigned long kh = h / (s.heads / s.kvHeads);
            unsigned long qb = (b * s.heads + h) * s.seqLen * D;
            unsigned long kb = (b * s.kvHeads + kh) * s.kvLen * D;
            unsigned long r = 0UL;
            while (r < s.seqLen) {
                long jmax = (long)s.kvLen - 1L;
                if (causal && (long)r + off < jmax) jmax = (long)r + off;
                long jmin = 0L;
                if (s.window > 0UL && (long)r + off - (long)s.window + 1L > 0L) jmin = (long)r + off - (long)s.window + 1L;
                double mx = -1.0e300;
                double sum = 0.0;
                long j = jmin;
                while (j <= jmax) {
                    double dot = 0.0;
                    unsigned long d = 0UL;
                    while (d < D) {
                        double qv = (double)*q.at(qb + r * D + d);
                        double kv = (double)*k.at(kb + (unsigned long)j * D + d);
                        if (lin) dot = dot + phi_d(qv) * phi_d(kv);
                        else dot = dot + qv * kv;
                        d = d + 1UL;
                    }
                    if (lin) { *w.at((unsigned long)j) = dot; sum = sum + dot; }
                    else { *w.at((unsigned long)j) = dot * scale; if (dot * scale > mx) mx = dot * scale; }
                    j = j + 1L;
                }
                if (!lin) {
                    j = jmin;
                    while (j <= jmax) {
                        double e = std::exp_d(*w.at((unsigned long)j) - mx);
                        *w.at((unsigned long)j) = e;
                        sum = sum + e;
                        j = j + 1L;
                    }
                }
                unsigned long d = 0UL;
                while (d < D) {
                    double a = 0.0;
                    j = jmin;
                    while (j <= jmax) {
                        a = a + *w.at((unsigned long)j) * (double)*v.at(kb + (unsigned long)j * D + d);
                        j = j + 1L;
                    }
                    out.push(jmax >= jmin ? (float)(a / sum) : (float)0.0);
                    d = d + 1UL;
                }
                r = r + 1UL;
            }
            h = h + 1UL;
        }
        b = b + 1UL;
    }
    w.free();
    return out;
}

static struct CudaAttnShape shape(unsigned long b, unsigned long h, unsigned long hkv,
                                  unsigned long sq, unsigned long sk, unsigned long d) {
    struct CudaAttnShape s;
    s.batch = b; s.heads = h; s.kvHeads = hkv; s.seqLen = sq; s.kvLen = sk; s.headDim = d; s.window = 0UL;
    return s;
}

static struct CudaAttnShape windowed(struct CudaAttnShape s, unsigned long w) {
    s.window = w;
    return s;
}

// Runs one GPU attention call in 'dtype' (0 = fp32) and compares with the
// double reference. lin: 0 flash, 1 linear.
static void run_case(const char* what, struct CudaAttnShape s, int dtype, int causal, int lin, double tol) {
    unsigned long qn = s.batch * s.heads * s.seqLen * s.headDim;
    unsigned long kn = s.batch * s.kvHeads * s.kvLen * s.headDim;
    float amp = lin ? (float)0.5 : (float)1.0;
    struct Vec<float> q = data(qn, 1UL, amp, dtype);
    struct Vec<float> k = data(kn, 2UL, amp, dtype);
    struct Vec<float> v = data(kn, 3UL, (float)1.0, dtype);
    struct Vec<float> want = ref_attn(&q, &k, &v, s, causal, lin);
    struct Vec<float> got = std::vec_with_cap(qn);
    unsigned long i = 0UL;
    while (i < qn) { got.push(0.0f); i = i + 1UL; }
    int ok = 0;
    if (dtype == 0) {
        struct CudaBuf dq = std::cuda_upload(q.at(0UL), qn);
        struct CudaBuf dk = std::cuda_upload(k.at(0UL), kn);
        struct CudaBuf dv = std::cuda_upload(v.at(0UL), kn);
        struct CudaBuf dout = std::cuda_alloc(qn);
        ok = lin ? std::cuda_linear_attention_ex(dq, dk, dv, dout, s, causal)
                 : std::cuda_flash_attention_ex(dq, dk, dv, dout, s, 0.0f, causal);
        ok = ok && std::cuda_read(dout, got.at(0UL), qn);
        std::cuda_free(&dq); std::cuda_free(&dk); std::cuda_free(&dv); std::cuda_free(&dout);
    } else {
        struct CudaHalfBuf hq = std::cuda_half_upload(q.at(0UL), qn, dtype);
        struct CudaHalfBuf hk = std::cuda_half_upload(k.at(0UL), kn, dtype);
        struct CudaHalfBuf hv = std::cuda_half_upload(v.at(0UL), kn, dtype);
        struct CudaHalfBuf hout = std::cuda_half_alloc(qn, dtype);
        struct CudaBuf dout = std::cuda_alloc(qn);
        ok = lin ? std::cuda_half_linear_attention(hq, hk, hv, hout, s, causal)
                 : std::cuda_half_flash_attention(hq, hk, hv, hout, s, 0.0f, causal);
        ok = ok && std::cuda_from_half(hout, dout) && std::cuda_read(dout, got.at(0UL), qn);
        std::cuda_half_free(&hq); std::cuda_half_free(&hk); std::cuda_half_free(&hv); std::cuda_half_free(&hout);
        std::cuda_free(&dout);
    }
    check(ok, what);
    double e = relerr(&got, &want);
    check(ok && e <= tol, what);
    unsafe { printf("  %-46s err %.2e (tol %.0e)\n", what, e, tol); }
    q.free(); k.free(); v.free(); want.free(); got.free();
}

int main() {
    if (!std::cuda_available()) { unsafe { printf("no CUDA GPU -- skipped\n"); } return 0; }
    unsafe { printf("device: %s (sm_%d)\n", std::cuda_device_name(), std::cuda_compute_capability()); }

    // Flash attention: fp32 (CUDA cores), fp16/bf16 (tensor cores up to D 128,
    // CUDA cores at 256); unaligned lengths, GQA 4/2, KV-cache offset.
    int c = 0;
    while (c < 2) {
        run_case(c ? "flash fp32 D64 GQA causal" : "flash fp32 D64 GQA", shape(2UL, 4UL, 2UL, 77UL, 77UL, 64UL), 0, c, 0, 2.0e-5);
        run_case(c ? "flash fp32 D128 kv-cache causal" : "flash fp32 D128 kv-cache", shape(1UL, 2UL, 1UL, 50UL, 130UL, 128UL), 0, c, 0, 2.0e-5);
        run_case(c ? "flash fp16 D64 GQA causal" : "flash fp16 D64 GQA", shape(2UL, 4UL, 2UL, 77UL, 77UL, 64UL), CUDA_FP16, c, 0, 4.0e-3);
        run_case(c ? "flash bf16 D128 kv-cache causal" : "flash bf16 D128 kv-cache", shape(1UL, 4UL, 2UL, 100UL, 200UL, 128UL), CUDA_BF16, c, 0, 2.5e-2);
        run_case(c ? "flash fp16 D96 causal" : "flash fp16 D96", shape(1UL, 2UL, 2UL, 65UL, 65UL, 96UL), CUDA_FP16, c, 0, 4.0e-3);
        run_case(c ? "flash bf16 D256 (CUDA cores) causal" : "flash bf16 D256 (CUDA cores)", shape(1UL, 2UL, 1UL, 40UL, 40UL, 256UL), CUDA_BF16, c, 0, 2.5e-2);
        run_case(c ? "flash fp32 D32 causal" : "flash fp32 D32", shape(1UL, 3UL, 1UL, 33UL, 97UL, 32UL), 0, c, 0, 2.0e-5);
        run_case(c ? "flash fp32 D256 (CUDA cores) causal" : "flash fp32 D256 (CUDA cores)", shape(1UL, 2UL, 1UL, 40UL, 72UL, 256UL), 0, c, 0, 2.0e-5);
        run_case(c ? "linear fp32 D256 (sequential) causal" : "linear fp32 D256 (two-pass)", shape(1UL, 2UL, 2UL, 50UL, 90UL, 256UL), 0, c, 1, 2.0e-5);
        run_case(c ? "linear bf16 D64 long causal" : "linear bf16 D64 long", shape(1UL, 2UL, 1UL, 300UL, 1000UL, 64UL), CUDA_BF16, c, 1, 2.5e-2);
        // Linear attention
        run_case(c ? "linear fp32 D64 GQA causal" : "linear fp32 D64 GQA", shape(2UL, 4UL, 2UL, 77UL, 77UL, 64UL), 0, c, 1, 2.0e-5);
        run_case(c ? "linear fp32 D96 kv-cache causal" : "linear fp32 D96 kv-cache", shape(1UL, 2UL, 2UL, 65UL, 130UL, 96UL), 0, c, 1, 2.0e-5);
        run_case(c ? "linear fp16 D128 causal" : "linear fp16 D128", shape(1UL, 2UL, 1UL, 50UL, 50UL, 128UL), CUDA_FP16, c, 1, 4.0e-3);
        run_case(c ? "linear bf16 D256 causal" : "linear bf16 D256", shape(1UL, 2UL, 1UL, 40UL, 300UL, 256UL), CUDA_BF16, c, 1, 2.5e-2);
        run_case(c ? "linear fp32 D32 causal" : "linear fp32 D32", shape(1UL, 2UL, 2UL, 33UL, 97UL, 32UL), 0, c, 1, 2.0e-5);
        c = c + 1;
    }
    // The same 16-bit flash cases through this library's own kernels (the
    // loop above used cuDNN's fused attention where available).
    int hadCudnn = std::cuda_attention_use_cudnn(1);
    std::cuda_attention_use_cudnn(0);
    unsafe { printf("  (cuDNN fused attention: %s)\n", hadCudnn ? "used above" : "not available"); }
    c = 0;
    while (c < 2) {
        run_case(c ? "own flash fp16 D64 GQA causal" : "own flash fp16 D64 GQA", shape(2UL, 4UL, 2UL, 77UL, 77UL, 64UL), CUDA_FP16, c, 0, 4.0e-3);
        run_case(c ? "own flash bf16 D128 kv-cache causal" : "own flash bf16 D128 kv-cache", shape(1UL, 4UL, 2UL, 100UL, 200UL, 128UL), CUDA_BF16, c, 0, 2.5e-2);
        run_case(c ? "own flash fp16 D96 causal" : "own flash fp16 D96", shape(1UL, 2UL, 2UL, 65UL, 65UL, 96UL), CUDA_FP16, c, 0, 4.0e-3);
        c = c + 1;
    }
    run_case("own flash fp16 D64 window 17", windowed(shape(1UL, 2UL, 2UL, 90UL, 90UL, 64UL), 17UL), CUDA_FP16, 0, 0, 4.0e-3);
    run_case("own flash bf16 D128 window 100 causal", windowed(shape(1UL, 2UL, 2UL, 130UL, 130UL, 128UL), 100UL), CUDA_BF16, 1, 0, 2.5e-2);
    std::cuda_attention_use_cudnn(1);
    // Sliding window (flash), with and without causal, narrower and wider than a tile
    run_case("flash fp32 D64 window 40 causal", windowed(shape(1UL, 2UL, 1UL, 150UL, 200UL, 64UL), 40UL), 0, 1, 0, 2.0e-5);
    run_case("flash fp16 D64 window 17", windowed(shape(1UL, 2UL, 2UL, 90UL, 90UL, 64UL), 17UL), CUDA_FP16, 0, 0, 4.0e-3);
    run_case("flash bf16 D128 window 100 causal", windowed(shape(1UL, 2UL, 2UL, 130UL, 130UL, 128UL), 100UL), CUDA_BF16, 1, 0, 2.5e-2);

    // Same result as attention.h's CPU flash / linear attention for one head.
    struct CudaAttnShape one = shape(1UL, 1UL, 1UL, 40UL, 56UL, 64UL);
    unsigned long qn = 40UL * 64UL;
    unsigned long kn = 56UL * 64UL;
    struct Vec<float> q = data(qn, 4UL, (float)0.5, 0);
    struct Vec<float> k = data(kn, 5UL, (float)0.5, 0);
    struct Vec<float> v = data(kn, 6UL, (float)1.0, 0);
    struct CudaBuf dq = std::cuda_upload(q.at(0UL), qn);
    struct CudaBuf dk = std::cuda_upload(k.at(0UL), kn);
    struct CudaBuf dv = std::cuda_upload(v.at(0UL), kn);
    struct CudaBuf dout = std::cuda_alloc(qn);
    struct Vec<float> got = std::vec_with_cap(qn);
    struct Vec<float> cpu = std::vec_with_cap(qn);
    unsigned long i = 0UL;
    while (i < qn) { got.push(0.0f); cpu.push(0.0f); i = i + 1UL; }
    &Tensor tq = std::tensor_from_2d(q.at(0UL), 40UL, 64UL, 0);
    &Tensor tk = std::tensor_from_2d(k.at(0UL), 56UL, 64UL, 0);
    &Tensor tv = std::tensor_from_2d(v.at(0UL), 56UL, 64UL, 0);
    int l = 0;
    while (l < 2) {
        &Tensor tr = l ? std::linear_attention_forward(tq, tk, tv) : std::flash_attention_forward(tq, tk, tv, 16UL);
        i = 0UL;
        while (i < qn) { unsafe { *cpu.at(i) = tr->data[i]; } i = i + 1UL; }
        tr.free();
        // the rocm_*-compatible argument form
        int ok = l ? std::cuda_linear_attention(dq, dk, dv, dout, 1UL, 40UL, 56UL, 64UL, 0)
                   : std::cuda_flash_attention(dq, dk, dv, dout, 1UL, 40UL, 56UL, 64UL, (float)0.0, 0);
        ok = ok && std::cuda_read(dout, got.at(0UL), qn);
        double e = relerr(&got, &cpu);
        check(ok && e <= 1.0e-5, l ? "linear matches attention.h" : "flash matches attention.h");
        unsafe { printf("  %-46s err %.2e\n", l ? "linear vs attention.h linear_attention_forward" : "flash vs attention.h flash_attention_forward", e); }
        l = l + 1;
    }
    tq.free(); tk.free(); tv.free();

    // Host-array fp32 versions, 2 heads, causal
    struct CudaAttnShape two = shape(1UL, 2UL, 2UL, 20UL, 28UL, 64UL);
    l = 0;
    while (l < 2) {
        struct Vec<float> want2 = ref_attn(&q, &k, &v, two, 1, l);
        int ok = l ? std::cuda_linear_attention_f32(q.at(0UL), k.at(0UL), v.at(0UL), got.at(0UL), 2UL, 20UL, 28UL, 64UL, 1)
                   : std::cuda_flash_attention_f32(q.at(0UL), k.at(0UL), v.at(0UL), got.at(0UL), 2UL, 20UL, 28UL, 64UL, (float)0.0, 1);
        struct Vec<float> got2 = std::vec_with_cap(want2.length());
        i = 0UL;
        while (i < want2.length()) { got2.push(*got.at(i)); i = i + 1UL; }
        check(ok && relerr(&got2, &want2) <= 2.0e-5, l ? "linear _f32 host arrays" : "flash _f32 host arrays");
        want2.free(); got2.free();
        l = l + 1;
    }

    // Graph capture: both ops recorded once (their variants are built above) and replayed.
    struct CudaBuf dout2 = std::cuda_alloc(qn);
    check(std::cuda_capture_begin(), "attention capture begin");
    std::cuda_flash_attention_ex(dq, dk, dv, dout, one, (float)0.0, 1);
    std::cuda_linear_attention_ex(dq, dk, dv, dout2, one, 0);
    struct CudaGraph g = std::cuda_capture_end();
    check(g.exec != 0UL, "attention capture end");
    check(std::cuda_graph_launch(g) && std::cuda_graph_launch(g), "attention graph launch");
    struct Vec<float> want = ref_attn(&q, &k, &v, one, 1, 0);
    check(std::cuda_read(dout, got.at(0UL), qn) && relerr(&got, &want) <= 2.0e-5, "graph replay flash");
    want.free();
    want = ref_attn(&q, &k, &v, one, 0, 1);
    check(std::cuda_read(dout2, got.at(0UL), qn) && relerr(&got, &want) <= 2.0e-5, "graph replay linear");
    want.free();
    std::cuda_graph_free(&g);

    // Rejected shapes
    check(!std::cuda_flash_attention_ex(dq, dk, dv, dout, shape(1UL, 1UL, 1UL, 40UL, 56UL, 48UL), (float)0.0, 0), "head dim 48 rejected");
    check(!std::cuda_flash_attention_ex(dq, dk, dv, dout, shape(1UL, 3UL, 2UL, 4UL, 4UL, 32UL), (float)0.0, 0), "heads % kvHeads rejected");
    check(!std::cuda_flash_attention(dq, dk, dv, dout, 1UL, 41UL, 56UL, 64UL, (float)0.0, 0), "short buffer rejected");
    check(!std::cuda_linear_attention(dq, dk, dv, dout, 1UL, 40UL, 20UL, 64UL, 1), "causal linear kvLen < seqLen rejected");
    check(!std::cuda_linear_attention_ex(dq, dk, dv, dout, windowed(one, 8UL), 0), "linear with window rejected");
    check(std::cuda_attention_head_dim_ok(128UL) && !std::cuda_attention_head_dim_ok(80UL), "head dim query");

    std::cuda_free(&dq); std::cuda_free(&dk); std::cuda_free(&dv); std::cuda_free(&dout); std::cuda_free(&dout2);
    q.free(); k.free(); v.free(); got.free(); cpu.free();
    unsafe { printf("cuda attention: %d checks, %d failures\n", g_checks, g_fails); }
    return g_fails == 0 ? 0 : 1;
}
