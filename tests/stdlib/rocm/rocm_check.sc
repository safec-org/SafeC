// Correctness check for std/ml/gpu_rocm against CPU reference results.
// Needs a ROCm GPU; run with tests/stdlib/rocm/run.sh.
#include <std/ml/gpu_rocm.h>
#include <std/collections/vec.h>
#include <std/math.h>
#include <std/ml/tensor_half.h>
#include <std/ml/float16.h>
#include <std/ml/attention.h>
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

static struct Vec<float> download(struct RocmBuf b) {
    struct Vec<float> v = std::vec_with_cap(b.len);
    unsigned long i = 0UL;
    while (i < b.len) { v.push(0.0f); i = i + 1UL; }
    std::rocm_read(b, v.at(0UL), b.len);
    return v;
}

// fp16 / bf16: GPU rounding matches float16.h bit for bit; 16-bit gemm
// against an fp32 CPU gemm of the same rounded inputs; 16-bit SGD exact.
static void check_half() {
    unsigned long n = 4096UL;
    struct Vec<float> v = std::vec_with_cap(n);
    v.push((float)0.0); v.push((float)-0.0); v.push((float)1.0); v.push((float)-1.0);
    v.push((float)65504.0);          // largest fp16
    v.push((float)65520.0);          // rounds to fp16 infinity
    v.push((float)1.0e-8);           // below the fp16 subnormal range
    v.push((float)6.0e-5);           // fp16 subnormal
    v.push((float)1.00048828125);    // 1 + 2^-11: fp16 tie, to even (1.0)
    v.push((float)1.00146484375);    // 1 + 3 * 2^-11: fp16 tie, up
    v.push((float)1.00390625);       // 1 + 2^-8: bf16 tie, to even
    v.push((float)1.01171875);       // 1 + 3 * 2^-8: bf16 tie, up
    v.push((float)3.0e38);
    while (v.length() < n) {
        unsigned long i = v.length();
        double scale = 1.0;
        unsigned long c = i % 5UL;
        if (c == 0UL) scale = 1.0e-6;
        if (c == 1UL) scale = 1.0e-3;
        if (c == 3UL) scale = 100.0;
        if (c == 4UL) scale = 3.0e4;
        v.push((float)((double)((i * 7919UL) % 20001UL) - 10000.0) * (float)(scale / 10000.0));
    }
    struct RocmBuf dv = std::rocm_upload(v.at(0UL), n);
    int d = ROCM_FP16;
    while (d <= ROCM_BF16) {
        int bf = d == ROCM_BF16;
        // round on the GPU, compare bits with the CPU encoder
        struct RocmHalfBuf h = std::rocm_half_alloc(n, d);
        check(h.dev != 0UL && std::rocm_to_half(dv, h), bf ? "bf16 convert ran" : "fp16 convert ran");
        struct Vec<unsigned short> bits = std::vec_with_cap(n);
        unsigned long i = 0UL;
        while (i < n) { bits.push((unsigned short)0); i = i + 1UL; }
        check(std::rocm_half_read(h, bits.at(0UL), n), "half read");
        unsigned long bad = 0UL;
        i = 0UL;
        while (i < n) { if (*bits.at(i) != std::half_encode(*v.at(i), d)) bad = bad + 1UL; i = i + 1UL; }
        check(bad == 0UL, bf ? "bf16 rounding matches float16.h" : "fp16 rounding matches float16.h");
        // decode on the GPU: exact
        struct RocmBuf back = std::rocm_alloc(n);
        check(std::rocm_from_half(h, back), "half decode ran");
        struct Vec<float> fb = download(back);
        bad = 0UL;
        i = 0UL;
        while (i < n) { if (*fb.at(i) != std::half_decode(*bits.at(i), d)) bad = bad + 1UL; i = i + 1UL; }
        check(bad == 0UL, bf ? "bf16 decode exact" : "fp16 decode exact");
        // rocm_half_upload rounds the same way
        struct RocmHalfBuf hu = std::rocm_half_upload(v.at(0UL), n, d);
        struct Vec<unsigned short> ubits = std::vec_with_cap(n);
        i = 0UL;
        while (i < n) { ubits.push((unsigned short)0); i = i + 1UL; }
        check(hu.dev != 0UL && std::rocm_half_read(hu, ubits.at(0UL), n), "half upload");
        bad = 0UL;
        i = 0UL;
        while (i < n) { if (*ubits.at(i) != *bits.at(i)) bad = bad + 1UL; i = i + 1UL; }
        check(bad == 0UL, bf ? "bf16 upload bits" : "fp16 upload bits");

        // gemm: 16-bit inputs, fp32 accumulation
        unsigned long m = 64UL;
        unsigned long k = 96UL;
        unsigned long nn = 80UL;
        struct Vec<float> a = seq(m * k, 0.011f, -0.5f);
        struct Vec<float> b = seq(k * nn, 0.007f, -0.3f);
        struct Vec<float> ar = std::vec_with_cap(m * k);
        struct Vec<float> br = std::vec_with_cap(k * nn);
        i = 0UL;
        while (i < m * k) { ar.push(std::half_round(*a.at(i), d)); i = i + 1UL; }
        i = 0UL;
        while (i < k * nn) { br.push(std::half_round(*b.at(i), d)); i = i + 1UL; }
        struct RocmHalfBuf ha = std::rocm_half_upload(a.at(0UL), m * k, d);
        struct RocmHalfBuf hb = std::rocm_half_upload(b.at(0UL), k * nn, d);
        struct RocmBuf c32 = std::rocm_alloc(m * nn);
        struct RocmHalfBuf c16 = std::rocm_half_alloc(m * nn, d);
        struct Vec<unsigned short> cb = std::vec_with_cap(m * nn);
        i = 0UL;
        while (i < m * nn) { cb.push((unsigned short)0); i = i + 1UL; }
        int t = 0;
        while (t < 4) {
            int ta = t & 1;
            int tb = (t >> 1) & 1;
            struct Vec<float> ref = cpu_gemm(&ar, ta, &br, tb, m, nn, k);
            check(std::rocm_half_gemm_f32(ha, ta, hb, tb, c32, m, nn, k), "half gemm f32 ran");
            struct Vec<float> got = download(c32);
            check(maxdiff(&got, &ref) < 1e-4f, bf ? "bf16 gemm f32 values" : "fp16 gemm f32 values");
            got.free();
            // 16-bit output: within one 16-bit rounding of the fp32 result
            check(std::rocm_half_gemm(ha, ta, hb, tb, c16, m, nn, k) && std::rocm_half_read(c16, cb.at(0UL), m * nn),
                  "half gemm ran");
            bad = 0UL;
            i = 0UL;
            while (i < m * nn) {
                float r = *ref.at(i);
                float g = std::half_decode(*cb.at(i), d);
                float e = g - r;
                if (e < 0.0f) e = -e;
                float ar_ = r < 0.0f ? -r : r;
                if (e > ar_ * (bf ? (float)0.0079 : (float)0.00099) + (float)1.0e-6) bad = bad + 1UL;
                i = i + 1UL;
            }
            check(bad == 0UL, bf ? "bf16 gemm values" : "fp16 gemm values");
            ref.free();
            t = t + 1;
        }

        // sgd on 16-bit weights: bit-exact with the CPU computation
        struct Vec<unsigned short> wb = std::vec_with_cap(m * k);
        struct Vec<unsigned short> gbits = std::vec_with_cap(m * k);
        i = 0UL;
        while (i < m * k) { wb.push((unsigned short)0); gbits.push(std::half_encode(*v.at(i % n), d)); i = i + 1UL; }
        struct RocmHalfBuf hg = std::rocm_half_alloc(m * k, d);
        check(std::rocm_half_write(hg, gbits.at(0UL), m * k), "half write");
        check(std::rocm_half_read(ha, wb.at(0UL), m * k), "half read w");
        float lr = (float)0.25;
        check(std::rocm_half_sgd(ha, hg, lr), "half sgd ran");
        struct Vec<unsigned short> wn = std::vec_with_cap(m * k);
        i = 0UL;
        while (i < m * k) { wn.push((unsigned short)0); i = i + 1UL; }
        check(std::rocm_half_read(ha, wn.at(0UL), m * k), "half read sgd");
        bad = 0UL;
        i = 0UL;
        while (i < m * k) {
            float wv = std::half_decode(*wb.at(i), d);
            float gv = std::half_decode(*gbits.at(i), d);
            float prod = lr * gv;
            float r = wv - prod;
            if (*wn.at(i) != std::half_encode(r, d)) bad = bad + 1UL;
            i = i + 1UL;
        }
        check(bad == 0UL, bf ? "bf16 sgd bit-exact" : "fp16 sgd bit-exact");

        // training ops: relu / relu_grad copy bits, mse_grad and sgd2 round
        // once per value, all bit-exact with the CPU computation
        unsigned long obn = n > m * k ? n : m * k;   // also >= m * nn
        struct Vec<unsigned short> gb2 = std::vec_with_cap(n);
        struct Vec<unsigned short> ob = std::vec_with_cap(obn);
        i = 0UL;
        while (i < n) { gb2.push(std::half_encode(*v.at((i * 7UL + 3UL) % n), d)); i = i + 1UL; }
        i = 0UL;
        while (i < obn) { ob.push((unsigned short)0); i = i + 1UL; }
        struct RocmHalfBuf hg2 = std::rocm_half_alloc(n, d);
        struct RocmHalfBuf ho = std::rocm_half_alloc(n, d);
        check(std::rocm_half_write(hg2, gb2.at(0UL), n), "half write g");
        check(std::rocm_half_relu(h, ho) && std::rocm_half_read(ho, ob.at(0UL), n), "half relu ran");
        bad = 0UL;
        i = 0UL;
        while (i < n) {
            unsigned short want = std::half_decode(*bits.at(i), d) > 0.0f ? *bits.at(i) : (unsigned short)0;
            if (*ob.at(i) != want) bad = bad + 1UL;
            i = i + 1UL;
        }
        check(bad == 0UL, bf ? "bf16 relu bit-exact" : "fp16 relu bit-exact");
        check(std::rocm_half_relu_grad(h, hg2, ho) && std::rocm_half_read(ho, ob.at(0UL), n), "half relu_grad ran");
        bad = 0UL;
        i = 0UL;
        while (i < n) {
            unsigned short want = std::half_decode(*bits.at(i), d) > 0.0f ? *gb2.at(i) : (unsigned short)0;
            if (*ob.at(i) != want) bad = bad + 1UL;
            i = i + 1UL;
        }
        check(bad == 0UL, bf ? "bf16 relu_grad bit-exact" : "fp16 relu_grad bit-exact");

        unsigned long nm = m * k;
        struct RocmHalfBuf hy = std::rocm_half_upload(a.at(0UL), nm, d);
        struct RocmHalfBuf ht = std::rocm_half_upload(b.at(0UL), nm, d);
        struct RocmHalfBuf hdy = std::rocm_half_alloc(nm, d);
        struct RocmBuf loss = std::rocm_alloc(1UL);
        float lossHost[1];
        check(std::rocm_half_mse_grad(hy, ht, hdy, loss) && std::rocm_half_read(hdy, ob.at(0UL), nm) &&
              std::rocm_read(loss, &lossHost[0], 1UL), "half mse_grad ran");
        bad = 0UL;
        double lref = 0.0;
        i = 0UL;
        while (i < nm) {
            float dd = std::half_round(*a.at(i), d) - std::half_round(*b.at(i), d);
            if (*ob.at(i) != std::half_encode((float)2.0 * dd, d)) bad = bad + 1UL;
            lref = lref + (double)dd * (double)dd;
            i = i + 1UL;
        }
        check(bad == 0UL, bf ? "bf16 mse_grad dy bit-exact" : "fp16 mse_grad dy bit-exact");
        double lerr = (double)lossHost[0] - lref;
        if (lerr < 0.0) lerr = -lerr;
        check(lerr < lref * 1.0e-5, bf ? "bf16 mse_grad loss" : "fp16 mse_grad loss");

        // sgd2 == two sgd calls
        struct RocmHalfBuf w1a = std::rocm_half_upload(a.at(0UL), nm, d);
        struct RocmHalfBuf w2a = std::rocm_half_upload(b.at(0UL), k * nn, d);
        struct RocmHalfBuf w1b = std::rocm_half_upload(a.at(0UL), nm, d);
        struct RocmHalfBuf w2b = std::rocm_half_upload(b.at(0UL), k * nn, d);
        struct RocmHalfBuf g1 = std::rocm_half_upload(b.at(0UL), nm, d);
        struct RocmHalfBuf g2 = std::rocm_half_upload(b.at(0UL), k * nn, d);
        check(std::rocm_half_sgd2(w1a, g1, w2a, g2, (float)0.375), "half sgd2 ran");
        check(std::rocm_half_sgd(w1b, g1, (float)0.375) && std::rocm_half_sgd(w2b, g2, (float)0.375), "half sgd x2 ran");
        struct Vec<unsigned short> xa = std::vec_with_cap(k * nn);
        struct Vec<unsigned short> xb = std::vec_with_cap(k * nn);
        i = 0UL;
        while (i < k * nn) { xa.push((unsigned short)0); xb.push((unsigned short)0); i = i + 1UL; }
        bad = 0UL;
        check(std::rocm_half_read(w1a, xa.at(0UL), nm) && std::rocm_half_read(w1b, xb.at(0UL), nm), "half read w1");
        i = 0UL;
        while (i < nm) { if (*xa.at(i) != *xb.at(i)) bad = bad + 1UL; i = i + 1UL; }
        check(std::rocm_half_read(w2a, xa.at(0UL), k * nn) && std::rocm_half_read(w2b, xb.at(0UL), k * nn), "half read w2");
        i = 0UL;
        while (i < k * nn) { if (*xa.at(i) != *xb.at(i)) bad = bad + 1UL; i = i + 1UL; }
        check(bad == 0UL, bf ? "bf16 sgd2 == two sgd" : "fp16 sgd2 == two sgd");

        // gemm_relu == gemm, then relu on the bits
        struct RocmHalfBuf hb2 = std::rocm_half_upload(b.at(0UL), k * nn, d);
        struct RocmHalfBuf ha2 = std::rocm_half_upload(a.at(0UL), m * k, d);
        struct RocmHalfBuf cr = std::rocm_half_alloc(m * nn, d);
        check(std::rocm_half_gemm(ha2, 0, hb2, 0, c16, m, nn, k) && std::rocm_half_read(c16, cb.at(0UL), m * nn),
              "half gemm (for relu) ran");
        check(std::rocm_half_gemm_relu(ha2, 0, hb2, 0, cr, m, nn, k) && std::rocm_half_read(cr, ob.at(0UL), m * nn),
              "half gemm_relu ran");
        bad = 0UL;
        i = 0UL;
        while (i < m * nn) {
            unsigned short want = std::half_decode(*cb.at(i), d) > 0.0f ? *cb.at(i) : (unsigned short)0;
            if (*ob.at(i) != want) bad = bad + 1UL;
            i = i + 1UL;
        }
        check(bad == 0UL, bf ? "bf16 gemm_relu" : "fp16 gemm_relu");
        check(!std::rocm_half_relu_grad(hy, h, ho), "half relu_grad size check");   // g shorter than z

        std::rocm_half_free(&hg2); std::rocm_half_free(&ho); std::rocm_half_free(&hy); std::rocm_half_free(&ht);
        std::rocm_half_free(&hdy); std::rocm_free(&loss); std::rocm_half_free(&w1a); std::rocm_half_free(&w2a);
        std::rocm_half_free(&w1b); std::rocm_half_free(&w2b); std::rocm_half_free(&g1); std::rocm_half_free(&g2);
        std::rocm_half_free(&hb2); std::rocm_half_free(&ha2); std::rocm_half_free(&cr);
        gb2.free(); ob.free(); xa.free(); xb.free();

        // dtype mismatches are rejected
        struct RocmHalfBuf other = std::rocm_half_alloc(k * nn, bf ? ROCM_FP16 : ROCM_BF16);
        check(!std::rocm_half_gemm_f32(ha, 0, other, 0, c32, m, nn, k), "half gemm dtype check");
        std::rocm_half_free(&other);

        std::rocm_half_free(&h); std::rocm_half_free(&hu); std::rocm_half_free(&ha); std::rocm_half_free(&hb);
        std::rocm_half_free(&c16); std::rocm_half_free(&hg); std::rocm_free(&c32); std::rocm_free(&back);
        bits.free(); ubits.free(); fb.free(); a.free(); b.free(); ar.free(); br.free(); cb.free();
        wb.free(); gbits.free(); wn.free();
        d = d + 1;
    }
    std::rocm_free(&dv);
    v.free();
}

// ── Attention ────────────────────────────────────────────────────────────────
// CPU references over [heads][rows][d]: softmax attention (causal: key j is
// visible to query i when j <= i + kv - seq) and linear attention with
// phi(x) = elu(x) + 1.
static float phi_ref(float x) { return x > (float)0.0 ? x + (float)1.0 : (float)std::exp_d((double)x); }

static struct Vec<float> attn_ref(&Vec<float> q, &Vec<float> k, &Vec<float> v, unsigned long heads,
                                  unsigned long seq, unsigned long kv, unsigned long d, float scale,
                                  int causal, int linear) {
    struct Vec<float> o = std::vec_with_cap(heads * seq * d);
    unsigned long h = 0UL;
    while (h < heads) {
        unsigned long i = 0UL;
        while (i < seq) {
            struct Vec<double> w = std::vec_with_cap(kv);
            double mx = -1.0e300;
            unsigned long j = 0UL;
            while (j < kv) {
                double sv = 0.0;
                unsigned long c = 0UL;
                if (linear) {
                    while (c < d) { sv = sv + (double)phi_ref(*q.at((h * seq + i) * d + c)) * (double)phi_ref(*k.at((h * kv + j) * d + c)); c = c + 1UL; }
                } else {
                    while (c < d) { sv = sv + (double)*q.at((h * seq + i) * d + c) * (double)*k.at((h * kv + j) * d + c); c = c + 1UL; }
                    sv = sv * (double)scale;
                }
                int vis = !causal || (long)j <= (long)i + (long)kv - (long)seq;
                w.push(vis ? sv : -1.0e300);
                if (vis && sv > mx) mx = sv;
                j = j + 1UL;
            }
            double den = 0.0;
            j = 0UL;
            while (j < kv) {
                double x = *w.at(j);
                double p = x <= -1.0e299 ? 0.0 : (linear ? x : std::exp_d(x - mx));
                *w.at(j) = p;
                den = den + p;
                j = j + 1UL;
            }
            unsigned long c = 0UL;
            while (c < d) {
                double a = 0.0;
                j = 0UL;
                while (j < kv) { a = a + *w.at(j) * (double)*v.at((h * kv + j) * d + c); j = j + 1UL; }
                o.push(den > 0.0 ? (float)(a / den) : (float)0.0);
                c = c + 1UL;
            }
            w.free();
            i = i + 1UL;
        }
        h = h + 1UL;
    }
    return o;
}

static float relmax(&Vec<float> a, &Vec<float> b) {
    float m = 0.0f;
    float big = 0.0f;
    unsigned long i = 0UL;
    while (i < a.length()) {
        float x = *b.at(i);
        if (x < 0.0f) x = -x;
        if (x > big) big = x;
        i = i + 1UL;
    }
    m = maxdiff(a, b);
    return big > 0.0f ? m / big : m;
}

static void check_attention() {
    unsigned long heads = 3UL;
    unsigned long dims[3];
    dims[0] = 64UL; dims[1] = 40UL; dims[2] = 128UL;      // the 64 and 128 builds, and a masked one
    int di = 0;
    while (di < 3) {
        unsigned long d = dims[di];
        unsigned long cases[3][2];
        cases[0][0] = 70UL; cases[0][1] = 70UL;           // partial tiles and chunks
        cases[1][0] = 33UL; cases[1][1] = 100UL;          // fewer queries than keys (KV cache)
        cases[2][0] = 1UL;  cases[2][1] = 257UL;          // one query
        int ci = 0;
        while (ci < 3) {
            unsigned long sl = cases[ci][0];
            unsigned long kv = cases[ci][1];
            struct Vec<float> q = seq(heads * sl * d, 0.013f, -0.6f);
            struct Vec<float> k = seq(heads * kv * d, 0.011f, -0.55f);
            struct Vec<float> v = seq(heads * kv * d, 0.02f, -1.0f);
            struct Vec<float> got = std::vec_with_cap(heads * sl * d);
            unsigned long z = 0UL;
            while (z < heads * sl * d) { got.push(0.0f); z = z + 1UL; }
            int causal = 0;
            while (causal < 2) {
                float scale = (float)(1.0 / std::sqrt_d((double)d));
                struct Vec<float> ref = attn_ref(&q, &k, &v, heads, sl, kv, d, scale, causal, 0);
                int ok = std::rocm_flash_attention_f32(q.at(0UL), k.at(0UL), v.at(0UL), got.at(0UL),
                                                       heads, sl, kv, d, 0.0f, causal);
                check(ok && relmax(&got, &ref) < 1.0e-4f, "flash attention vs reference");
                if (!ok || relmax(&got, &ref) >= 1.0e-4f) {
                    unsafe { printf("    flash d=%lu sl=%lu kv=%lu causal=%d ok=%d err=%g\n", d, sl, kv, causal, ok, (double)relmax(&got, &ref)); }
                }
                ref.free();
                if (!causal || kv == sl) {
                    ref = attn_ref(&q, &k, &v, heads, sl, kv, d, 1.0f, causal, 1);
                    ok = std::rocm_linear_attention_f32(q.at(0UL), k.at(0UL), v.at(0UL), got.at(0UL),
                                                        heads, sl, kv, d, causal);
                    check(ok && relmax(&got, &ref) < 1.0e-4f, "linear attention vs reference");
                    if (!ok || relmax(&got, &ref) >= 1.0e-4f) {
                        unsafe { printf("    linear d=%lu sl=%lu kv=%lu causal=%d ok=%d err=%g\n", d, sl, kv, causal, ok, (double)relmax(&got, &ref)); }
                    }
                    ref.free();
                } else {
                    // causal linear attention needs kvLen == seqLen
                    check(!std::rocm_linear_attention_f32(q.at(0UL), k.at(0UL), v.at(0UL), got.at(0UL),
                                                          heads, sl, kv, d, 1), "causal linear rejects kv != sl");
                }
                causal = causal + 1;
            }
            // an explicit scale
            struct Vec<float> ref2 = attn_ref(&q, &k, &v, heads, sl, kv, d, 0.37f, 0, 0);
            check(std::rocm_flash_attention_f32(q.at(0UL), k.at(0UL), v.at(0UL), got.at(0UL), heads, sl, kv, d, 0.37f, 0) &&
                  relmax(&got, &ref2) < 1.0e-4f, "flash attention, explicit scale");
            ref2.free();
            q.free(); k.free(); v.free(); got.free();
            ci = ci + 1;
        }
        di = di + 1;
    }
    // same definitions as std/ml/attention.h's CPU functions (one head)
    unsigned long n1 = 50UL;
    unsigned long d1 = 32UL;
    struct Vec<float> q1 = seq(n1 * d1, 0.017f, -0.8f);
    struct Vec<float> k1 = seq(n1 * d1, 0.015f, -0.7f);
    struct Vec<float> v1 = seq(n1 * d1, 0.03f, -1.2f);
    &Tensor tq = std::tensor_from_2d(q1.at(0UL), n1, d1, 0);
    &Tensor tk = std::tensor_from_2d(k1.at(0UL), n1, d1, 0);
    &Tensor tv = std::tensor_from_2d(v1.at(0UL), n1, d1, 0);
    &Tensor sa = std::attention_forward(tq, tk, tv);
    &Tensor la = std::linear_attention_forward(tq, tk, tv);
    struct Vec<float> g1 = std::vec_with_cap(n1 * d1);
    struct Vec<float> c1 = std::vec_with_cap(n1 * d1);
    struct Vec<float> c2 = std::vec_with_cap(n1 * d1);
    unsigned long z = 0UL;
    while (z < n1 * d1) { g1.push(0.0f); unsafe { c1.push(sa->data[z]); c2.push(la->data[z]); } z = z + 1UL; }
    check(std::rocm_flash_attention_f32(q1.at(0UL), k1.at(0UL), v1.at(0UL), g1.at(0UL), 1UL, n1, n1, d1, 0.0f, 0) &&
          relmax(&g1, &c1) < 1.0e-4f, "flash attention == attention_forward");
    check(std::rocm_linear_attention_f32(q1.at(0UL), k1.at(0UL), v1.at(0UL), g1.at(0UL), 1UL, n1, n1, d1, 0) &&
          relmax(&g1, &c2) < 1.0e-4f, "linear attention == linear_attention_forward");
    // bad shapes
    struct RocmBuf big = std::rocm_alloc(n1 * 129UL);
    check(big.dev != 0UL && !std::rocm_flash_attention(big, big, big, big, 1UL, n1, n1, 129UL, 0.0f, 0) &&
          !std::rocm_linear_attention(big, big, big, big, 1UL, n1, n1, 129UL, 0), "headDim > 128 rejected");
    check(!std::rocm_flash_attention(big, big, big, big, 1UL, n1, n1 - 1UL, 64UL, 0.0f, 1), "causal flash rejects kv < sl");
    std::rocm_free(&big);
    sa.free(); la.free(); tq.free(); tk.free(); tv.free();
    q1.free(); k1.free(); v1.free(); g1.free(); c1.free(); c2.free();
}


int main() {
    if (!std::rocm_available()) { unsafe { printf("no ROCm GPU -- skipped\n"); } return 0; }
    unsafe { printf("device: %s\n", std::rocm_device_name()); }

    unsigned long n = 10007UL;   // not a multiple of the block size
    struct Vec<float> a = seq(n, 0.01f, 0.1f);
    struct Vec<float> b = seq(n, 0.02f, 0.5f);
    struct RocmBuf da = std::rocm_upload(a.at(0UL), n);
    struct RocmBuf db = std::rocm_upload(b.at(0UL), n);
    struct RocmBuf dout = std::rocm_alloc(n);
    check(da.dev != 0UL && db.dev != 0UL && dout.dev != 0UL, "upload/alloc");

    // elementwise vs CPU
    struct Vec<float> want = std::vec_with_cap(n);
    unsigned long i = 0UL;
    while (i < n) { want.push(*a.at(i) + *b.at(i)); i = i + 1UL; }
    check(std::rocm_add(da, db, dout), "add enqueued");
    struct Vec<float> got = download(dout);
    check(maxdiff(&got, &want) < 1e-5f, "add values");
    got.free();

    std::rocm_mul(da, db, dout);
    got = download(dout);
    i = 0UL;
    while (i < n) { *want.at(i) = *a.at(i) * *b.at(i); i = i + 1UL; }
    check(maxdiff(&got, &want) < 1e-4f, "mul values");
    got.free();

    std::rocm_scale(da, -3.0f, dout);
    std::rocm_relu(dout, dout);   // in place: all zero
    got = download(dout);
    i = 0UL;
    while (i < n) { *want.at(i) = 0.0f; i = i + 1UL; }
    check(maxdiff(&got, &want) == 0.0f, "scale + relu in place");
    got.free();

    std::rocm_exp(da, dout);
    got = download(dout);
    i = 0UL;
    while (i < n) { *want.at(i) = std::exp_f(*a.at(i)); i = i + 1UL; }
    check(maxdiff(&got, &want) < 1e-3f, "exp values");
    got.free();

    // sum
    struct RocmBuf d1 = std::rocm_alloc(1UL);
    std::rocm_sum(da, d1);
    got = download(d1);
    double cpuSum = 0.0;
    i = 0UL;
    while (i < n) { cpuSum = cpuSum + (double)*a.at(i); i = i + 1UL; }
    double rel = ((double)*got.at(0UL) - cpuSum) / cpuSum;
    if (rel < 0.0) rel = -rel;
    check(rel < 1e-5, "sum");
    got.free();

    // fused MSE gradient
    std::rocm_mse_grad(da, db, dout, d1);
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
    std::rocm_sgd(da, db, 0.5f);
    got = download(da);
    i = 0UL;
    while (i < n) { *want.at(i) = *a.at(i) - (float)0.5 * *b.at(i); i = i + 1UL; }
    check(maxdiff(&got, &want) < 1e-5f, "sgd");
    got.free();

    // size checks
    struct RocmBuf small = std::rocm_alloc(10UL);
    check(!std::rocm_add(small, db, dout), "length mismatch rejected");

    // gemm, every transpose combination, odd sizes
    unsigned long m = 67UL;
    unsigned long nn = 45UL;
    unsigned long k = 129UL;
    struct Vec<float> ga = seq(m * k, 0.003f, -0.1f);
    struct Vec<float> gb = seq(k * nn, 0.002f, -0.07f);
    struct RocmBuf dga = std::rocm_upload(ga.at(0UL), m * k);
    struct RocmBuf dgb = std::rocm_upload(gb.at(0UL), k * nn);
    struct RocmBuf dgc = std::rocm_alloc(m * nn);
    int t = 0;
    while (t < 4) {
        int ta = t & 1;
        int tb = (t >> 1) & 1;
        struct Vec<float> ref = cpu_gemm(&ga, ta, &gb, tb, m, nn, k);
        check(std::rocm_gemm(dga, ta, dgb, tb, dgc, m, nn, k), "gemm enqueued");
        struct Vec<float> gc = download(dgc);
        check(maxdiff(&gc, &ref) < 1e-4f, ta ? (tb ? "gemm tt" : "gemm tn") : (tb ? "gemm nt" : "gemm nn"));
        gc.free();
        ref.free();
        t = t + 1;
    }
    check(!std::rocm_gemm(dga, 0, dgb, 0, dgc, m, nn, k + 1UL), "gemm size check");

    // host-array API: custom kernel and rocBLAS agree with CPU
    struct Vec<float> ref = cpu_gemm(&ga, 0, &gb, 0, m, nn, k);
    struct Vec<float> hc = std::vec_with_cap(m * nn);
    i = 0UL;
    while (i < m * nn) { hc.push(0.0f); i = i + 1UL; }
    check(std::rocm_matmul_f32(ga.at(0UL), gb.at(0UL), hc.at(0UL), m, k, nn), "custom matmul ran");
    check(maxdiff(&hc, &ref) < 1e-4f, "custom matmul values");
    check(std::rocm_matmul_f32_blas(ga.at(0UL), gb.at(0UL), hc.at(0UL), m, k, nn), "blas matmul ran");
    check(maxdiff(&hc, &ref) < 1e-4f, "blas matmul values");

    // graph: record add + scale, replay twice
    check(std::rocm_capture_begin(), "capture begin");
    std::rocm_add(db, db, dout);
    std::rocm_scale(dout, 0.25f, dout);
    struct RocmGraph g = std::rocm_capture_end();
    check(g.exec != 0UL, "capture end");
    check(std::rocm_graph_launch(g) && std::rocm_graph_launch(g), "graph launch");
    got = download(dout);
    i = 0UL;
    while (i < n) { *want.at(i) = *b.at(i) * (float)0.5; i = i + 1UL; }
    check(maxdiff(&got, &want) < 1e-5f, "graph replay values");
    got.free();
    std::rocm_graph_free(&g);

    std::rocm_free(&da); std::rocm_free(&db); std::rocm_free(&dout); std::rocm_free(&d1);
    std::rocm_free(&small); std::rocm_free(&dga); std::rocm_free(&dgb); std::rocm_free(&dgc);
    check(da.dev == 0UL, "free clears handle");
    a.free(); b.free(); want.free(); ga.free(); gb.free(); ref.free(); hc.free();

    check_half();

    check_attention();
    unsafe { printf("rocm: %d checks, %d failures\n", g_checks, g_fails); }
    return g_fails == 0 ? 0 : 1;
}
