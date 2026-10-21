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
    struct Vec<float> v = std::Vec::with_cap(n);
    unsigned long i = 0UL;
    while (i < n) { v.push((float)((i * 37UL) % 101UL) * scale + offset); i = i + 1UL; }
    return v;
}

// CPU reference: c[m,n] = op(a) op(b), row-major, ta/tb = stored transposed.
static struct Vec<float> cpu_gemm(&Vec<float> a, int ta, &Vec<float> b, int tb,
                                  unsigned long m, unsigned long n, unsigned long k) {
    struct Vec<float> c = std::Vec::with_cap(m * n);
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
    struct Vec<float> v = std::Vec::with_cap(b.len);
    unsigned long i = 0UL;
    while (i < b.len) { v.push(0.0f); i = i + 1UL; }
    std::rocm_read(b, v.at(0UL), b.len);
    return v;
}

// fp16 / bf16: GPU rounding matches float16.h bit for bit; 16-bit gemm
// against an fp32 CPU gemm of the same rounded inputs; 16-bit SGD exact.
static void check_half() {
    unsigned long n = 4096UL;
    struct Vec<float> v = std::Vec::with_cap(n);
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
    struct RocmBuf dv = RocmBuf::upload(v.at(0UL), n);
    int d = ROCM_FP16;
    while (d <= ROCM_BF16) {
        int bf = d == ROCM_BF16;
        // round on the GPU, compare bits with the CPU encoder
        struct RocmHalfBuf h = RocmHalfBuf::alloc(n, d);
        check(h.dev != 0UL && std::rocm_to_half(dv, h), bf ? "bf16 convert ran" : "fp16 convert ran");
        struct Vec<unsigned short> bits = std::Vec::with_cap(n);
        unsigned long i = 0UL;
        while (i < n) { bits.push((unsigned short)0); i = i + 1UL; }
        check(std::rocm_half_read(h, bits.at(0UL), n), "half read");
        unsigned long bad = 0UL;
        i = 0UL;
        while (i < n) { if (*bits.at(i) != std::half_encode(*v.at(i), d)) bad = bad + 1UL; i = i + 1UL; }
        check(bad == 0UL, bf ? "bf16 rounding matches float16.h" : "fp16 rounding matches float16.h");
        // decode on the GPU: exact
        struct RocmBuf back = RocmBuf::alloc(n);
        check(std::rocm_from_half(h, back), "half decode ran");
        struct Vec<float> fb = download(back);
        bad = 0UL;
        i = 0UL;
        while (i < n) { if (*fb.at(i) != std::half_decode(*bits.at(i), d)) bad = bad + 1UL; i = i + 1UL; }
        check(bad == 0UL, bf ? "bf16 decode exact" : "fp16 decode exact");
        // RocmHalfBuf::upload rounds the same way
        struct RocmHalfBuf hu = RocmHalfBuf::upload(v.at(0UL), n, d);
        struct Vec<unsigned short> ubits = std::Vec::with_cap(n);
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
        struct Vec<float> ar = std::Vec::with_cap(m * k);
        struct Vec<float> br = std::Vec::with_cap(k * nn);
        i = 0UL;
        while (i < m * k) { ar.push(std::half_round(*a.at(i), d)); i = i + 1UL; }
        i = 0UL;
        while (i < k * nn) { br.push(std::half_round(*b.at(i), d)); i = i + 1UL; }
        struct RocmHalfBuf ha = RocmHalfBuf::upload(a.at(0UL), m * k, d);
        struct RocmHalfBuf hb = RocmHalfBuf::upload(b.at(0UL), k * nn, d);
        struct RocmBuf c32 = RocmBuf::alloc(m * nn);
        struct RocmHalfBuf c16 = RocmHalfBuf::alloc(m * nn, d);
        struct Vec<unsigned short> cb = std::Vec::with_cap(m * nn);
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
        struct Vec<unsigned short> wb = std::Vec::with_cap(m * k);
        struct Vec<unsigned short> gbits = std::Vec::with_cap(m * k);
        i = 0UL;
        while (i < m * k) { wb.push((unsigned short)0); gbits.push(std::half_encode(*v.at(i % n), d)); i = i + 1UL; }
        struct RocmHalfBuf hg = RocmHalfBuf::alloc(m * k, d);
        check(std::rocm_half_write(hg, gbits.at(0UL), m * k), "half write");
        check(std::rocm_half_read(ha, wb.at(0UL), m * k), "half read w");
        float lr = (float)0.25;
        check(std::rocm_half_sgd(ha, hg, lr), "half sgd ran");
        struct Vec<unsigned short> wn = std::Vec::with_cap(m * k);
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
        struct Vec<unsigned short> gb2 = std::Vec::with_cap(n);
        struct Vec<unsigned short> ob = std::Vec::with_cap(obn);
        i = 0UL;
        while (i < n) { gb2.push(std::half_encode(*v.at((i * 7UL + 3UL) % n), d)); i = i + 1UL; }
        i = 0UL;
        while (i < obn) { ob.push((unsigned short)0); i = i + 1UL; }
        struct RocmHalfBuf hg2 = RocmHalfBuf::alloc(n, d);
        struct RocmHalfBuf ho = RocmHalfBuf::alloc(n, d);
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
        struct RocmHalfBuf hy = RocmHalfBuf::upload(a.at(0UL), nm, d);
        struct RocmHalfBuf ht = RocmHalfBuf::upload(b.at(0UL), nm, d);
        struct RocmHalfBuf hdy = RocmHalfBuf::alloc(nm, d);
        struct RocmBuf loss = RocmBuf::alloc(1UL);
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
        struct RocmHalfBuf w1a = RocmHalfBuf::upload(a.at(0UL), nm, d);
        struct RocmHalfBuf w2a = RocmHalfBuf::upload(b.at(0UL), k * nn, d);
        struct RocmHalfBuf w1b = RocmHalfBuf::upload(a.at(0UL), nm, d);
        struct RocmHalfBuf w2b = RocmHalfBuf::upload(b.at(0UL), k * nn, d);
        struct RocmHalfBuf g1 = RocmHalfBuf::upload(b.at(0UL), nm, d);
        struct RocmHalfBuf g2 = RocmHalfBuf::upload(b.at(0UL), k * nn, d);
        check(std::rocm_half_sgd2(w1a, g1, w2a, g2, (float)0.375), "half sgd2 ran");
        check(std::rocm_half_sgd(w1b, g1, (float)0.375) && std::rocm_half_sgd(w2b, g2, (float)0.375), "half sgd x2 ran");
        struct Vec<unsigned short> xa = std::Vec::with_cap(k * nn);
        struct Vec<unsigned short> xb = std::Vec::with_cap(k * nn);
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
        struct RocmHalfBuf hb2 = RocmHalfBuf::upload(b.at(0UL), k * nn, d);
        struct RocmHalfBuf ha2 = RocmHalfBuf::upload(a.at(0UL), m * k, d);
        struct RocmHalfBuf cr = RocmHalfBuf::alloc(m * nn, d);
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

        hg2.free(); ho.free(); hy.free(); ht.free();
        hdy.free(); loss.free(); w1a.free(); w2a.free();
        w1b.free(); w2b.free(); g1.free(); g2.free();
        hb2.free(); ha2.free(); cr.free();
        gb2.free(); ob.free(); xa.free(); xb.free();

        // dtype mismatches are rejected
        struct RocmHalfBuf other = RocmHalfBuf::alloc(k * nn, bf ? ROCM_FP16 : ROCM_BF16);
        check(!std::rocm_half_gemm_f32(ha, 0, other, 0, c32, m, nn, k), "half gemm dtype check");
        other.free();

        h.free(); hu.free(); ha.free(); hb.free();
        c16.free(); hg.free(); c32.free(); back.free();
        bits.free(); ubits.free(); fb.free(); a.free(); b.free(); ar.free(); br.free(); cb.free();
        wb.free(); gbits.free(); wn.free();
        d = d + 1;
    }
    dv.free();
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
    struct Vec<float> o = std::Vec::with_cap(heads * seq * d);
    unsigned long h = 0UL;
    while (h < heads) {
        unsigned long i = 0UL;
        while (i < seq) {
            struct Vec<double> w = std::Vec::with_cap(kv);
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

// Decode-step ops: rocm_half_gemv (both dtypes, vector and scalar paths),
// rocm_rmsnorm (with and without weight, in place) and rocm_gelu.
static void check_decode_ops() {
    int dts[2];
    dts[0] = ROCM_BF16; dts[1] = ROCM_FP16;
    unsigned long ks[2];
    ks[0] = 1536UL; ks[1] = 100UL;               // k % 8 == 0 (16-byte loads) and not
    int di = 0;
    while (di < 2) {
        int ki = 0;
        while (ki < 2) {
            unsigned long n = 37UL;
            unsigned long k = ks[ki];
            struct Vec<float> w = seq(n * k, 0.0013f, -0.9f);
            struct Vec<float> x = seq(k, 0.017f, -0.7f);
            struct RocmHalfBuf dw = RocmHalfBuf::upload(w.at(0UL), n * k, dts[di]);
            struct Vec<unsigned short> bits = std::Vec::with_cap(n * k);
            unsigned long z = 0UL;
            while (z < n * k) { bits.push((unsigned short)0); z = z + 1UL; }
            std::rocm_half_read(dw, bits.at(0UL), n * k);   // the rounded weights the GPU uses
            struct RocmBuf dx = RocmBuf::upload(x.at(0UL), k);
            struct RocmBuf dy = RocmBuf::alloc(n);
            int ok = std::rocm_half_gemv(dw, dx, dy, n, k);
            struct Vec<float> got = download(dy);
            float err = (float)0.0;
            float big = (float)0.0;
            unsigned long r = 0UL;
            while (r < n) {
                double ref = 0.0;
                unsigned long c = 0UL;
                while (c < k) {
                    float wv = dts[di] == ROCM_BF16 ? std::bf16_to_f32(*bits.at(r * k + c)) : std::fp16_to_f32(*bits.at(r * k + c));
                    ref = ref + (double)wv * (double)*x.at(c);
                    c = c + 1UL;
                }
                float d = *got.at(r) - (float)ref;
                if (d < (float)0.0) d = -d;
                if (d > err) err = d;
                float a = (float)ref;
                if (a < (float)0.0) a = -a;
                if (a > big) big = a;
                r = r + 1UL;
            }
            check(ok && err <= (float)1e-5 * (big > (float)1.0 ? big : (float)1.0) * (float)k, "half gemv vs reference");
            got.free(); w.free(); x.free(); bits.free();
            dw.free(); dx.free(); dy.free();
            ki = ki + 1;
        }
        di = di + 1;
    }
    // rmsnorm: 3 rows of 1536, with weight into y, then without weight in place
    unsigned long rows = 3UL;
    unsigned long n = 1536UL;
    struct Vec<float> x = seq(rows * n, 0.011f, -0.8f);
    struct Vec<float> w = seq(n, 0.002f, 0.5f);
    struct RocmBuf dx = RocmBuf::upload(x.at(0UL), rows * n);
    struct RocmBuf dw = RocmBuf::upload(w.at(0UL), n);
    struct RocmBuf dy = RocmBuf::alloc(rows * n);
    int ok1 = std::rocm_rmsnorm(dx, dw, dy, rows, n, (float)1e-6);
    struct Vec<float> y1 = download(dy);
    struct RocmBuf none;
    none.dev = 0UL; none.len = 0UL;
    int ok2 = std::rocm_rmsnorm(dx, none, dx, rows, n, (float)1e-6);
    struct Vec<float> y2 = download(dx);
    float e1 = (float)0.0;
    float e2 = (float)0.0;
    unsigned long r = 0UL;
    while (r < rows) {
        double ss = 0.0;
        unsigned long c = 0UL;
        while (c < n) { double v = (double)*x.at(r * n + c); ss = ss + v * v; c = c + 1UL; }
        double inv = 1.0 / std::sqrt_d(ss / (double)n + 1e-6);
        c = 0UL;
        while (c < n) {
            float r1 = (float)((double)*x.at(r * n + c) * inv * (double)*w.at(c));
            float r2 = (float)((double)*x.at(r * n + c) * inv);
            float d1 = *y1.at(r * n + c) - r1; if (d1 < (float)0.0) d1 = -d1; if (d1 > e1) e1 = d1;
            float d2 = *y2.at(r * n + c) - r2; if (d2 < (float)0.0) d2 = -d2; if (d2 > e2) e2 = d2;
            c = c + 1UL;
        }
        r = r + 1UL;
    }
    check(ok1 && e1 < (float)1e-5, "rmsnorm with weight");
    check(ok2 && e2 < (float)1e-5, "rmsnorm without weight, in place");
    // gelu (tanh approximation)
    struct Vec<float> gx = seq(1000UL, 0.01f, -5.0f);
    struct RocmBuf dg = RocmBuf::upload(gx.at(0UL), 1000UL);
    struct RocmBuf dgo = RocmBuf::alloc(1000UL);
    int ok3 = std::rocm_gelu(dg, dgo);
    struct Vec<float> go = download(dgo);
    float e3 = (float)0.0;
    unsigned long i = 0UL;
    while (i < 1000UL) {
        double v = (double)*gx.at(i);
        double ref = 0.5 * v * (1.0 + std::tanh_d(0.7978845608028654 * (v + 0.044715 * v * v * v)));
        float d = *go.at(i) - (float)ref; if (d < (float)0.0) d = -d; if (d > e3) e3 = d;
        i = i + 1UL;
    }
    check(ok3 && e3 < (float)1e-5, "gelu (tanh) vs reference");
    // bad shapes are rejected
    struct RocmHalfBuf small = RocmHalfBuf::alloc(10UL, ROCM_BF16);
    check(!std::rocm_half_gemv(small, dg, dgo, 5UL, 4UL), "half gemv rejects a too-small weight buffer");
    check(!std::rocm_rmsnorm(dx, dw, dy, 4UL, n, (float)1e-6), "rmsnorm rejects too many rows");
    small.free();
    x.free(); w.free(); y1.free(); y2.free(); gx.free(); go.free();
    dx.free(); dw.free(); dy.free(); dg.free(); dgo.free();
}

static void check_attention() {
    unsigned long heads = 3UL;
    unsigned long dims[3];
    dims[0] = 64UL; dims[1] = 40UL; dims[2] = 128UL;      // the 64 and 128 builds, and a masked one
    int di = 0;
    while (di < 3) {
        unsigned long d = dims[di];
        unsigned long cases[4][2];
        cases[0][0] = 70UL; cases[0][1] = 70UL;           // partial tiles and chunks
        cases[1][0] = 33UL; cases[1][1] = 100UL;          // fewer queries than keys (KV cache)
        cases[2][0] = 1UL;  cases[2][1] = 257UL;          // one query
        cases[3][0] = 300UL; cases[3][1] = 300UL;         // several 128-row chunks and query blocks
        int ci = 0;
        while (ci < 4) {
            unsigned long sl = cases[ci][0];
            unsigned long kv = cases[ci][1];
            struct Vec<float> q = seq(heads * sl * d, 0.013f, -0.6f);
            struct Vec<float> k = seq(heads * kv * d, 0.011f, -0.55f);
            struct Vec<float> v = seq(heads * kv * d, 0.02f, -1.0f);
            struct Vec<float> got = std::Vec::with_cap(heads * sl * d);
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
    // causal linear attention over many chunks (the look-back chain)
    unsigned long ld[2];
    ld[0] = 64UL; ld[1] = 40UL;
    int li = 0;
    while (li < 2) {
        unsigned long n = 1100UL;
        struct Vec<float> q = seq(2UL * n * ld[li], 0.013f, -0.6f);
        struct Vec<float> k = seq(2UL * n * ld[li], 0.011f, -0.55f);
        struct Vec<float> v = seq(2UL * n * ld[li], 0.02f, -1.0f);
        struct Vec<float> got = std::Vec::with_cap(2UL * n * ld[li]);
        unsigned long zz = 0UL;
        while (zz < 2UL * n * ld[li]) { got.push(0.0f); zz = zz + 1UL; }
        struct Vec<float> ref = attn_ref(&q, &k, &v, 2UL, n, n, ld[li], 1.0f, 1, 1);
        int ok = std::rocm_linear_attention_f32(q.at(0UL), k.at(0UL), v.at(0UL), got.at(0UL), 2UL, n, n, ld[li], 1);
        check(ok && relmax(&got, &ref) < 1.0e-4f, "causal linear attention, 1100 rows");
        // the same call recorded into a graph (its per-call reset is a graph node) and replayed twice
        struct RocmBuf dq = RocmBuf::upload(q.at(0UL), 2UL * n * ld[li]);
        struct RocmBuf dk = RocmBuf::upload(k.at(0UL), 2UL * n * ld[li]);
        struct RocmBuf dv = RocmBuf::upload(v.at(0UL), 2UL * n * ld[li]);
        struct RocmBuf dout = RocmBuf::alloc(2UL * n * ld[li]);
        check(std::rocm_linear_attention(dq, dk, dv, dout, 2UL, n, n, ld[li], 1), "linear attention before capture");
        check(std::rocm_capture_begin(), "capture linear attention");
        std::rocm_linear_attention(dq, dk, dv, dout, 2UL, n, n, ld[li], 1);
        struct RocmGraph g = std::rocm_capture_end();
        check(g.exec != 0UL && std::rocm_graph_launch(g) && std::rocm_graph_launch(g), "linear attention graph replay");
        struct Vec<float> gg = download(dout);
        check(relmax(&gg, &ref) < 1.0e-4f, "linear attention graph values");
        gg.free();
        g.free();
        dq.free(); dk.free(); dv.free(); dout.free();
        ref.free(); q.free(); k.free(); v.free(); got.free();
        li = li + 1;
    }
    // same definitions as std/ml/attention.h's CPU functions (one head)
    unsigned long n1 = 50UL;
    unsigned long d1 = 32UL;
    struct Vec<float> q1 = seq(n1 * d1, 0.017f, -0.8f);
    struct Vec<float> k1 = seq(n1 * d1, 0.015f, -0.7f);
    struct Vec<float> v1 = seq(n1 * d1, 0.03f, -1.2f);
    &FloatTensor tq = FloatTensor::from_2d(q1.at(0UL), n1, d1, 0);
    &FloatTensor tk = FloatTensor::from_2d(k1.at(0UL), n1, d1, 0);
    &FloatTensor tv = FloatTensor::from_2d(v1.at(0UL), n1, d1, 0);
    &FloatTensor sa = std::attention_forward(tq, tk, tv);
    &FloatTensor la = std::linear_attention_forward(tq, tk, tv);
    struct Vec<float> g1 = std::Vec::with_cap(n1 * d1);
    struct Vec<float> c1 = std::Vec::with_cap(n1 * d1);
    struct Vec<float> c2 = std::Vec::with_cap(n1 * d1);
    unsigned long z = 0UL;
    while (z < n1 * d1) { g1.push(0.0f); unsafe { c1.push(sa->data[z]); c2.push(la->data[z]); } z = z + 1UL; }
    check(std::rocm_flash_attention_f32(q1.at(0UL), k1.at(0UL), v1.at(0UL), g1.at(0UL), 1UL, n1, n1, d1, 0.0f, 0) &&
          relmax(&g1, &c1) < 1.0e-4f, "flash attention == attention_forward");
    check(std::rocm_linear_attention_f32(q1.at(0UL), k1.at(0UL), v1.at(0UL), g1.at(0UL), 1UL, n1, n1, d1, 0) &&
          relmax(&g1, &c2) < 1.0e-4f, "linear attention == linear_attention_forward");
    // bad shapes
    struct RocmBuf big = RocmBuf::alloc(n1 * 129UL);
    check(big.dev != 0UL && !std::rocm_flash_attention(big, big, big, big, 1UL, n1, n1, 129UL, 0.0f, 0) &&
          !std::rocm_linear_attention(big, big, big, big, 1UL, n1, n1, 129UL, 0), "headDim > 128 rejected");
    check(!std::rocm_flash_attention(big, big, big, big, 1UL, n1, n1 - 1UL, 64UL, 0.0f, 1), "causal flash rejects kv < sl");
    big.free();
    sa.free(); la.free(); tq.free(); tk.free(); tv.free();
    q1.free(); k1.free(); v1.free(); g1.free(); c1.free(); c2.free();
}


// Mixed precision: fp16 <-> bf16, many-buffer conversion and unscaling with
// the overflow flag, master-weight SGD / Adam (against the CPU arithmetic,
// skipped when flagged), 16-bit elementwise ops and bias.
static struct Vec<unsigned short> read_half(struct RocmHalfBuf h) {
    struct Vec<unsigned short> v = std::Vec::with_cap(h.len);
    unsigned long i = 0UL;
    while (i < h.len) { v.push((unsigned short)0); i = i + 1UL; }
    std::rocm_half_read(h, v.at(0UL), h.len);
    return v;
}

static int half_nan_(unsigned short h, int d) {
    unsigned int u = (unsigned int)h & 0x7FFFU;
    return d == ROCM_BF16 ? u > 0x7F80U : u > 0x7C00U;
}

static void check_mixed_precision() {
    // fp16 <-> bf16 over every 16-bit pattern, against float16.h's bulk conversions
    unsigned long n = 65536UL;
    struct Vec<unsigned short> pat = std::Vec::with_cap(n);
    struct Vec<unsigned short> cpu = std::Vec::with_cap(n);
    unsigned long i = 0UL;
    while (i < n) { pat.push((unsigned short)i); cpu.push((unsigned short)0); i = i + 1UL; }
    int d = ROCM_FP16;
    while (d <= ROCM_BF16) {
        int to = d == ROCM_FP16 ? ROCM_BF16 : ROCM_FP16;
        struct RocmHalfBuf hs = RocmHalfBuf::alloc(n, d);
        struct RocmHalfBuf hd = RocmHalfBuf::alloc(n, to);
        check(std::rocm_half_write(hs, pat.at(0UL), n) && std::rocm_half_convert(hs, hd), "half convert ran");
        struct Vec<unsigned short> got = read_half(hd);
        if (d == ROCM_FP16) std::fp16_to_bf16_bulk(pat.at(0UL), cpu.at(0UL), n);
        else std::bf16_to_fp16_bulk(pat.at(0UL), cpu.at(0UL), n);
        unsigned long bad = 0UL;
        i = 0UL;
        while (i < n) {
            if (half_nan_(*pat.at(i), d)) { if (!half_nan_(*got.at(i), to)) bad = bad + 1UL; }
            else if (*got.at(i) != *cpu.at(i)) bad = bad + 1UL;
            i = i + 1UL;
        }
        check(bad == 0UL, d == ROCM_FP16 ? "fp16 -> bf16 == CPU" : "bf16 -> fp16 == CPU");
        got.free();
        hs.free();
        hd.free();
        d = d + 1;
    }

    // 40 buffers (two launches) to bf16 and back, lengths 1 .. 2000
    struct Vec<struct RocmBuf> f32s = std::Vec::new();
    struct Vec<struct RocmHalfBuf> h16s = std::Vec::new();
    struct Vec<struct RocmBuf> backs = std::Vec::new();
    struct Vec<float> host = seq(2000UL, (float)0.37, -(float)17.0);
    unsigned long b = 0UL;
    while (b < 40UL) {
        unsigned long len = 1UL + (b * 977UL) % 2000UL;
        f32s.push(RocmBuf::upload(host.at(0UL), len));
        h16s.push(RocmHalfBuf::alloc(len, ROCM_BF16));
        backs.push(RocmBuf::alloc(len));
        b = b + 1UL;
    }
    check(std::rocm_to_half_many(&f32s, &h16s) && std::rocm_from_half_many(&h16s, &backs), "many conversions ran");
    unsigned long bad = 0UL;
    b = 0UL;
    while (b < 40UL) {
        struct RocmBuf back = *backs.at(b);
        struct Vec<float> got = download(back);
        i = 0UL;
        while (i < back.len) {
            if (*got.at(i) != std::bf16_to_f32(std::f32_to_bf16(*host.at(i)))) bad = bad + 1UL;
            i = i + 1UL;
        }
        got.free();
        b = b + 1UL;
    }
    check(bad == 0UL, "40 buffers round-trip through bf16");
    // a list mixing fp16 and bf16 is refused
    struct RocmHalfBuf odd = RocmHalfBuf::alloc(5UL, ROCM_FP16);
    struct Vec<struct RocmHalfBuf> mixed = std::Vec::new();
    mixed.push(*h16s.at(0UL));
    mixed.push(odd);
    struct Vec<struct RocmBuf> two = std::Vec::new();
    two.push(*backs.at(0UL));
    two.push(*backs.at(1UL));
    check(!std::rocm_from_half_many(&mixed, &two), "mixed dtypes refused");

    // unscaling: fp16 gradients holding 512 x the value; then one overflow
    struct RocmBuf flag = RocmBuf::alloc(1UL);
    struct Vec<float> gv = seq(3000UL, (float)0.01, -(float)0.5);
    struct Vec<float> scaled = std::Vec::with_cap(3000UL);
    i = 0UL;
    while (i < 3000UL) { scaled.push(*gv.at(i) * (float)512.0); i = i + 1UL; }
    struct RocmHalfBuf g16a = RocmHalfBuf::upload(scaled.at(0UL), 3000UL, ROCM_FP16);
    struct RocmHalfBuf g16b = RocmHalfBuf::upload(scaled.at(0UL), 1000UL, ROCM_FP16);
    struct RocmBuf g32a = RocmBuf::alloc(3000UL);
    struct RocmBuf g32b = RocmBuf::alloc(1000UL);
    struct Vec<struct RocmHalfBuf> gs = std::Vec::new();
    gs.push(g16a); gs.push(g16b);
    struct Vec<struct RocmBuf> g32s = std::Vec::new();
    g32s.push(g32a); g32s.push(g32b);
    check(std::rocm_zero(flag) && std::rocm_half_unscale_many(&gs, &g32s, (float)1.0 / (float)512.0, flag), "unscale ran");
    struct Vec<float> fl = download(flag);
    struct Vec<float> u = download(g32a);
    bad = 0UL;
    i = 0UL;
    while (i < 3000UL) {
        float want = std::fp16_to_f32(std::f32_to_fp16(*scaled.at(i))) * ((float)1.0 / (float)512.0);
        if (*u.at(i) != want) bad = bad + 1UL;
        i = i + 1UL;
    }
    check(bad == 0UL && *fl.at(0UL) == (float)0.0, "unscaled values, no overflow");
    fl.free(); u.free();
    unsigned short inf = (unsigned short)0x7C00;
    std::rocm_half_write(g16b, &inf, 1UL);
    check(std::rocm_half_unscale_many(&gs, &g32s, (float)1.0 / (float)512.0, flag), "unscale ran again");
    fl = download(flag);
    check(*fl.at(0UL) == (float)1.0, "overflow flagged");
    fl.free();
    // fp32 gradients in place
    struct Vec<struct RocmBuf> one = std::Vec::new();
    one.push(g32a);
    check(std::rocm_zero(flag) && std::rocm_unscale_many(&one, &one, (float)4.0, flag), "fp32 unscale in place");
    u = download(g32a);
    fl = download(flag);
    check(*u.at(10UL) == std::fp16_to_f32(std::f32_to_fp16(*scaled.at(10UL))) * ((float)1.0 / (float)512.0) * (float)4.0 &&
          *fl.at(0UL) == (float)0.0, "fp32 unscale values");
    u.free(); fl.free();

    // master-weight Adam (coupled and decoupled decay) and momentum SGD, 3 steps
    unsigned long m = 777UL;
    struct Vec<float> w0 = seq(m, (float)0.003, -(float)0.1);
    struct Vec<float> g0 = seq(m, (float)0.011, -(float)0.4);
    int variant = 0;
    while (variant < 3) {
        struct RocmBuf dw = RocmBuf::upload(w0.at(0UL), m);
        struct RocmBuf dg = RocmBuf::upload(g0.at(0UL), m);
        struct RocmBuf dm = RocmBuf::alloc(m);
        struct RocmBuf dvv = RocmBuf::alloc(m);
        struct RocmHalfBuf wh = RocmHalfBuf::alloc(m, ROCM_BF16);
        std::rocm_zero(dm); std::rocm_zero(dvv); std::rocm_zero(flag);
        struct Vec<float> cw = std::Vec::with_cap(m);
        struct Vec<float> cm = std::Vec::with_cap(m);
        struct Vec<float> cv = std::Vec::with_cap(m);
        i = 0UL;
        while (i < m) { cw.push(*w0.at(i)); cm.push((float)0.0); cv.push((float)0.0); i = i + 1UL; }
        float lr = (float)0.01; float b1 = (float)0.9; float b2 = (float)0.999; float eps = (float)1.0e-8; float wd = (float)0.05; float mu = (float)0.9;
        unsigned long t = 1UL;
        while (t <= 3UL) {
            if (variant < 2)
                check(std::rocm_mp_adam(dw, dm, dvv, dg, wh, lr, b1, b2, eps, wd, variant, t, flag), "mp adam ran");
            else
                check(std::rocm_mp_sgd(dw, dg, dm, wh, lr, mu, wd, flag), "mp sgd ran");
            float bc1 = (float)1.0 - std::pow_f(b1, (float)t);
            float bc2 = (float)1.0 - std::pow_f(b2, (float)t);
            i = 0UL;
            while (i < m) {
                float g = *g0.at(i);
                float w = *cw.at(i);
                if (variant < 2) {
                    if (variant == 0) g = g + wd * w;
                    float mi = b1 * *cm.at(i) + ((float)1.0 - b1) * g;
                    float vi = b2 * *cv.at(i) + ((float)1.0 - b2) * g * g;
                    *cm.at(i) = mi; *cv.at(i) = vi;
                    float upd = (mi / bc1) / (std::sqrt_f(vi / bc2) + eps);
                    if (variant == 1) upd = upd + wd * w;
                    *cw.at(i) = w - lr * upd;
                } else {
                    float gi = g + wd * w;
                    float v = mu * *cm.at(i) + gi;
                    *cm.at(i) = v;
                    *cw.at(i) = w - lr * v;
                }
                i = i + 1UL;
            }
            t = t + 1UL;
        }
        struct Vec<float> gw = download(dw);
        struct Vec<unsigned short> gh = read_half(wh);
        float worst = (float)0.0;
        bad = 0UL;
        i = 0UL;
        while (i < m) {
            float dd = *gw.at(i) - *cw.at(i);
            if (dd < (float)0.0) dd = -dd;
            float cwi = *cw.at(i);
            float sc = cwi < (float)0.0 ? (float)0.0 - cwi : cwi;
            dd = dd / (sc > (float)1.0e-3 ? sc : (float)1.0e-3);
            if (dd > worst) worst = dd;
            if (*gh.at(i) != std::f32_to_bf16(*gw.at(i))) bad = bad + 1UL;
            i = i + 1UL;
        }
        check(worst < (float)1.0e-5, variant == 0 ? "mp adam == CPU" : (variant == 1 ? "mp adamw == CPU" : "mp momentum sgd == CPU"));
        check(bad == 0UL, "master weights' bf16 copy");
        // a flagged step changes nothing
        float one1 = (float)1.0;
        std::rocm_write(flag, &one1, 1UL);
        if (variant < 2) std::rocm_mp_adam(dw, dm, dvv, dg, wh, lr, b1, b2, eps, wd, variant, 4UL, flag);
        else std::rocm_mp_sgd(dw, dg, dm, wh, lr, mu, wd, flag);
        struct Vec<float> gw2 = download(dw);
        check(maxdiff(&gw, &gw2) == (float)0.0, "flagged step skipped");
        gw.free(); gw2.free(); gh.free(); cw.free(); cm.free(); cv.free();
        dw.free(); dg.free(); dm.free(); dvv.free();
        wh.free();
        variant = variant + 1;
    }

    // 16-bit elementwise and bias, against CPU arithmetic on the decoded values
    unsigned long e = 4099UL;
    struct Vec<float> xa = seq(e, (float)0.13, -(float)6.0);
    struct Vec<float> xb = seq(e, -(float)0.07, (float)3.0);
    d = ROCM_FP16;
    while (d <= ROCM_BF16) {
        struct RocmHalfBuf ha = RocmHalfBuf::upload(xa.at(0UL), e, d);
        struct RocmHalfBuf hb = RocmHalfBuf::upload(xb.at(0UL), e, d);
        struct RocmHalfBuf ho = RocmHalfBuf::alloc(e, d);
        struct Vec<unsigned short> ab = read_half(ha);
        struct Vec<unsigned short> bb = read_half(hb);
        int op = 0;
        while (op < 5) {
            int ok = 0;
            if (op == 0) ok = std::rocm_half_add(ha, hb, ho);
            if (op == 1) ok = std::rocm_half_sub(ha, hb, ho);
            if (op == 2) ok = std::rocm_half_mul(ha, hb, ho);
            if (op == 3) ok = std::rocm_half_scale(ha, (float)0.3, ho);
            if (op == 4) ok = std::rocm_half_axpy(ha, -(float)1.7, hb, ho);
            struct Vec<unsigned short> ob = read_half(ho);
            bad = 0UL;
            i = 0UL;
            while (i < e) {
                float x = std::half_decode(*ab.at(i), d);
                float y = std::half_decode(*bb.at(i), d);
                float r = op == 0 ? x + y : (op == 1 ? x - y : (op == 2 ? x * y : (op == 3 ? x * (float)0.3 : x + -(float)1.7 * y)));
                if (*ob.at(i) != std::half_encode(r, d)) bad = bad + 1UL;
                i = i + 1UL;
            }
            check(ok && bad == 0UL, "half elementwise == CPU");
            ob.free();
            op = op + 1;
        }
        // NCHW bias + relu: 3 images x 7 channels x 5x5 (175 values per image, e >= 525)
        struct Vec<float> bias = seq(7UL, (float)0.5, -(float)1.5);
        struct RocmHalfBuf hbias = RocmHalfBuf::upload(bias.at(0UL), 7UL, d);
        struct Vec<unsigned short> biasb = read_half(hbias);
        check(std::rocm_half_bias(ha, hbias, 3UL, 7UL, 25UL, 1), "half bias ran");
        struct Vec<unsigned short> yb = read_half(ha);
        bad = 0UL;
        i = 0UL;
        while (i < e) {
            unsigned short want = *ab.at(i);
            if (i < 525UL) {
                float r = std::half_decode(*ab.at(i), d) + std::half_decode(*biasb.at((i / 25UL) % 7UL), d);
                if (r < (float)0.0) r = (float)0.0;
                want = std::half_encode(r, d);
            }
            if (*yb.at(i) != want) bad = bad + 1UL;
            i = i + 1UL;
        }
        check(bad == 0UL, "half NCHW bias + relu == CPU");
        yb.free(); biasb.free(); bias.free(); ab.free(); bb.free();
        ha.free(); hb.free(); ho.free(); hbias.free();
        d = d + 1;
    }

    b = 0UL;
    while (b < 40UL) {
        f32s.at(b).free(); h16s.at(b).free(); backs.at(b).free();
        b = b + 1UL;
    }
    odd.free();
    g16a.free(); g16b.free();
    g32a.free(); g32b.free(); flag.free();
    f32s.free(); h16s.free(); backs.free(); mixed.free(); two.free(); gs.free(); g32s.free(); one.free();
    pat.free(); cpu.free(); host.free(); gv.free(); scaled.free(); w0.free(); g0.free(); xa.free(); xb.free();
}


int main() {
    if (!std::rocm_available()) { unsafe { printf("no ROCm GPU -- skipped\n"); } return 0; }
    unsafe { printf("device: %s\n", std::rocm_device_name()); }

    unsigned long n = 10007UL;   // not a multiple of the block size
    struct Vec<float> a = seq(n, 0.01f, 0.1f);
    struct Vec<float> b = seq(n, 0.02f, 0.5f);
    struct RocmBuf da = RocmBuf::upload(a.at(0UL), n);
    struct RocmBuf db = RocmBuf::upload(b.at(0UL), n);
    struct RocmBuf dout = RocmBuf::alloc(n);
    check(da.dev != 0UL && db.dev != 0UL && dout.dev != 0UL, "upload/alloc");

    // elementwise vs CPU
    struct Vec<float> want = std::Vec::with_cap(n);
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
    struct RocmBuf d1 = RocmBuf::alloc(1UL);
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
    struct RocmBuf small = RocmBuf::alloc(10UL);
    check(!std::rocm_add(small, db, dout), "length mismatch rejected");

    // gemm, every transpose combination, odd sizes
    unsigned long m = 67UL;
    unsigned long nn = 45UL;
    unsigned long k = 129UL;
    struct Vec<float> ga = seq(m * k, 0.003f, -0.1f);
    struct Vec<float> gb = seq(k * nn, 0.002f, -0.07f);
    struct RocmBuf dga = RocmBuf::upload(ga.at(0UL), m * k);
    struct RocmBuf dgb = RocmBuf::upload(gb.at(0UL), k * nn);
    struct RocmBuf dgc = RocmBuf::alloc(m * nn);
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
    struct Vec<float> hc = std::Vec::with_cap(m * nn);
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
    g.free();

    da.free(); db.free(); dout.free(); d1.free();
    small.free(); dga.free(); dgb.free(); dgc.free();
    check(da.dev == 0UL, "free clears handle");
    a.free(); b.free(); want.free(); ga.free(); gb.free(); ref.free(); hc.free();

    check_half();
    check_mixed_precision();
    check_decode_ops();

    // matrix-core kernels (when the GPU has them), then the CUDA-core ones
    int mf = std::rocm_attention_use_mfma(1);
    unsafe { printf("attention: %s kernels\n", mf ? "matrix-core" : "CUDA-core"); }
    check_attention();
    if (mf) {
        check(!std::rocm_attention_use_mfma(0), "matrix-core attention can be switched off");
        unsafe { printf("attention: CUDA-core kernels\n"); }
        check_attention();
        check(std::rocm_attention_use_mfma(1), "matrix-core attention switched back on");
    }
    unsafe { printf("rocm: %d checks, %d failures\n", g_checks, g_fails); }
    return g_fails == 0 ? 0 : 1;
}
