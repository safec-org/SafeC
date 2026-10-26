// Correctness check for std/ml/gpu_cuda against CPU reference results.
// Needs an NVIDIA GPU; run with tests/stdlib/cuda/run.sh.
#include <std/ml/gpu_cuda.h>
#include <std/ml/float16.h>
#include <std/ml/tensor_half.h>
#include <std/collections/vec.h>
#include <std/math.h>
#include <std/ml/gpu_cudnn.h>
#include <std/ml/cnn.h>
#include <std/ml/unet.h>
#include <std/dsp/fft_cuda.h>
#include <std/dsp/fft.h>
#include <std/dsp/dft.h>
#include <std/dsp/imaging.h>
#include <std/mem.h>
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

static struct Vec<float> download(struct CudaBuf b) {
    struct Vec<float> v = std::Vec::with_cap(b.len);
    unsigned long i = 0UL;
    while (i < b.len) { v.push(0.0f); i = i + 1UL; }
    std::cuda_read(b, v.at(0UL), b.len);
    return v;
}

// ── cuDNN (gpu_cudnn) and cuFFT (dsp/fft_cuda) ──────────────────────────────

static void fm_fill(&FeatureMap fm, unsigned long salt) {
    unsigned long n = fm.channels * fm.height * fm.width;
    unsigned long i = 0UL;
    unsafe { while (i < n) { fm.data[i] = (float)(((i + salt) * 37UL) % 101UL) * (float)0.02 - (float)1.0; i = i + 1UL; } }
}

static void conv_bias(&Conv2D layer) {
    unsigned long i = 0UL;
    unsafe { while (i < layer.outChannels) { layer.bias[i] = (float)i * (float)0.1 - (float)0.3; i = i + 1UL; } }
}

// max |a - b| over n floats, relative to max(1, max |b|)
static float fm_err(const float* a, const float* b, unsigned long n) {
    float m = 0.0f;
    float big = 1.0f;
    unsigned long i = 0UL;
    unsafe {
        while (i < n) {
            float d = a[i] - b[i];
            if (d < 0.0f) d = -d;
            if (d > m) m = d;
            float bb = b[i] < 0.0f ? -b[i] : b[i];
            if (bb > big) big = bb;
            i = i + 1UL;
        }
    }
    return m / big;
}

static unsigned long fm_size(const &FeatureMap fm) { return fm.channels * fm.height * fm.width; }

static struct Vec<float> fm_download(struct CudaBuf b, unsigned long off, unsigned long n) {
    struct Vec<float> all = download(b);
    struct Vec<float> v = std::Vec::with_cap(n);
    unsigned long i = 0UL;
    while (i < n) { v.push(*all.at(off + i)); i = i + 1UL; }
    all.free();
    return v;
}

static void check_cudnn() {
    // conv2d_forward_cuda vs Conv2D::forward: 3x3, stride 1, pad 1, odd sizes
    struct Conv2D l1 = Conv2D::new(3UL, 8UL, 3UL, 3UL, 1UL, 1UL);
    conv_bias(&l1);
    struct FeatureMap x1 = FeatureMap::new(3UL, 17UL, 13UL);
    fm_fill(&x1, 0UL);
    struct FeatureMap c1 = l1.forward(&x1);
    struct FeatureMap g1 = std::conv2d_forward_cuda(&l1, &x1);
    check(g1.channels == 8UL && g1.height == 17UL && g1.width == 13UL, "conv2d_forward_cuda shape");
    unsafe { check(fm_err((const float*)g1.data, (const float*)c1.data, fm_size(&c1)) < 1e-5f, "conv2d_forward_cuda values"); }
    g1.free();

    // queued conv on a batch of two images: 5x5, stride 2, pad 0; then + relu
    struct Conv2D l2 = Conv2D::new(4UL, 6UL, 5UL, 5UL, 2UL, 0UL);
    conv_bias(&l2);
    struct FeatureMap xa = FeatureMap::new(4UL, 21UL, 15UL);
    struct FeatureMap xb = FeatureMap::new(4UL, 21UL, 15UL);
    fm_fill(&xa, 3UL);
    fm_fill(&xb, 50UL);
    struct FeatureMap ca = l2.forward(&xa);
    struct FeatureMap cb = l2.forward(&xb);
    unsigned long inN = fm_size(&xa);
    unsigned long outN = fm_size(&ca);
    check(std::cuda_conv_out_dim(21UL, 5UL, 2UL, 0UL) == ca.height && std::cuda_conv_out_dim(15UL, 5UL, 2UL, 0UL) == ca.width,
          "conv out dim");
    struct CudaConv2D d2 = CudaConv2D::upload(&l2);
    check(d2.weight.dev != 0UL && d2.bias.dev != 0UL, "conv2d upload");
    struct CudaBuf dx = CudaBuf::alloc(2UL * inN);
    struct CudaBuf dy = CudaBuf::alloc(2UL * outN);
    unsafe {
        std::cuda_write(dx, xa.data, inN);
        struct CudaBuf dx2 = dx;
        dx2.dev = dx.dev + inN * 4UL;
        dx2.len = inN;
        std::cuda_write(dx2, xb.data, inN);
    }
    check(d2.forward(dx, dy, 2UL, 21UL, 15UL), "CudaConv2D::forward batch ran");
    struct Vec<float> ya = fm_download(dy, 0UL, outN);
    struct Vec<float> yb = fm_download(dy, outN, outN);
    unsafe {
        check(fm_err(ya.at(0UL), (const float*)ca.data, outN) < 1e-5f &&
              fm_err(yb.at(0UL), (const float*)cb.data, outN) < 1e-5f, "CudaConv2D::forward batch values");
    }
    ya.free(); yb.free();
    check(d2.forward_relu(dx, dy, 2UL, 21UL, 15UL), "CudaConv2D::forward_relu ran");
    ya = fm_download(dy, outN, outN);
    unsafe {
        unsigned long i = 0UL;
        while (i < outN) { if (cb.data[i] < 0.0f) cb.data[i] = 0.0f; i = i + 1UL; }
        check(fm_err(ya.at(0UL), (const float*)cb.data, outN) < 1e-5f, "CudaConv2D::forward_relu values");
    }
    ya.free();
    check(!d2.forward(dx, dy, 3UL, 21UL, 15UL), "CudaConv2D::forward rejects short buffers");

    // the cached plan replays inside a graph
    check(std::cuda_capture_begin(), "conv capture begin");
    d2.forward(dx, dy, 2UL, 21UL, 15UL);
    struct CudaGraph cg = std::cuda_capture_end();
    check(cg.exec != 0UL, "conv capture end");
    std::cuda_scale(dy, 0.0f, dy);
    check(std::cuda_graph_launch(cg), "conv graph launch");
    ya = fm_download(dy, 0UL, outN);
    unsafe { check(fm_err(ya.at(0UL), (const float*)ca.data, outN) < 1e-5f, "conv graph values"); }
    ya.free();
    cg.free();

    // TF32 tensor cores: same result to ~1e-3
    check(std::cuda_set_tf32(1), "tf32 on");
    check(d2.forward(dx, dy, 2UL, 21UL, 15UL), "tf32 conv ran");
    ya = fm_download(dy, 0UL, outN);
    unsafe { check(fm_err(ya.at(0UL), (const float*)ca.data, outN) < 1e-2f, "tf32 conv values"); }
    ya.free();
    std::cuda_set_tf32(0);

    // pooling, upsample and concat against cnn.sc
    struct FeatureMap mp = xa.maxpool2d(3UL, 2UL);
    struct FeatureMap ap = xa.avgpool2d(3UL, 2UL);
    unsigned long pn = fm_size(&mp);
    struct CudaBuf dp = CudaBuf::alloc(2UL * pn);
    check(std::cuda_maxpool2d(dx, dp, 2UL, 4UL, 21UL, 15UL, 3UL, 2UL), "cuda_maxpool2d ran");
    ya = fm_download(dp, 0UL, pn);
    unsafe { check(fm_err(ya.at(0UL), (const float*)mp.data, pn) == 0.0f, "cuda_maxpool2d values"); }
    ya.free();
    check(std::cuda_avgpool2d(dx, dp, 2UL, 4UL, 21UL, 15UL, 3UL, 2UL), "cuda_avgpool2d ran");
    ya = fm_download(dp, 0UL, pn);
    unsafe { check(fm_err(ya.at(0UL), (const float*)ap.data, pn) < 1e-6f, "cuda_avgpool2d values"); }
    ya.free();

    struct FeatureMap up = xa.upsample2x_nearest();
    struct CudaBuf du = CudaBuf::alloc(4UL * inN);
    check(std::cuda_upsample2x_nearest(dx, du, 4UL, 21UL, 15UL), "cuda_upsample2x ran");
    ya = fm_download(du, 0UL, 4UL * inN);
    unsafe { check(fm_err(ya.at(0UL), (const float*)up.data, 4UL * inN) == 0.0f, "cuda_upsample2x values"); }
    ya.free();

    struct FeatureMap cat = std::concat_channels(&xa, &xb);
    struct CudaBuf dxa = CudaBuf::upload(xa.data, inN);
    struct CudaBuf dxb = CudaBuf::upload(xb.data, inN);
    struct CudaBuf dc = CudaBuf::alloc(2UL * inN);
    check(std::cuda_concat_channels(dxa, 4UL, dxb, 4UL, dc, 1UL, 21UL * 15UL), "cuda_concat_channels ran");
    ya = download(dc);
    unsafe { check(fm_err(ya.at(0UL), (const float*)cat.data, 2UL * inN) == 0.0f, "cuda_concat_channels values"); }
    ya.free();

    // batch-2 concat of 4x4 planes (the float4 copy path)
    struct CudaBuf dc2 = CudaBuf::alloc(2UL * 2UL * 4UL * 16UL);
    check(std::cuda_concat_channels(dx, 2UL, dx, 2UL, dc2, 2UL, 16UL) &&
          std::cuda_concat_channels(dx, 3UL, dx, 1UL, dc2, 2UL, 16UL), "cuda_concat_channels batch ran");
    ya = download(dc2);
    yb = download(dx);
    unsafe {
        // out[b] = (a[b] with 3 channels, c[b] with 1 channel), a = c = dx read as [2, 3|1, 16]
        int same = 1;
        unsigned long b = 0UL;
        while (b < 2UL) {
            unsigned long i = 0UL;
            while (i < 64UL) {
                float want = i < 48UL ? *yb.at(b * 48UL + i) : *yb.at(b * 16UL + i - 48UL);
                if (*ya.at(b * 64UL + i) != want) same = 0;
                i = i + 1UL;
            }
            b = b + 1UL;
        }
        check(same, "cuda_concat_channels batch values");
    }
    ya.free(); yb.free();

    // strided row copies: float4 and scalar paths, bounds
    check(std::cuda_copy_rows(dx, 4UL, 12UL, dc2, 8UL, 16UL, 8UL, 5UL) &&
          std::cuda_copy_rows(dx, 1UL, 7UL, dc2, 100UL, 9UL, 5UL, 3UL), "cuda_copy_rows ran");
    ya = download(dc2);
    yb = download(dx);
    unsafe {
        int same = 1;
        unsigned long r = 0UL;
        while (r < 5UL) {
            unsigned long c = 0UL;
            while (c < 8UL) { if (*ya.at(8UL + r * 16UL + c) != *yb.at(4UL + r * 12UL + c)) same = 0; c = c + 1UL; }
            r = r + 1UL;
        }
        r = 0UL;
        while (r < 3UL) {
            unsigned long c = 0UL;
            while (c < 5UL) { if (*ya.at(100UL + r * 9UL + c) != *yb.at(1UL + r * 7UL + c)) same = 0; c = c + 1UL; }
            r = r + 1UL;
        }
        check(same, "cuda_copy_rows values");
    }
    ya.free(); yb.free();
    check(!std::cuda_copy_rows(dx, 0UL, 4UL, dc2, 0UL, 16UL, 8UL, 2UL) &&
          !std::cuda_copy_rows(dx, 0UL, 8UL, dc2, 250UL, 8UL, 8UL, 2UL), "cuda_copy_rows rejects overlap / overrun");
    dc2.free();

    // a cached conv plan run on other buffers (rebinds its variant pack)
    struct CudaBuf dy2 = CudaBuf::alloc(2UL * outN);
    check(d2.forward(dx, dy2, 2UL, 21UL, 15UL), "CudaConv2D::forward other buffers ran");
    yb = fm_download(dy2, 0UL, outN);
    unsafe { check(fm_err(yb.at(0UL), (const float*)ca.data, outN) < 1e-5f, "CudaConv2D::forward other buffers values"); }
    yb.free();
    dy2.free();

    // whole U-Net
    struct UNet net = UNet::new(3UL, 8UL, 16UL, 2UL);
    struct FeatureMap ux = FeatureMap::new(3UL, 32UL, 24UL);
    fm_fill(&ux, 7UL);
    struct FeatureMap uc = net.forward(&ux);
    struct FeatureMap ug = std::unet_forward_cuda(&net, &ux);
    check(ug.channels == 2UL && ug.height == 32UL && ug.width == 24UL, "unet_forward_cuda shape");
    unsafe { check(fm_err((const float*)ug.data, (const float*)uc.data, fm_size(&uc)) < 1e-4f, "unet_forward_cuda values"); }

    ug.free(); uc.free(); ux.free();
    net.free();
    cat.free(); up.free(); mp.free(); ap.free();
    dxa.free(); dxb.free(); dc.free(); du.free(); dp.free();
    dx.free(); dy.free();
    d2.free();
    ca.free(); cb.free(); xa.free(); xb.free();
    c1.free(); x1.free();
    l1.free(); l2.free();
}

// A slice over n Complex values at p (test buffers from cplx_new).
static []Complex cs(&Complex p, unsigned long n) {
    unsafe { return ((struct Complex*)p)[0..n]; }
}

static &heap Complex cplx_new(unsigned long n, unsigned long salt) {
    ?&heap Complex q = std::alloc(n * sizeof(struct Complex));
    match (q) {
        case some(p): {
            unsafe {
                unsigned long i = 0UL;
                while (i < n) {
                    p[i].re = (double)(((i + salt) * 37UL) % 101UL) * 0.02 - 1.0;
                    p[i].im = (double)(((i + salt) * 53UL) % 89UL) * 0.01 - 0.4;
                    i = i + 1UL;
                }
            }
            return p;
        }
        case null: { std::panic_at("cplx_new: out of memory", "cuda_check.sc", 0); }
    }
    std::panic_at("cplx_new: unreachable", "cuda_check.sc", 0);
}

// max |a - b| / max(1, max |b|) over n complex values
static double cplx_err(const &Complex a, const &Complex b, unsigned long n) {
    double m = 0.0;
    double big = 1.0;
    unsigned long i = 0UL;
    unsafe {
        while (i < n) {
            double dr = a[i].re - b[i].re;
            double di = a[i].im - b[i].im;
            double d = dr * dr + di * di;
            double bb = b[i].re * b[i].re + b[i].im * b[i].im;
            if (d > m) m = d;
            if (bb > big) big = bb;
            i = i + 1UL;
        }
    }
    return std::sqrt_d(m / big);
}

static void cplx_copy(&Complex dst, const &Complex src, unsigned long n) {
    unsigned long i = 0UL;
    unsafe { while (i < n) { dst[i] = src[i]; i = i + 1UL; } }
}

// fp16 / bf16 (CudaHalfBuf): GPU rounding matches float16.h bit for bit;
// 16-bit gemm against an fp32 CPU gemm of the same rounded inputs; SGD,
// MSE grad, ReLU (grad) exact on the 16-bit values.
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
    struct CudaBuf dv = CudaBuf::upload(v.at(0UL), n);
    int d = CUDA_FP16;
    while (d <= CUDA_BF16) {
        int bf = d == CUDA_BF16;
        // round on the GPU, compare bits with the CPU encoder
        struct CudaHalfBuf h = CudaHalfBuf::alloc(n, d);
        check(h.dev != 0UL && std::cuda_to_half(dv, h), bf ? "bf16 convert ran" : "fp16 convert ran");
        struct Vec<unsigned short> bits = std::Vec::with_cap(n);
        unsigned long i = 0UL;
        while (i < n) { bits.push((unsigned short)0); i = i + 1UL; }
        check(std::cuda_half_read(h, bits.at(0UL), n), "half read");
        unsigned long bad = 0UL;
        i = 0UL;
        while (i < n) { if (*bits.at(i) != std::half_encode(*v.at(i), d)) bad = bad + 1UL; i = i + 1UL; }
        check(bad == 0UL, bf ? "bf16 rounding matches float16.h" : "fp16 rounding matches float16.h");
        // decode on the GPU: exact
        struct CudaBuf back = CudaBuf::alloc(n);
        check(std::cuda_from_half(h, back), "half decode ran");
        struct Vec<float> fb = download(back);
        bad = 0UL;
        i = 0UL;
        while (i < n) { if (*fb.at(i) != std::half_decode(*bits.at(i), d)) bad = bad + 1UL; i = i + 1UL; }
        check(bad == 0UL, bf ? "bf16 decode exact" : "fp16 decode exact");
        // CudaHalfBuf::upload rounds the same way
        struct CudaHalfBuf hu = CudaHalfBuf::upload(v.at(0UL), n, d);
        struct Vec<unsigned short> ubits = std::Vec::with_cap(n);
        i = 0UL;
        while (i < n) { ubits.push((unsigned short)0); i = i + 1UL; }
        check(hu.dev != 0UL && std::cuda_half_read(hu, ubits.at(0UL), n), "half upload");
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
        struct CudaHalfBuf ha = CudaHalfBuf::upload(a.at(0UL), m * k, d);
        struct CudaHalfBuf hb = CudaHalfBuf::upload(b.at(0UL), k * nn, d);
        struct CudaBuf c32 = CudaBuf::alloc(m * nn);
        struct CudaHalfBuf c16 = CudaHalfBuf::alloc(m * nn, d);
        struct Vec<unsigned short> cb = std::Vec::with_cap(m * nn);
        i = 0UL;
        while (i < m * nn) { cb.push((unsigned short)0); i = i + 1UL; }
        int t = 0;
        while (t < 4) {
            int ta = t & 1;
            int tb = (t >> 1) & 1;
            struct Vec<float> ref = cpu_gemm(&ar, ta, &br, tb, m, nn, k);
            check(std::cuda_half_gemm_f32(ha, ta, hb, tb, c32, m, nn, k), "half gemm f32 ran");
            struct Vec<float> got = download(c32);
            check(maxdiff(&got, &ref) < 1e-4f, bf ? "bf16 gemm f32 values" : "fp16 gemm f32 values");
            got.free();
            // 16-bit output: within one 16-bit rounding of the fp32 result
            check(std::cuda_half_gemm(ha, ta, hb, tb, c16, m, nn, k) && std::cuda_half_read(c16, cb.at(0UL), m * nn),
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
        struct CudaHalfBuf hg = CudaHalfBuf::alloc(m * k, d);
        check(std::cuda_half_write(hg, gbits.at(0UL), m * k), "half write");
        check(std::cuda_half_read(ha, wb.at(0UL), m * k), "half read w");
        float lr = (float)0.25;
        check(std::cuda_half_sgd(ha, hg, lr), "half sgd ran");
        struct Vec<unsigned short> wn = std::Vec::with_cap(m * k);
        i = 0UL;
        while (i < m * k) { wn.push((unsigned short)0); i = i + 1UL; }
        check(std::cuda_half_read(ha, wn.at(0UL), m * k), "half read sgd");
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

        // training ops against the CPU on decoded values
        unsigned long tn = 3000UL;
        struct Vec<float> y = seq(tn, 0.013f, -0.6f);
        struct Vec<float> tt = seq(tn, 0.009f, -0.2f);
        struct CudaHalfBuf hy = CudaHalfBuf::upload(y.at(0UL), tn, d);
        struct CudaHalfBuf ht = CudaHalfBuf::upload(tt.at(0UL), tn, d);
        struct CudaHalfBuf hd = CudaHalfBuf::alloc(tn, d);
        struct CudaHalfBuf hr = CudaHalfBuf::alloc(tn, d);
        struct CudaBuf loss = CudaBuf::alloc(1UL);
        struct Vec<unsigned short> yb = std::Vec::with_cap(tn);
        struct Vec<unsigned short> tb_ = std::Vec::with_cap(tn);
        struct Vec<unsigned short> ob = std::Vec::with_cap(tn);
        i = 0UL;
        while (i < tn) { yb.push((unsigned short)0); tb_.push((unsigned short)0); ob.push((unsigned short)0); i = i + 1UL; }
        std::cuda_half_read(hy, yb.at(0UL), tn);
        std::cuda_half_read(ht, tb_.at(0UL), tn);
        check(std::cuda_half_mse_grad(hy, ht, hd, loss) && std::cuda_half_read(hd, ob.at(0UL), tn), "half mse_grad ran");
        double lsum = 0.0;
        bad = 0UL;
        i = 0UL;
        while (i < tn) {
            float dd = std::half_decode(*yb.at(i), d) - std::half_decode(*tb_.at(i), d);
            lsum = lsum + (double)(dd * dd);
            if (*ob.at(i) != std::half_encode((float)2.0 * dd, d)) bad = bad + 1UL;
            i = i + 1UL;
        }
        struct Vec<float> lv = download(loss);
        double lerr = ((double)*lv.at(0UL) - lsum) / lsum;
        if (lerr < 0.0) lerr = -lerr;
        check(bad == 0UL && lerr < 1e-5, bf ? "bf16 mse_grad values" : "fp16 mse_grad values");
        lv.free();
        check(std::cuda_half_relu(hy, hr) && std::cuda_half_read(hr, ob.at(0UL), tn), "half relu ran");
        bad = 0UL;
        i = 0UL;
        while (i < tn) {
            unsigned short want = std::half_decode(*yb.at(i), d) > 0.0f ? *yb.at(i) : (unsigned short)0;
            if (*ob.at(i) != want) bad = bad + 1UL;
            i = i + 1UL;
        }
        check(bad == 0UL, bf ? "bf16 relu values" : "fp16 relu values");
        check(std::cuda_half_relu_grad(hy, ht, hr) && std::cuda_half_read(hr, ob.at(0UL), tn), "half relu_grad ran");
        bad = 0UL;
        i = 0UL;
        while (i < tn) {
            unsigned short want = std::half_decode(*yb.at(i), d) > 0.0f ? *tb_.at(i) : (unsigned short)0;
            if (*ob.at(i) != want) bad = bad + 1UL;
            i = i + 1UL;
        }
        check(bad == 0UL, bf ? "bf16 relu_grad values" : "fp16 relu_grad values");
        // gemm + relu epilogue vs gemm then relu, both 16-bit
        struct CudaHalfBuf c16r = CudaHalfBuf::alloc(m * nn, d);
        struct Vec<unsigned short> cr = std::Vec::with_cap(m * nn);
        i = 0UL;
        while (i < m * nn) { cr.push((unsigned short)0); i = i + 1UL; }
        check(std::cuda_half_gemm(ha, 0, hb, 0, c16, m, nn, k) && std::cuda_half_read(c16, cb.at(0UL), m * nn) &&
              std::cuda_half_gemm_relu(ha, 0, hb, 0, c16r, m, nn, k) && std::cuda_half_read(c16r, cr.at(0UL), m * nn),
              "half gemm_relu ran");
        bad = 0UL;
        i = 0UL;
        while (i < m * nn) {
            float g = std::half_decode(*cb.at(i), d);
            float want = g > (float)0.0 ? g : (float)0.0;
            float e = std::half_decode(*cr.at(i), d) - want;
            if (e < 0.0f) e = -e;
            float aw = want < 0.0f ? -want : want;
            if (e > aw * (bf ? (float)0.0079 : (float)0.00099) + (float)1.0e-6) bad = bad + 1UL;
            i = i + 1UL;
        }
        check(bad == 0UL, bf ? "bf16 gemm_relu values" : "fp16 gemm_relu values");
        // sgd2 = two sgd calls, bit for bit
        struct CudaHalfBuf w1 = CudaHalfBuf::upload(y.at(0UL), tn, d);
        struct CudaHalfBuf w2 = CudaHalfBuf::upload(tt.at(0UL), 1000UL, d);
        struct CudaHalfBuf w1b = CudaHalfBuf::upload(y.at(0UL), tn, d);
        struct CudaHalfBuf w2b = CudaHalfBuf::upload(tt.at(0UL), 1000UL, d);
        struct CudaHalfBuf g2 = CudaHalfBuf::upload(y.at(0UL), 1000UL, d);
        check(std::cuda_half_sgd2(w1, ht, w2, g2, 0.125f) && std::cuda_half_sgd(w1b, ht, 0.125f) &&
              std::cuda_half_sgd(w2b, g2, 0.125f), "half sgd2 ran");
        struct Vec<unsigned short> o2 = std::Vec::with_cap(tn);
        i = 0UL;
        while (i < tn) { o2.push((unsigned short)0); i = i + 1UL; }
        std::cuda_half_read(w1, ob.at(0UL), tn);
        std::cuda_half_read(w1b, o2.at(0UL), tn);
        bad = 0UL;
        i = 0UL;
        while (i < tn) { if (*ob.at(i) != *o2.at(i)) bad = bad + 1UL; i = i + 1UL; }
        std::cuda_half_read(w2, ob.at(0UL), 1000UL);
        std::cuda_half_read(w2b, o2.at(0UL), 1000UL);
        i = 0UL;
        while (i < 1000UL) { if (*ob.at(i) != *o2.at(i)) bad = bad + 1UL; i = i + 1UL; }
        check(bad == 0UL, bf ? "bf16 sgd2 matches sgd" : "fp16 sgd2 matches sgd");
        hy.free(); ht.free(); hd.free(); hr.free();
        c16r.free(); w1.free(); w2.free(); w1b.free();
        w2b.free(); g2.free(); loss.free();
        y.free(); tt.free(); yb.free(); tb_.free(); ob.free(); o2.free(); cr.free();

        // dtype mismatches are rejected
        struct CudaHalfBuf other = CudaHalfBuf::alloc(k * nn, bf ? CUDA_FP16 : CUDA_BF16);
        check(!std::cuda_half_gemm_f32(ha, 0, other, 0, c32, m, nn, k), "half gemm dtype check");
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

// Large host <-> device copies (the pinned-lane path), odd sizes, and the
// ordering of a write after queued kernels that read the old contents.
// Decode-step ops: cuda_half_gemv (both dtypes, vector and scalar paths),
// cuda_rmsnorm (with and without weight, in place) and cuda_gelu.
static void check_decode_ops() {
    int dts[2];
    dts[0] = CUDA_BF16; dts[1] = CUDA_FP16;
    // k % 8 == 0 (16-byte loads; 2056 leaves a partial 256-wide step) and not;
    // n picks 4, 2 and 1 warps per row on a 132-SM GPU; ki == 3 runs the
    // 1536 case on a weight view 2 bytes in (unaligned: the scalar path).
    unsigned long ks[4];
    ks[0] = 1536UL; ks[1] = 2056UL; ks[2] = 100UL; ks[3] = 1536UL;
    unsigned long ns[3];
    ns[0] = 37UL; ns[1] = 1500UL; ns[2] = 3000UL;
    int di = 0;
    while (di < 2) {
        int ki = 0;
        while (ki < 12) {
            unsigned long n = ns[ki / 4];
            unsigned long k = ks[ki % 4];
            unsigned long off = ki % 4 == 3 ? 1UL : 0UL;
            struct Vec<float> w = seq(n * k, 0.0013f, -0.9f);
            struct Vec<float> x = seq(k, 0.017f, -0.7f);
            struct CudaHalfBuf dw = CudaHalfBuf::upload(w.at(0UL), n * k + off, dts[di]);
            struct Vec<unsigned short> bits = std::Vec::with_cap(n * k + off);
            unsigned long z = 0UL;
            while (z < n * k + off) { bits.push((unsigned short)0); z = z + 1UL; }
            std::cuda_half_read(dw, bits.at(0UL), n * k + off);   // the rounded weights the GPU uses
            struct CudaHalfBuf view = dw;
            view.dev = dw.dev + 2UL * off;
            view.len = n * k;
            struct CudaBuf dx = CudaBuf::upload(x.at(0UL), k);
            struct CudaBuf dy = CudaBuf::alloc(n);
            int ok = std::cuda_half_gemv(view, dx, dy, n, k);
            struct Vec<float> got = download(dy);
            float err = (float)0.0;
            float big = (float)0.0;
            unsigned long r = 0UL;
            while (r < n) {
                double ref = 0.0;
                unsigned long c = 0UL;
                while (c < k) {
                    float wv = dts[di] == CUDA_BF16 ? std::bf16_to_f32(*bits.at(off + r * k + c)) : std::fp16_to_f32(*bits.at(off + r * k + c));
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
    struct CudaBuf dx = CudaBuf::upload(x.at(0UL), rows * n);
    struct CudaBuf dw = CudaBuf::upload(w.at(0UL), n);
    struct CudaBuf dy = CudaBuf::alloc(rows * n);
    int ok1 = std::cuda_rmsnorm(dx, dw, dy, rows, n, (float)1e-6);
    struct Vec<float> y1 = download(dy);
    struct CudaBuf none;
    none.dev = 0UL; none.len = 0UL;
    int ok2 = std::cuda_rmsnorm(dx, none, dx, rows, n, (float)1e-6);
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
    struct CudaBuf dg = CudaBuf::upload(gx.at(0UL), 1000UL);
    struct CudaBuf dgo = CudaBuf::alloc(1000UL);
    int ok3 = std::cuda_gelu(dg, dgo);
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
    struct CudaHalfBuf small = CudaHalfBuf::alloc(10UL, CUDA_BF16);
    check(!std::cuda_half_gemv(small, dg, dgo, 5UL, 4UL), "half gemv rejects a too-small weight buffer");
    check(!std::cuda_rmsnorm(dx, dw, dy, 4UL, n, (float)1e-6), "rmsnorm rejects too many rows");
    small.free();
    x.free(); w.free(); y1.free(); y2.free(); gx.free(); go.free();
    dx.free(); dw.free(); dy.free(); dg.free(); dgo.free();
}

static void check_transfers() {
    unsigned long sizes[3];
    sizes[0] = 262145UL;     // just over 1 MiB of floats: one lane
    sizes[1] = 1048577UL;    // ~4 MiB: four lanes, ragged tail
    sizes[2] = 5000003UL;    // ~20 MiB: several chunks per lane
    int si = 0;
    while (si < 3) {
        unsigned long n;
        unsafe { n = sizes[si]; }
        struct Vec<float> h = seq(n, 0.001f, -3.0f);
        struct CudaBuf d = CudaBuf::upload(h.at(0UL), n);
        struct Vec<float> back = download(d);
        check(maxdiff(&h, &back) == 0.0f, "large upload/read round trip");
        back.free();
        // out = d * 2 queued, then d overwritten: out must see the old d
        struct CudaBuf out = CudaBuf::alloc(n);
        std::cuda_scale(d, 2.0f, out);
        struct Vec<float> h2 = seq(n, 0.002f, 1.0f);
        check(std::cuda_write(d, h2.at(0UL), n), "large write ran");
        back = download(out);
        int ok = 1;
        unsigned long i = 0UL;
        while (i < n + 7918UL) {
            unsigned long at = i < n ? i : n - 1UL;   // every 7919th value and the last
            float got = *back.at(at);
            float want = *h.at(at);
            if (got != want + want) ok = 0;
            i = i + 7919UL;
        }
        check(ok, "large write ordered after queued reads");
        back.free();
        back = download(d);
        check(maxdiff(&h2, &back) == 0.0f, "large write values");
        back.free();
        // raw byte copies at an offset
        unsafe { check(std::cuda_copy_to_device(d.dev + 4UL, ((const unsigned char*)h.at(0UL))[0..(n - 1UL) * 4UL]), "cuda_copy_to_device ran"); }
        struct Vec<float> raw = std::Vec::with_cap(n);
        unsigned long k = 0UL;
        while (k < n) { unsafe { raw.push(0.0f); } k = k + 1UL; }
        unsafe { check(std::cuda_copy_to_host(((unsigned char*)raw.at(0UL))[0..(n - 1UL) * 4UL], d.dev + 4UL), "cuda_copy_to_host ran"); }
        unsafe { check(*raw.at(0UL) == *h.at(0UL) && *raw.at(n - 2UL) == *h.at(n - 2UL), "raw byte copies at an offset"); }
        raw.free(); h.free(); h2.free();
        d.free();
        out.free();
        si = si + 1;
    }
}

static void check_cufft() {
    // 1D double, power of two, against fft.h
    unsigned long n = 1024UL;
    &heap Complex x = cplx_new(n, 0UL);
    &heap Complex ref = cplx_new(n, 0UL);
    &heap Complex got = cplx_new(n, 0UL);
    std::fft(cs(ref, n));
    check(std::cuda_fft(cs(got, n)), "cuda_fft ran");
    check(cplx_err(got, ref, n) < 1e-12, "cuda_fft values");
    check(std::cuda_ifft(cs(got, n)), "cuda_ifft ran");
    check(cplx_err(got, x, n) < 1e-12, "cuda_ifft round trip");

    // non-power-of-two length against dft.h
    unsigned long m = 1000UL;
    &heap Complex y = cplx_new(m, 5UL);
    &heap Complex yref = cplx_new(m, 0UL);
    std::dft(cs(y, m), cs(yref, m));
    check(std::cuda_fft(cs(y, m)), "cuda_fft n=1000 ran");
    check(cplx_err(y, yref, m) < 1e-10, "cuda_fft n=1000 values");

    // batch of 4 x 256 against per-row fft
    unsigned long bn = 256UL;
    &heap Complex bx = cplx_new(4UL * bn, 9UL);
    &heap Complex bref = cplx_new(4UL * bn, 9UL);
    unsigned long r = 0UL;
    unsafe { while (r < 4UL) { std::fft(cs(bref, 4UL * bn)[r * bn..r * bn + bn]); r = r + 1UL; } }
    check(std::cuda_fft_batch(cs(bx, 4UL * bn), bn, 0), "cuda_fft_batch ran");
    check(cplx_err(bx, bref, 4UL * bn) < 1e-12, "cuda_fft_batch values");

    // 2D against imaging.h
    unsigned long h = 32UL;
    unsigned long w = 64UL;
    &heap Complex img = cplx_new(h * w, 11UL);
    &heap Complex iref = cplx_new(h * w, 11UL);
    std::fft2d(cs(iref, h * w), h, w);
    check(std::cuda_fft2d(cs(img, h * w), h, w), "cuda_fft2d ran");
    check(cplx_err(img, iref, h * w) < 1e-12, "cuda_fft2d values");
    check(std::cuda_ifft2d(cs(img, h * w), h, w), "cuda_ifft2d ran");
    &heap Complex iorig = cplx_new(h * w, 11UL);
    check(cplx_err(img, iorig, h * w) < 1e-12, "cuda_ifft2d round trip");

    // device fp32: c2c batch of 3 x 1024, r2c, c2r
    unsigned long fb = 3UL;
    &heap Complex fx = cplx_new(fb * n, 21UL);
    &heap Complex fref = cplx_new(fb * n, 21UL);
    r = 0UL;
    unsafe { while (r < fb) { std::fft(cs(fref, fb * n)[r * n..r * n + n]); r = r + 1UL; } }
    struct Vec<float> hf = std::Vec::with_cap(2UL * fb * n);
    unsigned long i = 0UL;
    unsafe { while (i < fb * n) { hf.push((float)fx[i].re); hf.push((float)fx[i].im); i = i + 1UL; } }
    struct CudaBuf din = CudaBuf::upload(hf.at(0UL), 2UL * fb * n);
    struct CudaBuf dfreq = CudaBuf::alloc(2UL * fb * n);
    check(std::cuda_fft_c2c(din, dfreq, n, fb, 0), "cuda_fft_c2c ran");
    struct Vec<float> gf = download(dfreq);
    &heap Complex gc = cplx_new(fb * n, 0UL);
    i = 0UL;
    unsafe { while (i < fb * n) { gc[i].re = (double)*gf.at(2UL * i); gc[i].im = (double)*gf.at(2UL * i + 1UL); i = i + 1UL; } }
    check(cplx_err(gc, fref, fb * n) < 1e-5, "cuda_fft_c2c values");
    gf.free();
    check(std::cuda_fft_c2c(dfreq, dfreq, n, fb, 1), "cuda_fft_c2c inverse in place");
    gf = download(dfreq);
    check(maxdiff(&gf, &hf) < 1e-5f, "cuda_fft_c2c round trip");
    gf.free();

    // r2c on the real parts of row 0, compared with fft_real
    struct Vec<float> re = std::Vec::with_cap(n);
    ?&heap double redq = std::alloc(n * sizeof(double));
    double* red;
    unsafe { red = (double*)redq; }
    i = 0UL;
    unsafe { while (i < n) { re.push((float)fx[i].re); red[i] = (double)(float)fx[i].re; i = i + 1UL; } }
    &heap Complex rref = cplx_new(n, 0UL);
    unsafe { std::fft_real(red[0..n], cs(rref, n)); }
    struct CudaBuf dre = CudaBuf::upload(re.at(0UL), n);
    struct CudaBuf dhalf = CudaBuf::alloc(2UL * (n / 2UL + 1UL));
    check(std::cuda_fft_r2c(dre, dhalf, n, 1UL), "cuda_fft_r2c ran");
    gf = download(dhalf);
    i = 0UL;
    unsafe { while (i <= n / 2UL) { gc[i].re = (double)*gf.at(2UL * i); gc[i].im = (double)*gf.at(2UL * i + 1UL); i = i + 1UL; } }
    check(cplx_err(gc, rref, n / 2UL + 1UL) < 1e-5, "cuda_fft_r2c values");
    gf.free();
    struct CudaBuf dback = CudaBuf::alloc(n);
    check(std::cuda_fft_c2r(dhalf, dback, n, 1UL), "cuda_fft_c2r ran");
    gf = download(dback);
    check(maxdiff(&gf, &re) < 1e-5f, "cuda_fft_c2r round trip");
    gf.free();

    din.free(); dfreq.free(); dre.free(); dhalf.free(); dback.free();
    hf.free(); re.free();
    unsafe {
        std::dealloc(x); std::dealloc(ref); std::dealloc(got);
        std::dealloc(y); std::dealloc(yref); std::dealloc(bx); std::dealloc(bref);
        std::dealloc(img); std::dealloc(iref); std::dealloc(iorig);
        std::dealloc(fx); std::dealloc(fref); std::dealloc(gc);
        std::dealloc(redq); std::dealloc(rref);
    }
}

// Mixed precision: fp16 <-> bf16, many-buffer conversion and unscaling with
// the overflow flag, master-weight SGD / Adam (against the CPU arithmetic,
// skipped when flagged), 16-bit elementwise ops and bias.
static struct Vec<unsigned short> read_half(struct CudaHalfBuf h) {
    struct Vec<unsigned short> v = std::Vec::with_cap(h.len);
    unsigned long i = 0UL;
    while (i < h.len) { v.push((unsigned short)0); i = i + 1UL; }
    std::cuda_half_read(h, v.at(0UL), h.len);
    return v;
}

static int half_nan_(unsigned short h, int d) {
    unsigned int u = (unsigned int)h & 0x7FFFU;
    return d == CUDA_BF16 ? u > 0x7F80U : u > 0x7C00U;
}

static void check_mixed_precision() {
    // fp16 <-> bf16 over every 16-bit pattern, against float16.h's bulk conversions
    unsigned long n = 65536UL;
    struct Vec<unsigned short> pat = std::Vec::with_cap(n);
    struct Vec<unsigned short> cpu = std::Vec::with_cap(n);
    unsigned long i = 0UL;
    while (i < n) { pat.push((unsigned short)i); cpu.push((unsigned short)0); i = i + 1UL; }
    int d = CUDA_FP16;
    while (d <= CUDA_BF16) {
        int to = d == CUDA_FP16 ? CUDA_BF16 : CUDA_FP16;
        struct CudaHalfBuf hs = CudaHalfBuf::alloc(n, d);
        struct CudaHalfBuf hd = CudaHalfBuf::alloc(n, to);
        check(std::cuda_half_write(hs, pat.at(0UL), n) && std::cuda_half_convert(hs, hd), "half convert ran");
        struct Vec<unsigned short> got = read_half(hd);
        if (d == CUDA_FP16) std::fp16_to_bf16_bulk(pat.at(0UL), cpu.at(0UL), n);
        else std::bf16_to_fp16_bulk(pat.at(0UL), cpu.at(0UL), n);
        unsigned long bad = 0UL;
        i = 0UL;
        while (i < n) {
            if (half_nan_(*pat.at(i), d)) { if (!half_nan_(*got.at(i), to)) bad = bad + 1UL; }
            else if (*got.at(i) != *cpu.at(i)) bad = bad + 1UL;
            i = i + 1UL;
        }
        check(bad == 0UL, d == CUDA_FP16 ? "fp16 -> bf16 == CPU" : "bf16 -> fp16 == CPU");
        got.free();
        hs.free();
        hd.free();
        d = d + 1;
    }

    // 40 buffers (two launches) to bf16 and back, lengths 1 .. 2000
    struct Vec<struct CudaBuf> f32s = std::Vec::new();
    struct Vec<struct CudaHalfBuf> h16s = std::Vec::new();
    struct Vec<struct CudaBuf> backs = std::Vec::new();
    struct Vec<float> host = seq(2000UL, (float)0.37, -(float)17.0);
    unsigned long b = 0UL;
    while (b < 40UL) {
        unsigned long len = 1UL + (b * 977UL) % 2000UL;
        f32s.push(CudaBuf::upload(host.at(0UL), len));
        h16s.push(CudaHalfBuf::alloc(len, CUDA_BF16));
        backs.push(CudaBuf::alloc(len));
        b = b + 1UL;
    }
    check(std::cuda_to_half_many(&f32s, &h16s) && std::cuda_from_half_many(&h16s, &backs), "many conversions ran");
    unsigned long bad = 0UL;
    b = 0UL;
    while (b < 40UL) {
        struct CudaBuf back = *backs.at(b);
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
    struct CudaHalfBuf odd = CudaHalfBuf::alloc(5UL, CUDA_FP16);
    struct Vec<struct CudaHalfBuf> mixed = std::Vec::new();
    mixed.push(*h16s.at(0UL));
    mixed.push(odd);
    struct Vec<struct CudaBuf> two = std::Vec::new();
    two.push(*backs.at(0UL));
    two.push(*backs.at(1UL));
    check(!std::cuda_from_half_many(&mixed, &two), "mixed dtypes refused");

    // unscaling: fp16 gradients holding 512 x the value; then one overflow
    struct CudaBuf flag = CudaBuf::alloc(1UL);
    struct Vec<float> gv = seq(3000UL, (float)0.01, -(float)0.5);
    struct Vec<float> scaled = std::Vec::with_cap(3000UL);
    i = 0UL;
    while (i < 3000UL) { scaled.push(*gv.at(i) * (float)512.0); i = i + 1UL; }
    struct CudaHalfBuf g16a = CudaHalfBuf::upload(scaled.at(0UL), 3000UL, CUDA_FP16);
    struct CudaHalfBuf g16b = CudaHalfBuf::upload(scaled.at(0UL), 1000UL, CUDA_FP16);
    struct CudaBuf g32a = CudaBuf::alloc(3000UL);
    struct CudaBuf g32b = CudaBuf::alloc(1000UL);
    struct Vec<struct CudaHalfBuf> gs = std::Vec::new();
    gs.push(g16a); gs.push(g16b);
    struct Vec<struct CudaBuf> g32s = std::Vec::new();
    g32s.push(g32a); g32s.push(g32b);
    check(std::cuda_zero(flag) && std::cuda_half_unscale_many(&gs, &g32s, (float)1.0 / (float)512.0, flag), "unscale ran");
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
    std::cuda_half_write(g16b, &inf, 1UL);
    check(std::cuda_half_unscale_many(&gs, &g32s, (float)1.0 / (float)512.0, flag), "unscale ran again");
    fl = download(flag);
    check(*fl.at(0UL) == (float)1.0, "overflow flagged");
    fl.free();
    // fp32 gradients in place
    struct Vec<struct CudaBuf> one = std::Vec::new();
    one.push(g32a);
    check(std::cuda_zero(flag) && std::cuda_unscale_many(&one, &one, (float)4.0, flag), "fp32 unscale in place");
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
        struct CudaBuf dw = CudaBuf::upload(w0.at(0UL), m);
        struct CudaBuf dg = CudaBuf::upload(g0.at(0UL), m);
        struct CudaBuf dm = CudaBuf::alloc(m);
        struct CudaBuf dvv = CudaBuf::alloc(m);
        struct CudaHalfBuf wh = CudaHalfBuf::alloc(m, CUDA_BF16);
        std::cuda_zero(dm); std::cuda_zero(dvv); std::cuda_zero(flag);
        struct Vec<float> cw = std::Vec::with_cap(m);
        struct Vec<float> cm = std::Vec::with_cap(m);
        struct Vec<float> cv = std::Vec::with_cap(m);
        i = 0UL;
        while (i < m) { cw.push(*w0.at(i)); cm.push((float)0.0); cv.push((float)0.0); i = i + 1UL; }
        float lr = (float)0.01; float b1 = (float)0.9; float b2 = (float)0.999; float eps = (float)1.0e-8; float wd = (float)0.05; float mu = (float)0.9;
        unsigned long t = 1UL;
        while (t <= 3UL) {
            if (variant < 2)
                check(std::cuda_mp_adam(dw, dm, dvv, dg, wh, lr, b1, b2, eps, wd, variant, t, flag), "mp adam ran");
            else
                check(std::cuda_mp_sgd(dw, dg, dm, wh, lr, mu, wd, flag), "mp sgd ran");
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
        std::cuda_write(flag, &one1, 1UL);
        if (variant < 2) std::cuda_mp_adam(dw, dm, dvv, dg, wh, lr, b1, b2, eps, wd, variant, 4UL, flag);
        else std::cuda_mp_sgd(dw, dg, dm, wh, lr, mu, wd, flag);
        struct Vec<float> gw2 = download(dw);
        check(maxdiff(&gw, &gw2) == (float)0.0, "flagged step skipped");
        gw.free(); gw2.free(); gh.free(); cw.free(); cm.free(); cv.free();
        dw.free(); dg.free(); dm.free(); dvv.free();
        wh.free();
        variant = variant + 1;
    }

    // cuda_mp_adam_many == cuda_mp_adam per tensor, bit for bit: 30 tensors
    // (two launches), odd and tiny lengths, an unaligned view, with and
    // without 16-bit copies, and a flagged step.
    unsigned long nt = 30UL;
    int pass = 0;
    while (pass < 2) {
        struct Vec<struct CudaBuf> aw = std::Vec::new();
        struct Vec<struct CudaBuf> am = std::Vec::new();
        struct Vec<struct CudaBuf> av = std::Vec::new();
        struct Vec<struct CudaBuf> ag = std::Vec::new();
        struct Vec<struct CudaHalfBuf> ah = std::Vec::new();
        struct Vec<struct CudaBuf> bw = std::Vec::new();
        struct Vec<struct CudaBuf> bm = std::Vec::new();
        struct Vec<struct CudaBuf> bv = std::Vec::new();
        struct Vec<struct CudaHalfBuf> bh = std::Vec::new();
        struct Vec<struct CudaBuf> keep = std::Vec::new();   // allocations behind views
        unsigned long k = 0UL;
        while (k < nt) {
            unsigned long n = 1UL + (k * 397UL) % 5003UL;
            struct Vec<float> wv = seq(n + 1UL, (float)0.003, -(float)0.2);
            struct Vec<float> gv = seq(n + 1UL, (float)0.017, -(float)0.5);
            struct CudaBuf w1 = CudaBuf::upload(wv.at(0UL), n + 1UL);
            struct CudaBuf w2 = CudaBuf::upload(wv.at(0UL), n + 1UL);
            struct CudaBuf gg = CudaBuf::upload(gv.at(0UL), n + 1UL);
            keep.push(w1); keep.push(w2); keep.push(gg);
            struct CudaBuf vw1 = w1; struct CudaBuf vw2 = w2; struct CudaBuf vg = gg;
            if (k == 7UL) { vw1.dev = vw1.dev + 4UL; vw2.dev = vw2.dev + 4UL; vg.dev = vg.dev + 4UL; }   // unaligned
            vw1.len = n; vw2.len = n; vg.len = n;
            struct CudaBuf m1 = CudaBuf::alloc(n); struct CudaBuf v1 = CudaBuf::alloc(n);
            struct CudaBuf m2 = CudaBuf::alloc(n); struct CudaBuf v2 = CudaBuf::alloc(n);
            std::cuda_zero(m1); std::cuda_zero(v1); std::cuda_zero(m2); std::cuda_zero(v2);
            aw.push(vw1); am.push(m1); av.push(v1); ag.push(vg);
            bw.push(vw2); bm.push(m2); bv.push(v2);
            if (pass == 0) { ah.push(CudaHalfBuf::alloc(n, CUDA_BF16)); bh.push(CudaHalfBuf::alloc(n, CUDA_BF16)); }
            wv.free(); gv.free();
            k = k + 1UL;
        }
        std::cuda_zero(flag);
        struct CudaHalfBuf noHalf;
        noHalf.dev = 0UL; noHalf.len = 0UL; noHalf.dtype = CUDA_BF16;
        int okAll = 1;
        unsigned long t = 1UL;
        while (t <= 2UL) {
            k = 0UL;
            while (k < nt) {
                okAll = okAll && std::cuda_mp_adam(*aw.at(k), *am.at(k), *av.at(k), *ag.at(k), pass == 0 ? *ah.at(k) : noHalf,
                                                   (float)0.01, (float)0.9, (float)0.999, (float)1.0e-8, (float)0.05, 1, t, flag);
                k = k + 1UL;
            }
            okAll = okAll && std::cuda_mp_adam_many(&bw, &bm, &bv, &ag, &bh, (float)0.01, (float)0.9, (float)0.999,
                                                    (float)1.0e-8, (float)0.05, 1, t, flag);
            t = t + 1UL;
        }
        // a flagged step changes nothing
        float one2 = (float)1.0;
        std::cuda_write(flag, &one2, 1UL);
        okAll = okAll && std::cuda_mp_adam_many(&bw, &bm, &bv, &ag, &bh, (float)0.01, (float)0.9, (float)0.999,
                                                (float)1.0e-8, (float)0.05, 1, 3UL, flag);
        unsigned long diff = 0UL;
        k = 0UL;
        while (k < nt) {
            struct Vec<float> x1 = download(*aw.at(k));
            struct Vec<float> x2 = download(*bw.at(k));
            struct Vec<float> y1 = download(*av.at(k));
            struct Vec<float> y2 = download(*bv.at(k));
            if (maxdiff(&x1, &x2) != (float)0.0 || maxdiff(&y1, &y2) != (float)0.0) diff = diff + 1UL;
            if (pass == 0) {
                struct Vec<unsigned short> h1 = read_half(*ah.at(k));
                struct Vec<unsigned short> h2 = read_half(*bh.at(k));
                unsigned long q = 0UL;
                while (q < h1.length()) { if (*h1.at(q) != *h2.at(q)) { diff = diff + 1UL; q = h1.length(); } q = q + 1UL; }
                h1.free(); h2.free();
                ah.at(k).free(); bh.at(k).free();
            }
            x1.free(); x2.free(); y1.free(); y2.free();
            am.at(k).free(); av.at(k).free(); bm.at(k).free(); bv.at(k).free();
            k = k + 1UL;
        }
        check(okAll && diff == 0UL, pass == 0 ? "mp adam many == per tensor (bf16 copies, flagged step skipped)"
                                              : "mp adam many == per tensor (no 16-bit copies)");
        k = 0UL;
        while (k < keep.length()) { keep.at(k).free(); k = k + 1UL; }
        aw.free(); am.free(); av.free(); ag.free(); ah.free(); bw.free(); bm.free(); bv.free(); bh.free(); keep.free();
        pass = pass + 1;
    }
    // mismatched lists are refused
    struct Vec<struct CudaBuf> e1 = std::Vec::new();
    struct Vec<struct CudaBuf> e2 = std::Vec::new();
    struct Vec<struct CudaHalfBuf> e3 = std::Vec::new();
    struct CudaBuf lone = CudaBuf::alloc(8UL);
    e1.push(lone);
    check(!std::cuda_mp_adam_many(&e1, &e2, &e2, &e2, &e3, (float)0.01, (float)0.9, (float)0.999, (float)1.0e-8,
                                  (float)0.0, 1, 1UL, flag), "mp adam many rejects mismatched lists");
    lone.free();
    e1.free(); e2.free(); e3.free();

    // 16-bit elementwise and bias, against CPU arithmetic on the decoded values
    unsigned long e = 4099UL;
    struct Vec<float> xa = seq(e, (float)0.13, -(float)6.0);
    struct Vec<float> xb = seq(e, -(float)0.07, (float)3.0);
    d = CUDA_FP16;
    while (d <= CUDA_BF16) {
        struct CudaHalfBuf ha = CudaHalfBuf::upload(xa.at(0UL), e, d);
        struct CudaHalfBuf hb = CudaHalfBuf::upload(xb.at(0UL), e, d);
        struct CudaHalfBuf ho = CudaHalfBuf::alloc(e, d);
        struct Vec<unsigned short> ab = read_half(ha);
        struct Vec<unsigned short> bb = read_half(hb);
        int op = 0;
        while (op < 5) {
            int ok = 0;
            if (op == 0) ok = std::cuda_half_add(ha, hb, ho);
            if (op == 1) ok = std::cuda_half_sub(ha, hb, ho);
            if (op == 2) ok = std::cuda_half_mul(ha, hb, ho);
            if (op == 3) ok = std::cuda_half_scale(ha, (float)0.3, ho);
            if (op == 4) ok = std::cuda_half_axpy(ha, -(float)1.7, hb, ho);
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
        struct CudaHalfBuf hbias = CudaHalfBuf::upload(bias.at(0UL), 7UL, d);
        struct Vec<unsigned short> biasb = read_half(hbias);
        check(std::cuda_half_bias(ha, hbias, 3UL, 7UL, 25UL, 1), "half bias ran");
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


// 16-bit cuDNN convolution and pooling against cnn.sc on the same rounded
// inputs (fp32 accumulation, so within a couple of 16-bit roundings).
static struct Conv2D conv_rounded_(const &Conv2D l, int d) {
    struct Conv2D r = Conv2D::new(l.inChannels, l.outChannels, l.kH, l.kW, l.stride, l.padding);
    unsigned long wn = l.outChannels * l.inChannels * l.kH * l.kW;
    unsafe {
        unsigned long i = 0UL;
        while (i < wn) { r.weight[i] = std::half_round(l.weight[i], d); i = i + 1UL; }
        i = 0UL;
        while (i < l.outChannels) { r.bias[i] = std::half_round(l.bias[i], d); i = i + 1UL; }
    }
    return r;
}

static void fm_round_(&FeatureMap fm, int d) {
    unsigned long n = fm_size(fm);
    unsafe { unsigned long i = 0UL; while (i < n) { fm.data[i] = std::half_round(fm.data[i], d); i = i + 1UL; } }
}

static struct Vec<float> half_download_(struct CudaHalfBuf h, unsigned long off, unsigned long n) {
    struct Vec<unsigned short> bits = std::Vec::with_cap(h.len);
    unsigned long i = 0UL;
    while (i < h.len) { bits.push((unsigned short)0); i = i + 1UL; }
    std::cuda_half_read(h, bits.at(0UL), h.len);
    struct Vec<float> v = std::Vec::with_cap(n);
    i = 0UL;
    while (i < n) { v.push(std::half_decode(*bits.at(off + i), h.dtype)); i = i + 1UL; }
    bits.free();
    return v;
}

static void check_cudnn_half() {
    int d = CUDA_FP16;
    while (d <= CUDA_BF16) {
        float tol = d == CUDA_FP16 ? (float)3.0e-3 : (float)2.0e-2;
        // 4 -> 6 channels, 5x5, stride 2, two images; 8 -> 8, 3x3, stride 1, pad 1;
        // 8 -> 5, 1x1 (the strided-batched GEMM path)
        int shape = 0;
        while (shape < 3) {
            unsigned long inC = shape == 0 ? 4UL : 8UL;
            unsigned long outC = shape == 0 ? 6UL : (shape == 1 ? 8UL : 5UL);
            unsigned long k = shape == 0 ? 5UL : (shape == 1 ? 3UL : 1UL);
            unsigned long st = shape == 0 ? 2UL : 1UL;
            unsigned long pd = shape == 1 ? 1UL : 0UL;
            struct Conv2D l = Conv2D::new(inC, outC, k, k, st, pd);
            conv_bias(&l);
            struct Conv2D lr = conv_rounded_(&l, d);
            struct FeatureMap xa = FeatureMap::new(inC, 21UL, 15UL);
            struct FeatureMap xb = FeatureMap::new(inC, 21UL, 15UL);
            fm_fill(&xa, 3UL);
            fm_fill(&xb, 50UL);
            fm_round_(&xa, d);
            fm_round_(&xb, d);
            struct FeatureMap ca = lr.forward(&xa);
            struct FeatureMap cb = lr.forward(&xb);
            unsigned long inN = fm_size(&xa);
            unsigned long outN = fm_size(&ca);
            struct Vec<float> both = std::Vec::with_cap(2UL * inN);
            unsafe {
                unsigned long i = 0UL;
                while (i < inN) { both.push(xa.data[i]); i = i + 1UL; }
                i = 0UL;
                while (i < inN) { both.push(xb.data[i]); i = i + 1UL; }
            }
            struct CudaHalfConv2D hl = CudaHalfConv2D::upload(&l, d);
            struct CudaHalfBuf hx = CudaHalfBuf::upload(both.at(0UL), 2UL * inN, d);
            struct CudaHalfBuf hy = CudaHalfBuf::alloc(2UL * outN, d);
            check(hl.weight.dev != 0UL && hx.dev != 0UL && hy.dev != 0UL, "half conv upload");
            check(hl.forward(hx, hy, 2UL, 21UL, 15UL), d == CUDA_FP16 ? "fp16 conv ran" : "bf16 conv ran");
            struct Vec<float> ya = half_download_(hy, 0UL, outN);
            struct Vec<float> yb = half_download_(hy, outN, outN);
            unsafe {
                check(fm_err(ya.at(0UL), (const float*)ca.data, outN) < tol && fm_err(yb.at(0UL), (const float*)cb.data, outN) < tol,
                      d == CUDA_FP16 ? "fp16 conv values" : "bf16 conv values");
            }
            ya.free(); yb.free();
            check(hl.forward_relu(hx, hy, 2UL, 21UL, 15UL), "half conv relu ran");
            yb = half_download_(hy, outN, outN);
            unsafe {
                unsigned long i = 0UL;
                while (i < outN) { if (cb.data[i] < (float)0.0) cb.data[i] = (float)0.0; i = i + 1UL; }
                check(fm_err(yb.at(0UL), (const float*)cb.data, outN) < tol, "half conv relu values");
            }
            yb.free();
            if (shape == 0) {
                // pooling: max is exact on rounded values, avg within one rounding
                struct FeatureMap mp = xa.maxpool2d(3UL, 2UL);
                struct FeatureMap ap = xa.avgpool2d(3UL, 2UL);
                unsigned long pn = fm_size(&mp);
                struct CudaHalfBuf hp = CudaHalfBuf::alloc(2UL * pn, d);
                check(std::cuda_half_maxpool2d(hx, hp, 2UL, inC, 21UL, 15UL, 3UL, 2UL), "half maxpool ran");
                struct Vec<float> pa = half_download_(hp, 0UL, pn);
                unsafe { check(fm_err(pa.at(0UL), (const float*)mp.data, pn) == (float)0.0, "half maxpool values"); }
                pa.free();
                check(std::cuda_half_avgpool2d(hx, hp, 2UL, inC, 21UL, 15UL, 3UL, 2UL), "half avgpool ran");
                pa = half_download_(hp, 0UL, pn);
                unsafe { check(fm_err(pa.at(0UL), (const float*)ap.data, pn) < tol, "half avgpool values"); }
                pa.free();
                struct CudaHalfBuf other = CudaHalfBuf::alloc(2UL * pn, d == CUDA_FP16 ? CUDA_BF16 : CUDA_FP16);
                check(!std::cuda_half_maxpool2d(hx, other, 2UL, inC, 21UL, 15UL, 3UL, 2UL), "half pool rejects mixed dtypes");
                other.free();
                hp.free();
                mp.free(); ap.free();
            }
            hx.free(); hy.free();
            hl.free();
            both.free();
            ca.free(); cb.free(); xa.free(); xb.free();
            l.free(); lr.free();
            shape = shape + 1;
        }
        d = d + 1;
    }
}


int main() {
    if (!std::cuda_available()) { unsafe { printf("no CUDA GPU -- skipped\n"); } return 0; }
    unsafe { printf("device: %s\n", std::cuda_device_name()); }

    unsigned long n = 10007UL;   // not a multiple of the block size
    struct Vec<float> a = seq(n, 0.01f, 0.1f);
    struct Vec<float> b = seq(n, 0.02f, 0.5f);
    struct CudaBuf da = CudaBuf::upload(a.at(0UL), n);
    struct CudaBuf db = CudaBuf::upload(b.at(0UL), n);
    struct CudaBuf dout = CudaBuf::alloc(n);
    check(da.dev != 0UL && db.dev != 0UL && dout.dev != 0UL, "upload/alloc");

    // elementwise vs CPU
    struct Vec<float> want = std::Vec::with_cap(n);
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
    struct CudaBuf d1 = CudaBuf::alloc(1UL);
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
    struct CudaBuf small = CudaBuf::alloc(10UL);
    check(!std::cuda_add(small, db, dout), "length mismatch rejected");

    // gemm, every transpose combination, odd sizes
    unsigned long m = 67UL;
    unsigned long nn = 45UL;
    unsigned long k = 129UL;
    struct Vec<float> ga = seq(m * k, 0.003f, -0.1f);
    struct Vec<float> gb = seq(k * nn, 0.002f, -0.07f);
    struct CudaBuf dga = CudaBuf::upload(ga.at(0UL), m * k);
    struct CudaBuf dgb = CudaBuf::upload(gb.at(0UL), k * nn);
    struct CudaBuf dgc = CudaBuf::alloc(m * nn);
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
    struct Vec<float> hc = std::Vec::with_cap(m * nn);
    i = 0UL;
    while (i < m * nn) { hc.push(0.0f); i = i + 1UL; }
    unsafe {
        check(std::cuda_matmul_f32(ga.at(0UL), gb.at(0UL), hc.at(0UL), m, k, nn), "custom matmul ran");
    }
    check(maxdiff(&hc, &ref) < 1e-4f, "custom matmul values");
    unsafe {
        check(std::cuda_matmul_f32_blas(ga.at(0UL), gb.at(0UL), hc.at(0UL), m, k, nn), "blas matmul ran");
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
    struct CudaBuf dg = CudaBuf::alloc(n);
    std::cuda_relu_grad(dout, da, dg);
    got = download(dg);
    i = 0UL;
    while (i < n) { float z = *a.at(i) - *b.at(i); *want.at(i) = 0.0f; if (z > 0.0f) *want.at(i) = *a.at(i); i = i + 1UL; }
    check(maxdiff(&got, &want) == 0.0f, "relu_grad values");
    got.free();
    struct CudaBuf w1 = CudaBuf::upload(a.at(0UL), n);
    struct CudaBuf w2 = CudaBuf::upload(ga.at(0UL), m * k);
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
    struct Vec<float> hv = std::Vec::with_cap(n);
    i = 0UL;
    while (i < n) { hv.push(0.0f); i = i + 1UL; }
    unsafe {
        check(std::cuda_add_f32(a.at(0UL), b.at(0UL), hv.at(0UL), n), "add_f32 ran");
    }
    i = 0UL;
    while (i < n) { *want.at(i) = *a.at(i) + *b.at(i); i = i + 1UL; }
    check(maxdiff(&hv, &want) < 1e-5f, "add_f32 values");
    float hsum = 0.0f;
    unsafe { check(std::cuda_sum_f32(a.at(0UL), &hsum, n), "sum_f32 ran"); }
    rel = ((double)hsum - cpuSum) / cpuSum;
    if (rel < 0.0) rel = -rel;
    check(rel < 1e-5, "sum_f32 value");
    unsafe {
        // (the raw device-pointer API — cuda_upload_persistent & co. — was
        // removed; CudaBuf / CudaHalfBuf cover it and are checked above)
    }
    dg.free(); w1.free(); w2.free();
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
    g.free();

    da.free(); db.free(); dout.free(); d1.free();
    small.free(); dga.free(); dgb.free(); dgc.free();
    check(da.dev == 0UL, "free clears handle");
    a.free(); b.free(); want.free(); ga.free(); gb.free(); ref.free(); hc.free();

    check_half();
    check_mixed_precision();
    check_decode_ops();
    check_transfers();
    check_cudnn();
    check_cudnn_half();
    check_cufft();

    unsafe { printf("cuda: %d checks, %d failures\n", g_checks, g_fails); }
    return g_fails == 0 ? 0 : 1;
}
