// Correctness check for std/ml/gpu_miopen (MIOpen) and std/dsp/fft_rocm
// (hipFFT) against the CPU implementations in cnn.h, unet.h, fft.h, dft.h
// and imaging.h. Same checks as tests/stdlib/cuda's cuDNN/cuFFT sections.
// Needs an AMD GPU; run with tests/stdlib/rocm/run.sh.
#include <std/ml/gpu_rocm.h>
#include <std/collections/vec.h>
#include <std/math.h>
#include <std/ml/gpu_miopen.h>
#include <std/ml/cnn.h>
#include <std/ml/unet.h>
#include <std/dsp/fft_rocm.h>
#include <std/dsp/fft.h>
#include <std/dsp/dft.h>
#include <std/dsp/imaging.h>
#include <std/mem.h>
#include <std/panic.h>
#include <std/ml/tensor_half.h>
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

static struct Vec<float> download(struct RocmBuf b) {
    struct Vec<float> v = std::vec_with_cap(b.len);
    unsigned long i = 0UL;
    while (i < b.len) { v.push(0.0f); i = i + 1UL; }
    std::rocm_read(b, v.at(0UL), b.len);
    return v;
}

static void fm_fill(&FeatureMap fm, unsigned long salt) {
    unsigned long n = fm.channels * fm.height * fm.width;
    unsigned long i = 0UL;
    unsafe { while (i < n) { fm.data[i] = (float)(((i + salt) * 37UL) % 101UL) * (float)0.02 - (float)1.0; i = i + 1UL; } }
}

static void conv_bias(&Conv2D layer) {
    unsigned long i = 0UL;
    unsafe { while (i < layer.outChannels) { layer.bias[i] = (float)i * (float)0.1 - (float)0.3; i = i + 1UL; } }
}

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

static struct Vec<float> fm_download(struct RocmBuf b, unsigned long off, unsigned long n) {
    struct Vec<float> all = download(b);
    struct Vec<float> v = std::vec_with_cap(n);
    unsigned long i = 0UL;
    while (i < n) { v.push(*all.at(off + i)); i = i + 1UL; }
    all.free();
    return v;
}

static void check_miopen() {
    // conv2d_forward_rocm vs Conv2D::forward: 3x3, stride 1, pad 1, odd sizes
    struct Conv2D l1 = std::conv2d_new(3UL, 8UL, 3UL, 3UL, 1UL, 1UL);
    conv_bias(&l1);
    struct FeatureMap x1 = std::feature_map_new(3UL, 17UL, 13UL);
    fm_fill(&x1, 0UL);
    struct FeatureMap c1 = l1.forward(&x1);
    struct FeatureMap g1 = std::conv2d_forward_rocm(&l1, &x1);
    check(g1.channels == 8UL && g1.height == 17UL && g1.width == 13UL, "conv2d_forward_rocm shape");
    unsafe { check(fm_err((const float*)g1.data, (const float*)c1.data, fm_size(&c1)) < 1e-5f, "conv2d_forward_rocm values"); }
    g1.free();

    // queued conv on a batch of two images: 5x5, stride 2, pad 0; then + relu
    struct Conv2D l2 = std::conv2d_new(4UL, 6UL, 5UL, 5UL, 2UL, 0UL);
    conv_bias(&l2);
    struct FeatureMap xa = std::feature_map_new(4UL, 21UL, 15UL);
    struct FeatureMap xb = std::feature_map_new(4UL, 21UL, 15UL);
    fm_fill(&xa, 3UL);
    fm_fill(&xb, 50UL);
    struct FeatureMap ca = l2.forward(&xa);
    struct FeatureMap cb = l2.forward(&xb);
    unsigned long inN = fm_size(&xa);
    unsigned long outN = fm_size(&ca);
    check(std::rocm_conv_out_dim(21UL, 5UL, 2UL, 0UL) == ca.height && std::rocm_conv_out_dim(15UL, 5UL, 2UL, 0UL) == ca.width,
          "conv out dim");
    struct RocmConv2D d2 = std::rocm_conv2d_upload(&l2);
    check(d2.weight.dev != 0UL && d2.bias.dev != 0UL, "conv2d upload");
    struct RocmBuf dx = std::rocm_alloc(2UL * inN);
    struct RocmBuf dy = std::rocm_alloc(2UL * outN);
    unsafe {
        std::rocm_write(dx, xa.data, inN);
        struct RocmBuf dx2 = dx;
        dx2.dev = dx.dev + inN * 4UL;
        dx2.len = inN;
        std::rocm_write(dx2, xb.data, inN);
    }
    check(d2.forward(dx, dy, 2UL, 21UL, 15UL), "RocmConv2D::forward batch ran");
    struct Vec<float> ya = fm_download(dy, 0UL, outN);
    struct Vec<float> yb = fm_download(dy, outN, outN);
    unsafe {
        check(fm_err(ya.at(0UL), (const float*)ca.data, outN) < 1e-5f &&
              fm_err(yb.at(0UL), (const float*)cb.data, outN) < 1e-5f, "RocmConv2D::forward batch values");
    }
    ya.free(); yb.free();
    check(d2.forward_relu(dx, dy, 2UL, 21UL, 15UL), "RocmConv2D::forward_relu ran");
    ya = fm_download(dy, outN, outN);
    unsafe {
        unsigned long i = 0UL;
        while (i < outN) { if (cb.data[i] < 0.0f) cb.data[i] = 0.0f; i = i + 1UL; }
        check(fm_err(ya.at(0UL), (const float*)cb.data, outN) < 1e-5f, "RocmConv2D::forward_relu values");
    }
    ya.free();
    check(!d2.forward(dx, dy, 3UL, 21UL, 15UL), "RocmConv2D::forward rejects short buffers");

    // the cached plan replays inside a graph
    check(std::rocm_capture_begin(), "conv capture begin");
    d2.forward(dx, dy, 2UL, 21UL, 15UL);
    struct RocmGraph cg = std::rocm_capture_end();
    check(cg.exec != 0UL, "conv capture end");
    std::rocm_scale(dy, 0.0f, dy);
    check(std::rocm_graph_launch(cg), "conv graph launch");
    ya = fm_download(dy, 0UL, outN);
    unsafe { check(fm_err(ya.at(0UL), (const float*)ca.data, outN) < 1e-5f, "conv graph values"); }
    ya.free();
    cg.free();


    // pooling, upsample and concat against cnn.sc
    struct FeatureMap mp = std::maxpool2d_forward(&xa, 3UL, 2UL);
    struct FeatureMap ap = std::avgpool2d_forward(&xa, 3UL, 2UL);
    unsigned long pn = fm_size(&mp);
    struct RocmBuf dp = std::rocm_alloc(2UL * pn);
    check(std::rocm_maxpool2d(dx, dp, 2UL, 4UL, 21UL, 15UL, 3UL, 2UL), "rocm_maxpool2d ran");
    ya = fm_download(dp, 0UL, pn);
    unsafe { check(fm_err(ya.at(0UL), (const float*)mp.data, pn) == 0.0f, "rocm_maxpool2d values"); }
    ya.free();
    check(std::rocm_avgpool2d(dx, dp, 2UL, 4UL, 21UL, 15UL, 3UL, 2UL), "rocm_avgpool2d ran");
    ya = fm_download(dp, 0UL, pn);
    unsafe { check(fm_err(ya.at(0UL), (const float*)ap.data, pn) < 1e-6f, "rocm_avgpool2d values"); }
    ya.free();

    struct FeatureMap up = std::upsample2x_nearest(&xa);
    struct RocmBuf du = std::rocm_alloc(4UL * inN);
    check(std::rocm_upsample2x_nearest(dx, du, 4UL, 21UL, 15UL), "rocm_upsample2x ran");
    ya = fm_download(du, 0UL, 4UL * inN);
    unsafe { check(fm_err(ya.at(0UL), (const float*)up.data, 4UL * inN) == 0.0f, "rocm_upsample2x values"); }
    ya.free();

    struct FeatureMap cat = std::concat_channels(&xa, &xb);
    struct RocmBuf dxa = std::rocm_upload(xa.data, inN);
    struct RocmBuf dxb = std::rocm_upload(xb.data, inN);
    struct RocmBuf dc = std::rocm_alloc(2UL * inN);
    check(std::rocm_concat_channels(dxa, 4UL, dxb, 4UL, dc, 1UL, 21UL * 15UL), "rocm_concat_channels ran");
    ya = download(dc);
    unsafe { check(fm_err(ya.at(0UL), (const float*)cat.data, 2UL * inN) == 0.0f, "rocm_concat_channels values"); }
    ya.free();

    // whole U-Net
    struct UNet net = std::unet_new(3UL, 8UL, 16UL, 2UL);
    struct FeatureMap ux = std::feature_map_new(3UL, 32UL, 24UL);
    fm_fill(&ux, 7UL);
    struct FeatureMap uc = net.forward(&ux);
    struct FeatureMap ug = std::unet_forward_rocm(&net, &ux);
    check(ug.channels == 2UL && ug.height == 32UL && ug.width == 24UL, "unet_forward_rocm shape");
    unsafe { check(fm_err((const float*)ug.data, (const float*)uc.data, fm_size(&uc)) < 1e-4f, "unet_forward_rocm values"); }

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
        case null: { std::panic_at("cplx_new: out of memory", "rocm_check.sc", 0); }
    }
    std::panic_at("cplx_new: unreachable", "rocm_check.sc", 0);
}

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

static void check_hipfft() {
    // 1D double, power of two, against fft.h
    unsigned long n = 1024UL;
    &heap Complex x = cplx_new(n, 0UL);
    &heap Complex ref = cplx_new(n, 0UL);
    &heap Complex got = cplx_new(n, 0UL);
    std::fft(ref, n);
    check(std::rocm_fft(got, n), "rocm_fft ran");
    check(cplx_err(got, ref, n) < 1e-12, "rocm_fft values");
    check(std::rocm_ifft(got, n), "rocm_ifft ran");
    check(cplx_err(got, x, n) < 1e-12, "rocm_ifft round trip");

    // non-power-of-two length against dft.h
    unsigned long m = 1000UL;
    &heap Complex y = cplx_new(m, 5UL);
    &heap Complex yref = cplx_new(m, 0UL);
    std::dft(y, yref, m);
    check(std::rocm_fft(y, m), "rocm_fft n=1000 ran");
    check(cplx_err(y, yref, m) < 1e-10, "rocm_fft n=1000 values");

    // batch of 4 x 256 against per-row fft
    unsigned long bn = 256UL;
    &heap Complex bx = cplx_new(4UL * bn, 9UL);
    &heap Complex bref = cplx_new(4UL * bn, 9UL);
    unsigned long r = 0UL;
    unsafe { while (r < 4UL) { std::fft(&bref[r * bn], bn); r = r + 1UL; } }
    check(std::rocm_fft_batch(bx, bn, 4UL, 0), "rocm_fft_batch ran");
    check(cplx_err(bx, bref, 4UL * bn) < 1e-12, "rocm_fft_batch values");

    // 2D against imaging.h
    unsigned long h = 32UL;
    unsigned long w = 64UL;
    &heap Complex img = cplx_new(h * w, 11UL);
    &heap Complex iref = cplx_new(h * w, 11UL);
    std::fft2d(iref, h, w);
    check(std::rocm_fft2d(img, h, w), "rocm_fft2d ran");
    check(cplx_err(img, iref, h * w) < 1e-12, "rocm_fft2d values");
    check(std::rocm_ifft2d(img, h, w), "rocm_ifft2d ran");
    &heap Complex iorig = cplx_new(h * w, 11UL);
    check(cplx_err(img, iorig, h * w) < 1e-12, "rocm_ifft2d round trip");

    // device fp32: c2c batch of 3 x 1024, r2c, c2r
    unsigned long fb = 3UL;
    &heap Complex fx = cplx_new(fb * n, 21UL);
    &heap Complex fref = cplx_new(fb * n, 21UL);
    r = 0UL;
    unsafe { while (r < fb) { std::fft(&fref[r * n], n); r = r + 1UL; } }
    struct Vec<float> hf = std::vec_with_cap(2UL * fb * n);
    unsigned long i = 0UL;
    unsafe { while (i < fb * n) { hf.push((float)fx[i].re); hf.push((float)fx[i].im); i = i + 1UL; } }
    struct RocmBuf din = std::rocm_upload(hf.at(0UL), 2UL * fb * n);
    struct RocmBuf dfreq = std::rocm_alloc(2UL * fb * n);
    check(std::rocm_fft_c2c(din, dfreq, n, fb, 0), "rocm_fft_c2c ran");
    struct Vec<float> gf = download(dfreq);
    &heap Complex gc = cplx_new(fb * n, 0UL);
    i = 0UL;
    unsafe { while (i < fb * n) { gc[i].re = (double)*gf.at(2UL * i); gc[i].im = (double)*gf.at(2UL * i + 1UL); i = i + 1UL; } }
    check(cplx_err(gc, fref, fb * n) < 1e-5, "rocm_fft_c2c values");
    gf.free();
    check(std::rocm_fft_c2c(dfreq, dfreq, n, fb, 1), "rocm_fft_c2c inverse in place");
    gf = download(dfreq);
    check(maxdiff(&gf, &hf) < 1e-5f, "rocm_fft_c2c round trip");
    gf.free();

    // r2c on the real parts of row 0, compared with fft_real
    struct Vec<float> re = std::vec_with_cap(n);
    ?&heap double redq = std::alloc(n * sizeof(double));
    double* red;
    unsafe { red = (double*)redq; }
    i = 0UL;
    unsafe { while (i < n) { re.push((float)fx[i].re); red[i] = (double)(float)fx[i].re; i = i + 1UL; } }
    &heap Complex rref = cplx_new(n, 0UL);
    unsafe { std::fft_real((const double*)red, rref, n); }
    struct RocmBuf dre = std::rocm_upload(re.at(0UL), n);
    struct RocmBuf dhalf = std::rocm_alloc(2UL * (n / 2UL + 1UL));
    check(std::rocm_fft_r2c(dre, dhalf, n, 1UL), "rocm_fft_r2c ran");
    gf = download(dhalf);
    i = 0UL;
    unsafe { while (i <= n / 2UL) { gc[i].re = (double)*gf.at(2UL * i); gc[i].im = (double)*gf.at(2UL * i + 1UL); i = i + 1UL; } }
    check(cplx_err(gc, rref, n / 2UL + 1UL) < 1e-5, "rocm_fft_r2c values");
    gf.free();
    struct RocmBuf dback = std::rocm_alloc(n);
    check(std::rocm_fft_c2r(dhalf, dback, n, 1UL), "rocm_fft_c2r ran");
    gf = download(dback);
    check(maxdiff(&gf, &re) < 1e-5f, "rocm_fft_c2r round trip");
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


// 16-bit MIOpen convolution and pooling against cnn.sc on the same rounded
// inputs (fp32 accumulation, so within a couple of 16-bit roundings).
static struct Conv2D conv_rounded_(const &Conv2D l, int d) {
    struct Conv2D r = std::conv2d_new(l.inChannels, l.outChannels, l.kH, l.kW, l.stride, l.padding);
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

static struct Vec<float> half_download_(struct RocmHalfBuf h, unsigned long off, unsigned long n) {
    struct Vec<unsigned short> bits = std::vec_with_cap(h.len);
    unsigned long i = 0UL;
    while (i < h.len) { bits.push((unsigned short)0); i = i + 1UL; }
    std::rocm_half_read(h, bits.at(0UL), h.len);
    struct Vec<float> v = std::vec_with_cap(n);
    i = 0UL;
    while (i < n) { v.push(std::half_decode(*bits.at(off + i), h.dtype)); i = i + 1UL; }
    bits.free();
    return v;
}

static void check_miopen_half() {
    int d = ROCM_FP16;
    while (d <= ROCM_BF16) {
        float tol = d == ROCM_FP16 ? (float)3.0e-3 : (float)2.0e-2;
        // 4 -> 6 channels, 5x5, stride 2, two images; 8 -> 8, 3x3, stride 1, pad 1
        int shape = 0;
        while (shape < 2) {
            unsigned long inC = shape == 0 ? 4UL : 8UL;
            unsigned long outC = shape == 0 ? 6UL : 8UL;
            unsigned long k = shape == 0 ? 5UL : 3UL;
            unsigned long st = shape == 0 ? 2UL : 1UL;
            unsigned long pd = shape == 0 ? 0UL : 1UL;
            struct Conv2D l = std::conv2d_new(inC, outC, k, k, st, pd);
            conv_bias(&l);
            struct Conv2D lr = conv_rounded_(&l, d);
            struct FeatureMap xa = std::feature_map_new(inC, 21UL, 15UL);
            struct FeatureMap xb = std::feature_map_new(inC, 21UL, 15UL);
            fm_fill(&xa, 3UL);
            fm_fill(&xb, 50UL);
            fm_round_(&xa, d);
            fm_round_(&xb, d);
            struct FeatureMap ca = lr.forward(&xa);
            struct FeatureMap cb = lr.forward(&xb);
            unsigned long inN = fm_size(&xa);
            unsigned long outN = fm_size(&ca);
            struct Vec<float> both = std::vec_with_cap(2UL * inN);
            unsafe {
                unsigned long i = 0UL;
                while (i < inN) { both.push(xa.data[i]); i = i + 1UL; }
                i = 0UL;
                while (i < inN) { both.push(xb.data[i]); i = i + 1UL; }
            }
            struct RocmHalfConv2D hl = std::rocm_half_conv2d_upload(&l, d);
            struct RocmHalfBuf hx = std::rocm_half_upload(both.at(0UL), 2UL * inN, d);
            struct RocmHalfBuf hy = std::rocm_half_alloc(2UL * outN, d);
            check(hl.weight.dev != 0UL && hx.dev != 0UL && hy.dev != 0UL, "half conv upload");
            check(hl.forward(hx, hy, 2UL, 21UL, 15UL), d == ROCM_FP16 ? "fp16 conv ran" : "bf16 conv ran");
            struct Vec<float> ya = half_download_(hy, 0UL, outN);
            struct Vec<float> yb = half_download_(hy, outN, outN);
            unsafe {
                check(fm_err(ya.at(0UL), (const float*)ca.data, outN) < tol && fm_err(yb.at(0UL), (const float*)cb.data, outN) < tol,
                      d == ROCM_FP16 ? "fp16 conv values" : "bf16 conv values");
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
                struct FeatureMap mp = std::maxpool2d_forward(&xa, 3UL, 2UL);
                struct FeatureMap ap = std::avgpool2d_forward(&xa, 3UL, 2UL);
                unsigned long pn = fm_size(&mp);
                struct RocmHalfBuf hp = std::rocm_half_alloc(2UL * pn, d);
                check(std::rocm_half_maxpool2d(hx, hp, 2UL, inC, 21UL, 15UL, 3UL, 2UL), "half maxpool ran");
                struct Vec<float> pa = half_download_(hp, 0UL, pn);
                unsafe { check(fm_err(pa.at(0UL), (const float*)mp.data, pn) == (float)0.0, "half maxpool values"); }
                pa.free();
                check(std::rocm_half_avgpool2d(hx, hp, 2UL, inC, 21UL, 15UL, 3UL, 2UL), "half avgpool ran");
                pa = half_download_(hp, 0UL, pn);
                unsafe { check(fm_err(pa.at(0UL), (const float*)ap.data, pn) < tol, "half avgpool values"); }
                pa.free();
                struct RocmHalfBuf other = std::rocm_half_alloc(2UL * pn, d == ROCM_FP16 ? ROCM_BF16 : ROCM_FP16);
                check(!std::rocm_half_maxpool2d(hx, other, 2UL, inC, 21UL, 15UL, 3UL, 2UL), "half pool rejects mixed dtypes");
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
    if (!std::rocm_available()) { unsafe { printf("no ROCm GPU -- skipping\n"); } return 0; }
    unsafe { printf("device: %s\n", std::rocm_device_name()); }
    check_miopen();
    check_miopen_half();
    check_hipfft();
    unsafe { printf("rocm dnn/fft: %d checks, %d failures\n", g_checks, g_fails); }
    return g_fails == 0 ? 0 : 1;
}
