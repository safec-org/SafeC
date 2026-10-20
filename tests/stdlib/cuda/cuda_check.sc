// Correctness check for std/ml/gpu_cuda against CPU reference results.
// Needs an NVIDIA GPU; run with tests/stdlib/cuda/run.sh.
#include <std/ml/gpu_cuda.h>
#include <std/ml/float16.h>
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
    struct Vec<float> v = std::vec_with_cap(n);
    unsigned long i = 0UL;
    while (i < n) { v.push(*all.at(off + i)); i = i + 1UL; }
    all.free();
    return v;
}

static void check_cudnn() {
    // conv2d_forward_cuda vs conv2d_forward: 3x3, stride 1, pad 1, odd sizes
    struct Conv2D l1 = std::conv2d_new(3UL, 8UL, 3UL, 3UL, 1UL, 1UL);
    conv_bias(&l1);
    struct FeatureMap x1 = std::feature_map_new(3UL, 17UL, 13UL);
    fm_fill(&x1, 0UL);
    struct FeatureMap c1 = std::conv2d_forward(&l1, &x1);
    struct FeatureMap g1 = std::conv2d_forward_cuda(&l1, &x1);
    check(g1.channels == 8UL && g1.height == 17UL && g1.width == 13UL, "conv2d_forward_cuda shape");
    unsafe { check(fm_err((const float*)g1.data, (const float*)c1.data, fm_size(&c1)) < 1e-5f, "conv2d_forward_cuda values"); }
    std::feature_map_free(&g1);

    // queued conv on a batch of two images: 5x5, stride 2, pad 0; then + relu
    struct Conv2D l2 = std::conv2d_new(4UL, 6UL, 5UL, 5UL, 2UL, 0UL);
    conv_bias(&l2);
    struct FeatureMap xa = std::feature_map_new(4UL, 21UL, 15UL);
    struct FeatureMap xb = std::feature_map_new(4UL, 21UL, 15UL);
    fm_fill(&xa, 3UL);
    fm_fill(&xb, 50UL);
    struct FeatureMap ca = std::conv2d_forward(&l2, &xa);
    struct FeatureMap cb = std::conv2d_forward(&l2, &xb);
    unsigned long inN = fm_size(&xa);
    unsigned long outN = fm_size(&ca);
    check(std::cuda_conv_out_dim(21UL, 5UL, 2UL, 0UL) == ca.height && std::cuda_conv_out_dim(15UL, 5UL, 2UL, 0UL) == ca.width,
          "conv out dim");
    struct CudaConv2D d2 = std::cuda_conv2d_upload(&l2);
    check(d2.weight.dev != 0UL && d2.bias.dev != 0UL, "conv2d upload");
    struct CudaBuf dx = std::cuda_alloc(2UL * inN);
    struct CudaBuf dy = std::cuda_alloc(2UL * outN);
    unsafe {
        std::cuda_write(dx, xa.data, inN);
        struct CudaBuf dx2 = dx;
        dx2.dev = dx.dev + inN * 4UL;
        dx2.len = inN;
        std::cuda_write(dx2, xb.data, inN);
    }
    check(std::cuda_conv2d(&d2, dx, dy, 2UL, 21UL, 15UL), "cuda_conv2d batch ran");
    struct Vec<float> ya = fm_download(dy, 0UL, outN);
    struct Vec<float> yb = fm_download(dy, outN, outN);
    unsafe {
        check(fm_err((const float*)ya.at(0UL), (const float*)ca.data, outN) < 1e-5f &&
              fm_err((const float*)yb.at(0UL), (const float*)cb.data, outN) < 1e-5f, "cuda_conv2d batch values");
    }
    ya.free(); yb.free();
    check(std::cuda_conv2d_relu(&d2, dx, dy, 2UL, 21UL, 15UL), "cuda_conv2d_relu ran");
    ya = fm_download(dy, outN, outN);
    unsafe {
        unsigned long i = 0UL;
        while (i < outN) { if (cb.data[i] < 0.0f) cb.data[i] = 0.0f; i = i + 1UL; }
        check(fm_err((const float*)ya.at(0UL), (const float*)cb.data, outN) < 1e-5f, "cuda_conv2d_relu values");
    }
    ya.free();
    check(!std::cuda_conv2d(&d2, dx, dy, 3UL, 21UL, 15UL), "cuda_conv2d rejects short buffers");

    // the cached plan replays inside a graph
    check(std::cuda_capture_begin(), "conv capture begin");
    std::cuda_conv2d(&d2, dx, dy, 2UL, 21UL, 15UL);
    struct CudaGraph cg = std::cuda_capture_end();
    check(cg.exec != 0UL, "conv capture end");
    std::cuda_scale(dy, 0.0f, dy);
    check(std::cuda_graph_launch(cg), "conv graph launch");
    ya = fm_download(dy, 0UL, outN);
    unsafe { check(fm_err((const float*)ya.at(0UL), (const float*)ca.data, outN) < 1e-5f, "conv graph values"); }
    ya.free();
    std::cuda_graph_free(&cg);

    // TF32 tensor cores: same result to ~1e-3
    check(std::cuda_set_tf32(1), "tf32 on");
    check(std::cuda_conv2d(&d2, dx, dy, 2UL, 21UL, 15UL), "tf32 conv ran");
    ya = fm_download(dy, 0UL, outN);
    unsafe { check(fm_err((const float*)ya.at(0UL), (const float*)ca.data, outN) < 1e-2f, "tf32 conv values"); }
    ya.free();
    std::cuda_set_tf32(0);

    // pooling, upsample and concat against cnn.sc
    struct FeatureMap mp = std::maxpool2d_forward(&xa, 3UL, 2UL);
    struct FeatureMap ap = std::avgpool2d_forward(&xa, 3UL, 2UL);
    unsigned long pn = fm_size(&mp);
    struct CudaBuf dp = std::cuda_alloc(2UL * pn);
    check(std::cuda_maxpool2d(dx, dp, 2UL, 4UL, 21UL, 15UL, 3UL, 2UL), "cuda_maxpool2d ran");
    ya = fm_download(dp, 0UL, pn);
    unsafe { check(fm_err((const float*)ya.at(0UL), (const float*)mp.data, pn) == 0.0f, "cuda_maxpool2d values"); }
    ya.free();
    check(std::cuda_avgpool2d(dx, dp, 2UL, 4UL, 21UL, 15UL, 3UL, 2UL), "cuda_avgpool2d ran");
    ya = fm_download(dp, 0UL, pn);
    unsafe { check(fm_err((const float*)ya.at(0UL), (const float*)ap.data, pn) < 1e-6f, "cuda_avgpool2d values"); }
    ya.free();

    struct FeatureMap up = std::upsample2x_nearest(&xa);
    struct CudaBuf du = std::cuda_alloc(4UL * inN);
    check(std::cuda_upsample2x_nearest(dx, du, 4UL, 21UL, 15UL), "cuda_upsample2x ran");
    ya = fm_download(du, 0UL, 4UL * inN);
    unsafe { check(fm_err((const float*)ya.at(0UL), (const float*)up.data, 4UL * inN) == 0.0f, "cuda_upsample2x values"); }
    ya.free();

    struct FeatureMap cat = std::concat_channels(&xa, &xb);
    struct CudaBuf dxa = std::cuda_upload(xa.data, inN);
    struct CudaBuf dxb = std::cuda_upload(xb.data, inN);
    struct CudaBuf dc = std::cuda_alloc(2UL * inN);
    check(std::cuda_concat_channels(dxa, 4UL, dxb, 4UL, dc, 1UL, 21UL * 15UL), "cuda_concat_channels ran");
    ya = download(dc);
    unsafe { check(fm_err((const float*)ya.at(0UL), (const float*)cat.data, 2UL * inN) == 0.0f, "cuda_concat_channels values"); }
    ya.free();

    // whole U-Net
    struct UNet net = std::unet_new(3UL, 8UL, 16UL, 2UL);
    struct FeatureMap ux = std::feature_map_new(3UL, 32UL, 24UL);
    fm_fill(&ux, 7UL);
    struct FeatureMap uc = std::unet_forward(&net, &ux);
    struct FeatureMap ug = std::unet_forward_cuda(&net, &ux);
    check(ug.channels == 2UL && ug.height == 32UL && ug.width == 24UL, "unet_forward_cuda shape");
    unsafe { check(fm_err((const float*)ug.data, (const float*)uc.data, fm_size(&uc)) < 1e-4f, "unet_forward_cuda values"); }

    std::feature_map_free(&ug); std::feature_map_free(&uc); std::feature_map_free(&ux);
    std::unet_free(&net);
    std::feature_map_free(&cat); std::feature_map_free(&up); std::feature_map_free(&mp); std::feature_map_free(&ap);
    std::cuda_free(&dxa); std::cuda_free(&dxb); std::cuda_free(&dc); std::cuda_free(&du); std::cuda_free(&dp);
    std::cuda_free(&dx); std::cuda_free(&dy);
    std::cuda_conv2d_free(&d2);
    std::feature_map_free(&ca); std::feature_map_free(&cb); std::feature_map_free(&xa); std::feature_map_free(&xb);
    std::feature_map_free(&c1); std::feature_map_free(&x1);
    std::conv2d_free(&l1); std::conv2d_free(&l2);
}

static struct Complex* cplx_new(unsigned long n, unsigned long salt) {
    struct Complex* p;
    unsafe {
        p = (struct Complex*)std::alloc(n * sizeof(struct Complex));
        unsigned long i = 0UL;
        while (i < n) {
            p[i].re = (double)(((i + salt) * 37UL) % 101UL) * 0.02 - 1.0;
            p[i].im = (double)(((i + salt) * 53UL) % 89UL) * 0.01 - 0.4;
            i = i + 1UL;
        }
    }
    return p;
}

// max |a - b| / max(1, max |b|) over n complex values
static double cplx_err(const struct Complex* a, const struct Complex* b, unsigned long n) {
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

static void cplx_copy(struct Complex* dst, const struct Complex* src, unsigned long n) {
    unsigned long i = 0UL;
    unsafe { while (i < n) { dst[i] = src[i]; i = i + 1UL; } }
}

static void check_cufft() {
    // 1D double, power of two, against fft.h
    unsigned long n = 1024UL;
    struct Complex* x = cplx_new(n, 0UL);
    struct Complex* ref = cplx_new(n, 0UL);
    struct Complex* got = cplx_new(n, 0UL);
    std::fft(ref, n);
    check(std::cuda_fft(got, n), "cuda_fft ran");
    check(cplx_err(got, ref, n) < 1e-12, "cuda_fft values");
    check(std::cuda_ifft(got, n), "cuda_ifft ran");
    check(cplx_err(got, x, n) < 1e-12, "cuda_ifft round trip");

    // non-power-of-two length against dft.h
    unsigned long m = 1000UL;
    struct Complex* y = cplx_new(m, 5UL);
    struct Complex* yref = cplx_new(m, 0UL);
    std::dft(y, yref, m);
    check(std::cuda_fft(y, m), "cuda_fft n=1000 ran");
    check(cplx_err(y, yref, m) < 1e-10, "cuda_fft n=1000 values");

    // batch of 4 x 256 against per-row fft
    unsigned long bn = 256UL;
    struct Complex* bx = cplx_new(4UL * bn, 9UL);
    struct Complex* bref = cplx_new(4UL * bn, 9UL);
    unsigned long r = 0UL;
    unsafe { while (r < 4UL) { std::fft(bref + r * bn, bn); r = r + 1UL; } }
    check(std::cuda_fft_batch(bx, bn, 4UL, 0), "cuda_fft_batch ran");
    check(cplx_err(bx, bref, 4UL * bn) < 1e-12, "cuda_fft_batch values");

    // 2D against imaging.h
    unsigned long h = 32UL;
    unsigned long w = 64UL;
    struct Complex* img = cplx_new(h * w, 11UL);
    struct Complex* iref = cplx_new(h * w, 11UL);
    std::fft2d(iref, h, w);
    check(std::cuda_fft2d(img, h, w), "cuda_fft2d ran");
    check(cplx_err(img, iref, h * w) < 1e-12, "cuda_fft2d values");
    check(std::cuda_ifft2d(img, h, w), "cuda_ifft2d ran");
    struct Complex* iorig = cplx_new(h * w, 11UL);
    check(cplx_err(img, iorig, h * w) < 1e-12, "cuda_ifft2d round trip");

    // device fp32: c2c batch of 3 x 1024, r2c, c2r
    unsigned long fb = 3UL;
    struct Complex* fx = cplx_new(fb * n, 21UL);
    struct Complex* fref = cplx_new(fb * n, 21UL);
    r = 0UL;
    unsafe { while (r < fb) { std::fft(fref + r * n, n); r = r + 1UL; } }
    struct Vec<float> hf = std::vec_with_cap(2UL * fb * n);
    unsigned long i = 0UL;
    unsafe { while (i < fb * n) { hf.push((float)fx[i].re); hf.push((float)fx[i].im); i = i + 1UL; } }
    struct CudaBuf din = std::cuda_upload(hf.at(0UL), 2UL * fb * n);
    struct CudaBuf dfreq = std::cuda_alloc(2UL * fb * n);
    check(std::cuda_fft_c2c(din, dfreq, n, fb, 0), "cuda_fft_c2c ran");
    struct Vec<float> gf = download(dfreq);
    struct Complex* gc = cplx_new(fb * n, 0UL);
    i = 0UL;
    unsafe { while (i < fb * n) { gc[i].re = (double)*gf.at(2UL * i); gc[i].im = (double)*gf.at(2UL * i + 1UL); i = i + 1UL; } }
    check(cplx_err(gc, fref, fb * n) < 1e-5, "cuda_fft_c2c values");
    gf.free();
    check(std::cuda_fft_c2c(dfreq, dfreq, n, fb, 1), "cuda_fft_c2c inverse in place");
    gf = download(dfreq);
    check(maxdiff(&gf, &hf) < 1e-5f, "cuda_fft_c2c round trip");
    gf.free();

    // r2c on the real parts of row 0, compared with fft_real
    struct Vec<float> re = std::vec_with_cap(n);
    double* red;
    unsafe { red = (double*)std::alloc(n * sizeof(double)); }
    i = 0UL;
    unsafe { while (i < n) { re.push((float)fx[i].re); red[i] = (double)(float)fx[i].re; i = i + 1UL; } }
    struct Complex* rref = cplx_new(n, 0UL);
    unsafe { std::fft_real((const double*)red, rref, n); }
    struct CudaBuf dre = std::cuda_upload(re.at(0UL), n);
    struct CudaBuf dhalf = std::cuda_alloc(2UL * (n / 2UL + 1UL));
    check(std::cuda_fft_r2c(dre, dhalf, n, 1UL), "cuda_fft_r2c ran");
    gf = download(dhalf);
    i = 0UL;
    unsafe { while (i <= n / 2UL) { gc[i].re = (double)*gf.at(2UL * i); gc[i].im = (double)*gf.at(2UL * i + 1UL); i = i + 1UL; } }
    check(cplx_err(gc, rref, n / 2UL + 1UL) < 1e-5, "cuda_fft_r2c values");
    gf.free();
    struct CudaBuf dback = std::cuda_alloc(n);
    check(std::cuda_fft_c2r(dhalf, dback, n, 1UL), "cuda_fft_c2r ran");
    gf = download(dback);
    check(maxdiff(&gf, &re) < 1e-5f, "cuda_fft_c2r round trip");
    gf.free();

    std::cuda_free(&din); std::cuda_free(&dfreq); std::cuda_free(&dre); std::cuda_free(&dhalf); std::cuda_free(&dback);
    hf.free(); re.free();
    unsafe {
        std::dealloc((void*)x); std::dealloc((void*)ref); std::dealloc((void*)got);
        std::dealloc((void*)y); std::dealloc((void*)yref); std::dealloc((void*)bx); std::dealloc((void*)bref);
        std::dealloc((void*)img); std::dealloc((void*)iref); std::dealloc((void*)iorig);
        std::dealloc((void*)fx); std::dealloc((void*)fref); std::dealloc((void*)gc);
        std::dealloc((void*)red); std::dealloc((void*)rref);
    }
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

    check_cudnn();
    check_cufft();

    unsafe { printf("cuda: %d checks, %d failures\n", g_checks, g_fails); }
    return g_fails == 0 ? 0 : 1;
}
