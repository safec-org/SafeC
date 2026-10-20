// std::dsp — fft/dft, window, biquad, filter (FIR/IIR), fixed, convolution,
// dct, resample, stft, comb, audio_buffer, timer_wheel, imaging, ztransform,
// cqt, cwt, minphase
#include <std/test/test.h>
#include <std/mem.h>
#include <std/panic.h>
#include <std/math.h>
#include <std/dsp/complex_dsp.h>
#include <std/dsp/fft.h>
#include <std/dsp/dft.h>
#include <std/dsp/window.h>
#include <std/dsp/biquad.h>
#include <std/dsp/filter.h>
#include <std/dsp/fixed.h>
#include <std/dsp/convolution.h>
#include <std/dsp/dct.h>
#include <std/dsp/resample.h>
#include <std/dsp/stft.h>
#include <std/dsp/comb.h>
#include <std/dsp/audio_buffer.h>
#include <std/dsp/timer_wheel.h>
#include <std/dsp/imaging.h>
#include <std/dsp/ztransform.h>
#include <std/dsp/cqt.h>
#include <std/dsp/cwt.h>
#include <std/dsp/minphase.h>
#include <std/dsp/dsp.h>

static int near_d(double a, double b, double eps) {
    double d = a - b;
    if (d < 0.0) d = -d;
    return d <= eps;
}

static int dsp_log2_t(unsigned long n) {
    int b = 0;
    while (n > 1UL) { n = n >> 1; b = b + 1; }
    return b;
}

static double rnd_state = 0.123;
static double rnd() {
    unsafe {
        rnd_state = rnd_state * 3.9 * (1.0 - rnd_state);   // logistic map, deterministic
        return rnd_state - 0.5;
    }
}

void t_fft() {
    unsafe {
        struct Complex a[64];
        struct Complex b[64];
        struct Complex orig[64];
        int i = 0;
        while (i < 64) {
            a[i] = complex_new(rnd(), rnd());
            orig[i] = a[i];
            i = i + 1;
        }
        dft((const struct Complex*)a, (struct Complex*)b, 64UL);
        ASSERT_TRUE(fft((struct Complex*)a, 64UL));
        double maxErr = 0.0;
        i = 0;
        while (i < 64) {
            double e = (a[i] - b[i]).abs();
            if (e > maxErr) maxErr = e;
            i = i + 1;
        }
        ASSERT_TRUE(maxErr < 1e-9);
        ASSERT_TRUE(ifft((struct Complex*)a, 64UL));
        maxErr = 0.0;
        i = 0;
        while (i < 64) {
            double e = (a[i] - orig[i]).abs();
            if (e > maxErr) maxErr = e;
            i = i + 1;
        }
        ASSERT_TRUE(maxErr < 1e-12);
        // idft undoes dft (complex input, so a sign slip in either shows)
        idft((const struct Complex*)b, (struct Complex*)a, 64UL);
        maxErr = 0.0;
        i = 0;
        while (i < 64) {
            double e = (a[i] - orig[i]).abs();
            if (e > maxErr) maxErr = e;
            i = i + 1;
        }
        ASSERT_TRUE(maxErr < 1e-12);
        ASSERT_FALSE(fft((struct Complex*)a, 48UL));   // not a power of two

        // Pure tone lands in its bin.
        double x[32];
        i = 0;
        while (i < 32) { x[i] = cos_d(2.0 * PI_D() * 4.0 * (double)i / 32.0); i = i + 1; }
        fft_real((const double*)x, (struct Complex*)a, 32UL);
        ASSERT_TRUE(near_d(a[4].abs(), 16.0, 1e-9));
        ASSERT_TRUE(near_d(a[5].abs(), 0.0, 1e-9));

        // Fixed-point FFT approximates the float one.
        struct FComplex f[16];
        i = 0;
        while (i < 16) { f[i] = fcomplex_new(fixed_from_float(x[i * 2] * 0.25), fixed_from_int(0)); i = i + 1; }
        ASSERT_TRUE(ffft((struct FComplex*)f, 16UL));
        struct Complex fc = fcomplex_to_float(f[4]);
        ASSERT_TRUE(near_d(fc.abs(), 2.0, 1e-3));
    }
}


// Larger transforms: the cache-blocked stages (n > 2048), the threaded
// path (n >= 32768), batches and the half-length real transform.
static struct Complex g_fa[131072];
static struct Complex g_fb[131072];
static double g_fx[16384];

static double max_diff(const struct Complex* a, const struct Complex* b, unsigned long n) {
    double m = 0.0;
    unsigned long i = 0UL;
    unsafe { while (i < n) { double e = (a[i] - b[i]).abs(); if (e > m) m = e; i = i + 1UL; } }
    return m;
}

void t_fft_large() {
    unsafe {
        ASSERT_TRUE(fft_prepare(131072UL));
        ASSERT_FALSE(fft_prepare(1000UL));
        // n = 4096 against the O(n^2) DFT.
        unsigned long n = 4096UL;
        unsigned long i = 0UL;
        while (i < n) { g_fa[i] = complex_new(rnd(), rnd()); i = i + 1UL; }
        dft((const struct Complex*)g_fa, (struct Complex*)g_fb, n);
        ASSERT_TRUE(fft((struct Complex*)g_fa, n));
        ASSERT_TRUE(max_diff((const struct Complex*)g_fa, (const struct Complex*)g_fb, n) < 1e-9);

        // n = 2^17, threaded: a sum of two tones lands in exactly two bins,
        // fft_serial agrees, and ifft round-trips.
        n = 131072UL;
        i = 0UL;
        while (i < n) {
            double t = (double)i / (double)n;
            g_fa[i] = complex_new(cos_d(2.0 * PI_D() * 1000.0 * t), sin_d(2.0 * PI_D() * 1000.0 * t) + 0.5 * cos_d(2.0 * PI_D() * 77.0 * t));
            g_fb[i] = g_fa[i];
            i = i + 1UL;
        }
        ASSERT_TRUE(fft((struct Complex*)g_fa, n));
        ASSERT_TRUE(near_d(g_fa[1000].abs(), (double)n, 1e-6));
        ASSERT_TRUE(near_d(g_fa[77].abs(), 0.25 * (double)n, 1e-6));
        ASSERT_TRUE(near_d(g_fa[n - 77UL].abs(), 0.25 * (double)n, 1e-6));
        ASSERT_TRUE(g_fa[500].abs() < 1e-6);
        ASSERT_TRUE(fft_serial((struct Complex*)g_fb, n));
        ASSERT_TRUE(max_diff((const struct Complex*)g_fa, (const struct Complex*)g_fb, n) < 1e-9);
        ASSERT_TRUE(ifft((struct Complex*)g_fa, n));
        double t1 = 1.0 / (double)n;
        ASSERT_TRUE(near_d(g_fa[1].re, cos_d(2.0 * PI_D() * 1000.0 * t1), 1e-12));
        ASSERT_TRUE(near_d(g_fa[1].im, sin_d(2.0 * PI_D() * 1000.0 * t1) + 0.5 * cos_d(2.0 * PI_D() * 77.0 * t1), 1e-12));

        // fft_batch == row-by-row fft_serial; inverse batch round-trips.
        unsigned long len = 256UL;
        unsigned long rows = 64UL;
        i = 0UL;
        while (i < len * rows) { g_fa[i] = complex_new(rnd(), rnd()); g_fb[i] = g_fa[i]; i = i + 1UL; }
        ASSERT_TRUE(fft_batch((struct Complex*)g_fa, len, rows, 0));
        unsigned long r = 0UL;
        while (r < rows) { fft_serial((struct Complex*)&g_fb[r * len], len); r = r + 1UL; }
        ASSERT_TRUE(max_diff((const struct Complex*)g_fa, (const struct Complex*)g_fb, len * rows) < 1e-12);
        ASSERT_TRUE(fft_batch((struct Complex*)g_fa, len, rows, 1));
        ASSERT_FALSE(fft_batch((struct Complex*)g_fa, 100UL, rows, 0));

        // fft_real / fft_real_batch == the complex transform of (x, 0).
        n = 1024UL;
        rows = 16UL;
        i = 0UL;
        while (i < n * rows) { g_fx[i] = rnd(); i = i + 1UL; }
        ASSERT_TRUE(fft_real_batch((const double*)g_fx, (struct Complex*)g_fa, n, rows));
        r = 0UL;
        double worst = 0.0;
        while (r < rows) {
            i = 0UL;
            while (i < n) { g_fb[i] = complex_new(g_fx[r * n + i], 0.0); i = i + 1UL; }
            fft_serial((struct Complex*)g_fb, n);
            double e = max_diff((const struct Complex*)&g_fa[r * n], (const struct Complex*)g_fb, n);
            if (e > worst) worst = e;
            r = r + 1UL;
        }
        ASSERT_TRUE(worst < 1e-10);
        double two[2];
        two[0] = 3.0; two[1] = -1.0;
        ASSERT_TRUE(fft_real((const double*)two, (struct Complex*)g_fa, 2UL));
        ASSERT_TRUE(near_d(g_fa[0].re, 2.0, 1e-15) && near_d(g_fa[1].re, 4.0, 1e-15));
        ASSERT_TRUE(fft_real((const double*)two, (struct Complex*)g_fa, 1UL));
        ASSERT_TRUE(near_d(g_fa[0].re, 3.0, 1e-15));
    }
}


// Block paths against their reference: overlap-add conv_fft vs conv_direct,
// and the windowed FIR block vs one sample at a time (split unevenly, so
// state carries across calls and chunk edges).
static double g_cx[20000];
static double g_ch[300];
static double g_cy1[20300];
static double g_cy2[20300];

void t_dsp_blocks() {
    unsafe {
        unsigned long i = 0UL;
        while (i < 20000UL) { g_cx[i] = rnd(); i = i + 1UL; }
        i = 0UL;
        while (i < 300UL) { g_ch[i] = rnd(); i = i + 1UL; }
        conv_direct((const double*)g_cx, 20000UL, (const double*)g_ch, 300UL, (double*)g_cy1);
        conv_fft((const double*)g_cx, 20000UL, (const double*)g_ch, 300UL, (double*)g_cy2);   // overlap-add
        double worst = 0.0;
        i = 0UL;
        while (i < 20299UL) { double e = g_cy1[i] - g_cy2[i]; if (e < 0.0) e = -e; if (e > worst) worst = e; i = i + 1UL; }
        ASSERT_TRUE(worst < 1e-9);
        conv_fft((const double*)g_cx, 2000UL, (const double*)g_ch, 300UL, (double*)g_cy2);    // one packed FFT
        conv_direct((const double*)g_cx, 2000UL, (const double*)g_ch, 300UL, (double*)g_cy1);
        worst = 0.0;
        i = 0UL;
        while (i < 2299UL) { double e = g_cy1[i] - g_cy2[i]; if (e < 0.0) e = -e; if (e > worst) worst = e; i = i + 1UL; }
        ASSERT_TRUE(worst < 1e-9);

        unsigned long taps[3];
        taps[0] = 1UL; taps[1] = 31UL; taps[2] = 300UL;   // 300 > the stack window: per-sample path
        int c = 0;
        while (c < 3) {
            unsigned long T = taps[c];
            double hist1[300];
            double hist2[300];
            struct FirFilter a = fir_init((const double*)g_ch, T, (double*)hist1);
            struct FirFilter b = fir_init((const double*)g_ch, T, (double*)hist2);
            a.reset();
            b.reset();
            i = 0UL;
            while (i < 3000UL) { g_cy1[i] = a.process(g_cx[i]); i = i + 1UL; }
            fir_process_block(&b, (const double*)g_cx, (double*)g_cy2, 7UL);
            fir_process_block(&b, (const double*)&g_cx[7], (double*)&g_cy2[7], 1100UL);
            unsigned long k = 1107UL;
            while (k < 1200UL) { g_cy2[k] = b.process(g_cx[k]); k = k + 1UL; }   // mixed with process()
            fir_process_block(&b, (const double*)&g_cx[1200], (double*)&g_cy2[1200], 1800UL);
            worst = 0.0;
            i = 0UL;
            while (i < 3000UL) { double e = g_cy1[i] - g_cy2[i]; if (e < 0.0) e = -e; if (e > worst) worst = e; i = i + 1UL; }
            ASSERT_TRUE(worst < 1e-10);
            c = c + 1;
        }

        struct Biquad q1 = biquad_lowpass(48000.0, 1000.0, 0.7071);
        struct Biquad q2 = biquad_lowpass(48000.0, 1000.0, 0.7071);
        i = 0UL;
        while (i < 1000UL) { g_cy1[i] = q1.process(g_cx[i]); i = i + 1UL; }
        biquad_process_block(&q2, (const double*)g_cx, (double*)g_cy2, 400UL);
        biquad_process_block(&q2, (const double*)&g_cx[400], (double*)&g_cy2[400], 600UL);
        // the block form may use FMA, so allow last-bit differences
        worst = 0.0;
        i = 0UL;
        while (i < 1000UL) { double e = g_cy1[i] - g_cy2[i]; if (e < 0.0) e = -e; if (e > worst) worst = e; i = i + 1UL; }
        ASSERT_TRUE(worst < 1e-12);
    }
}

void t_window_fixed() {
    unsafe {
        double w[9];
        window_hann((double*)w, 9UL);
        ASSERT_TRUE(near_d(w[0], 0.0, 1e-12));
        ASSERT_TRUE(near_d(w[4], 1.0, 1e-12));
        window_hamming((double*)w, 9UL);
        ASSERT_TRUE(near_d(w[0], 0.08, 1e-12));
        window_blackman((double*)w, 9UL);
        ASSERT_TRUE(near_d(w[4], 1.0, 1e-12));

        Fixed a = fixed_from_float(1.5);
        Fixed b = fixed_from_float(-2.25);
        ASSERT_TRUE(near_d(fixed_to_float(fixed_mul(a, b)), -3.375, 1e-6));
        ASSERT_TRUE(near_d(fixed_to_float(fixed_div(b, a)), -1.5, 1e-6));
        ASSERT_TRUE(near_d(fixed_to_float(fixed_sqrt(fixed_from_float(2.0))), 1.41421356, 1e-6));
        ASSERT_TRUE(near_d(fixed_to_float(fixed_sqrt(fixed_from_float(100.0))), 10.0, 1e-6));
        ASSERT_TRUE(near_d(fixed_to_float(fixed_sqrt(fixed_from_float(0.0001))), 0.01, 1e-5));
        ASSERT_EQ(fixed_to_int(fixed_from_int(7)), 7);
        ASSERT_TRUE(near_d(fixed_to_float(fixed_abs(b)), 2.25, 1e-6));
    }
}

void t_biquad_filters() {
    unsafe {
        struct Biquad lp = biquad_lowpass(48000.0, 1000.0, 0.70710678);
        ASSERT_TRUE(near_d(biquad_response_mag(&lp, 48000.0, 1.0), 1.0, 1e-4));
        ASSERT_TRUE(near_d(biquad_response_mag(&lp, 48000.0, 1000.0), 0.70710678, 1e-3));
        ASSERT_TRUE(biquad_response_mag(&lp, 48000.0, 10000.0) < 0.05);
        double y = 0.0;
        int i = 0;
        while (i < 2000) { y = lp.process(1.0); i = i + 1; }
        ASSERT_TRUE(near_d(y, 1.0, 1e-6));
        struct Biquad hp = biquad_highpass(48000.0, 1000.0, 0.70710678);
        ASSERT_TRUE(biquad_response_mag(&hp, 48000.0, 10.0) < 0.01);
        struct Biquad nt = biquad_notch(48000.0, 1000.0, 2.0);
        ASSERT_TRUE(biquad_response_mag(&nt, 48000.0, 1000.0) < 1e-6);
        struct Biquad pk = biquad_peaking(48000.0, 1000.0, 1.0, 6.0);
        ASSERT_TRUE(near_d(20.0 * log10_d(biquad_response_mag(&pk, 48000.0, 1000.0)), 6.0, 1e-3));

        // FIR impulse response == coefficients.
        double coeffs[4];
        coeffs[0] = 0.1; coeffs[1] = 0.2; coeffs[2] = 0.3; coeffs[3] = 0.4;
        double hist[4];
        safe_memset(&hist[0], 0, sizeof(double) * 4UL);
        struct FirFilter fir = fir_init((const double*)coeffs, 4UL, (double*)hist);
        ASSERT_TRUE(near_d(fir.process(1.0), 0.1, 1e-12));
        ASSERT_TRUE(near_d(fir.process(0.0), 0.2, 1e-12));
        ASSERT_TRUE(near_d(fir.process(0.0), 0.3, 1e-12));
        ASSERT_TRUE(near_d(fir.process(0.0), 0.4, 1e-12));
        ASSERT_TRUE(near_d(fir.process(0.0), 0.0, 1e-12));

        // IIR with the biquad's coefficients matches the biquad sample-for-sample.
        struct Biquad ref = biquad_lowpass(48000.0, 2000.0, 1.0);
        double bc[3];
        double ac[3];
        bc[0] = ref.b0; bc[1] = ref.b1; bc[2] = ref.b2;
        ac[0] = 1.0; ac[1] = ref.a1; ac[2] = ref.a2;
        double xh[3];
        double yh[3];
        safe_memset(&xh[0], 0, sizeof(double) * 3UL);
        safe_memset(&yh[0], 0, sizeof(double) * 3UL);
        struct IirFilter iir = iir_init((const double*)bc, 3UL, (const double*)ac, 3UL, (double*)xh, (double*)yh);
        double maxErr = 0.0;
        i = 0;
        while (i < 200) {
            double in = rnd();
            double e = iir.process(in) - ref.process(in);
            if (e < 0.0) e = -e;
            if (e > maxErr) maxErr = e;
            i = i + 1;
        }
        ASSERT_TRUE(maxErr < 1e-12);

        // N-th order bilinear transform agrees with the 2nd-order one.
        double B[3];
        double A[3];
        B[0] = 1.0; B[1] = 0.0; B[2] = 0.0;
        A[0] = 1.0; A[1] = 1.41421356 / 6283.0; A[2] = 1.0 / (6283.0 * 6283.0);
        double bo[3];
        double ao[3];
        bilinear_nth_order((const double*)B, (const double*)A, 2UL, 48000.0, (double*)bo, (double*)ao);
        struct Biquad b2 = bilinear_2nd_order(B[0], B[1], B[2], A[0], A[1], A[2], 48000.0);
        ASSERT_TRUE(near_d(bo[0], b2.b0, 1e-12));
        ASSERT_TRUE(near_d(ao[2], b2.a2, 1e-12));
        ASSERT_TRUE(near_d(ztransform_response_mag((const double*)bo, (const double*)ao, 2UL, 48000.0, 500.0),
                           biquad_response_mag(&b2, 48000.0, 500.0), 1e-9));
    }
}

void t_conv_dct_resample() {
    unsafe {
        double x[20];
        double h[7];
        double y1[26];
        double y2[26];
        int i = 0;
        while (i < 20) { x[i] = rnd(); i = i + 1; }
        i = 0;
        while (i < 7) { h[i] = rnd(); i = i + 1; }
        conv_direct((const double*)x, 20UL, (const double*)h, 7UL, (double*)y1);
        conv_fft((const double*)x, 20UL, (const double*)h, 7UL, (double*)y2);
        double maxErr = 0.0;
        i = 0;
        while (i < 26) { double e = y1[i] - y2[i]; if (e < 0.0) e = -e; if (e > maxErr) maxErr = e; i = i + 1; }
        ASSERT_TRUE(maxErr < 1e-10);

        double c[20];
        double r[20];
        dct2((const double*)x, (double*)c, 20UL);
        idct3((const double*)c, (double*)r, 20UL);
        maxErr = 0.0;
        i = 0;
        while (i < 20) { double e = r[i] - x[i]; if (e < 0.0) e = -e; if (e > maxErr) maxErr = e; i = i + 1; }
        ASSERT_TRUE(maxErr < 1e-10);

        double ramp[5];
        i = 0;
        while (i < 5) { ramp[i] = (double)i; i = i + 1; }
        double up[9];
        resample_linear((const double*)ramp, 5UL, (double*)up, 9UL);
        ASSERT_TRUE(near_d(up[0], 0.0, 1e-12));
        ASSERT_TRUE(near_d(up[8], 4.0, 1e-12));
        ASSERT_TRUE(near_d(up[1], 0.5, 1e-12));
    }
}

void t_stft() {
    unsafe {
        double sig[256];
        int i = 0;
        while (i < 256) { sig[i] = sin_d(0.3 * (double)i) + 0.2 * rnd(); i = i + 1; }
        unsigned long nf = stft_num_frames(256UL, 64UL, 16UL);
        ASSERT_EQ(nf, 13);
        ?&heap Complex specq = alloc(sizeof(struct Complex) * nf * 64UL);
        ?&heap double outq = alloc_zeroed(sizeof(double) * 256UL);
        ASSERT_FALSE(specq.is_null());
        ASSERT_FALSE(outq.is_null());
        if (specq.is_null() || outq.is_null()) return;
        &Complex spec = std::expect_ref(specq, "specq is null");
        &double out = std::expect_ref(outq, "outq is null");
        stft_forward((const double*)sig, 256UL, 64UL, 16UL, STFT_WIN_HANN, spec);
        stft_inverse((const struct Complex*)spec, nf, 64UL, 16UL, STFT_WIN_HANN, out);
        // Interior samples (fully overlapped) reconstruct exactly.
        double maxErr = 0.0;
        i = 64;
        while (i < 192) { double e = out[i] - sig[i]; if (e < 0.0) e = -e; if (e > maxErr) maxErr = e; i = i + 1; }
        ASSERT_TRUE(maxErr < 1e-9);
        unsigned long fs[2];
        fs[0] = 32UL; fs[1] = 64UL;
        ASSERT_TRUE(near_d(mr_stft_loss((const double*)sig, (const double*)sig, 256UL, (const unsigned long*)fs, 2UL), 0.0, 1e-9));
        ASSERT_TRUE(mr_stft_loss((const double*)sig, (const double*)out, 256UL, (const unsigned long*)fs, 2UL) >= 0.0);
        dealloc(specq);
        dealloc(outq);
    }
}

static int g_fired = 0;
static void on_timer(?&int ctx) { unsafe { g_fired = g_fired + 1; } }

void t_comb_audio_timer() {
    unsafe {
        double hist[3];
        safe_memset(&hist[0], 0, sizeof(double) * 3UL);
        struct CombFF ff = comb_ff_init((double*)hist, 3UL, 0.5);
        ASSERT_TRUE(near_d(ff.process(1.0), 1.0, 1e-12));
        ff.process(0.0);
        ff.process(0.0);
        ASSERT_TRUE(near_d(ff.process(0.0), 0.5, 1e-12));

        ?&heap AudioBuffer abq = alloc(sizeof(struct AudioBuffer));
        ?&heap TimerWheel twq = alloc(sizeof(struct TimerWheel));
        ASSERT_FALSE(abq.is_null());
        ASSERT_FALSE(twq.is_null());
        if (abq.is_null() || twq.is_null()) return;
        &AudioBuffer ab = std::expect_ref(abq, "abq is null");
        *ab = audio_buf_init(2UL);
        Fixed frames[8];
        int i = 0;
        while (i < 8) { frames[i] = fixed_from_int(i); i = i + 1; }
        ASSERT_EQ(ab.write_frames(frames, 4UL), 4);
        ASSERT_EQ(ab.readable_frames(), 4);
        Fixed outf[8];
        ASSERT_EQ(ab.read_frames(outf, 2UL), 2);
        ASSERT_EQ(fixed_to_int(outf[3]), 3);
        ASSERT_EQ(ab.readable_frames(), 2);
        dealloc(abq);

        &TimerWheel tw = std::expect_ref(twq, "twq is null");
        timer_wheel_init(tw);
        timer_add(tw, 3UL, on_timer, null);
        int pid = timer_add_periodic(tw, 5UL, on_timer, null);
        i = 0;
        while (i < 12) { tw.tick(); i = i + 1; }
        ASSERT_EQ(g_fired, 3);   // one-shot at 3, periodic at 5 and 10
        tw.cancel(pid);
        i = 0;
        while (i < 10) { tw.tick(); i = i + 1; }
        ASSERT_EQ(g_fired, 3);
        dealloc(twq);
    }
}

void t_imaging_cqt_cwt_minphase() {
    unsafe {
        double img[16];
        int i = 0;
        while (i < 16) { img[i] = (double)i; i = i + 1; }
        double k[9];
        safe_memset(&k[0], 0, sizeof(double) * 9UL);
        k[4] = 1.0;   // identity
        double out[16];
        conv2d((const double*)img, 4UL, 4UL, (const double*)k, 3UL, 3UL, (double*)out);
        ASSERT_TRUE(near_d(out[5], 5.0, 1e-12));
        double g[25];
        gaussian_kernel((double*)g, 5UL, 1.0);
        double sum = 0.0;
        i = 0;
        while (i < 25) { sum = sum + g[i]; i = i + 1; }
        ASSERT_TRUE(near_d(sum, 1.0, 1e-9));

        struct Complex c2[16];
        i = 0;
        while (i < 16) { c2[i] = complex_new(img[i], 0.0); i = i + 1; }
        fft2d((struct Complex*)c2, 4UL, 4UL);
        ASSERT_TRUE(near_d(c2[0].re, 120.0, 1e-9));
        ifft2d((struct Complex*)c2, 4UL, 4UL);
        ASSERT_TRUE(near_d(c2[7].re, 7.0, 1e-9));

        // CQT: a 440 Hz tone peaks in the bin whose centre is 440 Hz.
        double fs = 8000.0;
        ?&heap double toneq = alloc(sizeof(double) * 8000UL);
        ?&heap Complex cwq = alloc(sizeof(struct Complex) * 3UL * 512UL);
        ASSERT_FALSE(toneq.is_null());
        ASSERT_FALSE(cwq.is_null());
        if (toneq.is_null() || cwq.is_null()) return;
        &double tone = std::expect_ref(toneq, "toneq is null");
        i = 0;
        while (i < 8000) { tone[i] = sin_d(2.0 * PI_D() * 440.0 * (double)i / fs); i = i + 1; }
        struct Complex bins[24];
        cqt_forward((const double*)tone, 8000UL, fs, 220.0, 24UL, 12UL, (struct Complex*)bins);
        int best = 0;
        i = 1;
        while (i < 24) { if (bins[i].abs() > bins[best].abs()) best = i; i = i + 1; }
        ASSERT_EQ(best, 12);

        double scales[3];
        scales[0] = 4.0; scales[1] = 16.0; scales[2] = 64.0;
        &Complex cw = std::expect_ref(cwq, "cwq is null");
        cwt_morlet((const double*)tone, 512UL, (const double*)scales, 3UL, 6.0, cw);
        ASSERT_TRUE(cw[256].abs() >= 0.0);
        dealloc(cwq);
        dealloc(toneq);

        // h = [0.2, 1.0] has its zero at z = -5 (outside the unit circle);
        // its minimum-phase equivalent reflects that zero inside: [1.0, 0.2].
        double fir[2];
        fir[0] = 0.2; fir[1] = 1.0;
        double mp[256];   // output is fftLen samples (see minphase.h)
        minimum_phase((const double*)fir, 2UL, 256UL, (double*)mp);
        ASSERT_TRUE(near_d(mp[0], 1.0, 1e-3));
        ASSERT_TRUE(near_d(mp[1], 0.2, 1e-3));
        ASSERT_TRUE(near_d(mp[2], 0.0, 1e-3));
    }
}

// ── Single precision ────────────────────────────────────────────────────────
// Every _f32 path against the double one on the same input. Float keeps
// about 7 digits, so tolerances scale with the output's size.

static struct Complex32 g_ga[131072];
static struct Complex32 g_gb[131072];
static float g_gx[20300];
static float g_gy1[20300];
static float g_gy2[20300];

static double max_diff32(const struct Complex32* a, const struct Complex* b, unsigned long n) {
    double m = 0.0;
    unsigned long i = 0UL;
    unsafe {
        while (i < n) {
            double dr = (double)a[i].re - b[i].re;
            double di = (double)a[i].im - b[i].im;
            double e = sqrt_d(dr * dr + di * di);
            if (e > m) m = e;
            i = i + 1UL;
        }
    }
    return m;
}

static double max_diff32f(const struct Complex32* a, const struct Complex32* b, unsigned long n) {
    double m = 0.0;
    unsigned long i = 0UL;
    unsafe { while (i < n) { double e = (double)(a[i] - b[i]).abs(); if (e > m) m = e; i = i + 1UL; } }
    return m;
}

static double max_abs_diff_fd(const float* a, const double* b, unsigned long n) {
    double m = 0.0;
    unsigned long i = 0UL;
    unsafe { while (i < n) { double e = (double)a[i] - b[i]; if (e < 0.0) e = -e; if (e > m) m = e; i = i + 1UL; } }
    return m;
}

static double max_abs_diff_ff(const float* a, const float* b, unsigned long n) {
    double m = 0.0;
    unsigned long i = 0UL;
    unsafe { while (i < n) { double e = (double)(a[i] - b[i]); if (e < 0.0) e = -e; if (e > m) m = e; i = i + 1UL; } }
    return m;
}

void t_fft_f32() {
    unsafe {
        ASSERT_TRUE(fft_prepare_f32(131072UL));
        ASSERT_FALSE(fft_prepare_f32(1000UL));
        // Every size from 2 to 2^17 (codelet, both table kinds, threads)
        // against the double transform of the same (float) values.
        unsigned long n = 2UL;
        while (n <= 131072UL) {
            unsigned long i = 0UL;
            while (i < n) {
                g_ga[i] = complex32_new((float)rnd(), (float)rnd());
                g_fa[i] = complex_from_f32(g_ga[i]);
                i = i + 1UL;
            }
            ASSERT_TRUE(fft_f32((struct Complex32*)g_ga, n));
            ASSERT_TRUE(fft((struct Complex*)g_fa, n));
            ASSERT_TRUE(max_diff32((const struct Complex32*)g_ga, (const struct Complex*)g_fa, n) < 2e-6 * sqrt_d((double)n) * (double)dsp_log2_t(n));
            ASSERT_TRUE(ifft_f32((struct Complex32*)g_ga, n));
            ASSERT_TRUE(ifft((struct Complex*)g_fa, n));
            ASSERT_TRUE(max_diff32((const struct Complex32*)g_ga, (const struct Complex*)g_fa, n) < 1e-6 * (double)dsp_log2_t(n));
            n = n * 2UL;
        }
        // fft_serial_f32 == fft_f32; the tones of t_fft_large land in their bins.
        n = 131072UL;
        unsigned long i = 0UL;
        while (i < n) {
            double t = (double)i / (double)n;
            g_ga[i] = complex32_new((float)cos_d(2.0 * PI_D() * 1000.0 * t), (float)(sin_d(2.0 * PI_D() * 1000.0 * t) + 0.5 * cos_d(2.0 * PI_D() * 77.0 * t)));
            g_gb[i] = g_ga[i];
            i = i + 1UL;
        }
        ASSERT_TRUE(fft_f32((struct Complex32*)g_ga, n));
        ASSERT_TRUE(near_d((double)g_ga[1000].abs(), (double)n, 1e-6 * (double)n));
        ASSERT_TRUE(near_d((double)g_ga[77].abs(), 0.25 * (double)n, 1e-6 * (double)n));
        ASSERT_TRUE((double)g_ga[500].abs() < 1e-6 * (double)n);
        ASSERT_TRUE(fft_serial_f32((struct Complex32*)g_gb, n));
        ASSERT_TRUE(max_diff32f((const struct Complex32*)g_ga, (const struct Complex32*)g_gb, n) < 1e-7 * (double)n);
        ASSERT_TRUE(ifft_serial_f32((struct Complex32*)g_gb, n));
        ASSERT_TRUE(near_d((double)g_gb[1].re, cos_d(2.0 * PI_D() * 1000.0 / (double)n), 1e-5));

        // batch == row by row; real == complex of (x, 0); real batch.
        unsigned long len = 256UL;
        unsigned long rows = 64UL;
        i = 0UL;
        while (i < len * rows) { g_ga[i] = complex32_new((float)rnd(), (float)rnd()); g_gb[i] = g_ga[i]; i = i + 1UL; }
        ASSERT_TRUE(fft_batch_f32((struct Complex32*)g_ga, len, rows, 0));
        unsigned long r = 0UL;
        while (r < rows) { fft_serial_f32((struct Complex32*)&g_gb[r * len], len); r = r + 1UL; }
        ASSERT_TRUE(max_diff32f((const struct Complex32*)g_ga, (const struct Complex32*)g_gb, len * rows) < 1e-5);
        ASSERT_TRUE(fft_batch_f32((struct Complex32*)g_ga, len, rows, 1));
        ASSERT_FALSE(fft_batch_f32((struct Complex32*)g_ga, 100UL, rows, 0));
        n = 1024UL;
        rows = 16UL;
        i = 0UL;
        while (i < n * rows) { g_gx[i] = (float)rnd(); i = i + 1UL; }
        ASSERT_TRUE(fft_real_batch_f32((const float*)g_gx, (struct Complex32*)g_ga, n, rows));
        double worst = 0.0;
        r = 0UL;
        while (r < rows) {
            i = 0UL;
            while (i < n) { g_fb[i] = complex_new((double)g_gx[r * n + i], 0.0); i = i + 1UL; }
            fft_serial((struct Complex*)g_fb, n);
            double e = max_diff32((const struct Complex32*)&g_ga[r * n], (const struct Complex*)g_fb, n);
            if (e > worst) worst = e;
            r = r + 1UL;
        }
        ASSERT_TRUE(worst < 1e-4);
        float two[2];
        two[0] = 3.0; two[1] = -1.0;
        ASSERT_TRUE(fft_real_f32((const float*)two, (struct Complex32*)g_ga, 2UL));
        ASSERT_TRUE(near_d((double)g_ga[0].re, 2.0, 1e-6) && near_d((double)g_ga[1].re, 4.0, 1e-6));
    }
}

void t_dsp_f32() {
    unsafe {
        unsigned long i = 0UL;
        while (i < 20000UL) { g_gx[i] = (float)rnd(); g_cx[i] = (double)g_gx[i]; i = i + 1UL; }
        float hf[300];
        i = 0UL;
        while (i < 300UL) { hf[i] = (float)rnd(); g_ch[i] = (double)hf[i]; i = i + 1UL; }

        // dot
        double dd = dsp_dot_f64((const double*)g_cx, (const double*)g_ch, 300UL);
        float df = dsp_dot_f32((const float*)g_gx, (const float*)hf, 300UL);
        ASSERT_TRUE(near_d((double)df, dd, 1e-4));
        ASSERT_TRUE(near_d((double)dsp_dot_f32((const float*)g_gx, (const float*)hf, 13UL),
                           dsp_dot_f64((const double*)g_cx, (const double*)g_ch, 13UL), 1e-5));

        // convolution: direct and both FFT paths against the double direct one
        conv_direct((const double*)g_cx, 20000UL, (const double*)g_ch, 300UL, (double*)g_cy1);
        conv_direct_f32((const float*)g_gx, 20000UL, (const float*)hf, 300UL, (float*)g_gy1);
        conv_fft_f32((const float*)g_gx, 20000UL, (const float*)hf, 300UL, (float*)g_gy2);   // overlap-add
        ASSERT_TRUE(max_abs_diff_fd((const float*)g_gy1, (const double*)g_cy1, 20299UL) < 1e-4);
        ASSERT_TRUE(max_abs_diff_fd((const float*)g_gy2, (const double*)g_cy1, 20299UL) < 1e-4);
        conv_direct((const double*)g_cx, 2000UL, (const double*)g_ch, 300UL, (double*)g_cy1);
        conv_fft_f32((const float*)g_gx, 2000UL, (const float*)hf, 300UL, (float*)g_gy2);    // one packed FFT
        ASSERT_TRUE(max_abs_diff_fd((const float*)g_gy2, (const double*)g_cy1, 2299UL) < 1e-4);

        // FIR: block == per sample (mixed, uneven splits), and close to double
        unsigned long taps[3];
        taps[0] = 1UL; taps[1] = 31UL; taps[2] = 300UL;
        int c = 0;
        while (c < 3) {
            unsigned long T = taps[c];
            float hist1[300];
            float hist2[300];
            double histd[300];
            struct FirFilter32 a = fir_init_f32((const float*)hf, T, (float*)hist1);
            struct FirFilter32 b = fir_init_f32((const float*)hf, T, (float*)hist2);
            struct FirFilter d = fir_init((const double*)g_ch, T, (double*)histd);
            a.reset();
            b.reset();
            d.reset();
            i = 0UL;
            while (i < 3000UL) { g_gy1[i] = a.process(g_gx[i]); g_cy1[i] = d.process(g_cx[i]); i = i + 1UL; }
            fir_process_block_f32(&b, (const float*)g_gx, (float*)g_gy2, 7UL);
            fir_process_block_f32(&b, (const float*)&g_gx[7], (float*)&g_gy2[7], 1100UL);
            unsigned long k = 1107UL;
            while (k < 1200UL) { g_gy2[k] = b.process(g_gx[k]); k = k + 1UL; }
            fir_process_block_f32(&b, (const float*)&g_gx[1200], (float*)&g_gy2[1200], 1800UL);
            ASSERT_TRUE(max_abs_diff_ff((const float*)g_gy1, (const float*)g_gy2, 3000UL) < 1e-5);
            ASSERT_TRUE(max_abs_diff_fd((const float*)g_gy2, (const double*)g_cy1, 3000UL) < 1e-4);
            c = c + 1;
        }

        // Biquad32 from a double design: per sample == block, close to double
        struct Biquad qd = biquad_lowpass(48000.0, 1000.0, 0.7071);
        struct Biquad32 q1 = biquad_to_f32(qd);
        struct Biquad32 q2 = biquad_to_f32(qd);
        i = 0UL;
        while (i < 1000UL) { g_gy1[i] = q1.process(g_gx[i]); g_cy1[i] = qd.process(g_cx[i]); i = i + 1UL; }
        biquad_process_block_f32(&q2, (const float*)g_gx, (float*)g_gy2, 400UL);
        biquad_process_block_f32(&q2, (const float*)&g_gx[400], (float*)&g_gy2[400], 600UL);
        ASSERT_TRUE(max_abs_diff_ff((const float*)g_gy1, (const float*)g_gy2, 1000UL) < 1e-5);
        ASSERT_TRUE(max_abs_diff_fd((const float*)g_gy2, (const double*)g_cy1, 1000UL) < 1e-4);
        q1.reset();
        ASSERT_TRUE(q1.z1 == (float)0.0 && q1.z2 == (float)0.0);

        // windows
        float wf[257];
        double wd[257];
        window_hann_f32((float*)wf, 257UL);
        window_hann((double*)wd, 257UL);
        ASSERT_TRUE(max_abs_diff_fd((const float*)wf, (const double*)wd, 257UL) < 1e-7);
        window_hamming_f32((float*)wf, 257UL);
        window_hamming((double*)wd, 257UL);
        ASSERT_TRUE(max_abs_diff_fd((const float*)wf, (const double*)wd, 257UL) < 1e-7);
        window_blackman_f32((float*)wf, 257UL);
        window_blackman((double*)wd, 257UL);
        ASSERT_TRUE(max_abs_diff_fd((const float*)wf, (const double*)wd, 257UL) < 1e-7);
        window_rectangular_f32((float*)wf, 1UL);
        ASSERT_TRUE(wf[0] == (float)1.0);
        window_hann_f32((float*)wf, 9UL);
        float ones[9];
        i = 0UL;
        while (i < 9UL) { ones[i] = (float)2.0; i = i + 1UL; }
        window_apply_f32((float*)ones, (const float*)wf, 9UL);
        ASSERT_TRUE(near_d((double)ones[4], 2.0, 1e-6) && near_d((double)ones[0], 0.0, 1e-6));

        // resample_sinc against the windowed-sinc sum written out directly
        resample_sinc((const double*)g_cx, 1000UL, (double*)g_cy1, 1470UL, 16UL);
        double rworst = 0.0;
        i = 0UL;
        while (i < 1470UL) {
            double scale = 1000.0 / 1470.0;
            double src = (double)i * scale;
            unsigned long ctr = (unsigned long)src;
            unsigned long k0 = ctr >= 16UL ? ctr - 16UL : 0UL;
            unsigned long k1 = ctr + 16UL < 1000UL ? ctr + 16UL : 999UL;
            double ref = 0.0;
            unsigned long k = k0;
            while (k <= k1) {
                double d = src - (double)k;
                double sv = d == 0.0 ? 1.0 : sin_d(PI_D() * d) / (PI_D() * d);   // cutoff 1 (upsampling)
                ref = ref + g_cx[k] * sv * 0.5 * (1.0 + cos_d(PI_D() * d / 16.0));
                k = k + 1UL;
            }
            double e = ref - g_cy1[i];
            if (e < 0.0) e = -e;
            if (e > rworst) rworst = e;
            i = i + 1UL;
        }
        // (resample_sinc carries the kernel's angles from output to output, so
        // it differs from this per-position evaluation by the rounding of the
        // positions themselves: ~1e-13 at positions near 1000)
        ASSERT_TRUE(rworst < 1e-10);
        resample_sinc((const double*)g_cx, 1470UL, (double*)g_cy2, 1000UL, 16UL);   // downsampling: cutoff < 1
        resample_sinc_f32((const float*)g_gx, 1470UL, (float*)g_gy2, 1000UL, 16UL);
        ASSERT_TRUE(max_abs_diff_fd((const float*)g_gy2, (const double*)g_cy2, 1000UL) < 1e-6);
        // resample: float samples give the double result rounded
        resample_sinc_f32((const float*)g_gx, 1000UL, (float*)g_gy1, 1470UL, 16UL);
        ASSERT_TRUE(max_abs_diff_fd((const float*)g_gy1, (const double*)g_cy1, 1470UL) < 1e-6);
        resample_linear((const double*)g_cx, 1000UL, (double*)g_cy1, 777UL);
        resample_linear_f32((const float*)g_gx, 1000UL, (float*)g_gy1, 777UL);
        ASSERT_TRUE(max_abs_diff_fd((const float*)g_gy1, (const double*)g_cy1, 777UL) < 1e-6);
        resample_nearest((const double*)g_cx, 1000UL, (double*)g_cy1, 333UL);
        resample_nearest_f32((const float*)g_gx, 1000UL, (float*)g_gy1, 333UL);
        ASSERT_TRUE(max_abs_diff_fd((const float*)g_gy1, (const double*)g_cy1, 333UL) == 0.0);

        // STFT: forward close to double; inverse reconstructs the interior
        unsigned long nf = stft_num_frames(1024UL, 128UL, 32UL);
        stft_forward((const double*)g_cx, 1024UL, 128UL, 32UL, STFT_WIN_HANN, (struct Complex*)g_fa);
        stft_forward_f32((const float*)g_gx, 1024UL, 128UL, 32UL, STFT_WIN_HANN, (struct Complex32*)g_ga);
        ASSERT_TRUE(max_diff32((const struct Complex32*)g_ga, (const struct Complex*)g_fa, nf * 128UL) < 1e-5);
        i = 0UL;
        while (i < 1024UL) { g_gy1[i] = (float)0.0; i = i + 1UL; }
        stft_inverse_f32((const struct Complex32*)g_ga, nf, 128UL, 32UL, STFT_WIN_HANN, (float*)g_gy1);
        ASSERT_TRUE(max_abs_diff_ff((const float*)&g_gy1[128], (const float*)&g_gx[128], 768UL) < 1e-5);
    }
}

void t_dsp_f32_more() {
    unsafe {
        unsigned long i = 0UL;
        while (i < 4096UL) { g_gx[i] = (float)rnd(); g_cx[i] = (double)g_gx[i]; i = i + 1UL; }

        // IIR (a biquad's coefficients as a 2nd-order IIR) close to double
        struct Biquad bq = biquad_lowpass(48000.0, 2000.0, 0.7071);
        double bd[3]; double ad[3];
        bd[0] = bq.b0; bd[1] = bq.b1; bd[2] = bq.b2; ad[0] = 1.0; ad[1] = bq.a1; ad[2] = bq.a2;
        float bf[3]; float af[3];
        int c = 0;
        while (c < 3) { bf[c] = (float)bd[c]; af[c] = (float)ad[c]; c = c + 1; }
        double xd[3]; double yd[3]; float xf[3]; float yf[3];
        struct IirFilter id = iir_init((const double*)bd, 3UL, (const double*)ad, 3UL, (double*)xd, (double*)yd);
        struct IirFilter32 iff = iir_init_f32((const float*)bf, 3UL, (const float*)af, 3UL, (float*)xf, (float*)yf);
        id.reset();
        iff.reset();
        iir_process_block(&id, (const double*)g_cx, (double*)g_cy1, 2000UL);
        iir_process_block_f32(&iff, (const float*)g_gx, (float*)g_gy1, 2000UL);
        ASSERT_TRUE(max_abs_diff_fd((const float*)g_gy1, (const double*)g_cy1, 2000UL) < 1e-4);

        // comb filters and Karplus-Strong against double
        float hf1[37]; double hd1[37];
        struct CombFF32 cf = comb_ff_init_f32((float*)hf1, 37UL, (float)0.6);
        struct CombFF cd = comb_ff_init((double*)hd1, 37UL, 0.6);
        struct CombFB32 bf32 = comb_fb_init_f32((float*)hf1, 37UL, (float)0.6);
        struct CombFB bd64 = comb_fb_init((double*)hd1, 37UL, 0.6);
        cf.reset();
        cd.reset();
        double worst = 0.0;
        i = 0UL;
        while (i < 500UL) {
            double e = (double)cf.process(g_gx[i]) - cd.process(g_cx[i]);
            if (e < 0.0) e = -e;
            if (e > worst) worst = e;
            i = i + 1UL;
        }
        ASSERT_TRUE(worst < 1e-6);
        bf32.reset();
        bd64.reset();
        worst = 0.0;
        i = 0UL;
        while (i < 500UL) {
            double e = (double)bf32.process(g_gx[i]) - bd64.process(g_cx[i]);
            if (e < 0.0) e = -e;
            if (e > worst) worst = e;
            i = i + 1UL;
        }
        ASSERT_TRUE(worst < 1e-5);
        float kbf[50]; double kbd[50];
        struct KarplusStrong32 kf = ks_init_f32((float*)kbf, 50UL, (float)0.996);
        struct KarplusStrong kd = ks_init((double*)kbd, 50UL, 0.996);
        kf.reset((const float*)g_gx, 50UL);
        kd.reset((const double*)g_cx, 50UL);
        worst = 0.0;
        i = 0UL;
        while (i < 1000UL) {
            double e = (double)kf.process() - kd.process();
            if (e < 0.0) e = -e;
            if (e > worst) worst = e;
            i = i + 1UL;
        }
        ASSERT_TRUE(worst < 1e-5);

        // conv2d / fft2d / ifft2d
        double gk[25];
        gaussian_kernel((double*)gk, 5UL, 1.0);
        float gkf[25];
        c = 0;
        while (c < 25) { gkf[c] = (float)gk[c]; c = c + 1; }
        conv2d((const double*)g_cx, 64UL, 64UL, (const double*)gk, 5UL, 5UL, (double*)g_cy1);
        conv2d_f32((const float*)g_gx, 64UL, 64UL, (const float*)gkf, 5UL, 5UL, (float*)g_gy1);
        ASSERT_TRUE(max_abs_diff_fd((const float*)g_gy1, (const double*)g_cy1, 4096UL) < 1e-6);
        i = 0UL;
        while (i < 4096UL) { g_ga[i] = complex32_new(g_gx[i], (float)0.0); g_fa[i] = complex_new(g_cx[i], 0.0); i = i + 1UL; }
        fft2d_f32((struct Complex32*)g_ga, 64UL, 64UL);
        fft2d((struct Complex*)g_fa, 64UL, 64UL);
        ASSERT_TRUE(max_diff32((const struct Complex32*)g_ga, (const struct Complex*)g_fa, 4096UL) < 1e-4);
        ifft2d_f32((struct Complex32*)g_ga, 64UL, 64UL);
        ASSERT_TRUE(near_d((double)g_ga[77].re, g_cx[77], 1e-6) && near_d((double)g_ga[77].im, 0.0, 1e-6));

        // DCT round trip and DFT against the FFT
        dct2_f32((const float*)g_gx, (float*)g_gy1, 100UL);
        dct2((const double*)g_cx, (double*)g_cy1, 100UL);
        ASSERT_TRUE(max_abs_diff_fd((const float*)g_gy1, (const double*)g_cy1, 100UL) < 1e-5);
        idct3_f32((const float*)g_gy1, (float*)g_gy2, 100UL);
        ASSERT_TRUE(max_abs_diff_ff((const float*)g_gy2, (const float*)g_gx, 100UL) < 1e-6);
        i = 0UL;
        while (i < 64UL) { g_gb[i] = complex32_new(g_gx[i], g_gx[i + 64UL]); i = i + 1UL; }
        dft_f32((const struct Complex32*)g_gb, (struct Complex32*)g_ga, 64UL);
        fft_f32((struct Complex32*)g_gb, 64UL);
        ASSERT_TRUE(max_diff32f((const struct Complex32*)g_ga, (const struct Complex32*)g_gb, 64UL) < 1e-5);
        idft_f32((const struct Complex32*)g_ga, (struct Complex32*)g_gb, 64UL);
        ASSERT_TRUE(near_d((double)g_gb[5].re, (double)g_gx[5], 1e-6) && near_d((double)g_gb[5].im, (double)g_gx[69], 1e-6));
        dft_real_f32((const float*)g_gx, (struct Complex32*)g_ga, 64UL);
        i = 0UL;
        while (i < 64UL) { g_gb[i] = complex32_new(g_gx[i], (float)0.0); i = i + 1UL; }
        fft_f32((struct Complex32*)g_gb, 64UL);
        ASSERT_TRUE(max_diff32f((const struct Complex32*)g_ga, (const struct Complex32*)g_gb, 64UL) < 1e-5);
    }
}

int main() {
    struct TestSuite t = test_suite_init();
    unsafe {
        t.add("fft/dft", t_fft);
        t.add("fft large/batch/real", t_fft_large);
        t.add("conv_fft/fir/biquad blocks", t_dsp_blocks);
        t.add("window/fixed", t_window_fixed);
        t.add("biquad/fir/iir/ztransform", t_biquad_filters);
        t.add("conv/dct/resample", t_conv_dct_resample);
        t.add("stft", t_stft);
        t.add("comb/audio_buffer/timer_wheel", t_comb_audio_timer);
        t.add("imaging/cqt/cwt/minphase", t_imaging_cqt_cwt_minphase);
        t.add("fft f32", t_fft_f32);
        t.add("dsp f32", t_dsp_f32);
        t.add("iir/comb/imaging/dct/dft f32", t_dsp_f32_more);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
