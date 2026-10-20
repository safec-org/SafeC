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

static int near_d(double a, double b, double eps) {
    double d = a - b;
    if (d < 0.0) d = -d;
    return d <= eps;
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

int main() {
    struct TestSuite t = test_suite_init();
    unsafe {
        t.add("fft/dft", t_fft);
        t.add("window/fixed", t_window_fixed);
        t.add("biquad/fir/iir/ztransform", t_biquad_filters);
        t.add("conv/dct/resample", t_conv_dct_resample);
        t.add("stft", t_stft);
        t.add("comb/audio_buffer/timer_wheel", t_comb_audio_timer);
        t.add("imaging/cqt/cwt/minphase", t_imaging_cqt_cwt_minphase);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
