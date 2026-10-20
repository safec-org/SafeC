// SafeC Standard Library — resampling core, written once for both sample types.
//
// Internal: included by resample.sc (double samples) and resample_f32.sc
// (float samples), each its own translation unit. Positions, the kernel
// and the sums are double either way; only loads and stores use RS_T. The
// includer defines RS_T and the public names RS_NEAREST, RS_LINEAR, RS_SINC.
#pragma once

void RS_NEAREST(const &RS_T in, unsigned long inLen, &RS_T out, unsigned long outLen) {
    if (inLen == 0UL || outLen == 0UL) { return; }
    double scale = (double)inLen / (double)outLen;
    unsigned long i = 0UL;
    while (i < outLen) {
        double srcPos = (double)i * scale;
        unsigned long idx = (unsigned long)(srcPos + 0.5);
        if (idx >= inLen) { idx = inLen - 1UL; }
        RS_T v;
        unsafe {
            v = in[idx];
            out[i] = v;
        }
        i = i + 1UL;
    }
}

void RS_LINEAR(const &RS_T in, unsigned long inLen, &RS_T out, unsigned long outLen) {
    if (inLen == 0UL || outLen == 0UL) { return; }
    if (inLen == 1UL) {
        RS_T v;
        unsafe { v = in[0]; }
        unsigned long i = 0UL;
        while (i < outLen) {
            unsafe { out[i] = v; }
            i = i + 1UL;
        }
        return;
    }
    double scale = 0.0;
    if (outLen > 1UL) {
        scale = (double)(inLen - 1UL) / (double)(outLen - 1UL);
    }
    unsigned long i = 0UL;
    while (i < outLen) {
        double srcPos = (double)i * scale;
        unsigned long idx0 = (unsigned long)srcPos;
        unsigned long idx1 = idx0 + 1UL;
        if (idx1 >= inLen) { idx1 = inLen - 1UL; }
        double frac = srcPos - (double)idx0;
        double v0;
        double v1;
        unsafe {
            v0 = (double)in[idx0];
            v1 = (double)in[idx1];
            out[i] = (RS_T)(v0 + frac * (v1 - v0));
        }
        i = i + 1UL;
    }
}

static double resample_sinc_(double x) {
    if (x == 0.0) { return 1.0; }
    double px = SAFEC_DSP_PI_RESAMPLE * x;
    return sin_d(px) / px;
}

// Each output is a windowed-sinc sum over the taps k = kStart + t around
// its source position, with x = srcPos - kStart:
//   weight_t = sin(pi c (x - t)) / (pi (x - t)) * (1 + cos(pi (x - t) / hw)) / 2
// sin(pi c (x - t)) and cos(pi (x - t) / hw) expand by the angle-difference
// formulas into four trig values per output (of x) times tables of the tap
// angles (of t), made once per call (and the four values themselves are
// carried from output to output by rotation). The taps are then independent, so they
// run four to a vector. Outputs whose x is a whole number (a tap at x - t = 0)
// take the per-tap form below, as do kernels wider than the tables. Outputs
// are independent and run on the thread pool for longer signals.
#define RS_DMAX   8UL    // largest kStart step advanced by rotation
#define RS_ANCHOR 64UL   // exact trig at least this often
#ifdef __SAFEC_FREESTANDING__
#define RS_TAB 72UL
#else
#define RS_TAB 264UL
#endif

struct ResampleJob {
    const RS_T*   in;
    RS_T*         out;
    unsigned long inLen;
    unsigned long hw;
    double        scale;
    double        cutoff;
};

// One output by direct per-tap evaluation (any x, any hw).
static double resample_one_(const &ResampleJob j, double x, unsigned long kStart, unsigned long kEnd)
    __attribute__((always_inline)) {
    double c = j.cutoff;
    double wb = SAFEC_DSP_PI_RESAMPLE / (double)j.hw;
    double sum = 0.0;
    unsigned long k = kStart;
    while (k <= kEnd) {
        double d = x - (double)(k - kStart);
        double sincVal = d == 0.0 ? c : sin_d(SAFEC_DSP_PI_RESAMPLE * c * d) / (SAFEC_DSP_PI_RESAMPLE * d);
        double v;
        unsafe { v = (double)j.in[k]; }
        sum = __builtin_fmuladd(v, sincVal * 0.5 * (1.0 + cos_d(wb * d)), sum);
        k = k + 1UL;
    }
    return sum;
}

static void resample_outs_body_(const &ResampleJob j, unsigned long lo, unsigned long hi) __attribute__((always_inline)) {
    double c = j.cutoff;
    double wb = SAFEC_DSP_PI_RESAMPLE / (double)j.hw;
    double pic = SAFEC_DSP_PI_RESAMPLE * c;
    int tabbed = 2UL * j.hw + 1UL <= RS_TAB - 3UL;
    // c1/s1 = cos/sin(pi c t), c2/s2 = cos/sin(wb t), tt = t (doubles)
    double c1[RS_TAB]; double s1[RS_TAB]; double c2[RS_TAB]; double s2[RS_TAB]; double tt[RS_TAB];
    if (tabbed) {
        unsigned long t = 0UL;
        while (t < RS_TAB) {
            double td = (double)t;
            c1[t] = cos_d(pic * td); s1[t] = sin_d(pic * td);
            c2[t] = cos_d(wb * td);  s2[t] = sin_d(wb * td);
            tt[t] = td;
            t = t + 1UL;
        }
    }
    // From one output to the next x moves by scale - d, d = the step of
    // kStart (a small integer), so sin/cos(pic x) and sin/cos(wb x) advance
    // by rotations from these tables; every RS_ANCHOR outputs, or after a
    // larger step, they are recomputed exactly so rounding can't build up.
    double rcp[RS_DMAX + 1UL]; double rsp[RS_DMAX + 1UL]; double rcw[RS_DMAX + 1UL]; double rsw[RS_DMAX + 1UL];
    unsigned long dd = 0UL;
    while (dd <= RS_DMAX) {
        double step = j.scale - (double)dd;
        rcp[dd] = cos_d(pic * step); rsp[dd] = sin_d(pic * step);
        rcw[dd] = cos_d(wb * step);  rsw[dd] = sin_d(wb * step);
        dd = dd + 1UL;
    }
    double ps = 0.0; double pc = 1.0; double ws = 0.0; double wc = 1.0;
    unsigned long prevK = 0UL;
    unsigned long i = lo;
    while (i < hi) {
        double srcPos = (double)i * j.scale;
        unsigned long center = (unsigned long)srcPos;
        unsigned long kStart = center >= j.hw ? center - j.hw : 0UL;
        unsigned long kEnd = center + j.hw;
        if (kEnd >= j.inLen) { kEnd = j.inLen - 1UL; }
        double x = srcPos - (double)kStart;
        unsigned long d = kStart - prevK;
        if (i == lo || (i - lo) % RS_ANCHOR == 0UL || kStart < prevK || d > RS_DMAX) {
            ps = sin_d(pic * x); pc = cos_d(pic * x);
            ws = sin_d(wb * x);  wc = cos_d(wb * x);
        } else {
            double t = __builtin_fmuladd(ps, rcp[d], pc * rsp[d]);
            pc = __builtin_fmuladd(pc, rcp[d], -ps * rsp[d]);
            ps = t;
            t = __builtin_fmuladd(ws, rcw[d], wc * rsw[d]);
            wc = __builtin_fmuladd(wc, rcw[d], -ws * rsw[d]);
            ws = t;
        }
        prevK = kStart;
        double sum = 0.0;
        if (!tabbed || x == (double)(unsigned long)x) {
            sum = resample_one_(j, x, kStart, kEnd);
        } else {
            unsigned long m = kEnd - kStart + 1UL;
            f64x4 vps; vps[0] = ps; vps[1] = ps; vps[2] = ps; vps[3] = ps;
            f64x4 vpc; vpc[0] = pc; vpc[1] = pc; vpc[2] = pc; vpc[3] = pc;
            f64x4 vws; vws[0] = ws; vws[1] = ws; vws[2] = ws; vws[3] = ws;
            f64x4 vwc; vwc[0] = wc; vwc[1] = wc; vwc[2] = wc; vwc[3] = wc;
            f64x4 vx;  vx[0] = x;   vx[1] = x;   vx[2] = x;   vx[3] = x;
            f64x4 half; half[0] = 0.5; half[1] = 0.5; half[2] = 0.5; half[3] = 0.5;
            f64x4 acc = half - half;
            unsigned long t = 0UL;
            unsafe {
                const RS_T* src = &j.in[kStart];
                // The table rows run 3 past m, so the last group may read
                // them; its extra lanes are dropped by the zero inputs.
                while (t < m) {
                    f64x4 v;
                    v[0] = (double)src[t];
                    v[1] = t + 1UL < m ? (double)src[t + 1UL] : 0.0;
                    v[2] = t + 2UL < m ? (double)src[t + 2UL] : 0.0;
                    v[3] = t + 3UL < m ? (double)src[t + 3UL] : 0.0;
                    f64x4 num = vps * *(const f64x4*)&c1[t] - vpc * *(const f64x4*)&s1[t];   // sin(pi c (x - t))
                    f64x4 win = __builtin_fmuladd(vwc, *(const f64x4*)&c2[t], vws * *(const f64x4*)&s2[t]);   // cos(wb (x - t))
                    f64x4 d = vx - *(const f64x4*)&tt[t];
                    f64x4 w = num * (half + half * win) / d;
                    acc = __builtin_fmuladd(v, w, acc);
                    t = t + 4UL;
                }
            }
            sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) / SAFEC_DSP_PI_RESAMPLE;   // the 1/pi of each weight
        }
        unsafe { j.out[i] = (RS_T)sum; }
        i = i + 1UL;
    }
}

static void resample_outs_base_(const &ResampleJob j, unsigned long lo, unsigned long hi) { resample_outs_body_(j, lo, hi); }
#if defined(__x86_64__)
static void resample_outs_fma_(const &ResampleJob j, unsigned long lo, unsigned long hi)
    __attribute__((target("avx2,fma"))) { resample_outs_body_(j, lo, hi); }
#endif

static void resample_outs_(&ResampleJob j, unsigned long lo, unsigned long hi) {
#if defined(__x86_64__)
    if (simd_level() >= SIMD_AVX2) { resample_outs_fma_(j, lo, hi); return; }
#endif
    resample_outs_base_(j, lo, hi);
}

void RS_SINC(const &RS_T in, unsigned long inLen, &RS_T out, unsigned long outLen,
                    unsigned long kernelHalfWidth) {
    if (inLen == 0UL || outLen == 0UL || kernelHalfWidth == 0UL) { return; }
    double ratio = (double)outLen / (double)inLen;
    struct ResampleJob j;
    unsafe {
        j.in = (const RS_T*)in;
        j.out = (RS_T*)out;
    }
    j.inLen = inLen;
    j.hw = kernelHalfWidth;
    j.scale = (double)inLen / (double)outLen;
    j.cutoff = ratio > 1.0 ? 1.0 : ratio;
    if (outLen * kernelHalfWidth >= 131072UL) parallel_for(outLen, 2048UL, resample_outs_, &j);
    else resample_outs_(&j, 0UL, outLen);
}
