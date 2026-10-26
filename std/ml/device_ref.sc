// SafeC Standard Library — the device kernels (device_kernels.cu) on the
// CPU, over host addresses: what a backend runs for a kernel it has no GPU
// version of (the Metal backend, whose buffers the CPU can address directly),
// and the reference the GPU kernels are tested against. Same arguments, same
// arithmetic, one loop per kernel.
#pragma once
#include <std/stddef.h>
#include <std/ml/device.h>
#include <std/ml/device_kernels.h>
#include <std/math.h>
#include <std/simd/gemm.h>
#include <std/ml/float16.h>

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);

// The EW codes (device_kernels.cu).
#define REF_EW_ADD 0
#define REF_EW_SUB 1
#define REF_EW_MUL 2
#define REF_EW_SCALE 3
#define REF_EW_RELU 4
#define REF_EW_FILL 5
#define REF_EW_COPY 6
#define REF_EW_ACC 7
#define REF_EW_DEC 8
#define REF_EW_ACC_MUL 9
#define REF_EW_ACC_SCALE 10
#define REF_EW_ACC_RELU 11
#define REF_EW_ACC_K 12
#define REF_EW_ACC_SIGB 13
#define REF_EW_ACC_TANHB 14

static float dref_sig_(float x) { return (float)1.0 / ((float)1.0 + exp_f(-x)); }

static float dref_gelu_(float x) {
    float c = (float)0.7978845608028654;
    return (float)0.5 * x * ((float)1.0 + tanh_f(c * (x + (float)0.044715 * x * x * x)));
}

static float dref_gelu_d_(float x) {
    float c = (float)0.7978845608028654;
    float t = tanh_f(c * (x + (float)0.044715 * x * x * x));
    float du = c * ((float)1.0 + (float)0.134145 * x * x);
    return (float)0.5 * ((float)1.0 + t) + (float)0.5 * x * ((float)1.0 - t * t) * du;
}

static float dref_abs_(float v) { return v < (float)0.0 ? -v : v; }
static float dref_max_(float a, float b) { return a > b ? a : b; }
static float dref_min_(float a, float b) { return a < b ? a : b; }

static void dref_ew_(const &DevArgs a, unsigned long n) {
    unsafe {
        int op = (int)a.u[0];
        const float* x = (const float*)a.p[0];
        const float* y = (const float*)a.p[1];
        float* o = (float*)a.p[2];
        const float* z = (const float*)a.p[3];
        float k = a.f[0];
        float k2 = a.f[1];
        unsigned long i = 0UL;
        if (a.u[1] != 0U && ((op >= REF_EW_ACC && op <= REF_EW_ACC_TANHB) || (op >= 40 && op <= 51))) {
            // write instead of add: out was an unwritten zero gradient
            while (i < n) { o[i] = (float)0.0; i = i + 1UL; }
            i = 0UL;
        }
        while (i < n) {
            if (op == REF_EW_ADD) o[i] = x[i] + y[i];
            else if (op == REF_EW_SUB) o[i] = x[i] - y[i];
            else if (op == REF_EW_MUL) o[i] = x[i] * y[i];
            else if (op == REF_EW_SCALE) o[i] = x[i] * k;
            else if (op == REF_EW_RELU) o[i] = x[i] > (float)0.0 ? x[i] : (float)0.0;
            else if (op == REF_EW_FILL) o[i] = k;
            else if (op == REF_EW_COPY) o[i] = x[i];
            else if (op == REF_EW_ACC) o[i] = o[i] + x[i];
            else if (op == REF_EW_DEC) o[i] = o[i] - x[i];
            else if (op == REF_EW_ACC_MUL) o[i] = o[i] + x[i] * y[i];
            else if (op == REF_EW_ACC_SCALE) o[i] = o[i] + x[i] * k;
            else if (op == REF_EW_ACC_RELU) { if (y[i] > (float)0.0) o[i] = o[i] + x[i]; }
            else if (op == REF_EW_ACC_K) o[i] = o[i] + k;
            else if (op == REF_EW_ACC_SIGB) { float s = y[i]; o[i] = o[i] + x[i] * s * ((float)1.0 - s); }
            else if (op == REF_EW_ACC_TANHB) { float t = y[i]; o[i] = o[i] + x[i] * ((float)1.0 - t * t); }
            else if (op == 20) o[i] = exp_f(x[i]);
            else if (op == 21) o[i] = log_f(x[i]);
            else if (op == 22) o[i] = dref_abs_(x[i]);
            else if (op == 23) o[i] = x[i] > (float)0.0 ? x[i] : k * x[i];
            else if (op == 24) o[i] = dref_min_(dref_max_(x[i], k), k2);
            else if (op == 25) o[i] = dref_sig_(x[i]);
            else if (op == 26) o[i] = tanh_f(x[i]);
            else if (op == 27) o[i] = dref_gelu_(x[i]);
            else if (op == 28) o[i] = x[i] * dref_sig_(x[i]);
            else if (op == 29) o[i] = x[i] + k;
            else if (op == 30) o[i] = x[i] <= y[i] ? x[i] : y[i];
            else if (op == 31) o[i] = x[i] > (float)0.0 ? x[i] : k * (exp_f(x[i]) - (float)1.0);
            else if (op == 32) o[i] = o[i] + k * (x[i] - o[i]);
            else if (op == 33) o[i] = x[i] * y[0];
            else if (op == 40) o[i] = o[i] + y[0];
            else if (op == 41) o[i] = o[i] + x[i] * y[i];
            else if (op == 42) o[i] = o[i] + x[i] / y[i];
            else if (op == 43) { float v = y[i]; o[i] = o[i] + (v > (float)0.0 ? x[i] : (v < (float)0.0 ? -x[i] : (float)0.0)); }
            else if (op == 44) o[i] = o[i] + x[i] * (y[i] > (float)0.0 ? (float)1.0 : k);
            else if (op == 45) { if (y[i] == z[i]) o[i] = o[i] + x[i]; }
            else if (op == 46) o[i] = o[i] + x[i] * dref_gelu_d_(y[i]);
            else if (op == 47) { float s = dref_sig_(y[i]); o[i] = o[i] + x[i] * (s + y[i] * s * ((float)1.0 - s)); }
            else if (op == 48) { if (y[i] <= z[i]) o[i] = o[i] + x[i]; }
            else if (op == 49) { if (!(z[i] <= y[i])) o[i] = o[i] + x[i]; }
            else if (op == 50) o[i] = o[i] + x[i] * (y[i] > (float)0.0 ? (float)1.0 : z[i] + k);
            else if (op == 51) o[i] = o[i] + k * y[0];
            i = i + 1UL;
        }
    }
}

static float dref_lse_(const float* x, unsigned long C) {
    unsafe {
        float mx = x[0];
        unsigned long c = 1UL;
        while (c < C) { mx = dref_max_(mx, x[c]); c = c + 1UL; }
        float s = (float)0.0;
        c = 0UL;
        while (c < C) { s = s + exp_f(x[c] - mx); c = c + 1UL; }
        return mx + log_f(s);
    }
}

// The fused activation after a normalization (device_kernels.cu's fused_act / fused_dy).
static float dref_fact_(const &DevArgs a, float v) {
    if (a.u[5] == 0U || v > (float)0.0) return v;
    return a.u[5] == 1U ? (float)0.0 : a.f[1] * v;
}
static float dref_fdy_(const &DevArgs a, unsigned long i) {
    unsafe {
        float dy = ((const float*)a.p[0])[i];
        if (a.u[5] == 0U) return dy;
        if (((const float*)a.p[7])[i] > (float)0.0) return dy;
        return a.u[5] == 1U ? (float)0.0 : a.f[1] * dy;
    }
}

// The statistic of value i: its channel (G = 0) or its (image, group).
static unsigned long dref_norm_stat_(unsigned long i, unsigned long C, unsigned long P, unsigned long G) {
    unsigned long c = (i / P) % C;
    return G == 0UL ? c : (i / (C * P)) * G + c / (C / G);
}

struct DrefWin_ {
    unsigned long h0; unsigned long h1; unsigned long w0; unsigned long w1; unsigned long padded;
};

static struct DrefWin_ dref_pool_win_(const &DevArgs a, unsigned long oh, unsigned long ow) {
    long H = (long)a.u[1]; long W = (long)a.u[2];
    long kH = (long)a.f[0]; long kW = (long)a.f[1]; long sH = (long)a.f[2]; long sW = (long)a.f[3];
    long pH = (long)a.f[4]; long pW = (long)a.f[5];
    long h0 = (long)oh * sH - pH; long w0 = (long)ow * sW - pW;
    long h1 = h0 + kH; long w1 = w0 + kW;
    long hp = h1 < H + pH ? h1 : H + pH;
    long wp = w1 < W + pW ? w1 : W + pW;
    struct DrefWin_ r;
    r.padded = (unsigned long)((hp - h0) * (wp - w0));
    r.h0 = h0 < 0L ? 0UL : (unsigned long)h0;
    r.w0 = w0 < 0L ? 0UL : (unsigned long)w0;
    r.h1 = h1 > H ? (unsigned long)H : (unsigned long)h1;
    r.w1 = w1 > W ? (unsigned long)W : (unsigned long)w1;
    return r;
}

struct DrefTaps_ { unsigned long idx[4]; float w[4]; unsigned long n; };

static float dref_rs_src_(unsigned long o, unsigned long in, unsigned long out, unsigned long align, int clampZero) {
    if (align != 0UL) return out <= 1UL ? (float)0.0 : (float)o * (float)(in - 1UL) / (float)(out - 1UL);
    float s = ((float)o + (float)0.5) * (float)in / (float)out - (float)0.5;
    if (clampZero && s < (float)0.0) s = (float)0.0;
    return s;
}

static struct DrefTaps_ dref_rs_taps_(unsigned long o, unsigned long in, unsigned long out, unsigned long mode,
                                      unsigned long align) {
    struct DrefTaps_ t;
    unsafe {
        unsigned long k = 0UL;
        while (k < 4UL) { t.idx[k] = 0UL; t.w[k] = (float)0.0; k = k + 1UL; }
        if (mode == 0UL) {
            unsigned long i = (unsigned long)floor_f((float)o * ((float)in / (float)out));
            t.idx[0] = i < in ? i : in - 1UL; t.w[0] = (float)1.0; t.n = 1UL;
        } else if (mode == 1UL) {
            float s = dref_rs_src_(o, in, out, align, 1);
            unsigned long i0 = (unsigned long)s;
            if (i0 > in - 1UL) i0 = in - 1UL;
            float l = s - (float)i0;
            t.idx[0] = i0; t.idx[1] = i0 + 1UL < in ? i0 + 1UL : in - 1UL;
            t.w[0] = (float)1.0 - l; t.w[1] = l; t.n = 2UL;
        } else {
            float s = dref_rs_src_(o, in, out, align, 0);
            float f = floor_f(s);
            float x = s - f;
            float A = (float)-0.75;
            float x1 = x + (float)1.0; float x2 = (float)1.0 - x; float x3 = (float)2.0 - x;
            t.w[0] = ((A * x1 - (float)5.0 * A) * x1 + (float)8.0 * A) * x1 - (float)4.0 * A;
            t.w[1] = ((A + (float)2.0) * x - (A + (float)3.0)) * x * x + (float)1.0;
            t.w[2] = ((A + (float)2.0) * x2 - (A + (float)3.0)) * x2 * x2 + (float)1.0;
            t.w[3] = ((A * x3 - (float)5.0 * A) * x3 + (float)8.0 * A) * x3 - (float)4.0 * A;
            long i = (long)f;
            k = 0UL;
            while (k < 4UL) {
                long j = i - 1L + (long)k;
                t.idx[k] = j < 0L ? 0UL : (j >= (long)in ? in - 1UL : (unsigned long)j);
                k = k + 1UL;
            }
            t.n = 4UL;
        }
    }
    return t;
}

// The normalization, pooling and resizing kernels (DK_BIAS_CHANNELS_BWD .. DK_RESIZE).
static int dref_spatial_(int kernel, unsigned long n, const &DevArgs a) {
    unsafe {
        float* P0 = (float*)a.p[0]; float* P1 = (float*)a.p[1]; float* P2 = (float*)a.p[2]; float* P3 = (float*)a.p[3];
        float* P4 = (float*)a.p[4]; float* P5 = (float*)a.p[5]; float* P6 = (float*)a.p[6];
        unsigned long U0 = (unsigned long)a.u[0]; unsigned long U1 = (unsigned long)a.u[1];
        unsigned long U2 = (unsigned long)a.u[2]; unsigned long U3 = (unsigned long)a.u[3];
        unsigned long U4 = (unsigned long)a.u[4]; unsigned long U5 = (unsigned long)a.u[5];
        unsigned long U6 = (unsigned long)a.u[6];
        unsigned long i = 0UL;
        if (kernel == DK_BIAS_CHANNELS_BWD) {
            unsigned long c = 0UL;
            while (c < U0) {
                double s = 0.0;
                unsigned long q = 0UL;
                while (q < U2 * U1) { s = s + (double)P0[((q / U1) * U0 + c) * U1 + q % U1]; q = q + 1UL; }
                P1[c] = P1[c] + (float)s;
                c = c + 1UL;
            }
            return 1;
        }
        if (kernel == DK_BN_PARTIAL) {
            double* part = (double*)a.p[1];
            unsigned long M = U2 * U1;
            unsigned long u = 0UL;
            while (u < U0 * U3) {
                unsigned long c = u / U3; unsigned long sl = u % U3;
                double s = 0.0; double cnt = 0.0;
                unsigned long q = 0UL;
                while (q < M) {
                    if ((q / 256UL) % U3 == sl) { s = s + (double)P0[((q / U1) * U0 + c) * U1 + q % U1]; cnt = cnt + 1.0; }
                    q = q + 1UL;
                }
                double mu = cnt > 0.0 ? s / cnt : 0.0;
                double sq = 0.0;
                q = 0UL;
                while (q < M) {
                    if ((q / 256UL) % U3 == sl) { double d = (double)P0[((q / U1) * U0 + c) * U1 + q % U1] - mu; sq = sq + d * d; }
                    q = q + 1UL;
                }
                part[u * 3UL] = cnt; part[u * 3UL + 1UL] = mu; part[u * 3UL + 2UL] = sq;
                u = u + 1UL;
            }
            return 1;
        }
        if (kernel == DK_BN_FINISH) {
            const double* part = (const double*)a.p[0];
            while (i < n) {
                float mean; float var;
                if (U4 != 0UL) {
                    double cn = 0.0; double mu = 0.0; double m2 = 0.0;
                    unsigned long sl = 0UL;
                    while (sl < U3) {
                        const double* p = &part[(i * U3 + sl) * 3UL];
                        if (p[0] != 0.0) {
                            double nn = cn + p[0]; double d = p[1] - mu;
                            mu = mu + d * p[0] / nn;
                            m2 = m2 + p[2] + d * d * cn * p[0] / nn;
                            cn = nn;
                        }
                        sl = sl + 1UL;
                    }
                    mean = (float)mu;
                    var = (float)(m2 / cn);
                    float unb = cn > 1.0 ? (float)(m2 / (cn - 1.0)) : var;
                    P3[i] = ((float)1.0 - a.f[0]) * P3[i] + a.f[0] * mean;
                    P4[i] = ((float)1.0 - a.f[0]) * P4[i] + a.f[0] * unb;
                } else {
                    mean = P3[i]; var = P4[i];
                }
                P1[i] = mean;
                P2[i] = (float)1.0 / sqrt_f(var + a.f[1]);
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_GN_STATS) {
            unsigned long u = 0UL;
            while (u < U1) {
                const float* x = (const float*)&P0[u * U0];
                double s = 0.0;
                unsigned long q = 0UL;
                while (q < U0) { s = s + (double)x[q]; q = q + 1UL; }
                double mu = s / (double)U0;
                double sq = 0.0;
                q = 0UL;
                while (q < U0) { double d = (double)x[q] - mu; sq = sq + d * d; q = q + 1UL; }
                P1[u] = (float)mu;
                P2[u] = (float)1.0 / sqrt_f((float)(sq / (double)U0) + a.f[0]);
                u = u + 1UL;
            }
            return 1;
        }
        if (kernel == DK_NORM_APPLY) {
            while (i < n) {
                unsigned long c = (i / U1) % U0;
                unsigned long st = dref_norm_stat_(i, U0, U1, U2);
                P1[i] = dref_fact_(a, P4[c] * (P0[i] - P2[st]) * P3[st] + P5[c]);
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_NORM_SUMS) {
            unsigned long c = 0UL;
            while (c < U0) {
                double sd = 0.0; double sx = 0.0;
                unsigned long q = 0UL;
                while (q < U3 * U1) {
                    unsigned long j = ((q / U1) * U0 + c) * U1 + q % U1;
                    unsigned long st = dref_norm_stat_(j, U0, U1, U2);
                    float dy = dref_fdy_(a, j);
                    sd = sd + (double)dy;
                    sx = sx + (double)(dy * (P1[j] - P2[st]) * P3[st]);
                    q = q + 1UL;
                }
                if (P4 != (float*)0) P4[c] = P4[c] + (float)sx;
                if (P5 != (float*)0) P5[c] = P5[c] + (float)sd;
                if (P6 != (float*)0) { P6[c] = P6[c] + (float)sd; P6[U0 + c] = P6[U0 + c] + (float)sx; }
                c = c + 1UL;
            }
            return 1;
        }
        if (kernel == DK_BN_BWD_DX) {
            while (i < n) {
                unsigned long c = (i / U1) % U0;
                float inv = P3[c];
                float m1 = U2 != 0UL ? P5[c] * a.f[0] : (float)0.0;
                float m2 = U2 != 0UL ? P5[U0 + c] * a.f[0] : (float)0.0;
                float xh = (P1[i] - P2[c]) * inv;
                P6[i] = (a.u[6] != 0U ? (float)0.0 : P6[i]) + P4[c] * inv * (dref_fdy_(a, i) - m1 - xh * m2);
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_GN_BWD) {
            unsigned long Cg = U0 / U2;
            unsigned long M = Cg * U1;
            unsigned long u = 0UL;
            while (u < U3) {
                unsigned long base = u * M;
                unsigned long g = u % U2;
                float mean = P2[u]; float inv = P3[u];
                double s1 = 0.0; double s2 = 0.0;
                unsigned long q = 0UL;
                while (q < M) {
                    float dxh = dref_fdy_(a, base + q) * P4[g * Cg + q / U1];
                    s1 = s1 + (double)dxh;
                    s2 = s2 + (double)(dxh * (P1[base + q] - mean) * inv);
                    q = q + 1UL;
                }
                float m1 = (float)(s1 / (double)M); float m2 = (float)(s2 / (double)M);
                q = 0UL;
                while (q < M) {
                    float dxh = dref_fdy_(a, base + q) * P4[g * Cg + q / U1];
                    float xh = (P1[base + q] - mean) * inv;
                    P5[base + q] = P5[base + q] + inv * (dxh - m1 - xh * m2);
                    q = q + 1UL;
                }
                u = u + 1UL;
            }
            return 1;
        }
        if (kernel == DK_POOL) {
            unsigned long H = U1; unsigned long W = U2; unsigned long oH = U3; unsigned long oW = U4;
            while (i < n) {
                unsigned long ow = i % oW; unsigned long oh = (i / oW) % oH; unsigned long pl = i / (oW * oH);
                struct DrefWin_ r = dref_pool_win_(a, oh, ow);
                const float* img = (const float*)&P0[pl * H * W];
                if (U5 == 0UL) {
                    unsigned long best = r.h0 * W + r.w0;
                    float bv = img[best];
                    unsigned long y = r.h0;
                    while (y < r.h1) {
                        unsigned long x = r.w0;
                        while (x < r.w1) { if (img[y * W + x] > bv) { bv = img[y * W + x]; best = y * W + x; } x = x + 1UL; }
                        y = y + 1UL;
                    }
                    if (U6 != 0UL) P2[pl * H * W + best] = P2[pl * H * W + best] + P1[i]; else P1[i] = bv;
                } else {
                    float div = U5 == 1UL ? (float)r.padded : (float)((r.h1 - r.h0) * (r.w1 - r.w0));
                    float acc = (float)0.0;
                    float share = U6 != 0UL ? P1[i] / div : (float)0.0;
                    unsigned long y = r.h0;
                    while (y < r.h1) {
                        unsigned long x = r.w0;
                        while (x < r.w1) {
                            if (U6 != 0UL) P2[pl * H * W + y * W + x] = P2[pl * H * W + y * W + x] + share;
                            else acc = acc + img[y * W + x];
                            x = x + 1UL;
                        }
                        y = y + 1UL;
                    }
                    if (U6 == 0UL) P1[i] = acc / div;
                }
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_POOL_BWD) {
            // the adjoint of DK_POOL, by scatter (u6 = 1: dx was an unwritten zero gradient)
            if (U6 != 0UL) { while (i < n) { P2[i] = (float)0.0; i = i + 1UL; } }
            unsigned long planes = n / (U1 * U2);
            struct DevArgs f = *a;
            f.u[6] = 1U;
            device_ref_launch_(DK_POOL, planes * U3 * U4, &f);
            return 1;
        }
        if (kernel == DK_GPOOL) {
            unsigned long planes = n / 256UL;
            unsigned long pl = 0UL;
            while (pl < planes) {
                const float* x = (const float*)&P0[pl * U0];
                if (U1 != 0UL) {
                    if (U2 != 0UL) {
                        float share = P1[pl] / (float)U0;
                        if (U6 != 0UL) { unsigned long z = 0UL; while (z < U0) { P2[pl * U0 + z] = (float)0.0; z = z + 1UL; } }
                        unsigned long q = 0UL;
                        while (q < U0) { P2[pl * U0 + q] = P2[pl * U0 + q] + share; q = q + 1UL; }
                    } else {
                        double s = 0.0;
                        unsigned long q = 0UL;
                        while (q < U0) { s = s + (double)x[q]; q = q + 1UL; }
                        P1[pl] = (float)(s / (double)U0);
                    }
                } else {
                    unsigned long best = 0UL;
                    unsigned long q = 1UL;
                    while (q < U0) { if (x[q] > x[best]) best = q; q = q + 1UL; }
                    if (U2 != 0UL) P2[pl * U0 + best] = P2[pl * U0 + best] + P1[pl]; else P1[pl] = x[best];
                }
                pl = pl + 1UL;
            }
            return 1;
        }
        if (kernel == DK_UPSAMPLE) {
            unsigned long W = U0; unsigned long H = U1; unsigned long f = U2;
            unsigned long oW = W * f; unsigned long oH = H * f;
            while (i < n) {
                if (U3 == 0UL) {
                    unsigned long ow = i % oW; unsigned long oh = (i / oW) % oH; unsigned long pl = i / (oW * oH);
                    P1[i] = P0[(pl * H + oh / f) * W + ow / f];
                } else {
                    unsigned long w = i % W; unsigned long h = (i / W) % H; unsigned long pl = i / (W * H);
                    float s = (float)0.0;
                    unsigned long y = 0UL;
                    while (y < f) {
                        unsigned long x = 0UL;
                        while (x < f) { s = s + P0[(pl * oH + h * f + y) * oW + w * f + x]; x = x + 1UL; }
                        y = y + 1UL;
                    }
                    P1[i] = (U6 != 0UL ? (float)0.0 : P1[i]) + s;
                }
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_RESIZE) {
            unsigned long H = U0; unsigned long W = U1; unsigned long oH = U2; unsigned long oW = U3;
            while (i < n) {
                unsigned long ow = i % oW; unsigned long oh = (i / oW) % oH; unsigned long pl = i / (oW * oH);
                struct DrefTaps_ ty = dref_rs_taps_(oh, H, oH, U4, U5);
                struct DrefTaps_ tx = dref_rs_taps_(ow, W, oW, U4, U5);
                unsigned long base = pl * H * W;
                float acc = (float)0.0;
                unsigned long y = 0UL;
                while (y < ty.n) {
                    float rv = (float)0.0;
                    unsigned long x = 0UL;
                    while (x < tx.n) { rv = rv + tx.w[x] * P0[base + ty.idx[y] * W + tx.idx[x]]; x = x + 1UL; }
                    acc = acc + ty.w[y] * rv;
                    y = y + 1UL;
                }
                P1[i] = acc;
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_RESIZE_BWD) {
            // one axis's adjoint: pass 0 along x into t (overwritten), pass 1 along y into dx
            unsigned long H = U0; unsigned long W = U1; unsigned long oH = U2; unsigned long oW = U3;
            int pass = a.f[0] != (float)0.0;
            if (pass && U6 != 0UL) { while (i < n) { P1[i] = (float)0.0; i = i + 1UL; } i = 0UL; }
            unsigned long rows = pass ? n / (H * W) * oH : n / W;      // source rows: planes*oH either way
            unsigned long inLen = pass ? H : W;
            unsigned long outLen = pass ? oH : oW;
            if (!pass) { while (i < n) { P1[i] = (float)0.0; i = i + 1UL; } }
            unsigned long r = 0UL;
            while (r < rows) {
                unsigned long o = pass ? r % oH : 0UL;
                unsigned long pl = pass ? r / oH : r;
                unsigned long cnt = pass ? W : oW;
                unsigned long q = 0UL;
                while (q < cnt) {
                    if (pass) {
                        // t[pl, o, q] spreads over dx[pl, taps(o), q]
                        struct DrefTaps_ t = dref_rs_taps_(o, inLen, outLen, U4, U5);
                        float v = P0[(pl * oH + o) * W + q];
                        unsigned long k = 0UL;
                        while (k < t.n) { unsigned long at = (pl * H + t.idx[k]) * W + q; P1[at] = P1[at] + t.w[k] * v; k = k + 1UL; }
                    } else {
                        // dout[r, q] spreads over t[r, taps(q)]
                        struct DrefTaps_ t = dref_rs_taps_(q, inLen, outLen, U4, U5);
                        float v = P0[r * oW + q];
                        unsigned long k = 0UL;
                        while (k < t.n) { unsigned long at = r * W + t.idx[k]; P1[at] = P1[at] + t.w[k] * v; k = k + 1UL; }
                    }
                    q = q + 1UL;
                }
                r = r + 1UL;
            }
            return 1;
        }
    }
    return 0;
}

// attention.h's masks: is key j visible from position pos (flags: 1 causal, 2 window w)?
static int dref_attn_vis_(unsigned int flags, unsigned int w, long pos, long j) {
    if ((flags & 1U) != 0U && j > pos) return 0;
    if ((flags & 2U) != 0U && j <= pos - (long)w) return 0;
    return 1;
}

// DK_SOFTMAX_ROWS with a mask: u2 flags, u3 window, u4 (int) keys - queries.
static void dref_softmax_mask_(const &DevArgs a, unsigned long n) {
    unsafe {
        const float* X = (const float*)a.p[0];
        float* O = (float*)a.p[1];
        unsigned long C = (unsigned long)a.u[0];
        unsigned long r = 0UL;
        while (r < n) {
            long pos = (long)r + (long)(int)a.u[4];
            const float* x = &X[r * C];
            float* o = &O[r * C];
            float mx = (float)-3.0e38;
            unsigned long c = 0UL;
            while (c < C) { if (dref_attn_vis_(a.u[2], a.u[3], pos, (long)c) && x[c] > mx) mx = x[c]; c = c + 1UL; }
            float s = (float)0.0;
            c = 0UL;
            while (c < C) { if (dref_attn_vis_(a.u[2], a.u[3], pos, (long)c)) s = s + exp_f(x[c] - mx); c = c + 1UL; }
            float inv = s > (float)0.0 ? (float)1.0 / s : (float)0.0;
            c = 0UL;
            while (c < C) { o[c] = dref_attn_vis_(a.u[2], a.u[3], pos, (long)c) ? exp_f(x[c] - mx) * inv : (float)0.0; c = c + 1UL; }
            r = r + 1UL;
        }
    }
}

// The fused attention kernels (DK_ATTN_FWD, DK_ATTN_BWD_Q, DK_ATTN_BWD_KV;
// device_kernels.cu has the arguments): one (head, row) at a time, whatever
// the launch's thread count (the GPU kernels' is a tiling's).
#define DREF_ATTN_MAXD 128UL
static int dref_attn_(int kernel, unsigned long n, const &DevArgs a) {
    unsafe {
        const float* Q = (const float*)a.p[0]; const float* K = (const float*)a.p[1]; const float* V = (const float*)a.p[2];
        float* P3 = (float*)a.p[3]; float* LSE = (float*)a.p[4]; const float* G = (const float*)a.p[5];
        float* P6 = (float*)a.p[6]; float* DS = (float*)a.p[7];
        unsigned long S = (unsigned long)a.u[0]; unsigned long L = (unsigned long)a.u[1]; unsigned long H = (unsigned long)a.u[2];
        unsigned long D = (unsigned long)a.u[3]; unsigned long E = (unsigned long)a.u[4];
        unsigned int fl = a.u[5] & 3U; unsigned int w = a.u[6];
        float sc = a.f[0];
        long off = (long)L - (long)S;
        float acc[DREF_ATTN_MAXD];
        float acc2[DREF_ATTN_MAXD];
        unsigned long rowsAll = H * (kernel == DK_ATTN_BWD_KV ? L : S);
        unsigned long t = 0UL;
        while (t < rowsAll) {
            if (kernel == DK_ATTN_BWD_KV) {
                unsigned long h = t / L; unsigned long j = t % L;
                const float* kr = &K[(j * H + h) * D];
                const float* vr = &V[(j * H + h) * E];
                unsigned long d = 0UL;
                while (d < D) { acc[d] = (float)0.0; d = d + 1UL; }
                d = 0UL;
                while (d < E) { acc2[d] = (float)0.0; d = d + 1UL; }
                unsigned long i = 0UL;
                while (i < S) {
                    if (dref_attn_vis_(fl, w, (long)i + off, (long)j)) {
                        const float* qr = &Q[(i * H + h) * D];
                        const float* gr = &G[(i * H + h) * E];
                        float s = (float)0.0;
                        d = 0UL;
                        while (d < D) { s = s + qr[d] * kr[d]; d = d + 1UL; }
                        float p = exp_f(s * sc - LSE[h * S + i]);
                        float dp = (float)0.0;
                        d = 0UL;
                        while (d < E) { acc2[d] = acc2[d] + p * gr[d]; dp = dp + gr[d] * vr[d]; d = d + 1UL; }
                        float ds = p * (dp - DS[h * S + i]);
                        d = 0UL;
                        while (d < D) { acc[d] = acc[d] + ds * qr[d]; d = d + 1UL; }
                    }
                    i = i + 1UL;
                }
                if (P3 != (float*)0) {
                    float* o = &P3[(j * H + h) * D];
                    d = 0UL;
                    while (d < D) { o[d] = ((a.u[5] & 8U) != 0U ? (float)0.0 : o[d]) + sc * acc[d]; d = d + 1UL; }
                }
                if (P6 != (float*)0) {
                    float* o = &P6[(j * H + h) * E];
                    d = 0UL;
                    while (d < E) { o[d] = ((a.u[5] & 16U) != 0U ? (float)0.0 : o[d]) + acc2[d]; d = d + 1UL; }
                }
                t = t + 1UL;
                continue;
            }
            unsigned long h = t / S; unsigned long i = t % S;
            const float* qr = &Q[(i * H + h) * D];
            long pos = (long)i + off;
            unsigned long d = 0UL;
            if (kernel == DK_ATTN_FWD) {
                while (d < E) { acc[d] = (float)0.0; d = d + 1UL; }
                float m = (float)-3.0e38;
                float l = (float)0.0;
                unsigned long j = 0UL;
                while (j < L) {
                    if (dref_attn_vis_(fl, w, pos, (long)j)) {
                        const float* kr = &K[(j * H + h) * D];
                        const float* vr = &V[(j * H + h) * E];
                        float s = (float)0.0;
                        d = 0UL;
                        while (d < D) { s = s + qr[d] * kr[d]; d = d + 1UL; }
                        s = s * sc;
                        float mn = m > s ? m : s;
                        float c = exp_f(m - mn);
                        float p = exp_f(s - mn);
                        l = l * c + p;
                        d = 0UL;
                        while (d < E) { acc[d] = acc[d] * c + p * vr[d]; d = d + 1UL; }
                        m = mn;
                    }
                    j = j + 1UL;
                }
                float inv = l > (float)0.0 ? (float)1.0 / l : (float)0.0;
                float* o = &P3[(i * H + h) * E];
                d = 0UL;
                while (d < E) { o[d] = acc[d] * inv; d = d + 1UL; }
                LSE[t] = l > (float)0.0 ? m + log_f(l) : (float)3.0e38;
            } else {
                const float* orow = (const float*)&P3[(i * H + h) * E];
                const float* gr = &G[(i * H + h) * E];
                float Di = (float)0.0;
                while (d < E) { Di = Di + gr[d] * orow[d]; d = d + 1UL; }
                DS[t] = Di;
                if (P6 != (float*)0) {
                    d = 0UL;
                    while (d < D) { acc[d] = (float)0.0; d = d + 1UL; }
                    unsigned long j = 0UL;
                    while (j < L) {
                        if (dref_attn_vis_(fl, w, pos, (long)j)) {
                            const float* kr = &K[(j * H + h) * D];
                            const float* vr = &V[(j * H + h) * E];
                            float s = (float)0.0;
                            d = 0UL;
                            while (d < D) { s = s + qr[d] * kr[d]; d = d + 1UL; }
                            float p = exp_f(s * sc - LSE[t]);
                            float dp = (float)0.0;
                            d = 0UL;
                            while (d < E) { dp = dp + gr[d] * vr[d]; d = d + 1UL; }
                            float ds = p * (dp - Di);
                            d = 0UL;
                            while (d < D) { acc[d] = acc[d] + ds * kr[d]; d = d + 1UL; }
                        }
                        j = j + 1UL;
                    }
                    float* o = &P6[(i * H + h) * D];
                    d = 0UL;
                    while (d < D) { o[d] = ((a.u[5] & 4U) != 0U ? (float)0.0 : o[d]) + sc * acc[d]; d = d + 1UL; }
                }
            }
            t = t + 1UL;
        }
        return 1;
    }
}

// The CPU version of one DK_* kernel over n threads; 0 for an unknown kernel.
int device_ref_launch_(int kernel, unsigned long n, const &DevArgs a) {
    unsafe {
        float* P0 = (float*)a.p[0]; float* P1 = (float*)a.p[1]; float* P2 = (float*)a.p[2]; float* P3 = (float*)a.p[3];
        float* P4 = (float*)a.p[4]; float* P5 = (float*)a.p[5]; float* P6 = (float*)a.p[6];
        unsigned long U0 = (unsigned long)a.u[0]; unsigned long U1 = (unsigned long)a.u[1];
        unsigned long U2 = (unsigned long)a.u[2]; unsigned long U3 = (unsigned long)a.u[3];
        unsigned long i = 0UL;
        if (kernel == DK_EW || (kernel >= DK_EWOP_0 && kernel <= DK_EWOP_51)) { dref_ew_(a, n); return 1; }
        if (kernel == DK_SUM) {
            float acc = (float)0.0;
            while (i < n) { float v = P0[i]; acc = acc + (a.u[0] != 0U ? v * v : v); i = i + 1UL; }
            P1[0] = P1[0] + acc;
            return 1;
        }
        if (kernel == DK_BIAS_ROWS) { while (i < n) { P2[i] = P0[i] + P1[i % U0]; i = i + 1UL; } return 1; }
        if (kernel == DK_MUL_ROW) { while (i < n) { P2[i] = P0[i] * P1[i % U0]; i = i + 1UL; } return 1; }
        if (kernel == DK_COLSUM_ACC) {
            while (i < U0) {
                float s = (float)0.0;
                unsigned long r = 0UL;
                while (r < U1) { s = s + P0[r * U0 + i] * (P2 != (float*)0 ? P2[r * U0 + i] : (float)1.0); r = r + 1UL; }
                P1[i] = P1[i] + s;
                i = i + 1UL;
            }
            return 1;
        }
        int fresh = a.u[6] != 0U;   // write an unwritten zero gradient instead of adding to it
        if (kernel == DK_MUL_ROW_ACC) { while (i < n) { P2[i] = (fresh ? (float)0.0 : P2[i]) + P0[i] * P1[i % U0]; i = i + 1UL; } return 1; }
        if (kernel == DK_ROWSUM) {
            while (i < n) {
                float s = (float)0.0;
                unsigned long c = 0UL;
                while (c < U0) { s = s + P0[i * U0 + c]; c = c + 1UL; }
                P1[i] = s;
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_ROWBCAST_ACC) { while (i < n) { P1[i] = (fresh ? (float)0.0 : P1[i]) + P0[i / U0]; i = i + 1UL; } return 1; }
        if (kernel == DK_CONCAT) {
            unsigned long W = U0 + U1;
            while (i < n) {
                unsigned long r = i / W; unsigned long c = i % W;
                P2[i] = c < U0 ? P0[r * U0 + c] : P1[r * U1 + c - U0];
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_SLICE) {
            while (i < n) {
                unsigned long r = i / U2; unsigned long c = i % U2;
                if (U3 == 1UL) P1[r * U0 + U1 + c] = P1[r * U0 + U1 + c] + P0[i];
                else if (U3 == 2UL) P1[i] = (a.u[6] != 0U ? (float)0.0 : P1[i]) + P0[r * U0 + U1 + c];
                else P1[i] = P0[r * U0 + U1 + c];
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_GATHER_ROWS) {
            while (i < n) { unsigned long r = i / U0; unsigned long c = i % U0; P2[i] = P0[(unsigned long)P1[r] * U0 + c]; i = i + 1UL; }
            return 1;
        }
        if (kernel == DK_SCATTER_ROWS) {
            while (i < n) {
                unsigned long r = i / U0; unsigned long c = i % U0;
                unsigned long k = (unsigned long)P1[r] * U0 + c;
                P2[k] = P2[k] + P0[i];
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_SOFTMAX_ROWS) {
            if (a.u[2] != 0U) { dref_softmax_mask_(a, n); return 1; }
            while (i < n) {
                const float* x = (const float*)&P0[i * U0];
                float* o = &P1[i * U0];
                float l = dref_lse_(x, U0);
                unsigned long c = 0UL;
                while (c < U0) { o[c] = U1 != 0UL ? x[c] - l : exp_f(x[c] - l); c = c + 1UL; }
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_SOFTMAX_BWD) {
            while (i < n) {
                const float* g = (const float*)&P0[i * U0];
                const float* y = (const float*)&P1[i * U0];
                float* dx = &P2[i * U0];
                float s = (float)0.0;
                unsigned long c = 0UL;
                if (U1 != 0UL) {
                    while (c < U0) { s = s + g[c]; c = c + 1UL; }
                    c = 0UL;
                    while (c < U0) { dx[c] = dx[c] + g[c] - exp_f(y[c]) * s; c = c + 1UL; }
                } else {
                    while (c < U0) { s = s + g[c] * y[c]; c = c + 1UL; }
                    c = 0UL;
                    while (c < U0) { dx[c] = dx[c] + y[c] * (g[c] - s); c = c + 1UL; }
                }
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_CE_FWD) {
            float acc = (float)0.0;
            while (i < n) {
                const float* x = (const float*)&P0[i * U0];
                acc = acc + (dref_lse_(x, U0) - x[(unsigned long)P1[i]]) * a.f[0];
                i = i + 1UL;
            }
            P2[0] = P2[0] + acc;
            return 1;
        }
        if (kernel == DK_CE_BWD) {
            float sc = P3[0] * a.f[0];
            while (i < n) {
                const float* x = (const float*)&P0[i * U0];
                float* d = &P2[i * U0];
                float l = dref_lse_(x, U0);
                unsigned long lab = (unsigned long)P1[i];
                unsigned long c = 0UL;
                while (c < U0) { d[c] = (fresh ? (float)0.0 : d[c]) + sc * (exp_f(x[c] - l) - (c == lab ? (float)1.0 : (float)0.0)); c = c + 1UL; }
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_EWLOSS_FWD) {
            float acc = (float)0.0;
            while (i < n) {
                float x = P0[i]; float t = P1[i]; float v = (float)0.0;
                if (U0 == 0UL) v = dref_max_(x, (float)0.0) - x * t + log_f((float)1.0 + exp_f(-dref_abs_(x)));
                else { float d = x - t; float ad = dref_abs_(d); float k = a.f[1]; v = ad <= k ? (float)0.5 * d * d : k * (ad - (float)0.5 * k); }
                acc = acc + v * a.f[0];
                i = i + 1UL;
            }
            P2[0] = P2[0] + acc;
            return 1;
        }
        if (kernel == DK_EWLOSS_BWD) {
            float sc = P3[0] * a.f[0];
            while (i < n) {
                float x = P0[i]; float t = P1[i]; float g = (float)0.0;
                if (U0 == 0UL) g = dref_sig_(x) - t;
                else { float d = x - t; float k = a.f[1]; g = dref_min_(dref_max_(d, -k), k); }
                P2[i] = (fresh ? (float)0.0 : P2[i]) + sc * g;
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_LAYERNORM) {
            while (i < n) {
                const float* x = (const float*)&P0[i * U0];
                float* o = &P1[i * U0];
                float m = (float)0.0;
                unsigned long c = 0UL;
                while (c < U0) { m = m + x[c]; c = c + 1UL; }
                m = m / (float)U0;
                float v = (float)0.0;
                c = 0UL;
                while (c < U0) { float d = x[c] - m; v = v + d * d; c = c + 1UL; }
                float is = (float)1.0 / sqrt_f(v / (float)U0 + a.f[0]);
                P2[i] = is;
                c = 0UL;
                while (c < U0) { o[c] = (x[c] - m) * is; c = c + 1UL; }
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_LAYERNORM_BWD) {
            while (i < n) {
                const float* g = (const float*)&P0[i * U0];
                const float* y = (const float*)&P1[i * U0];
                float* dx = &P3[i * U0];
                float mg = (float)0.0; float mgy = (float)0.0;
                unsigned long c = 0UL;
                while (c < U0) { mg = mg + g[c]; mgy = mgy + g[c] * y[c]; c = c + 1UL; }
                mg = mg / (float)U0; mgy = mgy / (float)U0;
                float is = P2[i];
                c = 0UL;
                while (c < U0) { dx[c] = dx[c] + is * (g[c] - mg - y[c] * mgy); c = c + 1UL; }
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_OPT) {
            if (P5 != (float*)0 && P5[2] != (float)0.0) return 1;   // an overflowed fp16 step
            int kind = (int)a.u[0];
            float lr = a.f[0]; float wd = a.f[1]; float mu = a.f[2]; float b1 = a.f[3]; float b2 = a.f[4]; float eps = a.f[5];
            float cs = P4 != (float*)0 ? P4[0] : (float)1.0;
            while (i < n) {
                float g0 = P1[i] * cs;
                float wi = P0[i];
                if (kind == 0) P0[i] = wi - lr * (g0 + wd * wi);
                else if (kind == 1 || kind == 2) {
                    float g = g0 + wd * wi; float mi = mu * P2[i] + g; P2[i] = mi;
                    P0[i] = wi - lr * (kind == 2 ? g + mu * mi : mi);
                } else if (kind == 6) {
                    float g = g0 + wd * wi; float vi = P3[i] + g * g; P3[i] = vi;
                    P0[i] = wi - lr * g / (sqrt_f(vi) + eps);
                } else if (kind == 7) {
                    float c = b1 * P2[i] + ((float)1.0 - b1) * g0;
                    float sg = c > (float)0.0 ? (float)1.0 : (c < (float)0.0 ? (float)-1.0 : (float)0.0);
                    P0[i] = wi - lr * (sg + wd * wi);
                    P2[i] = b2 * P2[i] + ((float)1.0 - b2) * g0;
                } else if (kind == 3) {
                    float g = g0 + wd * wi; float vi = b2 * P3[i] + ((float)1.0 - b2) * g * g; P3[i] = vi;
                    P0[i] = wi - lr * g / (sqrt_f(vi) + eps);
                } else {
                    float l2 = kind == 5 ? (float)0.0 : wd;
                    float dec = kind == 5 ? wd : (float)0.0;
                    float g = g0 + l2 * wi;
                    float mi = b1 * P2[i] + ((float)1.0 - b1) * g;
                    float vi = b2 * P3[i] + ((float)1.0 - b2) * g * g;
                    P2[i] = mi; P3[i] = vi;
                    P0[i] = wi - lr * ((mi * a.f[6]) / (sqrt_f(vi) * a.f[7] + eps) + dec * wi);
                }
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_OPT_MULTI) {
            const unsigned long* T = (const unsigned long*)a.p[0];
            unsigned long e = 0UL;
            while (e < U1) {
                struct DevArgs one = *a;
                one.p[0] = T[e * 6UL]; one.p[1] = T[e * 6UL + 1UL]; one.p[2] = T[e * 6UL + 2UL]; one.p[3] = T[e * 6UL + 3UL];
                one.p[4] = 0UL;
                device_ref_launch_(DK_OPT, T[e * 6UL + 4UL], &one);
                e = e + 1UL;
            }
            return 1;
        }
        if (kernel == DK_AMP_UNSCALE) {
            float inv = P1[1];
            while (i < n) {
                float v = P0[i] * inv;
                if (v != v || v - v != (float)0.0) P1[2] = (float)1.0;   // NaN or infinite
                P0[i] = v;
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_AMP_UNSCALE_MULTI) {
            const unsigned long* T = (const unsigned long*)a.p[0];
            unsigned long e = 0UL;
            while (e < U1) {
                struct DevArgs one = *a;
                one.p[0] = T[e * 3UL];
                device_ref_launch_(DK_AMP_UNSCALE, T[e * 3UL + 1UL], &one);
                e = e + 1UL;
            }
            return 1;
        }
        if (kernel == DK_AMP_UPDATE) {
            if (P0[2] != (float)0.0) { P0[0] = dref_max_(P0[0] * (float)0.5, (float)1.0); P0[3] = (float)0.0; }
            else {
                P0[3] = P0[3] + (float)1.0;
                if (P0[3] >= (float)a.u[0]) { P0[0] = dref_min_(P0[0] * (float)2.0, (float)16777216.0); P0[3] = (float)0.0; }
            }
            P0[1] = (float)1.0 / P0[0];
            P0[2] = (float)0.0;
            return 1;
        }
        if (kernel == DK_CLIP_SCALE) {
            float nrm = sqrt_f(P0[0]);
            P1[0] = (nrm > a.f[0] && nrm > (float)0.0) ? a.f[0] / nrm : (float)1.0;
            return 1;
        }
        if (kernel == DK_IM2COL || kernel == DK_COL2IM) {
            unsigned long C = U0; unsigned long H = U1; unsigned long W = U2; unsigned long KH = U3;
            unsigned long KW = (unsigned long)a.u[4]; unsigned long OH = (unsigned long)a.u[5]; unsigned long OW = (unsigned long)a.u[6];
            unsigned long B = (unsigned long)a.f[6];
            long sh = (long)a.f[0]; long sw = (long)a.f[1]; long ph = (long)a.f[2]; long pw = (long)a.f[3];
            long dh = (long)a.f[4]; long dw = (long)a.f[5];
            unsigned long OP = OH * OW;
            unsigned long BP = B * OP;
            while (i < n) {
                unsigned long q = i % BP; unsigned long r = i / BP;
                unsigned long b = q / OP; unsigned long p = q % OP; unsigned long oh = p / OW; unsigned long ow = p % OW;
                unsigned long kw = r % KW; unsigned long kh = (r / KW) % KH; unsigned long c = r / (KW * KH);
                long y = (long)oh * sh - ph + (long)kh * dh;
                long x = (long)ow * sw - pw + (long)kw * dw;
                int inside = y >= 0L && y < (long)H && x >= 0L && x < (long)W;
                if (kernel == DK_IM2COL) {
                    P1[i] = inside ? P0[((b * C + c) * H + (unsigned long)y) * W + (unsigned long)x] : (float)0.0;
                } else if (inside) {
                    unsigned long k = ((b * C + c) * H + (unsigned long)y) * W + (unsigned long)x;
                    P1[k] = P1[k] + P0[i];
                }
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_SWAP01) {
            while (i < n) {
                unsigned long p = i % U2; unsigned long t = i / U2; unsigned long bi = t % U1; unsigned long ai = t / U1;
                unsigned long o = (bi * U0 + ai) * U2 + p;
                if (U3 != 0UL) P1[o] = P1[o] + P0[i]; else P1[o] = P0[i];
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_BIAS_CHANNELS) { while (i < n) { P0[i] = P0[i] + P1[(i / U1) % U0]; i = i + 1UL; } return 1; }
        if (kernel >= DK_BIAS_CHANNELS_BWD && kernel <= DK_RESIZE_BWD) return dref_spatial_(kernel, n, a);
        if (kernel == DK_TRANSPOSE) {
            while (i < n) {
                unsigned long r = i / U1; unsigned long c = i % U1;
                unsigned long k = c * U0 + r;
                if (U2 != 0UL) P1[k] = (a.u[6] != 0U ? (float)0.0 : P1[k]) + P0[i];
                else P1[k] = P0[i];
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_ATTN_FWD || kernel == DK_ATTN_BWD_Q || kernel == DK_ATTN_BWD_KV) return dref_attn_(kernel, n, a);
        if (kernel == DK_SPMM) {
            while (i < n) {
                unsigned long r = i / U0; unsigned long j = i % U0;
                unsigned long lo = (unsigned long)P0[r]; unsigned long hi = (unsigned long)P0[r + 1UL];
                float s = (float)0.0;
                unsigned long e = lo;
                while (e < hi) { s = s + P2[e] * P3[(unsigned long)P1[e] * U0 + j]; e = e + 1UL; }
                if (U1 != 0UL) P4[i] = P4[i] + s; else P4[i] = s;
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_SPMM_TGRAD) {
            while (i < n) {
                unsigned long e = i / U0; unsigned long j = i % U0;
                unsigned long r = (unsigned long)P5[e]; unsigned long col = (unsigned long)P1[e];
                P4[col * U0 + j] = P4[col * U0 + j] + P2[e] * P3[r * U0 + j];
                i = i + 1UL;
            }
            return 1;
        }
        if (kernel == DK_SPMM_VGRAD) {
            while (i < n) {
                unsigned long r = (unsigned long)P5[i]; unsigned long col = (unsigned long)P1[i];
                float s = (float)0.0;
                unsigned long j = 0UL;
                while (j < U0) { s = s + P4[r * U0 + j] * P3[col * U0 + j]; j = j + 1UL; }
                P6[i] = P6[i] + s;
                i = i + 1UL;
            }
            return 1;
        }
    }
    return 0;
}

// op(x) of 16-bit values as fp32 (rows x cols of the stored matrix, ld)
static float* dref_widen_(unsigned long src, int dtype, unsigned long rows, unsigned long ld) {
    unsafe {
        float* w = (float*)malloc(sizeof(float) * rows * ld + 4UL);
        const unsigned short* h = (const unsigned short*)src;
        unsigned long i = 0UL;
        while (i < rows * ld) { w[i] = dtype == DEVICE_FP16 ? fp16_to_f32(h[i]) : bf16_to_f32(h[i]); i = i + 1UL; }
        return w;
    }
}

int device_ref_gemm_batched_(const &DevGemm g) {
    unsigned long i = 0UL;
    if (g.ctype != DEVICE_FP32) return 0;
    if (g.dtype != DEVICE_FP32) {
        // widened to fp32 (the stored extents: a is m x k or k x m, b k x n or n x k)
        while (i < g.batch) {
            float* a = dref_widen_(g.a + 2UL * i * g.sa, g.dtype, g.ta ? g.k : g.m, g.lda);
            float* b = dref_widen_(g.b + 2UL * i * g.sb, g.dtype, g.tb ? g.n : g.k, g.ldb);
            unsafe {
                device_ref_gemm_(g.ta, g.tb, g.m, g.n, g.k, g.alpha, (unsigned long)a, g.lda, (unsigned long)b, g.ldb,
                                 g.beta, g.c + 4UL * i * g.sc, g.ldc);
                free((void*)a); free((void*)b);
            }
            i = i + 1UL;
        }
        return 1;
    }
    while (i < g.batch) {
        device_ref_gemm_(g.ta, g.tb, g.m, g.n, g.k, g.alpha, g.a + 4UL * i * g.sa, g.lda, g.b + 4UL * i * g.sb, g.ldb,
                         g.beta, g.c + 4UL * i * g.sc, g.ldc);
        i = i + 1UL;
    }
    return 1;
}

// sgemm on host addresses, with the device GEMM's arguments.
int device_ref_gemm_(int ta, int tb, unsigned long m, unsigned long n, unsigned long k, float alpha,
                     unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb,
                     float beta, unsigned long c, unsigned long ldc) {
    unsafe {
        sgemm_n_(ta ? SGEMM_TRANS : SGEMM_NO_TRANS, tb ? SGEMM_TRANS : SGEMM_NO_TRANS, m, n, k, alpha,
              (const &float)(const float*)a, lda, (const &float)(const float*)b, ldb, beta, (&float)(float*)c, ldc);
    }
    return 1;
}

} // namespace std
