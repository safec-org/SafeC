// SafeC Standard Library — FFT on AMD GPUs (see fft_rocm.h).
//
// hipFFT plans are created with hipfftPlan1d / hipfftPlan2d (each plan
// allocates its own work area) and bound to gpu_rocm's stream. On AMD a
// hipfftHandle is a pointer (rocm_ffi.h's HipfftPlan_).
// safeguard: link linux -L${ROCM_PATH:-/opt/rocm}/lib -lhipfft -lamdhip64
#pragma once
#include <std/stddef.h>
#include <std/dsp/fft_rocm.h>
#include <std/dsp/complex_dsp.h>
#include <std/ml/gpu_rocm.h>
#include <std/ml/rocm_ffi.h>
#include <std/mem.h>

namespace std {

// ── hipFFT C ABI (hipfft/hipfft.h) ───────────────────────────────────────────
extern int hipfftPlan1d(HipfftPlan_** plan, int nx, int type, int batch);
extern int hipfftPlan2d(HipfftPlan_** plan, int nx, int ny, int type);
extern int hipfftSetStream(HipfftPlan_* plan, RocmNativeStream* stream);
extern int hipfftDestroy(HipfftPlan_* plan);
extern int hipfftExecC2C(HipfftPlan_* plan, HipDevPtr_* idata, HipDevPtr_* odata, int direction);
extern int hipfftExecR2C(HipfftPlan_* plan, HipDevPtr_* idata, HipDevPtr_* odata);
extern int hipfftExecC2R(HipfftPlan_* plan, HipDevPtr_* idata, HipDevPtr_* odata);
extern int hipfftExecZ2Z(HipfftPlan_* plan, HipDevPtr_* idata, HipDevPtr_* odata, int direction);

// The HIP runtime is declared in std/ml/rocm_ffi.h.

#define RFFT_R2C      42     // HIPFFT_R2C (0x2a)
#define RFFT_C2R      44     // HIPFFT_C2R (0x2c)
#define RFFT_C2C      41     // HIPFFT_C2C (0x29)
#define RFFT_Z2Z      105    // HIPFFT_Z2Z (0x69)
#define RFFT_FORWARD  -1     // HIPFFT_FORWARD
#define RFFT_INVERSE  1      // HIPFFT_BACKWARD
#define RFFT_H2D      1      // hipMemcpyHostToDevice
#define RFFT_D2H      2      // hipMemcpyDeviceToHost
#define RFFT_PLANS    16

// Plan cache: (type, nx, ny, batch) -> hipfftHandle; ny == 0 for 1D plans.
static int g_rfft_used[RFFT_PLANS];
static int g_rfft_type[RFFT_PLANS];
static unsigned long g_rfft_nx[RFFT_PLANS];
static unsigned long g_rfft_ny[RFFT_PLANS];
static unsigned long g_rfft_batch[RFFT_PLANS];
static HipfftPlan_* g_rfft_plan[RFFT_PLANS];
static int g_rfft_evict = 0;
static int g_rfft_init = 0;

// Device scratch for the host-array functions; grows as needed.
static HipDevPtr_* g_rfft_scratch = null;
static unsigned long g_rfft_scratch_bytes = 0UL;

static int rfft_dim_ok_(unsigned long v) { return v > 0UL && v <= 2147483647UL; }

static RocmNativeStream* rfft_stream_() {
    ?&RocmNativeStream s = rocm_stream_handle();
    unsafe { return (RocmNativeStream*)s; }
}

// Returns a hipFFT plan on gpu_rocm's stream, or null.
static HipfftPlan_* rfft_plan_(int type, unsigned long nx, unsigned long ny, unsigned long batch) {
    if (!rocm_available() || !rfft_dim_ok_(nx) || !rfft_dim_ok_(batch) ||
        (ny != 0UL && (!rfft_dim_ok_(ny) || batch != 1UL))) return null;
    if (!g_rfft_init) {
        int i = 0;
        while (i < RFFT_PLANS) { g_rfft_used[i] = 0; i = i + 1; }
        g_rfft_init = 1;
    }
    int slot = -1;
    int s = 0;
    while (s < RFFT_PLANS) {
        if (g_rfft_used[s] != 0) {
            if (g_rfft_type[s] == type && g_rfft_nx[s] == nx && g_rfft_ny[s] == ny && g_rfft_batch[s] == batch)
                return g_rfft_plan[s];
        } else if (slot < 0) {
            slot = s;
        }
        s = s + 1;
    }
    unsafe {
        RocmNativeStream* stream = rfft_stream_();
        if (slot < 0) {
            slot = g_rfft_evict;
            g_rfft_evict = (g_rfft_evict + 1) % RFFT_PLANS;
            hipStreamSynchronize(stream);   // queued work may still use the plan
            hipfftDestroy(g_rfft_plan[slot]);
            g_rfft_used[slot] = 0;
        }
        HipfftPlan_* plan = null;
        int r = ny == 0UL ? hipfftPlan1d(&plan, (int)nx, type, (int)batch)
                          : hipfftPlan2d(&plan, (int)nx, (int)ny, type);
        if (r != 0 || plan == null) return null;
        if (hipfftSetStream(plan, stream) != 0) { hipfftDestroy(plan); return null; }
        g_rfft_plan[slot] = plan;
        g_rfft_type[slot] = type;
        g_rfft_nx[slot] = nx;
        g_rfft_ny[slot] = ny;
        g_rfft_batch[slot] = batch;
        g_rfft_used[slot] = 1;
        return plan;
    }
}

// ── Host arrays (double) ─────────────────────────────────────────────────────

static HipDevPtr_* rfft_scratch_(unsigned long bytes) {
    if (bytes <= g_rfft_scratch_bytes) return g_rfft_scratch;
    unsafe {
        if (g_rfft_scratch != null) {
            hipStreamSynchronize(rfft_stream_());
            hipFree(g_rfft_scratch);
            g_rfft_scratch = null;
            g_rfft_scratch_bytes = 0UL;
        }
        if (hipMalloc(&g_rfft_scratch, bytes) != 0) { g_rfft_scratch = null; return null; }
    }
    g_rfft_scratch_bytes = bytes;
    return g_rfft_scratch;
}

// In-place Z2Z on host data: total = nx * max(ny,1) * batch complex values.
static int rfft_host_z2z_(&Complex data, unsigned long nx, unsigned long ny, unsigned long batch, int inverse) {
    HipfftPlan_* plan = rfft_plan_(RFFT_Z2Z, nx, ny, batch);
    if (plan == null) return 0;
    unsigned long len = ny == 0UL ? nx : nx * ny;
    unsigned long total = len * batch;
    unsigned long bytes = checked_mul_size(total, sizeof(struct Complex));
    HipDevPtr_* d = rfft_scratch_(bytes);
    if (d == null) return 0;
    unsafe {
        RocmNativeStream* stream = rfft_stream_();
        if (hipMemcpyAsync((unsigned char*)d, (const unsigned char*)data, bytes, RFFT_H2D, stream) != 0) return 0;
        if (hipfftExecZ2Z(plan, d, d, inverse ? RFFT_INVERSE : RFFT_FORWARD) != 0) return 0;
        if (hipMemcpyAsync((unsigned char*)data, (const unsigned char*)d, bytes, RFFT_D2H, stream) != 0) return 0;
        if (hipStreamSynchronize(stream) != 0) return 0;
        if (inverse) {
            double inv = 1.0 / (double)len;
            unsigned long i = 0UL;
            while (i < total) {
                data[i].re = data[i].re * inv;
                data[i].im = data[i].im * inv;
                i = i + 1UL;
            }
        }
    }
    return 1;
}

int rocm_fft_n_(&Complex data, unsigned long n) { return rfft_host_z2z_(data, n, 0UL, 1UL, 0); }
int rocm_ifft_n_(&Complex data, unsigned long n) { return rfft_host_z2z_(data, n, 0UL, 1UL, 1); }

int rocm_fft_batch_n_(&Complex data, unsigned long n, unsigned long batch, int inverse) {
    return rfft_host_z2z_(data, n, 0UL, batch, inverse);
}

int rocm_fft2d([]Complex img, unsigned long h, unsigned long w) {
    if (h == 0UL || w == 0UL || img.len / w < h) { return 0; }
    unsafe { return rfft_host_z2z_((&Complex)img.ptr, h, w, 1UL, 0); }
}
int rocm_ifft2d([]Complex img, unsigned long h, unsigned long w) {
    if (h == 0UL || w == 0UL || img.len / w < h) { return 0; }
    unsafe { return rfft_host_z2z_((&Complex)img.ptr, h, w, 1UL, 1); }
}

// ── Queued device ops (fp32) ─────────────────────────────────────────────────

// buf[0 .. count) *= 1/n, queued.
static int rfft_normalize_(struct RocmBuf buf, unsigned long count, unsigned long n) {
    struct RocmBuf view;
    view.dev = buf.dev;
    view.len = count;
    float k = (float)1.0 / (float)n;
    return rocm_scale(view, k, view);
}

int rocm_fft_c2c(struct RocmBuf in, struct RocmBuf out, unsigned long n, unsigned long batch, int inverse) {
    unsigned long floats = 2UL * n * batch;
    if (in.dev == 0UL || out.dev == 0UL || in.len < floats || out.len < floats) return 0;
    HipfftPlan_* plan = rfft_plan_(RFFT_C2C, n, 0UL, batch);
    if (plan == null) return 0;
    unsafe {
        if (hipfftExecC2C(plan, (HipDevPtr_*)in.dev, (HipDevPtr_*)out.dev, inverse ? RFFT_INVERSE : RFFT_FORWARD) != 0)
            return 0;
    }
    return inverse ? rfft_normalize_(out, floats, n) : 1;
}

int rocm_fft_r2c(struct RocmBuf in, struct RocmBuf out, unsigned long n, unsigned long batch) {
    if (in.dev == 0UL || out.dev == 0UL || in.len < n * batch || out.len < 2UL * (n / 2UL + 1UL) * batch)
        return 0;
    HipfftPlan_* plan = rfft_plan_(RFFT_R2C, n, 0UL, batch);
    if (plan == null) return 0;
    unsafe { return hipfftExecR2C(plan, (HipDevPtr_*)in.dev, (HipDevPtr_*)out.dev) == 0; }
}

int rocm_fft_c2r(struct RocmBuf in, struct RocmBuf out, unsigned long n, unsigned long batch) {
    if (in.dev == 0UL || out.dev == 0UL || in.len < 2UL * (n / 2UL + 1UL) * batch || out.len < n * batch)
        return 0;
    HipfftPlan_* plan = rfft_plan_(RFFT_C2R, n, 0UL, batch);
    if (plan == null) return 0;
    unsafe {
        if (hipfftExecC2R(plan, (HipDevPtr_*)in.dev, (HipDevPtr_*)out.dev) != 0) return 0;
    }
    return rfft_normalize_(out, n * batch, n);
}

// ── Slice API (the public signatures; the (reference, length) cores above
// are the *_n_ functions) ─────────────────────────────────────────────────────

int rocm_fft([]Complex data) {
    if (data.len == 0UL) return 0;
    unsafe { return rocm_fft_n_((&Complex)data.ptr, data.len); }
}

int rocm_ifft([]Complex data) {
    if (data.len == 0UL) return 0;
    unsafe { return rocm_ifft_n_((&Complex)data.ptr, data.len); }
}

int rocm_fft_batch([]Complex data, unsigned long n, int inverse) {
    if (n == 0UL || data.len == 0UL || data.len % n != 0UL) return 0;
    unsafe { return rocm_fft_batch_n_((&Complex)data.ptr, n, data.len / n, inverse); }
}

} // namespace std
