// SafeC Standard Library — FFT on NVIDIA GPUs (see fft_cuda.h).
//
// cuFFT plans are created with the legacy cufftPlan1d / cufftPlan2d calls
// (each plan allocates its own work area) and bound to gpu_cuda's stream.
#pragma once
#include <std/stddef.h>
#include <std/dsp/fft_cuda.h>
#include <std/dsp/complex_dsp.h>
#include <std/ml/gpu_cuda.h>
#include <std/mem.h>

namespace std {

// ── cuFFT C ABI (cufft.h; cufftHandle is an int) ─────────────────────────────
extern int cufftPlan1d(int* plan, int nx, int type, int batch);
extern int cufftPlan2d(int* plan, int nx, int ny, int type);
extern int cufftSetStream(int plan, void* stream);
extern int cufftDestroy(int plan);
extern int cufftExecC2C(int plan, void* idata, void* odata, int direction);
extern int cufftExecR2C(int plan, void* idata, void* odata);
extern int cufftExecC2R(int plan, void* idata, void* odata);
extern int cufftExecZ2Z(int plan, void* idata, void* odata, int direction);

extern int cuMemAlloc_v2(unsigned long long* dptr, size_t bytes);
extern int cuMemFree_v2(unsigned long long dptr);
extern int cuMemcpyHtoDAsync_v2(unsigned long long dst, const void* src, size_t bytes, void* stream);
extern int cuMemcpyDtoHAsync_v2(void* dst, unsigned long long src, size_t bytes, void* stream);
extern int cuStreamSynchronize(void* stream);

#define FFT_R2C      42     // CUFFT_R2C (0x2a)
#define FFT_C2R      44     // CUFFT_C2R (0x2c)
#define FFT_C2C      41     // CUFFT_C2C (0x29)
#define FFT_Z2Z      105    // CUFFT_Z2Z (0x69)
#define FFT_FORWARD  -1     // CUFFT_FORWARD
#define FFT_INVERSE  1      // CUFFT_INVERSE
#define FFT_PLANS    16

// Plan cache: (type, nx, ny, batch) -> cufftHandle; ny == 0 for 1D plans.
static int g_fft_used[FFT_PLANS];
static int g_fft_type[FFT_PLANS];
static unsigned long g_fft_nx[FFT_PLANS];
static unsigned long g_fft_ny[FFT_PLANS];
static unsigned long g_fft_batch[FFT_PLANS];
static int g_fft_plan[FFT_PLANS];
static int g_fft_evict = 0;
static int g_fft_init = 0;

// Device scratch for the host-array functions; grows as needed.
static unsigned long long g_fft_scratch = 0ULL;
static unsigned long g_fft_scratch_bytes = 0UL;

static int fft_dim_ok_(unsigned long v) { return v > 0UL && v <= 2147483647UL; }

// Returns a cuFFT plan on gpu_cuda's stream, or -1.
static int fft_plan_(int type, unsigned long nx, unsigned long ny, unsigned long batch) {
    if (!cuda_available() || !fft_dim_ok_(nx) || !fft_dim_ok_(batch) ||
        (ny != 0UL && (!fft_dim_ok_(ny) || batch != 1UL))) return -1;
    if (!g_fft_init) {
        int i = 0;
        while (i < FFT_PLANS) { g_fft_used[i] = 0; i = i + 1; }
        g_fft_init = 1;
    }
    int slot = -1;
    int s = 0;
    while (s < FFT_PLANS) {
        if (g_fft_used[s] != 0) {
            if (g_fft_type[s] == type && g_fft_nx[s] == nx && g_fft_ny[s] == ny && g_fft_batch[s] == batch)
                return g_fft_plan[s];
        } else if (slot < 0) {
            slot = s;
        }
        s = s + 1;
    }
    unsafe {
        if (slot < 0) {
            slot = g_fft_evict;
            g_fft_evict = (g_fft_evict + 1) % FFT_PLANS;
            cuStreamSynchronize(cuda_stream_handle());   // queued work may still use the plan
            cufftDestroy(g_fft_plan[slot]);
            g_fft_used[slot] = 0;
        }
        int plan = 0;
        int r = ny == 0UL ? cufftPlan1d(&plan, (int)nx, type, (int)batch)
                          : cufftPlan2d(&plan, (int)nx, (int)ny, type);
        if (r != 0) return -1;
        if (cufftSetStream(plan, cuda_stream_handle()) != 0) { cufftDestroy(plan); return -1; }
        g_fft_plan[slot] = plan;
        g_fft_type[slot] = type;
        g_fft_nx[slot] = nx;
        g_fft_ny[slot] = ny;
        g_fft_batch[slot] = batch;
        g_fft_used[slot] = 1;
        return plan;
    }
}

// ── Host arrays (double) ─────────────────────────────────────────────────────

static unsigned long long fft_scratch_(unsigned long bytes) {
    if (bytes <= g_fft_scratch_bytes) return g_fft_scratch;
    unsafe {
        if (g_fft_scratch != 0ULL) {
            cuStreamSynchronize(cuda_stream_handle());
            cuMemFree_v2(g_fft_scratch);
            g_fft_scratch = 0ULL;
            g_fft_scratch_bytes = 0UL;
        }
        if (cuMemAlloc_v2(&g_fft_scratch, bytes) != 0) { g_fft_scratch = 0ULL; return 0ULL; }
    }
    g_fft_scratch_bytes = bytes;
    return g_fft_scratch;
}

// In-place Z2Z on host data: total = nx * max(ny,1) * batch complex values,
// 'norm' = the transform length the inverse divides by.
static int fft_host_z2z_(struct Complex* data, unsigned long nx, unsigned long ny, unsigned long batch,
                         int inverse) {
    int plan = fft_plan_(FFT_Z2Z, nx, ny, batch);
    if (plan < 0) return 0;
    unsigned long len = ny == 0UL ? nx : nx * ny;
    unsigned long total = len * batch;
    unsigned long bytes = checked_mul_size(total, sizeof(struct Complex));
    unsigned long long d = fft_scratch_(bytes);
    if (d == 0ULL) return 0;
    unsafe {
        void* stream = cuda_stream_handle();
        if (cuMemcpyHtoDAsync_v2(d, (const void*)data, bytes, stream) != 0) return 0;
        if (cufftExecZ2Z(plan, (void*)(unsigned long)d, (void*)(unsigned long)d,
                         inverse ? FFT_INVERSE : FFT_FORWARD) != 0) return 0;
        if (cuMemcpyDtoHAsync_v2((void*)data, d, bytes, stream) != 0) return 0;
        if (cuStreamSynchronize(stream) != 0) return 0;
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

int cuda_fft(&Complex data, unsigned long n) { return fft_host_z2z_(data, n, 0UL, 1UL, 0); }
int cuda_ifft(&Complex data, unsigned long n) { return fft_host_z2z_(data, n, 0UL, 1UL, 1); }

int cuda_fft_batch(&Complex data, unsigned long n, unsigned long batch, int inverse) {
    return fft_host_z2z_(data, n, 0UL, batch, inverse);
}

int cuda_fft2d(&Complex img, unsigned long h, unsigned long w) { return fft_host_z2z_(img, h, w, 1UL, 0); }
int cuda_ifft2d(&Complex img, unsigned long h, unsigned long w) { return fft_host_z2z_(img, h, w, 1UL, 1); }

// ── Queued device ops (fp32) ─────────────────────────────────────────────────

// buf[0 .. count) *= 1/n, queued.
static int fft_normalize_(struct CudaBuf buf, unsigned long count, unsigned long n) {
    struct CudaBuf view;
    view.dev = buf.dev;
    view.len = count;
    float k = (float)1.0 / (float)n;
    return cuda_scale(view, k, view);
}

int cuda_fft_c2c(struct CudaBuf in, struct CudaBuf out, unsigned long n, unsigned long batch, int inverse) {
    unsigned long floats = 2UL * n * batch;
    if (in.dev == 0UL || out.dev == 0UL || in.len < floats || out.len < floats) return 0;
    int plan = fft_plan_(FFT_C2C, n, 0UL, batch);
    if (plan < 0) return 0;
    unsafe {
        if (cufftExecC2C(plan, (void*)in.dev, (void*)out.dev, inverse ? FFT_INVERSE : FFT_FORWARD) != 0)
            return 0;
    }
    return inverse ? fft_normalize_(out, floats, n) : 1;
}

int cuda_fft_r2c(struct CudaBuf in, struct CudaBuf out, unsigned long n, unsigned long batch) {
    if (in.dev == 0UL || out.dev == 0UL || in.len < n * batch || out.len < 2UL * (n / 2UL + 1UL) * batch)
        return 0;
    int plan = fft_plan_(FFT_R2C, n, 0UL, batch);
    if (plan < 0) return 0;
    unsafe { return cufftExecR2C(plan, (void*)in.dev, (void*)out.dev) == 0; }
}

int cuda_fft_c2r(struct CudaBuf in, struct CudaBuf out, unsigned long n, unsigned long batch) {
    if (in.dev == 0UL || out.dev == 0UL || in.len < 2UL * (n / 2UL + 1UL) * batch || out.len < n * batch)
        return 0;
    int plan = fft_plan_(FFT_C2R, n, 0UL, batch);
    if (plan < 0) return 0;
    unsafe {
        if (cufftExecC2R(plan, (void*)in.dev, (void*)out.dev) != 0) return 0;
    }
    return fft_normalize_(out, n * batch, n);
}

} // namespace std
