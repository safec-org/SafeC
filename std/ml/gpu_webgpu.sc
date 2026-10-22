// SafeC Standard Library — GPU array ops via WebGPU (see gpu_webgpu.h): a
// thin layer over device_webgpu.sc's WebGpuBackend. Buffers are blocks of
// the backend's device heap; ops are its kernel launches and GEMMs; the
// host-array ops upload their inputs into temporary buffers, run, read the
// result back and free them.
#pragma once
#include <std/stddef.h>
#include <std/ml/gpu_webgpu.h>
#include <std/ml/device.h>
#include <std/ml/device_kernels.h>
#include <std/ml/device_webgpu.h>
#include <std/ml/float16.h>

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);
extern void* memcpy(void* dst, const void* src, size_t n);

int webgpu_available() { return WebGpuBackend::available(); }

static unsigned long wgb_esize_(int dtype) {
    if (dtype == 0) return 4UL;
    if (dtype == WEBGPU_FP16 || dtype == WEBGPU_BF16) return 2UL;
    return 0UL;
}

static struct WebGpuBuf wgb_make_(unsigned long handle, unsigned long bytes, int dtype) {
    struct WebGpuBuf b;
    b.handle = handle;
    b.bytes = handle == 0UL ? 0UL : bytes;
    b.dtype = dtype;
    return b;
}

// A buffer of 'bytes' holding 'src' (when given).
static struct WebGpuBuf wgb_new_(unsigned long bytes, int dtype, const void* src) {
    unsigned long h = WebGpuBackend::alloc((bytes + 3UL) & ~3UL);
    if (h == 0UL) return wgb_make_(0UL, 0UL, dtype);
    struct WebGpuBuf b = wgb_make_(dev_tagged_(h, (unsigned long)dtype), bytes, dtype);
    unsafe {
        if (src != (const void*)0 && !b.write((const &void)src, bytes)) { b.free(); return wgb_make_(0UL, 0UL, dtype); }
    }
    return b;
}

int WebGpuBuf::valid() const { return self.handle != 0UL; }

void WebGpuBuf::free() {
    if (self.handle != 0UL) WebGpuBackend::release(dev_addr_(self.handle));
    self.handle = 0UL;
    self.bytes = 0UL;
}

int WebGpuBuf::read(&void dst, unsigned long bytes) const {
    if (self.handle == 0UL || bytes > self.bytes) return 0;
    unsafe { return WebGpuBackend::download_bytes((void*)dst, dev_addr_(self.handle), bytes); }
}

int WebGpuBuf::read_f32(&float dst, unsigned long n) const {
    unsigned long es = wgb_esize_(self.dtype);
    if (self.handle == 0UL || es == 0UL || n * es > self.bytes) return 0;
    if (self.dtype == 0) { unsafe { return self.read((&void)(void*)dst, 4UL * n); } }
    unsafe {
        unsigned short* bits = (unsigned short*)malloc(2UL * n + 2UL);
        if (bits == (unsigned short*)0) return 0;
        int ok = self.read((&void)(void*)bits, 2UL * n);
        if (ok && self.dtype == WEBGPU_FP16) fp16_to_f32_bulk((const &unsigned short)bits, dst, n);
        if (ok && self.dtype == WEBGPU_BF16) bf16_to_f32_bulk((const &unsigned short)bits, dst, n);
        free((void*)bits);
        return ok;
    }
}

// Whole words go to the device: an odd tail is padded within the buffer.
int WebGpuBuf::write(const &void src, unsigned long bytes) {
    if (self.handle == 0UL || bytes > self.bytes) return 0;
    if ((bytes & 3UL) == 0UL) { unsafe { return WebGpuBackend::upload_bytes(dev_addr_(self.handle), (const void*)src, bytes); } }
    unsafe {
        unsigned long words = (bytes + 3UL) & ~3UL;
        unsigned char* tmp = (unsigned char*)malloc(words);
        if (tmp == (unsigned char*)0) return 0;
        memcpy((void*)tmp, (const void*)src, bytes);
        unsigned long i = bytes;
        while (i < words) { tmp[i] = (unsigned char)0; i = i + 1UL; }
        int ok = WebGpuBackend::upload_bytes(dev_addr_(self.handle), (const void*)tmp, words);
        free((void*)tmp);
        return ok;
    }
}

struct WebGpuBuf WebGpuBuf::alloc(unsigned long n, int dtype) {
    unsigned long es = wgb_esize_(dtype);
    if (es == 0UL) return wgb_make_(0UL, 0UL, dtype);
    unsafe { return wgb_new_(n * es, dtype, (const void*)0); }
}

struct WebGpuBuf WebGpuBuf::upload(const &float host, unsigned long n) {
    unsafe { return wgb_new_(4UL * n, 0, (const void*)host); }
}

struct WebGpuBuf WebGpuBuf::upload_bits(const &unsigned short bits, unsigned long n, int dtype) {
    if (dtype != WEBGPU_FP16 && dtype != WEBGPU_BF16) return wgb_make_(0UL, 0UL, dtype);
    unsafe { return wgb_new_(2UL * n, dtype, (const void*)bits); }
}

static struct WebGpuBuf wgb_upload_half_(const &float host, unsigned long n, int dtype) {
    unsafe {
        unsigned short* bits = (unsigned short*)malloc(2UL * n + 2UL);
        if (bits == (unsigned short*)0) return wgb_make_(0UL, 0UL, dtype);
        if (dtype == WEBGPU_FP16) f32_to_fp16_bulk(host, (&unsigned short)bits, n);
        else f32_to_bf16_bulk(host, (&unsigned short)bits, n);
        struct WebGpuBuf b = WebGpuBuf::upload_bits((const &unsigned short)bits, n, dtype);
        free((void*)bits);
        return b;
    }
}

struct WebGpuBuf WebGpuBuf::upload_f16(const &float host, unsigned long n) { return wgb_upload_half_(host, n, WEBGPU_FP16); }
struct WebGpuBuf WebGpuBuf::upload_bf16(const &float host, unsigned long n) { return wgb_upload_half_(host, n, WEBGPU_BF16); }

// ── Device-buffer ops ─────────────────────────────────────────────────────────
// A DK_* kernel over n threads (device.sc's launch convention: u7 = n).
static int wgb_launch_(int kernel, unsigned long n, struct DevArgs a) {
    if (n == 0UL) return 1;
    unsafe { a.u[7] = (unsigned int)n; }
    return WebGpuBackend::launch(kernel, n, &a);
}

int webgpu_matmul(struct WebGpuBuf a, struct WebGpuBuf b, struct WebGpuBuf out,
                  unsigned long M, unsigned long K, unsigned long N) {
    if (!a.valid() || !b.valid() || !out.valid() || a.dtype != b.dtype) return 0;
    if (M * K * wgb_esize_(a.dtype) > a.bytes || K * N * wgb_esize_(b.dtype) > b.bytes ||
        M * N * wgb_esize_(out.dtype) > out.bytes)
        return 0;
    struct DevGemm g;
    g.dtype = a.dtype; g.ctype = out.dtype;
    g.ta = 0; g.tb = 0; g.m = M; g.n = N; g.k = K; g.alpha = (float)1.0;
    g.a = dev_addr_(a.handle); g.lda = K; g.sa = 0UL;
    g.b = dev_addr_(b.handle); g.ldb = N; g.sb = 0UL;
    g.beta = (float)0.0; g.c = dev_addr_(out.handle); g.ldc = N; g.sc = 0UL; g.batch = 1UL;
    return WebGpuBackend::gemm_batched(&g);
}

int webgpu_sgd_update(struct WebGpuBuf w, struct WebGpuBuf g, float lr, unsigned long n) {
    if (!w.valid() || !g.valid()) return 0;
    struct DevArgs a = dev_args_();
    unsafe { a.p[0] = g.handle; a.p[2] = w.handle; a.u[0] = 10U; a.f[0] = -lr; }   // EW_ACC_SCALE: w += g * -lr
    return wgb_launch_(DK_EWOP_10, n, a);
}

int webgpu_sync() { return WebGpuBackend::sync(); }

// ── Host arrays ───────────────────────────────────────────────────────────────
// out = op(a, b) through temporary buffers: EW op code ew (k: its constant),
// or wk_map's op (map >= 0).
static int wgb_host_ew_(int ew, int map, const &float a, const ?&float b, float k, &float out, unsigned long n) {
    if (n == 0UL) return 1;
    if (!webgpu_available()) return 0;
    struct WebGpuBuf ba = WebGpuBuf::upload(a, n);
    struct WebGpuBuf bb = wgb_make_(0UL, 0UL, 0);
    match (b) {
        case some(bp): { bb = WebGpuBuf::upload(bp, n); }
        case null: { }
    }
    struct WebGpuBuf bo = WebGpuBuf::alloc(n, 0);
    int ok = ba.valid() && bo.valid();
    unsafe { if (b != null && !bb.valid()) ok = 0; }
    if (ok) {
        if (map >= 0) {
            ok = WebGpuBackend::map(map, ba.handle, bb.handle, bo.handle, n);
        } else {
            struct DevArgs x = dev_args_();
            unsafe { x.p[0] = ba.handle; x.p[1] = bb.handle; x.p[2] = bo.handle; x.u[0] = (unsigned int)ew; x.f[0] = k; }
            ok = wgb_launch_(device_ew_kernel_(ew), n, x);
        }
    }
    unsafe { if (ok) ok = bo.read((&void)(void*)out, 4UL * n); }
    ba.free(); bb.free(); bo.free();
    return ok;
}

int webgpu_add_f32(const &float a, const &float b, &float out, unsigned long n) { return wgb_host_ew_(0, -1, a, b, (float)0.0, out, n); }
int webgpu_sub_f32(const &float a, const &float b, &float out, unsigned long n) { return wgb_host_ew_(1, -1, a, b, (float)0.0, out, n); }
int webgpu_mul_f32(const &float a, const &float b, &float out, unsigned long n) { return wgb_host_ew_(2, -1, a, b, (float)0.0, out, n); }
int webgpu_div_f32(const &float a, const &float b, &float out, unsigned long n) { return wgb_host_ew_(-1, 0, a, b, (float)0.0, out, n); }
int webgpu_pow_f32(const &float a, const &float b, &float out, unsigned long n) { return wgb_host_ew_(-1, 1, a, b, (float)0.0, out, n); }
int webgpu_relu_f32(const &float a, &float out, unsigned long n) { return wgb_host_ew_(4, -1, a, null, (float)0.0, out, n); }
int webgpu_log_f32(const &float a, &float out, unsigned long n) { return wgb_host_ew_(21, -1, a, null, (float)0.0, out, n); }
int webgpu_exp_f32(const &float a, &float out, unsigned long n) { return wgb_host_ew_(20, -1, a, null, (float)0.0, out, n); }
int webgpu_sqrt_f32(const &float a, &float out, unsigned long n) { return wgb_host_ew_(-1, 2, a, null, (float)0.0, out, n); }
int webgpu_scale_f32(const &float a, float k, &float out, unsigned long n) { return wgb_host_ew_(3, -1, a, null, k, out, n); }

int webgpu_sum_f32(const &float a, &float out, unsigned long n) {
    if (!webgpu_available()) return 0;
    float zero = (float)0.0;
    struct WebGpuBuf ba = WebGpuBuf::upload(a, n);
    struct WebGpuBuf bo;
    unsafe { bo = WebGpuBuf::upload((const &float)&zero, 1UL); }
    int ok = ba.valid() && bo.valid();
    if (ok && n > 0UL) {
        struct DevArgs x = dev_args_();
        unsafe { x.p[0] = ba.handle; x.p[1] = bo.handle; }
        ok = wgb_launch_(DK_SUM, n, x);
    }
    unsafe { if (ok) ok = bo.read((&void)(void*)out, 4UL); }
    ba.free(); bo.free();
    return ok;
}

int webgpu_matmul_f32(const &float a, const &float b, &float out,
                      unsigned long M, unsigned long K, unsigned long N) {
    if (!webgpu_available()) return 0;
    struct WebGpuBuf ba = WebGpuBuf::upload(a, M * K);
    struct WebGpuBuf bb = WebGpuBuf::upload(b, K * N);
    struct WebGpuBuf bo = WebGpuBuf::alloc(M * N, 0);
    int ok = ba.valid() && bb.valid() && bo.valid() && webgpu_matmul(ba, bb, bo, M, K, N);
    unsafe { if (ok) ok = bo.read((&void)(void*)out, 4UL * M * N); }
    ba.free(); bb.free(); bo.free();
    return ok;
}

} // namespace std
