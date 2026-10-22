#pragma once
// SafeC Standard Library — the WebGPU device backend's interface
// (device_webgpu.sc): the static backend interface device.h describes, plus
// the few calls gpu_webgpu.h's host-array API makes on the same device,
// queue and kernels. Programs reach the backend through device.h (the
// webgpu feature, DEVICE_WEBGPU); this header is for gpu_webgpu.sc.
#include <std/stddef.h>
#include <std/ml/device.h>

namespace std {

// Handles are byte offsets into the backend's one device heap (never 0),
// with device.h's element-type tag in the top bits.
struct WebGpuBackend {
    int unused_;

    static int available();
    static unsigned long alloc(unsigned long bytes);
    static void release(unsigned long handle);
    static int upload(unsigned long dst, const &float src, unsigned long n);
    static int download(&float dst, unsigned long src, unsigned long n);
    static int copy(unsigned long dst, unsigned long src, unsigned long n);
    static int sync();
    static int gemm(int ta, int tb, unsigned long m, unsigned long n, unsigned long k, float alpha,
                    unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb,
                    float beta, unsigned long c, unsigned long ldc);
    static int gemm_batched(const &DevGemm g);
    static int conv(const &DevConv c);
    static int storage16();
    static int launch(int kernel, unsigned long n, const &DevArgs args);
    static unsigned long offset(unsigned long handle, unsigned long bytes);

    // bytes (a multiple of 4) to / from device memory at a 4-byte aligned handle
    static int upload_bytes(unsigned long dst, const void* src, unsigned long bytes);
    static int download_bytes(void* dst, unsigned long src, unsigned long bytes);
    // gpu_webgpu.h's elementwise ops without a DK_* kernel (device_kernels.wgsl's
    // wk_map): out = a / b (op 0), pow(a, b) (1), sqrt(a) (2) over n elements
    static int map(int op, unsigned long a, unsigned long b, unsigned long out, unsigned long n);
};

} // namespace std
