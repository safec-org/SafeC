#pragma once
// SafeC Standard Library — GPU array ops via WebGPU (the native webgpu.h C
// API of wgpu-native or Dawn: the API a browser's WebGPU exposes to
// JavaScript, called directly).
//
// A thin layer over the WebGPU tensor backend (device_webgpu.sc,
// DEVICE_WEBGPU): the same device, queue, device heap and WGSL kernels
// (device_kernels.wgsl), so these ops and std::ml tensors on DEVICE_WEBGPU
// share one stream and run in order with each other. Every op is one of
// the backend's kernels -- the DK_* elementwise ops and sum, wk_gemm for
// matrix products, and wk_map for division, powers and square roots, which
// the tensor ops don't need -- not a second copy of them. For training,
// use the tensor backend (t.place(DEVICE_WEBGPU), TrainConfig.device):
// these are for code that wants a GPU array without tensors.
//
// Link a WebGPU implementation as device_webgpu.sc describes (wgpu-native:
// -lwgpu_native); the webgpu feature isn't needed to call these (it
// registers the tensor backend). Without a WebGPU adapter every op returns
// 0 and every constructor an invalid buffer.
namespace std {

// 1 if a WebGPU adapter and device are available (initializes them).
int webgpu_available();

// ── Device buffers ────────────────────────────────────────────────────────────
// Element types of a buffer (WebGpuBuf.dtype): 0 = fp32, or 16-bit values.
#define WEBGPU_FP16 1   // the values of device.h's DEVICE_FP16 / DEVICE_BF16
#define WEBGPU_BF16 2

// Memory on the WebGPU device -- the WebGPU twin of gpu_mps.h's MpsBuf.
// 'handle' is the backend's device handle (a byte offset into its heap, the
// element type in its top two bits as device.h's DEV_TAG_*), 0 for an
// invalid (failed or freed) buffer; every op returns 0 for one. 'bytes' is
// its size, 'dtype' its element type: 0 = fp32, WEBGPU_FP16 or WEBGPU_BF16.
//
// Ops on buffers are queued on the device's stream and return without
// waiting; read() and read_f32() wait for the work queued before them, and
// write() is ordered after it, so a buffer may be written, used and read
// back without explicit synchronization. free() doesn't wait either: work
// already queued still reads the memory before anything queued after it can
// reuse it. Owned by the caller: free() exactly once.
struct WebGpuBuf {
    unsigned long handle;
    unsigned long bytes;
    int           dtype;

    int  valid() const;   // handle != 0
    void free();          // back to the device heap; handle := 0
    // Copies the first dst.len (<= this.bytes) bytes into 'dst', once the work
    // queued before has run. 1 = done.
    int  read([]unsigned char dst) const;
    // The first n elements as fp32 (16-bit ones converted on the CPU).
    int  read_f32([]float dst) const;
    // internal: the (reference, length) core of read_f32()
    int  read_f32_n_(&float dst, unsigned long n) const;
    // Copies src.len bytes (<= this.bytes) from 'src' into the buffer, after the
    // work queued before. 1 = done.
    int  write([]const unsigned char src);

    static struct WebGpuBuf alloc(unsigned long n, int dtype);       // n uninitialized elements
    static struct WebGpuBuf upload([]const float host);
    // internal: the (reference, length) core of upload()
    static struct WebGpuBuf upload_n_(const &float host, unsigned long n);          // n fp32 values
    // n fp32 values rounded on the CPU (float16.h's f32_to_fp16_bulk /
    // f32_to_bf16_bulk) and stored as fp16 / bf16.
    static struct WebGpuBuf upload_f16([]const float host);
    // internal: the (reference, length) core of upload_f16()
    static struct WebGpuBuf upload_f16_n_(const &float host, unsigned long n);
    static struct WebGpuBuf upload_bf16([]const float host);
    // internal: the (reference, length) core of upload_bf16()
    static struct WebGpuBuf upload_bf16_n_(const &float host, unsigned long n);
    // n raw fp16 or bf16 values (dtype WEBGPU_FP16 / WEBGPU_BF16), copied as is.
    static struct WebGpuBuf upload_bits([]const unsigned short bits, int dtype);
    // internal: the (reference, length) core of upload_bits()
    static struct WebGpuBuf upload_bits_n_(const &unsigned short bits, unsigned long n, int dtype);
};

// out [M, N] = a [M, K] . b [K, N], row-major. a and b share a dtype (fp32,
// fp16 or bf16; products accumulate in fp32), out is any dtype. Queued.
int webgpu_matmul(struct WebGpuBuf a, struct WebGpuBuf b, struct WebGpuBuf out,
                  unsigned long M, unsigned long K, unsigned long N);
// w[i] -= lr * g[i] for i < n, in place (any dtypes). Queued.
int webgpu_sgd_update(struct WebGpuBuf w, struct WebGpuBuf g, float lr, unsigned long n);
// Waits for everything queued on the device (tensor work too). 1 on success.
int webgpu_sync();

// ── Host arrays ───────────────────────────────────────────────────────────────
// out[i] = a[i] op b[i] (or op a[i]) for i in [0, n): the inputs uploaded,
// the op run, the result read back. 1 = done, 0 = no WebGPU (out untouched).
int webgpu_add_f32([]const float a, []const float b, []float out);
// internal: the (reference, length) core of webgpu_add_f32()
int webgpu_add_f32_n_(const &float a, const &float b, &float out, unsigned long n);
int webgpu_sub_f32([]const float a, []const float b, []float out);
// internal: the (reference, length) core of webgpu_sub_f32()
int webgpu_sub_f32_n_(const &float a, const &float b, &float out, unsigned long n);
int webgpu_mul_f32([]const float a, []const float b, []float out);
// internal: the (reference, length) core of webgpu_mul_f32()
int webgpu_mul_f32_n_(const &float a, const &float b, &float out, unsigned long n);
int webgpu_div_f32([]const float a, []const float b, []float out);
// internal: the (reference, length) core of webgpu_div_f32()
int webgpu_div_f32_n_(const &float a, const &float b, &float out, unsigned long n);
int webgpu_pow_f32([]const float a, []const float b, []float out);
// internal: the (reference, length) core of webgpu_pow_f32()
int webgpu_pow_f32_n_(const &float a, const &float b, &float out, unsigned long n);
int webgpu_relu_f32([]const float a, []float out);
// internal: the (reference, length) core of webgpu_relu_f32()
int webgpu_relu_f32_n_(const &float a, &float out, unsigned long n);
int webgpu_log_f32([]const float a, []float out);
// internal: the (reference, length) core of webgpu_log_f32()
int webgpu_log_f32_n_(const &float a, &float out, unsigned long n);
int webgpu_exp_f32([]const float a, []float out);
// internal: the (reference, length) core of webgpu_exp_f32()
int webgpu_exp_f32_n_(const &float a, &float out, unsigned long n);
int webgpu_sqrt_f32([]const float a, []float out);
// internal: the (reference, length) core of webgpu_sqrt_f32()
int webgpu_sqrt_f32_n_(const &float a, &float out, unsigned long n);
int webgpu_scale_f32([]const float a, float k, []float out);
// internal: the (reference, length) core of webgpu_scale_f32()
int webgpu_scale_f32_n_(const &float a, float k, &float out, unsigned long n);
// out[0] = sum(a[0..n)): a reduction per workgroup, added atomically.
int webgpu_sum_f32([]const float a, []float out);
// internal: the (reference, length) core of webgpu_sum_f32()
int webgpu_sum_f32_n_(const &float a, &float out, unsigned long n);
// out [M, N] = a [M, K] . b [K, N].
int webgpu_matmul_f32(const &float a, const &float b, &float out,
                      unsigned long M, unsigned long K, unsigned long N);

} // namespace std
