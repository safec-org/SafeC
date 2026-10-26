#pragma once
// SafeC Standard Library — array ops on a Vulkan GPU (SPIR-V), cross-vendor:
// NVIDIA, AMD, Intel, Apple through MoltenVK, wherever a Vulkan 1.2 driver
// is.
//
// A thin layer over device_vulkan.sc's VulkanBackend (the tensor device
// DEVICE_VULKAN): the same device memory, command stream and SPIR-V kernels
// (device_kernels.cu, compiled ahead of time and embedded), reached through
// device.h's internal API, so nothing here is a second copy of a kernel or
// of the Vulkan setup. Using it registers the Vulkan backend whether or not
// the program was built with the vulkan feature; link -lvulkan (on macOS
// Homebrew's vulkan-loader and molten-vk: -L/opt/homebrew/lib -lvulkan).
// Requirements and limits are device_vulkan.sc's.
//
// Two tiers, as gpu_mps.h has: SpirvBuf device buffers and the ops on them,
// queued on the device's stream without waiting (a chain of ops runs back
// to back; reading a buffer waits for the work before it), and host-array
// spirv_*_f32 functions that upload, run one op and read the result back.
// For whole models, place tensors on DEVICE_VULKAN instead (device.h).
namespace std {

// Element types of a buffer (SpirvBuf.dtype): 0 = fp32, or 16-bit values.
#define SPIRV_FP16 1   // same values as gpu_mps.h's MPS_FP16 / MPS_BF16
#define SPIRV_BF16 2

// A Vulkan device buffer. 'handle' is its buffer device address (what the
// kernels and VulkanTensor.device_buffer() name it by), 0 for an invalid
// (failed or freed) buffer; every op returns 0 for one. 'bytes' is the size
// it was made with, 'dtype' its element type (informational: the ops below
// take fp32). Buffers come from the device's memory cache; free() returns
// one to it, exactly once, and may wait for queued work that uses it.
struct SpirvBuf {
    unsigned long handle;
    unsigned long bytes;
    int           dtype;

    int  valid() const;   // handle != 0
    void free();          // back to the cache; handle := 0
    // Copies the first 'bytes' (<= this.bytes, whole 32-bit words) into
    // 'dst' once the queued work is done (waits). 1 = done.
    int  read([]unsigned char dst) const;
    // Copies src.len bytes (<= this.bytes, whole 32-bit words) from 'src' into the
    // buffer, in order with the queued work (ops queued before it read the
    // old contents). 1 = done.
    int  write([]const unsigned char src);

    static struct SpirvBuf alloc(unsigned long n, int dtype);       // n uninitialized elements
    static struct SpirvBuf upload(const &float host, unsigned long n);          // n fp32 values
};

// ── Ops on device buffers (fp32, n elements; queued, no wait) ───────────────
// out may be an input (in place). 1 = queued, 0 = no device or an invalid
// or too small buffer.
int spirv_add(struct SpirvBuf a, struct SpirvBuf b, struct SpirvBuf out, unsigned long n);
int spirv_sub(struct SpirvBuf a, struct SpirvBuf b, struct SpirvBuf out, unsigned long n);
int spirv_mul(struct SpirvBuf a, struct SpirvBuf b, struct SpirvBuf out, unsigned long n);
int spirv_div(struct SpirvBuf a, struct SpirvBuf b, struct SpirvBuf out, unsigned long n);
// a^b as exp(b log a) and sqrt(a) as exp(log(a) / 2): for a >= 0 (a negative
// base gives NaN, as powf does for a non-integer exponent).
int spirv_pow(struct SpirvBuf a, struct SpirvBuf b, struct SpirvBuf out, unsigned long n);
int spirv_sqrt(struct SpirvBuf a, struct SpirvBuf out, unsigned long n);
int spirv_relu(struct SpirvBuf a, struct SpirvBuf out, unsigned long n);
int spirv_log(struct SpirvBuf a, struct SpirvBuf out, unsigned long n);
int spirv_exp(struct SpirvBuf a, struct SpirvBuf out, unsigned long n);
int spirv_scale(struct SpirvBuf a, float k, struct SpirvBuf out, unsigned long n);
// out[0] = the sum of a's n values (out must not be a)
int spirv_sum(struct SpirvBuf a, struct SpirvBuf out, unsigned long n);
// out [M, N] = a [M, K] . b [K, N], row-major
int spirv_matmul(struct SpirvBuf a, struct SpirvBuf b, struct SpirvBuf out, unsigned long M, unsigned long K,
                 unsigned long N);
// Waits for every queued op. 1 on success.
int spirv_sync();

// ── Host arrays: upload, one op, read back (waits) ──────────────────────────
// 1 = done on the GPU, 0 = no device (out untouched): compute on the CPU.
int spirv_add_f32(const &float a, const &float b, &float out, unsigned long n);
int spirv_sub_f32(const &float a, const &float b, &float out, unsigned long n);
int spirv_mul_f32(const &float a, const &float b, &float out, unsigned long n);
int spirv_div_f32(const &float a, const &float b, &float out, unsigned long n);
int spirv_pow_f32(const &float a, const &float b, &float out, unsigned long n);
int spirv_relu_f32(const &float a, &float out, unsigned long n);
int spirv_log_f32(const &float a, &float out, unsigned long n);
int spirv_exp_f32(const &float a, &float out, unsigned long n);
int spirv_sqrt_f32(const &float a, &float out, unsigned long n);
int spirv_scale_f32(const &float a, float k, &float out, unsigned long n);
int spirv_sum_f32(const &float a, &float out, unsigned long n);   // out[0]
int spirv_matmul_f32(const &float a, const &float b, &float out,
                      unsigned long M, unsigned long K, unsigned long N);

// 1 if a Vulkan GPU the backend can use is present (initializes it).
int spirv_available();

} // namespace std
