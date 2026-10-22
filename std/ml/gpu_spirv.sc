// SafeC Standard Library — array ops on a Vulkan GPU (see gpu_spirv.h): the
// device backend's memory and kernels (device_vulkan.sc) through device.h's
// internal API, which also counts them in device_stats(DEVICE_VULKAN).
// safeguard: link linux -lvulkan
// safeguard: link macos -L${VULKAN_SDK}/lib -L/opt/homebrew/lib -L/usr/local/lib -lvulkan
// safeguard: link windows -L${VULKAN_SDK}/Lib -lvulkan-1
#pragma once
#include <std/stddef.h>
#include <std/ml/gpu_spirv.h>
#include <std/ml/device.h>
#include <std/ml/device_kernels.h>

namespace std {

// device_kernels.cu's elementwise op codes (tensor.h's EW_*)
#define SPV_EW_ADD 0
#define SPV_EW_SUB 1
#define SPV_EW_MUL 2
#define SPV_EW_SCALE 3
#define SPV_EW_RELU 4
#define SPV_EW_FILL 5
#define SPV_EW_EXP 20
#define SPV_EW_LOG 21
#define SPV_EW_ACC_LOGB 42   // out (+)= x / y: written, a quotient

static int g_spirv_state = 0;   // 0 = not yet, 1 = ready, -1 = no device

int spirv_available() {
    if (g_spirv_state == 0) {
        device_vulkan_register_();
        g_spirv_state = device_available(DEVICE_VULKAN) ? 1 : -1;
    }
    return g_spirv_state == 1;
}

int SpirvBuf::valid() const { return self.handle != 0UL; }

void SpirvBuf::free() {
    if (self.handle == 0UL) return;
    device_mem_free_(DEVICE_VULKAN, self.handle, self.bytes);
    self.handle = 0UL;
}

int SpirvBuf::read(&void dst, unsigned long bytes) const {
    if (self.handle == 0UL || bytes > self.bytes || bytes % 4UL != 0UL) return 0;
    unsafe { return device_download_(DEVICE_VULKAN, (&float)dst, self.handle, bytes / 4UL); }
}

int SpirvBuf::write(const &void src, unsigned long bytes) {
    if (self.handle == 0UL || bytes > self.bytes || bytes % 4UL != 0UL) return 0;
    unsafe { return device_upload_(DEVICE_VULKAN, self.handle, (const &float)src, bytes / 4UL); }
}

struct SpirvBuf SpirvBuf::alloc(unsigned long n, int dtype) {
    struct SpirvBuf b;
    b.handle = 0UL;
    b.dtype = dtype;
    b.bytes = dev_bytes_(dtype == 0 ? 0UL : 1UL, n);
    if (!spirv_available()) return b;
    b.handle = device_mem_alloc_(DEVICE_VULKAN, b.bytes);
    return b;
}

struct SpirvBuf SpirvBuf::upload(const &float host, unsigned long n) {
    struct SpirvBuf b = SpirvBuf::alloc(n, 0);
    if (b.handle != 0UL && !device_upload_(DEVICE_VULKAN, b.handle, host, n)) b.free();
    return b;
}

// Buffer b holds n fp32 values.
static int spv_holds_(struct SpirvBuf b, unsigned long n) { return b.handle != 0UL && b.bytes >= 4UL * n; }

// out = op(x, y) elementwise (k: the op's constant; write: an accumulating op's u1)
static int spv_ew_(int op, unsigned long x, unsigned long y, unsigned long out, unsigned long n, float k, unsigned int write) {
    struct DevArgs a = dev_args_();
    unsafe { a.p[0] = x; a.p[1] = y; a.p[2] = out; a.u[0] = (unsigned int)op; a.u[1] = write; a.f[0] = k; }
    return device_launch_(DEVICE_VULKAN, DK_EW, n, &a);
}

static int spv_binary_(int op, struct SpirvBuf a, struct SpirvBuf b, struct SpirvBuf out, unsigned long n, unsigned int write) {
    if (!spirv_available() || !spv_holds_(a, n) || !spv_holds_(b, n) || !spv_holds_(out, n)) return 0;
    return spv_ew_(op, a.handle, b.handle, out.handle, n, 0.0f, write);
}

static int spv_unary_(int op, struct SpirvBuf a, struct SpirvBuf out, unsigned long n, float k) {
    if (!spirv_available() || !spv_holds_(a, n) || !spv_holds_(out, n)) return 0;
    return spv_ew_(op, a.handle, 0UL, out.handle, n, k, 0U);
}

int spirv_add(struct SpirvBuf a, struct SpirvBuf b, struct SpirvBuf out, unsigned long n) { return spv_binary_(SPV_EW_ADD, a, b, out, n, 0U); }
int spirv_sub(struct SpirvBuf a, struct SpirvBuf b, struct SpirvBuf out, unsigned long n) { return spv_binary_(SPV_EW_SUB, a, b, out, n, 0U); }
int spirv_mul(struct SpirvBuf a, struct SpirvBuf b, struct SpirvBuf out, unsigned long n) { return spv_binary_(SPV_EW_MUL, a, b, out, n, 0U); }
int spirv_div(struct SpirvBuf a, struct SpirvBuf b, struct SpirvBuf out, unsigned long n) { return spv_binary_(SPV_EW_ACC_LOGB, a, b, out, n, 1U); }
int spirv_relu(struct SpirvBuf a, struct SpirvBuf out, unsigned long n) { return spv_unary_(SPV_EW_RELU, a, out, n, 0.0f); }
int spirv_log(struct SpirvBuf a, struct SpirvBuf out, unsigned long n) { return spv_unary_(SPV_EW_LOG, a, out, n, 0.0f); }
int spirv_exp(struct SpirvBuf a, struct SpirvBuf out, unsigned long n) { return spv_unary_(SPV_EW_EXP, a, out, n, 0.0f); }
int spirv_scale(struct SpirvBuf a, float k, struct SpirvBuf out, unsigned long n) { return spv_unary_(SPV_EW_SCALE, a, out, n, k); }

int spirv_pow(struct SpirvBuf a, struct SpirvBuf b, struct SpirvBuf out, unsigned long n) {
    // out = exp(b log a), in place in out (b read before out is written where they alias)
    if (!spirv_available() || !spv_holds_(a, n) || !spv_holds_(b, n) || !spv_holds_(out, n)) return 0;
    if (out.handle == b.handle) {
        struct SpirvBuf t = SpirvBuf::alloc(n, 0);
        if (!t.valid()) return 0;
        int ok = spv_ew_(SPV_EW_LOG, a.handle, 0UL, t.handle, n, 0.0f, 0U) &&
                 spv_ew_(SPV_EW_MUL, t.handle, b.handle, out.handle, n, 0.0f, 0U) &&
                 spv_ew_(SPV_EW_EXP, out.handle, 0UL, out.handle, n, 0.0f, 0U);
        t.free();
        return ok;
    }
    return spv_ew_(SPV_EW_LOG, a.handle, 0UL, out.handle, n, 0.0f, 0U) &&
           spv_ew_(SPV_EW_MUL, out.handle, b.handle, out.handle, n, 0.0f, 0U) &&
           spv_ew_(SPV_EW_EXP, out.handle, 0UL, out.handle, n, 0.0f, 0U);
}

int spirv_sqrt(struct SpirvBuf a, struct SpirvBuf out, unsigned long n) {
    if (!spirv_available() || !spv_holds_(a, n) || !spv_holds_(out, n)) return 0;
    return spv_ew_(SPV_EW_LOG, a.handle, 0UL, out.handle, n, 0.0f, 0U) &&
           spv_ew_(SPV_EW_SCALE, out.handle, 0UL, out.handle, n, 0.5f, 0U) &&
           spv_ew_(SPV_EW_EXP, out.handle, 0UL, out.handle, n, 0.0f, 0U);
}

int spirv_sum(struct SpirvBuf a, struct SpirvBuf out, unsigned long n) {
    if (!spirv_available() || !spv_holds_(a, n) || !spv_holds_(out, 1UL) || a.handle == out.handle) return 0;
    if (!spv_ew_(SPV_EW_FILL, 0UL, 0UL, out.handle, 1UL, 0.0f, 0U)) return 0;
    if (n == 0UL) return 1;
    struct DevArgs s = dev_args_();
    unsafe { s.p[0] = a.handle; s.p[1] = out.handle; s.u[0] = 0U; }   // a sum, not of squares
    return device_launch_(DEVICE_VULKAN, DK_SUM, n, &s);
}

int spirv_matmul(struct SpirvBuf a, struct SpirvBuf b, struct SpirvBuf out, unsigned long M, unsigned long K,
                 unsigned long N) {
    if (!spirv_available() || !spv_holds_(a, M * K) || !spv_holds_(b, K * N) || !spv_holds_(out, M * N)) return 0;
    return device_gemm_(DEVICE_VULKAN, 0, 0, M, N, K, 1.0f, a.handle, K, b.handle, N, 0.0f, out.handle, N);
}

int spirv_sync() {
    if (!spirv_available()) return 0;
    return device_sync(DEVICE_VULKAN);
}

// ── Host arrays ──────────────────────────────────────────────────────────────
// Uploads the inputs (n1 and n2 values; b absent when n2 is 0), runs op on
// the buffers, reads nOut values back.
typedef fn int(struct SpirvBuf, struct SpirvBuf, struct SpirvBuf, unsigned long) SpvBinaryOp;

static int spv_host_binary_(SpvBinaryOp op, const &float a, const &float b, &float out, unsigned long n) {
    if (!spirv_available()) return 0;
    if (n == 0UL) return 1;
    struct SpirvBuf ba = SpirvBuf::upload(a, n);
    struct SpirvBuf bb = SpirvBuf::upload(b, n);
    struct SpirvBuf bo = SpirvBuf::alloc(n, 0);
    int ok = ba.valid() && bb.valid() && bo.valid() && op(ba, bb, bo, n);
    if (ok) { unsafe { ok = bo.read((&void)out, 4UL * n); } }
    ba.free(); bb.free(); bo.free();
    return ok;
}

// The unary ops as binary ones (b unused), with spirv_scale's constant in g_spv_k.
static float g_spv_k = 0.0f;
static int spv_relu2_(struct SpirvBuf a, struct SpirvBuf b, struct SpirvBuf o, unsigned long n) { return spirv_relu(a, o, n); }
static int spv_log2_(struct SpirvBuf a, struct SpirvBuf b, struct SpirvBuf o, unsigned long n) { return spirv_log(a, o, n); }
static int spv_exp2_(struct SpirvBuf a, struct SpirvBuf b, struct SpirvBuf o, unsigned long n) { return spirv_exp(a, o, n); }
static int spv_sqrt2_(struct SpirvBuf a, struct SpirvBuf b, struct SpirvBuf o, unsigned long n) { return spirv_sqrt(a, o, n); }
static int spv_scale2_(struct SpirvBuf a, struct SpirvBuf b, struct SpirvBuf o, unsigned long n) { return spirv_scale(a, g_spv_k, o, n); }

static int spv_host_unary_(SpvBinaryOp op, const &float a, &float out, unsigned long n) {
    if (!spirv_available()) return 0;
    if (n == 0UL) return 1;
    struct SpirvBuf ba = SpirvBuf::upload(a, n);
    struct SpirvBuf bo = SpirvBuf::alloc(n, 0);
    int ok = ba.valid() && bo.valid() && op(ba, ba, bo, n);
    if (ok) { unsafe { ok = bo.read((&void)out, 4UL * n); } }
    ba.free(); bo.free();
    return ok;
}

int spirv_add_f32(const &float a, const &float b, &float out, unsigned long n) { return spv_host_binary_(spirv_add, a, b, out, n); }
int spirv_sub_f32(const &float a, const &float b, &float out, unsigned long n) { return spv_host_binary_(spirv_sub, a, b, out, n); }
int spirv_mul_f32(const &float a, const &float b, &float out, unsigned long n) { return spv_host_binary_(spirv_mul, a, b, out, n); }
int spirv_div_f32(const &float a, const &float b, &float out, unsigned long n) { return spv_host_binary_(spirv_div, a, b, out, n); }
int spirv_pow_f32(const &float a, const &float b, &float out, unsigned long n) { return spv_host_binary_(spirv_pow, a, b, out, n); }
int spirv_relu_f32(const &float a, &float out, unsigned long n) { return spv_host_unary_(spv_relu2_, a, out, n); }
int spirv_log_f32(const &float a, &float out, unsigned long n) { return spv_host_unary_(spv_log2_, a, out, n); }
int spirv_exp_f32(const &float a, &float out, unsigned long n) { return spv_host_unary_(spv_exp2_, a, out, n); }
int spirv_sqrt_f32(const &float a, &float out, unsigned long n) { return spv_host_unary_(spv_sqrt2_, a, out, n); }

int spirv_scale_f32(const &float a, float k, &float out, unsigned long n) {
    g_spv_k = k;
    return spv_host_unary_(spv_scale2_, a, out, n);
}

int spirv_sum_f32(const &float a, &float out, unsigned long n) {
    if (!spirv_available()) return 0;
    struct SpirvBuf ba = SpirvBuf::upload(a, n == 0UL ? 1UL : n);
    struct SpirvBuf bo = SpirvBuf::alloc(1UL, 0);
    int ok = (n == 0UL || ba.valid()) && bo.valid() && spirv_sum(ba, bo, n);
    if (ok) { unsafe { ok = bo.read((&void)out, 4UL); } }
    ba.free(); bo.free();
    return ok;
}

int spirv_matmul_f32(const &float a, const &float b, &float out,
                      unsigned long M, unsigned long K, unsigned long N) {
    if (!spirv_available()) return 0;
    if (M == 0UL || N == 0UL) return 1;
    struct SpirvBuf ba = SpirvBuf::upload(a, M * K == 0UL ? 1UL : M * K);
    struct SpirvBuf bb = SpirvBuf::upload(b, K * N == 0UL ? 1UL : K * N);
    struct SpirvBuf bo = SpirvBuf::alloc(M * N, 0);
    int ok = ba.valid() && bb.valid() && bo.valid() && spirv_matmul(ba, bb, bo, M, K, N);
    if (ok) { unsafe { ok = bo.read((&void)out, 4UL * M * N); } }
    ba.free(); bb.free(); bo.free();
    return ok;
}

} // namespace std
