// SafeC Standard Library — the device kernels behind std::ml's tensor ops on
// ROCm (compiled at run time by hiprtc), CUDA (NVRTC) and Metal
// (newLibraryWithSource): one source for all three. gen_device_kernels.sh
// turns this file into device_kernels.h (the source as a string and the
// kernel ids); edit this file, then regenerate.
//
// Every kernel takes one DevArgs (device.h): device addresses in p[],
// integers in u[], floats in f[]. It runs over n threads (n passed as u[7])
// in blocks of 256; per-row kernels use one thread per row.

// ── Portability: CUDA / HIP and Metal ────────────────────────────────────────
// The source is CUDA C++; as Metal Shading Language (C++14 with address
// spaces) a few things differ, and the macros below absorb them, expanding
// to nothing (or to the CUDA spelling) for hiprtc and NVRTC:
//   - pointers and references name the memory they point into: dk_global
//     (device memory), dk_shared (a block's shared memory), dk_thread (a
//     thread's own variables);
//   - Metal has no 64-bit 'long long' (DK_U64) and no double (dk_double, a
//     wide accumulator: float there; the reductions it serves stay
//     two-pass, so fp32 holds the tolerance);
//   - a kernel receives its arguments in constant memory and its thread
//     indices as parameters (DK_KERNEL declares both, under the CUDA names
//     threadIdx, blockIdx, blockDim and gridDim), so a helper that reads
//     them takes DK_IDS after its parameters and is called with DK_IDA;
//   - block reductions (block_add, block_sum_d) use simdgroup sums and a
//     small threadgroup scratch the kernel declares (dk_red);
//   - attention tiles are 8 x 8 instead of 16 x 16 (ATTN_T): Metal's 32 KB
//     of threadgroup memory doesn't hold four 16-row tiles of 128 floats;
//     a second build with DK_ATTN64 keeps 16 x 16 for head dims up to 64;
//   - integer division by a run-time value costs dozens of cycles on Apple
//     GPUs, so the hot index arithmetic divides with DK_UDIV / DK_UMOD: on
//     Metal through a float reciprocal made exact by correction (dk_udiv,
//     the reciprocal declared by DK_RECIP), elsewhere plain / and %;
//   - the kernels with a thread per row (softmax, layer norm, cross-entropy,
//     row sums) take a simdgroup per row on Metal (GSR, MpsBackend launches
//     32 threads per row), its lanes splitting the columns: a few hundred
//     rows are too few threads to keep an Apple GPU busy.
#if defined(__METAL_VERSION__)
#include <metal_stdlib>
using namespace metal;
#define DK_U64 ulong
#define dk_double float
#define dk_global device
#define dk_shared threadgroup
#define dk_thread thread
#define __device__
#define __forceinline__ inline __attribute__((always_inline))
#define __shared__ threadgroup
#define __syncthreads() threadgroup_barrier(mem_flags::mem_threadgroup)
struct DevArgs_ {
    ulong p[8];
    unsigned int u[8];
    float f[8];
};
// const DevArgs& names the kernel's argument struct in constant memory.
#define DevArgs constant DevArgs_
#define DK_KERNEL(name) kernel void name(constant DevArgs_& a [[buffer(0)]], \
    uint3 threadIdx [[thread_position_in_threadgroup]], uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint3 blockDim [[threads_per_threadgroup]], uint3 gridDim [[threadgroups_per_grid]], \
    uint dk_lane [[thread_index_in_simdgroup]], uint dk_simd [[simdgroup_index_in_threadgroup]], \
    threadgroup float* dk_red [[threadgroup(0)]])
#define DK_IDS , uint3 threadIdx, uint3 blockIdx, uint3 blockDim, uint3 gridDim, uint dk_lane, uint dk_simd, \
    threadgroup float* dk_red
#define DK_IDA , threadIdx, blockIdx, blockDim, gridDim, dk_lane, dk_simd, dk_red
// float atomics and the C math library's float functions
inline void atomicAdd(device float* p, float v) { atomic_fetch_add_explicit((device atomic_float*)p, v, memory_order_relaxed); }
inline float expf(float x) { return metal::exp(x); }
inline float logf(float x) { return metal::log(x); }
inline float tanhf(float x) { return metal::tanh(x); }
inline float sqrtf(float x) { return metal::sqrt(x); }
inline float rsqrtf(float x) { return metal::rsqrt(x); }
inline float fabsf(float x) { return metal::fabs(x); }
inline float floorf(float x) { return metal::floor(x); }
inline float ceilf(float x) { return metal::ceil(x); }
inline float fminf(float x, float y) { return metal::fmin(x, y); }
inline float fmaxf(float x, float y) { return metal::fmax(x, y); }
inline float fmaf(float x, float y, float z) { return metal::fma(x, y, z); }
inline unsigned int __float_as_uint(float f) { return as_type<uint>(f); }
inline float __uint_as_float(unsigned int u) { return as_type<float>(u); }
inline unsigned int dk_udiv(unsigned int q, unsigned int d, float inv) {
    unsigned int r = (unsigned int)((float)q * inv);
    while (r * d > q) r--;
    while ((r + 1u) * d <= q) r++;
    return r;
}
#define DK_RECIP(inv, d) float inv = 1.0f / (float)(d);
#define DK_UDIV(q, d, inv) dk_udiv((q), (d), (inv))
#define DK_UMOD(q, d, inv) ((q) - dk_udiv((q), (d), (inv)) * (d))
// rows r for this simdgroup (a grid of 32 threads per row); dk_lane splits the columns
#define GSR(r, n) for (unsigned int r = (blockIdx.x * blockDim.x + threadIdx.x) / 32u; r < (n); r += blockDim.x * gridDim.x / 32u)
#else
#define DK_U64 unsigned long long
#define dk_double double
#define dk_global
#define dk_shared
#define dk_thread
#define DK_KERNEL(name) extern "C" __global__ void name(DevArgs a)
#define DK_IDS
#define DK_IDA
#define DK_RECIP(inv, d)
#define DK_UDIV(q, d, inv) q / d
#define DK_UMOD(q, d, inv) q % d
struct DevArgs {
    DK_U64 p[8];
    unsigned int u[8];
    float f[8];
};
#endif

#define GS(i, n) for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// ── Element types ────────────────────────────────────────────────────────────
// A buffer's element type travels in its handle's top two bits (device.h's
// DEV_TAG_*): 0 fp32, 1 fp16, 2 bf16 (activations stored in 16 bits). F(k)
// views argument k's buffer: loads widen to float, stores round to its type,
// so every kernel reads and writes any of them. Atomics target fp32 buffers
// (gradients, sums) through FA(k). The source compiles twice: with
// DK_TAGGED for launches with a 16-bit argument, and plain (views are bare
// float pointers) for the rest.
#define DEV_ADDR_MASK 0x3fffffffffffffffULL
__device__ unsigned short f2h_bits(float f);
__device__ float h2f_bits(unsigned short h);
__device__ unsigned short f2b_bits(float f);
__device__ float b2f_bits(unsigned short b);
#ifdef DK_TAGGED
struct Ref {
    DK_U64 p;   // the element's address
    unsigned int tag;
    __device__ float get() const {
        if (tag == 0u) return *(const dk_global float*)p;
        unsigned short h = *(const dk_global unsigned short*)p;
        return tag == 1u ? h2f_bits(h) : b2f_bits(h);
    }
    __device__ void set(float v) const {
        if (tag == 0u) *(dk_global float*)p = v;
        else *(dk_global unsigned short*)p = tag == 1u ? f2h_bits(v) : f2b_bits(v);
    }
    __device__ operator float() const { return get(); }
    __device__ const dk_thread Ref& operator=(float v) const { set(v); return *this; }
    __device__ const dk_thread Ref& operator=(const dk_thread Ref& o) const { set(o.get()); return *this; }
    __device__ const dk_thread Ref& operator+=(float v) const { set(get() + v); return *this; }
    __device__ const dk_thread Ref& operator-=(float v) const { set(get() - v); return *this; }
};
struct TP {
    DK_U64 p;   // tagged handle
    __device__ TP(DK_U64 h) : p(h) {}
    __device__ unsigned int tag() const { return (unsigned int)(p >> 62); }
    __device__ DK_U64 base() const { return p & DEV_ADDR_MASK; }
    __device__ Ref operator[](DK_U64 i) const {
        Ref r; r.tag = tag(); r.p = base() + i * (r.tag == 0u ? 4ull : 2ull); return r;
    }
    __device__ TP operator+(DK_U64 k) const { return TP(p + k * (tag() == 0u ? 4ull : 2ull)); }
    __device__ explicit operator bool() const { return base() != 0ull; }
};
#else
// The plain build (no argument is 16-bit): views are float pointers.
struct TP {
    DK_U64 p;
    __device__ TP(DK_U64 h) : p(h) {}
    __device__ dk_global float& operator[](DK_U64 i) const { return ((dk_global float*)p)[i]; }
    __device__ TP operator+(DK_U64 k) const { return TP(p + k * 4ull); }
    __device__ explicit operator bool() const { return p != 0ull; }
};
#endif
#define F(k) TP(a.p[k])
#define CF(k) TP(a.p[k])
#define FA(k) ((dk_global float*)(a.p[k] & DEV_ADDR_MASK))
#define N (a.u[7])

// Gradient kernels that add to their output take u6 = 1 to write it instead
// (it was an unwritten zero gradient): ACCUM(out, v) is out (+)= v.
#define ACCUM(o, v) do { if (a.u[6]) (o) = (v); else (o) += (v); } while (0)

// ── Elementwise: op codes match tensor.h's EW_* ─────────────────────────────
// p0 = in a, p1 = in b, p2 = out, p3 = aux (a third operand); f0 = k, f1 = k2.
// The accumulating forms (EW_ACC .. and 40 ..) add to out, or with u1 = 1
// write it (out was all zeros: an unwritten gradient).
#define EW_ADD        0
#define EW_SUB        1
#define EW_MUL        2
#define EW_SCALE      3
#define EW_RELU       4
#define EW_FILL       5
#define EW_COPY       6
#define EW_ACC        7
#define EW_DEC        8
#define EW_ACC_MUL    9
#define EW_ACC_SCALE  10
#define EW_ACC_RELU   11
#define EW_ACC_K      12
#define EW_ACC_SIGB   13
#define EW_ACC_TANHB  14
#define EW_EXP        20
#define EW_LOG        21
#define EW_ABS        22
#define EW_LEAKY      23
#define EW_CLAMP      24
#define EW_SIGMOID    25
#define EW_TANH       26
#define EW_GELU       27
#define EW_SILU       28
#define EW_ADDK       29
#define EW_MIN        30
#define EW_ELU        31
#define EW_LERP       32   // out += k * (a - out), in place (a soft update)
#define EW_MULS       33   // out = a * b[0] (b a device scalar)
#define EW_ACC_PK     40   // out += *b (a device scalar: a seed gradient)
#define EW_ACC_EXPB   41   // out += g * y            (a = g, b = y)
#define EW_ACC_LOGB   42   // out += g / x            (a = g, b = x)
#define EW_ACC_ABSB   43   // out += g * sign(x)
#define EW_ACC_LEAKYB 44   // out += g * (x > 0 ? 1 : k)
#define EW_ACC_CLAMPB 45   // out += x == y ? g : 0   (aux = y)
#define EW_ACC_GELUB  46   // out += g * gelu'(x)
#define EW_ACC_SILUB  47   // out += g * silu'(x)
#define EW_ACC_MINA   48   // out += x <= other ? g : 0 (b = x, aux = other)
#define EW_ACC_MINB   49   // out += x <  other ? 0 : g ... (b = x of 'a' side, aux = this side)
#define EW_ACC_ELUB   50   // out += g * (x > 0 ? 1 : y + k) (b = x, aux = y)
#define EW_ACC_PSCALE 51   // out += *b * k * ... (b device scalar seed): out += k * b[0]

__device__ float gelu_f(float x) {
    float c = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}
__device__ float gelu_d(float x) {
    float c = 0.7978845608028654f;
    float t = tanhf(c * (x + 0.044715f * x * x * x));
    float du = c * (1.0f + 0.134145f * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du;
}
__device__ float sig_f(float x) { return 1.0f / (1.0f + expf(-x)); }

// One elementwise result from the operands' values (y0: b[0] for the ops
// reading a device scalar); ow: the accumulating forms write.
__device__ __forceinline__ float ew_val(int op, float x, float y, float z, float o, float k, float k2, bool ow) {
    float ob = ow ? 0.0f : o;
    switch (op) {
    case EW_ADD: return x + y;
    case EW_SUB: return x - y;
    case EW_MUL: return x * y;
    case EW_SCALE: return x * k;
    case EW_RELU: return x > 0.0f ? x : 0.0f;
    case EW_FILL: return k;
    case EW_COPY: return x;
    case EW_ACC: return ob + x;
    case EW_DEC: return ob - x;
    case EW_ACC_MUL: return fmaf(x, y, ob);
    case EW_ACC_SCALE: return fmaf(x, k, ob);
    case EW_ACC_RELU: return ob + (y > 0.0f ? x : 0.0f);
    case EW_ACC_K: return ob + k;
    case EW_ACC_SIGB: return fmaf(x * y, 1.0f - y, ob);
    case EW_ACC_TANHB: return fmaf(x, 1.0f - y * y, ob);
    case EW_EXP: return expf(x);
    case EW_LOG: return logf(x);
    case EW_ABS: return fabsf(x);
    case EW_LEAKY: return x > 0.0f ? x : k * x;
    case EW_CLAMP: return fminf(fmaxf(x, k), k2);
    case EW_SIGMOID: return sig_f(x);
    case EW_TANH: return tanhf(x);
    case EW_GELU: return gelu_f(x);
    case EW_SILU: return x * sig_f(x);
    case EW_ADDK: return x + k;
    case EW_MIN: return x <= y ? x : y;
    case EW_ELU: return x > 0.0f ? x : k * (expf(x) - 1.0f);
    case EW_LERP: return fmaf(k, x - o, o);
    case EW_MULS: return x * y;
    case EW_ACC_PK: return ob + y;
    case EW_ACC_EXPB: return fmaf(x, y, ob);
    case EW_ACC_LOGB: return ob + x / y;
    case EW_ACC_ABSB: return ob + (y > 0.0f ? x : (y < 0.0f ? -x : 0.0f));
    case EW_ACC_LEAKYB: return ob + x * (y > 0.0f ? 1.0f : k);
    case EW_ACC_CLAMPB: return ob + (y == z ? x : 0.0f);
    case EW_ACC_GELUB: return fmaf(x, gelu_d(y), ob);
    case EW_ACC_SILUB: { float s = sig_f(y); return fmaf(x, s + y * s * (1.0f - s), ob); }
    case EW_ACC_MINA: return ob + (y <= z ? x : 0.0f);
    case EW_ACC_MINB: return ob + (!(z <= y) ? x : 0.0f);
    case EW_ACC_ELUB: return ob + x * (y > 0.0f ? 1.0f : z + k);
    case EW_ACC_PSCALE: return ob + k * y;
    }
    return 0.0f;
}
// Which operands an op reads: x, y per element, y as a device scalar, z, o.
__device__ __forceinline__ bool ew_uses_x(int op) { return op != EW_FILL && op != EW_ACC_K && op != EW_ACC_PK && op != EW_ACC_PSCALE; }
__device__ __forceinline__ bool ew_y_scalar(int op) { return op == EW_MULS || op == EW_ACC_PK || op == EW_ACC_PSCALE; }
__device__ __forceinline__ bool ew_uses_y(int op) {
    return op == EW_ADD || op == EW_SUB || op == EW_MUL || op == EW_ACC_MUL || op == EW_ACC_RELU || op == EW_ACC_SIGB ||
           op == EW_ACC_TANHB || op == EW_MIN || (op >= EW_ACC_EXPB && op <= EW_ACC_ELUB);
}
__device__ __forceinline__ bool ew_uses_z(int op) { return op == EW_ACC_CLAMPB || op == EW_ACC_MINA || op == EW_ACC_MINB || op == EW_ACC_ELUB; }
__device__ __forceinline__ bool ew_reads_o(int op, bool ow) {
    return op == EW_LERP || (!ow && ((op >= EW_ACC && op <= EW_ACC_TANHB) || (op >= EW_ACC_PK && op <= EW_ACC_PSCALE)));
}

#ifdef DK_TAGGED
// Two 16-bit values of type t from one 32-bit word, and back.
__device__ __forceinline__ float h16_lo(unsigned int w, unsigned int t) { unsigned short h = (unsigned short)(w & 0xffffu); return t == 1u ? h2f_bits(h) : b2f_bits(h); }
__device__ __forceinline__ float h16_hi(unsigned int w, unsigned int t) { unsigned short h = (unsigned short)(w >> 16); return t == 1u ? h2f_bits(h) : b2f_bits(h); }
__device__ __forceinline__ unsigned int h16_pack(float lo, float hi, unsigned int t) {
    unsigned int a = t == 1u ? f2h_bits(lo) : f2b_bits(lo), b = t == 1u ? f2h_bits(hi) : f2b_bits(hi);
    return a | (b << 16);
}
// Every element operand of type t (the output's) and 4-byte aligned: two
// values per thread through 32-bit words.
__device__ bool ew_pairable(const DevArgs& a, int op, bool ow, unsigned int t) {
    if (t == 0u) return false;
    DK_U64 ps[4];
    bool use[4];
    ps[0] = a.p[0]; use[0] = ew_uses_x(op);
    ps[1] = a.p[1]; use[1] = ew_uses_y(op);
    ps[2] = a.p[2]; use[2] = true;
    ps[3] = a.p[3]; use[3] = ew_uses_z(op);
    for (int q = 0; q < 4; q++)
        if (use[q] && ((unsigned int)(ps[q] >> 62) != t || (ps[q] & 3ull) != 0ull)) return false;
    return true;
}
#endif

// One instantiation per op: a tight loop, no per-element dispatch.
template <int OPC>
__device__ __forceinline__ void ew_run(const DevArgs& a DK_IDS) {
    const int OP = OPC >= 0 ? OPC : (int)a.u[0];
    TP x = CF(0); TP y = CF(1); TP o = F(2); TP z = CF(3);
    float k = a.f[0], k2 = a.f[1];
    bool ow = a.u[1] != 0;
    const bool ux = ew_uses_x(OP), uy = ew_uses_y(OP), ys = ew_y_scalar(OP), uz = ew_uses_z(OP);
    bool uo = ew_reads_o(OP, ow);
    float y0 = ys ? (float)y[0] : 0.0f;
#ifdef DK_TAGGED
    unsigned int t = (unsigned int)(a.p[2] >> 62);
    if (ew_pairable(a, OP, ow, t)) {
        const dk_global unsigned int* xw = (const dk_global unsigned int*)(a.p[0] & DEV_ADDR_MASK);
        const dk_global unsigned int* yw = (const dk_global unsigned int*)(a.p[1] & DEV_ADDR_MASK);
        const dk_global unsigned int* zw = (const dk_global unsigned int*)(a.p[3] & DEV_ADDR_MASK);
        dk_global unsigned int* ow_ = (dk_global unsigned int*)(a.p[2] & DEV_ADDR_MASK);
        unsigned int n2 = N / 2u;
        GS(j, n2) {
            unsigned int xv = ux ? xw[j] : 0u, yv = uy ? yw[j] : 0u, zv = uz ? zw[j] : 0u, ov = uo ? ow_[j] : 0u;
            float lo, hi;
            if (t == 1u) {
                lo = ew_val(OP, h16_lo(xv, 1u), uy ? h16_lo(yv, 1u) : y0, h16_lo(zv, 1u), h16_lo(ov, 1u), k, k2, ow);
                hi = ew_val(OP, h16_hi(xv, 1u), uy ? h16_hi(yv, 1u) : y0, h16_hi(zv, 1u), h16_hi(ov, 1u), k, k2, ow);
                ow_[j] = h16_pack(lo, hi, 1u);
            } else {
                lo = ew_val(OP, h16_lo(xv, 2u), uy ? h16_lo(yv, 2u) : y0, h16_lo(zv, 2u), h16_lo(ov, 2u), k, k2, ow);
                hi = ew_val(OP, h16_hi(xv, 2u), uy ? h16_hi(yv, 2u) : y0, h16_hi(zv, 2u), h16_hi(ov, 2u), k, k2, ow);
                ow_[j] = h16_pack(lo, hi, 2u);
            }
        }
        if ((N & 1u) && blockIdx.x == 0 && threadIdx.x == 0) {
            unsigned int i = N - 1u;
            o[i] = ew_val(OP, ux ? (float)x[i] : 0.0f, uy ? (float)y[i] : y0, uz ? (float)z[i] : 0.0f,
                          uo ? (float)o[i] : 0.0f, k, k2, ow);
        }
        return;
    }
#endif
    GS(i, N) {
        o[i] = ew_val(OP, ux ? (float)x[i] : 0.0f, uy ? (float)y[i] : y0, uz ? (float)z[i] : 0.0f,
                      uo ? (float)o[i] : 0.0f, k, k2, ow);
    }
}

// Any op (u0); dk_ewop_<code> below run one op each (device.sc picks them).
DK_KERNEL(dk_ew) { ew_run<-1>(a DK_IDA); }
DK_KERNEL(dk_ewop_0) { ew_run<0>(a DK_IDA); }
DK_KERNEL(dk_ewop_1) { ew_run<1>(a DK_IDA); }
DK_KERNEL(dk_ewop_2) { ew_run<2>(a DK_IDA); }
DK_KERNEL(dk_ewop_3) { ew_run<3>(a DK_IDA); }
DK_KERNEL(dk_ewop_4) { ew_run<4>(a DK_IDA); }
DK_KERNEL(dk_ewop_5) { ew_run<5>(a DK_IDA); }
DK_KERNEL(dk_ewop_6) { ew_run<6>(a DK_IDA); }
DK_KERNEL(dk_ewop_7) { ew_run<7>(a DK_IDA); }
DK_KERNEL(dk_ewop_8) { ew_run<8>(a DK_IDA); }
DK_KERNEL(dk_ewop_9) { ew_run<9>(a DK_IDA); }
DK_KERNEL(dk_ewop_10) { ew_run<10>(a DK_IDA); }
DK_KERNEL(dk_ewop_11) { ew_run<11>(a DK_IDA); }
DK_KERNEL(dk_ewop_12) { ew_run<12>(a DK_IDA); }
DK_KERNEL(dk_ewop_13) { ew_run<13>(a DK_IDA); }
DK_KERNEL(dk_ewop_14) { ew_run<14>(a DK_IDA); }
DK_KERNEL(dk_ewop_20) { ew_run<20>(a DK_IDA); }
DK_KERNEL(dk_ewop_21) { ew_run<21>(a DK_IDA); }
DK_KERNEL(dk_ewop_22) { ew_run<22>(a DK_IDA); }
DK_KERNEL(dk_ewop_23) { ew_run<23>(a DK_IDA); }
DK_KERNEL(dk_ewop_24) { ew_run<24>(a DK_IDA); }
DK_KERNEL(dk_ewop_25) { ew_run<25>(a DK_IDA); }
DK_KERNEL(dk_ewop_26) { ew_run<26>(a DK_IDA); }
DK_KERNEL(dk_ewop_27) { ew_run<27>(a DK_IDA); }
DK_KERNEL(dk_ewop_28) { ew_run<28>(a DK_IDA); }
DK_KERNEL(dk_ewop_29) { ew_run<29>(a DK_IDA); }
DK_KERNEL(dk_ewop_30) { ew_run<30>(a DK_IDA); }
DK_KERNEL(dk_ewop_31) { ew_run<31>(a DK_IDA); }
DK_KERNEL(dk_ewop_32) { ew_run<32>(a DK_IDA); }
DK_KERNEL(dk_ewop_33) { ew_run<33>(a DK_IDA); }
DK_KERNEL(dk_ewop_40) { ew_run<40>(a DK_IDA); }
DK_KERNEL(dk_ewop_41) { ew_run<41>(a DK_IDA); }
DK_KERNEL(dk_ewop_42) { ew_run<42>(a DK_IDA); }
DK_KERNEL(dk_ewop_43) { ew_run<43>(a DK_IDA); }
DK_KERNEL(dk_ewop_44) { ew_run<44>(a DK_IDA); }
DK_KERNEL(dk_ewop_45) { ew_run<45>(a DK_IDA); }
DK_KERNEL(dk_ewop_46) { ew_run<46>(a DK_IDA); }
DK_KERNEL(dk_ewop_47) { ew_run<47>(a DK_IDA); }
DK_KERNEL(dk_ewop_48) { ew_run<48>(a DK_IDA); }
DK_KERNEL(dk_ewop_49) { ew_run<49>(a DK_IDA); }
DK_KERNEL(dk_ewop_50) { ew_run<50>(a DK_IDA); }
DK_KERNEL(dk_ewop_51) { ew_run<51>(a DK_IDA); }

// ── Reductions (256 threads per block, one atomic per block) ─────────────────
__device__ void block_add(float v, dk_global float* out DK_IDS) {
#if defined(__METAL_VERSION__)
    // each simdgroup's sum, then the first simdgroup adds those
    v = simd_sum(v);
    if (dk_lane == 0u) dk_red[dk_simd] = v;
    __syncthreads();
    if (dk_simd == 0u) {
        float s = simd_sum(threadIdx.x < (blockDim.x + 31u) / 32u ? dk_red[threadIdx.x] : 0.0f);
        if (threadIdx.x == 0u) atomicAdd(out, s);
    }
#else
    __shared__ float s[256];
    s[threadIdx.x] = v;
    __syncthreads();
    for (unsigned int w = 128; w > 0; w >>= 1) {
        if (threadIdx.x < w) s[threadIdx.x] += s[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, s[0]);
#endif
}

// p0 = in, p1 = out[0] (zeroed by the caller); u0: 0 sum, 1 sum of squares
DK_KERNEL(dk_sum) {
    TP x = CF(0);
    float acc = 0.0f;
    GS(i, N) { float v = x[i]; acc += a.u[0] ? v * v : v; }
    block_add(acc, FA(1) DK_IDA);
}

// ── Rows and columns: u0 = cols (rows = n / cols for row-shaped n) ──────────
// out[r,c] = x[r,c] + b[c]                                    n = rows*cols
DK_KERNEL(dk_bias_rows) {
    unsigned int C = a.u[0];
    GS(i, N) F(2)[i] = CF(0)[i] + CF(1)[i % C];
}
// out[r,c] = x[r,c] * g[c]                                    n = rows*cols
DK_KERNEL(dk_mul_row) {
    unsigned int C = a.u[0];
    GS(i, N) F(2)[i] = CF(0)[i] * CF(1)[i % C];
}
// gb[c] += sum_r d[r,c] (* x[r,c] when p2 != 0); u0 = cols, u1 = rows, u2 = S
// row slices (0 or 1: one). Units are (256 columns, row slice); a thread per
// column.                                                    n = ceil(cols/256)*S*256
DK_KERNEL(dk_colsum_acc) {
    unsigned int C = a.u[0], R = a.u[1], S = a.u[2] > 1 ? a.u[2] : 1;
    TP d = CF(0); dk_global float* gb = FA(1); TP x = CF(2);
    for (unsigned int u = blockIdx.x; u < ((C + 255) / 256) * S; u += gridDim.x) {
        unsigned int c = (u / S) * 256 + threadIdx.x, sl = u % S;
        if (c < C) {
            float s = 0.0f;
            if (x) { for (unsigned int r = sl; r < R; r += S) s += d[r * C + c] * x[r * C + c]; }
            else   { for (unsigned int r = sl; r < R; r += S) s += d[r * C + c]; }
            if (S > 1) atomicAdd(&gb[c], s); else gb[c] += s;
        }
    }
}
// dx[r,c] += d[r,c] * g[c]                                    n = rows*cols
DK_KERNEL(dk_mul_row_acc) {
    unsigned int C = a.u[0];
    GS(i, N) ACCUM(F(2)[i], CF(0)[i] * CF(1)[i % C]);
}
// out[r] = sum_c x[r,c]                                       n = rows
DK_KERNEL(dk_rowsum) {
    unsigned int C = a.u[0];
#if defined(__METAL_VERSION__)
    GSR(r, N) {
        float s = 0.0f;
        for (unsigned int c = dk_lane; c < C; c += 32u) s += CF(0)[r * C + c];
        s = simd_sum(s);
        if (dk_lane == 0u) F(1)[r] = s;
    }
#else
    GS(r, N) { float s = 0.0f; for (unsigned int c = 0; c < C; c++) s += CF(0)[r * C + c]; F(1)[r] = s; }
#endif
}
// dx[r,c] += g[r]                                             n = rows*cols
DK_KERNEL(dk_rowbcast_acc) {
    unsigned int C = a.u[0];
    GS(i, N) ACCUM(F(1)[i], CF(0)[i / C]);
}
// out [rows, p+q] = x [rows, p] ++ y [rows, q]; u0 = p, u1 = q   n = rows*(p+q)
DK_KERNEL(dk_concat) {
    unsigned int P = a.u[0], Q = a.u[1], W = P + Q;
    GS(i, N) { unsigned int r = i / W, c = i % W; F(2)[i] = c < P ? CF(0)[r * P + c] : CF(1)[r * Q + c - P]; }
}
// Columns [start, start+len) of a wide [rows, cols] tensor and a narrow
// [rows, len] one; u0 = cols, u1 = start, u2 = len, u3 = mode:
//   0  narrow (p1)  = wide (p0) columns          (slice)
//   1  wide (p1)   += narrow (p0)                (slice backward)
//   2  narrow (p1) += wide (p0) columns          (concat backward)   n = rows*len
DK_KERNEL(dk_slice) {
    unsigned int C = a.u[0], S = a.u[1], L = a.u[2];
    GS(i, N) {
        unsigned int r = i / L, c = i % L;
        if (a.u[3] == 1) F(1)[r * C + S + c] += CF(0)[i];
        else if (a.u[3] == 2) ACCUM(F(1)[i], CF(0)[r * C + S + c]);
        else F(1)[i] = CF(0)[r * C + S + c];
    }
}
// out[i, :] = table[idx[i], :]; u0 = d                        n = m*d
DK_KERNEL(dk_gather_rows) {
    unsigned int D = a.u[0];
    GS(i, N) { unsigned int r = i / D, c = i % D; F(2)[i] = CF(0)[(unsigned int)CF(1)[r] * D + c]; }
}
// table.grad[idx[i], :] += g[i, :]                            n = m*d
DK_KERNEL(dk_scatter_rows) {
    unsigned int D = a.u[0];
    GS(i, N) { unsigned int r = i / D, c = i % D; atomicAdd(&FA(2)[(unsigned int)CF(1)[r] * D + c], CF(0)[i]); }
}

// ── Attention masks (attention.h) ────────────────────────────────────────────
// Query row i sits at position pos = i + (keys - queries); flags bit 0
// (causal) hides keys j > pos, bit 1 (a sliding window of w) keys j <= pos - w.
__device__ __forceinline__ bool attn_vis(unsigned int flags, unsigned int w, int pos, int j) {
    if ((flags & 1u) && j > pos) return false;
    if ((flags & 2u) && j <= pos - (int)w) return false;
    return true;
}

// ── Softmax family: one thread per row, u0 = cols ────────────────────────────
// u1: 0 softmax, 1 log-softmax. A masked softmax (u1 = 0): u2 = mask flags,
// u3 = window, u4 = (int) keys - queries; hidden entries and rows with no
// visible key come out 0.
DK_KERNEL(dk_softmax_rows) {
    unsigned int C = a.u[0];
#if defined(__METAL_VERSION__)
    GSR(r, N) {
        TP x = CF(0) + r * C; TP o = F(1) + r * C;
        unsigned int fl = a.u[2];
        int pos = (int)r + (int)a.u[4];
        float mx = -3.0e38f;
        for (unsigned int c = dk_lane; c < C; c += 32u) if (!fl || attn_vis(fl, a.u[3], pos, (int)c)) mx = fmaxf(mx, x[c]);
        mx = simd_max(mx);
        float s = 0.0f;
        for (unsigned int c = dk_lane; c < C; c += 32u) if (!fl || attn_vis(fl, a.u[3], pos, (int)c)) s += expf(x[c] - mx);
        s = simd_sum(s);
        if (fl) {
            float inv = s > 0.0f ? 1.0f / s : 0.0f;
            for (unsigned int c = dk_lane; c < C; c += 32u) o[c] = attn_vis(fl, a.u[3], pos, (int)c) ? expf(x[c] - mx) * inv : 0.0f;
        } else if (a.u[1]) {
            float l = mx + logf(s);
            for (unsigned int c = dk_lane; c < C; c += 32u) o[c] = x[c] - l;
        } else {
            float inv = 1.0f / s;
            for (unsigned int c = dk_lane; c < C; c += 32u) o[c] = expf(x[c] - mx) * inv;
        }
    }
#else
    GS(r, N) {
        TP x = CF(0) + r * C; TP o = F(1) + r * C;
        if (a.u[2]) {
            int pos = (int)r + (int)a.u[4];
            float mx = -3.0e38f;
            for (unsigned int c = 0; c < C; c++) if (attn_vis(a.u[2], a.u[3], pos, (int)c)) mx = fmaxf(mx, x[c]);
            float s = 0.0f;
            for (unsigned int c = 0; c < C; c++) if (attn_vis(a.u[2], a.u[3], pos, (int)c)) s += expf(x[c] - mx);
            float inv = s > 0.0f ? 1.0f / s : 0.0f;
            for (unsigned int c = 0; c < C; c++) o[c] = attn_vis(a.u[2], a.u[3], pos, (int)c) ? expf(x[c] - mx) * inv : 0.0f;
            continue;
        }
        float mx = x[0]; for (unsigned int c = 1; c < C; c++) mx = fmaxf(mx, x[c]);
        float s = 0.0f; for (unsigned int c = 0; c < C; c++) s += expf(x[c] - mx);
        if (a.u[1]) { float l = mx + logf(s); for (unsigned int c = 0; c < C; c++) o[c] = x[c] - l; }
        else { float inv = 1.0f / s; for (unsigned int c = 0; c < C; c++) o[c] = expf(x[c] - mx) * inv; }
    }
#endif
}
// dx += y * (g - sum(g*y))  (softmax), dx += g - exp(y) * sum(g) (log-softmax)
// p0 = g, p1 = y (the forward output), p2 = dx
DK_KERNEL(dk_softmax_bwd) {
    unsigned int C = a.u[0];
#if defined(__METAL_VERSION__)
    GSR(r, N) {
        TP g = CF(0) + r * C; TP y = CF(1) + r * C; TP dx = F(2) + r * C;
        float s = 0.0f;
        if (a.u[1]) {
            for (unsigned int c = dk_lane; c < C; c += 32u) s += g[c];
            s = simd_sum(s);
            for (unsigned int c = dk_lane; c < C; c += 32u) dx[c] += g[c] - expf(y[c]) * s;
        } else {
            for (unsigned int c = dk_lane; c < C; c += 32u) s += g[c] * y[c];
            s = simd_sum(s);
            for (unsigned int c = dk_lane; c < C; c += 32u) dx[c] += y[c] * (g[c] - s);
        }
    }
#else
    GS(r, N) {
        TP g = CF(0) + r * C; TP y = CF(1) + r * C; TP dx = F(2) + r * C;
        if (a.u[1]) {
            float s = 0.0f; for (unsigned int c = 0; c < C; c++) s += g[c];
            for (unsigned int c = 0; c < C; c++) dx[c] += g[c] - expf(y[c]) * s;
        } else {
            float s = 0.0f; for (unsigned int c = 0; c < C; c++) s += g[c] * y[c];
            for (unsigned int c = 0; c < C; c++) dx[c] += y[c] * (g[c] - s);
        }
    }
#endif
}

// ── Transpose ────────────────────────────────────────────────────────────────
// out [C, R] = in [R, C]^T, or with u2 = 1 out += it (u6 = 1: write); u0 = R,
// u1 = C                                                       n = R*C
DK_KERNEL(dk_transpose) {
    unsigned int R = a.u[0], C = a.u[1];
    GS(i, N) {
        unsigned int r = i / C, c = i % C;
        float v = CF(0)[i];
        if (a.u[2]) ACCUM(F(1)[c * R + r], v); else F(1)[c * R + r] = v;
    }
}

// ── Fused multi-head attention (attention.h) ─────────────────────────────────
// q [S, H*D], k [L, H*D], v [L, H*E], out [S, H*E] (head h is columns
// [h*D, (h+1)*D) of each row): out = softmax(scale q k^T, masked) v per head
// without forming the score matrix -- FlashAttention-2's tiling (Dao 2023):
// a block of 256 threads takes one head's tile of 16 rows and walks the
// other side in tiles of 16 through shared memory, 16 x 16 scores at a time,
// with an online softmax; the row's log-sum-exp is kept for the backward
// pass, which recomputes the scores from it. Thread t works on row t / 16
// of its tile and, across the head's columns, on every 16th one from t % 16
// (D, E <= ATTN_MAXD). Tiles a causal or windowed mask hides entirely are
// skipped. Arguments shared by the three kernels: u0 = S, u1 = L, u2 = H,
// u3 = D, u4 = E, u5 = flags (bits 0-1 the mask, as attn_vis; bits 2-4: dq,
// dk, dv are unwritten zero gradients: write them), u6 = window, f0 = scale;
// p4 = lse [H*S] (fp32), p5 = dout, p7 = D [H*S] (fp32, rowsum(dout * out),
// written by dk_attn_bwd_q for dk_attn_bwd_kv). Launched over
// 256 * H * ceil(rows / 16) threads (rows: S, or L for dk_attn_bwd_kv).
#if defined(__METAL_VERSION__) && defined(DK_ATTN64)
#define ATTN_MAXD 64   // a second Metal build for head dims up to 64, where 16 x 16 tiles fit
#define ATTN_T 16
#elif defined(__METAL_VERSION__)
#define ATTN_MAXD 128
#define ATTN_T 8   // 64 threads per block (see Portability above)
#else
#define ATTN_MAXD 128
#define ATTN_T 16
#endif
#define ATTN_LD (ATTN_MAXD + 1)
#define ATTN_PER (ATTN_MAXD / ATTN_T)
#define ATTN_HID (-3.0e38f)
// The key range a tile of rows [i0, i0 + 16) can see: [lo, hi).
__device__ __forceinline__ void attn_range(const DevArgs& a, unsigned int i0, dk_thread int& lo, dk_thread int& hi) {
    int off = (int)a.u[1] - (int)a.u[0];
    lo = 0; hi = (int)a.u[1];
    if (a.u[5] & 1u) { int m = (int)i0 + ATTN_T - 1 + off + 1; if (m < hi) hi = m; }
    if (a.u[5] & 2u) { int m = (int)i0 + off - (int)a.u[6] + 1; if (m > lo) lo = m; }
    if (lo < 0) lo = 0;
}
// rows [r0, r0 + 16) of a [rows, H*W] tensor's head h -> sh [16][ATTN_LD] (0 past the end)
__device__ __forceinline__ void attn_load(TP x, dk_shared float* sh, unsigned int r0, unsigned int rows, unsigned int H,
                                          unsigned int h, unsigned int W DK_IDS) {
    for (unsigned int k = threadIdx.x; k < ATTN_T * W; k += blockDim.x) {
        unsigned int r = k / W, c = k % W;
        sh[r * ATTN_LD + c] = r0 + r < rows ? (float)x[((r0 + r) * H + h) * W + c] : 0.0f;
    }
}
// p0 q, p1 k, p2 v, p3 out, p4 lse (written)
DK_KERNEL(dk_attn_fwd) {
    __shared__ float Qs[ATTN_T * ATTN_LD], Ks[ATTN_T * ATTN_LD], Vs[ATTN_T * ATTN_LD], Ps[ATTN_T * (ATTN_T + 1)];
    unsigned int S = a.u[0], L = a.u[1], H = a.u[2], D = a.u[3], E = a.u[4], fl = a.u[5] & 3u;
    int off = (int)L - (int)S;
    unsigned int tiles = (S + ATTN_T - 1) / ATTN_T;
    unsigned int r = threadIdx.x / ATTN_T, ln = threadIdx.x % ATTN_T;
    for (unsigned int u = blockIdx.x; u < H * tiles; u += gridDim.x) {
        unsigned int h = u / tiles, i0 = (u % tiles) * ATTN_T, i = i0 + r;
        attn_load(CF(0), Qs, i0, S, H, h, D DK_IDA);
        float acc[ATTN_PER];
        for (unsigned int t = 0; t < ATTN_PER; t++) acc[t] = 0.0f;
        float m = ATTN_HID, l = 0.0f;
        int lo, hi;
        attn_range(a, i0, lo, hi);
        for (int j0 = (lo / ATTN_T) * ATTN_T; j0 < hi; j0 += ATTN_T) {
            __syncthreads();
            attn_load(CF(1), Ks, (unsigned int)j0, L, H, h, D DK_IDA);
            attn_load(CF(2), Vs, (unsigned int)j0, L, H, h, E DK_IDA);
            __syncthreads();
            unsigned int j = (unsigned int)j0 + ln;
            float sc = ATTN_HID;
            if (i < S && j < L && attn_vis(fl, a.u[6], (int)i + off, (int)j)) {
                float s = 0.0f;
                for (unsigned int d = 0; d < D; d++) s += Qs[r * ATTN_LD + d] * Ks[ln * ATTN_LD + d];
                sc = s * a.f[0];
            }
            Ps[r * (ATTN_T + 1) + ln] = sc;
            __syncthreads();
            float mt = ATTN_HID;
            for (unsigned int c = 0; c < ATTN_T; c++) mt = fmaxf(mt, Ps[r * (ATTN_T + 1) + c]);
            if (mt > ATTN_HID) {
                float mn = fmaxf(m, mt), cr = expf(m - mn);
                float pv[ATTN_T], ps = 0.0f;
                for (unsigned int c = 0; c < ATTN_T; c++) {
                    float sv = Ps[r * (ATTN_T + 1) + c];
                    pv[c] = sv > ATTN_HID ? expf(sv - mn) : 0.0f;
                    ps += pv[c];
                }
                l = l * cr + ps;
                for (unsigned int t = 0; t < ATTN_PER; t++) {
                    unsigned int e = ln + t * ATTN_T;
                    if (e < E) {
                        float x = acc[t] * cr;
                        for (unsigned int c = 0; c < ATTN_T; c++) x += pv[c] * Vs[c * ATTN_LD + e];
                        acc[t] = x;
                    }
                }
                m = mn;
            }
        }
        if (i < S) {
            float inv = l > 0.0f ? 1.0f / l : 0.0f;
            TP o = F(3) + (i * H + h) * E;
            for (unsigned int t = 0; t < ATTN_PER; t++) { unsigned int e = ln + t * ATTN_T; if (e < E) o[e] = acc[t] * inv; }
            if (ln == 0) FA(4)[h * S + i] = l > 0.0f ? m + logf(l) : 3.0e38f;   // no visible key: every p is 0
        }
        __syncthreads();
    }
}
// p0 q, p1 k, p2 v, p3 out, p6 dq (+)= scale * sum_j ds_ij k_j (0: none),
// ds_ij = p_ij (dout_i . v_j - D_i)
DK_KERNEL(dk_attn_bwd_q) {
    __shared__ float Qs[ATTN_T * ATTN_LD], Gs[ATTN_T * ATTN_LD], Ks[ATTN_T * ATTN_LD], Vs[ATTN_T * ATTN_LD];
    __shared__ float Ds[ATTN_T * (ATTN_T + 1)], Rd[ATTN_T], Rl[ATTN_T];
    unsigned int S = a.u[0], L = a.u[1], H = a.u[2], D = a.u[3], E = a.u[4], fl = a.u[5] & 3u;
    int off = (int)L - (int)S;
    unsigned int tiles = (S + ATTN_T - 1) / ATTN_T;
    unsigned int r = threadIdx.x / ATTN_T, ln = threadIdx.x % ATTN_T;
    for (unsigned int u = blockIdx.x; u < H * tiles; u += gridDim.x) {
        unsigned int h = u / tiles, i0 = (u % tiles) * ATTN_T, i = i0 + r;
        attn_load(CF(5), Gs, i0, S, H, h, E DK_IDA);
        attn_load(CF(3), Ks, i0, S, H, h, E DK_IDA);   // the output rows, for D (Ks is free until the key tiles)
        __syncthreads();
        float part = 0.0f;
        for (unsigned int e = ln; e < E; e += ATTN_T) part += Gs[r * ATTN_LD + e] * Ks[r * ATTN_LD + e];
        Ds[r * (ATTN_T + 1) + ln] = part;
        __syncthreads();
        if (ln == 0) {
            float Di = 0.0f;
            for (unsigned int c = 0; c < ATTN_T; c++) Di += Ds[r * (ATTN_T + 1) + c];
            Rd[r] = Di;
            Rl[r] = i < S ? FA(4)[h * S + i] : 0.0f;
            if (i < S) FA(7)[h * S + i] = Di;
        }
        if (!a.p[6]) { __syncthreads(); continue; }
        attn_load(CF(0), Qs, i0, S, H, h, D DK_IDA);
        float dq[ATTN_PER];
        for (unsigned int t = 0; t < ATTN_PER; t++) dq[t] = 0.0f;
        int lo, hi;
        attn_range(a, i0, lo, hi);
        for (int j0 = (lo / ATTN_T) * ATTN_T; j0 < hi; j0 += ATTN_T) {
            __syncthreads();
            attn_load(CF(1), Ks, (unsigned int)j0, L, H, h, D DK_IDA);
            attn_load(CF(2), Vs, (unsigned int)j0, L, H, h, E DK_IDA);
            __syncthreads();
            unsigned int j = (unsigned int)j0 + ln;
            float ds = 0.0f;
            if (i < S && j < L && attn_vis(fl, a.u[6], (int)i + off, (int)j)) {
                float s = 0.0f, dp = 0.0f;
                for (unsigned int d = 0; d < D; d++) s += Qs[r * ATTN_LD + d] * Ks[ln * ATTN_LD + d];
                for (unsigned int e = 0; e < E; e++) dp += Gs[r * ATTN_LD + e] * Vs[ln * ATTN_LD + e];
                ds = expf(s * a.f[0] - Rl[r]) * (dp - Rd[r]);
            }
            Ds[r * (ATTN_T + 1) + ln] = ds;
            __syncthreads();
            for (unsigned int t = 0; t < ATTN_PER; t++) {
                unsigned int d = ln + t * ATTN_T;
                if (d < D) {
                    float x = dq[t];
                    for (unsigned int c = 0; c < ATTN_T; c++) x += Ds[r * (ATTN_T + 1) + c] * Ks[c * ATTN_LD + d];
                    dq[t] = x;
                }
            }
        }
        if (i < S) {
            TP o = F(6) + (i * H + h) * D;
            for (unsigned int t = 0; t < ATTN_PER; t++) {
                unsigned int d = ln + t * ATTN_T;
                if (d < D) { if (a.u[5] & 4u) o[d] = a.f[0] * dq[t]; else o[d] += a.f[0] * dq[t]; }
            }
        }
        __syncthreads();
    }
}
// p0 q, p1 k, p2 v, p3 dk (+)= scale * sum_i ds_ij q_i, p6 dv (+)= sum_i p_ij
// dout_i (either 0: none); after dk_attn_bwd_q. A block takes 16 keys.
DK_KERNEL(dk_attn_bwd_kv) {
    __shared__ float Ks[ATTN_T * ATTN_LD], Vs[ATTN_T * ATTN_LD], Qs[ATTN_T * ATTN_LD], Gs[ATTN_T * ATTN_LD];
    __shared__ float Ps[ATTN_T * (ATTN_T + 1)], Ds[ATTN_T * (ATTN_T + 1)], Rd[ATTN_T], Rl[ATTN_T];
    unsigned int S = a.u[0], L = a.u[1], H = a.u[2], D = a.u[3], E = a.u[4], fl = a.u[5] & 3u;
    int off = (int)L - (int)S;
    unsigned int tiles = (L + ATTN_T - 1) / ATTN_T;
    unsigned int r = threadIdx.x / ATTN_T, ln = threadIdx.x % ATTN_T;
    for (unsigned int u = blockIdx.x; u < H * tiles; u += gridDim.x) {
        unsigned int h = u / tiles, j0 = (u % tiles) * ATTN_T, j = j0 + r;
        attn_load(CF(1), Ks, j0, L, H, h, D DK_IDA);
        attn_load(CF(2), Vs, j0, L, H, h, E DK_IDA);
        float dk[ATTN_PER], dv[ATTN_PER];
        for (unsigned int t = 0; t < ATTN_PER; t++) { dk[t] = 0.0f; dv[t] = 0.0f; }
        // the queries that can see keys [j0, j0 + 16): [qlo, qhi)
        int qlo = 0, qhi = (int)S;
        if (fl & 1u) { int m2 = (int)j0 - off; if (m2 > qlo) qlo = m2; }
        if (fl & 2u) { int m2 = (int)j0 + ATTN_T - 1 - off + (int)a.u[6]; if (m2 < qhi) qhi = m2; }
        if (qlo < 0) qlo = 0;
        for (int i0 = (qlo / ATTN_T) * ATTN_T; i0 < qhi; i0 += ATTN_T) {
            __syncthreads();
            attn_load(CF(0), Qs, (unsigned int)i0, S, H, h, D DK_IDA);
            attn_load(CF(5), Gs, (unsigned int)i0, S, H, h, E DK_IDA);
            if (threadIdx.x < ATTN_T) {
                unsigned int i = (unsigned int)i0 + threadIdx.x;
                Rl[threadIdx.x] = i < S ? FA(4)[h * S + i] : 0.0f;
                Rd[threadIdx.x] = i < S ? FA(7)[h * S + i] : 0.0f;
            }
            __syncthreads();
            // thread (key r, query ln)
            unsigned int i = (unsigned int)i0 + ln;
            float p = 0.0f, ds = 0.0f;
            if (i < S && j < L && attn_vis(fl, a.u[6], (int)i + off, (int)j)) {
                float s = 0.0f, dp = 0.0f;
                for (unsigned int d = 0; d < D; d++) s += Qs[ln * ATTN_LD + d] * Ks[r * ATTN_LD + d];
                for (unsigned int e = 0; e < E; e++) dp += Gs[ln * ATTN_LD + e] * Vs[r * ATTN_LD + e];
                p = expf(s * a.f[0] - Rl[ln]);
                ds = p * (dp - Rd[ln]);
            }
            Ps[r * (ATTN_T + 1) + ln] = p;
            Ds[r * (ATTN_T + 1) + ln] = ds;
            __syncthreads();
            for (unsigned int t = 0; t < ATTN_PER; t++) {
                unsigned int d = ln + t * ATTN_T;
                if (d < E) {
                    float x = dv[t];
                    for (unsigned int c = 0; c < ATTN_T; c++) x += Ps[r * (ATTN_T + 1) + c] * Gs[c * ATTN_LD + d];
                    dv[t] = x;
                }
                if (d < D) {
                    float x = dk[t];
                    for (unsigned int c = 0; c < ATTN_T; c++) x += Ds[r * (ATTN_T + 1) + c] * Qs[c * ATTN_LD + d];
                    dk[t] = x;
                }
            }
        }
        if (j < L) {
            if (a.p[3]) {
                TP o = F(3) + (j * H + h) * D;
                for (unsigned int t = 0; t < ATTN_PER; t++) {
                    unsigned int d = ln + t * ATTN_T;
                    if (d < D) { if (a.u[5] & 8u) o[d] = a.f[0] * dk[t]; else o[d] += a.f[0] * dk[t]; }
                }
            }
            if (a.p[6]) {
                TP o = F(6) + (j * H + h) * E;
                for (unsigned int t = 0; t < ATTN_PER; t++) {
                    unsigned int e = ln + t * ATTN_T;
                    if (e < E) { if (a.u[5] & 16u) o[e] = dv[t]; else o[e] += dv[t]; }
                }
            }
        }
        __syncthreads();
    }
}

// ── Fused losses -> loss[0] (zeroed by the caller), averaged ─────────────────
// Cross-entropy on class indices: p0 = logits [rows, C], p1 = labels [rows],
// p2 = loss[0]; u0 = C; f0 = 1/rows                            n = rows
DK_KERNEL(dk_ce_fwd) {
    unsigned int C = a.u[0];
    float acc = 0.0f;
#if defined(__METAL_VERSION__)
    GSR(r, N) {
        TP x = CF(0) + r * C;
        float mx = -3.0e38f;
        for (unsigned int c = dk_lane; c < C; c += 32u) mx = fmaxf(mx, x[c]);
        mx = simd_max(mx);
        float s = 0.0f;
        for (unsigned int c = dk_lane; c < C; c += 32u) s += expf(x[c] - mx);
        s = simd_sum(s);
        unsigned int lab = (unsigned int)CF(1)[r];
        if (dk_lane == 0u) acc += (mx + logf(s) - x[lab]) * a.f[0];
    }
#else
    GS(r, N) {
        TP x = CF(0) + r * C;
        float mx = x[0]; for (unsigned int c = 1; c < C; c++) mx = fmaxf(mx, x[c]);
        float s = 0.0f; for (unsigned int c = 0; c < C; c++) s += expf(x[c] - mx);
        unsigned int lab = (unsigned int)CF(1)[r];
        acc += (mx + logf(s) - x[lab]) * a.f[0];
    }
#endif
    block_add(acc, FA(2) DK_IDA);
}
// dlogits += seed[0] * (softmax - onehot) / rows; p3 = seed (the loss's grad)
DK_KERNEL(dk_ce_bwd) {
    unsigned int C = a.u[0];
#if defined(__METAL_VERSION__)
    GSR(r, N) {
        TP x = CF(0) + r * C; TP d = F(2) + r * C;
        float sc = CF(3)[0] * a.f[0];
        float mx = -3.0e38f;
        for (unsigned int c = dk_lane; c < C; c += 32u) mx = fmaxf(mx, x[c]);
        mx = simd_max(mx);
        float s = 0.0f;
        for (unsigned int c = dk_lane; c < C; c += 32u) s += expf(x[c] - mx);
        s = simd_sum(s);
        unsigned int lab = (unsigned int)CF(1)[r];
        float inv = 1.0f / s;
        for (unsigned int c = dk_lane; c < C; c += 32u) ACCUM(d[c], sc * (expf(x[c] - mx) * inv - (c == lab ? 1.0f : 0.0f)));
    }
#else
    GS(r, N) {
        TP x = CF(0) + r * C; TP d = F(2) + r * C;
        float sc = CF(3)[0] * a.f[0];
        float mx = x[0]; for (unsigned int c = 1; c < C; c++) mx = fmaxf(mx, x[c]);
        float s = 0.0f; for (unsigned int c = 0; c < C; c++) s += expf(x[c] - mx);
        unsigned int lab = (unsigned int)CF(1)[r];
        float inv = 1.0f / s;
        for (unsigned int c = 0; c < C; c++) ACCUM(d[c], sc * (expf(x[c] - mx) * inv - (c == lab ? 1.0f : 0.0f)));
    }
#endif
}
// Elementwise losses: u0 = 0 BCE on logits, 1 Huber (f1 = delta); p0 = x,
// p1 = target, p2 = loss[0], f0 = 1/n                          n = elements
DK_KERNEL(dk_ewloss_fwd) {
    float acc = 0.0f;
    GS(i, N) {
        float x = CF(0)[i], t = CF(1)[i], v;
        if (a.u[0] == 0) v = fmaxf(x, 0.0f) - x * t + logf(1.0f + expf(-fabsf(x)));
        else { float d = x - t, ad = fabsf(d), k = a.f[1]; v = ad <= k ? 0.5f * d * d : k * (ad - 0.5f * k); }
        acc += v * a.f[0];
    }
    block_add(acc, FA(2) DK_IDA);
}
// dx += seed[0] / n * (sigmoid(x) - t)  |  clamp(x - t, -delta, delta); p2 = dx, p3 = seed
DK_KERNEL(dk_ewloss_bwd) {
    float sc = CF(3)[0] * a.f[0];
    GS(i, N) {
        float x = CF(0)[i], t = CF(1)[i], g;
        if (a.u[0] == 0) g = sig_f(x) - t;
        else { float d = x - t, k = a.f[1]; g = fminf(fmaxf(d, -k), k); }
        ACCUM(F(2)[i], sc * g);
    }
}

// ── Layer norm: one thread per row, u0 = cols, f0 = eps ──────────────────────
// p0 = x, p1 = xhat out, p2 = inv std [rows]
DK_KERNEL(dk_layernorm) {
    unsigned int C = a.u[0];
#if defined(__METAL_VERSION__)
    GSR(r, N) {
        TP x = CF(0) + r * C; TP o = F(1) + r * C;
        float m = 0.0f;
        for (unsigned int c = dk_lane; c < C; c += 32u) m += x[c];
        m = simd_sum(m) / (float)C;
        float v = 0.0f;
        for (unsigned int c = dk_lane; c < C; c += 32u) { float d = x[c] - m; v += d * d; }
        float is = rsqrtf(simd_sum(v) / (float)C + a.f[0]);
        if (dk_lane == 0u) F(2)[r] = is;
        for (unsigned int c = dk_lane; c < C; c += 32u) o[c] = (x[c] - m) * is;
    }
#else
    GS(r, N) {
        TP x = CF(0) + r * C; TP o = F(1) + r * C;
        float m = 0.0f; for (unsigned int c = 0; c < C; c++) m += x[c]; m /= (float)C;
        float v = 0.0f; for (unsigned int c = 0; c < C; c++) { float d = x[c] - m; v += d * d; }
        float is = rsqrtf(v / (float)C + a.f[0]);
        F(2)[r] = is;
        for (unsigned int c = 0; c < C; c++) o[c] = (x[c] - m) * is;
    }
#endif
}
// dx += inv * (g - mean(g) - xhat * mean(g * xhat)); p0 = g, p1 = xhat, p2 = inv, p3 = dx
DK_KERNEL(dk_layernorm_bwd) {
    unsigned int C = a.u[0];
#if defined(__METAL_VERSION__)
    GSR(r, N) {
        TP g = CF(0) + r * C; TP y = CF(1) + r * C; TP dx = F(3) + r * C;
        float mg = 0.0f, mgy = 0.0f;
        for (unsigned int c = dk_lane; c < C; c += 32u) { mg += g[c]; mgy += g[c] * y[c]; }
        mg = simd_sum(mg) / (float)C; mgy = simd_sum(mgy) / (float)C;
        float is = CF(2)[r];
        for (unsigned int c = dk_lane; c < C; c += 32u) dx[c] += is * (g[c] - mg - y[c] * mgy);
    }
#else
    GS(r, N) {
        TP g = CF(0) + r * C; TP y = CF(1) + r * C; TP dx = F(3) + r * C;
        float mg = 0.0f, mgy = 0.0f;
        for (unsigned int c = 0; c < C; c++) { mg += g[c]; mgy += g[c] * y[c]; }
        mg /= (float)C; mgy /= (float)C;
        float is = CF(2)[r];
        for (unsigned int c = 0; c < C; c++) dx[c] += is * (g[c] - mg - y[c] * mgy);
    }
#endif
}

// ── Optimizer step over one parameter (nn.h's OPT_* arithmetic) ─────────────
// p0 = w, p1 = g, p2 = m, p3 = v, p4 = scale[0] (gradient clip factor, or 0 = none)
// u0 = kind; f0 lr, f1 wd, f2 momentum, f3 beta1, f4 beta2, f5 eps, f6 1/bc1, f7 1/sqrt(bc2)
#define OPT_SGD      0
#define OPT_MOMENTUM 1
#define OPT_NESTEROV 2
#define OPT_RMSPROP  3
#define OPT_ADAM     4
#define OPT_ADAMW    5
#define OPT_ADAGRAD  6
#define OPT_LION     7
__device__ void opt_one(const DevArgs& a, dk_global float* w, const dk_global float* gr, dk_global float* m, dk_global float* v,
                        unsigned int i, float cs) {
    int kind = (int)a.u[0];
    float lr = a.f[0], wd = a.f[1], mu = a.f[2], b1 = a.f[3], b2 = a.f[4], eps = a.f[5];
    {
        float g0 = gr[i] * cs;
        if (kind == OPT_SGD) { w[i] -= lr * (g0 + wd * w[i]); }
        else if (kind == OPT_MOMENTUM || kind == OPT_NESTEROV) {
            float g = g0 + wd * w[i]; float mi = mu * m[i] + g; m[i] = mi;
            w[i] -= lr * (kind == OPT_NESTEROV ? g + mu * mi : mi);
        } else if (kind == OPT_ADAGRAD) {
            float g = g0 + wd * w[i]; float vi = v[i] + g * g; v[i] = vi;
            w[i] -= lr * g / (sqrtf(vi) + eps);
        } else if (kind == OPT_LION) {
            float c = b1 * m[i] + (1.0f - b1) * g0;
            float sg = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f);
            w[i] -= lr * (sg + wd * w[i]);
            m[i] = b2 * m[i] + (1.0f - b2) * g0;
        } else if (kind == OPT_RMSPROP) {
            float g = g0 + wd * w[i]; float vi = b2 * v[i] + (1.0f - b2) * g * g; v[i] = vi;
            w[i] -= lr * g / (sqrtf(vi) + eps);
        } else {
            float l2 = kind == OPT_ADAMW ? 0.0f : wd, dec = kind == OPT_ADAMW ? wd : 0.0f;
            float g = g0 + l2 * w[i];
            float mi = b1 * m[i] + (1.0f - b1) * g, vi = b2 * v[i] + (1.0f - b2) * g * g;
            m[i] = mi; v[i] = vi;
            w[i] -= lr * ((mi * a.f[6]) / (sqrtf(vi) * a.f[7] + eps) + dec * w[i]);
        }
    }
}
// p5 (when set): the loss scale's state; an overflowed step changes nothing.
DK_KERNEL(dk_opt) {
    TP clip = CF(4);
    if (a.p[5] && CF(5)[2] != 0.0f) return;
    float cs = clip ? clip[0] : 1.0f;
    GS(i, N) opt_one(a, FA(0), FA(1), FA(2), FA(3), i, cs);
}
// Every parameter's step in one launch: p0 = a table of u1 entries
// {w, g, m, v, n, start} (u64 each; start = the entry's first thread),
// otherwise dk_opt's arguments                               n = total values
DK_KERNEL(dk_opt_multi) {
    const dk_global DK_U64* T = (const dk_global DK_U64*)a.p[0];
    unsigned int cnt = a.u[1];
    if (a.p[5] && CF(5)[2] != 0.0f) return;
    GS(i, N) {
        unsigned int lo = 0, hi = cnt - 1;
        while (lo < hi) { unsigned int mid = (lo + hi + 1) / 2; if (T[mid * 6 + 5] <= i) lo = mid; else hi = mid - 1; }
        const dk_global DK_U64* e = T + lo * 6;
        opt_one(a, (dk_global float*)e[0], (const dk_global float*)e[1], (dk_global float*)e[2], (dk_global float*)e[3],
                i - (unsigned int)e[5], 1.0f);
    }
}
// scale[0] = sumsq[0] > maxNorm^2 ? maxNorm / sqrt(sumsq[0]) : 1; p0 = sumsq, p1 = scale, f0 = maxNorm
DK_KERNEL(dk_clip_scale) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float n = sqrtf(CF(0)[0]);
        F(1)[0] = (n > a.f[0] && n > 0.0f) ? a.f[0] / n : 1.0f;
    }
}

// ── Convolution helpers (convnet.h's layout: [batch, C*H*W] rows, NCHW) ─────
// u0 C, u1 H, u2 W, u3 kH, u4 kW, u5 OH, u6 OW; f0 strideH, f1 strideW,
// f2 padH, f3 padW, f4 dilH, f5 dilW (integers carried as floats)
// Batched: col [C*kH*kW, B*OH*OW] for B images (f6 = B); p0 = x [B, C*H*W], p1 = col
//                                                            n = C*kH*kW*B*OH*OW
#define CONV_DECODE \
    unsigned int C = a.u[0], H = a.u[1], W = a.u[2], KH = a.u[3], KW = a.u[4], OH = a.u[5], OW = a.u[6]; \
    unsigned int B = (unsigned int)a.f[6]; \
    int sh = (int)a.f[0], sw = (int)a.f[1], ph = (int)a.f[2], pw = (int)a.f[3], dh = (int)a.f[4], dw = (int)a.f[5];
DK_KERNEL(dk_im2col) {
    CONV_DECODE
    unsigned int OP = OH * OW, BP = B * OP;
    DK_RECIP(iBP, BP) DK_RECIP(iOP, OP) DK_RECIP(iOW, OW) DK_RECIP(iKW, KW) DK_RECIP(iKH, KH) DK_RECIP(iKK, KW * KH)
    GS(i, N) {
        unsigned int q = DK_UMOD(i, BP, iBP), r = DK_UDIV(i, BP, iBP);                 // column, row of col
        unsigned int b = DK_UDIV(q, OP, iOP), p = DK_UMOD(q, OP, iOP), oh = DK_UDIV(p, OW, iOW), ow = DK_UMOD(p, OW, iOW);
        unsigned int kw = DK_UMOD(r, KW, iKW), kh = DK_UMOD((DK_UDIV(r, KW, iKW)), KH, iKH), c = DK_UDIV(r, (KW * KH), iKK);
        int y = (int)oh * sh - ph + (int)kh * dh, x = (int)ow * sw - pw + (int)kw * dw;
        F(1)[i] = (y >= 0 && y < (int)H && x >= 0 && x < (int)W)
            ? CF(0)[((b * C + c) * H + (unsigned int)y) * W + (unsigned int)x] : 0.0f;
    }
}
// x [B, C*H*W] += col2im(col): the adjoint of dk_im2col (atomics: windows overlap)
DK_KERNEL(dk_col2im) {
    CONV_DECODE
    unsigned int OP = OH * OW, BP = B * OP;
    DK_RECIP(iBP, BP) DK_RECIP(iOP, OP) DK_RECIP(iOW, OW) DK_RECIP(iKW, KW) DK_RECIP(iKH, KH) DK_RECIP(iKK, KW * KH)
    GS(i, N) {
        unsigned int q = DK_UMOD(i, BP, iBP), r = DK_UDIV(i, BP, iBP);
        unsigned int b = DK_UDIV(q, OP, iOP), p = DK_UMOD(q, OP, iOP), oh = DK_UDIV(p, OW, iOW), ow = DK_UMOD(p, OW, iOW);
        unsigned int kw = DK_UMOD(r, KW, iKW), kh = DK_UMOD((DK_UDIV(r, KW, iKW)), KH, iKH), c = DK_UDIV(r, (KW * KH), iKK);
        int y = (int)oh * sh - ph + (int)kh * dh, x = (int)ow * sw - pw + (int)kw * dw;
        if (y >= 0 && y < (int)H && x >= 0 && x < (int)W)
            atomicAdd(&FA(1)[((b * C + c) * H + (unsigned int)y) * W + (unsigned int)x], CF(0)[i]);
    }
}
// out [B2, A, P] (+)= in [A, B2, P] with the first two axes swapped; u0 A, u1 B2, u2 P,
// u3 = 1 to accumulate                                        n = A*B2*P
DK_KERNEL(dk_swap01) {
    unsigned int A = a.u[0], B2 = a.u[1], P = a.u[2];
    DK_RECIP(iP, P) DK_RECIP(iB2, B2)
    GS(i, N) {
        unsigned int p = DK_UMOD(i, P, iP), t = DK_UDIV(i, P, iP), bi = DK_UMOD(t, B2, iB2), ai = DK_UDIV(t, B2, iB2);   // i indexes 'in'
        unsigned int o = (bi * A + ai) * P + p;
        if (a.u[3]) F(1)[o] += CF(0)[i]; else F(1)[o] = CF(0)[i];
    }
}
// q / d through a float reciprocal (inv = 1/d), made exact by correction:
// integer division costs dozens of instructions per element.
__device__ unsigned int udiv(unsigned int q, unsigned int d, float inv) {
    unsigned int r = (unsigned int)((float)q * inv);
    while (r * d > q) r--;
    while ((r + 1u) * d <= q) r++;
    return r;
}
// The channel of value i in [batch, C, P] (iP = 1/P, iC = 1/C).
__device__ unsigned int chan_of(unsigned int i, unsigned int C, unsigned int P, float iP, float iC) {
    unsigned int t = udiv(i, P, iP);
    return t - udiv(t, C, iC) * C;
}
#ifdef DK_TAGGED
// Pairs of 16-bit values (type T) through 32-bit words: word j holds
// elements 2j and 2j + 1.
template <int T> __device__ __forceinline__ float pk_lo(unsigned int w) { return T == 1 ? h2f_bits((unsigned short)(w & 0xffffu)) : b2f_bits((unsigned short)(w & 0xffffu)); }
template <int T> __device__ __forceinline__ float pk_hi(unsigned int w) { return T == 1 ? h2f_bits((unsigned short)(w >> 16)) : b2f_bits((unsigned short)(w >> 16)); }
template <int T> __device__ __forceinline__ unsigned int pk(float lo, float hi) {
    return T == 1 ? ((unsigned int)f2h_bits(lo) | ((unsigned int)f2h_bits(hi) << 16))
                  : ((unsigned int)f2b_bits(lo) | ((unsigned int)f2b_bits(hi) << 16));
}
__device__ __forceinline__ const dk_global unsigned int* pw(DK_U64 p) { return (const dk_global unsigned int*)(p & DEV_ADDR_MASK); }
__device__ __forceinline__ dk_global unsigned int* pwm(DK_U64 p) { return (dk_global unsigned int*)(p & DEV_ADDR_MASK); }
// The 16-bit type every listed handle shares (4-byte aligned, nonzero), or 0.
__device__ unsigned int pair_tag(DK_U64 p0, DK_U64 p1, DK_U64 p2, DK_U64 p3) {
    unsigned int t = (unsigned int)(p0 >> 62);
    DK_U64 ps[4] = { p0, p1, p2, p3 };
    if (t == 0u) return 0u;
    for (int q = 0; q < 4; q++) {
        if (ps[q] == 0ull) continue;
        if ((unsigned int)(ps[q] >> 62) != t || (ps[q] & 3ull) != 0ull) return 0u;
    }
    return t;
}
#endif

// out[b, c, p] += bias[c]; u0 = channels, u1 = plane            n = batch*C*plane
#ifdef DK_TAGGED
template <int T> __device__ void bias_ch_pairs(const DevArgs& a DK_IDS) {
    unsigned int C = a.u[0], P = a.u[1];
    float iP = 1.0f / (float)P, iC = 1.0f / (float)C;
    dk_global unsigned int* o = pwm(a.p[0]);
    GS(j, N / 2u) {
        float b = FA(1)[chan_of(2u * j, C, P, iP, iC)];
        unsigned int w = o[j];
        o[j] = pk<T>(pk_lo<T>(w) + b, pk_hi<T>(w) + b);
    }
}
#endif
DK_KERNEL(dk_bias_channels) {
    unsigned int C = a.u[0], P = a.u[1];
#ifdef DK_TAGGED
    if ((P & 1u) == 0u && (N & 1u) == 0u) {
        unsigned int t = pair_tag(a.p[0], 0ull, 0ull, 0ull);
        if (t == 1u) { bias_ch_pairs<1>(a DK_IDA); return; }
        if (t == 2u) { bias_ch_pairs<2>(a DK_IDA); return; }
    }
#endif
    float iP = 1.0f / (float)P, iC = 1.0f / (float)C;
    GS(i, N) F(0)[i] += CF(1)[chan_of(i, C, P, iP, iC)];
}
// Block-per-unit kernels run n = units * 256 threads: block b takes units
// b, b + gridDim.x, ...; its threads split the unit's values.
#define UNITS(u, U) for (unsigned int u = blockIdx.x; u < (U); u += gridDim.x)
#define TSPLIT(q, M) for (unsigned int q = threadIdx.x; q < (M); q += blockDim.x)
// The block's total of v, returned to every thread.
__device__ dk_double block_sum_d(dk_double v DK_IDS) {
#if defined(__METAL_VERSION__)
    __syncthreads();   // the previous call's readers are done with dk_red
    v = simd_sum(v);
    if (dk_lane == 0u) dk_red[dk_simd] = v;
    __syncthreads();
    float s = 0.0f;
    for (unsigned int k = 0; k < (blockDim.x + 31u) / 32u; k++) s += dk_red[k];
    return s;
#else
    __shared__ dk_double s[256];
    __syncthreads();
    s[threadIdx.x] = v;
    __syncthreads();
    for (unsigned int w = 128; w > 0; w >>= 1) {
        if (threadIdx.x < w) s[threadIdx.x] += s[threadIdx.x + w];
        __syncthreads();
    }
    return s[0];
#endif
}
// Per-channel reductions over [batch, C, plane] split each channel's M =
// batch*plane values into S slices (u3), a block each: slice s takes the
// 256-value runs r with r % S == s. Units are (channel, slice) = (u / S, u % S).
#define SLICE_LOOP(q, M, s, S) for (unsigned int q = (s) * blockDim.x + threadIdx.x; q < (M); q += (S) * blockDim.x)
// The index of the q-th value of channel c (needs iP = 1/P in scope).
__device__ unsigned int chan_at(unsigned int q, unsigned int c, unsigned int C, unsigned int P, float iP) {
    unsigned int b = udiv(q, P, iP);
    return (b * C + c) * P + (q - b * P);
}
#define CHAN_AT(q, c, C, P) chan_at((q), (c), (C), (P), iP)
// gb[c] += sum over batch and plane of d; u0 = C, u1 = plane, u2 = batch, u3 = S   n = C*S*256
DK_KERNEL(dk_bias_channels_bwd) {
    unsigned int C = a.u[0], P = a.u[1], M = a.u[2] * a.u[1], S = a.u[3];
    float iP = 1.0f / (float)P;
#ifdef DK_TAGGED
    unsigned int t = (P & 1u) == 0u ? pair_tag(a.p[0], 0ull, 0ull, 0ull) : 0u;
    if (t != 0u) {
        // pairs: the channel's M / 2 words
        const dk_global unsigned int* d = pw(a.p[0]);
        UNITS(u, C * S) {
            unsigned int c = u / S, sl = u % S;
            dk_double s = 0.0;
            SLICE_LOOP(q, M / 2u, sl, S) {
                unsigned int w = d[CHAN_AT(2u * q, c, C, P) / 2u];
                s += t == 1u ? (dk_double)pk_lo<1>(w) + (dk_double)pk_hi<1>(w) : (dk_double)pk_lo<2>(w) + (dk_double)pk_hi<2>(w);
            }
            s = block_sum_d(s DK_IDA);
            if (threadIdx.x == 0) atomicAdd(&FA(1)[c], (float)s);
        }
        return;
    }
#endif
    UNITS(u, C * S) {
        unsigned int c = u / S, sl = u % S;
        dk_double s = 0.0;
        SLICE_LOOP(q, M, sl, S) s += (dk_double)CF(0)[CHAN_AT(q, c, C, P)];
        s = block_sum_d(s DK_IDA);
        if (threadIdx.x == 0) atomicAdd(&FA(1)[c], (float)s);
    }
}

// ── Batch / group normalization (convnet.h's layout [batch, C*plane]) ───────
// Batch norm statistics, pass 1: per (channel, slice) the count, mean and sum
// of squared deviations, as doubles in p1 [C*S*3]. p0 x; u0 C, u1 plane, u2 batch, u3 S
DK_KERNEL(dk_bn_partial) {
    unsigned int C = a.u[0], P = a.u[1], M = a.u[2] * a.u[1], S = a.u[3];
    float iP = 1.0f / (float)P;
    dk_global dk_double* part = (dk_global dk_double*)a.p[1];
#ifdef DK_TAGGED
    unsigned int t = (P & 1u) == 0u ? pair_tag(a.p[0], 0ull, 0ull, 0ull) : 0u;
    if (t != 0u) {
        const dk_global unsigned int* xw = pw(a.p[0]);
        UNITS(u, C * S) {
            unsigned int c = u / S, sl = u % S;
            dk_double s = 0.0, cnt = 0.0;
            SLICE_LOOP(q, M / 2u, sl, S) {
                unsigned int w = xw[CHAN_AT(2u * q, c, C, P) / 2u];
                float lo = t == 1u ? pk_lo<1>(w) : pk_lo<2>(w), hi = t == 1u ? pk_hi<1>(w) : pk_hi<2>(w);
                s += (dk_double)lo + (dk_double)hi; cnt += 2.0;
            }
            cnt = block_sum_d(cnt DK_IDA);
            dk_double tot = block_sum_d(s DK_IDA);
            dk_double mu = cnt > 0.0 ? tot / cnt : 0.0;
            dk_double sq = 0.0;
            SLICE_LOOP(q, M / 2u, sl, S) {
                unsigned int w = xw[CHAN_AT(2u * q, c, C, P) / 2u];
                dk_double lo = (dk_double)(t == 1u ? pk_lo<1>(w) : pk_lo<2>(w)) - mu, hi = (dk_double)(t == 1u ? pk_hi<1>(w) : pk_hi<2>(w)) - mu;
                sq += lo * lo + hi * hi;
            }
            sq = block_sum_d(sq DK_IDA);
            if (threadIdx.x == 0) { part[u * 3] = cnt; part[u * 3 + 1] = mu; part[u * 3 + 2] = sq; }
        }
        return;
    }
#endif
    UNITS(u, C * S) {
        unsigned int c = u / S, sl = u % S;
        dk_double s = 0.0, cnt = 0.0;
        SLICE_LOOP(q, M, sl, S) { s += (dk_double)CF(0)[CHAN_AT(q, c, C, P)]; cnt += 1.0; }
        cnt = block_sum_d(cnt DK_IDA);
        dk_double tot = block_sum_d(s DK_IDA);
        dk_double mu = cnt > 0.0 ? tot / cnt : 0.0;
        dk_double sq = 0.0;
        SLICE_LOOP(q, M, sl, S) { dk_double d = (dk_double)CF(0)[CHAN_AT(q, c, C, P)] - mu; sq += d * d; }
        sq = block_sum_d(sq DK_IDA);
        if (threadIdx.x == 0) { part[u * 3] = cnt; part[u * 3 + 1] = mu; part[u * 3 + 2] = sq; }
    }
}
// Pass 2, one thread per channel: the slices combined (Chan et al.), the
// running statistics updated (training), mean and invstd. p0 partials, p1
// mean[C], p2 invstd[C], p3 running mean, p4 running var; u0 C, u3 S,
// u4 training; f0 momentum, f1 eps                              n = C
DK_KERNEL(dk_bn_finish) {
    unsigned int S = a.u[3];
    const dk_global dk_double* part = (const dk_global dk_double*)a.p[0];
    GS(c, N) {
        float mean, var;
        if (a.u[4]) {
            dk_double n = 0.0, mu = 0.0, m2 = 0.0;
            for (unsigned int s = 0; s < S; s++) {
                const dk_global dk_double* p = part + (c * S + s) * 3;
                if (p[0] == 0.0) continue;
                dk_double nn = n + p[0], d = p[1] - mu;
                mu += d * p[0] / nn;
                m2 += p[2] + d * d * n * p[0] / nn;
                n = nn;
            }
            mean = (float)mu;
            var = (float)(m2 / n);
            float unb = n > 1.0 ? (float)(m2 / (n - 1.0)) : var;
            F(3)[c] = (1.0f - a.f[0]) * F(3)[c] + a.f[0] * mean;
            F(4)[c] = (1.0f - a.f[0]) * F(4)[c] + a.f[0] * unb;
        } else {
            mean = CF(3)[c];
            var = CF(4)[c];
        }
        F(1)[c] = mean;
        F(2)[c] = 1.0f / sqrtf(var + a.f[1]);
    }
}
// Group norm statistics per (image, group) unit of M contiguous values:
// p0 x, p1 mean[U], p2 invstd[U]; u0 M, u1 U; f0 eps            n = U*256
DK_KERNEL(dk_gn_stats) {
    unsigned int M = a.u[0];
    UNITS(u, a.u[1]) {
        TP x = CF(0) + (DK_U64)u * M;
        dk_double s = 0.0;
        TSPLIT(q, M) s += (dk_double)x[q];
        dk_double mu = block_sum_d(s DK_IDA) / (dk_double)M;
        dk_double sq = 0.0;
        TSPLIT(q, M) { dk_double d = (dk_double)x[q] - mu; sq += d * d; }
        sq = block_sum_d(sq DK_IDA);
        if (threadIdx.x == 0) { F(1)[u] = (float)mu; F(2)[u] = 1.0f / sqrtf((float)(sq / (dk_double)M) + a.f[0]); }
    }
}
// The statistic of value i: its channel (batch norm, G = 0) or its (image, group).
__device__ unsigned int norm_stat(unsigned int i, unsigned int C, unsigned int P, unsigned int G, float iP, float iC) {
    unsigned int t = udiv(i, P, iP), img = udiv(t, C, iC), c = t - img * C;
    return G == 0 ? c : img * G + c / (C / G);
}
// A normalization fused with the activation after it: u5 = 0 none, 1 ReLU,
// 2 leaky ReLU (slope f1). The backward kernels take the fused output in p7
// and use dy * act'(y) for dy.
__device__ float fused_act(const DevArgs& a, float v) {
    return a.u[5] == 0 || v > 0.0f ? v : (a.u[5] == 1 ? 0.0f : a.f[1] * v);
}
__device__ float fused_dy(const DevArgs& a, DK_U64 i) {
    float dy = TP(a.p[0])[i];
    if (a.u[5] == 0) return dy;
    return TP(a.p[7])[i] > 0.0f ? dy : (a.u[5] == 1 ? 0.0f : a.f[1] * dy);
}
// out = gamma[c] (x - mean) invstd + beta[c]: p0 x, p1 out, p2 mean, p3 invstd,
// p4 gamma, p5 beta; u0 C, u1 plane, u2 groups (0: batch norm)  n = values
#ifdef DK_TAGGED
template <int T> __device__ void norm_apply_pairs(const DevArgs& a DK_IDS) {
    unsigned int C = a.u[0], P = a.u[1], G = a.u[2];
    float iP = 1.0f / (float)P, iC = 1.0f / (float)C;
    const dk_global unsigned int* x = pw(a.p[0]);
    dk_global unsigned int* o = pwm(a.p[1]);
    GS(j, N / 2u) {
        unsigned int i = 2u * j, c = chan_of(i, C, P, iP, iC), st = norm_stat(i, C, P, G, iP, iC);
        float g = FA(4)[c], b = FA(5)[c], m = FA(2)[st], is = FA(3)[st];
        unsigned int w = x[j];
        o[j] = pk<T>(fused_act(a, g * (pk_lo<T>(w) - m) * is + b), fused_act(a, g * (pk_hi<T>(w) - m) * is + b));
    }
}
#endif
DK_KERNEL(dk_norm_apply) {
    unsigned int C = a.u[0], P = a.u[1], G = a.u[2];
#ifdef DK_TAGGED
    if ((P & 1u) == 0u && (N & 1u) == 0u) {
        unsigned int t = pair_tag(a.p[0], a.p[1], 0ull, 0ull);
        if (t == 1u) { norm_apply_pairs<1>(a DK_IDA); return; }
        if (t == 2u) { norm_apply_pairs<2>(a DK_IDA); return; }
    }
#endif
    float iP = 1.0f / (float)P, iC = 1.0f / (float)C;
    GS(i, N) {
        unsigned int c = chan_of(i, C, P, iP, iC), st = norm_stat(i, C, P, G, iP, iC);
        F(1)[i] = fused_act(a, CF(4)[c] * (CF(0)[i] - CF(2)[st]) * CF(3)[st] + CF(5)[c]);
    }
}
// Per channel: dgamma[c] += sum dy xh, dbeta[c] += sum dy, and (p6, when set,
// zeroed by the caller) sums[c] += sum dy, sums[C + c] += sum dy xh. p0 dy,
// p1 x, p2 mean, p3 invstd, p4 dgamma, p5 dbeta (either may be 0); u0 C,
// u1 plane, u2 groups, u3 batch, u4 S slices                    n = C*S*256
#ifdef DK_TAGGED
// dy * act'(y) for a pair
template <int T> __device__ __forceinline__ void fused_dy2(const DevArgs& a, unsigned int j, dk_thread float& lo, dk_thread float& hi) {
    unsigned int w = pw(a.p[0])[j];
    lo = pk_lo<T>(w); hi = pk_hi<T>(w);
    if (a.u[5] != 0u) {
        unsigned int y = pw(a.p[7])[j];
        float s = a.u[5] == 1u ? 0.0f : a.f[1];
        if (!(pk_lo<T>(y) > 0.0f)) lo *= s;
        if (!(pk_hi<T>(y) > 0.0f)) hi *= s;
    }
}
template <int T> __device__ void norm_sums_pairs(const DevArgs& a DK_IDS) {
    unsigned int C = a.u[0], P = a.u[1], G = a.u[2], M = a.u[3] * a.u[1], S = a.u[4];
    float iP = 1.0f / (float)P, iC = 1.0f / (float)C;
    const dk_global unsigned int* xw = pw(a.p[1]);
    UNITS(u, C * S) {
        unsigned int c = u / S, sl = u % S;
        dk_double sd = 0.0, sx = 0.0;
        SLICE_LOOP(q, M / 2u, sl, S) {
            unsigned int i = CHAN_AT(2u * q, c, C, P), st = G == 0 ? c : norm_stat(i, C, P, G, iP, iC);
            float dlo, dhi;
            fused_dy2<T>(a, i / 2u, dlo, dhi);
            float m = FA(2)[st], is = FA(3)[st];
            unsigned int w = xw[i / 2u];
            sd += (dk_double)dlo + (dk_double)dhi;
            sx += (dk_double)(dlo * (pk_lo<T>(w) - m) * is) + (dk_double)(dhi * (pk_hi<T>(w) - m) * is);
        }
        sd = block_sum_d(sd DK_IDA);
        sx = block_sum_d(sx DK_IDA);
        if (threadIdx.x == 0) {
            if (a.p[4]) atomicAdd(&FA(4)[c], (float)sx);
            if (a.p[5]) atomicAdd(&FA(5)[c], (float)sd);
            if (a.p[6]) { atomicAdd(&FA(6)[c], (float)sd); atomicAdd(&FA(6)[C + c], (float)sx); }
        }
    }
}
#endif
DK_KERNEL(dk_norm_sums) {
    unsigned int C = a.u[0], P = a.u[1], G = a.u[2], M = a.u[3] * a.u[1], S = a.u[4];
#ifdef DK_TAGGED
    if ((P & 1u) == 0u) {
        unsigned int t = pair_tag(a.p[0], a.p[1], a.u[5] != 0u ? a.p[7] : 0ull, 0ull);
        if (t == 1u) { norm_sums_pairs<1>(a DK_IDA); return; }
        if (t == 2u) { norm_sums_pairs<2>(a DK_IDA); return; }
    }
#endif
    float iP = 1.0f / (float)P, iC = 1.0f / (float)C;
    UNITS(u, C * S) {
        unsigned int c = u / S, sl = u % S;
        dk_double sd = 0.0, sx = 0.0;
        SLICE_LOOP(q, M, sl, S) {
            unsigned int i = CHAN_AT(q, c, C, P), st = G == 0 ? c : norm_stat(i, C, P, G, iP, iC);
            float dy = fused_dy(a, i);
            sd += (dk_double)dy;
            sx += (dk_double)(dy * (CF(1)[i] - CF(2)[st]) * CF(3)[st]);
        }
        sd = block_sum_d(sd DK_IDA);
        sx = block_sum_d(sx DK_IDA);
        if (threadIdx.x == 0) {
            if (a.p[4]) atomicAdd(&FA(4)[c], (float)sx);
            if (a.p[5]) atomicAdd(&FA(5)[c], (float)sd);
            if (a.p[6]) { atomicAdd(&FA(6)[c], (float)sd); atomicAdd(&FA(6)[C + c], (float)sx); }
        }
    }
}
// Batch norm: dx += gamma invstd (dy - m1 - xh m2), m1 = sum dy / N, m2 = sum dy xh / N
// in training (0 in eval). p0 dy, p1 x, p2 mean, p3 invstd, p4 gamma, p5 sums,
// p6 dx; u0 C, u1 plane, u2 training; f0 1/N                    n = values
#ifdef DK_TAGGED
template <int T> __device__ void bn_bwd_dx_pairs(const DevArgs& a DK_IDS) {
    unsigned int C = a.u[0], P = a.u[1];
    float iP = 1.0f / (float)P, iC = 1.0f / (float)C;
    const dk_global unsigned int* xw = pw(a.p[1]);
    dk_global unsigned int* dx = pwm(a.p[6]);
    GS(j, N / 2u) {
        unsigned int c = chan_of(2u * j, C, P, iP, iC);
        float inv = FA(3)[c], mean = FA(2)[c], g = FA(4)[c] * inv;
        float m1 = a.u[2] ? FA(5)[c] * a.f[0] : 0.0f, m2 = a.u[2] ? FA(5)[C + c] * a.f[0] : 0.0f;
        float dlo, dhi;
        fused_dy2<T>(a, j, dlo, dhi);
        unsigned int w = xw[j];
        float xlo = (pk_lo<T>(w) - mean) * inv, xhi = (pk_hi<T>(w) - mean) * inv;
        float vlo = g * (dlo - m1 - xlo * m2), vhi = g * (dhi - m1 - xhi * m2);
        if (!a.u[6]) { unsigned int o = dx[j]; vlo += pk_lo<T>(o); vhi += pk_hi<T>(o); }
        dx[j] = pk<T>(vlo, vhi);
    }
}
#endif
DK_KERNEL(dk_bn_bwd_dx) {
    unsigned int C = a.u[0], P = a.u[1];
#ifdef DK_TAGGED
    if ((P & 1u) == 0u && (N & 1u) == 0u) {
        unsigned int t = pair_tag(a.p[0], a.p[1], a.p[6], a.u[5] != 0u ? a.p[7] : 0ull);
        if (t == 1u) { bn_bwd_dx_pairs<1>(a DK_IDA); return; }
        if (t == 2u) { bn_bwd_dx_pairs<2>(a DK_IDA); return; }
    }
#endif
    float iP = 1.0f / (float)P, iC = 1.0f / (float)C;
    GS(i, N) {
        unsigned int c = chan_of(i, C, P, iP, iC);
        float inv = CF(3)[c];
        float m1 = a.u[2] ? CF(5)[c] * a.f[0] : 0.0f, m2 = a.u[2] ? CF(5)[C + c] * a.f[0] : 0.0f;
        float xh = (CF(1)[i] - CF(2)[c]) * inv;
        ACCUM(F(6)[i], CF(4)[c] * inv * (fused_dy(a, i) - m1 - xh * m2));
    }
}
// Group norm: per (image, group) unit, dxh = dy gamma[c],
// dx += invstd (dxh - mean(dxh) - xh mean(dxh xh)). p0 dy, p1 x, p2 mean,
// p3 invstd, p4 gamma, p5 dx; u0 C, u1 plane, u2 groups, u3 units   n = U*256
DK_KERNEL(dk_gn_bwd) {
    unsigned int C = a.u[0], P = a.u[1], G = a.u[2];
    unsigned int Cg = C / G, M = Cg * P;
    float iP = 1.0f / (float)P;
    UNITS(u, a.u[3]) {
        DK_U64 base = (DK_U64)u * M;
        unsigned int g = u % G;
        float mean = CF(2)[u], inv = CF(3)[u];
        dk_double s1 = 0.0, s2 = 0.0;
        TSPLIT(q, M) {
            float dxh = fused_dy(a, base + q) * CF(4)[g * Cg + udiv(q, P, iP)];
            s1 += (dk_double)dxh;
            s2 += (dk_double)(dxh * (CF(1)[base + q] - mean) * inv);
        }
        float m1 = (float)(block_sum_d(s1 DK_IDA) / (dk_double)M), m2 = (float)(block_sum_d(s2 DK_IDA) / (dk_double)M);
        TSPLIT(q, M) {
            float dxh = fused_dy(a, base + q) * CF(4)[g * Cg + udiv(q, P, iP)];
            float xh = (CF(1)[base + q] - mean) * inv;
            F(5)[base + q] += inv * (dxh - m1 - xh * m2);
        }
    }
}

// ── Pooling ([batch, C*H*W] planes) ──────────────────────────────────────────
// u0 C, u1 H, u2 W, u3 oH, u4 oW, u5 mode (0 max, 1 avg over the padded
// window, 2 avg over the part inside the image); f0..f5 kH, kW, sH, sW, pH, pW.
struct PoolWin { unsigned int h0, h1, w0, w1, padded; };
__device__ PoolWin pool_win(const DevArgs& a, unsigned int oh, unsigned int ow) {
    unsigned int H = a.u[1], W = a.u[2];
    int kH = (int)a.f[0], kW = (int)a.f[1], sH = (int)a.f[2], sW = (int)a.f[3], pH = (int)a.f[4], pW = (int)a.f[5];
    int h0 = (int)oh * sH - pH, w0 = (int)ow * sW - pW, h1 = h0 + kH, w1 = w0 + kW;
    int hp = h1 < (int)H + pH ? h1 : (int)H + pH, wp = w1 < (int)W + pW ? w1 : (int)W + pW;
    PoolWin r;
    r.padded = (unsigned int)((hp - h0) * (wp - w0));
    r.h0 = h0 < 0 ? 0 : h0; r.w0 = w0 < 0 ? 0 : w0;
    r.h1 = h1 > (int)H ? H : h1; r.w1 = w1 > (int)W ? W : w1;
    return r;
}
// p0 x, p1 out [batch, C*oH*oW] (forward); p0 x, p1 dout, p2 dx (backward: u6 = 1)
// n = batch*C*oH*oW
DK_KERNEL(dk_pool) {
    unsigned int H = a.u[1], W = a.u[2], oH = a.u[3], oW = a.u[4], mode = a.u[5];
    float iW = 1.0f / (float)oW, iH = 1.0f / (float)oH;
    GS(i, N) {
        unsigned int t = udiv(i, oW, iW), ow = i - t * oW, pl = udiv(t, oH, iH), oh = t - pl * oH;
        PoolWin r = pool_win(a, oh, ow);
        TP img = CF(0) + (DK_U64)pl * H * W;
        if (mode == 0) {
            unsigned int best = r.h0 * W + r.w0;
            float bv = img[best];
            for (unsigned int y = r.h0; y < r.h1; y++)
                for (unsigned int x = r.w0; x < r.w1; x++) if (img[y * W + x] > bv) { bv = img[y * W + x]; best = y * W + x; }
            if (a.u[6]) atomicAdd(&FA(2)[(DK_U64)pl * H * W + best], CF(1)[i]); else F(1)[i] = bv;
        } else {
            float div = mode == 1 ? (float)r.padded : (float)((r.h1 - r.h0) * (r.w1 - r.w0));
            if (a.u[6]) {
                float share = CF(1)[i] / div;
                for (unsigned int y = r.h0; y < r.h1; y++)
                    for (unsigned int x = r.w0; x < r.w1; x++) atomicAdd(&FA(2)[(DK_U64)pl * H * W + y * W + x], share);
            } else {
                float acc = 0.0f;
                for (unsigned int y = r.h0; y < r.h1; y++) for (unsigned int x = r.w0; x < r.w1; x++) acc += img[y * W + x];
                F(1)[i] = acc / div;
            }
        }
    }
}
// The pooling's backward as a gather (no atomics, any element type): input
// (y, x) sums the gradients of the windows that contain it -- for max
// pooling, of those whose (first) largest value it is. p0 x, p1 dout, p2 dx;
// geometry as dk_pool; u6 = 1: dx was an unwritten zero gradient   n = planes*H*W
DK_KERNEL(dk_pool_bwd) {
    unsigned int H = a.u[1], W = a.u[2], oH = a.u[3], oW = a.u[4], mode = a.u[5];
    int kH = (int)a.f[0], kW = (int)a.f[1], sH = (int)a.f[2], sW = (int)a.f[3], pH = (int)a.f[4], pW = (int)a.f[5];
    if (kH == sH && kW == sW && pH == 0 && pW == 0 &&
        (!a.u[6] || (oH * (unsigned int)kH == H && oW * (unsigned int)kW == W))) {
        // non-overlapping windows partition the inputs: a thread per window
        // adds into its own inputs (n counts inputs; the windows are fewer)
        float iW = 1.0f / (float)oW, iH = 1.0f / (float)oH;
        unsigned int planes = N / (H * W);
        GS(i, planes * oH * oW) {
            unsigned int t = udiv(i, oW, iW), ow = i - t * oW, pl = udiv(t, oH, iH), oh = t - pl * oH;
            PoolWin r = pool_win(a, oh, ow);
            TP img = CF(0) + (DK_U64)pl * H * W;
            TP dx = F(2) + (DK_U64)pl * H * W;
            float d = CF(1)[i];
            if (mode == 0) {
                unsigned int best = r.h0 * W + r.w0;
                float bv = img[best];
                for (unsigned int yy = r.h0; yy < r.h1; yy++)
                    for (unsigned int xx = r.w0; xx < r.w1; xx++) if (img[yy * W + xx] > bv) { bv = img[yy * W + xx]; best = yy * W + xx; }
                if (a.u[6]) {
                    // exact tiling, unwritten gradient: write the window
                    for (unsigned int yy = r.h0; yy < r.h1; yy++)
                        for (unsigned int xx = r.w0; xx < r.w1; xx++) dx[yy * W + xx] = yy * W + xx == best ? d : 0.0f;
                } else dx[best] += d;
            } else {
                float share = d / (mode == 1 ? (float)r.padded : (float)((r.h1 - r.h0) * (r.w1 - r.w0)));
                for (unsigned int yy = r.h0; yy < r.h1; yy++)
                    for (unsigned int xx = r.w0; xx < r.w1; xx++) ACCUM(dx[yy * W + xx], share);
            }
        }
        return;
    }
    float iW = 1.0f / (float)W, iH = 1.0f / (float)H, iK = 1.0f / (float)kH, iKw = 1.0f / (float)kW;
    GS(i, N) {
        unsigned int t = udiv(i, W, iW), x = i - t * W, pl = udiv(t, H, iH), y = t - pl * H;
        TP img = CF(0) + (DK_U64)pl * H * W;
        TP dout = CF(1) + (DK_U64)pl * oH * oW;
        int oh0, oh1, ow0, ow1;
        if (kH == sH && kW == sW && pH == 0 && pW == 0) {
            // non-overlapping windows: exactly one holds (y, x)
            oh0 = oh1 = (int)udiv(y, (unsigned int)kH, iK);
            ow0 = ow1 = (int)udiv(x, (unsigned int)kW, iKw);
        } else {
            oh0 = ((int)y + pH - kH + sH) / sH; oh1 = ((int)y + pH) / sH;   // windows over row y
            ow0 = ((int)x + pW - kW + sW) / sW; ow1 = ((int)x + pW) / sW;
            if ((int)y + pH - kH + 1 < 0) oh0 = 0;
            if ((int)x + pW - kW + 1 < 0) ow0 = 0;
        }
        if (oh1 > (int)oH - 1) oh1 = (int)oH - 1;
        if (ow1 > (int)oW - 1) ow1 = (int)oW - 1;
        float s = 0.0f;
        for (int oh = oh0; oh <= oh1; oh++) {
            for (int ow = ow0; ow <= ow1; ow++) {
                PoolWin r = pool_win(a, (unsigned int)oh, (unsigned int)ow);
                if (y < r.h0 || y >= r.h1 || x < r.w0 || x >= r.w1) continue;
                float d = dout[(unsigned int)oh * oW + (unsigned int)ow];
                if (mode == 0) {
                    unsigned int best = r.h0 * W + r.w0;
                    float bv = img[best];
                    for (unsigned int yy = r.h0; yy < r.h1; yy++)
                        for (unsigned int xx = r.w0; xx < r.w1; xx++) if (img[yy * W + xx] > bv) { bv = img[yy * W + xx]; best = yy * W + xx; }
                    if (best == y * W + x) s += d;
                } else {
                    s += d / (mode == 1 ? (float)r.padded : (float)((r.h1 - r.h0) * (r.w1 - r.w0)));
                }
            }
        }
        ACCUM(F(2)[i], s);
    }
}
// Global pooling, one plane of u0 values per unit: p0 x, p1 out [planes];
// backward (u2 = 1): p1 dout, p2 dx. u1 = 0 max, 1 avg           n = planes*256
DK_KERNEL(dk_gpool) {
    unsigned int P = a.u[0];
    UNITS(pl, N / 256) {
        TP x = CF(0) + (DK_U64)pl * P;
        if (a.u[1]) {
            if (a.u[2]) { float share = CF(1)[pl] / (float)P; TSPLIT(q, P) ACCUM(F(2)[(DK_U64)pl * P + q], share); }
            else { dk_double s = 0.0; TSPLIT(q, P) s += (dk_double)x[q]; s = block_sum_d(s DK_IDA); if (threadIdx.x == 0) F(1)[pl] = (float)(s / (dk_double)P); }
        } else if (threadIdx.x == 0) {
            unsigned int best = 0;
            for (unsigned int q = 1; q < P; q++) if (x[q] > x[best]) best = q;
            if (a.u[2]) F(2)[(DK_U64)pl * P + best] += CF(1)[pl]; else F(1)[pl] = x[best];
        }
    }
}

// ── Upsampling and resizing (planes of H x W -> oH x oW) ─────────────────────
// Nearest by an integer factor u2: forward (u3 = 0) p0 x, p1 out, one thread
// per output; backward (u3 = 1) p0 dout, p1 dx, one per input. u0 W, u1 H
DK_KERNEL(dk_upsample) {
    unsigned int W = a.u[0], H = a.u[1], f = a.u[2], oW = W * f, oH = H * f;
    GS(i, N) {
        if (a.u[3] == 0) {
            unsigned int ow = i % oW, oh = (i / oW) % oH, pl = i / (oW * oH);
            F(1)[i] = CF(0)[((DK_U64)pl * H + oh / f) * W + ow / f];
        } else {
            unsigned int w = i % W, h = (i / W) % H, pl = i / (W * H);
            float s = 0.0f;
            for (unsigned int y = 0; y < f; y++)
                for (unsigned int x = 0; x < f; x++) s += CF(0)[((DK_U64)pl * oH + h * f + y) * oW + w * f + x];
            ACCUM(F(1)[i], s);
        }
    }
}
// Up to four taps per axis (unused ones weigh 0), kept in registers: every
// loop over them is unrolled.
struct Taps { unsigned int i0, i1, i2, i3; float w0, w1, w2, w3; };
__device__ float rs_src(unsigned int o, unsigned int in, unsigned int out, unsigned int align, int clampZero) {
    if (align) return out <= 1 ? 0.0f : (float)o * (float)(in - 1) / (float)(out - 1);
    float s = ((float)o + 0.5f) * (float)in / (float)out - 0.5f;
    if (clampZero && s < 0.0f) s = 0.0f;
    return s;
}
__device__ unsigned int rs_clamp(int j, unsigned int in) { return j < 0 ? 0 : (j >= (int)in ? in - 1 : (unsigned int)j); }
__device__ Taps rs_taps(unsigned int o, unsigned int in, unsigned int out, unsigned int mode, unsigned int align) {
    Taps t;
    t.w0 = t.w1 = t.w2 = t.w3 = 0.0f;
    if (mode == 0) {
        unsigned int i = (unsigned int)floorf((float)o * ((float)in / (float)out));
        t.i0 = t.i1 = t.i2 = t.i3 = i < in ? i : in - 1; t.w0 = 1.0f;
    } else if (mode == 1) {
        float s = rs_src(o, in, out, align, 1);
        unsigned int i0 = (unsigned int)s;
        if (i0 > in - 1) i0 = in - 1;
        float l = s - (float)i0;
        t.i0 = i0; t.i1 = t.i2 = t.i3 = i0 + 1 < in ? i0 + 1 : in - 1;
        t.w0 = 1.0f - l; t.w1 = l;
    } else {
        float s = rs_src(o, in, out, align, 0), f = floorf(s), x = s - f, A = -0.75f;
        float x1 = x + 1.0f, x2 = 1.0f - x, x3 = 2.0f - x;
        t.w0 = ((A * x1 - 5.0f * A) * x1 + 8.0f * A) * x1 - 4.0f * A;
        t.w1 = ((A + 2.0f) * x - (A + 3.0f)) * x * x + 1.0f;
        t.w2 = ((A + 2.0f) * x2 - (A + 3.0f)) * x2 * x2 + 1.0f;
        t.w3 = ((A * x3 - 5.0f * A) * x3 + 8.0f * A) * x3 - 4.0f * A;
        int i = (int)f;
        t.i0 = rs_clamp(i - 1, in); t.i1 = rs_clamp(i, in); t.i2 = rs_clamp(i + 1, in); t.i3 = rs_clamp(i + 2, in);
    }
    return t;
}
// sum_k w_k row[i_k]
__device__ float rs_dot(const dk_thread Taps& t, TP row) {
    float s = 0.0f;
    if (t.w0 != 0.0f) s += t.w0 * row[t.i0];
    if (t.w1 != 0.0f) s += t.w1 * row[t.i1];
    if (t.w2 != 0.0f) s += t.w2 * row[t.i2];
    if (t.w3 != 0.0f) s += t.w3 * row[t.i3];
    return s;
}
// convnet.h's RESIZE_* (u4), alignCorners u5: p0 x, p1 out. u0 H, u1 W, u2 oH,
// u3 oW                                                       n = planes*oH*oW
DK_KERNEL(dk_resize) {
    unsigned int H = a.u[0], W = a.u[1], oH = a.u[2], oW = a.u[3];
    GS(i, N) {
        unsigned int ow = i % oW, oh = (i / oW) % oH, pl = i / (oW * oH);
        Taps ty = rs_taps(oh, H, oH, a.u[4], a.u[5]), tx = rs_taps(ow, W, oW, a.u[4], a.u[5]);
        TP img = CF(0) + (DK_U64)pl * H * W;
        float acc = 0.0f;
        if (ty.w0 != 0.0f) acc += ty.w0 * rs_dot(tx, img + ty.i0 * W);
        if (ty.w1 != 0.0f) acc += ty.w1 * rs_dot(tx, img + ty.i1 * W);
        if (ty.w2 != 0.0f) acc += ty.w2 * rs_dot(tx, img + ty.i2 * W);
        if (ty.w3 != 0.0f) acc += ty.w3 * rs_dot(tx, img + ty.i3 * W);
        F(1)[i] = acc;
    }
}
// The weight with which output o's taps read input i along one axis (0: none).
__device__ float rs_weight(unsigned int o, unsigned int i, unsigned int in, unsigned int out, unsigned int mode,
                           unsigned int align) {
    Taps t = rs_taps(o, in, out, mode, align);
    return (t.i0 == i ? t.w0 : 0.0f) + (t.i1 == i ? t.w1 : 0.0f) + (t.i2 == i ? t.w2 : 0.0f) + (t.i3 == i ? t.w3 : 0.0f);
}
// The outputs that can read input i: their source coordinate lies in
// [i - 1, i + 1) for bilinear taps (floor(s), floor(s) + 1) and [i - 2, i + 2)
// for bicubic ones (floor(s) - 1 .. floor(s) + 2), borders clamped; a margin
// of one either side for rounding.
__device__ void rs_range(unsigned int i, unsigned int in, unsigned int out, unsigned int mode, unsigned int align,
                         dk_thread int* lo, dk_thread int* hi) {
    float a0, a1;
    float lo_s = mode == 1 ? (float)i - 1.0f : (float)i - 2.0f, hi_s = mode == 1 ? (float)i + 1.0f : (float)i + 2.0f;
    if (mode == 0) { float r = (float)out / (float)in; a0 = (float)i * r; a1 = (float)(i + 1) * r; }
    else if (align) {
        float r = in > 1 ? (float)(out - 1) / (float)(in - 1) : (float)out;
        a0 = lo_s * r; a1 = hi_s * r;
    } else {
        float r = (float)out / (float)in;
        a0 = (lo_s + 0.5f) * r - 0.5f; a1 = (hi_s + 0.5f) * r - 0.5f;
    }
    int l = (int)floorf(a0) - 1, h = (int)ceilf(a1) + 1;
    *lo = l < 0 ? 0 : l;
    *hi = h > (int)out - 1 ? (int)out - 1 : h;
}
// The resize's backward as two gathers (no atomics), one axis at a time:
// pass 0 (f0 = 0): t[pl, oy, x] = sum over ox reading x of dout[pl, oy, ox] wx;
// pass 1 (f0 = 1): dx[pl, y, x] (+)= sum over oy reading y of t[pl, oy, x] wy.
// p0 source, p1 destination; u0 H, u1 W, u2 oH, u3 oW, u4 mode, u5 align
// n = planes*oH*W (pass 0), planes*H*W (pass 1)
DK_KERNEL(dk_resize_bwd) {
    unsigned int H = a.u[0], W = a.u[1], oH = a.u[2], oW = a.u[3], mode = a.u[4], align = a.u[5];
    GS(i, N) {
        unsigned int x = i % W, r = (i / W);
        int q0, q1;
        float s = 0.0f;
        if (a.f[0] == 0.0f) {
            // r = pl * oH + oy: a row of dout
            rs_range(x, W, oW, mode, align, &q0, &q1);
            TP d = CF(0) + (DK_U64)r * oW;
            for (int ox = q0; ox <= q1; ox++) {
                float w = rs_weight((unsigned int)ox, x, W, oW, mode, align);
                if (w != 0.0f) s += w * d[ox];
            }
            F(1)[i] = s;
        } else {
            unsigned int y = r % H, pl = r / H;
            rs_range(y, H, oH, mode, align, &q0, &q1);
            TP t = CF(0) + (DK_U64)pl * oH * W + x;
            for (int oy = q0; oy <= q1; oy++) {
                float w = rs_weight((unsigned int)oy, y, H, oH, mode, align);
                if (w != 0.0f) s += w * t[(unsigned int)oy * W];
            }
            ACCUM(F(1)[i], s);
        }
    }
}

// ── 16-bit storage (mixed precision) ─────────────────────────────────────────
// fp16 / bf16 values held as unsigned shorts, converted with round to
// nearest even (the same bits as float16.h's f32_to_fp16 / f32_to_bf16).
// Hardware conversions: v_cvt on AMD (_Float16), cvt.rn on NVIDIA, half on
// Apple GPUs.
#if defined(__METAL_VERSION__)
unsigned short f2h_bits(float f) { return as_type<ushort>(half(f)); }
float h2f_bits(unsigned short b) { return float(as_type<half>(b)); }
#elif defined(__HIPCC_RTC__) || defined(__HIP__)
__device__ unsigned short f2h_bits(float f) { _Float16 h = (_Float16)f; return *(unsigned short*)&h; }
__device__ float h2f_bits(unsigned short b) { return (float)*(_Float16*)&b; }
#else
__device__ unsigned short f2h_bits(float f) { unsigned short r; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(f)); return r; }
__device__ float h2f_bits(unsigned short b) { float r; asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(b)); return r; }
#endif
__device__ unsigned short f2b_bits(float f) {
    unsigned int x = __float_as_uint(f);
    if ((x & 0x7f800000u) == 0x7f800000u && (x & 0x7fffffu)) return (unsigned short)((x >> 16) | 0x40u);
    return (unsigned short)((x + 0x7fffu + ((x >> 16) & 1u)) >> 16);
}
__device__ float b2f_bits(unsigned short b) { return __uint_as_float((unsigned int)b << 16); }
// The dynamic loss scale's state (one per device, 4 floats): [scale,
// 1/scale, overflow flag, good steps].
// g *= 1/scale, the flag raised on a non-finite value: p0 g, p1 state   n = values
DK_KERNEL(dk_amp_unscale) {
    TP st = F(1);
    float inv = st[1];
    GS(i, N) {
        float v = F(0)[i] * inv;
        if (!isfinite(v)) st[2] = 1.0f;
        F(0)[i] = v;
    }
}
// Every gradient of a table at once: p0 = u1 entries {g, n, start} (u64),
// p1 state                                                     n = total values
DK_KERNEL(dk_amp_unscale_multi) {
    const dk_global DK_U64* T = (const dk_global DK_U64*)a.p[0];
    unsigned int cnt = a.u[1];
    TP st = F(1);
    float inv = st[1];
    GS(i, N) {
        unsigned int lo = 0, hi = cnt - 1;
        while (lo < hi) { unsigned int mid = (lo + hi + 1) / 2; if (T[mid * 3 + 2] <= i) lo = mid; else hi = mid - 1; }
        dk_global float* g = (dk_global float*)T[lo * 3];
        unsigned int j = i - (unsigned int)T[lo * 3 + 2];
        float v = g[j] * inv;
        if (!isfinite(v)) st[2] = 1.0f;
        g[j] = v;
    }
}
// After a step: an overflow halves the scale, u0 good steps in a row double
// it; the flag is cleared. p0 state                             n = 1
DK_KERNEL(dk_amp_update) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        TP st = F(0);
        if (st[2] != 0.0f) { st[0] = fmaxf(st[0] * 0.5f, 1.0f); st[3] = 0.0f; }
        else {
            st[3] += 1.0f;
            if (st[3] >= (float)a.u[0]) { st[0] = fminf(st[0] * 2.0f, 16777216.0f); st[3] = 0.0f; }
        }
        st[1] = 1.0f / st[0];
        st[2] = 0.0f;
    }
}

// ── Sparse (CSR) x dense: sparse.h ───────────────────────────────────────────
// out [rows, n] (+)= S [rows, cols] . X [cols, n]; p0 rowPtr, p1 colIdx (as
// float32 indices), p2 values, p3 X, p4 out; u0 = n, u1 = 1 to accumulate   n = rows*ncols
DK_KERNEL(dk_spmm) {
    unsigned int NC = a.u[0];
    GS(i, N) {
        unsigned int r = i / NC, j = i % NC;
        unsigned int lo = (unsigned int)CF(0)[r], hi = (unsigned int)CF(0)[r + 1];
        float s = 0.0f;
        for (unsigned int e = lo; e < hi; e++) s += CF(2)[e] * CF(3)[(unsigned int)CF(1)[e] * NC + j];
        if (a.u[1]) F(4)[i] += s; else F(4)[i] = s;
    }
}
// dX[cols, n] += S^T . G: one thread per (nonzero, column)          n = nnz*ncols
// p5 = row of each nonzero
DK_KERNEL(dk_spmm_tgrad) {
    unsigned int NC = a.u[0];
    GS(i, N) {
        unsigned int e = i / NC, j = i % NC;
        unsigned int r = (unsigned int)CF(5)[e], col = (unsigned int)CF(1)[e];
        atomicAdd(&FA(4)[col * NC + j], CF(2)[e] * CF(3)[r * NC + j]);
    }
}
// dvalues[e] += G[row(e), :] . X[col(e), :]                        n = nnz
// p1 colIdx, p3 X, p4 G, p5 row of each nonzero, p6 dvalues
DK_KERNEL(dk_spmm_vgrad) {
    unsigned int NC = a.u[0];
    GS(e, N) {
        unsigned int r = (unsigned int)CF(5)[e], col = (unsigned int)CF(1)[e];
        float s = 0.0f;
        for (unsigned int j = 0; j < NC; j++) s += CF(4)[r * NC + j] * CF(3)[col * NC + j];
        F(6)[e] += s;
    }
}

// ── Metal only: GEMMs and copies (rocBLAS / cuBLAS and the copy engines' role
// on the other backends; MpsBackend launches these by name) ────────────────
#if defined(__METAL_VERSION__)
// A strided batch of c_i = alpha op(a_i) op(b_i) + beta c_i (device.h's
// DevGemm), row-major: a and b fp32, fp16 or bf16 (MK_T), c any of them
// (ct), products accumulated in fp32 on the 8 x 8 simdgroup matrix units.
// A threadgroup of four simdgroups takes a BM x BN tile of c (32 x 32 or
// 64 x 64: mk_gemm<BM>_<op(a)><op(b)>_<type>), k in steps of BK through
// threadgroup memory; the grid is (n / BN, m / BM, batch). Tiles inside the
// matrices load without bounds checks (a divergent load per element costs
// 3x) and store straight from the matrix units, alpha and beta applied as
// products with diagonal matrices; edge tiles check every element.
struct DkGemm {
    ulong a, b, c;          // addresses
    ulong lda, ldb, ldc;
    ulong sa, sb, sc;       // batch strides, in elements
    uint m, n, k, ta, tb, at, ct;
    float alpha, beta;
};
template <uint T> inline float mk_ld(ulong base, uint i) {
    if (T == 0u) return ((const device float*)base)[i];
    ushort h = ((const device ushort*)base)[i];
    return T == 1u ? h2f_bits(h) : b2f_bits(h);
}
inline float mk_ld_any(ulong base, ulong i, uint t) {
    if (t == 0u) return ((const device float*)base)[i];
    ushort h = ((const device ushort*)base)[i];
    return t == 1u ? h2f_bits(h) : b2f_bits(h);
}
inline void mk_st(ulong base, ulong i, uint t, float v) {
    if (t == 0u) ((device float*)base)[i] = v;
    else ((device ushort*)base)[i] = t == 1u ? f2h_bits(v) : f2b_bits(v);
}
// One BK step of the tile product. The tiles are As [BM][BK] and Bs [BK][BN],
// or k-major As [BK][BM] (AK) and n-major Bs [BN][BK] (BNM) where that lets
// neighbouring threads store to neighbouring words; the matrix loads
// transpose them back.
template <uint BM, uint BN, uint BK, uint SM, uint SN, bool AK = false, bool BNM = false>
inline void mk_mma(threadgroup float* As, threadgroup float* Bs, thread simdgroup_float8x8 (&acc)[SM][SN], uint sr, uint sc) {
    #pragma clang loop unroll(full)
    for (uint kk = 0; kk < BK; kk += 8u) {
        simdgroup_float8x8 fa[SM], fb[SN];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < SM; i++) {
            if (AK) simdgroup_load(fa[i], &As[kk * BM + sr + i * 8u], BM, ulong2(0, 0), true);
            else simdgroup_load(fa[i], &As[(sr + i * 8u) * BK + kk], BK);
        }
        #pragma clang loop unroll(full)
        for (uint j = 0; j < SN; j++) {
            if (BNM) simdgroup_load(fb[j], &Bs[(sc + j * 8u) * BK + kk], BK, ulong2(0, 0), true);
            else simdgroup_load(fb[j], &Bs[kk * BN + sc + j * 8u], BN);
        }
        #pragma clang loop unroll(full)
        for (uint i = 0; i < SM; i++)
            #pragma clang loop unroll(full)
            for (uint j = 0; j < SN; j++) simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
}

template <uint T, bool TA, bool TB, uint BM, uint BN, uint BK>
inline void mk_gemm(constant DkGemm& g, uint3 tg, uint tid, uint sg, threadgroup float* As, threadgroup float* Bs) {
    constexpr uint SM = BM / 16u, SN = BN / 16u;   // 8 x 8 fragments per simdgroup (2 x 2 simdgroups)
    constexpr ulong es = T == 0u ? 4ul : 2ul;
    uint m0 = tg.y * BM, n0 = tg.x * BN;
    ulong A = g.a + (ulong)tg.z * g.sa * es, B = g.b + (ulong)tg.z * g.sb * es;
    ulong C = g.c + (ulong)tg.z * g.sc * (g.ct == 0u ? 4ul : 2ul);
    uint sr = (sg / 2u) * SM * 8u, sc = (sg % 2u) * SN * 8u;
    uint lda = (uint)g.lda, ldb = (uint)g.ldb, Mx = g.m, Nx = g.n, Kx = g.k;
    bool inside = m0 + BM <= Mx && n0 + BN <= Nx;
    simdgroup_float8x8 acc[SM][SN];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < SM; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < SN; j++) acc[i][j] = simdgroup_float8x8(0.0f);
    for (uint k0 = 0; k0 < Kx; k0 += BK) {
        // neighbouring threads read neighbouring addresses either way round
        if (inside && k0 + BK <= Kx) {
            #pragma clang loop unroll(full)
            for (uint q = 0; q < BM * BK / 128u; q++) {
                uint e = tid + q * 128u, r = TA ? e % BM : e / BK, c = TA ? e / BM : e % BK;
                As[TA ? c * BM + r : r * BK + c] = mk_ld<T>(A, TA ? (k0 + c) * lda + m0 + r : (m0 + r) * lda + k0 + c);
            }
            #pragma clang loop unroll(full)
            for (uint q = 0; q < BK * BN / 128u; q++) {
                uint e = tid + q * 128u, r = TB ? e % BK : e / BN, c = TB ? e / BK : e % BN;
                Bs[TB ? c * BK + r : r * BN + c] = mk_ld<T>(B, TB ? (n0 + c) * ldb + k0 + r : (k0 + r) * ldb + n0 + c);
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint q = 0; q < BM * BK / 128u; q++) {
                uint e = tid + q * 128u, r = TA ? e % BM : e / BK, c = TA ? e / BM : e % BK;
                uint gm = m0 + r, gk = k0 + c;
                float v = 0.0f;
                if (gm < Mx && gk < Kx) v = mk_ld<T>(A, TA ? gk * lda + gm : gm * lda + gk);
                As[TA ? c * BM + r : r * BK + c] = v;
            }
            #pragma clang loop unroll(full)
            for (uint q = 0; q < BK * BN / 128u; q++) {
                uint e = tid + q * 128u, r = TB ? e % BK : e / BN, c = TB ? e / BK : e % BN;
                uint gk = k0 + r, gn = n0 + c;
                float v = 0.0f;
                if (gk < Kx && gn < Nx) v = mk_ld<T>(B, TB ? gn * ldb + gk : gk * ldb + gn);
                Bs[TB ? c * BK + r : r * BN + c] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mk_mma<BM, BN, BK, SM, SN, TA, TB>(As, Bs, acc, sr, sc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (inside && g.ct == 0u) {
        // c = alpha acc + beta c on the matrix units (diagonal alpha, beta)
        simdgroup_float8x8 da = simdgroup_float8x8(g.alpha), db = simdgroup_float8x8(g.beta);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < SM; i++)
            #pragma clang loop unroll(full)
            for (uint j = 0; j < SN; j++) {
                device float* o = (device float*)C + (ulong)(m0 + sr + i * 8u) * g.ldc + n0 + sc + j * 8u;
                simdgroup_float8x8 r = acc[i][j];
                if (g.alpha != 1.0f) simdgroup_multiply(r, r, da);
                if (g.beta != 0.0f) {
                    simdgroup_float8x8 old;
                    simdgroup_load(old, o, g.ldc);
                    simdgroup_multiply_accumulate(r, old, db, r);
                }
                simdgroup_store(r, o, g.ldc);
            }
        return;
    }
    // an edge tile or a 16-bit c: each simdgroup's fragments through its own
    // 8 x 8 of threadgroup memory, then element by element
    threadgroup float* Cs = As + sg * 64u;
    uint lane = tid % 32u;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < SM; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < SN; j++) {
            simdgroup_store(acc[i][j], Cs, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (uint q = 0; q < 2u; q++) {
                uint e = lane + q * 32u;
                uint gm = m0 + sr + i * 8u + e / 8u, gn = n0 + sc + j * 8u + e % 8u;
                if (gm < Mx && gn < Nx) {
                    ulong o = (ulong)gm * g.ldc + gn;
                    float v = g.alpha * Cs[e];
                    if (g.beta != 0.0f) v += g.beta * mk_ld_any(C, o, g.ct);   // beta 0: c is not read
                    mk_st(C, o, g.ct, v);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
}
#define MK_GEMM(name, T, TA, TB, BM, BN, BK) \
kernel void name(constant DkGemm& g [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float As[BM * BK], Bs[BK * BN]; \
    mk_gemm<T, TA, TB, BM, BN, BK>(g, tg, tid, sg, As, Bs); \
}
#define MK_GEMMS(T, sfx) \
    MK_GEMM(mk_gemm32_nn_##sfx, T, false, false, 32u, 32u, 32u) \
    MK_GEMM(mk_gemm32_tn_##sfx, T, true, false, 32u, 32u, 32u) \
    MK_GEMM(mk_gemm32_nt_##sfx, T, false, true, 32u, 32u, 32u) \
    MK_GEMM(mk_gemm32_tt_##sfx, T, true, true, 32u, 32u, 32u) \
    MK_GEMM(mk_gemm64_nn_##sfx, T, false, false, 64u, 64u, 16u) \
    MK_GEMM(mk_gemm64_tn_##sfx, T, true, false, 64u, 64u, 16u) \
    MK_GEMM(mk_gemm64_nt_##sfx, T, false, true, 64u, 64u, 16u) \
    MK_GEMM(mk_gemm64_tt_##sfx, T, true, true, 64u, 64u, 16u)
MK_GEMMS(0u, f32)
MK_GEMMS(1u, f16)
MK_GEMMS(2u, bf16)
// Convolutions as implicit GEMMs (MpsBackend::conv: MIOpen / cuDNN's role),
// the column matrix gathered tile by tile instead of written out by im2col.
// In the terms of a plain convolution -- input I [B, Ci, Hi, Wi], weights
// W [Co, Ci, kH, kW], output O [B, Co, Ho, Wo] (a transposed convolution is
// one with I and O swapped) --
//   mk_conv_fwd     O = conv(I, W):        m = co, n = (b, oy, ox), k = (ci, ky, kx)
//   mk_conv_bwd_x   I = conv^T(O, W):      m = ci, n = (b, y, x),   k = (co, ky, kx),
//                   a gather over the windows holding (y, x), no atomics
//   mk_conv_bwd_w   W += O . I-columns:    m = co, n = (ci, ky, kx), k = (b, oy, ox),
//                   split over k (grid z), the slices added atomically (fp32)
// Gathered operands load from a clamped address and select 0 outside the
// image, so neighbouring threads never diverge on a load.
struct DkConv {
    ulong x, w, y;          // I, W, O
    uint B, Ci, Hi, Wi, Co, Ho, Wo, kH, kW, sH, sW, pH, pW, dH, dW;
    uint kSplit;            // mk_conv_bwd_w: k per grid z slice
};
// q / d and q % d through a reciprocal, for q < 2^31
struct MkDiv {
    uint d; float inv;
    inline uint div(uint q) const { return dk_udiv(q, d, inv); }
};
inline MkDiv mk_div(uint d) { MkDiv r; r.d = d; r.inv = 1.0f / (float)d; return r; }

// Writes the tile (m0, n0) of an [M, B x P] product into a [B, M, P] tensor
// (O or I): straight from the matrix units when the tile is whole and P a
// multiple of 8, else element by element.
template <uint T, uint BM, uint BN, uint SM, uint SN>
inline void mk_conv_store(thread simdgroup_float8x8 (&acc)[SM][SN], ulong out, uint M, uint Ntot, uint P, MkDiv dP,
                          uint m0, uint n0, uint sr, uint sc, uint tid, uint sg, threadgroup float* scratch) {
    if (T == 0u && m0 + BM <= M && n0 + BN <= Ntot && P % 8u == 0u) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < SM; i++)
            #pragma clang loop unroll(full)
            for (uint j = 0; j < SN; j++) {
                uint n = n0 + sc + j * 8u, b = dP.div(n), p = n - b * P, m = m0 + sr + i * 8u;
                simdgroup_store(acc[i][j], (device float*)out + ((ulong)b * M + m) * P + p, P);
            }
        return;
    }
    threadgroup float* Cs = scratch + sg * 64u;
    uint lane = tid % 32u;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < SM; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < SN; j++) {
            simdgroup_store(acc[i][j], Cs, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (uint q = 0; q < 2u; q++) {
                uint e = lane + q * 32u, m = m0 + sr + i * 8u + e / 8u, n = n0 + sc + j * 8u + e % 8u;
                if (m < M && n < Ntot) {
                    uint b = dP.div(n), p = n - b * P;
                    mk_st(out, ((ulong)b * M + m) * P + p, T, Cs[e]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
}

// The tile shapes (BM x BN, k steps of BK, four simdgroups as (4 / WN) x WN):
// 32 x 32 and 64 x 64, and for few output channels 32 x 128 and 8 x 128.
// When the channels summed over (Ci forward, Co for the input gradient) are
// a multiple of BK, k runs over (tap, channel) instead of (channel, tap): a
// step of k is then one filter tap and BK channels, so a thread finds its
// pixel's tap position once per step and its loads differ by a channel
// stride. The weight gradient's k = (b, oy, ox) likewise takes whole steps
// within one image (and one row of it when Wo allows).
#define MK_TILE_SETUP \
    constexpr uint WM = 4u / WN, SM = BM / (WM * 8u), SN = BN / (WN * 8u); \
    uint m0 = tg.y * BM, n0 = tg.x * BN, sr = (sg / WN) * SM * 8u, sc = (sg % WN) * SN * 8u; \
    simdgroup_float8x8 acc[SM][SN]; \
    _Pragma("clang loop unroll(full)") \
    for (uint i = 0; i < SM; i++) \
        _Pragma("clang loop unroll(full)") \
        for (uint j = 0; j < SN; j++) acc[i][j] = simdgroup_float8x8(0.0f);

template <uint T, uint BM, uint BN, uint BK, uint WN>
inline void mk_conv_fwd(constant DkConv& c, uint3 tg, uint tid, uint sg, threadgroup float* As, threadgroup float* Bs) {
    MK_TILE_SETUP
    uint KK = c.kH * c.kW, M = c.Co, K = c.Ci * KK, Po = c.Ho * c.Wo, Ntot = c.B * Po, HW = c.Hi * c.Wi;
    bool cb = c.Ci % BK == 0u;
    MkDiv dKK = mk_div(KK), dKW = mk_div(c.kW), dPo = mk_div(Po), dWo = mk_div(c.Wo), dCi = mk_div(c.Ci);
    // this thread's column of the B tile: output pixel n = (b, oy, ox), fixed
    uint n = min(n0 + tid % BN, Ntot - 1u), b = dPo.div(n), op = n - b * Po, oy = dWo.div(op), ox = op - oy * c.Wo;
    int y0 = (int)(oy * c.sH) - (int)c.pH, x0 = (int)(ox * c.sW) - (int)c.pW;
    bool nIn = n0 + tid % BN < Ntot;
    ulong img = c.x + (ulong)b * c.Ci * HW * (T == 0u ? 4ul : 2ul);
    for (uint k0 = 0; k0 < K; k0 += BK) {
        if (cb) {
            uint tap = dCi.div(k0), ci0 = k0 - tap * c.Ci, ky = dKW.div(tap), kx = tap - ky * c.kW;
            #pragma clang loop unroll(full)
            for (uint q = 0; q < BM * BK / 128u; q++) {
                uint e = tid + q * 128u, r = e / BK, cc = e % BK, gm = m0 + r;
                float v = mk_ld<T>(c.w, (min(gm, M - 1u) * c.Ci + ci0 + cc) * KK + tap);
                As[r * BK + cc] = gm < M ? v : 0.0f;
            }
            int y = y0 + (int)(ky * c.dH), x = x0 + (int)(kx * c.dW);
            bool ok = nIn && y >= 0 && y < (int)c.Hi && x >= 0 && x < (int)c.Wi;
            uint pix = (uint)clamp(y, 0, (int)c.Hi - 1) * c.Wi + (uint)clamp(x, 0, (int)c.Wi - 1);
            #pragma clang loop unroll(full)
            for (uint q = 0; q < BK * BN / 128u; q++) {
                uint r = (tid + q * 128u) / BN;
                float v = mk_ld<T>(img, (ci0 + r) * HW + pix);
                Bs[r * BN + tid % BN] = ok ? v : 0.0f;
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint q = 0; q < BM * BK / 128u; q++) {
                uint e = tid + q * 128u, r = e / BK, cc = e % BK, gm = m0 + r, gk = k0 + cc;
                float v = mk_ld<T>(c.w, min(gm, M - 1u) * K + min(gk, K - 1u));
                As[r * BK + cc] = gm < M && gk < K ? v : 0.0f;
            }
            #pragma clang loop unroll(full)
            for (uint q = 0; q < BK * BN / 128u; q++) {
                uint r = (tid + q * 128u) / BN, gk = k0 + r;
                uint kc = min(gk, K - 1u), ci = dKK.div(kc), kr = kc - ci * KK, ky = dKW.div(kr), kx = kr - ky * c.kW;
                int y = y0 + (int)(ky * c.dH), x = x0 + (int)(kx * c.dW);
                bool ok = nIn && gk < K && y >= 0 && y < (int)c.Hi && x >= 0 && x < (int)c.Wi;
                uint pix = (uint)clamp(y, 0, (int)c.Hi - 1) * c.Wi + (uint)clamp(x, 0, (int)c.Wi - 1);
                float v = mk_ld<T>(img, ci * HW + pix);
                Bs[r * BN + tid % BN] = ok ? v : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mk_mma<BM, BN, BK, SM, SN>(As, Bs, acc, sr, sc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mk_conv_store<T, BM, BN, SM, SN>(acc, c.y, M, Ntot, Po, dPo, m0, n0, sr, sc, tid, sg, As);
}

// The input gradient, by stride phase (grid z, without dilation): the input
// pixels y = y0 + j sH (and likewise x) whose windows reach them through the
// taps ky = ry + i sH only, so no step of k is spent on taps that miss --
// 3 in 4 of them for a stride-2 transposed convolution's forward pass.
template <uint T, uint BM, uint BN, uint BK, uint WN>
inline void mk_conv_bwd_x(constant DkConv& c, uint3 tg, uint tid, uint sg, threadgroup float* As, threadgroup float* Bs) {
    MK_TILE_SETUP
    uint KK = c.kH * c.kW, M = c.Ci, Pi = c.Hi * c.Wi, Po = c.Ho * c.Wo;
    bool phased = c.dH == 1u && c.dW == 1u;
    // the phase: taps ky = ry + i sH, kx = rx + j sW; pixels from (yb, xb) in steps of the stride
    uint sH = phased ? c.sH : 1u, sW = phased ? c.sW : 1u;
    uint ry = tg.z / sW, rx = tg.z - ry * sW;
    uint yb = (ry + sH - c.pH % sH) % sH, xb = (rx + sW - c.pW % sW) % sW;
    uint py = yb < c.Hi ? (c.Hi - yb + sH - 1u) / sH : 0u, px = xb < c.Wi ? (c.Wi - xb + sW - 1u) / sW : 0u;
    uint ty = phased ? (ry < c.kH ? (c.kH - ry + sH - 1u) / sH : 0u) : c.kH;
    uint tx = phased ? (rx < c.kW ? (c.kW - rx + sW - 1u) / sW : 0u) : c.kW;
    uint Pp = py * px, Ntot = c.B * Pp, K = c.Co * ty * tx;
    if (n0 >= Ntot) return;   // a phase with fewer pixels: the whole threadgroup
    bool cb = c.Co % BK == 0u;
    MkDiv dPp = mk_div(max(Pp, 1u)), dPx = mk_div(max(px, 1u)), dTT = mk_div(max(ty * tx, 1u)), dTx = mk_div(max(tx, 1u)),
          dSH = mk_div(c.sH), dSW = mk_div(c.sW), dCo = mk_div(c.Co);
    // this thread's column: input pixel n = (b, y, x)
    uint n = min(n0 + tid % BN, Ntot - 1u), b = dPp.div(n), ip = n - b * Pp, jy = dPx.div(ip), jx = ip - jy * px;
    uint y = yb + jy * sH, x = xb + jx * sW;
    int ny0 = (int)(y + c.pH), nx0 = (int)(x + c.pW);
    bool nIn = n0 + tid % BN < Ntot;
    ulong img = c.y + (ulong)b * c.Co * Po * (T == 0u ? 4ul : 2ul);
    for (uint k0 = 0; k0 < K; k0 += BK) {
        if (cb) {
            // A (m = ci, k = (tap, co)) = W[co, ci, ky, kx]: neighbouring threads take neighbouring ci
            uint t = dCo.div(k0), co0 = k0 - t * c.Co, iy = dTx.div(t), ix = t - iy * tx;
            uint ky = phased ? ry + iy * sH : iy, kx = phased ? rx + ix * sW : ix;
            #pragma clang loop unroll(full)
            for (uint q = 0; q < BM * BK / 128u; q++) {
                uint e = tid + q * 128u, r = e % BM, cc = e / BM, gm = m0 + r;
                float v = mk_ld<T>(c.w, ((co0 + cc) * c.Ci + min(gm, M - 1u)) * KK + ky * c.kW + kx);
                As[cc * BM + r] = gm < M ? v : 0.0f;   // k-major
            }
            // B (k = (tap, co), n = (b, y, x)) = O[b, co, oy, ox], the window at
            // (oy, ox) holding (y, x) through this tap
            int ny = ny0 - (int)(ky * c.dH), nx = nx0 - (int)(kx * c.dW);
            uint oy = dSH.div((uint)max(ny, 0)), ox = dSW.div((uint)max(nx, 0));
            bool ok = nIn && ny >= 0 && nx >= 0 && oy * c.sH == (uint)ny && ox * c.sW == (uint)nx && oy < c.Ho && ox < c.Wo;
            uint pix = min(oy, c.Ho - 1u) * c.Wo + min(ox, c.Wo - 1u);
            #pragma clang loop unroll(full)
            for (uint q = 0; q < BK * BN / 128u; q++) {
                uint r = (tid + q * 128u) / BN;
                float v = mk_ld<T>(img, (co0 + r) * Po + pix);
                Bs[r * BN + tid % BN] = ok ? v : 0.0f;
            }
        } else {
            // k = (co, tap)
            #pragma clang loop unroll(full)
            for (uint q = 0; q < BM * BK / 128u; q++) {
                uint e = tid + q * 128u, r = e % BM, cc = e / BM, gm = m0 + r, gk = k0 + cc;
                uint kc = min(gk, K - 1u), co = dTT.div(kc), t = kc - co * ty * tx, iy = dTx.div(t), ix = t - iy * tx;
                uint ky = phased ? ry + iy * sH : iy, kx = phased ? rx + ix * sW : ix;
                float v = mk_ld<T>(c.w, (co * c.Ci + min(gm, M - 1u)) * KK + ky * c.kW + kx);
                As[cc * BM + r] = gm < M && gk < K ? v : 0.0f;
            }
            #pragma clang loop unroll(full)
            for (uint q = 0; q < BK * BN / 128u; q++) {
                uint r = (tid + q * 128u) / BN, gk = k0 + r;
                uint kc = min(gk, K - 1u), co = dTT.div(kc), t = kc - co * ty * tx, iy = dTx.div(t), ix = t - iy * tx;
                uint ky = phased ? ry + iy * sH : iy, kx = phased ? rx + ix * sW : ix;
                int ny = ny0 - (int)(ky * c.dH), nx = nx0 - (int)(kx * c.dW);
                uint oy = dSH.div((uint)max(ny, 0)), ox = dSW.div((uint)max(nx, 0));
                bool ok = nIn && gk < K && ny >= 0 && nx >= 0 && oy * c.sH == (uint)ny && ox * c.sW == (uint)nx && oy < c.Ho && ox < c.Wo;
                float v = mk_ld<T>(img, co * Po + min(oy, c.Ho - 1u) * c.Wo + min(ox, c.Wo - 1u));
                Bs[r * BN + tid % BN] = ok ? v : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mk_mma<BM, BN, BK, SM, SN, true>(As, Bs, acc, sr, sc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sH == 1u && sW == 1u) {
        mk_conv_store<T, BM, BN, SM, SN>(acc, c.x, M, Ntot, Pi, dPp, m0, n0, sr, sc, tid, sg, As);
        return;
    }
    // a phase's pixels are strided: element by element
    threadgroup float* Cs = As + sg * 64u;
    uint lane = tid % 32u;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < SM; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < SN; j++) {
            simdgroup_store(acc[i][j], Cs, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (uint q = 0; q < 2u; q++) {
                uint e = lane + q * 32u, m = m0 + sr + i * 8u + e / 8u, nn = n0 + sc + j * 8u + e % 8u;
                if (m < M && nn < Ntot) {
                    uint bb = dPp.div(nn), pp = nn - bb * Pp, qy = dPx.div(pp), qx = pp - qy * px;
                    mk_st(c.x, ((ulong)bb * M + m) * Pi + (yb + qy * sH) * c.Wi + xb + qx * sW, T, Cs[e]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
}

// fp32 only (atomics); W must be zeroed first
template <uint BM, uint BN, uint BK, uint WN>
inline void mk_conv_bwd_w(constant DkConv& c, uint3 tg, uint tid, uint sg, threadgroup float* As, threadgroup float* Bs) {
    MK_TILE_SETUP
    uint KK = c.kH * c.kW, M = c.Co, Ntot = c.Ci * KK, Po = c.Ho * c.Wo, K = c.B * Po, HW = c.Hi * c.Wi;
    uint kBeg = tg.z * c.kSplit, kEnd = min(K, kBeg + c.kSplit);
    bool img1 = Po % BK == 0u && c.kSplit % BK == 0u, row1 = img1 && c.Wo % BK == 0u;   // a step within one image / row
    MkDiv dPo = mk_div(Po), dWo = mk_div(c.Wo), dKK = mk_div(KK), dKW = mk_div(c.kW);
    // this thread's column of B: n = (ci, ky, kx)
    uint n = min(n0 + tid % BN, Ntot - 1u), ci = dKK.div(n), kr = n - ci * KK, ky = dKW.div(kr), kx = kr - ky * c.kW;
    int yo = (int)(ky * c.dH) - (int)c.pH, xo = (int)(kx * c.dW) - (int)c.pW;
    bool nIn = n0 + tid % BN < Ntot;
    const device float* O = (const device float*)c.y;
    const device float* I = (const device float*)c.x;
    for (uint k0 = kBeg; k0 < kEnd; k0 += BK) {
        if (img1) {
            uint b = dPo.div(k0), p0 = k0 - b * Po;
            // A (m = co, k = (b, p)) = O[b, co, p]
            #pragma clang loop unroll(full)
            for (uint q = 0; q < BM * BK / 128u; q++) {
                uint e = tid + q * 128u, r = e / BK, cc = e % BK, gm = m0 + r;
                float v = O[((ulong)b * c.Co + min(gm, M - 1u)) * Po + p0 + cc];
                As[r * BK + cc] = gm < M ? v : 0.0f;
            }
            const device float* src = I + ((ulong)b * c.Ci + ci) * HW;
            uint oy0 = dWo.div(p0), ox0 = p0 - oy0 * c.Wo;
            #pragma clang loop unroll(full)
            for (uint q = 0; q < BK * BN / 128u; q++) {
                uint r = (tid + q * 128u) / BN;
                uint oy = oy0, ox = ox0 + r;
                if (!row1) { uint p = p0 + r; oy = dWo.div(p); ox = p - oy * c.Wo; }
                int y = (int)(oy * c.sH) + yo, x = (int)(ox * c.sW) + xo;
                bool ok = nIn && y >= 0 && y < (int)c.Hi && x >= 0 && x < (int)c.Wi;
                float v = src[(uint)clamp(y, 0, (int)c.Hi - 1) * c.Wi + (uint)clamp(x, 0, (int)c.Wi - 1)];
                Bs[r * BN + tid % BN] = ok ? v : 0.0f;
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint q = 0; q < BM * BK / 128u; q++) {
                uint e = tid + q * 128u, r = e / BK, cc = e % BK, gm = m0 + r, gk = k0 + cc;
                uint kc = min(gk, kEnd - 1u), b = dPo.div(kc), op = kc - b * Po;
                float v = O[((ulong)b * c.Co + min(gm, M - 1u)) * Po + op];
                As[r * BK + cc] = gm < M && gk < kEnd ? v : 0.0f;
            }
            #pragma clang loop unroll(full)
            for (uint q = 0; q < BK * BN / 128u; q++) {
                uint r = (tid + q * 128u) / BN, gk = k0 + r;
                uint kc = min(gk, kEnd - 1u), b = dPo.div(kc), op = kc - b * Po, oy = dWo.div(op), ox = op - oy * c.Wo;
                int y = (int)(oy * c.sH) + yo, x = (int)(ox * c.sW) + xo;
                bool ok = nIn && gk < kEnd && y >= 0 && y < (int)c.Hi && x >= 0 && x < (int)c.Wi;
                float v = I[(((ulong)b * c.Ci + ci) * c.Hi + (uint)clamp(y, 0, (int)c.Hi - 1)) * c.Wi + (uint)clamp(x, 0, (int)c.Wi - 1)];
                Bs[r * BN + tid % BN] = ok ? v : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mk_mma<BM, BN, BK, SM, SN>(As, Bs, acc, sr, sc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* Cs = As + sg * 64u;
    uint lane = tid % 32u;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < SM; i++)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < SN; j++) {
            simdgroup_store(acc[i][j], Cs, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (uint q = 0; q < 2u; q++) {
                uint e = lane + q * 32u, m = m0 + sr + i * 8u + e / 8u, nn = n0 + sc + j * 8u + e % 8u;
                if (m < M && nn < Ntot) atomicAdd((device float*)c.w + (ulong)m * Ntot + nn, Cs[e]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
}

// mk_conv_<op><shape>_<type>: shapes 32 (32 x 32), 64 (64 x 64), w (32 x 128), s (8 x 128)
#define MK_CONV(name, call, BM, BN) \
kernel void name(constant DkConv& c [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float As[BM * 16u > 8u * 64u ? BM * 16u : 8u * 64u], Bs[16u * BN]; \
    call(c, tg, tid, sg, As, Bs); \
}
#define MK_CONV_SHAPES(op, T, sfx) \
    MK_CONV(mk_conv_##op##32_##sfx, (mk_conv_##op<T, 32u, 32u, 16u, 2u>), 32u, 32u) \
    MK_CONV(mk_conv_##op##64_##sfx, (mk_conv_##op<T, 64u, 64u, 16u, 2u>), 64u, 64u) \
    MK_CONV(mk_conv_##op##w_##sfx, (mk_conv_##op<T, 32u, 128u, 16u, 4u>), 32u, 128u) \
    MK_CONV(mk_conv_##op##s_##sfx, (mk_conv_##op<T, 8u, 128u, 16u, 4u>), 8u, 128u)
MK_CONV_SHAPES(fwd, 0u, f32)
MK_CONV_SHAPES(fwd, 1u, f16)
MK_CONV_SHAPES(fwd, 2u, bf16)
MK_CONV_SHAPES(bwd_x, 0u, f32)
MK_CONV_SHAPES(bwd_x, 1u, f16)
MK_CONV_SHAPES(bwd_x, 2u, bf16)
MK_CONV(mk_conv_bwd_w32_f32, (mk_conv_bwd_w<32u, 32u, 16u, 2u>), 32u, 32u)
MK_CONV(mk_conv_bwd_w64_f32, (mk_conv_bwd_w<64u, 64u, 16u, 2u>), 64u, 64u)
MK_CONV(mk_conv_bwd_ww_f32, (mk_conv_bwd_w<32u, 128u, 16u, 4u>), 32u, 128u)
MK_CONV(mk_conv_bwd_ws_f32, (mk_conv_bwd_w<8u, 128u, 16u, 4u>), 8u, 128u)
// dst[i] = src[i] for n 32-bit words: p0 src, p2 dst, u7 = n (bit-exact,
// unlike a float copy of words holding two 16-bit values)
kernel void mk_copy(constant DevArgs_& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i < a.u[7]) ((device uint*)a.p[2])[i] = ((const device uint*)a.p[0])[i];
}
// dst[i] = src[i] for n words of host data that came with the dispatch
// (setBytes): an upload ordered with the queued work. p2 dst, u7 = n
kernel void mk_upload(constant DevArgs_& a [[buffer(0)]], constant uint* src [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < a.u[7]) ((device uint*)a.p[2])[i] = src[i];
}
#endif
