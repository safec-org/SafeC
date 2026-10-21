#pragma once
#include <std/collections/vec.h>
// SafeC Standard Library — NVIDIA GPU backend (CUDA).
//
// Three layers:
//
//  1. Device-resident buffers (CudaBuf) and queued device ops — the fast
//     path. Upload once, run any number of ops on the GPU, read back only
//     what you need. Matmuls go to cuBLAS (cuBLASLt for the fused
//     GEMM + ReLU); everything else runs this library's own CUDA kernels
//     (elementwise math, ReLU and its gradient, in-place SGD, sum
//     reduction, a fused MSE-loss gradient), compiled at first use with
//     NVRTC for the GPU that is present (sm_80, sm_90, ...) — no offline
//     build step and no per-architecture binaries.
//  2. Host-array convenience ops (cuda_*_f32) — upload, compute, read back
//     in one call. Simple, but each call pays two PCIe transfers; use (1)
//     inside loops.
//  3. Device-address handles (cuda_*_persistent, fp16/bf16) — the older
//     '&void'-handle API, kept for existing callers; it runs on the same
//     context and kernels as (1).
//
// Ops on layer (1) are queued in order on one CUDA stream and return as
// soon as they're enqueued; cuda_read() and cuda_sync() wait. A sequence
// of queued ops can be recorded once into a CudaGraph and replayed with a
// single launch, which removes per-kernel launch overhead from small,
// repeated workloads such as a training step.
//
// The backend uses the device's primary context (the one the CUDA runtime,
// and so PyTorch, also uses) on the GPU with the most SMs, and honours
// CUDA_VISIBLE_DEVICES. fp32 GEMMs run in full fp32 unless TF32 is turned
// on with cuda_set_tf32 (see below).
// Link with: -lcuda -lcublas -lcublasLt -lnvrtc (CUDA 12 or newer).
namespace std {

// ── Device and buffers ───────────────────────────────────────────────────────

// 1 if a CUDA GPU is usable (initializes the backend on first call).
int cuda_available();

// Name of the selected GPU ("NVIDIA H200"), or "" if none.
const &char cuda_device_name();

// A float32 buffer in GPU memory. 'dev' is the device address — never
// dereferenced on the host — and 0 for an invalid (failed or freed)
// buffer; 'len' is the element count every op checks against.
struct CudaBuf {
    unsigned long dev;
    unsigned long len;
};

struct CudaBuf cuda_alloc(unsigned long n);                       // uninitialized; dev == 0 on failure
struct CudaBuf cuda_upload(const &float host, unsigned long n);  // dev == 0 on failure
int  cuda_write(struct CudaBuf dst, const &float host, unsigned long n);  // 1 on success
int  cuda_read(struct CudaBuf src, &float host, unsigned long n);         // waits for queued work
// cuda_write returns once 'host' may be reused. Copies of 1 MiB and up run
// as 4 parallel lanes (3 helper threads started on first use plus the
// caller), each staging its slice through pinned buffers with the DMA of
// one chunk overlapping the copy of the next: ~2x the driver's own
// pageable-memory path, which is bounded by one core's memcpy rate.
void cuda_free(&CudaBuf b);                                       // b.dev := 0
int  cuda_sync();                                                 // wait for all queued work

// TF32 tensor cores for every later cuda_gemm / cuda_gemm_relu (sm_80 and
// newer): inputs are rounded to a 10-bit mantissa, accumulation stays fp32.
// Several times the matmul throughput, ~1e-3 relative error. Off by default.
// Returns 1 if the setting was applied.
int  cuda_set_tf32(int on);

// ── Queued device ops (1 = enqueued, 0 = invalid buffer/size or no GPU) ─────
// Matrices are row-major. 'ta'/'tb' = 1 means that operand is stored
// transposed (A as k x m, B as n x k).

// c[m,n] = op(a)[m,k] * op(b)[k,n]  (cuBLAS sgemm)
int cuda_gemm(struct CudaBuf a, int ta, struct CudaBuf b, int tb, struct CudaBuf c,
              unsigned long m, unsigned long n, unsigned long k);
// c = relu(op(a) * op(b)) in one kernel (cuBLASLt ReLU epilogue).
int cuda_gemm_relu(struct CudaBuf a, int ta, struct CudaBuf b, int tb, struct CudaBuf c,
                   unsigned long m, unsigned long n, unsigned long k);

int cuda_add(struct CudaBuf a, struct CudaBuf b, struct CudaBuf out);   // out = a + b
int cuda_sub(struct CudaBuf a, struct CudaBuf b, struct CudaBuf out);
int cuda_mul(struct CudaBuf a, struct CudaBuf b, struct CudaBuf out);
int cuda_div(struct CudaBuf a, struct CudaBuf b, struct CudaBuf out);
int cuda_pow(struct CudaBuf a, struct CudaBuf b, struct CudaBuf out);
int cuda_scale(struct CudaBuf a, float k, struct CudaBuf out);         // out = a * k
int cuda_relu(struct CudaBuf a, struct CudaBuf out);
int cuda_exp(struct CudaBuf a, struct CudaBuf out);
int cuda_log(struct CudaBuf a, struct CudaBuf out);
int cuda_sqrt(struct CudaBuf a, struct CudaBuf out);
int cuda_gelu(struct CudaBuf a, struct CudaBuf out);   // tanh approximation (gelu_pytorch_tanh)
// RMSNorm per row: y[r] = x[r] / sqrt(mean(x[r]^2) + eps) * w, over 'rows'
// rows of 'n' values; w has n values, or dev == 0 for no weight. y may be x.
int cuda_rmsnorm(struct CudaBuf x, struct CudaBuf w, struct CudaBuf y, unsigned long rows, unsigned long n, float eps);
// out = z > 0 ? g : 0. 'z' may be the ReLU input or its output (same sign test).
int cuda_relu_grad(struct CudaBuf z, struct CudaBuf g, struct CudaBuf out);
int cuda_sgd(struct CudaBuf w, struct CudaBuf g, float lr);            // w -= lr * g, in place
// Both SGD updates of a two-tensor model in one launch: w1 -= lr*g1, w2 -= lr*g2.
int cuda_sgd2(struct CudaBuf w1, struct CudaBuf g1, struct CudaBuf w2, struct CudaBuf g2, float lr);
int cuda_sum(struct CudaBuf a, struct CudaBuf out1);                   // out1[0] = sum(a)
// Fused MSE gradient: dy = 2 (y - t) and loss1[0] = sum((y - t)^2), in one pass.
int cuda_mse_grad(struct CudaBuf y, struct CudaBuf t, struct CudaBuf dy, struct CudaBuf loss1);

// dst[dstOff .. dstOff+n) = src[srcOff .. srcOff+n)  (device-to-device copy)
int cuda_copy(struct CudaBuf src, unsigned long srcOff, struct CudaBuf dst, unsigned long dstOff, unsigned long n);
// Strided 2D copy, one queued kernel: for r < rows, c < rowLen,
//   dst[dstOff + r*dstStride + c] = src[srcOff + r*srcStride + c].
// Rows must not overlap (rowLen <= both strides when rows > 1).
int cuda_copy_rows(struct CudaBuf src, unsigned long srcOff, unsigned long srcStride,
                   struct CudaBuf dst, unsigned long dstOff, unsigned long dstStride,
                   unsigned long rowLen, unsigned long rows);
// Nearest-neighbour 2x upsample of 'planes' row-major h x w planes into
// planes x 2h x 2w: out[p][y][x] = in[p][y/2][x/2] (cnn.h's upsample2x_nearest).
int cuda_upsample2x_nearest(struct CudaBuf in, struct CudaBuf out,
                            unsigned long planes, unsigned long h, unsigned long w);

// ── Graphs: record queued ops once, replay with one launch ───────────────────

struct CudaGraph {
    unsigned long exec;   // 0 = invalid
};

// Ops issued between begin and end are recorded instead of run. Don't call
// cuda_read/cuda_write/cuda_sync/cuda_alloc while recording.
int  cuda_capture_begin();
struct CudaGraph cuda_capture_end();          // exec == 0 on failure
int  cuda_graph_launch(struct CudaGraph g);   // queued like any other op
void cuda_graph_free(&CudaGraph g);

// ── fp16 / bf16 ──────────────────────────────────────────────────────────────
// 16-bit storage, fp32 arithmetic (the same API as gpu_rocm.h's RocmHalfBuf):
// half the memory and bandwidth of a CudaBuf. Every kernel widens on load
// and rounds to nearest even once per stored value, giving exactly the bits
// float16.h's f32_to_fp16 / f32_to_bf16 give on the CPU. GEMMs run on
// tensor cores through cuBLAS / cuBLASLt with fp32 accumulation. All ops are
// queued and can be recorded into a CudaGraph.

#define CUDA_FP16 1   // same values as tensor_half.h's TENSOR_FP16 / TENSOR_BF16
#define CUDA_BF16 2

// 'len' fp16 or bf16 values ('dtype') in GPU memory; dev == 0 if invalid.
struct CudaHalfBuf {
    unsigned long dev;
    unsigned long len;
    int           dtype;
};

struct CudaHalfBuf cuda_half_alloc(unsigned long n, int dtype);                   // uninitialized
struct CudaHalfBuf cuda_half_upload(const &float host, unsigned long n, int dtype); // rounded on the GPU
int  cuda_half_write(struct CudaHalfBuf dst, const &unsigned short bits, unsigned long n);  // raw bits
int  cuda_half_read(struct CudaHalfBuf src, &unsigned short bits, unsigned long n);         // waits
void cuda_half_free(&CudaHalfBuf b);

// Conversions over dst.len (to_half) / src.len (from_half) elements.
int cuda_to_half(struct CudaBuf src, struct CudaHalfBuf dst);     // round to dst.dtype
int cuda_from_half(struct CudaHalfBuf src, struct CudaBuf dst);   // exact

// c[m,n] = op(a)[m,k] * op(b)[k,n] (ta/tb as in cuda_gemm), a and b of the
// same dtype; c in that dtype (_relu: then max(0, .) in the epilogue), or
// fp32 with _f32.
int cuda_half_gemm(struct CudaHalfBuf a, int ta, struct CudaHalfBuf b, int tb, struct CudaHalfBuf c,
                   unsigned long m, unsigned long n, unsigned long k);
int cuda_half_gemm_relu(struct CudaHalfBuf a, int ta, struct CudaHalfBuf b, int tb, struct CudaHalfBuf c,
                        unsigned long m, unsigned long n, unsigned long k);
int cuda_half_gemm_f32(struct CudaHalfBuf a, int ta, struct CudaHalfBuf b, int tb, struct CudaBuf c,
                       unsigned long m, unsigned long n, unsigned long k);
// 'batch' products c_i = op(a_i) op(b_i) in one cuBLAS call, a_i starting
// at element i * strideA of a (likewise b, c); a stride of 0 reuses one
// matrix for every product (e.g. a weight shared by a batch). One dtype.
int cuda_half_gemm_strided(struct CudaHalfBuf a, int ta, unsigned long strideA, struct CudaHalfBuf b, int tb,
                           unsigned long strideB, struct CudaHalfBuf c, unsigned long strideC,
                           unsigned long m, unsigned long n, unsigned long k, unsigned long batch);
// Matrix-vector product for decoding one token at a time: y = w x with w an
// [n][k] fp16/bf16 matrix (row-major, e.g. a linear layer's weight) and x,
// y fp32 (x is not rounded to 16 bits). Memory-bound: on an H200 1.3-2.2x
// faster than a one-row cuda_half_gemm_f32 on 1536-6144-wide projections
// (e.g. 1536 x 2048: 3.2 vs 7.0 us), 4.47 TB/s on a 262144 x 1536 LM head.
// Fastest when k is a multiple of 8 (16-byte loads; scalar otherwise).
int cuda_half_gemv(struct CudaHalfBuf w, struct CudaBuf x, struct CudaBuf y, unsigned long n, unsigned long k);

// Training ops, as their CudaBuf versions above (loss stays fp32).
int cuda_half_relu(struct CudaHalfBuf a, struct CudaHalfBuf out);
int cuda_half_relu_grad(struct CudaHalfBuf z, struct CudaHalfBuf g, struct CudaHalfBuf out);
int cuda_half_mse_grad(struct CudaHalfBuf y, struct CudaHalfBuf t, struct CudaHalfBuf dy, struct CudaBuf loss1);
int cuda_half_sgd(struct CudaHalfBuf w, struct CudaHalfBuf g, float lr);   // w = round(w - lr * g)
int cuda_half_sgd2(struct CudaHalfBuf w1, struct CudaHalfBuf g1, struct CudaHalfBuf w2, struct CudaHalfBuf g2,
                   float lr);

// ── Mixed precision ──────────────────────────────────────────────────────────
// The usual layout: fp32 master weights and optimizer state, a 16-bit copy
// of the weights for the forward / backward passes, 16-bit or fp32
// gradients with the loss scale in them. One step:
//   cuda_half_gemm ... (forward / backward in fp16 or bf16)
//   zero the flag; cuda_half_unscale_many(grads16, grads32, 1 / scale, flag)
//   cuda_mp_adam(w32, m, v, g32, w16, ..., flag)       // skipped on the GPU if flag != 0
//   read the flag (4 bytes) to back the loss scale off or grow it
// so an overflowed step costs no host round trip before the update. All of
// these are queued and can be recorded into a CudaGraph.
//
// dst = src converted between fp16 and bf16 (src.len elements).
int cuda_half_convert(struct CudaHalfBuf src, struct CudaHalfBuf dst);
// Every src[i] -> dst[i] (lengths dst[i].len / src[i].len as in
// cuda_to_half / cuda_from_half), 32 buffers per launch: a model's
// parameters in one or a few launches instead of one per tensor.
int cuda_to_half_many(const &Vec<struct CudaBuf> src, const &Vec<struct CudaHalfBuf> dst);
int cuda_from_half_many(const &Vec<struct CudaHalfBuf> src, const &Vec<struct CudaBuf> dst);
// Gradient unscaling: dst[i] = src[i] * scale in fp32 (16-bit or fp32
// sources; the fp32 form may work in place, dst == src), and flag[0] = 1
// when any result is inf or NaN. The flag is only ever set: zero it
// (cuda_zero) before the step.
int cuda_half_unscale_many(const &Vec<struct CudaHalfBuf> src, const &Vec<struct CudaBuf> dst, float scale,
                           struct CudaBuf flag);
int cuda_unscale_many(const &Vec<struct CudaBuf> src, const &Vec<struct CudaBuf> dst, float scale,
                      struct CudaBuf flag);
int cuda_zero(struct CudaBuf b);                     // b = 0 (all of b.len)
// Optimizer steps on fp32 master weights w with gradient g, as nn.h's
// optimizer_step (OPT_SGD / OPT_MOMENTUM when vel is valid, OPT_ADAM /
// OPT_ADAMW when decoupled; 'step' is the 1-based step count for the bias
// correction). The updated weights are also written, rounded, to wHalf when
// it is valid. Skipped entirely when skip is valid and skip[0] != 0. Pass an
// invalid buffer (dev == 0) for any optional argument.
int cuda_mp_sgd(struct CudaBuf w, struct CudaBuf g, struct CudaBuf vel, struct CudaHalfBuf wHalf,
                float lr, float momentum, float weightDecay, struct CudaBuf skip);
int cuda_mp_adam(struct CudaBuf w, struct CudaBuf m, struct CudaBuf v, struct CudaBuf g, struct CudaHalfBuf wHalf,
                 float lr, float beta1, float beta2, float eps, float weightDecay, int decoupled,
                 unsigned long step, struct CudaBuf skip);
// cuda_mp_adam on every tensor of a model, 24 tensors per launch (w[i], m[i],
// v[i], g[i] of w[i].len values; wHalf empty for no 16-bit copies, else one
// per tensor, all of one dtype). The same arithmetic per value; one launch
// instead of one per tensor matters for models with many small tensors
// (biases, norms): GPT-2 small's 148 tensors take 1.11 instead of 1.84 ms
// on an H200 (fused torch.optim.AdamW: 1.37 ms, 1.61 with the bf16 copy).
int cuda_mp_adam_many(const &Vec<struct CudaBuf> w, const &Vec<struct CudaBuf> m, const &Vec<struct CudaBuf> v,
                      const &Vec<struct CudaBuf> g, const &Vec<struct CudaHalfBuf> wHalf, float lr, float beta1,
                      float beta2, float eps, float weightDecay, int decoupled, unsigned long step,
                      struct CudaBuf skip);
// 16-bit elementwise ops (fp32 arithmetic, one rounding per value; a, b and
// out of one dtype and a.len elements).
int cuda_half_add(struct CudaHalfBuf a, struct CudaHalfBuf b, struct CudaHalfBuf out);
int cuda_half_sub(struct CudaHalfBuf a, struct CudaHalfBuf b, struct CudaHalfBuf out);
int cuda_half_mul(struct CudaHalfBuf a, struct CudaHalfBuf b, struct CudaHalfBuf out);
int cuda_half_scale(struct CudaHalfBuf a, float k, struct CudaHalfBuf out);
int cuda_half_axpy(struct CudaHalfBuf a, float k, struct CudaHalfBuf b, struct CudaHalfBuf out);   // a + k b
// y[r][c][p] += bias[c] (rows x channels x plane, row-major), then ReLU when
// relu: a linear layer's bias (plane 1) or an NCHW convolution's.
int cuda_half_bias(struct CudaHalfBuf y, struct CudaHalfBuf bias, unsigned long rows, unsigned long channels,
                   unsigned long plane, int relu);

// 16-bit max / avg pooling on NCHW (x: [batch, channels, height, width]),
// kernel x kernel window, given stride, no padding: the same windows as
// gpu_cudnn.h's cuda_maxpool2d. fp32 accumulation, one rounding; max
// pooling copies the largest value's bits.
int cuda_half_maxpool2d(struct CudaHalfBuf x, struct CudaHalfBuf y, unsigned long batch, unsigned long channels,
                        unsigned long height, unsigned long width, unsigned long kernel, unsigned long stride);
int cuda_half_avgpool2d(struct CudaHalfBuf x, struct CudaHalfBuf y, unsigned long batch, unsigned long channels,
                        unsigned long height, unsigned long width, unsigned long kernel, unsigned long stride);

// ── Interop: for libraries layered on this backend (gpu_cudnn.h, dsp/fft_cuda.h)

// The CUstream every op above is queued on (a cudaStream_t to the runtime
// API, cuDNN and cuFFT); null if there is no GPU. Work another library
// queues on it is ordered with this backend's ops and recorded by graph
// capture.
?&void cuda_stream_handle();
// 1 if cuda_set_tf32(1) is in effect.
int cuda_tf32_enabled();
// Compute capability of the selected GPU as major * 10 + minor (80 on an
// A100, 90 on an H100/H200); 0 if there is no GPU.
int cuda_compute_capability();
// Streaming multiprocessors on the selected GPU (132 on an H100/H200 SXM); 0 if none.
int cuda_sm_count();
// Compiles CUDA C++ source with NVRTC for the selected GPU (-arch=sm_<cc>,
// -std=c++17) and loads it as a CUmodule, for libraries that bring their
// own kernels (gpu_cuda_attention.h). Null on failure, after printing the
// compile log. Look kernels up with cuModuleGetFunction and launch them on
// cuda_stream_handle().
?&void cuda_load_module(const &char source, const &char name);
// Byte copies between host memory and a raw device address (e.g. a
// CudaBuf's dev plus an offset, or memory from cuMemAlloc), through the
// same path as cuda_write / cuda_read (pinned lanes from 1 MiB up).
// cuda_copy_to_device returns once 'host' may be reused; cuda_copy_to_host
// waits for queued work first. 1 on success.
int cuda_copy_to_device(unsigned long dev, const &void host, unsigned long bytes);
int cuda_copy_to_host(&void host, unsigned long dev, unsigned long bytes);

// ── Host-array convenience ops (upload → compute → read back) ────────────────
// 1 on success, 0 on failure (no GPU, allocation failure).

int cuda_add_f32(const &float a, const &float b, &float out, unsigned long n);
int cuda_sub_f32(const &float a, const &float b, &float out, unsigned long n);
int cuda_mul_f32(const &float a, const &float b, &float out, unsigned long n);
int cuda_div_f32(const &float a, const &float b, &float out, unsigned long n);
int cuda_pow_f32(const &float a, const &float b, &float out, unsigned long n);
int cuda_relu_f32(const &float a, &float out, unsigned long n);
int cuda_log_f32(const &float a, &float out, unsigned long n);
int cuda_exp_f32(const &float a, &float out, unsigned long n);
int cuda_sqrt_f32(const &float a, &float out, unsigned long n);
int cuda_scale_f32(const &float a, float k, &float out, unsigned long n);
int cuda_sum_f32(const &float a, &float out, unsigned long n);

// out[M,N] = a[M,K] * b[K,N] with this library's own tiled CUDA kernel
// (64x64 tiles in shared memory, 4x4 per thread). Kept as a dependency-free
// reference; cuda_matmul_f32_blas is faster at every size.
int cuda_matmul_f32(const &float a, const &float b, &float out,
                    unsigned long M, unsigned long K, unsigned long N);
// Same product through cuBLAS.
int cuda_matmul_f32_blas(const &float a, const &float b, &float out,
                         unsigned long M, unsigned long K, unsigned long N);

// ── Raw device pointers (older API) ──────────────────────────────────────────
// Same context and kernels as the CudaBuf layer; a '&void' here is a device
// address. Prefer CudaBuf in new code: it carries its length, so every op
// can check sizes.

// 1 if the backend initialized (same as cuda_available()).
int cuda_persistent_available();
// Uploads 'bytes' of host memory to a new device buffer; null on failure.
?&void cuda_upload_persistent(const &float data, unsigned long bytes);
// Frees a buffer from any cuda_upload_*_persistent function.
void cuda_release_persistent(&void devPtr);
// out[M,N] (host) = devA[M,K] * devB[K,N], both already on the device.
int cuda_matmul_f32_persistent(&void devA, &void devB, &float out,
                                unsigned long M, unsigned long K, unsigned long N);
// devW[i] -= lr * devGrad[i], in place on the device; nothing is read back.
int cuda_sgd_update_f32(&void devW, &void devGrad, float lr, unsigned long n);

// fp16 / bf16 storage (2 bytes per element), fp32 arithmetic: each kernel
// widens on load and rounds to nearest-even on store. The upload functions
// convert from float32 on the host; 'out' receives raw fp16/bf16 bits (see
// float16.h's fp16_to_f32_bulk / bf16_to_f32_bulk).
?&void cuda_upload_f16_persistent(const &float data, unsigned long n);
?&void cuda_upload_bf16_persistent(const &float data, unsigned long n);
int cuda_matmul_f16_persistent(&void devA, &void devB, &unsigned short out,
                                unsigned long M, unsigned long K, unsigned long N);
int cuda_matmul_bf16_persistent(&void devA, &void devB, &unsigned short out,
                                 unsigned long M, unsigned long K, unsigned long N);
int cuda_sgd_update_f16(&void devW, &void devGrad, float lr, unsigned long n);
int cuda_sgd_update_bf16(&void devW, &void devGrad, float lr, unsigned long n);

} // namespace std
