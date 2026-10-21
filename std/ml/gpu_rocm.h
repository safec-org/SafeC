#pragma once
// SafeC Standard Library — AMD GPU backend (ROCm / HIP).
//
// Two layers:
//
//  1. Device-resident buffers (RocmBuf) and queued device ops — the fast
//     path. Upload once, run any number of ops on the GPU, read back only
//     what you need. Matmuls go to rocBLAS; everything else runs custom
//     HIP kernels (elementwise math, ReLU and its gradient, in-place SGD,
//     sum reduction, and a fused MSE-loss gradient) that this library
//     compiles at first use with hiprtc for whatever GPU is present — no
//     offline build step and no per-architecture binaries.
//  2. Host-array convenience ops (rocm_*_f32) — upload, compute, read back
//     in one call. Simple, but each call pays two PCIe transfers; use (1)
//     inside loops.
//
// Ops on layer (1) are queued in order on one HIP stream and return as
// soon as they're enqueued; rocm_read() and rocm_sync() wait. A sequence
// of queued ops can be recorded once into a RocmGraph and replayed with a
// single launch, which removes per-kernel launch overhead from small,
// repeated workloads such as a training step.
//
// The backend picks the GPU with the most compute units (discrete over
// integrated), so on a machine with both an Instinct/Radeon card and an
// APU it uses the card. Link with: -lamdhip64 -lhiprtc -lrocblas
// (from /opt/rocm/lib).
namespace std {

// ── Device and buffers ───────────────────────────────────────────────────────

// 1 if a HIP GPU is usable (initializes the backend on first call).
int rocm_available();

// Name of the selected GPU ("AMD Instinct MI100"), or "" if none.
const &char rocm_device_name();

// A float32 buffer in GPU memory. 'dev' is the device address — never
// dereferenced on the host — and 0 for an invalid (failed or freed)
// buffer; 'len' is the element count every op checks against.
struct RocmBuf {
    unsigned long dev;
    unsigned long len;
};

struct RocmBuf rocm_alloc(unsigned long n);                       // uninitialized; dev == 0 on failure
struct RocmBuf rocm_upload(const &float host, unsigned long n);  // dev == 0 on failure
int  rocm_write(struct RocmBuf dst, const &float host, unsigned long n);  // 1 on success
int  rocm_read(struct RocmBuf src, &float host, unsigned long n);         // waits for queued work
void rocm_free(&RocmBuf b);                                       // b.dev := 0
int  rocm_sync();                                                 // wait for all queued work

// ── Queued device ops (1 = enqueued, 0 = invalid buffer/size or no GPU) ─────
// Matrices are row-major. 'ta'/'tb' = 1 means that operand is stored
// transposed (A as k x m, B as n x k).

// c[m,n] = op(a)[m,k] * op(b)[k,n]  (rocBLAS sgemm)
int rocm_gemm(struct RocmBuf a, int ta, struct RocmBuf b, int tb, struct RocmBuf c,
              unsigned long m, unsigned long n, unsigned long k);

int rocm_add(struct RocmBuf a, struct RocmBuf b, struct RocmBuf out);   // out = a + b
int rocm_sub(struct RocmBuf a, struct RocmBuf b, struct RocmBuf out);
int rocm_mul(struct RocmBuf a, struct RocmBuf b, struct RocmBuf out);
int rocm_div(struct RocmBuf a, struct RocmBuf b, struct RocmBuf out);
int rocm_pow(struct RocmBuf a, struct RocmBuf b, struct RocmBuf out);
int rocm_scale(struct RocmBuf a, float k, struct RocmBuf out);         // out = a * k
int rocm_relu(struct RocmBuf a, struct RocmBuf out);
int rocm_exp(struct RocmBuf a, struct RocmBuf out);
int rocm_log(struct RocmBuf a, struct RocmBuf out);
int rocm_sqrt(struct RocmBuf a, struct RocmBuf out);
int rocm_gelu(struct RocmBuf a, struct RocmBuf out);   // tanh approximation (gelu_pytorch_tanh)
// RMSNorm per row: y[r] = x[r] / sqrt(mean(x[r]^2) + eps) * w, over 'rows'
// rows of 'n' values; w has n values, or dev == 0 for no weight. y may be x.
int rocm_rmsnorm(struct RocmBuf x, struct RocmBuf w, struct RocmBuf y, unsigned long rows, unsigned long n, float eps);
int rocm_relu_grad(struct RocmBuf z, struct RocmBuf g, struct RocmBuf out);   // out = z > 0 ? g : 0
int rocm_sgd(struct RocmBuf w, struct RocmBuf g, float lr);            // w -= lr * g, in place
int rocm_sum(struct RocmBuf a, struct RocmBuf out1);                   // out1[0] = sum(a)
// Fused MSE gradient: dy = 2 (y - t) and loss1[0] = sum((y - t)^2), in one pass.
int rocm_mse_grad(struct RocmBuf y, struct RocmBuf t, struct RocmBuf dy, struct RocmBuf loss1);

// dst[dstOff .. dstOff+n) = src[srcOff .. srcOff+n)  (device-to-device copy)
int rocm_copy(struct RocmBuf src, unsigned long srcOff, struct RocmBuf dst, unsigned long dstOff, unsigned long n);
// out[b] = (a[b], c[b]) for 'batch' rows of na and nc elements, one launch
// (the channel concat of NCHW tensors: na = aChannels * height * width).
int rocm_concat(struct RocmBuf a, unsigned long na, struct RocmBuf c, unsigned long nc,
                struct RocmBuf out, unsigned long batch);
// Nearest-neighbour 2x upsample of 'planes' row-major h x w planes into
// planes x 2h x 2w: out[p][y][x] = in[p][y/2][x/2] (cnn.h's upsample2x_nearest).
int rocm_upsample2x_nearest(struct RocmBuf in, struct RocmBuf out,
                            unsigned long planes, unsigned long h, unsigned long w);
// out[b] = (upsample2x_nearest(x[b]), skip[b]) along channels, in one pass:
// x is [batch, xChannels, h, w], skip [batch, skipChannels, 2h, 2w], out
// [batch, xChannels + skipChannels, 2h, 2w] (a U-Net decoder step).
int rocm_upsample2x_concat(struct RocmBuf x, unsigned long xChannels, struct RocmBuf skip,
                           unsigned long skipChannels, struct RocmBuf out,
                           unsigned long batch, unsigned long h, unsigned long w);
// NCHW per-channel bias, in place: y[b][c][i] += bias[c] for every one of the
// 'plane' (= height*width) elements, then max(0, .) if relu != 0. The
// epilogue gpu_miopen.h runs after a convolution.
int rocm_bias_channels(struct RocmBuf y, struct RocmBuf bias, unsigned long batch, unsigned long channels,
                       unsigned long plane, int relu);

// ── fp16 / bf16 ──────────────────────────────────────────────────────────────
// 16-bit storage, fp32 arithmetic: half the memory and bandwidth of a
// RocmBuf. Conversions round to nearest even and give exactly the bits
// float16.h's f32_to_fp16 / f32_to_bf16 give on the CPU. Matmuls run on
// rocBLAS (gemm_ex) with fp32 accumulation and round once at the end.

#define ROCM_FP16 1   // same values as tensor_half.h's TENSOR_FP16 / TENSOR_BF16
#define ROCM_BF16 2

// A buffer of 'len' fp16 or bf16 values ('dtype') in GPU memory; dev == 0
// for an invalid (failed or freed) buffer.
struct RocmHalfBuf {
    unsigned long dev;
    unsigned long len;
    int           dtype;
};

struct RocmHalfBuf rocm_half_alloc(unsigned long n, int dtype);                   // uninitialized
struct RocmHalfBuf rocm_half_upload(const &float host, unsigned long n, int dtype); // rounded on the GPU
int  rocm_half_write(struct RocmHalfBuf dst, const &unsigned short bits, unsigned long n);  // raw bits
int  rocm_half_read(struct RocmHalfBuf src, &unsigned short bits, unsigned long n);         // waits
void rocm_half_free(&RocmHalfBuf b);

// Queued conversions over dst.len elements.
int rocm_to_half(struct RocmBuf src, struct RocmHalfBuf dst);     // round to dst.dtype
int rocm_from_half(struct RocmHalfBuf src, struct RocmBuf dst);   // exact

// c[m,n] = op(a)[m,k] * op(b)[k,n], 16-bit a and b of the same dtype, fp32
// accumulation; c in the same 16-bit dtype, or fp32 with _f32.
int rocm_half_gemm(struct RocmHalfBuf a, int ta, struct RocmHalfBuf b, int tb, struct RocmHalfBuf c,
                   unsigned long m, unsigned long n, unsigned long k);
int rocm_half_gemm_f32(struct RocmHalfBuf a, int ta, struct RocmHalfBuf b, int tb, struct RocmBuf c,
                       unsigned long m, unsigned long n, unsigned long k);
// w = round(w - lr * g), computed in fp32, in place.
int rocm_half_sgd(struct RocmHalfBuf w, struct RocmHalfBuf g, float lr);

// c = relu(op(a) op(b)), all in a's dtype (the GEMM, then ReLU in place).
int rocm_half_gemm_relu(struct RocmHalfBuf a, int ta, struct RocmHalfBuf b, int tb, struct RocmHalfBuf c,
                        unsigned long m, unsigned long n, unsigned long k);
// Matrix-vector product for decoding one token at a time: y = w x with w an
// [n][k] fp16/bf16 matrix (row-major, e.g. a linear layer's weight) and x,
// y fp32 (x is not rounded to 16 bits). Memory-bound: 0.8-1.07 TB/s of
// weights on an MI100, 1.04-1.8x faster than a one-row rocm_half_gemm_f32
// (most on short outputs, e.g. 1536 x 2048: 7.8 vs 13.7 us). Fastest when k
// is a multiple of 8.
int rocm_half_gemv(struct RocmHalfBuf w, struct RocmBuf x, struct RocmBuf y, unsigned long n, unsigned long k);
// Training ops, as their RocmBuf versions (loss stays fp32).
int rocm_half_relu(struct RocmHalfBuf a, struct RocmHalfBuf out);
int rocm_half_relu_grad(struct RocmHalfBuf z, struct RocmHalfBuf g, struct RocmHalfBuf out);
int rocm_half_mse_grad(struct RocmHalfBuf y, struct RocmHalfBuf t, struct RocmHalfBuf dy, struct RocmBuf loss1);
// rocm_half_sgd on two weight/gradient pairs in one launch.
int rocm_half_sgd2(struct RocmHalfBuf w1, struct RocmHalfBuf g1, struct RocmHalfBuf w2, struct RocmHalfBuf g2,
                   float lr);

// ── Graphs: record queued ops once, replay with one launch ───────────────────

struct RocmGraph {
    unsigned long exec;   // 0 = invalid
};

// Ops issued between begin and end are recorded instead of run. Don't call
// rocm_read/rocm_write/rocm_sync while recording.
int  rocm_capture_begin();
struct RocmGraph rocm_capture_end();          // exec == 0 on failure
int  rocm_graph_launch(struct RocmGraph g);   // queued like any other op
void rocm_graph_free(&RocmGraph g);

// ── Interop: for libraries layered on this backend (gpu_miopen.h, dsp/fft_rocm.h)

// The hipStream_t every op above is queued on (MIOpen and hipFFT queue their
// work on it too); null if there is no GPU. Work another library queues on
// it is ordered with this backend's ops and recorded by graph capture.
?&void rocm_stream_handle();

// ── Attention ────────────────────────────────────────────────────────────────
// fp32, batched over heads: q and out are [heads][seqLen][headDim], k and v
// [heads][kvLen][headDim], all row-major; headDim <= 128. 1 on success, 0 on
// a bad shape or failure. Forward only, queued on the stream like every op.
//
// On GPUs with matrix cores (CDNA: MI100 / MI200 / MI300) both ops run on
// them (fp32 MFMA); elsewhere on CUDA-core kernels with the same results up
// to rounding.
//
// Flash attention: softmax(q k^T * scale) v without forming the score
// matrix -- K/V tiles in LDS and an online softmax per query row (Dao et
// al.), the same result as std/ml/attention.h's attention_forward up to
// rounding. scale <= 0 means 1/sqrt(headDim). causal: query i sees keys
// j <= i + (kvLen - seqLen) (so kvLen >= seqLen; with a KV cache the new
// queries are the last seqLen positions). A query with no visible key gets
// zeros.
int rocm_flash_attention(struct RocmBuf q, struct RocmBuf k, struct RocmBuf v, struct RocmBuf out,
                         unsigned long heads, unsigned long seqLen, unsigned long kvLen, unsigned long headDim,
                         float scale, int causal);
// Linear attention with phi(x) = elu(x) + 1, as attention.h's
// linear_attention_forward: out_i = phi(q_i) S / phi(q_i) z with S, z the
// sums of phi(k_j)^T v_j and phi(k_j). Chunked (128 rows on matrix cores,
// 32 otherwise): per-chunk states, a prefix over chunks, then each chunk's
// rows -- so causal (j <= i, which needs kvLen == seqLen) runs in parallel
// too. Cost O(n d^2) instead of
// O(n^2 d). The chunk states live in a scratch buffer this library keeps
// and grows; during graph capture a call that would need to grow it fails.
int rocm_linear_attention(struct RocmBuf q, struct RocmBuf k, struct RocmBuf v, struct RocmBuf out,
                          unsigned long heads, unsigned long seqLen, unsigned long kvLen, unsigned long headDim,
                          int causal);
// on = 0 makes both ops use the CUDA-core kernels even on matrix-core GPUs
// (for comparison and testing); on = 1 restores the default. Returns 1 if the
// matrix-core kernels are in use after the call.
int rocm_attention_use_mfma(int on);

// ── Host-array convenience ops (upload → compute → read back) ────────────────
// 1 on success, 0 on failure (no GPU, allocation failure).

int rocm_add_f32(const &float a, const &float b, &float out, unsigned long n);
int rocm_sub_f32(const &float a, const &float b, &float out, unsigned long n);
int rocm_mul_f32(const &float a, const &float b, &float out, unsigned long n);
int rocm_div_f32(const &float a, const &float b, &float out, unsigned long n);
int rocm_pow_f32(const &float a, const &float b, &float out, unsigned long n);
int rocm_relu_f32(const &float a, &float out, unsigned long n);
int rocm_log_f32(const &float a, &float out, unsigned long n);
int rocm_exp_f32(const &float a, &float out, unsigned long n);
int rocm_sqrt_f32(const &float a, &float out, unsigned long n);
int rocm_scale_f32(const &float a, float k, &float out, unsigned long n);
int rocm_sum_f32(const &float a, &float out1, unsigned long n);
// The attention ops above on host arrays (same layouts).
int rocm_flash_attention_f32(const &float q, const &float k, const &float v, &float out,
                             unsigned long heads, unsigned long seqLen, unsigned long kvLen,
                             unsigned long headDim, float scale, int causal);
int rocm_linear_attention_f32(const &float q, const &float k, const &float v, &float out,
                              unsigned long heads, unsigned long seqLen, unsigned long kvLen,
                              unsigned long headDim, int causal);

// out[M,N] = a[M,K] * b[K,N] with this library's own tiled HIP kernel
// (64x64 tiles in LDS, 4x4 per thread). Kept as a dependency-free
// reference; rocm_matmul_f32_blas is several times faster at every size.
int rocm_matmul_f32(const &float a, const &float b, &float out,
                    unsigned long M, unsigned long K, unsigned long N);
// Same product through rocBLAS.
int rocm_matmul_f32_blas(const &float a, const &float b, &float out,
                         unsigned long M, unsigned long K, unsigned long N);

} // namespace std
