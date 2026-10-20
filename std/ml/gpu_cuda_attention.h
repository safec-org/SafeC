#pragma once
// SafeC Standard Library — attention on NVIDIA GPUs (CUDA): flash attention
// and linear attention as this library's own CUDA kernels.
//
// Batched multi-head versions of attention.h's flash_attention_forward and
// linear_attention_forward, on gpu_cuda.h's device buffers and stream:
//
//  - Flash attention: softmax(Q K^T * scale) V without materializing the
//    score matrix. fp16/bf16 go to cuDNN's fused attention (the kernels
//    PyTorch's scaled_dot_product_attention uses on Hopper) when cuDNN 9 is
//    installed; otherwise, and for anything cuDNN rejects, this library's
//    FlashAttention-2 kernel runs them on tensor cores (mma.sync, fp32
//    accumulation) on sm_80+ for head dims up to 128. fp32, head dim 256 and
//    older GPUs use a CUDA-core kernel with the same algorithm.
//  - Linear attention: Katharopoulos et al.'s kernel-trick attention with
//    phi(x) = elu(x) + 1, in O((seqLen + kvLen) D^2) instead of
//    O(seqLen kvLen D). Non-causal runs phi(K)^T V once per kv head; causal
//    walks the sequence in 32-row chunks with a running state, as a linear
//    RNN would.
//
// Two calling forms:
//  - cuda_flash_attention / cuda_linear_attention (and their host-array
//    _f32 versions) take the same arguments as gpu_rocm.h's
//    rocm_flash_attention / rocm_linear_attention, so code moves between
//    AMD and NVIDIA GPUs unchanged: fp32, q and out [heads][seqLen][headDim],
//    k and v [heads][kvLen][headDim] (fold a batch into heads).
//  - The _ex and cuda_half_ versions take a CudaAttnShape: a batch, grouped-
//    query K/V (kvHeads dividing heads; query head h reads kv head
//    h / (heads / kvHeads)), and a sliding window. Q and out are then
//    [batch][heads][seqLen][headDim], K and V [batch][kvHeads][kvLen][headDim].
// All buffers are row-major and contiguous.
//
// Masking: query i sits at position pos = i + (kvLen - seqLen) -- the last
// seqLen keys are the queries' own tokens, any before them a KV cache (as
// gpu_mps.h's mps_flash_attn_bf16 and the ROCm ops count it). causal hides
// keys j > pos; a sliding window w > 0 (flash only) also hides keys
// j <= pos - w, i.e. the query sees its last w positions.
//
// Head dims: 32, 64, 96, 128 or 256. All math is fp32; the half versions
// read and write fp16 or bf16 (all four buffers of one dtype). Ops are
// queued on gpu_cuda's stream like every other op there and can be
// recorded into a CudaGraph. Each (kernel, dtype, head dim) variant is
// compiled with NVRTC on its first call (~0.3 s), so make one call of each
// shape before capturing a graph; non-causal linear attention also sizes
// its workspace on that call. Forward only, like attention.h. 1 = enqueued
// (the _f32 forms: done), 0 = bad shape/buffer or no GPU.
// Link with gpu_cuda.h's libraries (-lcuda -lcublas -lcublasLt -lnvrtc).
#include <std/ml/gpu_cuda.h>

namespace std {

struct CudaAttnShape {
    unsigned long batch;
    unsigned long heads;      // query heads
    unsigned long kvHeads;    // key/value heads; divides heads
    unsigned long seqLen;     // query length
    unsigned long kvLen;      // key/value length
    unsigned long headDim;    // 32, 64, 96, 128 or 256
    unsigned long window;     // flash attention sliding window; 0 = none
};

// 1 if 'headDim' is supported by the kernels below.
int cuda_attention_head_dim_ok(unsigned long headDim);

// fp16/bf16 flash attention uses cuDNN's fused attention engine when
// libcudnn.so.9 can be loaded (dlopen; no link dependency) and the GPU is
// sm_80 or newer, and this library's own kernels otherwise. on = 0 forces
// the own kernels. Returns 1 if cuDNN is in use after the call.
int cuda_attention_use_cudnn(int on);

// out = softmax(q k^T * scale [+ mask]) v. 'scale' <= 0 means
// 1 / sqrt(headDim). A query with no visible key gets zeros.
int cuda_flash_attention(struct CudaBuf q, struct CudaBuf k, struct CudaBuf v, struct CudaBuf out,
                         unsigned long heads, unsigned long seqLen, unsigned long kvLen, unsigned long headDim,
                         float scale, int causal);
int cuda_flash_attention_ex(struct CudaBuf q, struct CudaBuf k, struct CudaBuf v, struct CudaBuf out,
                            struct CudaAttnShape s, float scale, int causal);
int cuda_half_flash_attention(struct CudaHalfBuf q, struct CudaHalfBuf k, struct CudaHalfBuf v,
                              struct CudaHalfBuf out, struct CudaAttnShape s, float scale, int causal);

// out_i = phi(q_i) S / phi(q_i) z with S = sum_j phi(k_j) v_j^T and
// z = sum_j phi(k_j) over all keys (causal = 0) or over j <= pos (causal = 1,
// which needs kvLen >= seqLen). phi(x) = elu(x) + 1; no scale, since the
// division cancels it. The same result as attention.h's
// linear_attention_forward per head. s.window must be 0.
int cuda_linear_attention(struct CudaBuf q, struct CudaBuf k, struct CudaBuf v, struct CudaBuf out,
                          unsigned long heads, unsigned long seqLen, unsigned long kvLen, unsigned long headDim,
                          int causal);
int cuda_linear_attention_ex(struct CudaBuf q, struct CudaBuf k, struct CudaBuf v, struct CudaBuf out,
                             struct CudaAttnShape s, int causal);
int cuda_half_linear_attention(struct CudaHalfBuf q, struct CudaHalfBuf k, struct CudaHalfBuf v,
                               struct CudaHalfBuf out, struct CudaAttnShape s, int causal);

// The two fp32 ops on host arrays (upload, compute, read back), with the
// rocm_*_f32 arguments.
int cuda_flash_attention_f32(const &float q, const &float k, const &float v, &float out,
                             unsigned long heads, unsigned long seqLen, unsigned long kvLen,
                             unsigned long headDim, float scale, int causal);
int cuda_linear_attention_f32(const &float q, const &float k, const &float v, &float out,
                              unsigned long heads, unsigned long seqLen, unsigned long kvLen,
                              unsigned long headDim, int causal);

} // namespace std
