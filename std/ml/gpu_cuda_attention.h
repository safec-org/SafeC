#pragma once
// SafeC Standard Library — attention on NVIDIA GPUs (CUDA): flash attention
// and linear attention as this library's own CUDA kernels.
//
// Batched multi-head versions of attention.h's flash_attention_forward and
// linear_attention_forward, on gpu_cuda.h's device buffers and stream:
//
//  - cuda_flash_attention / cuda_half_flash_attention: softmax(Q K^T *
//    scale) V without materializing the score matrix (FlashAttention-2:
//    K/V tiles in shared memory, online softmax in registers). fp16/bf16
//    run on tensor cores (mma.sync, fp32 accumulation) on sm_80 and newer
//    for head dims up to 128; fp32, head dim 256 and older GPUs use a
//    CUDA-core kernel with the same algorithm.
//  - cuda_linear_attention / cuda_half_linear_attention: Katharopoulos et
//    al.'s kernel-trick attention with phi(x) = elu(x) + 1, in
//    O((seqLen + kvLen) D^2) instead of O(seqLen kvLen D). Non-causal runs
//    phi(K)^T V once per kv head; causal walks the sequence in 32-row
//    chunks with a running state, as a linear RNN would.
//
// Layout: Q and out are [batch, heads, seqLen, headDim], K and V are
// [batch, kvHeads, kvLen, headDim], all row-major and contiguous. kvHeads
// must divide heads; query head h reads kv head h / (heads / kvHeads)
// (grouped-query attention; kvHeads == heads is plain multi-head). The
// causal mask is aligned to the bottom right: query i sees keys
// j <= i + (kvLen - seqLen), which is the usual mask for a KV cache that
// holds kvLen - seqLen earlier tokens.
//
// Head dims: 32, 64, 96, 128 or 256. All math is fp32; the half versions
// read and write fp16 or bf16 (all four buffers of one dtype). Ops are
// queued on gpu_cuda's stream like every other op there and can be
// recorded into a CudaGraph. Each (kernel, dtype, head dim) variant is
// compiled with NVRTC on its first call (~0.3 s), so make one call of each
// shape before capturing a graph; cuda_linear_attention's non-causal path
// also sizes its workspace on that call. Forward only, like attention.h.
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
};

// 1 if 'headDim' is supported by the kernels below.
int cuda_attention_head_dim_ok(unsigned long headDim);

// out = softmax(Q K^T * scale [+ causal mask]) V. 'scale' <= 0 means
// 1 / sqrt(headDim). Queries with no visible key (causal, kvLen < seqLen)
// get zeros. 1 = enqueued, 0 = bad shape/buffer or no GPU.
int cuda_flash_attention(struct CudaBuf q, struct CudaBuf k, struct CudaBuf v, struct CudaBuf out,
                         struct CudaAttnShape s, float scale, int causal);
int cuda_half_flash_attention(struct CudaHalfBuf q, struct CudaHalfBuf k, struct CudaHalfBuf v,
                              struct CudaHalfBuf out, struct CudaAttnShape s, float scale, int causal);

// out_i = phi(q_i) . S / phi(q_i) . z with S = sum_j phi(k_j) v_j^T and
// z = sum_j phi(k_j) over all keys (causal = 0) or over j <= i + (kvLen -
// seqLen) (causal = 1, which needs kvLen >= seqLen). phi(x) = elu(x) + 1.
// The same result as attention.h's linear_attention_forward per head.
int cuda_linear_attention(struct CudaBuf q, struct CudaBuf k, struct CudaBuf v, struct CudaBuf out,
                          struct CudaAttnShape s, int causal);
int cuda_half_linear_attention(struct CudaHalfBuf q, struct CudaHalfBuf k, struct CudaHalfBuf v,
                               struct CudaHalfBuf out, struct CudaAttnShape s, int causal);

} // namespace std
