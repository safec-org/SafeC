#pragma once
// SafeC Standard Library — Attention (scaled dot-product + multi-head).
//
// Built on std/ml/tensor.h's Tensor. Every op here records its backward
// edges like tensor_matmul does, so attention layers train (trainer.h), and
// runs where its operands live (device.h): on a GPU the multi-head ops run
// as one fused kernel launch for every head -- QK^T, the softmax and the
// product with V, never forming the score matrix -- with a fused backward,
// and otherwise compose the device ops (transpose, matmul, masked softmax,
// column slices). Q/K/V are plain 2D tensors, [seqLen, dim] -- no batch
// dimension (call these once per sequence in a batch) and no positional
// encoding baked in (add it to Q/K yourself before calling, e.g. via
// tensor_add with a positional tensor).
//
// Masks: query row i sits at position pos = i + (kvLen - seqLen) (the last
// seqLen keys are the queries' own tokens, any before them a KV cache).
// causal hides keys j > pos; a sliding window w > 0 hides keys j <= pos - w
// (the query sees its last w positions). A query with no visible key gets
// zeros. These are gpu_rocm.h / gpu_cuda_attention.h / gpu_mps.h's masks.
//
// Scope note: standard scaled-dot-product/MHA, gated attention, gated
// MHA, linear attention, and flash attention are implemented and
// verified in this file. Multi-head Latent Attention (MLA)/EG-MLA,
// Shifted-Window (Swin) Attention, and Gated DeltaNet live in
// attention_advanced.h/.sc. std/ml/ROADMAP.md tracks anything still
// outstanding.
//
// Inference kernels: gpu_cuda_attention.h (NVIDIA) and gpu_mps.h's
// mps_flash_attention / mps_linear_attention (Apple) -- batched multi-head
// flash and linear attention, GQA, causal and sliding-window masks,
// fp32/fp16/bf16, with the same argument forms -- and gpu_rocm.h's
// rocm_flash_attention / rocm_linear_attention (AMD, fp32), on their own
// [heads][seqLen][headDim] buffers, forward only.
#include <std/ml/tensor.h>
#include <std/ml/nn.h>
#include <std/collections/vec.h>

namespace std {

// Row-wise softmax: for a [rows, cols] tensor, each output row sums to 1.
// Numerically stable (subtracts each row's max before exponentiating).
&FloatTensor FloatTensor::softmax_rows() const;
// The same over the keys a mask leaves visible (row r is query r of a
// [seqLen, kvLen] score matrix; see the masks above); hidden entries are 0.
&FloatTensor FloatTensor::softmax_rows_mask(int causal, unsigned long window) const;

// out[r][c] = in[r][c] transposed -> [cols, rows].
&FloatTensor FloatTensor::transpose() const;

// Copies out the column range [startCol, startCol+numCols) of every row.
// Records a backward edge (gradients go back into those columns) when the
// input requires grad.
&FloatTensor FloatTensor::slice_cols(unsigned long startCol, unsigned long numCols) const;

// Writes 'src' into 'dst's column range [startCol, startCol+numCols)
// (numCols == src's own column count), row by row, in place -- not part of
// the graph (tensor_concat_cols is the differentiable way to assemble
// columns).
void tensor_set_cols(&FloatTensor dst, unsigned long startCol, const &FloatTensor src);

// Single-head scaled dot-product attention: softmax(Q @ K^T / sqrt(headDim)) @ V.
// Q: [seqLen, headDim], K/V: [kvLen, headDim] -> result: [seqLen, headDim].
&FloatTensor attention_forward(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V);

// Multi-head attention: splits Q/K/V's dModel columns into 'numHeads'
// equal headDim-wide slices (dModel must be divisible by numHeads), runs
// attention_forward() per head, and concatenates the per-head outputs
// back into one [seqLen, dModel] result — the standard MHA shape from
// "Attention Is All You Need", without a separate output-projection
// matrix (apply your own W_O via tensor_matmul on the result if needed).
&FloatTensor mha_forward(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V, unsigned long numHeads);

// The two with a mask (causal 0/1, window 0 for none; see above).
&FloatTensor attention_forward_mask(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V, int causal,
                                    unsigned long window);
&FloatTensor mha_forward_mask(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V, unsigned long numHeads,
                              int causal, unsigned long window);

// Whether the multi-head ops on a GPU run as the fused kernels (1, the
// default) or as composed device ops (0: transpose, matmul, masked softmax
// per head -- for comparison and testing). Head dims over 128 always
// compose. Returns the previous setting.
int attention_set_fused(int on);

// ── Gated attention ──────────────────────────────────────────────────────────
// Output gating: sigmoid(gateLogits) elementwise-multiplied into the
// attention result before it's returned (the pattern used e.g. to
// modulate an attention block's contribution to the residual stream).
// 'gateLogits' must match the output shape: [seqLen, headDim] for
// gated_attention_forward, [seqLen, dModel] for gated_mha_forward.
&FloatTensor gated_attention_forward(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V,
                                 const &FloatTensor gateLogits);
&FloatTensor gated_mha_forward(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V,
                           unsigned long numHeads, const &FloatTensor gateLogits);

// ── Linear attention ─────────────────────────────────────────────────────────
// Kernel-trick reformulation (Katharopoulos et al., "Transformers are
// RNNs"): with feature map phi(x) = elu(x)+1 (elementwise, always
// positive), softmax attention's O(seqLen*kvLen*headDim) cost becomes
// O((seqLen+kvLen)*headDim^2) by computing phi(K)^T @ V once
// ([headDim,headDim]) instead of the full [seqLen,kvLen] score matrix:
//   out_i = (phi(Q_i) . sum_j phi(K_j) (x) V_j) / (phi(Q_i) . sum_j phi(K_j))
// Q: [seqLen,headDim], K/V: [kvLen,headDim] -> result: [seqLen,headDim].
&FloatTensor linear_attention_forward(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V);

// ── Flash attention ──────────────────────────────────────────────────────────
// Tiled, online-softmax attention (Dao et al.): processes K/V in blocks
// of 'kvBlockSize' rows, maintaining a running max/sum/weighted-output
// per query row and rescaling as each block's contribution arrives, so
// the full [seqLen,kvLen] score matrix is never materialized. Produces
// the same result as attention_forward() (up to floating-point rounding)
// — the point is the memory-access pattern, not a different answer.
// Q: [seqLen,headDim], K/V: [kvLen,headDim] -> result: [seqLen,headDim].
// When it records gradients, or on a GPU, it is the fused attention op (one
// row per thread with an online softmax: also never the full score matrix),
// whose backward recomputes the scores from each row's log-sum-exp.
&FloatTensor flash_attention_forward(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V,
                                 unsigned long kvBlockSize);

// ── Multi-head self-attention layer ──────────────────────────────────────────
// The projections around mha_forward as one layer:
//   out = mha_forward_mask(X Wq, X Wk, X Wv, numHeads, causal, window) Wo
// X: [seqLen, dModel]. The weights require grad (a Module: module.h).
struct MultiHeadAttention {
    &FloatTensor Wq;   // [dModel, dModel]
    &FloatTensor Wk;
    &FloatTensor Wv;
    &FloatTensor Wo;
    unsigned long numHeads;
    int causal;             // 0 from new(); 1 for a decoder (each token sees itself and earlier ones)
    unsigned long window;   // 0 from new(); > 0 a sliding window

    &FloatTensor forward(const &FloatTensor X) const;   // [seqLen, dModel] -> [seqLen, dModel]
    // Parameterized (module.h): Wq, Wk, Wv, Wo.
    void collect_params(&Vec<&FloatTensor> out) const;
    void collect_state(&Vec<&FloatTensor> out) const;
    void drop();
    void free() consume;   // = drop()

    // Weights U(-1/sqrt(dModel), 1/sqrt(dModel)) from rng; numHeads divides dModel.
    static struct MultiHeadAttention new(unsigned long dModel, unsigned long numHeads, &Rng rng);
};

// Internal: a [rows, cols] weight, U(-1/sqrt(rows), 1/sqrt(rows)) from rng,
// requiring grad.
&FloatTensor attn_weight_(unsigned long rows, unsigned long cols, &Rng rng);
// Internal: frees a temporary tensor a layer made (not one its arena owns).
void attn_release_(&FloatTensor t);
// Internal: ... unless 'out' is part of a graph (backward needs it).
void attn_release_tmp_(const &FloatTensor out, &FloatTensor t);

} // namespace std
