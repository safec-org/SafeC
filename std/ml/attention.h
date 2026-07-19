#pragma once
// SafeC Standard Library — Attention (scaled dot-product + multi-head).
//
// Built on std/ml/tensor.h's CPU Tensor. Inference-focused (the
// vLLM-inspired half of this library's brief), not a training library:
// softmax_rows() and the attention functions below are forward-only —
// they don't record autograd edges (see tensor.h's Tensor::gradFn) even
// on requiresGrad inputs. Q/K/V are always plain 2D tensors, [seqLen,
// dim] — no batch dimension (call these once per sequence in a batch)
// and no positional encoding baked in (add it to Q/K yourself before
// calling, e.g. via tensor_add with a precomputed positional tensor).
//
// Scope note: standard scaled-dot-product/MHA, gated attention, gated
// MHA, linear attention, and flash attention are implemented and
// verified in this file. Multi-head Latent Attention (MLA)/EG-MLA,
// Shifted-Window (Swin) Attention, and Gated DeltaNet live in
// attention_advanced.h/.sc. std/ml/ROADMAP.md tracks anything still
// outstanding.
#include <std/ml/tensor.h>

namespace std {

// Row-wise softmax: for a [rows, cols] tensor, each output row sums to 1.
// Numerically stable (subtracts each row's max before exponentiating).
// Forward-only — see this header's note above.
&Tensor softmax_rows(const &Tensor x);

// out[r][c] = in[r][c] transposed -> [cols, rows].
&Tensor tensor_transpose(const &Tensor in);

// Copies out the column range [startCol, startCol+numCols) of every row.
&Tensor tensor_slice_cols(const &Tensor in, unsigned long startCol, unsigned long numCols);

// Writes 'src' into 'dst's column range [startCol, startCol+numCols)
// (numCols == src's own column count), row by row. Used to reassemble
// per-head outputs into one [seqLen, dModel] tensor after mha_forward's
// per-head loop.
void tensor_set_cols(&Tensor dst, unsigned long startCol, const &Tensor src);

// Single-head scaled dot-product attention: softmax(Q @ K^T / sqrt(headDim)) @ V.
// Q: [seqLen, headDim], K/V: [kvLen, headDim] -> result: [seqLen, headDim].
&Tensor attention_forward(const &Tensor Q, const &Tensor K, const &Tensor V);

// Multi-head attention: splits Q/K/V's dModel columns into 'numHeads'
// equal headDim-wide slices (dModel must be divisible by numHeads), runs
// attention_forward() per head, and concatenates the per-head outputs
// back into one [seqLen, dModel] result — the standard MHA shape from
// "Attention Is All You Need", without a separate output-projection
// matrix (apply your own W_O via tensor_matmul on the result if needed).
&Tensor mha_forward(const &Tensor Q, const &Tensor K, const &Tensor V, unsigned long numHeads);

// ── Gated attention ──────────────────────────────────────────────────────────
// Output gating: sigmoid(gateLogits) elementwise-multiplied into the
// attention result before it's returned (the pattern used e.g. to
// modulate an attention block's contribution to the residual stream).
// 'gateLogits' must match the output shape: [seqLen, headDim] for
// gated_attention_forward, [seqLen, dModel] for gated_mha_forward.
&Tensor gated_attention_forward(const &Tensor Q, const &Tensor K, const &Tensor V,
                                 const &Tensor gateLogits);
&Tensor gated_mha_forward(const &Tensor Q, const &Tensor K, const &Tensor V,
                           unsigned long numHeads, const &Tensor gateLogits);

// ── Linear attention ─────────────────────────────────────────────────────────
// Kernel-trick reformulation (Katharopoulos et al., "Transformers are
// RNNs"): with feature map phi(x) = elu(x)+1 (elementwise, always
// positive), softmax attention's O(seqLen*kvLen*headDim) cost becomes
// O((seqLen+kvLen)*headDim^2) by computing phi(K)^T @ V once
// ([headDim,headDim]) instead of the full [seqLen,kvLen] score matrix:
//   out_i = (phi(Q_i) . sum_j phi(K_j) (x) V_j) / (phi(Q_i) . sum_j phi(K_j))
// Q: [seqLen,headDim], K/V: [kvLen,headDim] -> result: [seqLen,headDim].
&Tensor linear_attention_forward(const &Tensor Q, const &Tensor K, const &Tensor V);

// ── Flash attention ──────────────────────────────────────────────────────────
// Tiled, online-softmax attention (Dao et al.): processes K/V in blocks
// of 'kvBlockSize' rows, maintaining a running max/sum/weighted-output
// per query row and rescaling as each block's contribution arrives, so
// the full [seqLen,kvLen] score matrix is never materialized. Produces
// the same result as attention_forward() (up to floating-point rounding)
// — the point is the memory-access pattern, not a different answer.
// Q: [seqLen,headDim], K/V: [kvLen,headDim] -> result: [seqLen,headDim].
&Tensor flash_attention_forward(const &Tensor Q, const &Tensor K, const &Tensor V,
                                 unsigned long kvBlockSize);

} // namespace std
