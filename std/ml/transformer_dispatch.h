#pragma once
// SafeC Standard Library — device-selectable JiTBlock forward pass.
//
// transformer.h/.sc stay portable and dependency-free (CPU-only,
// tensor_matmul/tensor_relu, no framework dependency) — the same
// "portable core, opt-in extension" split as tensor.h vs tensor_blas.h/
// tensor_gpu.h. This file is the extension: jit_block_forward_on/
// jit_forward_on are the exact same computation as transformer.h's
// jit_block_forward/jit_forward, with every projection matmul (Q/K/V/Wo,
// the FFN's W1/W2) and the FFN's relu routed through tensor_dispatch.h's
// tensor_matmul_on/tensor_relu_on instead of always running on the CPU.
//
// Scope boundary, stated plainly: multi-head attention's own internals
// (mha_forward -> attention_forward's per-head QK^T/softmax/*V matmuls)
// are NOT device-routed here — they still always run on the CPU via
// attention.sc's existing functions. Threading device selection all the
// way through attention.sc's several layer variants (standard MHA,
// grouped/sliding-window attention, etc.) is real, separate work this
// file doesn't attempt; what's here covers the projection and
// feedforward matmuls, which dominate a typical block's FLOPs for
// moderate head counts, but not 100% of it.
#include <std/ml/transformer.h>
#include <std/ml/tensor_dispatch.h>

namespace std {

&Tensor jit_block_forward_on(const &JiTBlock block, const &Tensor x, enum Device device);

// Same signature as transformer.h's jit_forward, plus a device — see
// tensor_dispatch.h's own header comment for why enum Device's
// enumerators (DEVICE_CPU etc.) must be referenced unqualified rather
// than as std::DEVICE_CPU.
&Tensor jit_forward_on(const &Tensor patches, const &Tensor WPatchEmbed, const &Tensor cond,
                        struct JiTBlock* blocks, unsigned long numBlocks, enum Device device);

} // namespace std
