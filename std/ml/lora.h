#pragma once
// SafeC Standard Library — parameter-efficient fine-tuning: LoRA and QLoRA.
//
// LoRA (Hu et al. 2021) freezes a pretrained network and learns a low-rank
// update for each linear layer:
//   y = x W + b + (alpha / r) * (dropout(x) A) B
// with A [in, r] random (Kaiming-uniform) and B [r, out] zero, so the
// adapted network starts exactly equal to the base. Only A and B train:
// r * (in + out) values per layer instead of in * out. LoraMlp::merge folds
// the update into W for deployment (no extra cost at inference).
//
// QLoRA (Dettmers et al. 2023) does the same on top of a base whose
// weights are stored in 4-bit NormalFloat (NF4): each block of 64 weights
// keeps one absmax scale and a 4-bit index into the 16 NF4 levels (the
// quantiles of a standard normal, so normally distributed weights use
// every level evenly). Double quantization also stores those scales in 8
// bits. Weights are dequantized to float32 per forward pass, so the
// frozen base costs ~4.1-4.5 bits per weight at rest while gradients flow
// in float32 through it to the adapters.
//
// Both wrap an Mlp (nn.h) and are Modules (module.h), so they train like
// any other model:
//   struct LoraMlp ft = LoraMlp::attach(&pretrained, &cfg);
//   model_add_params(&ft, &opt);       // adapters only
//   module_fit(&ft, TRAIN_LOSS_CE, &opt, &data, &tc);
#include <std/ml/tensor.h>
#include <std/ml/nn.h>
#include <std/collections/vec.h>

namespace std {

struct LoraConfig {
    unsigned long rank;   // r
    float alpha;          // the update is scaled by alpha / r
    float dropout;        // on the adapter's input while training (grad enabled)
    unsigned long seed;

    // alpha = rank (scale 1), no dropout.
    static struct LoraConfig default(unsigned long rank);
};

// ── LoRA ─────────────────────────────────────────────────────────────────────
// Borrows the base Mlp and freezes its parameters (requiresGrad = 0) until
// LoraMlp::detach; the base must outlive the adapter.
struct LoraMlp {
    &Mlp base;
    struct Vec<&FloatTensor> A;    // per layer [in, r]
    struct Vec<&FloatTensor> B;    // per layer [r, out]
    struct LoraConfig cfg;
    float scale;              // alpha / r
    struct Rng rng;

    &FloatTensor forward(const &FloatTensor x);
    // Parameterized (module.h): the adapters (A_0, B_0, A_1, B_1, ...), which
    // are all that trains and all a LoRA file holds.
    void collect_params(&Vec<&FloatTensor> out) const;
    void collect_state(&Vec<&FloatTensor> out) const;
    // W += scale * A B in the base for every layer, then B = 0: the base alone
    // now computes what the adapted network did.
    void merge();
    // Frees the adapters and unfreezes the base.
    void detach();

    static struct LoraMlp attach(&Mlp base, const &LoraConfig cfg);
};

// ── NF4 quantization ─────────────────────────────────────────────────────────
#define NF4_BLOCK        64UL    // weights per absmax scale
#define NF4_SCALE_BLOCK  256UL   // scales per second-level scale (double quantization)

struct Nf4Tensor {
    &heap unsigned char codes;      // two 4-bit level indices per byte, low nibble first
    &heap float absmax;             // one per block (when not double-quantized)
    &heap unsigned char absmaxQ;    // int8 + 128 per block (double-quantized)
    &heap float absmaxScale;        // one per NF4_SCALE_BLOCK blocks
    float absmaxMean;               // subtracted before the 8-bit quantization
    unsigned long rows;
    unsigned long cols;
    unsigned long blockSize;
    unsigned long blocks;
    int doubleQuant;

    unsigned long bytes() const;   // storage, all parts
    void free();

    static struct Nf4Tensor quantize(const &FloatTensor w, unsigned long blockSize, int doubleQuant);
    &FloatTensor dequantize() const;   // [rows, cols], no grad
};

// ── QLoRA ────────────────────────────────────────────────────────────────────
// Copies the base: weights to NF4, biases as float32, all frozen. The base
// Mlp may be freed afterwards.
struct QLoraMlp {
    struct Vec<struct Nf4Tensor> W;
    struct Vec<&FloatTensor> bias;
    struct Vec<&FloatTensor> A;
    struct Vec<&FloatTensor> B;
    unsigned long nLayers;
    int act;
    int outAct;
    struct LoraConfig cfg;
    float scale;
    struct Rng rng;

    &FloatTensor forward(const &FloatTensor x);
    // Parameterized (module.h): the adapters, as for LoraMlp.
    void collect_params(&Vec<&FloatTensor> out) const;
    void collect_state(&Vec<&FloatTensor> out) const;
    unsigned long base_bytes() const;        // the frozen base at rest
    void free();

    static struct QLoraMlp attach(const &Mlp base, const &LoraConfig cfg, unsigned long blockSize, int doubleQuant);
};

} // namespace std
