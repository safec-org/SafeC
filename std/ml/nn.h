#pragma once
// SafeC Standard Library — trainable networks on std::ml's Autograd::get core.
//
// tensor.h / tensor_nn.h carry the Tensor type, reverse-mode Autograd::get and
// the activations. This file adds what training a model end to end needs:
//
//   * Rng — a small deterministic random generator (splitmix64), so every
//     initialization, shuffle and sample is reproducible from a seed.
//   * More differentiable ops: bias add, exp / log / abs, clamp, minimum,
//     row sums, column concatenation, row gather, log-softmax, detach.
//   * Losses: MSE, Huber, binary cross-entropy on logits, softmax
//     cross-entropy on class indices.
//   * Mlp — a stack of fully connected layers with an activation.
//   * Optimizer — SGD, momentum, Nesterov, RMSProp, Adagrad, Adam, AdamW
//     and Lion over a list of parameter tensors, with gradient-norm
//     clipping.
//
// trainer.h runs the training loop; autoencoder.h, gan.h and rl.h build
// their models from these pieces. Every model here works on rows: a batch
// is a [batch, features] tensor. Class labels are tensors of class indices
// stored as floats ([batch] or [batch, 1]), the same as datasets.h makes.
#include <std/ml/tensor.h>
#include <std/ml/tensor_nn.h>
#include <std/collections/vec.h>

namespace std {

// ── Random numbers ───────────────────────────────────────────────────────────
struct Rng {
    unsigned long state;
    int           hasSpare;   // Box-Muller's second value is cached
    float         spare;

    unsigned long u64();
    float uniform();                                      // [0, 1)
    float normal();                                       // mean 0, std 1
    unsigned long below(unsigned long n);                 // [0, n)
    void fill_uniform(&FloatTensor t, float lo, float hi);
    void fill_normal(&FloatTensor t, float mean, float std);
    // A tensor of the given shape filled from N(0, 1) (a VAE's epsilon, a
    // GAN's latent batch). Not part of any graph.
    &FloatTensor normal_tensor(unsigned long rows, unsigned long cols);

    static struct Rng new(unsigned long seed);
};

// ── More differentiable ops ──────────────────────────────────────────────────
// Shapes are [rows, cols] unless noted; every op records its backward
// edge like tensor_add does.
&FloatTensor tensor_add_bias(const &FloatTensor x, const &FloatTensor b);     // x [m,n] + b [n] on every row
&FloatTensor FloatTensor::add_scalar(float k) const;
&FloatTensor FloatTensor::exp() const;
&FloatTensor FloatTensor::log() const;   // natural log; a must be > 0
&FloatTensor FloatTensor::abs() const;
&FloatTensor FloatTensor::leaky_relu(float slope) const;
&FloatTensor FloatTensor::clamp(float lo, float hi) const;   // gradient 0 where clamped
&FloatTensor tensor_minimum(const &FloatTensor a, const &FloatTensor b);      // elementwise min
&FloatTensor FloatTensor::mean() const;   // -> 1 element
&FloatTensor FloatTensor::sum_rows() const;   // [m,n] -> [m,1]
&FloatTensor tensor_concat_cols(const &FloatTensor a, const &FloatTensor b);  // [m,p] ++ [m,q] -> [m,p+q]
// out[i] = table[idx[i]]: [K,d] and m indices -> [m,d]. Gradients
// scatter-add back into the table's rows (embeddings, VQ codebooks).
&FloatTensor tensor_gather_rows(const &FloatTensor table, const &FloatTensor idx);
&FloatTensor FloatTensor::log_softmax() const;   // last axis, numerically stable
// A copy that is not part of the graph: gradients stop here.
&FloatTensor FloatTensor::detach() const;

// Constant helpers (no gradient).
&FloatTensor FloatTensor::one_hot(unsigned long classes) const;   // m labels -> [m, classes]
&FloatTensor FloatTensor::argmax_rows() const;   // [m,n] -> [m] indices

// ── Losses (each returns a 1-element tensor, averaged over elements) ─────────
&FloatTensor tensor_mse_loss(const &FloatTensor pred, const &FloatTensor target);
// 0.5 d^2 for |d| <= delta, else delta (|d| - delta/2). Gradient to pred only.
&FloatTensor tensor_huber_loss(const &FloatTensor pred, const &FloatTensor target, float delta);
// Binary cross-entropy of sigmoid(logits) against targets in [0, 1],
// computed from the logits so it never takes log(0). Gradient to logits only.
&FloatTensor tensor_bce_logits_loss(const &FloatTensor logits, const &FloatTensor target);
// Softmax cross-entropy of logits [m, classes] against m class indices,
// averaged over rows. Gradient to logits only.
&FloatTensor tensor_cross_entropy_loss(const &FloatTensor logits, const &FloatTensor labels);
// Fraction of rows whose argmax equals the label.
float   tensor_accuracy(const &FloatTensor logits, const &FloatTensor labels);

// ── Activations by tag ───────────────────────────────────────────────────────
#define ACT_NONE     0
#define ACT_RELU     1
#define ACT_LEAKY    2   // leaky ReLU, slope 0.2 (the usual GAN choice)
#define ACT_TANH     3
#define ACT_SIGMOID  4
#define ACT_GELU     5
#define ACT_SILU     6
&FloatTensor FloatTensor::activate(int act) const;

// ── Multi-layer perceptron ───────────────────────────────────────────────────
// inDim -> hidden (x hiddenLayers) -> outDim, 'act' after every hidden
// layer and 'outAct' after the last. hiddenLayers 0 is a single linear
// layer. Weights are [in, out] (transpose a PyTorch nn.Linear's), biases
// [out], initialized like PyTorch's nn.Linear: U(-1/sqrt(in), 1/sqrt(in)).
// params holds W0, b0, W1, b1, ... and every one requires grad.
struct Mlp {
    struct Vec<&FloatTensor> params;
    unsigned long nLayers;
    unsigned long inDim;
    unsigned long outDim;
    int act;
    int outAct;

    &FloatTensor forward(const &FloatTensor x) const;   // [batch, inDim] -> [batch, outDim]
    // Parameterized (module.h): the trained tensors, and what a copy or a file carries.
    void collect_params(&Vec<&FloatTensor> out) const;
    void collect_state(&Vec<&FloatTensor> out) const;
    void free();

    static struct Mlp new(unsigned long inDim, unsigned long hidden, unsigned long hiddenLayers, unsigned long outDim,
                          int act, int outAct, &Rng rng);
};

// ── Saving and loading tensors ───────────────────────────────────────────────
// A list of tensors' shapes and values to a file and back: the magic
// "SCTN", the count, then per tensor its rank, shape and float32 values
// (all little-endian, 64-bit counts). Loading fills existing tensors and
// requires the same count and shapes. Return 1 on success, 0 when the file
// can't be opened, is truncated or doesn't match.
int tensors_save(const &Vec<&FloatTensor> ts, const &char path);
int tensors_load(&Vec<&FloatTensor> ts, const &char path);

// ── Optimizers ───────────────────────────────────────────────────────────────
#define OPT_SGD       0
#define OPT_MOMENTUM  1   // heavy ball, 'momentum' 0.9
#define OPT_NESTEROV  2
#define OPT_RMSPROP   3   // squared-gradient average 'beta2' (0.99)
#define OPT_ADAM      4   // 'beta1' 0.9, 'beta2' 0.999, 'eps' 1e-8
#define OPT_ADAMW     5   // Adam with decoupled weight decay ('weightDecay' 0.01)
#define OPT_ADAGRAD   6   // per-weight lr / sqrt(sum of squared gradients) (Duchi et al. 2011)
#define OPT_LION      7   // sign of the interpolated momentum (Chen et al. 2023); 'beta1' 0.9,
                          // 'beta2' 0.99, decoupled 'weightDecay'; use ~3-10x smaller lr than Adam

// Fields are public: change lr (a schedule), betas or weight decay
// between steps. State buffers are created when a parameter is added,
// always on the heap even inside a tensor arena.
struct Optimizer {
    struct Vec<&FloatTensor> params;
    struct Vec<&FloatTensor> m;   // first moment / velocity, one per parameter
    struct Vec<&FloatTensor> v;   // second moment, one per parameter
    int   kind;
    float lr;
    float momentum;
    float beta1;
    float beta2;
    float eps;
    float weightDecay;       // L2 for SGD / momentum / RMSProp / Adagrad / Adam, decoupled for AdamW / Lion
    unsigned long t;         // steps taken

    void add(&FloatTensor param);
    void zero_grad();
    // Scales every gradient so their global L2 norm is at most maxNorm and
    // returns the norm before scaling.
    float clip_grad_norm(float maxNorm);
    void step();
    void free();                           // frees the state, not the parameters

    static struct Optimizer new(int kind, float lr);
    // An empty optimizer with o's kind and hyperparameters (not its parameters
    // or state), as nn_rocm.h's RocmOptimizer::from.
    static struct Optimizer from(const &Optimizer o);
};

} // namespace std
