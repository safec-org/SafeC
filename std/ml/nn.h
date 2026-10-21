#pragma once
// SafeC Standard Library — trainable networks on std::ml's autograd core.
//
// tensor.h / tensor_nn.h carry the Tensor type, reverse-mode autograd and
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
};

struct Rng    rng_new(unsigned long seed);
unsigned long rng_u64(&Rng r);
float         rng_uniform(&Rng r);                    // [0, 1)
float         rng_normal(&Rng r);                     // mean 0, std 1
unsigned long rng_below(&Rng r, unsigned long n);     // [0, n)
void          rng_fill_uniform(&Rng r, &Tensor t, float lo, float hi);
void          rng_fill_normal(&Rng r, &Tensor t, float mean, float std);
// A tensor of the given shape filled from N(0, 1) (a VAE's epsilon, a
// GAN's latent batch). Not part of any graph.
&Tensor       rng_normal_tensor(&Rng r, unsigned long rows, unsigned long cols);

// ── More differentiable ops ──────────────────────────────────────────────────
// Shapes are [rows, cols] unless noted; every op records its backward
// edge like tensor_add does.
&Tensor tensor_add_bias(const &Tensor x, const &Tensor b);     // x [m,n] + b [n] on every row
&Tensor tensor_add_scalar(const &Tensor a, float k);
&Tensor tensor_exp(const &Tensor a);
&Tensor tensor_log(const &Tensor a);                           // natural log; a must be > 0
&Tensor tensor_abs(const &Tensor a);
&Tensor tensor_leaky_relu(const &Tensor a, float slope);
&Tensor tensor_clamp(const &Tensor a, float lo, float hi);     // gradient 0 where clamped
&Tensor tensor_minimum(const &Tensor a, const &Tensor b);      // elementwise min
&Tensor tensor_mean(const &Tensor a);                          // -> 1 element
&Tensor tensor_sum_rows(const &Tensor a);                      // [m,n] -> [m,1]
&Tensor tensor_concat_cols(const &Tensor a, const &Tensor b);  // [m,p] ++ [m,q] -> [m,p+q]
// out[i] = table[idx[i]]: [K,d] and m indices -> [m,d]. Gradients
// scatter-add back into the table's rows (embeddings, VQ codebooks).
&Tensor tensor_gather_rows(const &Tensor table, const &Tensor idx);
&Tensor tensor_log_softmax(const &Tensor a);                   // last axis, numerically stable
// A copy that is not part of the graph: gradients stop here.
&Tensor tensor_detach(const &Tensor a);

// Constant helpers (no gradient).
&Tensor tensor_one_hot(const &Tensor labels, unsigned long classes);   // m labels -> [m, classes]
&Tensor tensor_argmax_rows(const &Tensor a);                             // [m,n] -> [m] indices

// ── Losses (each returns a 1-element tensor, averaged over elements) ─────────
&Tensor tensor_mse_loss(const &Tensor pred, const &Tensor target);
// 0.5 d^2 for |d| <= delta, else delta (|d| - delta/2). Gradient to pred only.
&Tensor tensor_huber_loss(const &Tensor pred, const &Tensor target, float delta);
// Binary cross-entropy of sigmoid(logits) against targets in [0, 1],
// computed from the logits so it never takes log(0). Gradient to logits only.
&Tensor tensor_bce_logits_loss(const &Tensor logits, const &Tensor target);
// Softmax cross-entropy of logits [m, classes] against m class indices,
// averaged over rows. Gradient to logits only.
&Tensor tensor_cross_entropy_loss(const &Tensor logits, const &Tensor labels);
// Fraction of rows whose argmax equals the label.
float   tensor_accuracy(const &Tensor logits, const &Tensor labels);

// ── Activations by tag ───────────────────────────────────────────────────────
#define ACT_NONE     0
#define ACT_RELU     1
#define ACT_LEAKY    2   // leaky ReLU, slope 0.2 (the usual GAN choice)
#define ACT_TANH     3
#define ACT_SIGMOID  4
#define ACT_GELU     5
#define ACT_SILU     6
&Tensor tensor_activate(const &Tensor x, int act);

// ── Multi-layer perceptron ───────────────────────────────────────────────────
// inDim -> hidden (x hiddenLayers) -> outDim, 'act' after every hidden
// layer and 'outAct' after the last. hiddenLayers 0 is a single linear
// layer. Weights are [in, out] (transpose a PyTorch nn.Linear's), biases
// [out], initialized like PyTorch's nn.Linear: U(-1/sqrt(in), 1/sqrt(in)).
// params holds W0, b0, W1, b1, ... and every one requires grad.
struct Mlp {
    struct Vec<&Tensor> params;
    unsigned long nLayers;
    unsigned long inDim;
    unsigned long outDim;
    int act;
    int outAct;
};
struct Mlp mlp_new(unsigned long inDim, unsigned long hidden, unsigned long hiddenLayers,
                   unsigned long outDim, int act, int outAct, &Rng rng);
&Tensor mlp_forward(const &Mlp m, const &Tensor x);           // [batch, inDim] -> [batch, outDim]
unsigned long mlp_param_count(const &Mlp m);
// dst = src (same shapes), and dst = tau * src + (1 - tau) * dst: target
// networks in DQN and other off-policy RL.
void mlp_copy(&Mlp dst, const &Mlp src);
void mlp_soft_update(&Mlp dst, const &Mlp src, float tau);
void mlp_free(&Mlp m);

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
    struct Vec<&Tensor> params;
    struct Vec<&Tensor> m;   // first moment / velocity, one per parameter
    struct Vec<&Tensor> v;   // second moment, one per parameter
    int   kind;
    float lr;
    float momentum;
    float beta1;
    float beta2;
    float eps;
    float weightDecay;       // L2 for SGD / momentum / RMSProp / Adagrad / Adam, decoupled for AdamW / Lion
    unsigned long t;         // steps taken
};
struct Optimizer optimizer_new(int kind, float lr);
void  optimizer_add(&Optimizer o, &Tensor param);
void  optimizer_add_mlp(&Optimizer o, const &Mlp m);
void  optimizer_zero_grad(&Optimizer o);
// Scales every gradient so their global L2 norm is at most maxNorm and
// returns the norm before scaling.
float optimizer_clip_grad_norm(&Optimizer o, float maxNorm);
void  optimizer_step(&Optimizer o);
void  optimizer_free(&Optimizer o);   // frees the state, not the parameters

} // namespace std
