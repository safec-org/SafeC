#pragma once
// SafeC Standard Library — device-resident training on ROCm.
//
// nn.h, trainer.h, autoencoder.h, convnet.h, gan.h and rl.h train on the
// CPU with autograd. This file runs the same models on an AMD GPU with
// everything — parameters, activations, gradients, optimizer state, the
// dataset and the running loss — kept in GPU memory, so a training step
// moves nothing between host and device:
//
//   * RocmNet — a stack of layers with an activation after each, with a
//     hand-written backward pass: fully connected, convolution and
//     transposed convolution (any Conv2dSpec: rectangular kernels, strides,
//     padding, dilation, groups, output padding), max / avg / global
//     pooling, nearest upsampling, resizing (nearest / bilinear / bicubic),
//     reshape, batch norm (batch or running statistics), group / layer /
//     instance norm, dropout, spectral norm, and residual (add, with an
//     optional projection) and U-Net (concat) skips. Built from an Mlp, any
//     ConvNet (rocm_net_from_convnet) or layer by layer. Matmuls go to
//     rocBLAS (gpu_rocm.h's rocm_gemm); convolutions run on MIOpen when the
//     program links it, else as im2col / col2im + rocBLAS (rocm_conv_engine);
//     bias, activations, norms and their gradients are HIP kernels.
//     Activations can be fp16 or bf16, with fp32 weights, gradients and
//     optimizer state (rocm_net_set_precision).
//   * Losses that write their gradient and add to a device loss sum:
//     softmax cross-entropy on class indices, binary cross-entropy on
//     logits, MSE.
//   * RocmOptimizer — nn.h's optimizers (OPT_SGD ... OPT_LION) with the
//     same arithmetic, over all parameters in one launch, with optional
//     global gradient-norm clipping and weight clamping. Hyperparameters
//     and the step count live on the GPU, so a recorded step stays correct
//     when replayed.
//   * RocmVae — autoencoder.h's Vae, ConvVae and their conditional forms,
//     the reparameterization noise drawn on the GPU; RocmConvAe — its
//     ConvAutoencoder.
//   * RocmGan — gan.h's Gan and Dcgan: every loss including the gradient
//     penalty, conditional models, the averaged generator.
//   * RocmDqn / RocmPolicy — rl.h's DQN and REINFORCE / A2C / PPO agents,
//     following the host agents' random streams step for step.
//   * rocm_trainer_fit — trainer.h's trainer_fit on the GPU: the dataset is
//     uploaded once, each epoch uploads only its shuffled row order (the
//     same order trainer_fit's DataLoader uses for that seed), batches are
//     gathered on the GPU, and after a first eager step (which also lets
//     rocBLAS pick its kernels) steps are recorded once into RocmGraphs
//     (ROCM_TRAIN_CHUNK steps per graph) and replayed. The loss is read
//     back once per epoch.
//   * RocmTrainer, RocmGanTrainer, RocmDqnAgent, RocmPgAgent — the above
//     behind train_traits.h's traits, so train_fit / train_gan and code
//     written against the traits run here or on the CPU (train.h picks
//     the backend from the build's rocm feature).
//
// A small fully connected net (every activation of a batch fits in one
// workgroup's LDS: 2-32-32-2 or 4-64-64-2 at batch 32) runs forward, loss
// and backward of a step as one kernel in the rocm_net_*_step functions,
// since the fixed cost of each kernel (~1.3 us on an MI100, even inside a
// graph) outweighs its work there. Such nets still train faster on the CPU
// (nn.h) at small batches; the GPU path pays off from mid-sized layers
// (784-256-256-10: 3.3x faster than the CPU) and for convolutions.
//
//   struct Mlp net = mlp_new(784UL, 256UL, 2UL, 10UL, ACT_RELU, ACT_NONE, &rng);
//   struct RocmNet dnet = rocm_net_from_mlp(&net, 128UL);
//   struct RocmOptimizer opt = rocm_optimizer_new(OPT_ADAM, (float)0.001);
//   rocm_optimizer_add_net(&opt, &dnet);
//   struct TrainReport rep = rocm_trainer_fit(&dnet, rocm_net_ce_step, &opt, &train, &cfg);
//   rocm_net_download(&dnet, &net.params);   // back into the host Mlp
//
// Layout. Every buffer is row-major with one row per sample: a batch of
// images is [batch, C*H*W] with CHW rows (convnet.h's layout, i.e. NCHW), at
// the API and between layers alike.
//
// Every op is queued on gpu_rocm.h's stream and returns 1 when enqueued, 0
// on a bad shape or when there is no GPU. Gradients are overwritten (not
// accumulated) by each backward pass. Link with -lamdhip64 -lhiprtc
// -lrocblas (and -lMIOpen for MIOpen convolutions).
#include <std/ml/gpu_rocm.h>
#include <std/ml/nn.h>
#include <std/ml/datasets.h>
#include <std/ml/trainer.h>
#include <std/ml/train_traits.h>
#include <std/ml/autoencoder.h>
#include <std/ml/convnet.h>
#include <std/ml/gan.h>
#include <std/ml/rl.h>
#include <std/collections/vec.h>

namespace std {

// ── Layers and networks ──────────────────────────────────────────────────────

#define ROCM_LAYER_LINEAR         0
#define ROCM_LAYER_CONV           1   // convnet.h's tensor_conv2d_ex (any Conv2dSpec)
#define ROCM_LAYER_CONVT          2   // tensor_conv_transpose2d_ex
#define ROCM_LAYER_MAXPOOL        3   // tensor_maxpool2d_ex (any Pool2dSpec)
#define ROCM_LAYER_AVGPOOL        4   // tensor_avgpool2d_ex
#define ROCM_LAYER_GLOBAL_AVGPOOL 5   // -> C x 1 x 1
#define ROCM_LAYER_GLOBAL_MAXPOOL 6
#define ROCM_LAYER_UPSAMPLE       7   // nearest, integer factor
#define ROCM_LAYER_RESHAPE        8   // a new C x H x W of the same size
#define ROCM_LAYER_BATCHNORM      9   // tensor_batch_norm
#define ROCM_LAYER_GROUPNORM      10  // tensor_group_norm
#define ROCM_LAYER_SAVE           11  // keeps its input for a later ADD / CONCAT (passes it on)
#define ROCM_LAYER_ADD            12  // + a saved activation (through a 1x1 projection when 'flag')
#define ROCM_LAYER_CONCAT         13  // a saved activation's channels appended
#define ROCM_LAYER_RESIZE         14  // tensor_resize2d ('mode', 'flag' = alignCorners)
#define ROCM_LAYER_DROPOUT        15  // tensor_dropout / _dropout2d ('prob', 'flag' = by channel)

struct RocmLayer {
    int kind;                 // ROCM_LAYER_*
    int act;                  // ACT_* (nn.h) after the layer
    struct RocmBuf w;         // linear [in, out]; conv [outC, inC/groups*kH*kW]; convT [inC, outC/groups*kH*kW];
                              // norms: gamma; dev 0 for layers without parameters
    struct RocmBuf b;         // bias [out] / [outC]; norms: beta
    struct RocmBuf gw;        // gradients, written by each backward pass
    struct RocmBuf gb;
    unsigned long inC;        // the input's geometry (a fully connected layer's input is
    unsigned long inH;        // inC*inH*inW features)
    unsigned long inW;
    unsigned long outC;       // linear: out features (outH = outW = 1)
    unsigned long outH;
    unsigned long outW;
    struct Conv2dSpec conv;   // CONV / CONVT
    struct Pool2dSpec pool;   // MAXPOOL / AVGPOOL
    unsigned long factor;     // UPSAMPLE factor, GROUPNORM groups
    unsigned long slot;       // ADD / CONCAT: the SAVE layer's index in the net
    unsigned long skipC;      // ... and the saved activation's geometry
    unsigned long skipH;
    unsigned long skipW;
    int   skipAccum;          // ADD / CONCAT: adds its gradient to the saved one (a later consumer wrote it)
    int   mode;               // RESIZE mode
    int   flag;               // RESIZE alignCorners, ADD projection, DROPOUT by channel
    float prob;               // DROPOUT
    unsigned long seed;       // DROPOUT: the mask's random stream
    int   foldBias;           // CONV / CONVT: the batch norm after it applies its bias (and its gradient)
    int   alias;              // 1: h, 2: dh are another layer's buffers (a SAVE's input, an ADD's gradient)
    int   spectral;           // CONV / CONVT / LINEAR: the weight is spectrally normalized
    struct RocmBuf wsn;       // ... the normalized weight the layer computes with
    float momentum;           // BATCHNORM
    float eps;                // BATCHNORM / GROUPNORM
    struct RocmBuf runMean;   // BATCHNORM running statistics
    struct RocmBuf runVar;
    struct RocmBuf stats;     // norms: saved mean / inverse std; max pools: each window's argmax
    struct RocmBuf z;         // pre-activation, only for GELU / SiLU (their gradient needs it)
    struct RocmBuf h;         // output
    struct RocmBuf dh;        // gradient of the output (made the pre-activation's in place)
    struct RocmBuf col;       // convolutions (im2col engine): im2col / col2im scratch
    struct RocmBuf cm;        // ... and the channel-major side of its GEMMs
    struct RocmBuf part;      // bias-gradient partial sums
    struct RocmBuf aux;       // SAVE: the skip gradient; ADD: the projection's input gradient;
                              // RESIZE: tap tables; DROPOUT: draw counter; spectral norm: u, v, sigma
    struct RocmBuf wh;        // 16-bit nets: the weight rounded to their type, each forward pass
};

#define ROCM_MAX_SEGMENTS 8UL

struct RocmNet {
    struct Vec<struct RocmLayer> layers;
    unsigned long maxBatch;   // every buffer is sized for this many rows
    unsigned long inC;        // input geometry (set by the first layer, or rocm_net_set_input)
    unsigned long inH;
    unsigned long inW;
    int training;             // batch norms: batch statistics (1, the default) or running ones (0)
    unsigned long segments;   // batch norms normalize the batch's rows in this many equal parts
                              // (1..ROCM_MAX_SEGMENTS): stacked batches that two separate
                              // passes would have normalized separately, e.g. a GAN
                              // discriminator's real and generated rows
    int ok;                   // 0 once an upload, allocation or shape check failed
    int precision;            // activations: 0 fp32, ROCM_FP16 or ROCM_BF16 (rocm_net_set_precision)
    struct RocmBuf scale;     // 16-bit nets: the loss scale and its count of good steps
    struct RocmBuf xh;        // ... the input, its gradient, the output and its gradient
    struct RocmBuf dxh;       //     (16-bit; the output's are fp32)
    struct RocmBuf out32;
    struct RocmBuf dout32;
};

struct RocmNet rocm_net_new(unsigned long maxBatch);
// The input geometry for a net whose first layer takes it from the
// previous one (pooling, upsampling, reshape, norms). Before any layer.
int rocm_net_set_input(&RocmNet n, unsigned long channels, unsigned long height, unsigned long width);
// Appends a layer, uploading its parameters from host tensors (shapes as
// nn.h's Mlp / convnet.h). A layer's input must match the previous layer's
// output (conv: inC*inH*inW). Return 1, or 0 (and n.ok = 0).
int rocm_net_add_linear(&RocmNet n, const &Tensor w, const &Tensor b, int act);
int rocm_net_add_conv2d(&RocmNet n, const &Tensor w, const &Tensor b, unsigned long inC, unsigned long inH,
                        unsigned long inW, unsigned long kernel, unsigned long stride, unsigned long padding, int act);
int rocm_net_add_conv_transpose2d(&RocmNet n, const &Tensor w, const &Tensor b, unsigned long inC, unsigned long inH,
                                  unsigned long inW, unsigned long kernel, unsigned long stride,
                                  unsigned long padding, int act);
int rocm_net_add_conv2d_ex(&RocmNet n, const &Tensor w, const &Tensor b, unsigned long inC, unsigned long inH,
                           unsigned long inW, const &Conv2dSpec spec, int act);
int rocm_net_add_conv_transpose2d_ex(&RocmNet n, const &Tensor w, const &Tensor b, unsigned long inC,
                                     unsigned long inH, unsigned long inW, const &Conv2dSpec spec, int act);
// These take their input geometry from the previous layer (or rocm_net_set_input).
// rocm_net_add_save returns the SAVE layer's index, the slot that ADD /
// CONCAT name; ADD's projection weight is [C, skipC] (a 1x1 convolution
// with the stride that maps the saved size onto the current one).
unsigned long rocm_net_add_save(&RocmNet n);
int rocm_net_add_skip(&RocmNet n, unsigned long slot, ?&Tensor projW, ?&Tensor projB, int act);
int rocm_net_add_concat(&RocmNet n, unsigned long slot);
int rocm_net_add_resize(&RocmNet n, unsigned long outH, unsigned long outW, int mode, int alignCorners);
// p: drop probability; the mask is drawn on the GPU from 'seed' (not rng.h's stream).
int rocm_net_add_dropout(&RocmNet n, float p, int byChannel, unsigned long seed);
// Spectral normalization of the last layer's weight (CONV / CONVT / LINEAR),
// u [weight rows] the power iteration's vector (one iteration per training
// forward pass, as convnet.h).
int rocm_net_spectral_norm(&RocmNet n, const &Tensor u);
int rocm_net_add_maxpool(&RocmNet n, const &Pool2dSpec spec, int act);
int rocm_net_add_avgpool(&RocmNet n, const &Pool2dSpec spec, int act);
int rocm_net_add_global_pool(&RocmNet n, int max, int act);
int rocm_net_add_upsample(&RocmNet n, unsigned long factor, int act);
int rocm_net_add_reshape(&RocmNet n, unsigned long channels, unsigned long height, unsigned long width);
// gamma, beta [C]; runMean, runVar [C] (the starting running statistics).
int rocm_net_add_batchnorm(&RocmNet n, const &Tensor gamma, const &Tensor beta, const &Tensor runMean,
                           const &Tensor runVar, float momentum, float eps, int act);
int rocm_net_add_groupnorm(&RocmNet n, const &Tensor gamma, const &Tensor beta, unsigned long groups, float eps,
                           int act);
// Convolution engine for CONV / CONVT layers. ROCM_CONV_AUTO (the default)
// runs them on MIOpen when the program links it (-lMIOpen: implicit GEMM,
// Winograd, matrix cores), else on the im2col engine (im2col / col2im +
// rocBLAS); ROCM_CONV_IM2COL always uses the latter. MIOpen picks its
// kernels for each shape and batch size on the first forward pass, which
// must run outside graph capture (rocm_trainer_fit's first step does);
// that pass also times an fp32 shape's forward and backward on both
// engines and keeps the im2col engine for it when that is clearly faster
// (small convolutions, where MIOpen's cost per call dominates).
// Returns 1 if MIOpen is in use. Applies to layers added afterwards.
#define ROCM_CONV_AUTO   0
#define ROCM_CONV_IM2COL 1
int rocm_conv_engine(int engine);
// An Mlp's layers (its act / outAct), for batches of up to maxBatch rows.
struct RocmNet rocm_net_from_mlp(const &Mlp m, unsigned long maxBatch);
// A ConvNet's layers, parameters, buffers (running statistics, spectral
// norms' u) and training mode: every ConvNet layer kind. Dropout masks are
// drawn on the GPU (seeded from the layer's Rng), so they differ from the
// CPU's.
struct RocmNet rocm_net_from_convnet(const &ConvNet c, unsigned long maxBatch);
unsigned long rocm_net_in_dim(const &RocmNet n);
unsigned long rocm_net_out_dim(const &RocmNet n);
int  rocm_net_set_training(&RocmNet n, int training);   // returns the previous mode
// Mixed precision (after the last layer, before rocm_optimizer_add_net):
// activations and their gradients in ROCM_FP16 or ROCM_BF16 (0: fp32),
// the arithmetic in fp32. Weights, gradients and optimizer state stay fp32
// and are rounded to 16 bits for each forward pass; the input, the output
// (rocm_net_forward) and their gradients stay fp32, so losses and steps are
// unchanged. Convolutions need MIOpen (rocm_conv_engine) and linear layers
// run on 16-bit GEMMs. fp16 adds a dynamic loss scale, as torch's
// GradScaler: the output gradient is multiplied by it, the optimizer
// divides it out, skips a step whose gradients overflowed and halves it,
// and doubles it after 2000 good steps (from 65536). 0 when this net can't
// run in that precision. On an MI100 fp16 is the fast one (matrix cores
// for MIOpen's fp16 kernels); its bf16 convolutions run at about fp32's
// speed. rocm_net_loss_scale / _set_loss_scale read and set the scale (1
// for bf16).
int   rocm_net_set_precision(&RocmNet n, int dtype);
int   rocm_net_set_loss_scale(&RocmNet n, float scale);
float rocm_net_loss_scale(const &RocmNet n);   // waits
// Every parameterized layer's weight and bias (gamma and beta for norms),
// in order (W0, b0, W1, b1, ...: an Mlp's or a ConvNet's params), from or
// to host tensors of the same sizes. Download waits.
int rocm_net_download(const &RocmNet n, &Vec<&Tensor> params);
int rocm_net_upload(&RocmNet n, const &Vec<&Tensor> params);
// The non-trained buffers in a ConvNet's order: each batch norm's running
// mean and variance, each spectral norm's u.
int rocm_net_download_stats(const &RocmNet n, &Vec<&Tensor> buffers);
int rocm_net_upload_stats(&RocmNet n, const &Vec<&Tensor> buffers);
// Parameters and running statistics back into the ConvNet it was built from.
int rocm_net_download_convnet(const &RocmNet n, &ConvNet c);
void rocm_net_free(&RocmNet n);

// Forward pass on x [batch, inDim]. Returns the output [batch, outDim] (a
// view of the net's own buffer, valid until the next forward), dev == 0 on
// failure.
struct RocmBuf rocm_net_forward(&RocmNet n, struct RocmBuf x, unsigned long batch);
// Where a loss writes d(loss)/d(output) before rocm_net_backward.
struct RocmBuf rocm_net_grad_output(const &RocmNet n);
// Backward pass after rocm_net_forward(n, x, batch): every layer's gw / gb,
// and d(loss)/d(x) into dx when dx.dev != 0 (a generator trained through a
// discriminator, or a VAE's decoder).
int rocm_net_backward(&RocmNet n, struct RocmBuf x, unsigned long batch, struct RocmBuf dx);

// ── Losses ───────────────────────────────────────────────────────────────────
// Each writes d(loss)/d(input) into d and ADDS the loss to loss1[0] (zero
// it with rocm_zero to start a new sum), so a run of steps can sum its
// losses on the GPU and read them back once. 'scale' multiplies both.

// Softmax cross-entropy of logits [rows, classes] against class indices
// (rows floats), averaged over rows (tensor_cross_entropy_loss).
int rocm_cross_entropy(struct RocmBuf logits, struct RocmBuf labels, struct RocmBuf d, struct RocmBuf loss1,
                       unsigned long rows, unsigned long classes);
// scale * mean binary cross-entropy of sigmoid(logits) against targets in
// [0, 1] over n values (tensor_bce_logits_loss).
int rocm_bce_logits(struct RocmBuf logits, struct RocmBuf target, struct RocmBuf d, struct RocmBuf loss1,
                    unsigned long n, float scale);
// scale * mean squared error over n values (tensor_mse_loss).
int rocm_mse(struct RocmBuf pred, struct RocmBuf target, struct RocmBuf d, struct RocmBuf loss1,
             unsigned long n, float scale);

// Training steps for rocm_trainer_fit: forward, loss, backward on one batch.
// For a small fully connected net (see the top of this file) they run as
// one kernel that leaves only the gradients and the last layer's output
// current, not the other layers' activations.
int rocm_net_ce_step(&RocmNet n, struct RocmBuf x, struct RocmBuf y, unsigned long rows, struct RocmBuf loss1);
int rocm_net_mse_step(&RocmNet n, struct RocmBuf x, struct RocmBuf y, unsigned long rows, struct RocmBuf loss1);
int rocm_net_bce_step(&RocmNet n, struct RocmBuf x, struct RocmBuf y, unsigned long rows, struct RocmBuf loss1);
// Autoencoders: the target is x itself (y is ignored).
int rocm_net_recon_bce_step(&RocmNet n, struct RocmBuf x, struct RocmBuf y, unsigned long rows, struct RocmBuf loss1);
int rocm_net_recon_mse_step(&RocmNet n, struct RocmBuf x, struct RocmBuf y, unsigned long rows, struct RocmBuf loss1);

// ── Optimizer ────────────────────────────────────────────────────────────────
// nn.h's Optimizer on the GPU. Change lr, betas, weightDecay or clipNorm
// between steps like the host fields of an Optimizer: rocm_optimizer_step
// copies changed values to the GPU when it runs outside graph capture, and
// rocm_optimizer_sync does so explicitly (before replaying a recorded step).
struct RocmOptimizer {
    struct Vec<struct RocmBuf> w;   // parameters
    struct Vec<struct RocmBuf> g;   // their gradients
    struct Vec<struct RocmBuf> m;   // first moment / velocity
    struct Vec<struct RocmBuf> v;   // second moment
    int   kind;                     // OPT_*
    float lr;
    float momentum;
    float beta1;
    float beta2;
    float eps;
    float weightDecay;
    float clipNorm;                 // > 0: scale the gradients to this global L2 norm at most
    float clampWeights;             // > 0: clamp every weight to [-c, c] after each step (WGAN's critic)
    unsigned long t;                // steps taken (the GPU counts its own for the bias correction)
    struct RocmBuf state;           // GPU copy of the hyperparameters, step count, clip scale
    float sent[7];                  // the values last copied to 'state'
    struct RocmBuf lossScale;       // an fp16 net's loss scale (rocm_optimizer_add_net), or none
    int   ok;
};
// Same defaults as optimizer_new.
struct RocmOptimizer rocm_optimizer_new(int kind, float lr);
// An empty optimizer with o's kind and hyperparameters (not its state).
struct RocmOptimizer rocm_optimizer_from(const &Optimizer o);
int  rocm_optimizer_add(&RocmOptimizer o, struct RocmBuf w, struct RocmBuf g);
int  rocm_optimizer_add_net(&RocmOptimizer o, const &RocmNet n);
// One step over every parameter: one launch per 32 parameter tensors, plus
// as many for the gradient norm when clipping. Queued.
int  rocm_optimizer_step(&RocmOptimizer o);
int  rocm_optimizer_sync(&RocmOptimizer o);
void rocm_optimizer_free(&RocmOptimizer o);   // frees the state, not the parameters

// ── Row ops ──────────────────────────────────────────────────────────────────
// out[r] = (x[r], one constant plane of 'plane' values per class, 1 in the
// plane of label[r] and 0 elsewhere), x [rows, channels*plane]: a one-hot
// label appended to rows (plane 1, gan.h's and autoencoder.h's conditional
// models) or label channels appended to images (convnet.h's
// tensor_append_label_channels). labels: rows class indices.
int rocm_append_labels(struct RocmBuf x, struct RocmBuf labels, struct RocmBuf out, unsigned long rows,
                       unsigned long channels, unsigned long plane, unsigned long classes);
// dst[r][j] = src[r][offset + j] for j < dstCols (src rows of srcCols).
int rocm_slice_cols(struct RocmBuf src, struct RocmBuf dst, unsigned long rows, unsigned long srcCols,
                    unsigned long dstCols, unsigned long offset);

// ── Variational autoencoder ──────────────────────────────────────────────────
// autoencoder.h's Vae, conditional (classes > 0) or not, with the same loss:
// reconstruction summed over features plus beta * KL, averaged over the
// batch. The mu and log-variance heads run as one fused layer; the noise is
// drawn on the GPU. A CVAE reads each row's class index from y.
struct RocmVae {
    struct RocmNet enc;       // x (+ one-hot label) -> hidden
    struct RocmNet head;      // hidden -> (mu | logvar), [rows, 2*latent]
    struct RocmNet dec;       // z (+ one-hot label) -> x
    struct RocmBuf eps;       // [maxBatch, latent] this step's noise
    struct RocmBuf zbuf;      // [maxBatch, latent]
    struct RocmBuf dz;
    struct RocmBuf encIn;     // CVAE: [maxBatch, inDim + classes]
    struct RocmBuf decIn;     // CVAE: [maxBatch, latent + classes]
    struct RocmBuf decGrad;   // CVAE: d(loss)/d(decIn)
    struct RocmBuf counter;   // [1] noise draws so far (advanced on the GPU)
    unsigned long inDim;
    unsigned long encChannels;  // the encoder input's channels and plane: a CVAE's labels join as
    unsigned long encPlane;     // planes (Vae: inDim, 1; ConvVae: image channels, H*W)
    unsigned long latent;
    unsigned long classes;
    unsigned long maxBatch;
    int   recon;              // RECON_MSE / RECON_BCE
    float beta;
    unsigned long seed;
    int   ok;
};
struct RocmVae rocm_vae_new(const &Vae v, unsigned long maxBatch);
int  rocm_vae_download(const &RocmVae d, &Vae v);
// autoencoder.h's ConvVae (ConvNet encoder and decoder) as a RocmVae.
struct RocmVae rocm_conv_vae_new(const &ConvVae v, unsigned long maxBatch);
int  rocm_conv_vae_download(const &RocmVae d, &ConvVae v);
int  rocm_vae_step(&RocmVae v, struct RocmBuf x, struct RocmBuf y, unsigned long rows, struct RocmBuf loss1);
int  rocm_optimizer_add_vae(&RocmOptimizer o, const &RocmVae v);
void rocm_vae_free(&RocmVae v);

// ── Convolutional autoencoder ────────────────────────────────────────────────
// autoencoder.h's ConvAutoencoder: conv_autoencoder_loss's reconstruction
// loss (mean BCE on logits or MSE against x).
struct RocmConvAe {
    struct RocmNet enc;
    struct RocmNet dec;
    int recon;
    int ok;
};
struct RocmConvAe rocm_conv_autoencoder_new(const &ConvAutoencoder ae, unsigned long maxBatch);
int  rocm_conv_autoencoder_step(&RocmConvAe a, struct RocmBuf x, struct RocmBuf y, unsigned long rows,
                                struct RocmBuf loss1);
int  rocm_optimizer_add_conv_autoencoder(&RocmOptimizer o, const &RocmConvAe a);
int  rocm_conv_autoencoder_download(const &RocmConvAe a, &ConvAutoencoder ae);
void rocm_conv_autoencoder_free(&RocmConvAe a);

// ── GANs ─────────────────────────────────────────────────────────────────────
// gan.h's Gan (MLP) and Dcgan (ConvNet) on the GPU: every loss (GAN_*),
// nCritic critic steps per generator step, WGAN weight clipping, the
// gradient penalty (as gan.sc computes it), conditional models and the
// averaged generator, with the same arithmetic as gan_train_step. The
// discriminator step runs D once on the real and generated rows stacked,
// its batch norms normalizing the two halves separately (as two passes
// would). Noise is drawn on the GPU, so a step records into a graph.
struct RocmGan {
    struct RocmNet gen;
    struct RocmNet disc;
    struct RocmOptimizer optG;
    struct RocmOptimizer optD;
    struct Vec<struct RocmBuf> ema;      // averaged generator: each parameter, then batch norm statistics
    struct Vec<struct RocmBuf> emaSrc;   // the generator buffers they follow
    struct GanConfig cfg;
    unsigned long dataDim;   // values per data row
    unsigned long channels;  // labels join as extra planes of 'plane' values (MLP: dataDim, 1)
    unsigned long plane;
    unsigned long maxBatch;
    struct RocmBuf noise;    // this step's draws: per critic step a latent batch, then the generator's,
                             // then per critic step the interpolation weights (gradient penalty)
    struct RocmBuf gIn;      // conditional: G's input (noise + one-hot label)
    struct RocmBuf dData;    // D's data rows: real | fake (| xp | xm)
    struct RocmBuf dIn;      // conditional: the stack with label planes
    struct RocmBuf dGrad;    // d(loss)/d(D input)
    struct RocmBuf xh;       // gradient penalty: interpolates, their input gradient, coefficients
    struct RocmBuf xhIn;
    struct RocmBuf gData;
    struct RocmBuf coef;
    struct RocmBuf state;    // noise counter, steps, D loss, G loss, penalty, block count
    unsigned long steps;     // generator steps taken
    int ok;
};
struct RocmGan rocm_gan_new(const &Gan g, unsigned long maxBatch);
struct RocmGan rocm_dcgan_new(const &Dcgan g, unsigned long maxBatch);
// One gan_train_step on real [rows, dataDim]; labels: rows class indices
// for a conditional GAN (else ignored). Queued.
int rocm_gan_step(&RocmGan g, struct RocmBuf real, struct RocmBuf labels, unsigned long rows);
struct GanLosses rocm_gan_losses(const &RocmGan g);   // the last step's (waits)
// gan_train on the GPU: the same batches, steps recorded into a graph.
struct GanLosses rocm_gan_train(&RocmGan g, const &Dataset data, unsigned long epochs, unsigned long batchSize);
// The parameters (and batch norm statistics) of G, D and the averaged G
// back into the host model; the optimizers' moments stay on the GPU.
int rocm_gan_download(const &RocmGan d, &Gan g);
int rocm_dcgan_download(const &RocmGan d, &Dcgan g);
void rocm_gan_free(&RocmGan g);
// Internal (exposed for tests): the step with drawNoise 0 uses the values
// already in g.noise (rows-sized batches at strides of maxBatch, in the
// layout above).
int rocm_gan_step_(&RocmGan g, struct RocmBuf real, struct RocmBuf labels, unsigned long rows, int drawNoise);

// ── Reinforcement learning ───────────────────────────────────────────────────
// rl.h's Dqn and PolicyAgent with their networks, optimizer state and (for
// DQN) replay buffer on the GPU. The environment loop stays on the host:
// acting uploads one observation and reads back the action values, a DQN
// observation uploads the transition (and, when it trains, the sampled
// slots), a policy update uploads its rollout once. Every random choice
// (exploration, replay sampling, minibatch order, action sampling) is drawn
// on the host from the agent's own Rng exactly as rl.sc does, so a device
// agent built from a host one takes the same actions and updates up to
// rounding. Each update records once into a graph and is replayed.

struct RocmDqn {
    struct RocmNet q;
    struct RocmNet target;
    struct RocmOptimizer opt;
    struct Vec<struct RocmBuf> qParams;   // soft target updates: target <- q
    struct Vec<struct RocmBuf> tParams;
    struct DqnConfig cfg;
    struct Rng rng;
    unsigned long obsDim;
    unsigned long actions;
    unsigned long steps;            // environment steps observed
    unsigned long updates;          // gradient steps taken
    unsigned long size;             // transitions stored
    unsigned long pos;              // next slot
    struct RocmBuf replay;          // [bufferSize][obs | next | action | reward | terminal]
    struct RocmBuf idx;             // [batchSize] sampled slots
    struct RocmBuf ob;              // the sampled batch
    struct RocmBuf nx;
    struct RocmBuf act;
    struct RocmBuf rew;
    struct RocmBuf term;
    struct RocmBuf y;               // targets
    struct RocmBuf state;           // [counter, updates, loss, -, -, block count]
    struct RocmBuf obs1;            // acting: one observation
    &heap float   stage;            // host staging (a transition, the sampled slots, action values)
    struct RocmGraph graph;         // one update, recorded after the first
    int   ok;
};
// A device copy of d (networks, replay contents, counters and Rng state).
struct RocmDqn rocm_dqn_new(const &Dqn d);
float         rocm_dqn_epsilon(const &RocmDqn d);
unsigned long rocm_dqn_act(&RocmDqn d, const &Tensor obs, int explore);   // waits for the action values
// dqn_observe: stores the transition and queues an update when due. Returns
// 1 when an update was queued, 0 when none was due, -1 on failure (no
// read-back: rocm_dqn_loss gives the last update's loss).
int           rocm_dqn_observe(&RocmDqn d, const &Tensor obs, unsigned long action, float reward,
                               const &Tensor next, int terminal);
float         rocm_dqn_loss(const &RocmDqn d);   // waits
int           rocm_dqn_download(const &RocmDqn d, &Dqn h);   // networks, counters, Rng (not the replay buffer)
void          rocm_dqn_free(&RocmDqn d);

struct RocmPolicy {
    struct RocmNet policy;          // sized for maxRollout rows
    struct RocmNet value;
    struct RocmOptimizer opt;
    struct PgConfig cfg;
    struct Rng rng;
    unsigned long obsDim;
    unsigned long actions;
    unsigned long maxRollout;
    struct Vec<float> roll;         // host rollout rows: obs | next | action | reward | terminal | done
    unsigned long rollout;          // transitions stored
    struct RocmBuf data;            // the uploaded rollout, SoA: obs, next, action, reward, terminal, done
    struct RocmBuf v;               // values, next values, old log-probabilities, advantages, returns
    struct RocmBuf vn;
    struct RocmBuf logpOld;
    struct RocmBuf adv;
    struct RocmBuf ret;
    struct RocmBuf order;           // minibatch order (as floats) and its cursor
    struct RocmBuf cursor;
    struct RocmBuf mObs;            // the current minibatch
    struct RocmBuf mAct;
    struct RocmBuf mAdv;
    struct RocmBuf mRet;
    struct RocmBuf mLpo;
    struct RocmBuf state;           // [-, -, loss, entropy]
    struct RocmBuf obs1;
    &heap float   stage;
    struct RocmGraph graphs[2];     // a full and a short minibatch step
    unsigned long graphRows[2];
    float lastEntropy;
    int   ok;
};
// A device copy of a (networks, Rng state) taking rollouts of up to maxRollout steps.
struct RocmPolicy rocm_pg_new(const &PolicyAgent a, unsigned long maxRollout);
unsigned long rocm_pg_act(&RocmPolicy a, const &Tensor obs, int sample);   // waits for the logits
void  rocm_pg_record(&RocmPolicy a, const &Tensor obs, unsigned long action, float reward, const &Tensor next,
                     int terminated, int truncated);
// pg_update on the GPU; returns the last minibatch's loss (one read-back at
// the end), or -1 on failure (e.g. a rollout longer than maxRollout).
float rocm_pg_update(&RocmPolicy a);
int   rocm_pg_download(const &RocmPolicy d, &PolicyAgent a);   // networks and Rng state
void  rocm_pg_free(&RocmPolicy a);

// ── The training loop ────────────────────────────────────────────────────────
// A dataset in GPU memory and its batch cursor (rocm_trainer_fit's
// internals, usable for hand-written loops). xb / yb hold the current batch.
struct RocmBatches {
    struct RocmBuf x;         // [n, features]
    struct RocmBuf y;         // [n, targets]
    struct RocmBuf order;     // this epoch's row order (as floats)
    struct RocmBuf cursor;    // [2] first row of the next batch in 'order', and a block count
    struct RocmBuf xb;
    struct RocmBuf yb;
    struct RocmBuf loss;      // [1] this epoch's loss sum
    unsigned long n;
    unsigned long features;
    unsigned long targets;
    unsigned long batchSize;
    int   shuffle;
    struct Rng rng;
    &heap unsigned long perm;
    &heap float permf;
    int   ok;
};
struct RocmBatches rocm_batches_new(const &Dataset d, unsigned long batchSize, int shuffle, unsigned long seed);
// Starts an epoch: uploads the row order, rewinds the cursor, zeroes the loss.
int   rocm_batches_epoch(&RocmBatches b);
// Queued: xb, yb = the next 'rows' rows, and the cursor moves past them.
int   rocm_batches_gather(&RocmBatches b, unsigned long rows);
float rocm_batches_loss(&RocmBatches b);                          // waits; the epoch's loss sum
void  rocm_batches_free(&RocmBatches b);

// Internal: one queued step (gather, model step, optimizer).
generic<T>
int rocm_trainer_step_(&T model, fn int(&T, struct RocmBuf, struct RocmBuf, unsigned long, struct RocmBuf) stepFn,
                       &RocmOptimizer opt, &RocmBatches bt, unsigned long rows) {
    if (!rocm_batches_gather(bt, rows)) return 0;
    if (!stepFn(model, bt.xb, bt.yb, rows, bt.loss)) return 0;
    return rocm_optimizer_step(opt);
}

// Internal: runs 'count' steps of 'rows' rows each: eagerly the first time
// a batch size is seen, then recorded into *g once and replayed (the batch
// cursor and the step count advance on the GPU, so consecutive steps are
// the same work).
generic<T>
int rocm_trainer_run_(&T model, fn int(&T, struct RocmBuf, struct RocmBuf, unsigned long, struct RocmBuf) stepFn,
                      &RocmOptimizer opt, &RocmBatches bt, unsigned long rows, unsigned long count,
                      &RocmGraph g, &int seen) {
    unsigned long t0 = opt.t;
    int ok = 1;
    unsigned long i = 0UL;
    if (g.exec != 0UL) {
        ok = rocm_graph_launch(*g);
    } else if (*seen) {
        rocm_optimizer_sync(opt);
        if (rocm_capture_begin()) {
            int rec = 1;
            while (rec && i < count) { rec = rocm_trainer_step_(model, stepFn, opt, bt, rows); i = i + 1UL; }
            *g = rocm_capture_end();
            ok = rec && g.exec != 0UL && rocm_graph_launch(*g);
        } else {
            while (ok && i < count) { ok = rocm_trainer_step_(model, stepFn, opt, bt, rows); i = i + 1UL; }
        }
    } else {
        while (ok && i < count) { ok = rocm_trainer_step_(model, stepFn, opt, bt, rows); i = i + 1UL; }
        *seen = 1;
    }
    opt.t = t0 + count;
    return ok;
}

// Full batches recorded into one graph: a graph launch costs ~7 us on an
// MI100 however small, a step inside one ~1.3 us per kernel.
#ifndef ROCM_TRAIN_CHUNK
#define ROCM_TRAIN_CHUNK 4UL
#endif

// Internal: the graphs rocm_trainer_fit records (a chunk of full batches,
// one full batch, the short last batch) and whether each batch size has had
// its eager first step. They stay valid while the RocmBatches they read do.
struct RocmFitGraphs {
    struct RocmGraph full;
    struct RocmGraph tail;
    struct RocmGraph chunk;
    int seenFull;
    int seenTail;
};
struct RocmFitGraphs rocm_fit_graphs_new_();
void rocm_fit_graphs_free_(&RocmFitGraphs g);

// Internal: one epoch of rocm_trainer_fit (queued; the loss sum stays in
// bt.loss). Sets *batches; 0 on a GPU failure.
generic<T>
int rocm_trainer_epoch_(&T model, fn int(&T, struct RocmBuf, struct RocmBuf, unsigned long, struct RocmBuf) stepFn,
                        &RocmOptimizer opt, &RocmBatches bt, &RocmFitGraphs gs, &unsigned long batches) {
    int ok = rocm_batches_epoch(bt) && rocm_optimizer_sync(opt);
    unsigned long pos = 0UL;
    unsigned long n = 0UL;
    while (ok && pos < bt.n) {
        unsigned long rows = bt.n - pos;
        unsigned long count = 1UL;
        if (rows >= ROCM_TRAIN_CHUNK * bt.batchSize && gs.seenFull && ROCM_TRAIN_CHUNK > 1UL) {
            rows = bt.batchSize;
            count = ROCM_TRAIN_CHUNK;
            ok = rocm_trainer_run_(model, stepFn, opt, bt, rows, count, &gs.chunk, &gs.seenFull);
        } else if (rows >= bt.batchSize) {
            rows = bt.batchSize;
            ok = rocm_trainer_run_(model, stepFn, opt, bt, rows, 1UL, &gs.full, &gs.seenFull);
        } else {
            ok = rocm_trainer_run_(model, stepFn, opt, bt, rows, 1UL, &gs.tail, &gs.seenTail);
        }
        pos = pos + rows * count;
        n = n + count;
    }
    *batches = n;
    return ok;
}

// trainer_fit on the GPU. stepFn queues forward, loss and backward for
// 'rows' rows of xb / yb and adds the batch's loss to loss1[0]
// (rocm_net_ce_step, rocm_vae_step, ...). Same batches, epochs, lr decay
// and report as trainer_fit; clipNorm sets opt.clipNorm. A run that fails
// on the GPU reports steps == 0. RocmTrainer (below) is the same loop
// behind train_traits.h's Trainable.
generic<T>
struct TrainReport rocm_trainer_fit(&T model,
                                    fn int(&T, struct RocmBuf, struct RocmBuf, unsigned long, struct RocmBuf) stepFn,
                                    &RocmOptimizer opt, const &Dataset data, const &TrainConfig cfg) {
    struct TrainReport rep = train_report_new_();
    struct RocmBatches bt = rocm_batches_new(data, cfg.batchSize, cfg.shuffle, cfg.seed);
    if (!bt.ok || !opt.ok) { rocm_batches_free(&bt); return rep; }
    opt.clipNorm = cfg.clipNorm;
    struct RocmFitGraphs gs = rocm_fit_graphs_new_();
    int ok = 1;
    unsigned long steps = 0UL;
    unsigned long e = 0UL;
    while (ok && e < cfg.epochs) {
        unsigned long batches = 0UL;
        ok = rocm_trainer_epoch_(model, stepFn, opt, &bt, &gs, &batches);
        float sum = rocm_batches_loss(&bt);
        train_report_epoch_(&rep, e, batches > 0UL ? sum / (float)batches : (float)0.0, batches);
        steps = steps + batches;
        opt.lr = opt.lr * cfg.lrDecay;
        e = e + 1UL;
    }
    rep.steps = ok ? steps : 0UL;
    rocm_fit_graphs_free_(&gs);
    rocm_batches_free(&bt);
    return rep;
}

// ── The training traits (train_traits.h) ─────────────────────────────────────
// The device models above behind the backend-neutral interface; train.h
// names them without the Rocm prefix when the build enables the rocm
// feature. Each owns its host model in 'host' and a device copy in 'dev';
// sync() downloads 'dev' into 'host' (train_fit and train_gan do so at the
// end). Graph replay, the GPU-side optimizer state and the device-resident
// dataset are those of rocm_trainer_fit, rocm_gan_train and the RL agents.

// A host model H trained as a device model D (RocmNet, RocmVae, RocmConvAe):
// Trainable. 'ok' is 0 once building or uploading failed.
generic<H, D>
struct RocmTrainer {
    H host;
    D dev;
    struct RocmOptimizer opt;
    fn int(&D, struct RocmBuf, struct RocmBuf, unsigned long, struct RocmBuf) stepFn;
    fn int(const &D, &H) downloadFn;
    fn void(&D) freeFn;
    unsigned long maxBatch;
    struct RocmBatches bt;        // between fit_begin and fit_end
    struct RocmFitGraphs graphs;
    struct RocmBuf xs;            // train_step's batch and loss
    struct RocmBuf ys;
    struct RocmBuf loss1;
    int active;
    int ok;

    int   fit_begin(const &Dataset data, const &TrainConfig cfg);
    struct EpochLoss fit_epoch();
    void  fit_end();
    float train_step(const &Tensor x, const &Tensor y);   // -1 on failure
    float learning_rate();
    void  set_learning_rate(float lr);
    int   sync();
    void  free();
};

// Internal: b holds at least n floats (reallocated when smaller).
int rocm_trainer_reserve_(&RocmBuf b, unsigned long n);

generic<H, D>
int RocmTrainer::fit_begin(const &Dataset data, const &TrainConfig cfg) {
    self.fit_end();
    if (!self.ok || !self.opt.ok) return 0;
    self.bt = rocm_batches_new(data, cfg.batchSize, cfg.shuffle, cfg.seed);
    self.graphs = rocm_fit_graphs_new_();
    self.active = 1;
    if (!self.bt.ok || self.bt.batchSize > self.maxBatch) { self.fit_end(); return 0; }
    self.opt.clipNorm = cfg.clipNorm;
    return 1;
}

generic<H, D>
struct EpochLoss RocmTrainer::fit_epoch() {
    struct EpochLoss el;
    el.steps = 0UL;
    el.loss = (float)0.0;
    el.ok = 0;
    if (!self.active) return el;
    el.ok = rocm_trainer_epoch_(&self.dev, self.stepFn, &self.opt, &self.bt, &self.graphs, &el.steps);
    float sum = rocm_batches_loss(&self.bt);
    if (el.steps > 0UL) el.loss = sum / (float)el.steps;
    return el;
}

generic<H, D>
void RocmTrainer::fit_end() {
    if (!self.active) return;
    rocm_fit_graphs_free_(&self.graphs);
    rocm_batches_free(&self.bt);
    self.active = 0;
}

generic<H, D>
float RocmTrainer::train_step(const &Tensor x, const &Tensor y) {
    unsigned long rows = 0UL;
    unsigned long nx = 0UL;
    unsigned long ny = 0UL;
    unsafe { rows = x->shape[0]; nx = x->size; ny = y->size; }
    if (!self.ok || rows == 0UL || rows > self.maxBatch || !rocm_trainer_reserve_(&self.xs, nx) ||
        !rocm_trainer_reserve_(&self.ys, ny) || !rocm_trainer_reserve_(&self.loss1, 1UL)) return (float)-1.0;
    int ok = 0;
    unsafe {
        ok = rocm_write(self.xs, (const &float)(const float*)x->data, nx) &&
             rocm_write(self.ys, (const &float)(const float*)y->data, ny);
    }
    ok = ok && rocm_zero(self.loss1) && self.stepFn(&self.dev, self.xs, self.ys, rows, self.loss1) &&
         rocm_optimizer_step(&self.opt);
    float l = (float)-1.0;
    if (ok && !rocm_read(self.loss1, &l, 1UL)) l = (float)-1.0;
    return l;
}

generic<H, D>
float RocmTrainer::learning_rate() { return self.opt.lr; }

generic<H, D>
void RocmTrainer::set_learning_rate(float lr) { self.opt.lr = lr; }

generic<H, D>
int RocmTrainer::sync() { return self.ok && self.downloadFn(&self.dev, &self.host); }

generic<H, D>
void RocmTrainer::free() {
    self.fit_end();
    rocm_optimizer_free(&self.opt);
    self.freeFn(&self.dev);
    rocm_free(&self.xs);
    rocm_free(&self.ys);
    rocm_free(&self.loss1);
    self.ok = 0;
}

// 'opt' supplies the optimizer's kind and hyperparameters
// (rocm_optimizer_from); o.maxBatch sizes the device buffers. loss:
// TRAIN_LOSS_* (rocm_net_ce_step, _mse_step, _bce_step). o.precision
// (rocm_net_set_precision) applies to Mlp and ConvNet trainers; the others
// fail to build (ok = 0) when it is set.
struct RocmTrainer<struct Mlp, struct RocmNet> rocm_mlp_trainer_new(struct Mlp m, int loss, const &Optimizer opt,
                                                                    const &TrainerOptions o);
struct RocmTrainer<struct ConvNet, struct RocmNet> rocm_convnet_trainer_new(struct ConvNet n, int loss,
                                                                            const &Optimizer opt,
                                                                            const &TrainerOptions o);
struct RocmTrainer<struct Vae, struct RocmVae> rocm_vae_trainer_new(struct Vae v, const &Optimizer opt,
                                                                    const &TrainerOptions o);
struct RocmTrainer<struct ConvVae, struct RocmVae> rocm_conv_vae_trainer_new(struct ConvVae v, const &Optimizer opt,
                                                                             const &TrainerOptions o);
struct RocmTrainer<struct ConvAutoencoder, struct RocmConvAe> rocm_conv_ae_trainer_new(struct ConvAutoencoder ae,
                                                                                       const &Optimizer opt,
                                                                                       const &TrainerOptions o);
// An Mlp's parameters from its device net (rocm_net_download into m.params).
int rocm_net_download_mlp(const &RocmNet n, &Mlp m);

// A Gan or Dcgan (H) as a RocmGan: GanTrainable. train is rocm_gan_train;
// train_step uploads the batch, runs rocm_gan_step and reads the losses.
generic<H>
struct RocmGanTrainer {
    H host;
    struct RocmGan dev;
    fn int(const &RocmGan, &H) downloadFn;
    struct RocmBuf xs;            // train_step's batch and labels
    struct RocmBuf ls;

    struct GanLosses train_step(const &Tensor real, ?&Tensor labels);
    struct GanLosses train(const &Dataset data, unsigned long epochs, unsigned long batchSize);
    int  sync();
    void free();
};

generic<H>
struct GanLosses RocmGanTrainer::train_step(const &Tensor real, ?&Tensor labels) {
    struct GanLosses l;
    l.d = (float)0.0;
    l.g = (float)0.0;
    unsigned long rows = 0UL;
    unsigned long n = 0UL;
    unsafe { rows = real->shape[0]; n = real->size; }
    if (!rocm_trainer_reserve_(&self.xs, n) || !rocm_trainer_reserve_(&self.ls, rows)) return l;
    int ok = 0;
    unsafe { ok = rocm_write(self.xs, (const &float)(const float*)real->data, n); }
    match (labels) {
        case some(y): { unsafe { ok = ok && rocm_write(self.ls, (const &float)(const float*)y->data, rows); } }
        case null: { ok = ok && rocm_zero(self.ls); }
    }
    if (ok && rocm_gan_step(&self.dev, self.xs, self.ls, rows)) l = rocm_gan_losses(&self.dev);
    return l;
}

generic<H>
struct GanLosses RocmGanTrainer::train(const &Dataset data, unsigned long epochs, unsigned long batchSize) {
    return rocm_gan_train(&self.dev, data, epochs, batchSize);
}

generic<H>
int RocmGanTrainer::sync() { return self.downloadFn(&self.dev, &self.host); }

generic<H>
void RocmGanTrainer::free() {
    rocm_gan_free(&self.dev);
    rocm_free(&self.xs);
    rocm_free(&self.ls);
}

// o.maxBatch: the largest batch train / train_step will see.
struct RocmGanTrainer<struct Gan> rocm_gan_trainer_new(struct Gan g, const &TrainerOptions o);
struct RocmGanTrainer<struct Dcgan> rocm_dcgan_trainer_new(struct Dcgan g, const &TrainerOptions o);

// rl.h's Dqn as a RocmDqn: ValueAgent.
struct RocmDqnAgent {
    struct Dqn host;
    struct RocmDqn dev;

    float epsilon();
    unsigned long act(const &Tensor obs, int explore);
    int   observe(const &Tensor obs, unsigned long action, float reward, const &Tensor next, int terminal);
    float last_loss();             // waits
    unsigned long updates();
    int   sync();
    void  free();
};
struct RocmDqnAgent rocm_dqn_agent_new(struct Dqn d);

// rl.h's PolicyAgent as a RocmPolicy: PolicyLearner.
struct RocmPgAgent {
    struct PolicyAgent host;
    struct RocmPolicy dev;

    unsigned long act(const &Tensor obs, int sample);
    void  record(const &Tensor obs, unsigned long action, float reward, const &Tensor next, int terminated,
                 int truncated);
    float update();
    int   sync();
    void  free();
};
// o.maxBatch: the longest rollout (rocm_pg_new's maxRollout).
struct RocmPgAgent rocm_pg_agent_new(struct PolicyAgent a, const &TrainerOptions o);

} // namespace std
