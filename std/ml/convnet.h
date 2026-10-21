#pragma once
// SafeC Standard Library — trainable convolutional networks on std::ml's
// autograd core.
//
// cnn.h runs convolutions forward on FeatureMaps; this file makes them
// differentiable on Tensors so they train with tensor_backward and the
// optimizers in nn.h, alongside Mlp. Images travel as rows, the same as
// every other model in std::ml: a batch is a [batch, C*H*W] tensor, each
// row one image in CHW order (row[c*H*W + h*W + w]). The geometry (C, H,
// W) is passed to every op explicitly, since a 2D tensor does not carry it.
//
//   * tensor_conv2d / tensor_conv2d_ex — convolution with stride, zero
//     padding, dilation and groups (depthwise when groups == channels),
//     weight [outC, (inC/groups)*kH*kW] (a PyTorch Conv2d weight,
//     flattened), bias [outC].
//   * tensor_conv_transpose2d / _ex — its transpose (the gradient of conv2d
//     with respect to its input, run forward), weight [inC,
//     (outC/groups)*kH*kW] (a PyTorch ConvTranspose2d weight, flattened),
//     bias [outC], with output padding to reach every output size. The
//     usual learned upsampler of decoders and generators.
//   * Pooling: max / avg with any window, stride, padding and ceil mode;
//     global average / max.
//   * Resizing: nearest (integer factor, fast path), and nearest / bilinear
//     / bicubic to any size, with or without aligned corners.
//   * Normalization: batch norm with running statistics; group norm (layer
//     norm and instance norm as its two ends); spectral norm of a weight.
//   * Dropout, elementwise or by whole channels.
//   * ConvNet — a stack of those layers plus fully connected ones, with
//     saved activations for residual (add) and U-Net (concat) skips, built
//     layer by layer and parallel to Mlp. autoencoder.h (ConvAutoencoder,
//     ConvVae) and gan.h (Dcgan) take ConvNets as their encoders,
//     decoders, generators and discriminators.
//
// Output sides along each axis: conv2d (dim + 2*pad - dil*(k-1) - 1)/stride
// + 1 (rounded down, as PyTorch); conv_transpose2d (dim - 1)*stride - 2*pad
// + dil*(k-1) + outPad + 1; pooling as conv2d with dilation 1, rounded up
// in ceil mode. Both convolutions compute through im2col + sgemm (3x3
// stride-1 forward passes with 8+ channels through cnn.h's Winograd
// kernel); a large enough batch is split across std/sync/parallel.h's
// thread pool, one chunk of images per thread, with results independent of
// scheduling. Pooling, resizing and the norms split large inputs across the
// pool by plane, row or channel the same way.
#include <std/ml/tensor.h>
#include <std/ml/nn.h>
#include <std/collections/vec.h>

namespace std {

// ── Convolution ──────────────────────────────────────────────────────────────
// Every field per axis; conv2d_spec fills a square kernel with dilation 1,
// 1 group and no output padding. inC and outC must divide by groups;
// outPad (transposed convolution only) must be below stride or dilation.
struct Conv2dSpec {
    unsigned long kH;
    unsigned long kW;
    unsigned long strideH;
    unsigned long strideW;
    unsigned long padH;
    unsigned long padW;
    unsigned long dilationH;
    unsigned long dilationW;
    unsigned long groups;
    unsigned long outPadH;
    unsigned long outPadW;
};
struct Conv2dSpec conv2d_spec(unsigned long kernel, unsigned long stride, unsigned long padding);

// x is [batch, inC*inH*inW] (a 1D tensor is one image). Each returns
// [batch, outC*outH*outW] and records its backward edge to x, weight and
// bias, like tensor_add does. outC is bias's size.
&Tensor tensor_conv2d_ex(const &Tensor x, const &Tensor weight, const &Tensor bias,
                         unsigned long inC, unsigned long inH, unsigned long inW, const &Conv2dSpec spec);
&Tensor tensor_conv_transpose2d_ex(const &Tensor x, const &Tensor weight, const &Tensor bias,
                                   unsigned long inC, unsigned long inH, unsigned long inW, const &Conv2dSpec spec);
// The common case: kH x kW kernel, one stride and padding for both axes.
&Tensor tensor_conv2d(const &Tensor x, const &Tensor weight, const &Tensor bias,
                      unsigned long inC, unsigned long inH, unsigned long inW,
                      unsigned long kH, unsigned long kW,
                      unsigned long stride, unsigned long padding);
&Tensor tensor_conv_transpose2d(const &Tensor x, const &Tensor weight, const &Tensor bias,
                                unsigned long inC, unsigned long inH, unsigned long inW,
                                unsigned long kH, unsigned long kW,
                                unsigned long stride, unsigned long padding);

// Output side along one axis.
unsigned long conv_out_side(unsigned long dim, unsigned long kernel, unsigned long stride, unsigned long padding);
unsigned long conv_transpose_out_side(unsigned long dim, unsigned long kernel, unsigned long stride, unsigned long padding);
unsigned long conv_out_side_ex(unsigned long dim, unsigned long kernel, unsigned long stride,
                               unsigned long padding, unsigned long dilation);
unsigned long conv_transpose_out_side_ex(unsigned long dim, unsigned long kernel, unsigned long stride,
                                         unsigned long padding, unsigned long dilation, unsigned long outPad);

// ── Pooling ──────────────────────────────────────────────────────────────────
// Windows are clipped to the plane: max pooling ignores the padding (as
// PyTorch's MaxPool2d does) and sends each window's gradient to its
// (first) largest input. Avg pooling with countIncludePad divides by the
// window's extent inside the padded plane (PyTorch's AvgPool2d default),
// else by the positions inside the plane. padH / padW at most half the
// window. ceilMode rounds the output side up, as PyTorch's ceil_mode (a
// last window must still start inside the input or its left padding).
struct Pool2dSpec {
    unsigned long kH;
    unsigned long kW;
    unsigned long strideH;
    unsigned long strideW;
    unsigned long padH;
    unsigned long padW;
    int ceilMode;
    int countIncludePad;           // avg pooling only
};
// Square window; floor mode, padding counted.
struct Pool2dSpec pool2d_spec(unsigned long kernel, unsigned long stride, unsigned long padding);
unsigned long pool_out_side(unsigned long dim, unsigned long kernel, unsigned long stride,
                            unsigned long padding, int ceilMode);
&Tensor tensor_maxpool2d_ex(const &Tensor x, unsigned long channels, unsigned long height, unsigned long width,
                            const &Pool2dSpec spec);
&Tensor tensor_avgpool2d_ex(const &Tensor x, unsigned long channels, unsigned long height, unsigned long width,
                            const &Pool2dSpec spec);
&Tensor tensor_maxpool2d(const &Tensor x, unsigned long channels, unsigned long height, unsigned long width,
                         unsigned long kernel, unsigned long stride, unsigned long padding);
&Tensor tensor_avgpool2d(const &Tensor x, unsigned long channels, unsigned long height, unsigned long width,
                         unsigned long kernel, unsigned long stride, unsigned long padding, int countIncludePad);
// Global pooling: each channel's whole plane to one value, [batch, C*H*W]
// -> [batch, C] (a classifier head, or the squeeze of squeeze-and-excite).
// Max sends the gradient to the plane's (first) largest input.
&Tensor tensor_global_avgpool2d(const &Tensor x, unsigned long channels, unsigned long height, unsigned long width);
&Tensor tensor_global_maxpool2d(const &Tensor x, unsigned long channels, unsigned long height, unsigned long width);

// ── Resizing ─────────────────────────────────────────────────────────────────
// out[c][h*factor + i][w*factor + j] = in[c][h][w]; gradients sum back.
&Tensor tensor_upsample_nearest(const &Tensor x, unsigned long channels, unsigned long height, unsigned long width,
                                unsigned long factor);
// To any outH x outW (up or down), as PyTorch's interpolate(size=...):
//   RESIZE_NEAREST   source floor(o * in / out) ('nearest'; alignCorners ignored)
//   RESIZE_BILINEAR  the four nearest pixels
//   RESIZE_BICUBIC   the sixteen nearest, cubic convolution with a = -0.75
// alignCorners maps the corner pixels' centres onto each other (source o *
// (in - 1) / (out - 1)); otherwise pixel areas line up (source (o + 0.5) *
// in / out - 0.5, clamped at 0 for bilinear). Edges are clamped. Gradients
// scatter back with the same weights.
#define RESIZE_NEAREST  0
#define RESIZE_BILINEAR 1
#define RESIZE_BICUBIC  2
&Tensor tensor_resize2d(const &Tensor x, unsigned long channels, unsigned long height, unsigned long width,
                        unsigned long outH, unsigned long outW, int mode, int alignCorners);
// tensor_resize2d to factor x the size, bilinear, corners not aligned.
&Tensor tensor_upsample_bilinear(const &Tensor x, unsigned long channels, unsigned long height, unsigned long width,
                                 unsigned long factor);

// ── Normalization and dropout ────────────────────────────────────────────────
// Batch normalization (Ioffe & Szegedy 2015) over x [batch, channels*plane]:
// each channel is normalized over the batch and its plane (plane = H*W for
// feature maps, 1 for [batch, features] rows), then scaled by gamma and
// shifted by beta ([channels] each, trained). When 'training', the batch's
// statistics are used and runMean / runVar ([channels], no gradient) move
// toward them by 'momentum' (PyTorch: 0.1, with the unbiased variance);
// otherwise the running statistics are used. eps is added to the variance.
&Tensor tensor_batch_norm(const &Tensor x, const &Tensor gamma, const &Tensor beta,
                          &Tensor runMean, &Tensor runVar,
                          unsigned long channels, unsigned long plane,
                          int training, float momentum, float eps);
// Group normalization (Wu & He 2018): each image's channels split into
// 'groups' groups, each normalized over its channels and planes, then
// gamma / beta per channel. Batch-independent, the same in training and
// inference. groups 1 is layer norm over (C, H, W); groups == channels is
// instance norm.
&Tensor tensor_group_norm(const &Tensor x, const &Tensor gamma, const &Tensor beta,
                          unsigned long channels, unsigned long plane, unsigned long groups, float eps);
// Spectral normalization (Miyato et al. 2018): weight / sigma, sigma its
// largest singular value (the weight seen as a matrix [rows, size/rows]),
// estimated by power iteration from u ([rows], no gradient). In training
// 'iters' iterations update u; in inference u is kept. The gradient treats
// u and v as constants, as PyTorch does.
&Tensor tensor_spectral_norm(const &Tensor weight, &Tensor u, unsigned long iters, int training);
// Inverted dropout: in training each value is zeroed with probability p
// and the rest scaled by 1 / (1 - p); in inference x passes unchanged.
// tensor_dropout2d drops whole channel planes (spatial dropout).
&Tensor tensor_dropout(const &Tensor x, float p, int training, &Rng rng);
&Tensor tensor_dropout2d(const &Tensor x, unsigned long channels, unsigned long plane, float p,
                         int training, &Rng rng);

// ── Conditioning ─────────────────────────────────────────────────────────────
// One constant plane per class, 1 in the plane of each row's label (m
// class indices, stored as floats) and 0 elsewhere -> [m,
// classes*height*width], no gradient. tensor_append_label_channels puts
// them after x's channels (x [m, C*height*width] -> [m, (C+classes)*...]):
// how a conditional ConvVae / Dcgan feeds the label to its image-side net.
&Tensor tensor_label_planes(const &Tensor labels, unsigned long classes, unsigned long height, unsigned long width);
&Tensor tensor_append_label_channels(const &Tensor x, const &Tensor labels, unsigned long classes,
                                     unsigned long height, unsigned long width);

// ── ConvNet ──────────────────────────────────────────────────────────────────
#define CONV_LAYER_CONV            0
#define CONV_LAYER_DECONV          1   // transposed convolution
#define CONV_LAYER_MAXPOOL         2
#define CONV_LAYER_AVGPOOL         3
#define CONV_LAYER_UPSAMPLE        4   // nearest, by 'factor'
#define CONV_LAYER_DENSE           5   // fully connected over the flattened map -> outC x 1 x 1
#define CONV_LAYER_RESHAPE         6   // reinterpret the geometry (same size), no data change
#define CONV_LAYER_BATCHNORM       7
#define CONV_LAYER_RESIZE          8   // tensor_resize2d to outH x outW ('mode', 'flag' = alignCorners)
#define CONV_LAYER_GLOBAL_AVGPOOL  9   // -> C x 1 x 1
#define CONV_LAYER_GLOBAL_MAXPOOL  10
#define CONV_LAYER_GROUPNORM       11  // 'factor' groups
#define CONV_LAYER_DROPOUT         12  // 'prob', 'flag' = by channel
#define CONV_LAYER_SAVE            13  // keep the activation in slot 'slot'
#define CONV_LAYER_ADD             14  // + slot 'slot' (through a 1x1 projection when 'flag')
#define CONV_LAYER_CONCAT          15  // channel concat with slot 'slot'
#define CONV_LAYER_IDENTITY        16  // no-op (a folded batch norm)

struct ConvLayer {
    int kind;                    // CONV_LAYER_*
    int act;                     // ACT_* after the layer
    unsigned long inC;
    unsigned long inH;
    unsigned long inW;
    unsigned long outC;
    unsigned long outH;
    unsigned long outW;
    struct Conv2dSpec conv;      // CONV / DECONV, and ADD's projection
    struct Pool2dSpec pool;      // MAXPOOL / AVGPOOL
    unsigned long factor;        // UPSAMPLE factor, GROUPNORM groups
    int   mode;                  // RESIZE mode
    int   flag;                  // RESIZE alignCorners, DROPOUT by channel, ADD projection
    float prob;                  // DROPOUT
    unsigned long slot;          // SAVE / ADD / CONCAT
    unsigned long skipC;         // ADD / CONCAT: the saved activation's geometry
    unsigned long skipH;
    unsigned long skipW;
    unsigned long param;         // first trained tensor's index in params (weight / gamma; bias / beta next)
    unsigned long buffer;        // BATCHNORM running mean (variance next), spectral norm's u, DROPOUT's rng
    int   spectral;              // CONV / DECONV / DENSE: weight spectrally normalized (u in buffers[buffer])
};

// Start with the input geometry, then append layers; each one's input is
// the previous one's output. Weights are initialized like PyTorch's:
// U(-1/sqrt(fanIn), 1/sqrt(fanIn)) for weights and biases; the norms start
// at gamma 1, beta 0. params holds every layer's trained tensors, all
// requiring grad, on the heap; buffers holds what is not trained (batch
// norms' running statistics, spectral norms' u). 'training' (1 from
// convnet_new) picks batch statistics in batch norm and updates the running
// ones, iterates spectral norms and applies dropout; set it to 0 for
// inference with convnet_set_training. The models in autoencoder.h and
// gan.h do that in their inference functions (encode / decode / sample /
// generate / score).
//
//   struct ConvNet net = convnet_new(1UL, 28UL, 28UL);
//   convnet_conv(&net, 32UL, 3UL, 1UL, 1UL, ACT_NONE, &rng);    // 32 x 28 x 28
//   convnet_batchnorm(&net, ACT_RELU);
//   convnet_residual_block(&net, 64UL, 2UL, 1, ACT_RELU, &rng); // 64 x 14 x 14
//   convnet_global_avgpool(&net);                               // 64 x 1 x 1
//   convnet_dense(&net, 10UL, ACT_NONE, &rng);                  // logits
//   &Tensor logits = convnet_forward(&net, batch);              // [m, 784] -> [m, 10]
struct ConvNet {
    struct Vec<struct ConvLayer> layers;
    struct Vec<&Tensor> params;
    struct Vec<&Tensor> buffers;  // not trained by the optimizer
    struct Vec<struct Rng> rngs;  // one per dropout layer
    unsigned long slots;          // saved activations (CONV_LAYER_SAVE)
    int training;
    unsigned long inC;
    unsigned long inH;
    unsigned long inW;
    unsigned long outC;           // the last layer's output geometry
    unsigned long outH;
    unsigned long outW;
};
struct ConvNet convnet_new(unsigned long channels, unsigned long height, unsigned long width);
void convnet_conv(&ConvNet n, unsigned long outC, unsigned long kernel, unsigned long stride,
                  unsigned long padding, int act, &Rng rng);
void convnet_conv_ex(&ConvNet n, unsigned long outC, const &Conv2dSpec spec, int act, &Rng rng);
void convnet_deconv(&ConvNet n, unsigned long outC, unsigned long kernel, unsigned long stride,
                    unsigned long padding, int act, &Rng rng);
void convnet_deconv_ex(&ConvNet n, unsigned long outC, const &Conv2dSpec spec, int act, &Rng rng);
void convnet_maxpool(&ConvNet n, unsigned long kernel, unsigned long stride, unsigned long padding);
void convnet_avgpool(&ConvNet n, unsigned long kernel, unsigned long stride, unsigned long padding,
                     int countIncludePad);
void convnet_maxpool_ex(&ConvNet n, const &Pool2dSpec spec);
void convnet_avgpool_ex(&ConvNet n, const &Pool2dSpec spec);
void convnet_global_avgpool(&ConvNet n);                             // -> C x 1 x 1
void convnet_global_maxpool(&ConvNet n);
void convnet_upsample(&ConvNet n, unsigned long factor);            // nearest, integer factor
void convnet_upsample_bilinear(&ConvNet n, unsigned long factor);
void convnet_resize(&ConvNet n, unsigned long outH, unsigned long outW, int mode, int alignCorners);
// Norms over the current output's channels, then 'act': place them between
// a convolution (ACT_NONE) and its activation. Batch norm: momentum 0.1,
// eps 1e-5. Group norm: eps 1e-5; layer norm is 1 group, instance norm one
// group per channel.
void convnet_batchnorm(&ConvNet n, int act);
void convnet_groupnorm(&ConvNet n, unsigned long groups, int act);
void convnet_layernorm(&ConvNet n, int act);
void convnet_instancenorm(&ConvNet n, int act);
// Spectral normalization of the last CONV / DECONV / DENSE layer's weight
// (one power iteration per training forward pass): the usual way to keep a
// GAN critic Lipschitz.
void convnet_spectral_norm(&ConvNet n, &Rng rng);
// Dropout with probability p (byChannel: whole planes); its random stream
// is seeded from rng.
void convnet_dropout(&ConvNet n, float p, int byChannel, &Rng rng);
void convnet_dense(&ConvNet n, unsigned long outDim, int act, &Rng rng);
// channels*height*width must equal the current output size.
void convnet_reshape(&ConvNet n, unsigned long channels, unsigned long height, unsigned long width);

// Skips. convnet_save keeps the current activation and returns its slot.
// convnet_add_skip adds a slot to the current output, then 'act': through a
// learned 1x1 convolution (stride = the size ratio) when the channels or
// size differ -- ResNet's projection shortcut. convnet_concat_skip appends
// a slot's channels (same height and width) to the current output -- a
// U-Net skip. convnet_residual_block is ResNet's basic block: conv 3x3
// (stride) -> [batch norm] -> act -> conv 3x3 -> [batch norm] -> + shortcut
// -> act. Output side (dim - 1)/stride + 1.
unsigned long convnet_save(&ConvNet n);
void convnet_add_skip(&ConvNet n, unsigned long slot, int act, &Rng rng);
void convnet_concat_skip(&ConvNet n, unsigned long slot);
void convnet_residual_block(&ConvNet n, unsigned long outC, unsigned long stride, int batchNorm, int act, &Rng rng);

// An activation on the last layer's output (replaces the one it had).
void convnet_set_out_act(&ConvNet n, int act);
// Training (1) or inference (0) mode; returns the previous mode.
int  convnet_set_training(&ConvNet n, int training);
// For inference: folds every batch norm that directly follows a CONV /
// DECONV / DENSE layer (no activation or spectral norm between) into that
// layer's weight and bias, using the running statistics, and turns the
// batch norm into a no-op. Inference output is unchanged; do not train the
// net afterwards. Returns how many were folded.
unsigned long convnet_fold_batchnorm(&ConvNet n);

unsigned long convnet_in_dim(const &ConvNet n);    // inC*inH*inW
unsigned long convnet_out_dim(const &ConvNet n);   // outC*outH*outW
&Tensor convnet_forward(const &ConvNet n, const &Tensor x);   // [batch, in_dim] -> [batch, out_dim]
unsigned long convnet_param_count(const &ConvNet n);
// Same layers, a copy of the parameters and buffers (e.g. a GAN's averaged
// generator).
struct ConvNet convnet_clone(const &ConvNet n);
// dst = src and dst = tau * src + (1 - tau) * dst, as for Mlp (buffers
// included).
void convnet_copy(&ConvNet dst, const &ConvNet src);
void convnet_soft_update(&ConvNet dst, const &ConvNet src, float tau);
void optimizer_add_convnet(&Optimizer o, const &ConvNet n);
// Parameters and buffers to / from a file (tensors_save's format, nn.h).
// Loading needs a net built with the same layers; returns 1 on success, 0
// when the file is missing, unreadable or doesn't match.
int  convnet_save_file(const &ConvNet n, const &char path);
int  convnet_load_file(&ConvNet n, const &char path);
void convnet_free(&ConvNet n);

} // namespace std
