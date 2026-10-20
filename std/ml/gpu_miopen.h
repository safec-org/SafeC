#pragma once
// SafeC Standard Library — MIOpen convolution, pooling and U-Net on ROCm.
//
// The AMD counterpart of gpu_cudnn.h (same API with rocm_ in place of
// cuda_), layered on gpu_rocm.h: tensors are RocmBufs in NCHW order (cnn.h's
// CHW FeatureMap with a leading batch axis), and every op is queued on
// gpu_rocm's stream, so it orders with the RocmBuf ops there and can be
// recorded into a RocmGraph.
//
// Convolutions run through MIOpen. The first call for each shape asks
// MIOpen to time its algorithms on that call's buffers
// (miopenFindConvolutionForwardAlgorithm) and keeps the fastest, with its
// workspace, in a plan cache — so make that first call outside graph
// capture. The bias add and ReLU after a convolution are one fused HIP
// kernel (gpu_rocm.h's rocm_bias_channels).
// Link with: -lMIOpen plus gpu_rocm's -lamdhip64 -lhiprtc -lrocblas.
#include <std/ml/gpu_rocm.h>
#include <std/ml/cnn.h>
#include <std/ml/unet.h>

namespace std {

// A Conv2D (cnn.h) whose weight [outC, inC, kH, kW] and bias [outC] live on
// the GPU. weight.dev == 0 if the upload failed.
struct RocmConv2D {
    struct RocmBuf weight;
    struct RocmBuf bias;
    unsigned long  inChannels;
    unsigned long  outChannels;
    unsigned long  kH;
    unsigned long  kW;
    unsigned long  stride;
    unsigned long  padding;
};

struct RocmConv2D rocm_conv2d_upload(const &Conv2D layer);
void rocm_conv2d_free(&RocmConv2D layer);

// Output side length of a convolution or pooling window: (dim + 2*padding - kernel)/stride + 1.
unsigned long rocm_conv_out_dim(unsigned long dim, unsigned long kernel,
                                unsigned long stride, unsigned long padding);

// ── Queued ops (1 = enqueued, 0 = bad size / no GPU / MIOpen error) ────────
// x: [batch, inChannels, height, width]; y: [batch, outChannels, outH, outW]
// with outH/outW from rocm_conv_out_dim. Same arithmetic as conv2d_forward.

int rocm_conv2d(const &RocmConv2D layer, struct RocmBuf x, struct RocmBuf y,
                unsigned long batch, unsigned long height, unsigned long width);
// y = relu(conv(x) + bias).
int rocm_conv2d_relu(const &RocmConv2D layer, struct RocmBuf x, struct RocmBuf y,
                     unsigned long batch, unsigned long height, unsigned long width);

// kernel x kernel window, given stride, no padding (maxpool2d_forward /
// avgpool2d_forward). x: [batch, channels, height, width].
int rocm_maxpool2d(struct RocmBuf x, struct RocmBuf y, unsigned long batch, unsigned long channels,
                   unsigned long height, unsigned long width, unsigned long kernel, unsigned long stride);
int rocm_avgpool2d(struct RocmBuf x, struct RocmBuf y, unsigned long batch, unsigned long channels,
                   unsigned long height, unsigned long width, unsigned long kernel, unsigned long stride);

// out[b] = (a[b], c[b]) along channels, for each of 'batch' images whose
// planes are 'plane' = height*width elements (concat_channels).
int rocm_concat_channels(struct RocmBuf a, unsigned long aChannels, struct RocmBuf c, unsigned long cChannels,
                         struct RocmBuf out, unsigned long batch, unsigned long plane);

// ── U-Net (unet.h) on the GPU ────────────────────────────────────────────────
// Weights plus every intermediate activation, sized once for a fixed
// batch x height x width, so a forward pass allocates nothing. Each decoder
// step upsamples straight into its concat buffer (rocm_upsample2x_concat).
struct RocmUNet {
    struct RocmConv2D encConv1;
    struct RocmConv2D encConv2;
    struct RocmConv2D bottleneck;
    struct RocmConv2D decConv1;
    struct RocmConv2D decConv2;
    struct RocmBuf e1;     // [B, c1, H, W]
    struct RocmBuf p1;     // [B, c1, H/2, W/2]
    struct RocmBuf e2;     // [B, c2, H/2, W/2]
    struct RocmBuf p2;     // [B, c2, H/4, W/4]
    struct RocmBuf mid;    // [B, c2, H/4, W/4]
    struct RocmBuf cat1;   // [B, 2*c2, H/2, W/2]
    struct RocmBuf d1;     // [B, c1, H/2, W/2]
    struct RocmBuf cat2;   // [B, 2*c1, H, W]
    unsigned long  batch;
    unsigned long  height;
    unsigned long  width;
    int            ok;     // 0 if an upload or allocation failed
};

// height and width must each be divisible by 4.
struct RocmUNet rocm_unet_new(const &UNet net, unsigned long batch, unsigned long height, unsigned long width);
// y: [batch, outChannels, height, width]. Queued, like the ops above.
int rocm_unet_forward(const &RocmUNet net, struct RocmBuf x, struct RocmBuf y);
void rocm_unet_free(&RocmUNet net);

// ── Host convenience (upload → compute → read back) ──────────────────────────
// Same signatures and results as the CPU functions; each falls back to the
// CPU implementation if there is no usable GPU. Every call uploads the
// weights — inside a loop, upload once and use the queued ops above.
struct FeatureMap conv2d_forward_rocm(const &Conv2D layer, const &FeatureMap input);
struct FeatureMap unet_forward_rocm(&UNet net, const &FeatureMap input);

} // namespace std
