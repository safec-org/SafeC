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

    void free();
    int forward(struct RocmBuf x, struct RocmBuf y, unsigned long batch, unsigned long height, unsigned long width) const;
    // y = relu(conv(x) + bias).
    int forward_relu(struct RocmBuf x, struct RocmBuf y, unsigned long batch, unsigned long height,
                     unsigned long width) const;

    static struct RocmConv2D upload(const &Conv2D layer);
};

// Output side length of a convolution or pooling window: (dim + 2*padding - kernel)/stride + 1.
unsigned long rocm_conv_out_dim(unsigned long dim, unsigned long kernel,
                                unsigned long stride, unsigned long padding);

// ── Queued ops (1 = enqueued, 0 = bad size / no GPU / MIOpen error) ────────
// x: [batch, inChannels, height, width]; y: [batch, outChannels, outH, outW]
// with outH/outW from rocm_conv_out_dim. Same arithmetic as Conv2D::forward.

// kernel x kernel window, given stride, no padding (FeatureMap::maxpool2d /
// FeatureMap::avgpool2d). x: [batch, channels, height, width].
int rocm_maxpool2d(struct RocmBuf x, struct RocmBuf y, unsigned long batch, unsigned long channels,
                   unsigned long height, unsigned long width, unsigned long kernel, unsigned long stride);
int rocm_avgpool2d(struct RocmBuf x, struct RocmBuf y, unsigned long batch, unsigned long channels,
                   unsigned long height, unsigned long width, unsigned long kernel, unsigned long stride);

// ── fp16 / bf16 ──────────────────────────────────────────────────────────────
// The same convolution on 16-bit NCHW tensors (gpu_rocm.h's
// RocmHalfBuf): MIOpen's half / bfloat16 kernels, which accumulate in fp32
// and round once per output; the bias add and ReLU are one fused 16-bit
// kernel (rocm_half_bias). Weights are rounded from a Conv2D's fp32 ones on
// upload. x, y and the layer must share one dtype.
struct RocmHalfConv2D {
    struct RocmHalfBuf weight;
    struct RocmHalfBuf bias;
    unsigned long      inChannels;
    unsigned long      outChannels;
    unsigned long      kH;
    unsigned long      kW;
    unsigned long      stride;
    unsigned long      padding;

    void free();
    int forward(struct RocmHalfBuf x, struct RocmHalfBuf y, unsigned long batch, unsigned long height,
                unsigned long width) const;
    int forward_relu(struct RocmHalfBuf x, struct RocmHalfBuf y, unsigned long batch, unsigned long height,
                     unsigned long width) const;

    static struct RocmHalfConv2D upload(const &Conv2D layer, int dtype);   // ROCM_FP16 / ROCM_BF16
};
// 16-bit pooling is gpu_rocm.h's rocm_half_maxpool2d / rocm_half_avgpool2d
// (a HIP kernel: MIOpen has no bfloat16 pooling solver on every GPU).

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

    // y: [batch, outChannels, height, width]. Queued, like the ops above.
    int forward(struct RocmBuf x, struct RocmBuf y) const;
    void free();

    // height and width must each be divisible by 4.
    static struct RocmUNet new(const &UNet net, unsigned long batch, unsigned long height, unsigned long width);
};

// ── Host convenience (upload → compute → read back) ──────────────────────────
// Same signatures and results as the CPU functions; each falls back to the
// CPU implementation if there is no usable GPU. Every call uploads the
// weights — inside a loop, upload once and use the queued ops above.
struct FeatureMap conv2d_forward_rocm(const &Conv2D layer, const &FeatureMap input);
struct FeatureMap unet_forward_rocm(&UNet net, const &FeatureMap input);

} // namespace std
