#pragma once
// SafeC Standard Library — cuDNN convolution, pooling and U-Net on CUDA.
//
// The GPU counterpart of cnn.h / unet.h, layered on gpu_cuda.h: tensors are
// CudaBufs in NCHW order (cnn.h's CHW FeatureMap with a leading batch
// axis), and every op is queued on gpu_cuda's stream, so it orders with the
// CudaBuf ops there and can be recorded into a CudaGraph.
//
// Convolutions run through cuDNN. The first call for each shape (and for
// plain vs. +relu) builds the fused graph conv -> + bias (-> relu) with
// cuDNN's graph API, times the engine configs its heuristics suggest and
// the legacy API's best algorithm (cudnnFind…) on the call's buffers, and
// keeps the fastest, with its workspace, in a plan cache — so make that
// first call outside graph capture. Later calls cost a few microseconds of
// host time. fp32 convolutions run in full fp32 unless cuda_set_tf32(1) is in
// effect, which also lets cuDNN use TF32 tensor cores (same switch as the
// cuBLAS GEMMs).
// Link with: -lcudnn plus gpu_cuda's -lcuda -lcublas -lcublasLt -lnvrtc
// (cuDNN 9).
#include <std/ml/gpu_cuda.h>
#include <std/ml/cnn.h>
#include <std/ml/unet.h>

namespace std {

// A Conv2D (cnn.h) whose weight [outC, inC, kH, kW] and bias [outC] live on
// the GPU. weight.dev == 0 if the upload failed.
struct CudaConv2D {
    struct CudaBuf weight;
    struct CudaBuf bias;
    unsigned long  inChannels;
    unsigned long  outChannels;
    unsigned long  kH;
    unsigned long  kW;
    unsigned long  stride;
    unsigned long  padding;
};

struct CudaConv2D cuda_conv2d_upload(const &Conv2D layer);
void cuda_conv2d_free(&CudaConv2D layer);

// Output side length of a convolution or pooling window: (dim + 2*padding - kernel)/stride + 1.
unsigned long cuda_conv_out_dim(unsigned long dim, unsigned long kernel,
                                unsigned long stride, unsigned long padding);

// ── Queued ops (1 = enqueued, 0 = bad size / no GPU / cuDNN error) ─────────
// x: [batch, inChannels, height, width]; y: [batch, outChannels, outH, outW]
// with outH/outW from cuda_conv_out_dim. Same arithmetic as conv2d_forward.

int cuda_conv2d(const &CudaConv2D layer, struct CudaBuf x, struct CudaBuf y,
                unsigned long batch, unsigned long height, unsigned long width);
// y = relu(conv(x) + bias), fused.
int cuda_conv2d_relu(const &CudaConv2D layer, struct CudaBuf x, struct CudaBuf y,
                     unsigned long batch, unsigned long height, unsigned long width);

// kernel x kernel window, given stride, no padding (maxpool2d_forward /
// avgpool2d_forward). x: [batch, channels, height, width].
int cuda_maxpool2d(struct CudaBuf x, struct CudaBuf y, unsigned long batch, unsigned long channels,
                   unsigned long height, unsigned long width, unsigned long kernel, unsigned long stride);
int cuda_avgpool2d(struct CudaBuf x, struct CudaBuf y, unsigned long batch, unsigned long channels,
                   unsigned long height, unsigned long width, unsigned long kernel, unsigned long stride);

// ── fp16 / bf16 ──────────────────────────────────────────────────────────────
// The same convolution on 16-bit NCHW tensors (gpu_cuda.h's CudaHalfBuf):
// cuDNN's half / bfloat16 convolutions (fp32 compute, tensor cores where
// cuDNN picks them), rounding once per output; the bias add and ReLU are
// one fused 16-bit kernel (cuda_half_bias). Weights are rounded from a Conv2D's fp32 ones on
// upload. x, y and the layer must share one dtype.
struct CudaHalfConv2D {
    struct CudaHalfBuf weight;
    struct CudaHalfBuf bias;
    unsigned long      inChannels;
    unsigned long      outChannels;
    unsigned long      kH;
    unsigned long      kW;
    unsigned long      stride;
    unsigned long      padding;
};
struct CudaHalfConv2D cuda_half_conv2d_upload(const &Conv2D layer, int dtype);   // CUDA_FP16 / CUDA_BF16
void cuda_half_conv2d_free(&CudaHalfConv2D layer);
int cuda_half_conv2d(const &CudaHalfConv2D layer, struct CudaHalfBuf x, struct CudaHalfBuf y,
                     unsigned long batch, unsigned long height, unsigned long width);
int cuda_half_conv2d_relu(const &CudaHalfConv2D layer, struct CudaHalfBuf x, struct CudaHalfBuf y,
                          unsigned long batch, unsigned long height, unsigned long width);
// 16-bit pooling is gpu_cuda.h's cuda_half_maxpool2d / cuda_half_avgpool2d
// (a CUDA kernel, the same as on ROCm).

// out[b] = (a[b], c[b]) along channels, for each of 'batch' images whose
// planes are 'plane' = height*width elements (concat_channels).
int cuda_concat_channels(struct CudaBuf a, unsigned long aChannels, struct CudaBuf c, unsigned long cChannels,
                         struct CudaBuf out, unsigned long batch, unsigned long plane);

// ── U-Net (unet.h) on the GPU ────────────────────────────────────────────────
// Weights plus every intermediate activation, sized once for a fixed
// batch x height x width, so a forward pass allocates nothing.
struct CudaUNet {
    struct CudaConv2D encConv1;
    struct CudaConv2D encConv2;
    struct CudaConv2D bottleneck;
    struct CudaConv2D decConv1;
    struct CudaConv2D decConv2;
    struct CudaBuf e1;     // [B, c1, H, W]
    struct CudaBuf p1;     // [B, c1, H/2, W/2]
    struct CudaBuf e2;     // [B, c2, H/2, W/2]
    struct CudaBuf p2;     // [B, c2, H/4, W/4]
    struct CudaBuf mid;    // [B, c2, H/4, W/4]
    struct CudaBuf up;     // [B, c2, H/2, W/2], reused for [B, c1, H, W]
    struct CudaBuf cat1;   // [B, 2*c2, H/2, W/2]
    struct CudaBuf d1;     // [B, c1, H/2, W/2]
    struct CudaBuf cat2;   // [B, 2*c1, H, W]
    unsigned long  batch;
    unsigned long  height;
    unsigned long  width;
    int            ok;     // 0 if an upload or allocation failed
};

// height and width must each be divisible by 4.
struct CudaUNet cuda_unet_new(const &UNet net, unsigned long batch, unsigned long height, unsigned long width);
// y: [batch, outChannels, height, width]. Queued, like the ops above.
int cuda_unet_forward(const &CudaUNet net, struct CudaBuf x, struct CudaBuf y);
void cuda_unet_free(&CudaUNet net);

// ── Host convenience (upload → compute → read back) ──────────────────────────
// Same signatures and results as the CPU functions; each falls back to the
// CPU implementation if there is no usable GPU. Every call uploads the
// weights — inside a loop, upload once and use the queued ops above.
struct FeatureMap conv2d_forward_cuda(const &Conv2D layer, const &FeatureMap input);
struct FeatureMap unet_forward_cuda(&UNet net, const &FeatureMap input);

} // namespace std
