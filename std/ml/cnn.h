#pragma once
// SafeC Standard Library — CNN: 2D convolution + pooling over flat CHW
// feature maps.
//
// Feature maps are channels-first (CHW), flat row-major float32 buffers —
// deliberately not std::Tensor (which tops out at 2D and is wired for
// matmul-shaped autograd; convolution's 4D weight tensor and 3D
// activations don't fit that shape). Forward-only, like the rest of
// std/ml's newer additions (see attention.h). Conv2D supports stride and
// zero-padding; no dilation, no grouped/depthwise convolution.
#include <std/mem.h>

namespace std {

struct FeatureMap {
    &heap float   data;   // flat, channels*height*width, CHW order (data[c*H*W + h*W + w])
    unsigned long channels;
    unsigned long height;
    unsigned long width;
    int           fromArena; // 1 if data came from a TensorArena (FeatureMap::free leaves it)

    void free();
};

// Between TensorArena::begin and tensor_arena_end (std/ml/tensor.h), new
// feature maps -- the ones these functions return, and every layer's
// intermediate inside UNet::forward and the like -- come from the thread's
// TensorArena, as tensors do: no malloc (or page faults, for fresh memory)
// per layer. FeatureMap::free does nothing for them; TensorArena::reset
// releases them all, so don't use one after its arena is reset.
struct FeatureMap feature_map_new(unsigned long channels, unsigned long height, unsigned long width);

// out[oc][oh][ow] = bias[oc] + sum_ic sum_kh sum_kw
//                     weight[oc][ic][kh][kw] * in[ic][oh*stride+kh-padding][ow*stride+kw-padding]
// (input positions outside [0,height)x[0,width) — i.e. inside the zero
// padding — contribute 0). Output spatial size is
// (dim + 2*padding - kernel)/stride + 1; caller must choose dimensions
// that divide evenly (no implicit ceil/floor rounding surprises).
struct Conv2D {
    &heap float   weight; // [outChannels, inChannels, kH, kW]
    &heap float   bias;   // [outChannels]
    unsigned long inChannels;
    unsigned long outChannels;
    unsigned long kH;
    unsigned long kW;
    unsigned long stride;
    unsigned long padding;

    // Convolution + bias, computed as im2col + sgemm (std/simd/gemm.h): SIMD
    // on any target, and split across std/sync/parallel.h's thread pool when
    // the layer is large enough.
    struct FeatureMap forward(const &FeatureMap input) const;
    // relu(layer.forward(input)), in one pass.
    struct FeatureMap relu_forward(const &FeatureMap input) const;
    void free();
};

struct Conv2D conv2d_new(unsigned long inChannels, unsigned long outChannels,
                          unsigned long kH, unsigned long kW,
                          unsigned long stride, unsigned long padding);

// Max/avg pooling: kernel x kernel window, given stride, no padding.
// Channel count is preserved; caller must choose dimensions divisible by
// the kernel/stride combination.
struct FeatureMap maxpool2d_forward(const &FeatureMap input, unsigned long kernel, unsigned long stride);
struct FeatureMap avgpool2d_forward(const &FeatureMap input, unsigned long kernel, unsigned long stride);
// The same with 'padding' on every side; output side (dim + 2*padding -
// kernel)/stride + 1. Max pooling ignores the padded positions (as
// PyTorch's MaxPool2d does); avg pooling counts them as zeros in the
// divisor when 'countIncludePad' (PyTorch's AvgPool2d default) and
// divides by the positions inside the plane otherwise.
struct FeatureMap maxpool2d_forward_ex(const &FeatureMap input, unsigned long kernel, unsigned long stride,
                                       unsigned long padding);
struct FeatureMap avgpool2d_forward_ex(const &FeatureMap input, unsigned long kernel, unsigned long stride,
                                       unsigned long padding, int countIncludePad);
// Global pooling over each channel's whole plane: channels x 1 x 1 (the
// classifier head of ResNet-style networks).
struct FeatureMap global_avgpool2d(const &FeatureMap input);
struct FeatureMap global_maxpool2d(const &FeatureMap input);

// Residual connection: out = a + b, element by element (relu(a + b) when
// 'relu'); a and b must have the same shape.
struct FeatureMap feature_map_add(const &FeatureMap a, const &FeatureMap b, int relu);

// ResNet basic block (He et al. 2016): out = relu(conv2(relu(conv1(x))) + shortcut(x)),
// conv1 3x3 with 'stride', conv2 3x3 stride 1, both padding 1. The shortcut
// is x itself, or a 1x1 convolution with 'stride' when the stride is not 1
// or the channel count changes (the "projection" shortcut). Output side
// (dim - 1)/stride + 1. No batch norm: fold a trained model's batch norms
// into the convolutions' weights and biases.
struct ResidualBlock {
    struct Conv2D conv1;
    struct Conv2D conv2;
    struct Conv2D projection;  // used only when hasProjection
    int           hasProjection;

    struct FeatureMap forward(const &FeatureMap input) const;
    void free();
};
struct ResidualBlock residual_block_new(unsigned long inChannels, unsigned long outChannels, unsigned long stride);

// Nearest-neighbor 2x spatial upsample: out[c][2h+dh][2w+dw] = in[c][h][w]
// for dh,dw in {0,1}. Used by U-Net's decoder path (see unet.h) in place
// of a learned transposed convolution.
struct FeatureMap upsample2x_nearest(const &FeatureMap input);

// Concatenates 'a' and 'b' along the channel axis: out has
// a.channels + b.channels channels, a's copied first, then b's. Both
// must share the same height/width. Used to implement U-Net's encoder-
// to-decoder skip connections.
struct FeatureMap concat_channels(const &FeatureMap a, const &FeatureMap b);

} // namespace std
