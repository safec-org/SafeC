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
#include <std/mem.sc>

namespace std {

struct FeatureMap {
    &heap float   data;   // flat, channels*height*width, CHW order (data[c*H*W + h*W + w])
    unsigned long channels;
    unsigned long height;
    unsigned long width;
};

struct FeatureMap feature_map_new(unsigned long channels, unsigned long height, unsigned long width);
void feature_map_free(&FeatureMap fm);

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
};

struct Conv2D conv2d_new(unsigned long inChannels, unsigned long outChannels,
                          unsigned long kH, unsigned long kW,
                          unsigned long stride, unsigned long padding);
struct FeatureMap conv2d_forward(const &Conv2D layer, const &FeatureMap input);
void conv2d_free(&Conv2D layer);

// Max/avg pooling: kernel x kernel window, given stride, no padding.
// Channel count is preserved; caller must choose dimensions divisible by
// the kernel/stride combination.
struct FeatureMap maxpool2d_forward(const &FeatureMap input, unsigned long kernel, unsigned long stride);
struct FeatureMap avgpool2d_forward(const &FeatureMap input, unsigned long kernel, unsigned long stride);

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
