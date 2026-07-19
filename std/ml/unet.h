#pragma once
// SafeC Standard Library — U-Net: a small encoder/decoder CNN with skip
// connections (Ronneberger et al. 2015), the convolutional backbone
// classic DDPM-style diffusion models denoise with.
//
// A fixed 2-level (2 downsample/upsample stages) U-Net over
// std::FeatureMap (see cnn.h): each encoder stage is
// Conv2D(3x3,pad1) -> ReLU -> MaxPool2D(2x2), the bottleneck is one more
// Conv2D -> ReLU, and each decoder stage is
// upsample2x_nearest -> concat_channels(skip) -> Conv2D(3x3,pad1) ->
// ReLU (the final decoder stage skips the trailing ReLU, since this
// produces the network's raw output — e.g. a predicted noise map, not a
// bounded activation). Forward-only, like the rest of std/ml.
#include <std/ml/tensor.h>
#include <std/ml/cnn.h>

namespace std {

struct UNet {
    struct Conv2D encConv1; // inChannels -> c1
    struct Conv2D encConv2; // c1 -> c2
    struct Conv2D bottleneck; // c2 -> c2
    struct Conv2D decConv1; // (c2 + c2 skip) -> c1
    struct Conv2D decConv2; // (c1 + c1 skip) -> outChannels
};

struct UNet unet_new(unsigned long inChannels, unsigned long c1, unsigned long c2,
                      unsigned long outChannels);
// Input height/width must each be divisible by 4 (two 2x poolings).
struct FeatureMap unet_forward(struct UNet* net, const struct FeatureMap* input);
void unet_free(struct UNet* net);

} // namespace std
