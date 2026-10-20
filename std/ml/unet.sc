// SafeC Standard Library — U-Net implementation (see unet.h).
#pragma once
#include <std/ml/unet.h>
#include <std/ml/tensor.h>
#include <std/ml/cnn.h>
#include <std/mem.h>

namespace std {

struct UNet unet_new(unsigned long inChannels, unsigned long c1, unsigned long c2,
                      unsigned long outChannels) {
    struct UNet net;
    net.encConv1 = conv2d_new(inChannels, c1, 3UL, 3UL, 1UL, 1UL);
    net.encConv2 = conv2d_new(c1, c2, 3UL, 3UL, 1UL, 1UL);
    net.bottleneck = conv2d_new(c2, c2, 3UL, 3UL, 1UL, 1UL);
    net.decConv1 = conv2d_new(c2 + c2, c1, 3UL, 3UL, 1UL, 1UL);
    net.decConv2 = conv2d_new(c1 + c1, outChannels, 3UL, 3UL, 1UL, 1UL);
    return net;
}

void unet_free(&UNet net) {
    unsafe {
        conv2d_free((struct Conv2D*)&net->encConv1);
        conv2d_free((struct Conv2D*)&net->encConv2);
        conv2d_free((struct Conv2D*)&net->bottleneck);
        conv2d_free((struct Conv2D*)&net->decConv1);
        conv2d_free((struct Conv2D*)&net->decConv2);
    }
}

struct FeatureMap unet_forward(&UNet net, const &FeatureMap input) {

    struct FeatureMap e1 = conv2d_relu_forward(&net->encConv1, input);
    struct FeatureMap p1 = maxpool2d_forward(&e1, 2UL, 2UL);

    struct FeatureMap e2 = conv2d_relu_forward(&net->encConv2, &p1);
    feature_map_free(&p1);
    struct FeatureMap p2 = maxpool2d_forward(&e2, 2UL, 2UL);

    struct FeatureMap b = conv2d_relu_forward(&net->bottleneck, &p2);
    feature_map_free(&p2);

    struct FeatureMap u1 = upsample2x_nearest(&b);
    feature_map_free(&b);
    struct FeatureMap cat1 = concat_channels(&u1, &e2);
    feature_map_free(&u1);
    feature_map_free(&e2);
    struct FeatureMap d1 = conv2d_relu_forward(&net->decConv1, &cat1);
    feature_map_free(&cat1);

    struct FeatureMap u2 = upsample2x_nearest(&d1);
    feature_map_free(&d1);
    struct FeatureMap cat2 = concat_channels(&u2, &e1);
    feature_map_free(&u2);
    feature_map_free(&e1);
    struct FeatureMap out = conv2d_forward(&net->decConv2, &cat2);
    feature_map_free(&cat2);

    return out;
}

} // namespace std
