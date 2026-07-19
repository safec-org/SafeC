// SafeC Standard Library — U-Net implementation (see unet.h).
#pragma once
#include <std/ml/unet.h>
#include <std/ml/tensor.h>
#include <std/ml/cnn.h>
#include <std/ml/cnn.sc>
#include <std/mem.sc>

namespace std {

static struct FeatureMap __fm_relu(const struct FeatureMap* fm) {
    struct FeatureMap out;
    unsafe {
        out = feature_map_new(fm->channels, fm->height, fm->width);
        unsigned long size = fm->channels * fm->height * fm->width;
        unsigned long i = 0UL;
        while (i < size) {
            double v = fm->data[i];
            out.data[i] = (v > 0.0) ? v : 0.0;
            i = i + 1UL;
        }
    }
    return out;
}

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

void unet_free(struct UNet* net) {
    unsafe {
        conv2d_free((struct Conv2D*)&net->encConv1);
        conv2d_free((struct Conv2D*)&net->encConv2);
        conv2d_free((struct Conv2D*)&net->bottleneck);
        conv2d_free((struct Conv2D*)&net->decConv1);
        conv2d_free((struct Conv2D*)&net->decConv2);
    }
}

struct FeatureMap unet_forward(struct UNet* net, const struct FeatureMap* input) {
    struct Conv2D* pEnc1; struct Conv2D* pEnc2; struct Conv2D* pBottle;
    struct Conv2D* pDec1; struct Conv2D* pDec2;
    unsafe {
        pEnc1 = (struct Conv2D*)&net->encConv1; pEnc2 = (struct Conv2D*)&net->encConv2;
        pBottle = (struct Conv2D*)&net->bottleneck;
        pDec1 = (struct Conv2D*)&net->decConv1; pDec2 = (struct Conv2D*)&net->decConv2;
    }

    struct FeatureMap e1c = conv2d_forward(pEnc1, input);
    struct FeatureMap e1 = __fm_relu(&e1c);
    feature_map_free(&e1c);
    struct FeatureMap p1 = maxpool2d_forward(&e1, 2UL, 2UL);

    struct FeatureMap e2c = conv2d_forward(pEnc2, &p1);
    struct FeatureMap e2 = __fm_relu(&e2c);
    feature_map_free(&e2c);
    feature_map_free(&p1);
    struct FeatureMap p2 = maxpool2d_forward(&e2, 2UL, 2UL);

    struct FeatureMap bc = conv2d_forward(pBottle, &p2);
    struct FeatureMap b = __fm_relu(&bc);
    feature_map_free(&bc);
    feature_map_free(&p2);

    struct FeatureMap u1 = upsample2x_nearest(&b);
    feature_map_free(&b);
    struct FeatureMap cat1 = concat_channels(&u1, &e2);
    feature_map_free(&u1);
    feature_map_free(&e2);
    struct FeatureMap d1c = conv2d_forward(pDec1, &cat1);
    feature_map_free(&cat1);
    struct FeatureMap d1 = __fm_relu(&d1c);
    feature_map_free(&d1c);

    struct FeatureMap u2 = upsample2x_nearest(&d1);
    feature_map_free(&d1);
    struct FeatureMap cat2 = concat_channels(&u2, &e1);
    feature_map_free(&u2);
    feature_map_free(&e1);
    struct FeatureMap out = conv2d_forward(pDec2, &cat2);
    feature_map_free(&cat2);

    return out;
}

} // namespace std
