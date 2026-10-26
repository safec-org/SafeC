// SafeC Standard Library — U-Net implementation (see unet.h).
#pragma once
#include <std/ml/unet.h>
#include <std/ml/tensor.h>
#include <std/ml/cnn.h>
#include <std/mem.h>

namespace std {

struct UNet UNet::new(unsigned long inChannels, unsigned long c1, unsigned long c2, unsigned long outChannels) {
    struct UNet net;
    net.encConv1 = Conv2D::new(inChannels, c1, 3UL, 3UL, 1UL, 1UL);
    net.encConv2 = Conv2D::new(c1, c2, 3UL, 3UL, 1UL, 1UL);
    net.bottleneck = Conv2D::new(c2, c2, 3UL, 3UL, 1UL, 1UL);
    net.decConv1 = Conv2D::new(c2 + c2, c1, 3UL, 3UL, 1UL, 1UL);
    net.decConv2 = Conv2D::new(c1 + c1, outChannels, 3UL, 3UL, 1UL, 1UL);
    return net;
}

void UNet::free() { self.drop(); }

void UNet::drop() {
    unsafe {
        ((struct Conv2D*)&self->encConv1).free();
        ((struct Conv2D*)&self->encConv2).free();
        ((struct Conv2D*)&self->bottleneck).free();
        ((struct Conv2D*)&self->decConv1).free();
        ((struct Conv2D*)&self->decConv2).free();
    }
}

struct FeatureMap UNet::forward(const &FeatureMap input) {

    struct FeatureMap e1 = self->encConv1.relu_forward(input);
    struct FeatureMap p1 = e1.maxpool2d(2UL, 2UL);

    struct FeatureMap e2 = self->encConv2.relu_forward(&p1);
    p1.free();
    struct FeatureMap p2 = e2.maxpool2d(2UL, 2UL);

    struct FeatureMap b = self->bottleneck.relu_forward(&p2);
    p2.free();

    struct FeatureMap u1 = b.upsample2x_nearest();
    b.free();
    struct FeatureMap cat1 = concat_channels(&u1, &e2);
    u1.free();
    e2.free();
    struct FeatureMap d1 = self->decConv1.relu_forward(&cat1);
    cat1.free();

    struct FeatureMap u2 = d1.upsample2x_nearest();
    d1.free();
    struct FeatureMap cat2 = concat_channels(&u2, &e1);
    u2.free();
    e1.free();
    struct FeatureMap out = self->decConv2.forward(&cat2);
    cat2.free();

    return out;
}

} // namespace std
