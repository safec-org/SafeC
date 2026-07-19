// SafeC Standard Library — CNN implementation (see cnn.h).
#pragma once
#include <std/ml/cnn.h>
#include <std/mem.sc>

namespace std {

struct FeatureMap feature_map_new(unsigned long channels, unsigned long height, unsigned long width) {
    struct FeatureMap fm;
    unsigned long size = channels * height * width;
    unsafe {
        double* buf = (double*)malloc(sizeof(double) * size);
        unsigned long i = 0UL;
        while (i < size) { buf[i] = 0.0; i = i + 1UL; }
        fm.data = (&heap double)buf;
    }
    fm.channels = channels;
    fm.height = height;
    fm.width = width;
    return fm;
}

void feature_map_free(&FeatureMap fm) {
    unsafe { free((void*)fm->data); }
}

static double __cnn_init_weight(unsigned long i, unsigned long seed) {
    unsigned long h = (i + 1UL) * 2654435761UL + seed * 40503UL;
    h = (h ^ (h >> 13UL)) * 2246822519UL;
    return (((double)(h % 2000UL) / 1000.0) - 1.0) * 0.3; // [-0.3, 0.3)
}

struct Conv2D conv2d_new(unsigned long inChannels, unsigned long outChannels,
                          unsigned long kH, unsigned long kW,
                          unsigned long stride, unsigned long padding) {
    struct Conv2D layer;
    unsigned long wSize = outChannels * inChannels * kH * kW;
    unsafe {
        double* w = (double*)malloc(sizeof(double) * wSize);
        unsigned long i = 0UL;
        while (i < wSize) { w[i] = __cnn_init_weight(i, 11UL); i = i + 1UL; }
        layer.weight = (&heap double)w;

        double* b = (double*)malloc(sizeof(double) * outChannels);
        i = 0UL;
        while (i < outChannels) { b[i] = 0.0; i = i + 1UL; }
        layer.bias = (&heap double)b;
    }
    layer.inChannels = inChannels;
    layer.outChannels = outChannels;
    layer.kH = kH;
    layer.kW = kW;
    layer.stride = stride;
    layer.padding = padding;
    return layer;
}

void conv2d_free(&Conv2D layer) {
    unsafe { free((void*)layer->weight); free((void*)layer->bias); }
}

struct FeatureMap conv2d_forward(const &Conv2D layer, const &FeatureMap input) {
    unsigned long inH; unsigned long inW; unsigned long inC;
    unsigned long kH; unsigned long kW; unsigned long stride; unsigned long padding;
    unsigned long outC;
    unsafe {
        inC = input->channels; inH = input->height; inW = input->width;
        kH = layer->kH; kW = layer->kW; stride = layer->stride; padding = layer->padding;
        outC = layer->outChannels;
    }
    unsigned long outH = (inH + 2UL * padding - kH) / stride + 1UL;
    unsigned long outW = (inW + 2UL * padding - kW) / stride + 1UL;
    struct FeatureMap out = feature_map_new(outC, outH, outW);

    unsafe {
        unsigned long oc = 0UL;
        while (oc < outC) {
            double bias = layer->bias[oc];
            unsigned long oh = 0UL;
            while (oh < outH) {
                unsigned long ow = 0UL;
                while (ow < outW) {
                    double acc = bias;
                    unsigned long ic = 0UL;
                    while (ic < inC) {
                        unsigned long kh = 0UL;
                        while (kh < kH) {
                            long ih = (long)(oh * stride + kh) - (long)padding;
                            unsigned long kw = 0UL;
                            while (kw < kW) {
                                long iw = (long)(ow * stride + kw) - (long)padding;
                                if (ih >= 0L && iw >= 0L && (unsigned long)ih < inH && (unsigned long)iw < inW) {
                                    unsigned long inIdx = ic * inH * inW + (unsigned long)ih * inW + (unsigned long)iw;
                                    unsigned long wIdx = ((oc * inC + ic) * kH + kh) * kW + kw;
                                    acc = acc + layer->weight[wIdx] * input->data[inIdx];
                                }
                                kw = kw + 1UL;
                            }
                            kh = kh + 1UL;
                        }
                        ic = ic + 1UL;
                    }
                    out.data[oc * outH * outW + oh * outW + ow] = acc;
                    ow = ow + 1UL;
                }
                oh = oh + 1UL;
            }
            oc = oc + 1UL;
        }
    }
    return out;
}

struct FeatureMap maxpool2d_forward(const &FeatureMap input, unsigned long kernel, unsigned long stride) {
    unsigned long c; unsigned long h; unsigned long w;
    unsafe { c = input->channels; h = input->height; w = input->width; }
    unsigned long outH = (h - kernel) / stride + 1UL;
    unsigned long outW = (w - kernel) / stride + 1UL;
    struct FeatureMap out = feature_map_new(c, outH, outW);
    unsafe {
        unsigned long ch = 0UL;
        while (ch < c) {
            unsigned long oh = 0UL;
            while (oh < outH) {
                unsigned long ow = 0UL;
                while (ow < outW) {
                    double best = input->data[ch * h * w + (oh * stride) * w + (ow * stride)];
                    unsigned long kh = 0UL;
                    while (kh < kernel) {
                        unsigned long kw = 0UL;
                        while (kw < kernel) {
                            double v = input->data[ch * h * w + (oh * stride + kh) * w + (ow * stride + kw)];
                            if (v > best) best = v;
                            kw = kw + 1UL;
                        }
                        kh = kh + 1UL;
                    }
                    out.data[ch * outH * outW + oh * outW + ow] = best;
                    ow = ow + 1UL;
                }
                oh = oh + 1UL;
            }
            ch = ch + 1UL;
        }
    }
    return out;
}

struct FeatureMap avgpool2d_forward(const &FeatureMap input, unsigned long kernel, unsigned long stride) {
    unsigned long c; unsigned long h; unsigned long w;
    unsafe { c = input->channels; h = input->height; w = input->width; }
    unsigned long outH = (h - kernel) / stride + 1UL;
    unsigned long outW = (w - kernel) / stride + 1UL;
    struct FeatureMap out = feature_map_new(c, outH, outW);
    double count = (double)(kernel * kernel);
    unsafe {
        unsigned long ch = 0UL;
        while (ch < c) {
            unsigned long oh = 0UL;
            while (oh < outH) {
                unsigned long ow = 0UL;
                while (ow < outW) {
                    double sum = 0.0;
                    unsigned long kh = 0UL;
                    while (kh < kernel) {
                        unsigned long kw = 0UL;
                        while (kw < kernel) {
                            sum = sum + input->data[ch * h * w + (oh * stride + kh) * w + (ow * stride + kw)];
                            kw = kw + 1UL;
                        }
                        kh = kh + 1UL;
                    }
                    out.data[ch * outH * outW + oh * outW + ow] = sum / count;
                    ow = ow + 1UL;
                }
                oh = oh + 1UL;
            }
            ch = ch + 1UL;
        }
    }
    return out;
}

struct FeatureMap upsample2x_nearest(const &FeatureMap input) {
    unsigned long c; unsigned long h; unsigned long w;
    unsafe { c = input->channels; h = input->height; w = input->width; }
    unsigned long outH = h * 2UL;
    unsigned long outW = w * 2UL;
    struct FeatureMap out = feature_map_new(c, outH, outW);
    unsafe {
        unsigned long ch = 0UL;
        while (ch < c) {
            unsigned long y = 0UL;
            while (y < h) {
                unsigned long x = 0UL;
                while (x < w) {
                    double v = input->data[ch * h * w + y * w + x];
                    out.data[ch * outH * outW + (2UL * y) * outW + (2UL * x)] = v;
                    out.data[ch * outH * outW + (2UL * y) * outW + (2UL * x + 1UL)] = v;
                    out.data[ch * outH * outW + (2UL * y + 1UL) * outW + (2UL * x)] = v;
                    out.data[ch * outH * outW + (2UL * y + 1UL) * outW + (2UL * x + 1UL)] = v;
                    x = x + 1UL;
                }
                y = y + 1UL;
            }
            ch = ch + 1UL;
        }
    }
    return out;
}

struct FeatureMap concat_channels(const &FeatureMap a, const &FeatureMap b) {
    unsigned long ca; unsigned long cb; unsigned long h; unsigned long w;
    unsafe { ca = a->channels; cb = b->channels; h = a->height; w = a->width; }
    struct FeatureMap out = feature_map_new(ca + cb, h, w);
    unsigned long plane = h * w;
    unsafe {
        unsigned long i = 0UL;
        while (i < ca * plane) { out.data[i] = a->data[i]; i = i + 1UL; }
        i = 0UL;
        while (i < cb * plane) { out.data[ca * plane + i] = b->data[i]; i = i + 1UL; }
    }
    return out;
}

} // namespace std
