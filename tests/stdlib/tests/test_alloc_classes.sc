// std::alloc size classes, and a CPU convolution followed by large
// allocations. Sizes in (1MB, 1.5MB] used to map to a size class 33 one past
// the 33-entry cache tables, so alloc_raw_ read allocCacheCount_[33], which is
// whatever thread_local the linker put next — cnn.sc's im2col buffer pointer
// once conv2d_forward had run, giving a garbage "cached" block and a segfault.
#include <std/test/test.h>
#include <std/mem.h>
#include <std/collections/vec.h>
#include <std/ml/cnn.h>
#include <std/sync/parallel.h>

static int near(float a, float b) {
    float d = a - b;
    if (d < (float)0.0) d = (float)0.0 - d;
    return d <= (float)1.0e-3;
}

// Header word 1 holds the block's size class (all-ones when uncached).
static unsigned long block_class(void* p) {
    unsafe { return ((unsigned long*)((unsigned long)p - 16UL))[1]; }
}

void t_size_classes() {
    unsigned long sizes[9] = { 1048560UL, 1048576UL, 1048577UL, 1048580UL, 1310720UL,
                               1572864UL, 1572865UL, 2097152UL, 1572864UL - 8UL };
    unsafe {
        unsigned long i = 0UL;
        while (i < 9UL) {
            void* p = alloc_raw_(sizes[i]);
            ASSERT_TRUE(p != (void*)0);
            unsigned long cls = block_class(p);
            if (sizes[i] <= 1048576UL) ASSERT_EQ(cls, 32UL);
            else ASSERT_TRUE(cls == 0xFFFFFFFFFFFFFFFFUL);
            ((char*)p)[0] = (char)1;
            ((char*)p)[sizes[i] - 1UL] = (char)2;
            dealloc_raw_(p);
            i = i + 1UL;
        }
    }
}

// One conv against a direct reference, then allocations around 1MB.
static void conv_then_alloc(unsigned long inC, unsigned long outC, unsigned long k,
                            unsigned long stride, unsigned long pad,
                            unsigned long h, unsigned long w, int relu) {
    struct Conv2D l = conv2d_new(inC, outC, k, k, stride, pad);
    struct FeatureMap x = feature_map_new(inC, h, w);
    unsafe {
        unsigned long i = 0UL;
        while (i < inC * h * w) { x.data[i] = (float)((i * 7UL) % 11UL) * (float)0.1 - (float)0.5; i = i + 1UL; }
        i = 0UL;
        while (i < outC) { l.bias[i] = (float)i * (float)0.05 - (float)0.1; i = i + 1UL; }
    }
    struct FeatureMap y = relu ? conv2d_relu_forward(&l, &x) : conv2d_forward(&l, &x);
    unsigned long oh = (h + 2UL * pad - k) / stride + 1UL;
    unsigned long ow = (w + 2UL * pad - k) / stride + 1UL;
    ASSERT_EQ(y.channels, outC);
    ASSERT_EQ(y.height, oh);
    ASSERT_EQ(y.width, ow);
    unsafe {
        unsigned long oc = 0UL;
        while (oc < outC) {
            unsigned long p = 0UL;
            while (p < oh * ow) {
                float s = l.bias[oc];
                unsigned long ic = 0UL;
                while (ic < inC) {
                    unsigned long kh = 0UL;
                    while (kh < k) {
                        unsigned long kw = 0UL;
                        while (kw < k) {
                            long ih = (long)((p / ow) * stride + kh) - (long)pad;
                            long iw = (long)((p % ow) * stride + kw) - (long)pad;
                            if (ih >= 0L && iw >= 0L && (unsigned long)ih < h && (unsigned long)iw < w)
                                s = s + l.weight[((oc * inC + ic) * k + kh) * k + kw]
                                      * x.data[(ic * h + (unsigned long)ih) * w + (unsigned long)iw];
                            kw = kw + 1UL;
                        }
                        kh = kh + 1UL;
                    }
                    ic = ic + 1UL;
                }
                if (relu && s < (float)0.0) s = (float)0.0;
                ASSERT_TRUE(near(y.data[oc * oh * ow + p], s));
                p = p + 1UL;
            }
            oc = oc + 1UL;
        }
    }
    // The original repro: just over 1MB of floats, then a few more sizes,
    // each twice so the second round can come from the size-class cache.
    int round = 0;
    while (round < 2) {
        struct Vec<float> v = vec_with_cap(262145UL);
        unsigned long i = 0UL;
        while (i < 262145UL) { v.push((float)i); i = i + 1UL; }
        ASSERT_EQ(v.length(), 262145);
        ASSERT_TRUE(v.get(262144UL).default((float)-1.0) == (float)262144.0);
        v.free();
        t_size_classes();
        round = round + 1;
    }
    feature_map_free(&y);
    feature_map_free(&x);
    conv2d_free(&l);
}

void t_conv_then_alloc() {
    unsigned long threads[3] = { 1UL, 2UL, 4UL };
    unsafe {
        unsigned long saved = parallel_threads();
        unsigned long t = 0UL;
        while (t < 3UL) {
            parallel_set_threads(threads[t]);
            conv_then_alloc(3UL, 8UL, 3UL, 1UL, 1UL, 17UL, 13UL, 0);   // the original repro
            conv_then_alloc(4UL, 6UL, 5UL, 2UL, 0UL, 19UL, 23UL, 1);
            conv_then_alloc(16UL, 32UL, 3UL, 1UL, 1UL, 40UL, 36UL, 1); // several blocks, threaded
            conv_then_alloc(8UL, 5UL, 1UL, 1UL, 0UL, 9UL, 7UL, 0);     // 1x1 path
            conv_then_alloc(2UL, 3UL, 3UL, 1UL, 1UL, 1UL, 1UL, 0);     // one pixel
            t = t + 1UL;
        }
        parallel_set_threads(saved);
    }
}

int main() {
    struct TestSuite t = test_suite_init();
    unsafe {
        t.add("alloc size classes around 1MB", t_size_classes);
        t.add("conv2d then large allocations", t_conv_then_alloc);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
