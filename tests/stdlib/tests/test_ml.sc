// std::ml fp16/bf16: HalfTensor storage + ops, mixed-precision Autograd::get,
// dynamic loss scaling. Expected values are exact fp16/bf16 roundings
// (checked against IEEE binary16 / bfloat16 bit patterns).
#include <std/test/test.h>
#include <std/ml/tensor_half.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/cnn.h>
#include <std/ml/ssm.h>
#include <std/math.h>

extern void* malloc(unsigned long size);
extern void free(void* ptr);

static int near(float a, float b, float tol) {
    float d = a - b;
    if (d < (float)0.0) d = (float)0.0 - d;
    return d <= tol;
}

void t_round() {
    // 1/3: fp16 keeps 11 significant bits, bf16 keeps 8.
    float third = (float)1.0 / (float)3.0;
    ASSERT_TRUE(half_round(third, TENSOR_FP16) == (float)0.333251953125);
    ASSERT_TRUE(half_round(third, TENSOR_BF16) == (float)0.333984375);
    // fp16 overflows above 65504; bf16 has float32's range.
    ASSERT_TRUE(half_round((float)70000.0, TENSOR_FP16) > (float)1.0e30);
    ASSERT_TRUE(half_round((float)70000.0, TENSOR_BF16) == (float)70144.0);
    ASSERT_EQ(half_encode((float)1.0, TENSOR_FP16), 0x3C00);
    ASSERT_EQ(half_encode((float)1.0, TENSOR_BF16), 0x3F80);
    ASSERT_TRUE(half_decode((unsigned short)0x4000, TENSOR_FP16) == (float)2.0);
}

void t_convert() {
    unsafe {
        float v[4] = { (float)1.0, (float)-2.5, (float)0.1, (float)1000.0 };
        &FloatTensor t = FloatTensor::from_2d(((const float*)((const float*)v))[0..2UL * 2UL], 2UL, 2UL, 0);
        &HalfTensor h = HalfTensor::from_float(t);
        ASSERT_EQ(h.ndim, 2);
        ASSERT_EQ(h.size, 4);
        ASSERT_TRUE(h.at2(0UL, 0UL) == (float)1.0);
        ASSERT_TRUE(h.at2(0UL, 1UL) == (float)-2.5);
        ASSERT_TRUE(h.at2(1UL, 0UL) == (float)0.0999755859375);   // 0.1 in fp16
        ASSERT_TRUE(h.at2(1UL, 1UL) == (float)1000.0);
        &FloatTensor back = h.to_float();
        ASSERT_TRUE(back.at2(1UL, 0UL) == (float)0.0999755859375);
        h.set1(3UL, (float)3.0);
        ASSERT_TRUE(h.at1(3UL) == (float)3.0);

        &BF16Tensor b = BF16Tensor::from_float(t);
        ASSERT_TRUE(b.at1(2UL) == (float)0.10009765625);            // 0.1 in bf16
        b.free();
        h.free();
        back.free();
        t.free();
    }
}

void t_half_ops() {
    &HalfTensor a = HalfTensor::new_2d(2UL, 2UL, 0);
    &HalfTensor b = HalfTensor::new_2d(2UL, 2UL, 0);
    a.set2(0UL, 0UL, (float)1.0); a.set2(0UL, 1UL, (float)2.0);
    a.set2(1UL, 0UL, (float)3.0); a.set2(1UL, 1UL, (float)-4.0);
    b.set2(0UL, 0UL, (float)0.5); b.set2(0UL, 1UL, (float)1.0);
    b.set2(1UL, 0UL, (float)1.5); b.set2(1UL, 1UL, (float)2.0);

    &HalfTensor s = tensor_add(a, b);
    ASSERT_TRUE(s.at2(1UL, 1UL) == (float)-2.0);
    &HalfTensor d = tensor_sub(a, b);
    ASSERT_TRUE(d.at2(0UL, 0UL) == (float)0.5);
    &HalfTensor m = tensor_mul(a, b);
    ASSERT_TRUE(m.at2(1UL, 0UL) == (float)4.5);
    &HalfTensor k = a.scale((float)0.5);
    ASSERT_TRUE(k.at2(0UL, 1UL) == (float)1.0);

    // [[1,2],[3,-4]] @ [[0.5,1],[1.5,2]] = [[3.5,5],[-4.5,-5]]
    &HalfTensor p = tensor_matmul(a, b);
    unsafe {
        ASSERT_EQ(p.shape[0], 2);
        ASSERT_EQ(p.shape[1], 2);
    }
    ASSERT_TRUE(p.at2(0UL, 0UL) == (float)3.5);
    ASSERT_TRUE(p.at2(0UL, 1UL) == (float)5.0);
    ASSERT_TRUE(p.at2(1UL, 0UL) == (float)-4.5);
    ASSERT_TRUE(p.at2(1UL, 1UL) == (float)-5.0);
    ASSERT_TRUE(a.sum().at1(0UL) == (float)2.0);

    // a result is rounded to the element type: 1/3 in bf16
    &BF16Tensor x = BF16Tensor::new_1d(1UL, 0);
    x.set1(0UL, (float)1.0);
    ASSERT_TRUE(x.scale((float)1.0 / (float)3.0).at1(0UL) == (float)0.333984375);

    s.free(); d.free(); m.free(); k.free(); p.free(); a.free(); b.free(); x.free();
}

void t_matmul_accumulates_f32() {
    // 2048 * (1 * 1): an fp16 running sum would stall at 2048 + 1 == 2048;
    // float32 accumulation gives the exact 4096 (representable in fp16).
    &HalfTensor a = HalfTensor::new_2d(1UL, 4096UL, 0);
    &HalfTensor b = HalfTensor::new_2d(4096UL, 1UL, 0);
    a.fill((float)1.0);
    b.fill((float)1.0);
    &HalfTensor p = tensor_matmul(a, b);
    ASSERT_TRUE(p.at1(0UL) == (float)4096.0);
    p.free(); a.free(); b.free();
}

// Autograd on 16-bit tensors: gradients are float32 and match a
// FloatTensor graph over the same (rounded) values.
void t_half_autograd() {
    float av[4] = { (float)1.0, (float)2.0, (float)3.0, (float)-4.0 };
    float bv[4] = { (float)0.5, (float)1.0, (float)1.5, (float)2.0 };
    &HalfTensor a = HalfTensor::from_2d(av[0..2UL * 2UL], 2UL, 2UL, 1);
    &HalfTensor b = HalfTensor::from_2d(bv[0..2UL * 2UL], 2UL, 2UL, 1);
    // loss = sum((a @ b) * a)
    &HalfTensor loss = tensor_mul(tensor_matmul(a, b), a).sum();
    loss.backward();
    &FloatTensor fa = FloatTensor::from_2d(av[0..2UL * 2UL], 2UL, 2UL, 1);
    &FloatTensor fb = FloatTensor::from_2d(bv[0..2UL * 2UL], 2UL, 2UL, 1);
    tensor_mul(tensor_matmul(fa, fb), fa).sum().backward();
    ASSERT_TRUE(loss.at1(0UL) == (float)20.0);   // 3.5 + 10 - 13.5 + 20
    unsigned long i = 0UL;
    unsafe {
        while (i < 4UL) {
            ASSERT_TRUE(a->grad[i] == fa->grad[i]);
            ASSERT_TRUE(b->grad[i] == fb->grad[i]);
            i = i + 1UL;
        }
    }
    // bf16 through the same path, and a graph that mixes element types
    &BF16Tensor c = BF16Tensor::from_1d(av[0..4UL], 1);
    &FloatTensor y = tensor_mul(c.to_float(), c.to_float()).sum();   // sum(c^2)
    y.backward();
    unsafe { ASSERT_TRUE(c->grad[3] == (float)-8.0); }
    a.free(); b.free(); fa.free(); fb.free(); c.free();
}

// What the gemv check asks of a 16-bit weight matrix.
trait Gemv {
    float at2(unsigned long r, unsigned long c) const;
    void  set1(unsigned long i, float v);
    int   gemv(const &float x, &float y) const;
    int   gemv_batch(const &float x, unsigned long m, &float y) const;
}

// n * k large enough to take the threaded path, with a k that isn't a
// multiple of 8 (tail loop)
generic<W: Gemv>
static void check_gemv_(&W w, unsigned long n, unsigned long k) {
    float* x;
    float* y;
    unsafe {
        x = (float*)malloc(sizeof(float) * k);
        y = (float*)malloc(sizeof(float) * n);
    }
    unsigned long i = 0UL;
    while (i < n * k) { w.set1(i, (float)((i * 37UL) % 113UL) * (float)0.01 - (float)0.55); i = i + 1UL; }
    unsafe { i = 0UL; while (i < k) { x[i] = (float)((i * 11UL) % 29UL) * (float)0.03 - (float)0.4; i = i + 1UL; } }
    int ok;
    unsafe { ok = w.gemv(x[0..k], y[0..n]); }
    ASSERT_TRUE(ok);
    float worst = (float)0.0;
    unsigned long r = 0UL;
    while (r < n) {
        double ref = 0.0;
        unsigned long c = 0UL;
        unsafe { while (c < k) { ref = ref + (double)w.at2(r, c) * (double)x[c]; c = c + 1UL; } }
        float d;
        unsafe { d = y[r] - (float)ref; }
        if (d < (float)0.0) d = -d;
        if (d > worst) worst = d;
        r = r + 1UL;
    }
    ASSERT_TRUE(worst < (float)1e-3);
    // gemv_batch: m vectors in one pass equal m gemv calls
    unsigned long m = 5UL;
    float* xs;
    float* ys;
    unsafe {
        xs = (float*)malloc(sizeof(float) * k * m);
        ys = (float*)malloc(sizeof(float) * n * m);
        i = 0UL; while (i < k * m) { xs[i] = (float)((i * 13UL) % 31UL) * (float)0.02 - (float)0.3; i = i + 1UL; }
        ok = w.gemv_batch(xs[0..k * m], m, ys[0..n * m]);
    }
    ASSERT_TRUE(ok);
    int same = 1;
    unsigned long v = 0UL;
    while (v < m) {
        unsafe {
            w.gemv(xs[v * k..v * k + k], y[0..n]);
            r = 0UL; while (r < n) { if (y[r] != ys[v * n + r]) same = 0; r = r + 1UL; }
        }
        v = v + 1UL;
    }
    ASSERT_TRUE(same);
    unsafe { free((void*)x); free((void*)y); free((void*)xs); free((void*)ys); }
}

void t_half_gemv() {
    &HalfTensor w = HalfTensor::new_2d(301UL, 1029UL, 0);
    check_gemv_(w, 301UL, 1029UL);
    w.free();
    &BF16Tensor wb = BF16Tensor::new_2d(301UL, 1029UL, 0);
    check_gemv_(wb, 301UL, 1029UL);
    wb.free();
    // fp16 widening edge cases: subnormal, negative, +-inf (inf * 1 = inf)
    &HalfTensor e = HalfTensor::new_2d(4UL, 1UL, 0);
    unsafe {
        e->data[0].bits = (unsigned short)0x0001;   // smallest subnormal, 2^-24
        e->data[1].bits = (unsigned short)0xc500;   // -5
        e->data[2].bits = (unsigned short)0x7c00;   // +inf
        e->data[3].bits = (unsigned short)0xfc00;   // -inf
    }
    float one[1];
    float out[4];
    unsafe {
        one[0] = (float)1.0;
        ASSERT_TRUE(e.gemv(one[..], out[..]));
    }
    ASSERT_TRUE(out[0] == (float)5.960464477539063e-08);
    ASSERT_TRUE(out[1] == (float)-5.0);
    ASSERT_TRUE(out[2] > (float)3.0e38 && out[2] * (float)0.5 == out[2]);
    ASSERT_TRUE(out[3] < (float)-3.0e38 && out[3] * (float)0.5 == out[3]);
    // a 1D tensor is not a matrix
    &BF16Tensor v = BF16Tensor::new_1d(4UL, 0);
    unsafe { ASSERT_TRUE(!v.gemv(one[..], out[..])); }
    e.free(); v.free();
}

void t_cast_autograd() {
    unsafe {
        float av[2] = { (float)0.1, (float)0.2 };
        &FloatTensor x = FloatTensor::from_1d(((const float*)((const float*)av))[0..2UL], 1);
        &FloatTensor c = x.cast(TENSOR_FP16);
        ASSERT_TRUE(c.at1(0UL) == (float)0.0999755859375);
        &FloatTensor y = tensor_mul(c, c).sum();      // sum(cast(x)^2)
        y.backward();
        // Straight-through: dy/dx = 2 * cast(x).
        ASSERT_TRUE(near(x.grad[0], (float)2.0 * (float)0.0999755859375, (float)1.0e-6));
        ASSERT_TRUE(near(x.grad[1], (float)2.0 * c.at1(1UL), (float)1.0e-6));
    }
}

void t_matmul_lowp() {
    unsafe {
        float av[4] = { (float)1.0, (float)2.0, (float)3.0, (float)4.0 };
        float bv[4] = { (float)0.1, (float)0.0, (float)0.0, (float)0.1 };
        &FloatTensor a = FloatTensor::from_2d(((const float*)((const float*)av))[0..2UL * 2UL], 2UL, 2UL, 1);
        &FloatTensor b = FloatTensor::from_2d(((const float*)((const float*)bv))[0..2UL * 2UL], 2UL, 2UL, 1);
        &FloatTensor out = tensor_matmul_lowp(a, b, TENSOR_BF16);
        float b01 = half_round((float)0.1, TENSOR_BF16);          // 0.10009765625
        ASSERT_TRUE(out.at2(0UL, 0UL) == half_round((float)1.0 * b01, TENSOR_BF16));
        ASSERT_TRUE(out.at2(1UL, 1UL) == half_round((float)4.0 * b01, TENSOR_BF16));
        out.sum().backward();
        // d sum(A B)/dA[i][j] = sum_n B[j][n] (of the rounded B), per row.
        ASSERT_TRUE(near(a.grad[0], b01, (float)1.0e-6));
        ASSERT_TRUE(near(b.grad[0], (float)4.0, (float)1.0e-6));  // A[0][0] + A[1][0]
    }
}

void t_loss_scaler() {
    unsafe {
        struct LossScaler s = LossScaler::new();
        ASSERT_TRUE(s.scale == (float)65536.0);
        float v[2] = { (float)1.0, (float)2.0 };
        &FloatTensor p = FloatTensor::from_1d(((const float*)((const float*)v))[0..2UL], 1);
        p.scale(s.scale).sum().backward();    // grads = 65536
        ASSERT_TRUE(s.unscale(p) == 1);
        ASSERT_TRUE(p.grad[0] == (float)1.0);
        s.update(1);
        ASSERT_EQ(s.goodSteps, 1);

        // Overflowed gradient -> 0, scale halves, counter resets.
        float big = (float)3.0e38;
        p.grad[0] = big * (float)10.0;                             // +inf
        ASSERT_TRUE(s.unscale(p) == 0);
        s.update(0);
        ASSERT_TRUE(s.scale == (float)32768.0);
        ASSERT_EQ(s.goodSteps, 0);

        // Growth after growthInterval clean steps.
        s.growthInterval = 2;
        s.update(1);
        s.update(1);
        ASSERT_TRUE(s.scale == (float)65536.0);
    }
}


// The same training loop on the heap and in a TensorArena: identical
// losses and weights, parameters' gradients survive resets, and the arena
// stops growing after the first step.
static float arena_train(int useArena, &FloatTensor W, const &FloatTensor X, unsigned long* chunkBytes) {
    struct TensorArena a = TensorArena::new(65536UL);
    float last = (float)0.0;
    int step = 0;
    unsigned long used0 = 0UL;
    while (step < 20) {
        if (useArena) a.begin();
        W.zero_grad();
        &FloatTensor y = tensor_matmul(X, W);
        &FloatTensor r = y.relu();
        &FloatTensor l = tensor_mul(r, r).sum();
        l.backward();
        unsafe {
            last = l->data[0];
            unsigned long i = 0UL;
            while (i < W->size) { W->data[i] = W->data[i] - (float)0.01 * W->grad[i]; i = i + 1UL; }
        }
        if (useArena) {
            tensor_arena_end();
            if (step == 0) used0 = a.used();
            if (step == 19) { unsafe { *chunkBytes = a.used() == used0 ? 1UL : 0UL; } }
            a.reset();
            ASSERT_TRUE(a.used() == 0UL);
        }
        step = step + 1;
    }
    a.free();
    return last;
}

void t_tensor_arena() {
    float xs[64];
    float ws[32];
    int i = 0;
    while (i < 64) { xs[i] = (float)((i * 37) % 17 - 8) * (float)0.1; i = i + 1; }
    i = 0;
    while (i < 32) { ws[i] = (float)((i * 11) % 7 - 3) * (float)0.05; i = i + 1; }
    &FloatTensor X = FloatTensor::from_2d(xs[0..8UL * 8UL], 8UL, 8UL, 0);
    &FloatTensor W1 = FloatTensor::from_2d(ws[0..8UL * 4UL], 8UL, 4UL, 1);
    &FloatTensor W2 = FloatTensor::from_2d(ws[0..8UL * 4UL], 8UL, 4UL, 1);
    unsigned long same = 0UL;
    float a = arena_train(0, W1, X, &same);
    float b = arena_train(1, W2, X, &same);
    ASSERT_TRUE(a == b);
    ASSERT_TRUE(same == 1UL);              // every step used the same bytes
    int eq = 1;
    unsigned long k = 0UL;
    unsafe { while (k < 32UL) { if (W1->data[k] != W2->data[k]) eq = 0; k = k + 1UL; } }
    ASSERT_TRUE(eq);
    unsafe { ASSERT_TRUE(W2->grad != (float*)0 && W2->owner.is_null()); }   // parameters stay on the heap

    // a tensor made inside the arena knows its owner, and free() leaves it alone
    struct TensorArena ar = TensorArena::new(0UL);
    ar.begin();
    &FloatTensor t = FloatTensor::new_2d(3UL, 5UL, 0);
    tensor_arena_end();
    ASSERT_FALSE(t->owner.is_null());
    t.free();
    ASSERT_TRUE(ar.used() > 0UL);
    ar.free();
}

// conv2d against a direct sum: 3x3 stride 1 (Winograd F(2x2,3x3)) with and
// without padding, odd sizes so the last tiles hang over the edge, ReLU,
// and a 5x5 / stride-2 layer (im2col).
static double conv_ref_err(&Conv2D layer, unsigned long inC, unsigned long H, unsigned long W, int relu) {
    struct FeatureMap in = FeatureMap::new(inC, H, W);
    unsigned long n = inC * H * W;
    unsigned long i = 0UL;
    unsafe { while (i < n) { in.data[i] = (float)((double)((i * 7919UL) % 1000UL) / 500.0 - 1.0); i = i + 1UL; } }
    struct FeatureMap out = relu ? layer.relu_forward(&in) : layer.forward(&in);
    double worst = 0.0;
    unsafe {
        unsigned long oc = 0UL;
        while (oc < out.channels) {
            unsigned long oh = 0UL;
            while (oh < out.height) {
                unsigned long ow = 0UL;
                while (ow < out.width) {
                    double acc = (double)layer.bias[oc];
                    unsigned long ic = 0UL;
                    while (ic < inC) {
                        unsigned long kh = 0UL;
                        while (kh < layer.kH) {
                            unsigned long kw = 0UL;
                            while (kw < layer.kW) {
                                long ih = (long)(oh * layer.stride + kh) - (long)layer.padding;
                                long iw = (long)(ow * layer.stride + kw) - (long)layer.padding;
                                if (ih >= 0L && ih < (long)H && iw >= 0L && iw < (long)W)
                                    acc = acc + (double)layer.weight[((oc * inC + ic) * layer.kH + kh) * layer.kW + kw] *
                                                (double)in.data[(ic * H + (unsigned long)ih) * W + (unsigned long)iw];
                                kw = kw + 1UL;
                            }
                            kh = kh + 1UL;
                        }
                        ic = ic + 1UL;
                    }
                    if (relu && acc < 0.0) acc = 0.0;
                    double e = acc - (double)out.data[(oc * out.height + oh) * out.width + ow];
                    if (e < 0.0) e = -e;
                    if (e > worst) worst = e;
                    ow = ow + 1UL;
                }
                oh = oh + 1UL;
            }
            oc = oc + 1UL;
        }
    }
    in.free();
    out.free();
    return worst;
}

void t_conv2d() {
    struct Conv2D a = Conv2D::new(16UL, 24UL, 3UL, 3UL, 1UL, 1UL);
    ASSERT_TRUE(conv_ref_err(&a, 16UL, 13UL, 11UL, 0) < 1e-4);
    ASSERT_TRUE(conv_ref_err(&a, 16UL, 13UL, 11UL, 1) < 1e-4);
    ASSERT_TRUE(conv_ref_err(&a, 16UL, 40UL, 37UL, 0) < 1e-4);   // several tile blocks
    a.free();
    struct Conv2D b = Conv2D::new(12UL, 8UL, 3UL, 3UL, 1UL, 0UL);  // no padding
    ASSERT_TRUE(conv_ref_err(&b, 12UL, 15UL, 10UL, 0) < 1e-4);
    b.free();
    // Inside a TensorArena the output (and every intermediate) comes from
    // the arena: same values, FeatureMap::free leaves it to the reset.
    struct Conv2D d = Conv2D::new(8UL, 8UL, 3UL, 3UL, 1UL, 1UL);
    struct FeatureMap in = FeatureMap::new(8UL, 20UL, 20UL);
    unsafe { unsigned long q = 0UL; while (q < 3200UL) { in.data[q] = (float)((double)(q % 17UL) / 8.0 - 1.0); q = q + 1UL; } }
    struct FeatureMap heapOut = d.forward(&in);
    ASSERT_EQ(heapOut.fromArena, 0);
    struct TensorArena ar = TensorArena::new(0UL);
    ar.begin();
    struct FeatureMap arOut = d.relu_forward(&in);
    struct FeatureMap tmp = FeatureMap::new(4UL, 4UL, 4UL);
    tensor_arena_end();
    ASSERT_EQ(arOut.fromArena, 1);
    ASSERT_EQ(tmp.fromArena, 1);
    unsafe { ASSERT_TRUE(tmp.data[63] == (float)0.0); }
    ASSERT_TRUE(ar.used() >= 8UL * 20UL * 20UL * 4UL);
    double worst = 0.0;
    unsafe {
        unsigned long q = 0UL;
        while (q < 3200UL) {
            float h = heapOut.data[q] > (float)0.0 ? heapOut.data[q] : (float)0.0;
            double e = (double)(h - arOut.data[q]);
            if (e < 0.0) e = -e;
            if (e > worst) worst = e;
            q = q + 1UL;
        }
    }
    ASSERT_TRUE(worst == 0.0);
    arOut.free();   // no-op for an arena map
    tmp.free();
    ar.reset();
    ASSERT_EQ(ar.used(), 0);
    ar.free();
    heapOut.free();
    in.free();
    d.free();
    struct Conv2D c = Conv2D::new(5UL, 7UL, 5UL, 5UL, 2UL, 2UL);   // im2col
    ASSERT_TRUE(conv_ref_err(&c, 5UL, 17UL, 14UL, 0) < 1e-4);
    c.free();
}

// Pooling with padding (PyTorch semantics) and global pooling, against a
// direct double-precision reference.
static double pool_ref_err(&FeatureMap in, &FeatureMap out, unsigned long k, unsigned long s,
                           unsigned long p, int isMax, int includePad) {
    double worst = 0.0;
    unsafe {
        unsigned long c = 0UL;
        while (c < out.channels) {
            unsigned long oh = 0UL;
            while (oh < out.height) {
                unsigned long ow = 0UL;
                while (ow < out.width) {
                    double acc = isMax ? -1.0e30 : 0.0;
                    unsigned long cnt = 0UL;
                    unsigned long kh = 0UL;
                    while (kh < k) {
                        unsigned long kw = 0UL;
                        while (kw < k) {
                            long y = (long)(oh * s + kh) - (long)p;
                            long x = (long)(ow * s + kw) - (long)p;
                            if (y >= 0L && y < (long)in.height && x >= 0L && x < (long)in.width) {
                                double v = (double)in.data[(c * in.height + (unsigned long)y) * in.width + (unsigned long)x];
                                if (isMax) { if (v > acc) acc = v; } else acc = acc + v;
                                cnt = cnt + 1UL;
                            }
                            kw = kw + 1UL;
                        }
                        kh = kh + 1UL;
                    }
                    if (!isMax) acc = acc / (double)(includePad ? k * k : cnt);
                    double e = acc - (double)out.data[(c * out.height + oh) * out.width + ow];
                    if (e < 0.0) e = -e;
                    if (e > worst) worst = e;
                    ow = ow + 1UL;
                }
                oh = oh + 1UL;
            }
            c = c + 1UL;
        }
    }
    return worst;
}

static void fill_map(&FeatureMap m, unsigned long seed) {
    unsigned long n = m.channels * m.height * m.width;
    unsigned long i = 0UL;
    unsafe { while (i < n) { m.data[i] = (float)((double)(((i + seed) * 7919UL) % 1000UL) / 500.0 - 1.0); i = i + 1UL; } }
}

void t_pooling() {
    struct FeatureMap in = FeatureMap::new(3UL, 9UL, 7UL);
    fill_map(&in, 0UL);
    struct FeatureMap mx = in.maxpool2d_ex(3UL, 2UL, 1UL);   // ResNet stem pool
    ASSERT_EQ(mx.height, 5UL); ASSERT_EQ(mx.width, 4UL);
    ASSERT_TRUE(pool_ref_err(&in, &mx, 3UL, 2UL, 1UL, 1, 0) < 1e-6);
    struct FeatureMap a1 = in.avgpool2d_ex(3UL, 2UL, 1UL, 1);
    ASSERT_TRUE(pool_ref_err(&in, &a1, 3UL, 2UL, 1UL, 0, 1) < 1e-6);
    struct FeatureMap a0 = in.avgpool2d_ex(3UL, 2UL, 1UL, 0);
    ASSERT_TRUE(pool_ref_err(&in, &a0, 3UL, 2UL, 1UL, 0, 0) < 1e-6);
    struct FeatureMap np = in.maxpool2d(2UL, 2UL);   // unpadded path unchanged
    ASSERT_TRUE(pool_ref_err(&in, &np, 2UL, 2UL, 0UL, 1, 0) < 1e-6);
    // global pooling == a full-plane window
    struct FeatureMap ga = in.global_avgpool2d();
    struct FeatureMap gm = in.global_maxpool2d();
    ASSERT_EQ(ga.height, 1UL); ASSERT_EQ(ga.width, 1UL); ASSERT_EQ(gm.channels, 3UL);
    unsafe {
        unsigned long c = 0UL;
        while (c < 3UL) {
            double sum = 0.0; double best = -1.0e30;
            unsigned long i = 0UL;
            while (i < 63UL) {
                double v = (double)in.data[c * 63UL + i];
                sum = sum + v; if (v > best) best = v;
                i = i + 1UL;
            }
            double e = sum / 63.0 - (double)ga.data[c];
            ASSERT_TRUE(e < 1e-5 && e > -1e-5);
            ASSERT_TRUE((double)gm.data[c] == best);
            c = c + 1UL;
        }
    }
    ga.free(); gm.free(); np.free();
    a0.free(); a1.free(); mx.free(); in.free();
}

// feature_map_add and the ResNet basic block against the composition of
// its convolutions.
void t_residual() {
    struct FeatureMap a = FeatureMap::new(4UL, 6UL, 5UL);
    struct FeatureMap b = FeatureMap::new(4UL, 6UL, 5UL);
    fill_map(&a, 1UL); fill_map(&b, 77UL);
    struct FeatureMap s = feature_map_add(&a, &b, 0);
    struct FeatureMap r = feature_map_add(&a, &b, 1);
    unsafe {
        unsigned long i = 0UL;
        while (i < 120UL) {
            float v = a.data[i] + b.data[i];
            ASSERT_TRUE(s.data[i] == v);
            ASSERT_TRUE(r.data[i] == (v > (float)0.0 ? v : (float)0.0));
            i = i + 1UL;
        }
    }
    s.free(); r.free(); b.free();

    // stride 2 + channel change -> projection shortcut; stride 1 -> identity
    unsigned long strides[2];
    unsigned long outCs[2];
    strides[0] = 2UL; outCs[0] = 8UL;
    strides[1] = 1UL; outCs[1] = 4UL;
    int k = 0;
    while (k < 2) {
        struct ResidualBlock blk = ResidualBlock::new(4UL, outCs[k], strides[k]);
        ASSERT_EQ(blk.hasProjection, k == 0 ? 1 : 0);
        struct FeatureMap out = blk.forward(&a);
        ASSERT_EQ(out.channels, outCs[k]);
        ASSERT_EQ(out.height, (6UL - 1UL) / strides[k] + 1UL);
        ASSERT_EQ(out.width, (5UL - 1UL) / strides[k] + 1UL);
        struct FeatureMap h1 = blk.conv1.relu_forward(&a);
        struct FeatureMap h2 = blk.conv2.forward(&h1);
        unsigned long n = out.channels * out.height * out.width;
        unsafe {
            unsigned long i = 0UL;
            if (k == 0) {
                struct FeatureMap sc = blk.projection.forward(&a);
                while (i < n) {
                    float v = h2.data[i] + sc.data[i];
                    ASSERT_TRUE(out.data[i] == (v > (float)0.0 ? v : (float)0.0));
                    i = i + 1UL;
                }
                sc.free();
            } else {
                while (i < n) {
                    float v = h2.data[i] + a.data[i];
                    ASSERT_TRUE(out.data[i] == (v > (float)0.0 ? v : (float)0.0));
                    i = i + 1UL;
                }
            }
        }
        h1.free(); h2.free(); out.free();
        blk.free();
        k = k + 1;
    }
    a.free();
}

static &FloatTensor seeded_tensor_(unsigned long rows, unsigned long cols, unsigned long seed, float scale, float bias) {
    &FloatTensor t = FloatTensor::new_2d(rows, cols, 0);
    unsafe {
        unsigned long i = 0UL;
        while (i < rows * cols) {
            t->data[i] = (float)((double)(((i + seed) * 2654435761UL >> 7UL) % 1000UL) / 500.0 - 1.0) * scale + bias;
            i = i + 1UL;
        }
    }
    return t;
}

static void tensor_drop(&FloatTensor t) {
    unsafe { t->free(); }   // a heap tensor, freed below
    unsafe { free((void*)t); }
}

// The selective scan against a naive double-precision recurrence.
void t_ssm_scan() {
    unsigned long T = 23UL; unsigned long D = 40UL; unsigned long N = 16UL;
    &FloatTensor x = seeded_tensor_(T, D, 1UL, (float)1.0, (float)0.0);
    &FloatTensor dl = seeded_tensor_(T, D, 2UL, (float)0.04, (float)0.05);   // (0.01, 0.09)
    &FloatTensor A = seeded_tensor_(D, N, 3UL, (float)2.0, (float)-3.0);     // negative
    &FloatTensor B = seeded_tensor_(T, N, 4UL, (float)1.0, (float)0.0);
    &FloatTensor C = seeded_tensor_(T, N, 5UL, (float)1.0, (float)0.0);
    &FloatTensor Ds = seeded_tensor_(1UL, D, 6UL, (float)1.0, (float)0.0);
    &FloatTensor y = ssm_selective_scan(x, dl, A, B, C, Ds);
    double worst = 0.0;
    unsafe {
        double* h = (double*)malloc(sizeof(double) * N);
        unsigned long d = 0UL;
        while (d < D) {
            unsigned long n = 0UL;
            while (n < N) { h[n] = 0.0; n = n + 1UL; }
            unsigned long t = 0UL;
            while (t < T) {
                double dt = (double)dl->data[t * D + d];
                double xv = (double)x->data[t * D + d];
                double acc = 0.0;
                n = 0UL;
                while (n < N) {
                    h[n] = exp_d(dt * (double)A->data[d * N + n]) * h[n] + dt * (double)B->data[t * N + n] * xv;
                    acc = acc + (double)C->data[t * N + n] * h[n];
                    n = n + 1UL;
                }
                acc = acc + (double)Ds->data[d] * xv;
                double e = acc - (double)y->data[t * D + d];
                if (e < 0.0) e = -e;
                if (e > worst) worst = e;
                t = t + 1UL;
            }
            d = d + 1UL;
        }
        free((void*)h);
    }
    ASSERT_TRUE(worst < 1e-5);
    tensor_drop(x); tensor_drop(dl); tensor_drop(A); tensor_drop(B); tensor_drop(C); tensor_drop(Ds); tensor_drop(y);
}

// Mamba: the parallel forward equals token-by-token recurrent decoding,
// on both the serial and the threaded scan path.
void t_mamba() {
    unsigned long dms[2];
    dms[0] = 24UL; dms[1] = 160UL;
    int k = 0;
    while (k < 2) {
        unsigned long dm = dms[k];
        unsigned long T = 19UL;
        struct Mamba m = Mamba::new(dm, 16UL, 4UL, 2UL);
        ASSERT_EQ(m.dInner, 2UL * dm);
        ASSERT_EQ(m.dtRank, (dm + 15UL) / 16UL);
        &FloatTensor X = seeded_tensor_(T, dm, 9UL, (float)1.0, (float)0.0);
        &FloatTensor full = m.forward(X);
        unsigned long r0; unsigned long c0;
        unsafe { r0 = full->shape[0]; c0 = full->shape[1]; }
        ASSERT_EQ(r0, T);
        ASSERT_EQ(c0, dm);
        struct MambaState st = MambaState::new(&m);
        st.reset();
        float worst = (float)0.0;
        float mag = (float)0.0;
        unsigned long t = 0UL;
        while (t < T) {
            &FloatTensor xt = FloatTensor::new_2d(1UL, dm, 0);
            unsafe {
                unsigned long i = 0UL;
                while (i < dm) { xt->data[i] = X->data[t * dm + i]; i = i + 1UL; }
            }
            &FloatTensor yt = m.step(&st, xt);
            unsafe {
                unsigned long i = 0UL;
                while (i < dm) {
                    float a = full->data[t * dm + i];
                    float e = yt->data[i] - a;
                    if (e < (float)0.0) e = -e;
                    if (e > worst) worst = e;
                    if (a < (float)0.0) a = -a;
                    if (a > mag) mag = a;
                    i = i + 1UL;
                }
            }
            tensor_drop(xt); tensor_drop(yt);
            t = t + 1UL;
        }
        ASSERT_TRUE(mag > (float)1e-3);                 // not trivially zero
        ASSERT_TRUE(worst <= (float)1e-4 * mag + (float)1e-6);
        st.free();
        tensor_drop(X); tensor_drop(full);
        m.free();
        k = k + 1;
    }
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("fp16/bf16 rounding", t_round);
        t.add("tensor <-> half", t_convert);
        t.add("half ops", t_half_ops);
        t.add("half matmul f32 accumulate", t_matmul_accumulates_f32);
        t.add("half autograd", t_half_autograd);
        t.add("half gemv", t_half_gemv);
        t.add("Tensor::cast autograd", t_cast_autograd);
        t.add("tensor_matmul_lowp", t_matmul_lowp);
        t.add("loss scaler", t_loss_scaler);
        t.add("tensor arena", t_tensor_arena);
        t.add("conv2d (winograd / im2col)", t_conv2d);
        t.add("pooling (padded / global)", t_pooling);
        t.add("residual add + ResNet block", t_residual);
        t.add("selective scan", t_ssm_scan);
        t.add("mamba forward == recurrent steps", t_mamba);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
