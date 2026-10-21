// std::ml fp16/bf16: HalfTensor storage + ops, mixed-precision autograd,
// dynamic loss scaling. Expected values are exact fp16/bf16 roundings
// (checked against IEEE binary16 / bfloat16 bit patterns).
#include <std/test/test.h>
#include <std/ml/tensor_half.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/cnn.h>

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
        &Tensor t = tensor_from_2d((const float*)v, 2UL, 2UL, 0);
        struct HalfTensor h = tensor_to_half(t, TENSOR_FP16);
        ASSERT_EQ(h.ndim, 2);
        ASSERT_EQ(h.size, 4);
        ASSERT_TRUE(h.at2(0UL, 0UL) == (float)1.0);
        ASSERT_TRUE(h.at2(0UL, 1UL) == (float)-2.5);
        ASSERT_TRUE(h.at2(1UL, 0UL) == (float)0.0999755859375);   // 0.1 in fp16
        ASSERT_TRUE(h.at2(1UL, 1UL) == (float)1000.0);
        &Tensor back = half_to_tensor(&h, 0);
        ASSERT_TRUE(back.at2(1UL, 0UL) == (float)0.0999755859375);
        h.set1(3UL, (float)3.0);
        ASSERT_TRUE(h.at1(3UL) == (float)3.0);

        struct HalfTensor b = tensor_to_half(t, TENSOR_BF16);
        ASSERT_TRUE(b.at1(2UL) == (float)0.10009765625);            // 0.1 in bf16
        b.free();
        h.free();
        back.free();
        t.free();
    }
}

void t_half_ops() {
    struct HalfTensor a = half_tensor_new_2d(2UL, 2UL, TENSOR_FP16);
    struct HalfTensor b = half_tensor_new_2d(2UL, 2UL, TENSOR_FP16);
    a.set2(0UL, 0UL, (float)1.0); a.set2(0UL, 1UL, (float)2.0);
    a.set2(1UL, 0UL, (float)3.0); a.set2(1UL, 1UL, (float)-4.0);
    b.set2(0UL, 0UL, (float)0.5); b.set2(0UL, 1UL, (float)1.0);
    b.set2(1UL, 0UL, (float)1.5); b.set2(1UL, 1UL, (float)2.0);

    struct HalfTensor s = half_add(&a, &b);
    ASSERT_TRUE(s.at2(1UL, 1UL) == (float)-2.0);
    struct HalfTensor d = half_sub(&a, &b);
    ASSERT_TRUE(d.at2(0UL, 0UL) == (float)0.5);
    struct HalfTensor m = half_mul(&a, &b);
    ASSERT_TRUE(m.at2(1UL, 0UL) == (float)4.5);
    struct HalfTensor k = half_scale(&a, (float)0.5);
    ASSERT_TRUE(k.at2(0UL, 1UL) == (float)1.0);
    struct HalfTensor r = half_relu(&a);
    ASSERT_TRUE(r.at2(1UL, 1UL) == (float)0.0);
    ASSERT_TRUE(r.at2(1UL, 0UL) == (float)3.0);

    // [[1,2],[3,-4]] @ [[0.5,1],[1.5,2]] = [[3.5,5],[-4.5,-5]]
    struct HalfTensor p = half_matmul(&a, &b);
    ASSERT_EQ(p.shape[0], 2);
    ASSERT_EQ(p.shape[1], 2);
    ASSERT_TRUE(p.at2(0UL, 0UL) == (float)3.5);
    ASSERT_TRUE(p.at2(0UL, 1UL) == (float)5.0);
    ASSERT_TRUE(p.at2(1UL, 0UL) == (float)-4.5);
    ASSERT_TRUE(p.at2(1UL, 1UL) == (float)-5.0);
    ASSERT_TRUE(half_sum(&a) == (float)2.0);

    // Mismatched shape -> empty result.
    struct HalfTensor v = half_tensor_new_1d(3UL, TENSOR_FP16);
    struct HalfTensor bad = half_add(&a, &v);
    ASSERT_EQ(bad.size, 0);
    // Mismatched dtype -> empty result.
    struct HalfTensor bf = half_tensor_new_2d(2UL, 2UL, TENSOR_BF16);
    struct HalfTensor bad2 = half_matmul(&a, &bf);
    ASSERT_EQ(bad2.size, 0);

    s.free(); d.free(); m.free(); k.free(); r.free(); p.free();
    v.free(); bad.free(); bf.free(); bad2.free(); a.free(); b.free();
}

void t_matmul_accumulates_f32() {
    // 2048 * (1 * 1): an fp16 running sum would stall at 2048 + 1 == 2048;
    // float32 accumulation gives the exact 4096 (representable in fp16).
    struct HalfTensor a = half_tensor_new_2d(1UL, 4096UL, TENSOR_FP16);
    struct HalfTensor b = half_tensor_new_2d(4096UL, 1UL, TENSOR_FP16);
    unsigned long i = 0UL;
    while (i < 4096UL) { a.set1(i, (float)1.0); b.set1(i, (float)1.0); i = i + 1UL; }
    struct HalfTensor p = half_matmul(&a, &b);
    ASSERT_TRUE(p.at1(0UL) == (float)4096.0);
    p.free(); a.free(); b.free();
}

void t_half_gemv() {
    // both dtypes; n * k large enough to take the threaded path, with a
    // k that isn't a multiple of 8 (tail loop)
    int dts[2];
    dts[0] = TENSOR_FP16; dts[1] = TENSOR_BF16;
    int di = 0;
    while (di < 2) {
        unsigned long n = 301UL;
        unsigned long k = 1029UL;
        struct HalfTensor w = half_tensor_new_2d(n, k, dts[di]);
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
        unsafe { ok = half_gemv(&w, (const &float)x, (&float)y); }
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
        // half_gemv_batch: m vectors in one pass equal m half_gemv calls
        unsigned long m = 5UL;
        float* xs;
        float* ys;
        unsafe {
            xs = (float*)malloc(sizeof(float) * k * m);
            ys = (float*)malloc(sizeof(float) * n * m);
            i = 0UL; while (i < k * m) { xs[i] = (float)((i * 13UL) % 31UL) * (float)0.02 - (float)0.3; i = i + 1UL; }
            ok = half_gemv_batch(&w, (const &float)xs, m, (&float)ys);
        }
        ASSERT_TRUE(ok);
        int same = 1;
        unsigned long v = 0UL;
        while (v < m) {
            unsafe {
                half_gemv(&w, (const &float)&xs[v * k], (&float)y);
                r = 0UL; while (r < n) { if (y[r] != ys[v * n + r]) same = 0; r = r + 1UL; }
            }
            v = v + 1UL;
        }
        ASSERT_TRUE(same);
        unsafe { free((void*)x); free((void*)y); free((void*)xs); free((void*)ys); }
        w.free();
        di = di + 1;
    }
    // fp16 widening edge cases: subnormal, negative, +-inf (inf * 1 = inf)
    struct HalfTensor e = half_tensor_new_2d(4UL, 1UL, TENSOR_FP16);
    unsafe {
        e.data[0] = (unsigned short)0x0001;   // smallest subnormal, 2^-24
        e.data[1] = (unsigned short)0xc500;   // -5
        e.data[2] = (unsigned short)0x7c00;   // +inf
        e.data[3] = (unsigned short)0xfc00;   // -inf
    }
    float one[1];
    float out[4];
    unsafe {
        one[0] = (float)1.0;
        ASSERT_TRUE(half_gemv(&e, (const &float)&one[0], (&float)&out[0]));
    }
    ASSERT_TRUE(out[0] == (float)5.960464477539063e-08);
    ASSERT_TRUE(out[1] == (float)-5.0);
    ASSERT_TRUE(out[2] > (float)3.0e38 && out[2] * (float)0.5 == out[2]);
    ASSERT_TRUE(out[3] < (float)-3.0e38 && out[3] * (float)0.5 == out[3]);
    // a 1D tensor is not a matrix
    struct HalfTensor v = half_tensor_new_1d(4UL, TENSOR_BF16);
    unsafe { ASSERT_TRUE(!half_gemv(&v, (const &float)&one[0], (&float)&out[0])); }
    e.free(); v.free();
}

void t_cast_autograd() {
    unsafe {
        float av[2] = { (float)0.1, (float)0.2 };
        &Tensor x = tensor_from_1d((const float*)av, 2UL, 1);
        &Tensor c = tensor_cast(x, TENSOR_FP16);
        ASSERT_TRUE(c.at1(0UL) == (float)0.0999755859375);
        &Tensor y = tensor_sum(tensor_mul(c, c));      // sum(cast(x)^2)
        tensor_backward(y);
        // Straight-through: dy/dx = 2 * cast(x).
        ASSERT_TRUE(near(x.grad[0], (float)2.0 * (float)0.0999755859375, (float)1.0e-6));
        ASSERT_TRUE(near(x.grad[1], (float)2.0 * c.at1(1UL), (float)1.0e-6));
    }
}

void t_matmul_lowp() {
    unsafe {
        float av[4] = { (float)1.0, (float)2.0, (float)3.0, (float)4.0 };
        float bv[4] = { (float)0.1, (float)0.0, (float)0.0, (float)0.1 };
        &Tensor a = tensor_from_2d((const float*)av, 2UL, 2UL, 1);
        &Tensor b = tensor_from_2d((const float*)bv, 2UL, 2UL, 1);
        &Tensor out = tensor_matmul_lowp(a, b, TENSOR_BF16);
        float b01 = half_round((float)0.1, TENSOR_BF16);          // 0.10009765625
        ASSERT_TRUE(out.at2(0UL, 0UL) == half_round((float)1.0 * b01, TENSOR_BF16));
        ASSERT_TRUE(out.at2(1UL, 1UL) == half_round((float)4.0 * b01, TENSOR_BF16));
        tensor_backward(tensor_sum(out));
        // d sum(A B)/dA[i][j] = sum_n B[j][n] (of the rounded B), per row.
        ASSERT_TRUE(near(a.grad[0], b01, (float)1.0e-6));
        ASSERT_TRUE(near(b.grad[0], (float)4.0, (float)1.0e-6));  // A[0][0] + A[1][0]
    }
}

void t_loss_scaler() {
    unsafe {
        struct LossScaler s = loss_scaler_new();
        ASSERT_TRUE(s.scale == (float)65536.0);
        float v[2] = { (float)1.0, (float)2.0 };
        &Tensor p = tensor_from_1d((const float*)v, 2UL, 1);
        tensor_backward(tensor_sum(tensor_scale(p, s.scale)));    // grads = 65536
        ASSERT_TRUE(loss_scaler_unscale(&s, p) == 1);
        ASSERT_TRUE(p.grad[0] == (float)1.0);
        loss_scaler_update(&s, 1);
        ASSERT_EQ(s.goodSteps, 1);

        // Overflowed gradient -> 0, scale halves, counter resets.
        float big = (float)3.0e38;
        p.grad[0] = big * (float)10.0;                             // +inf
        ASSERT_TRUE(loss_scaler_unscale(&s, p) == 0);
        loss_scaler_update(&s, 0);
        ASSERT_TRUE(s.scale == (float)32768.0);
        ASSERT_EQ(s.goodSteps, 0);

        // Growth after growthInterval clean steps.
        s.growthInterval = 2;
        loss_scaler_update(&s, 1);
        loss_scaler_update(&s, 1);
        ASSERT_TRUE(s.scale == (float)65536.0);
    }
}


// The same training loop on the heap and in a TensorArena: identical
// losses and weights, parameters' gradients survive resets, and the arena
// stops growing after the first step.
static float arena_train(int useArena, &Tensor W, const &Tensor X, unsigned long* chunkBytes) {
    struct TensorArena a = tensor_arena_new(65536UL);
    float last = (float)0.0;
    int step = 0;
    unsigned long used0 = 0UL;
    while (step < 20) {
        if (useArena) tensor_arena_begin(&a);
        tensor_zero_grad(W);
        &Tensor y = tensor_matmul(X, W);
        &Tensor r = tensor_relu(y);
        &Tensor l = tensor_sum(tensor_mul(r, r));
        tensor_backward(l);
        unsafe {
            last = l->data[0];
            unsigned long i = 0UL;
            while (i < W->size) { W->data[i] = W->data[i] - (float)0.01 * W->grad[i]; i = i + 1UL; }
        }
        if (useArena) {
            tensor_arena_end();
            if (step == 0) used0 = tensor_arena_used(&a);
            if (step == 19) { unsafe { *chunkBytes = tensor_arena_used(&a) == used0 ? 1UL : 0UL; } }
            tensor_arena_reset(&a);
            ASSERT_TRUE(tensor_arena_used(&a) == 0UL);
        }
        step = step + 1;
    }
    tensor_arena_free(&a);
    return last;
}

void t_tensor_arena() {
    float xs[64];
    float ws[32];
    int i = 0;
    while (i < 64) { xs[i] = (float)((i * 37) % 17 - 8) * (float)0.1; i = i + 1; }
    i = 0;
    while (i < 32) { ws[i] = (float)((i * 11) % 7 - 3) * (float)0.05; i = i + 1; }
    &Tensor X = tensor_from_2d(&xs[0], 8UL, 8UL, 0);
    &Tensor W1 = tensor_from_2d(&ws[0], 8UL, 4UL, 1);
    &Tensor W2 = tensor_from_2d(&ws[0], 8UL, 4UL, 1);
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
    struct TensorArena ar = tensor_arena_new(0UL);
    tensor_arena_begin(&ar);
    &Tensor t = tensor_new_2d(3UL, 5UL, 0);
    tensor_arena_end();
    ASSERT_FALSE(t->owner.is_null());
    t.free();
    ASSERT_TRUE(tensor_arena_used(&ar) > 0UL);
    tensor_arena_free(&ar);
}

// conv2d against a direct sum: 3x3 stride 1 (Winograd F(2x2,3x3)) with and
// without padding, odd sizes so the last tiles hang over the edge, ReLU,
// and a 5x5 / stride-2 layer (im2col).
static double conv_ref_err(&Conv2D layer, unsigned long inC, unsigned long H, unsigned long W, int relu) {
    struct FeatureMap in = feature_map_new(inC, H, W);
    unsigned long n = inC * H * W;
    unsigned long i = 0UL;
    unsafe { while (i < n) { in.data[i] = (float)((double)((i * 7919UL) % 1000UL) / 500.0 - 1.0); i = i + 1UL; } }
    struct FeatureMap out = relu ? conv2d_relu_forward(layer, &in) : conv2d_forward(layer, &in);
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
    feature_map_free(&in);
    feature_map_free(&out);
    return worst;
}

void t_conv2d() {
    struct Conv2D a = conv2d_new(16UL, 24UL, 3UL, 3UL, 1UL, 1UL);
    ASSERT_TRUE(conv_ref_err(&a, 16UL, 13UL, 11UL, 0) < 1e-4);
    ASSERT_TRUE(conv_ref_err(&a, 16UL, 13UL, 11UL, 1) < 1e-4);
    ASSERT_TRUE(conv_ref_err(&a, 16UL, 40UL, 37UL, 0) < 1e-4);   // several tile blocks
    conv2d_free(&a);
    struct Conv2D b = conv2d_new(12UL, 8UL, 3UL, 3UL, 1UL, 0UL);  // no padding
    ASSERT_TRUE(conv_ref_err(&b, 12UL, 15UL, 10UL, 0) < 1e-4);
    conv2d_free(&b);
    // Inside a TensorArena the output (and every intermediate) comes from
    // the arena: same values, feature_map_free leaves it to the reset.
    struct Conv2D d = conv2d_new(8UL, 8UL, 3UL, 3UL, 1UL, 1UL);
    struct FeatureMap in = feature_map_new(8UL, 20UL, 20UL);
    unsafe { unsigned long q = 0UL; while (q < 3200UL) { in.data[q] = (float)((double)(q % 17UL) / 8.0 - 1.0); q = q + 1UL; } }
    struct FeatureMap heapOut = conv2d_forward(&d, &in);
    ASSERT_EQ(heapOut.fromArena, 0);
    struct TensorArena ar = tensor_arena_new(0UL);
    tensor_arena_begin(&ar);
    struct FeatureMap arOut = conv2d_relu_forward(&d, &in);
    struct FeatureMap tmp = feature_map_new(4UL, 4UL, 4UL);
    tensor_arena_end();
    ASSERT_EQ(arOut.fromArena, 1);
    ASSERT_EQ(tmp.fromArena, 1);
    unsafe { ASSERT_TRUE(tmp.data[63] == (float)0.0); }
    ASSERT_TRUE(tensor_arena_used(&ar) >= 8UL * 20UL * 20UL * 4UL);
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
    feature_map_free(&arOut);   // no-op for an arena map
    feature_map_free(&tmp);
    tensor_arena_reset(&ar);
    ASSERT_EQ(tensor_arena_used(&ar), 0);
    tensor_arena_free(&ar);
    feature_map_free(&heapOut);
    feature_map_free(&in);
    conv2d_free(&d);
    struct Conv2D c = conv2d_new(5UL, 7UL, 5UL, 5UL, 2UL, 2UL);   // im2col
    ASSERT_TRUE(conv_ref_err(&c, 5UL, 17UL, 14UL, 0) < 1e-4);
    conv2d_free(&c);
}

int main() {
    struct TestSuite t = test_suite_init();
    unsafe {
        t.add("fp16/bf16 rounding", t_round);
        t.add("tensor <-> half", t_convert);
        t.add("half ops", t_half_ops);
        t.add("half matmul f32 accumulate", t_matmul_accumulates_f32);
        t.add("half gemv", t_half_gemv);
        t.add("tensor_cast autograd", t_cast_autograd);
        t.add("tensor_matmul_lowp", t_matmul_lowp);
        t.add("loss scaler", t_loss_scaler);
        t.add("tensor arena", t_tensor_arena);
        t.add("conv2d (winograd / im2col)", t_conv2d);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
