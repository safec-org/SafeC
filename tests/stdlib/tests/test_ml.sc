// std::ml fp16/bf16: HalfTensor storage + ops, mixed-precision autograd,
// dynamic loss scaling. Expected values are exact fp16/bf16 roundings
// (checked against IEEE binary16 / bfloat16 bit patterns).
#include <std/test/test.h>
#include <std/ml/tensor_half.h>

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

int main() {
    struct TestSuite t = test_suite_init();
    unsafe {
        t.add("fp16/bf16 rounding", t_round);
        t.add("tensor <-> half", t_convert);
        t.add("half ops", t_half_ops);
        t.add("half matmul f32 accumulate", t_matmul_accumulates_f32);
        t.add("tensor_cast autograd", t_cast_autograd);
        t.add("tensor_matmul_lowp", t_matmul_lowp);
        t.add("loss scaler", t_loss_scaler);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
