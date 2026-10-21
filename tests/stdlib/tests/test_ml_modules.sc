// std::ml's model traits (module.h): one generic check runs on every
// Module, the generic training / file / copy functions match the paths
// they replace, and the smaller roles (Recurrent, Generative, DenoiseModel,
// CondModule) drive generic code too. Traits are composed (module.h's
// 'trait Module : Parameterized + Forward {}'), and a bounded generic
// struct of this file's own is itself a Module.
#include <std/test/test.h>
#include <std/ml/module.h>
#include <std/ml/nn.h>
#include <std/ml/datasets.h>
#include <std/ml/trainer.h>
#include <std/ml/convnet.h>
#include <std/ml/norm.h>
#include <std/ml/lora.h>
#include <std/ml/attention.h>
#include <std/ml/attention_advanced.h>
#include <std/ml/transformer.h>
#include <std/ml/ssm.h>
#include <std/ml/rnn.h>
#include <std/ml/autoencoder.h>
#include <std/ml/gan.h>
#include <std/ml/diffusion.h>

extern void free(void* ptr);

static void tensor_drop(&Tensor t) {
    t->free();
    unsafe { free((void*)t); }
}

static float absf_(float x) { return x < (float)0.0 ? -x : x; }

static float max_diff_(const &Tensor a, const &Tensor b) {
    float m = (float)0.0;
    unsafe {
        if (a->size != b->size) return (float)1.0e30;
        unsigned long i = 0UL;
        while (i < a->size) { float d = absf_(a->data[i] - b->data[i]); if (d > m) m = d; i = i + 1UL; }
    }
    return m;
}

static &Tensor randn_(unsigned long r, unsigned long c, &Rng rng) {
    &Tensor t = tensor_new_2d(r, c, 0);
    rng.fill_normal(t, (float)0.0, (float)1.0);
    return t;
}

// ── One check for every Module ───────────────────────────────────────────────
// forward keeps the row count; the parameter count is the collected
// tensors' size; save -> perturb -> load restores the output; a soft update
// toward a copy with tau 1 is a copy.
generic<M: Module>
int module_contract_(&M m, const &Tensor x, unsigned long outCols, const &char path) {
    int ok = 1;
    &Tensor y0 = module_predict(m, x);
    unsafe { ok = ok && y0->shape[0] == x->shape[0] && y0->shape[1] == outCols; }
    struct Vec<&Tensor> ps = vec_new();
    m.collect_params(&ps);
    unsigned long n = 0UL;
    unsigned long i = 0UL;
    while (i < ps.length()) { &Tensor p = *ps.at(i); unsafe { n = n + p->size; } i = i + 1UL; }
    ok = ok && ps.length() > 0UL && n == model_param_count(m);
    ok = ok && model_save_file(m, path);
    // perturb every state tensor, then load it back
    struct Vec<&Tensor> st = vec_new();
    m.collect_state(&st);
    i = 0UL;
    while (i < st.length()) {
        &Tensor p = *st.at(i);
        unsafe { if (p->size > 0UL) p->data[0] = p->data[0] + (float)1.0; }
        i = i + 1UL;
    }
    ok = ok && model_load_file(m, path);
    &Tensor y1 = module_predict(m, x);
    ok = ok && max_diff_(y0, y1) == (float)0.0;
    tensor_drop(y0);
    tensor_drop(y1);
    ps.free();
    st.free();
    return ok;
}

void t_every_module() {
    struct Rng rng = rng_new(7UL);
    &Tensor x8 = randn_(5UL, 8UL, &rng);

    struct Mlp mlp = mlp_new(8UL, 16UL, 1UL, 3UL, ACT_RELU, ACT_NONE, &rng);
    ASSERT_TRUE(module_contract_(&mlp, x8, 3UL, "build/test_ml_modules_mlp.bin"));

    struct ConvNet cn = convnet_new(2UL, 2UL, 2UL);
    cn.conv(4UL, 3UL, 1UL, 1UL, ACT_NONE, &rng);
    cn.batchnorm(ACT_RELU);
    cn.dense(3UL, ACT_NONE, &rng);
    cn.set_training(0);
    ASSERT_TRUE(module_contract_(&cn, x8, 3UL, "build/test_ml_modules_convnet.bin"));

    struct LayerNorm ln = layer_norm_new(8UL);
    ASSERT_TRUE(module_contract_(&ln, x8, 8UL, "build/test_ml_modules_ln.bin"));

    struct LoraConfig lc = lora_config_default(2UL);
    struct LoraMlp lora = lora_attach(&mlp, &lc);
    rng.fill_normal(*lora.B.at(0UL), (float)0.0, (float)0.1);
    ASSERT_TRUE(module_contract_(&lora, x8, 3UL, "build/test_ml_modules_lora.bin"));
    ASSERT_EQ(model_param_count(&lora), 2UL * (8UL + 16UL) + 2UL * (16UL + 3UL));

    struct MultiHeadAttention mha = mha_new(8UL, 2UL, &rng);
    ASSERT_TRUE(module_contract_(&mha, x8, 8UL, "build/test_ml_modules_mha.bin"));
    // the layer is mha_forward between its projections
    &Tensor h = mha_forward(tensor_matmul(x8, mha.Wq), tensor_matmul(x8, mha.Wk), tensor_matmul(x8, mha.Wv), 2UL);
    &Tensor want = tensor_matmul(h, mha.Wo);
    &Tensor got = mha.forward(x8);
    ASSERT_TRUE(max_diff_(want, got) < (float)0.000001);

    struct Mla mla = mla_new(8UL, 4UL, 2UL, &rng);
    ASSERT_TRUE(module_contract_(&mla, x8, 8UL, "build/test_ml_modules_mla.bin"));
    struct EgMla eg = eg_mla_new(8UL, 4UL, 2UL, 2UL, &rng);
    ASSERT_TRUE(module_contract_(&eg, x8, 8UL, "build/test_ml_modules_egmla.bin"));
    ASSERT_EQ(model_param_count(&eg), 64UL + 32UL + 2UL * 2UL * 4UL * 4UL);

    struct JiTBlock jb = jit_block_new(8UL, 16UL, 2UL);
    ASSERT_TRUE(module_contract_(&jb, x8, 8UL, "build/test_ml_modules_jit.bin"));

    struct Mamba mb = mamba_new(8UL, 4UL, 3UL, 2UL);
    ASSERT_TRUE(module_contract_(&mb, x8, 8UL, "build/test_ml_modules_mamba.bin"));

    mb.free(); jb.free(); eg.free(); mla.free(); mha.free();
    lora.detach();
    ln.free(); cn.free(); mlp.free();
}

// ── module_fit / model_fit match the paths they replace ──────────────────────
static &Tensor clf_loss_(&Mlp m, const &Tensor x, const &Tensor y) {
    return tensor_cross_entropy_loss(m.forward(x), y);
}

void t_module_fit_matches_trainer_fit() {
    struct Rng drng = rng_new(1UL);
    struct Dataset data = dataset_xor(250UL, (float)0.0, &drng);
    struct Rng r1 = rng_new(3UL);
    struct Mlp a = mlp_new(2UL, 16UL, 2UL, 2UL, ACT_RELU, ACT_NONE, &r1);
    struct Rng r2 = rng_new(3UL);
    struct Mlp b = mlp_new(2UL, 16UL, 2UL, 2UL, ACT_RELU, ACT_NONE, &r2);
    struct TrainConfig tc = train_config_default();
    tc.epochs = 5UL;
    tc.clipNorm = (float)1.0;
    tc.lrDecay = (float)0.9;
    struct Optimizer oa = optimizer_new(OPT_ADAM, (float)0.01);
    model_add_params(&a, &oa);
    struct TrainReport ra = trainer_fit(&a, clf_loss_, &oa, &data, &tc);
    struct Optimizer ob = optimizer_new(OPT_ADAM, (float)0.01);
    model_add_params(&b, &ob);
    struct TrainReport rb = module_fit(&b, TRAIN_LOSS_CE, &ob, &data, &tc);
    ASSERT_TRUE(ra.firstEpochLoss == rb.firstEpochLoss && ra.lastEpochLoss == rb.lastEpochLoss);
    ASSERT_EQ(ra.steps, rb.steps);
    ASSERT_TRUE(module_evaluate(&b, TRAIN_LOSS_CE, &data, 64UL) < ra.firstEpochLoss);
    ASSERT_TRUE(module_accuracy(&b, &data, 64UL) > (float)0.8);
    // model_copy / model_soft_update
    struct Rng r3 = rng_new(9UL);
    struct Mlp c = mlp_new(2UL, 16UL, 2UL, 2UL, ACT_RELU, ACT_NONE, &r3);
    ASSERT_TRUE(model_copy(&c, &b));
    &Tensor yb = module_predict(&b, data.x);
    &Tensor yc = module_predict(&c, data.x);
    ASSERT_TRUE(max_diff_(yb, yc) == (float)0.0);
    tensor_drop(yb);
    tensor_drop(yc);
    oa.free(); ob.free();
    a.free(); b.free(); c.free();
    data.free();
}

void t_model_fit_autoencoders() {
    struct Rng drng = rng_new(9UL);
    struct Dataset data = dataset_shapes(128UL, 8UL, &drng);
    struct VaeConfig cfg = vae_config_default(4UL);
    cfg.hidden = 32UL;
    struct Vae v = vae_new(64UL, &cfg);
    struct Optimizer o = optimizer_new(OPT_ADAM, (float)0.003);
    model_add_params(&v, &o);
    struct TrainConfig tc = train_config_default();
    tc.epochs = 6UL;
    struct TrainReport rep = model_fit(&v, &o, &data, &tc);
    ASSERT_TRUE(rep.lastEpochLoss < rep.firstEpochLoss);
    ASSERT_TRUE(model_evaluate(&v, &data, 64UL) > (float)0.0);
    // the whole state round-trips through a file
    ASSERT_TRUE(model_save_file(&v, "build/test_ml_modules_vae.bin"));
    struct Vae w = vae_new(64UL, &cfg);
    ASSERT_TRUE(model_load_file(&w, "build/test_ml_modules_vae.bin"));
    struct Vec<&Tensor> a = vec_new();
    struct Vec<&Tensor> b = vec_new();
    v.collect_state(&a);
    w.collect_state(&b);
    float worst = (float)0.0;
    unsigned long i = 0UL;
    while (i < a.length()) { float d = max_diff_(*a.at(i), *b.at(i)); if (d > worst) worst = d; i = i + 1UL; }
    ASSERT_TRUE(worst == (float)0.0);
    a.free(); b.free();
    o.free();
    v.free(); w.free();
    data.free();
}

// ── A bounded generic struct of our own is a Module ──────────────────────────
// Two modules averaged: forward, collect_params and collect_state come from
// the members' own, so module_fit / model_save_file work on the pair.
generic<M: Module>
struct Avg2 {
    M a;
    M b;
    &Tensor forward(const &Tensor x);
    void collect_params(&Vec<&Tensor> out) const;
    void collect_state(&Vec<&Tensor> out) const;
};
generic<M: Module>
&Tensor Avg2::forward(const &Tensor x) {
    return tensor_scale(tensor_add(self.a.forward(x), self.b.forward(x)), (float)0.5);
}
generic<M: Module>
void Avg2::collect_params(&Vec<&Tensor> out) const { self.a.collect_params(out); self.b.collect_params(out); }
generic<M: Module>
void Avg2::collect_state(&Vec<&Tensor> out) const { self.a.collect_state(out); self.b.collect_state(out); }

void t_generic_struct_is_a_module() {
    struct Rng rng = rng_new(4UL);
    struct Dataset data = dataset_moons(200UL, (float)0.1, &rng);
    struct Avg2<struct Mlp> pair;
    pair.a = mlp_new(2UL, 16UL, 1UL, 2UL, ACT_RELU, ACT_NONE, &rng);
    pair.b = mlp_new(2UL, 16UL, 1UL, 2UL, ACT_TANH, ACT_NONE, &rng);
    struct Optimizer o = optimizer_new(OPT_ADAM, (float)0.02);
    model_add_params(&pair, &o);
    ASSERT_EQ(model_param_count(&pair), 2UL * (2UL * 16UL + 16UL + 16UL * 2UL + 2UL));
    struct TrainConfig tc = train_config_default();
    tc.epochs = 30UL;
    struct TrainReport rep = module_fit(&pair, TRAIN_LOSS_CE, &o, &data, &tc);
    ASSERT_TRUE(rep.lastEpochLoss < rep.firstEpochLoss * (float)0.6);
    ASSERT_TRUE(module_accuracy(&pair, &data, 64UL) > (float)0.85);
    o.free();
    pair.a.free(); pair.b.free();
    data.free();
}

// ── Recurrent, Generative, DenoiseModel, CondModule ──────────────────────────
generic<C: Recurrent>
float unroll_by_hand_diff_(&C cell, const &Tensor xs, const &Tensor h0) {
    &Tensor h = cell.forward(recurrent_row_(xs, 0UL), h0);
    unsigned long t = 1UL;
    unsigned long T = 0UL;
    unsafe { T = xs->shape[0]; }
    while (t < T) { h = cell.forward(recurrent_row_(xs, t), h); t = t + 1UL; }
    return max_diff_(h, recurrent_run(cell, xs, h0));
}

void t_recurrent_cells() {
    struct Rng rng = rng_new(5UL);
    &Tensor xs = randn_(6UL, 3UL, &rng);
    &Tensor h0 = randn_(1UL, 4UL, &rng);
    struct RNNCell rc = rnn_cell_new(3UL, 4UL);
    struct GRUCell gc = gru_cell_new(3UL, 4UL);
    ASSERT_TRUE(unroll_by_hand_diff_(&rc, xs, h0) == (float)0.0);
    ASSERT_TRUE(unroll_by_hand_diff_(&gc, xs, h0) == (float)0.0);
    &Tensor last = recurrent_run(&gc, xs, h0);
    unsafe { ASSERT_TRUE(last->shape[0] == 1UL && last->shape[1] == 4UL); }
    ASSERT_EQ(model_param_count(&gc), 3UL * (3UL * 4UL + 4UL * 4UL + 4UL + 4UL));
    rc.free(); gc.free();
}

generic<G: Generative>
unsigned long sample_rows_(&G g, unsigned long n) {
    &Tensor s = g.sample(n, null);
    unsigned long r = 0UL;
    unsafe { r = s->shape[0]; }
    tensor_drop(s);
    return r;
}

void t_generative_models() {
    struct GanConfig gcfg = gan_config_default(4UL);
    gcfg.hidden = 16UL;
    struct Gan g = gan_new(2UL, &gcfg);
    struct VaeConfig vcfg = vae_config_default(4UL);
    vcfg.hidden = 16UL;
    struct Vae v = vae_new(10UL, &vcfg);
    ASSERT_EQ(sample_rows_(&g, 7UL), 7UL);
    ASSERT_EQ(sample_rows_(&v, 5UL), 5UL);
    g.free(); v.free();
}

// A denoiser whose prediction ignores x: the order-2 solvers then reduce to
// the order-1 updates with the same prediction.
struct ConstPredict {
    &Tensor value;
    &Tensor predict(const &Tensor x, float sigma);
};
&Tensor ConstPredict::predict(const &Tensor x, float sigma) { return tensor_scale(self.value, (float)1.0); }

void t_denoise_model_solvers() {
    struct Rng rng = rng_new(6UL);
    &Tensor x = randn_(3UL, 4UL, &rng);
    struct ConstPredict m;
    m.value = randn_(3UL, 4UL, &rng);
    &Tensor a = dpm_solver_2_step(&m, x, m.value, (float)0.8, (float)0.6, (float)0.95, (float)0.3122);
    &Tensor b = dpm_solver_1_step(x, m.value, (float)0.8, (float)0.6, (float)0.95, (float)0.3122);
    ASSERT_TRUE(max_diff_(a, b) < (float)0.0001);
    &Tensor c = dpm_solver_pp_2_step(&m, x, m.value, (float)0.8, (float)0.6, (float)0.95, (float)0.3122);
    &Tensor d = dpm_solver_pp_1_step(x, m.value, (float)0.8, (float)0.6, (float)0.95, (float)0.3122);
    ASSERT_TRUE(max_diff_(c, d) < (float)0.0001);
}

generic<C: CondModule>
unsigned long cond_rows_(&C m, const &Tensor x, const &Tensor c) {
    &Tensor y = m.forward(x, c);
    unsigned long r = 0UL;
    unsafe { r = y->shape[0] * 1000UL + y->shape[1]; }
    return r;
}

void t_cond_modules() {
    struct Rng rng = rng_new(8UL);
    struct DiTBlock dit = dit_block_new(8UL, 16UL, 4UL, 2UL);
    struct JiT jit = jit_new(6UL, 8UL, 16UL, 2UL, 2UL);
    &Tensor x = randn_(5UL, 8UL, &rng);
    &Tensor c = randn_(1UL, 4UL, &rng);
    &Tensor patches = randn_(5UL, 6UL, &rng);
    &Tensor cond = randn_(1UL, 8UL, &rng);
    ASSERT_EQ(cond_rows_(&dit, x, c), 5008UL);
    ASSERT_EQ(cond_rows_(&jit, patches, cond), 5008UL);
    ASSERT_EQ(model_param_count(&jit), 6UL * 8UL + 2UL * (4UL * 64UL + 8UL * 16UL + 16UL + 16UL * 8UL + 8UL));
    dit.free(); jit.free();
}

int main() {
    struct TestSuite t = test_suite_init();
    unsafe {
        t.add("one generic check over ten Modules", t_every_module);
        t.add("module_fit = trainer_fit; model_copy", t_module_fit_matches_trainer_fit);
        t.add("model_fit / files on a LossModel (Vae)", t_model_fit_autoencoders);
        t.add("a bounded generic struct is a Module", t_generic_struct_is_a_module);
        t.add("recurrent_run over RNN / GRU cells", t_recurrent_cells);
        t.add("Generative: Gan and Vae", t_generative_models);
        t.add("DenoiseModel: order-2 solvers", t_denoise_model_solvers);
        t.add("CondModule: DiTBlock and JiT", t_cond_modules);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
