// std::ml's training traits (train_traits.h) and the backend-neutral names
// (train.h) on the CPU backend: every trainer reached through the traits
// trains exactly as the direct API it wraps (trainer_fit, Gan::train, dqn_*,
// pg_*). tests/stdlib/rocm/train_backend_check.sc runs one program on both
// backends.
#include <std/test/test.h>
#include <std/ml/train.h>
#include <std/ml/module.h>
#include <std/ml/nn.h>
#include <std/ml/datasets.h>
#include <std/ml/trainer.h>
#include <std/ml/convnet.h>
#include <std/ml/autoencoder.h>
#include <std/ml/gan.h>
#include <std/ml/rl.h>

extern void free(void* ptr);

static void tensor_drop(&Tensor t) {
    t->free();
    unsafe { free((void*)t); }
}

static float absf_(float x) { return x < (float)0.0 ? -x : x; }

// max |a - b| over two parameter lists of the same shapes
static float params_diff_(const &Vec<&Tensor> a, const &Vec<&Tensor> b) {
    float worst = (float)0.0;
    if (a.length() != b.length()) return (float)1.0e30;
    unsigned long i = 0UL;
    while (i < a.length()) {
        &Tensor x = *a.at(i);
        &Tensor y = *b.at(i);
        unsafe {
            if (x->size != y->size) return (float)1.0e30;
            unsigned long j = 0UL;
            while (j < x->size) { float d = absf_(x->data[j] - y->data[j]); if (d > worst) worst = d; j = j + 1UL; }
        }
        i = i + 1UL;
    }
    return worst;
}

static int same_report_(const &TrainReport a, const &TrainReport b) {
    return a.steps == b.steps && a.epochs == b.epochs && a.firstEpochLoss == b.firstEpochLoss &&
           a.lastEpochLoss == b.lastEpochLoss && a.bestEpochLoss == b.bestEpochLoss;
}

// ── Trainable ────────────────────────────────────────────────────────────────
static &Tensor clf_loss_(&Mlp m, const &Tensor x, const &Tensor y) {
    return tensor_cross_entropy_loss(m.forward(x), y);
}
static &Tensor clf_logits_(&Mlp m, const &Tensor x) { return m.forward(x); }

void t_backend_name() {
    const char* name = ML_TRAIN_BACKEND;
    char c = 0;
    unsafe { c = name[0]; }
    ASSERT_TRUE(c == 'c');
}

// train_fit over the neutral MlpTrainer = trainer_fit, to the bit.
void t_fit_matches_trainer_fit() {
    struct Rng drng = rng_new(1UL);
    struct Dataset data = dataset_xor(250UL, (float)0.0, &drng);   // 7 full batches + a 26-row tail
    struct Rng r1 = rng_new(3UL);
    struct Mlp a = mlp_new(2UL, 16UL, 2UL, 2UL, ACT_RELU, ACT_NONE, &r1);
    struct Rng r2 = rng_new(3UL);
    struct Mlp b = mlp_new(2UL, 16UL, 2UL, 2UL, ACT_RELU, ACT_NONE, &r2);
    struct TrainConfig tc = train_config_default();
    tc.epochs = 6UL;
    tc.clipNorm = (float)1.0;
    tc.lrDecay = (float)0.9;
    struct Optimizer o = optimizer_new(OPT_ADAM, (float)0.01);
    model_add_params(&a, &o);
    struct TrainReport ra = trainer_fit(&a, clf_loss_, &o, &data, &tc);

    struct Optimizer adam = optimizer_new(OPT_ADAM, (float)0.01);
    struct TrainerOptions to = trainer_options_default();
    MlpTrainer t = mlp_trainer_new(b, TRAIN_LOSS_CE, &adam, &to);
    struct TrainReport rb = train_fit(&t, &data, &tc);
    ASSERT_TRUE(same_report_(&ra, &rb));
    ASSERT_EQ(rb.steps, 48UL);
    ASSERT_TRUE(params_diff_(&a.params, &t.host.params) == (float)0.0);
    // the host tensors are the ones passed in
    ASSERT_TRUE(params_diff_(&b.params, &t.host.params) == (float)0.0);
    ASSERT_TRUE(t.learning_rate() == o.lr);
    ASSERT_TRUE(trainer_accuracy(&t.host, clf_logits_, &data, 64UL) > (float)0.9);

    // a second run continues from the trained state
    struct TrainReport again = train_fit(&t, &data, &tc);
    ASSERT_TRUE(again.lastEpochLoss < rb.firstEpochLoss);
    t.free();
    o.free();
    adam.free();
    a.free();
    t.host.free();
    data.free();
}

// The loss kinds, and train_step on host batches.
void t_losses_and_train_step() {
    struct Rng rng = rng_new(5UL);
    struct Dataset data = dataset_sine(128UL, (float)0.0, &rng);
    struct Mlp net = mlp_new(1UL, 16UL, 1UL, 1UL, ACT_TANH, ACT_NONE, &rng);
    struct Optimizer adam = optimizer_new(OPT_ADAM, (float)0.02);
    struct TrainerOptions to = trainer_options_default();
    MlpTrainer t = mlp_trainer_new(net, TRAIN_LOSS_MSE, &adam, &to);
    float first = t.train_step(data.x, data.y);
    float last = first;
    int i = 0;
    while (i < 150) { last = t.train_step(data.x, data.y); i = i + 1; }
    ASSERT_TRUE(last < first * (float)0.2);
    t.free();
    t.host.free();

    // BCE on logits: a 0/1 target per row
    struct Dataset xr = dataset_xor(200UL, (float)0.0, &rng);
    struct Mlp n2 = mlp_new(2UL, 16UL, 2UL, 1UL, ACT_RELU, ACT_NONE, &rng);
    MlpTrainer t2 = mlp_trainer_new(n2, TRAIN_LOSS_BCE, &adam, &to);
    struct TrainConfig tc = train_config_default();
    tc.epochs = 40UL;
    struct TrainReport rep = train_fit(&t2, &xr, &tc);
    ASSERT_TRUE(rep.lastEpochLoss < rep.firstEpochLoss * (float)0.5);
    t2.free();
    t2.host.free();
    adam.free();
    xr.free();
    data.free();
}

static &Tensor conv_loss_(&ConvNet n, const &Tensor x, const &Tensor y) {
    return tensor_cross_entropy_loss(n.forward(x), y);
}

static struct ConvNet small_cnn_(unsigned long seed) {
    struct Rng rng = rng_new(seed);
    struct ConvNet n = convnet_new(1UL, 8UL, 8UL);
    n.conv(4UL, 3UL, 1UL, 1UL, ACT_RELU, &rng);
    n.maxpool(2UL, 2UL, 0UL);
    n.dense(4UL, ACT_NONE, &rng);
    return n;
}

void t_convnet_trainer() {
    struct Rng drng = rng_new(9UL);
    struct Dataset data = dataset_shapes(96UL, 8UL, &drng);
    struct ConvNet a = small_cnn_(21UL);
    struct ConvNet b = small_cnn_(21UL);
    struct TrainConfig tc = train_config_default();
    tc.epochs = 3UL;
    struct Optimizer o = optimizer_new(OPT_ADAM, (float)0.01);
    model_add_params(&a, &o);
    struct TrainReport ra = trainer_fit(&a, conv_loss_, &o, &data, &tc);
    struct Optimizer adam = optimizer_new(OPT_ADAM, (float)0.01);
    struct TrainerOptions to = trainer_options_default();
    ConvNetTrainer t = convnet_trainer_new(b, TRAIN_LOSS_CE, &adam, &to);
    struct TrainReport rb = train_fit(&t, &data, &tc);
    ASSERT_TRUE(same_report_(&ra, &rb));
    ASSERT_TRUE(params_diff_(&a.params, &t.host.params) == (float)0.0);
    t.free();
    o.free();
    adam.free();
    a.free();
    t.host.free();
    data.free();
}

void t_autoencoder_trainers() {
    struct Rng drng = rng_new(9UL);
    struct Dataset data = dataset_shapes(128UL, 8UL, &drng);
    struct VaeConfig cfg = vae_config_default(4UL);
    cfg.hidden = 32UL;
    struct Vae a = vae_new(64UL, &cfg);
    struct Vae b = vae_new(64UL, &cfg);
    struct TrainConfig tc = train_config_default();
    tc.epochs = 3UL;
    struct Optimizer o = optimizer_new(OPT_ADAM, (float)0.003);
    model_add_params(&a, &o);
    struct TrainReport ra = model_fit(&a, &o, &data, &tc);
    struct Optimizer adam = optimizer_new(OPT_ADAM, (float)0.003);
    struct TrainerOptions to = trainer_options_default();
    VaeTrainer t = vae_trainer_new(b, &adam, &to);
    struct TrainReport rb = train_fit(&t, &data, &tc);
    ASSERT_TRUE(same_report_(&ra, &rb));
    ASSERT_TRUE(params_diff_(&a.dec.params, &t.host.dec.params) == (float)0.0);
    ASSERT_TRUE(t.host.lastRecon == a.lastRecon);   // the trainer's host is the trained model
    t.free();
    o.free();
    a.free();
    t.host.free();

    // convolutional autoencoder
    struct Rng rng = rng_new(61UL);
    struct ConvNet enc = convnet_new(1UL, 8UL, 8UL);
    enc.conv(4UL, 3UL, 2UL, 1UL, ACT_RELU, &rng);
    struct ConvNet dec = convnet_new(4UL, 4UL, 4UL);
    dec.deconv(1UL, 4UL, 2UL, 1UL, ACT_NONE, &rng);
    ConvAeTrainer c = conv_ae_trainer_new(conv_autoencoder_new(enc, dec, RECON_BCE), &adam, &to);
    tc.epochs = 4UL;
    struct TrainReport rc = train_fit(&c, &data, &tc);
    ASSERT_EQ(rc.epochs, 4UL);
    ASSERT_TRUE(rc.lastEpochLoss < rc.firstEpochLoss);
    c.free();
    c.host.free();
    adam.free();
    data.free();
}

// ── GanTrainable ─────────────────────────────────────────────────────────────
void t_gan_trainer() {
    struct Rng drng = rng_new(4UL);
    struct Dataset data = dataset_gaussian_ring(128UL, 4UL, (float)2.0, (float)0.1, &drng);
    struct GanConfig cfg = gan_config_default(4UL);
    cfg.hidden = 16UL;
    cfg.emaDecay = (float)0.0;
    struct Gan a = gan_new(2UL, &cfg);
    struct Gan b = gan_new(2UL, &cfg);
    struct GanLosses la = a.train(&data, 2UL, 32UL);
    struct TrainerOptions to = trainer_options_default();
    GanTrainer t = gan_trainer_new(b, &to);
    struct GanLosses lb = train_gan(&t, &data, 2UL, 32UL);
    ASSERT_TRUE(la.d == lb.d && la.g == lb.g);
    ASSERT_EQ(t.host.steps, a.steps);
    ASSERT_TRUE(params_diff_(&a.gen.params, &t.host.gen.params) == (float)0.0);
    // one more step on a host batch, through the trait
    struct GanLosses ls = t.train_step(data.x, null);
    ASSERT_TRUE(ls.d == ls.d && ls.g == ls.g);   // finite
    ASSERT_EQ(t.host.steps, a.steps + 1UL);
    t.free();
    a.free();
    t.host.free();
    data.free();
}

// ── ValueAgent / PolicyLearner: generic loops ────────────────────────────────
// Random observations and rewards from 'env'; returns the actions' sum.
generic<A: ValueAgent>
unsigned long value_run_(&A a, &Rng env, unsigned long steps) {
    &Tensor o = tensor_new_1d(4UL, 0);
    &Tensor n = tensor_new_1d(4UL, 0);
    unsigned long sum = 0UL;
    unsigned long i = 0UL;
    while (i < steps) {
        env.fill_normal(o, (float)0.0, (float)1.0);
        env.fill_normal(n, (float)0.0, (float)1.0);
        unsigned long act = a.act(o, 1);
        sum = sum + act;
        float r = env.uniform();
        int t = env.below(10UL) == 0UL;
        a.observe(o, act, r, n, t);
        i = i + 1UL;
    }
    tensor_drop(o);
    tensor_drop(n);
    return sum;
}

generic<P: PolicyLearner>
float policy_run_(&P p, &Rng env, unsigned long rollout, int updates) {
    &Tensor o = tensor_new_1d(4UL, 0);
    &Tensor n = tensor_new_1d(4UL, 0);
    float l = (float)0.0;
    int u = 0;
    while (u < updates) {
        unsigned long i = 0UL;
        while (i < rollout) {
            env.fill_normal(o, (float)0.0, (float)1.0);
            env.fill_normal(n, (float)0.0, (float)1.0);
            unsigned long act = p.act(o, 1);
            float r = env.uniform();
            int t = env.below(20UL) == 0UL;
            int tr = !t && env.below(30UL) == 0UL;
            p.record(o, act, r, n, t, tr);
            i = i + 1UL;
        }
        l = p.update();
        u = u + 1;
    }
    tensor_drop(o);
    tensor_drop(n);
    return l;
}

void t_rl_agents() {
    struct DqnConfig cfg = dqn_config_default();
    cfg.hidden = 32UL;
    cfg.batchSize = 32UL;
    cfg.bufferSize = 500UL;
    cfg.learningStarts = 64UL;
    cfg.epsDecaySteps = 100UL;
    struct Dqn direct = dqn_new(4UL, 3UL, &cfg);
    DqnLearner d = dqn_learner_new(dqn_new(4UL, 3UL, &cfg));
    // the same loop by hand on the direct API
    struct Rng e1 = rng_new(91UL);
    &Tensor o = tensor_new_1d(4UL, 0);
    &Tensor n = tensor_new_1d(4UL, 0);
    unsigned long sum = 0UL;
    unsigned long i = 0UL;
    while (i < 160UL) {
        e1.fill_normal(o, (float)0.0, (float)1.0);
        e1.fill_normal(n, (float)0.0, (float)1.0);
        unsigned long act = direct.act(o, 1);
        sum = sum + act;
        float r = e1.uniform();
        int t = e1.below(10UL) == 0UL;
        direct.observe(o, act, r, n, t);
        i = i + 1UL;
    }
    struct Rng e2 = rng_new(91UL);
    unsigned long sum2 = value_run_(&d, &e2, 160UL);
    ASSERT_EQ(sum, sum2);
    ASSERT_EQ(d.updates(), direct.updates);
    ASSERT_TRUE(d.updates() > 0UL);
    ASSERT_TRUE(d.last_loss() == direct.lastLoss);
    ASSERT_TRUE(d.sync());
    ASSERT_TRUE(params_diff_(&direct.q.params, &d.host.q.params) == (float)0.0);
    d.free();
    d.host.free();
    direct.free();

    struct PgConfig pc = pg_config_default(PG_PPO);
    pc.hidden = 32UL;
    pc.minibatch = 24UL;
    pc.epochs = 3UL;
    struct PolicyAgent pd = pg_new(4UL, 3UL, &pc);
    struct TrainerOptions to = trainer_options_default();
    PgLearner p = pg_learner_new(pg_new(4UL, 3UL, &pc), &to);
    struct Rng e3 = rng_new(92UL);
    float lc = (float)0.0;
    int u = 0;
    while (u < 2) {
        i = 0UL;
        while (i < 100UL) {
            e3.fill_normal(o, (float)0.0, (float)1.0);
            e3.fill_normal(n, (float)0.0, (float)1.0);
            unsigned long act = pd.act(o, 1);
            float r = e3.uniform();
            int t = e3.below(20UL) == 0UL;
            int tr = !t && e3.below(30UL) == 0UL;
            pd.record(o, act, r, n, t, tr);
            i = i + 1UL;
        }
        lc = pd.update();
        u = u + 1;
    }
    struct Rng e4 = rng_new(92UL);
    float lp = policy_run_(&p, &e4, 100UL, 2);
    ASSERT_TRUE(lp == lc);
    ASSERT_TRUE(p.sync());
    ASSERT_TRUE(params_diff_(&pd.policy.params, &p.host.policy.params) == (float)0.0);
    p.free();
    p.host.free();
    pd.free();
    tensor_drop(o);
    tensor_drop(n);
}

int main() {
    struct TestSuite t = test_suite_init();
    unsafe {
        t.add("backend is the CPU without features", t_backend_name);
        t.add("train_fit (Trainable) = trainer_fit", t_fit_matches_trainer_fit);
        t.add("loss kinds, train_step", t_losses_and_train_step);
        t.add("ConvNet trainer = trainer_fit", t_convnet_trainer);
        t.add("VAE / conv autoencoder trainers", t_autoencoder_trainers);
        t.add("train_gan (GanTrainable) = Gan::train", t_gan_trainer);
        t.add("ValueAgent / PolicyLearner loops = dqn_* / pg_*", t_rl_agents);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
