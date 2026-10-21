// One training program, two backends: written only against train.h's
// backend-neutral names and train_traits.h's traits, built once without
// features (the CPU) and once with SAFEC_FEATURE_ROCM (run.sh does both).
// Each build checks that its models train and prints
//   RESULT <name> <value> <tolerance>
// lines; run.sh compares the two builds' values: equal up to the
// tolerance, relative to 1 + |cpu value| (the device follows the CPU's
// batches, initial weights and random streams except where it draws its own
// noise: VAEs, GANs).
#include <std/ml/train.h>
#include <std/ml/nn.h>
#include <std/ml/datasets.h>
#include <std/ml/convnet.h>
#include <std/ml/autoencoder.h>
#include <std/ml/gan.h>
#include <std/ml/rl.h>
extern int printf(const char* fmt, ...);
extern void free(void* ptr);

static int g_checks = 0;
static int g_fails = 0;

static void check(int ok, const char* what) {
    g_checks = g_checks + 1;
    if (!ok) {
        g_fails = g_fails + 1;
        unsafe { printf("  FAIL %s\n", what); }
    }
}

static void result(const char* name, double v, double tol) {
    unsafe { printf("RESULT %s %.7g %g\n", name, v, tol); }
}

static void tensor_drop(&Tensor t) {
    t->free();
    unsafe { free((void*)t); }
}

static &Tensor clf_logits_(&Mlp m, const &Tensor x) { return m.forward(x); }

// Any Trainable: fit, report, check the loss fell.
generic<M: Trainable>
struct TrainReport fit_case_(&M t, const char* name, const &Dataset d, const &TrainConfig tc, float fall) {
    struct TrainReport rep = train_fit(t, d, tc);
    unsafe { printf("  %-12s %lu steps, loss %.5f -> %.5f\n", name, rep.steps, (double)rep.firstEpochLoss,
                    (double)rep.lastEpochLoss); }
    check(rep.steps > 0UL && rep.lastEpochLoss < rep.firstEpochLoss * fall, name);
    return rep;
}

static void mlp_cases_(const &TrainerOptions to) {
    struct Rng drng = rng_new(1UL);
    struct Dataset data = dataset_xor(250UL, (float)0.0, &drng);   // 7 full batches + a 26-row tail
    struct Rng r = rng_new(3UL);
    struct Optimizer adam = optimizer_new(OPT_ADAM, (float)0.01);
    MlpTrainer t = mlp_trainer_new(mlp_new(2UL, 16UL, 2UL, 2UL, ACT_RELU, ACT_NONE, &r), TRAIN_LOSS_CE, &adam, to);
    struct TrainConfig tc = train_config_default();
    tc.epochs = 6UL;
    tc.clipNorm = (float)1.0;
    tc.lrDecay = (float)0.9;
    struct TrainReport rep = fit_case_(&t, "mlp ce", &data, &tc, (float)0.9);
    result("mlp.steps", (double)rep.steps, 0.0);
    result("mlp.first", (double)rep.firstEpochLoss, 0.0005);
    result("mlp.last", (double)rep.lastEpochLoss, 0.001);
    result("mlp.lr", (double)t.learning_rate(), 0.000001);
    // the host model is synced: evaluate it on the CPU
    result("mlp.acc", (double)trainer_accuracy(&t.host, clf_logits_, &data, 64UL), 0.01);
    unsafe { result("mlp.w0", (double)(*t.host.params.at(0UL))->data[0], 0.001); }
    t.free();
    t.host.free();

    // train_step on a host batch (MSE)
    struct Dataset sine = dataset_sine(64UL, (float)0.0, &drng);
    MlpTrainer s = mlp_trainer_new(mlp_new(1UL, 32UL, 1UL, 1UL, ACT_TANH, ACT_NONE, &r), TRAIN_LOSS_MSE, &adam, to);
    float first = s.train_step(sine.x, sine.y);
    float last = first;
    int i = 0;
    while (i < 40) { last = s.train_step(sine.x, sine.y); i = i + 1; }
    unsafe { printf("  %-12s loss %.5f -> %.5f\n", "train_step", (double)first, (double)last); }
    check(first > (float)0.0 && last < first * (float)0.5, "train_step");
    result("step.first", (double)first, 0.0005);
    result("step.last", (double)last, 0.002);
    s.free();
    s.host.free();
    adam.free();
    sine.free();
    data.free();
}

static void convnet_case_(const &TrainerOptions to) {
    struct Rng drng = rng_new(9UL);
    struct Dataset data = dataset_shapes(250UL, 8UL, &drng);
    struct Rng rng = rng_new(21UL);
    struct ConvNet n = convnet_new(1UL, 8UL, 8UL);
    n.conv(8UL, 3UL, 1UL, 1UL, ACT_RELU, &rng);
    n.maxpool(2UL, 2UL, 0UL);
    n.conv(8UL, 3UL, 1UL, 1UL, ACT_RELU, &rng);
    n.dense(4UL, ACT_NONE, &rng);
    struct Optimizer adam = optimizer_new(OPT_ADAM, (float)0.005);
    ConvNetTrainer t = convnet_trainer_new(n, TRAIN_LOSS_CE, &adam, to);
    struct TrainConfig tc = train_config_default();
    tc.epochs = 5UL;
    struct TrainReport rep = fit_case_(&t, "convnet ce", &data, &tc, (float)0.8);
    result("cnn.first", (double)rep.firstEpochLoss, 0.001);
    result("cnn.last", (double)rep.lastEpochLoss, 0.003);
    t.free();
    t.host.free();
    adam.free();
    data.free();
}

static void autoencoder_cases_(const &TrainerOptions to) {
    struct Rng drng = rng_new(9UL);
    struct Dataset data = dataset_shapes(256UL, 8UL, &drng);
    struct VaeConfig cfg = vae_config_default(4UL);
    cfg.hidden = 64UL;
    struct Optimizer adam = optimizer_new(OPT_ADAM, (float)0.003);
    VaeTrainer v = vae_trainer_new(vae_new(64UL, &cfg), &adam, to);
    struct TrainConfig tc = train_config_default();
    tc.epochs = 15UL;
    struct TrainReport rep = fit_case_(&v, "vae", &data, &tc, (float)0.8);
    // the device draws its own noise: the loss level agrees, not the values
    result("vae.first", (double)rep.firstEpochLoss, 0.03);
    result("vae.last", (double)rep.lastEpochLoss, 0.1);
    v.free();
    v.host.free();

    struct Rng rng = rng_new(61UL);
    struct ConvNet enc = convnet_new(1UL, 8UL, 8UL);
    enc.conv(4UL, 3UL, 2UL, 1UL, ACT_RELU, &rng);     // 4 x 4 x 4
    enc.conv(8UL, 3UL, 2UL, 1UL, ACT_RELU, &rng);     // 8 x 2 x 2
    struct ConvNet dec = convnet_new(8UL, 2UL, 2UL);
    dec.deconv(4UL, 4UL, 2UL, 1UL, ACT_RELU, &rng);   // 4 x 4 x 4
    dec.upsample_bilinear(2UL);                       // 4 x 8 x 8
    dec.conv(1UL, 3UL, 1UL, 1UL, ACT_NONE, &rng);     // 1 x 8 x 8
    ConvAeTrainer a = conv_ae_trainer_new(conv_autoencoder_new(enc, dec, RECON_BCE), &adam, to);
    tc.epochs = 4UL;
    struct TrainReport ra = fit_case_(&a, "conv ae", &data, &tc, (float)0.95);
    result("cae.first", (double)ra.firstEpochLoss, 0.0005);
    result("cae.last", (double)ra.lastEpochLoss, 0.001);
    a.free();
    a.host.free();
    adam.free();
    data.free();
}

// The device draws its own noise for a GAN: the losses stay finite and the
// host model is synced, but they don't follow the CPU's.
static void gan_case_(const &TrainerOptions to) {
    struct Rng drng = rng_new(4UL);
    struct Dataset data = dataset_gaussian_ring(256UL, 4UL, (float)2.0, (float)0.1, &drng);
    struct GanConfig cfg = gan_config_default(4UL);
    cfg.hidden = 32UL;
    GanTrainer g = gan_trainer_new(gan_new(2UL, &cfg), to);
    struct GanLosses l = train_gan(&g, &data, 5UL, 64UL);
    struct GanLosses s = g.train_step(data.x, null);
    unsafe { printf("  %-12s d %.4f g %.4f, one more step d %.4f g %.4f\n", "gan", (double)l.d, (double)l.g,
                    (double)s.d, (double)s.g); }
    check(l.d == l.d && l.g == l.g && l.d != (float)0.0 && s.d == s.d && s.g == s.g, "gan");
    &Tensor sample = g.host.generate(8UL, null);
    unsigned long rows = 0UL;
    unsafe { rows = sample->shape[0]; }
    check(rows == 8UL, "gan host generates");
    tensor_drop(sample);
    g.free();
    g.host.free();
    data.free();
}

// ── RL through the agent traits ──────────────────────────────────────────────
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
        sum = sum * 3UL + act;
        float r = env.uniform();
        int t = env.below(10UL) == 0UL;
        if (a.observe(o, act, r, n, t) < 0) check(0, "observe");
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

static void rl_cases_(const &TrainerOptions to) {
    struct DqnConfig cfg = dqn_config_default();
    cfg.hidden = 32UL;
    cfg.batchSize = 32UL;
    cfg.bufferSize = 500UL;
    cfg.learningStarts = 64UL;
    cfg.epsDecaySteps = 100UL;
    DqnLearner d = dqn_learner_new(dqn_new(4UL, 3UL, &cfg));
    struct Rng env = rng_new(91UL);
    unsigned long acts = value_run_(&d, &env, 160UL);
    float dl = d.last_loss();
    check(d.updates() > 0UL && d.sync(), "dqn");
    unsafe { printf("  %-12s %lu updates, loss %.5f\n", "dqn", d.updates(), (double)dl); }
    result("dqn.actions", (double)(acts % 1000003UL), 0.0);
    result("dqn.updates", (double)d.updates(), 0.0);
    result("dqn.loss", (double)dl, 0.001);
    unsafe { result("dqn.w0", (double)(*d.host.q.params.at(0UL))->data[0], 0.001); }
    d.free();
    d.host.free();

    struct PgConfig pc = pg_config_default(PG_PPO);
    pc.hidden = 32UL;
    pc.minibatch = 24UL;
    pc.epochs = 3UL;
    PgLearner p = pg_learner_new(pg_new(4UL, 3UL, &pc), to);
    struct Rng env2 = rng_new(92UL);
    float pl = policy_run_(&p, &env2, 100UL, 2);
    check(pl != (float)-1.0 && p.sync(), "ppo");
    unsafe { printf("  %-12s loss %.5f\n", "ppo", (double)pl); }
    result("ppo.loss", (double)pl, 0.001);
    unsafe { result("ppo.w0", (double)(*p.host.policy.params.at(0UL))->data[0], 0.001); }
    p.free();
    p.host.free();
}

int main() {
    unsafe { printf("train_backend_check: backend %s\n", ML_TRAIN_BACKEND); }
    struct TrainerOptions to = trainer_options_default();
    to.maxBatch = 256UL;
    mlp_cases_(&to);
    convnet_case_(&to);
    autoencoder_cases_(&to);
    gan_case_(&to);
    rl_cases_(&to);
    unsafe { printf("train_backend_check (%s): %d checks, %d failed\n", ML_TRAIN_BACKEND, g_checks, g_fails); }
    return g_fails == 0 ? 0 : 1;
}
