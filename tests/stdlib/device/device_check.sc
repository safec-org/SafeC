// The GPU device backends (device.h) against the CPU: every training
// workload of test_ml_device.sc, and the sparse product, on CHECK_DEVICE.
// Built by tests/stdlib/{rocm,cuda,mps}/run.sh with the backend's feature
// define (-DSAFEC_FEATURE_ROCM=1 -DCHECK_DEVICE=DEVICE_ROCM, ...), which is
// what links the backend in. Prints one line per check and exits nonzero on
// any mismatch.
#include <std/io.h>
#include <std/ml/device.h>
#include <std/ml/tensor.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/nn.h>
#include <std/ml/norm.h>
#include <std/ml/sparse.h>
#include <std/ml/datasets.h>
#include <std/ml/trainer.h>
#include <std/ml/module.h>
#include <std/ml/autoencoder.h>
#include <std/ml/gan.h>
#include <std/ml/rl.h>
#include <std/ml/lora.h>
#include <std/ml/convnet.h>

extern void* malloc(unsigned long n);
extern void free(void* p);
extern void* memcpy(void* d, const void* s, unsigned long n);

static float absf_(float v) { return v < (float)0.0 ? -v : v; }

// a and b agree to tol relative (plus tol absolute)
static int near_(float a, float b, float tol) { return absf_(a - b) <= tol * ((float)1.0 + absf_(a)); }

// ── Training on the reference device = training on the CPU ───────────────────
static int g_dev = DEVICE_CPU;
static float g_loss = (float)0.0;

static void w_mlp_(&Dataset d) {
    struct Rng rng = Rng::new(3UL);
    struct Mlp net = Mlp::new(2UL, 16UL, 2UL, 2UL, ACT_RELU, ACT_NONE, &rng);
    struct Optimizer opt = Optimizer::new(OPT_ADAM, (float)0.01);
    model_add_params(&net, &opt);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 3UL; tc.device = g_dev; tc.clipNorm = (float)1.0;
    struct TrainReport rep = module_fit(&net, TRAIN_LOSS_CE, &opt, d, &tc);
    g_loss = rep.lastEpochLoss;
    opt.free(); net.free();
}

static void w_vae_(&Dataset d, unsigned long classes) {
    struct VaeConfig cfg = VaeConfig::default(4UL);
    cfg.hidden = 32UL;
    cfg.classes = classes;
    struct Vae v = Vae::new(100UL, &cfg);
    struct Optimizer opt = Optimizer::new(OPT_ADAMW, (float)0.002);
    model_add_params(&v, &opt);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 2UL; tc.device = g_dev;
    struct TrainReport rep = model_fit(&v, &opt, d, &tc);
    g_loss = rep.lastEpochLoss;
    opt.free(); v.free();
}

static void w_gan_(&Dataset d, int loss) {
    struct GanConfig cfg = GanConfig::default(4UL);
    cfg.hidden = 16UL; cfg.loss = loss; cfg.emaDecay = (float)0.9;
    struct Gan g = Gan::new(2UL, &cfg);
    model_to(&g, g_dev);
    struct GanLosses l = g.train(d, 2UL, 64UL);
    g_loss = l.d + l.g;
    g.free();
}

static void w_dqn_() {
    struct DqnConfig cfg = DqnConfig::default();
    cfg.learningStarts = 32UL; cfg.bufferSize = 1000UL; cfg.batchSize = 32UL; cfg.hidden = 16UL;
    struct Dqn d = Dqn::new(4UL, 2UL, &cfg);
    model_to(&d, g_dev);
    struct Rng rng = Rng::new(5UL);
    &FloatTensor o = FloatTensor::new_1d(4UL, 0);
    &FloatTensor n = FloatTensor::new_1d(4UL, 0);
    unsigned long i = 0UL;
    while (i < 120UL) {
        rng.fill_normal(o, (float)0.0, (float)1.0);
        rng.fill_normal(n, (float)0.0, (float)1.0);
        d.act(o, 0);
        d.observe(o, rng.below(2UL), rng.uniform(), n, rng.below(10UL) == 0UL);
        i = i + 1UL;
    }
    g_loss = d.lastLoss;
    d.free(); o.free(); n.free();
}

static void w_ppo_() {
    struct PgConfig cfg = PgConfig::default(PG_PPO);
    cfg.hidden = 16UL; cfg.epochs = 2UL;
    struct PolicyAgent p = PolicyAgent::new(4UL, 2UL, &cfg);
    model_to(&p, g_dev);
    struct Rng rng = Rng::new(6UL);
    &FloatTensor o = FloatTensor::new_1d(4UL, 0);
    &FloatTensor n = FloatTensor::new_1d(4UL, 0);
    unsigned long i = 0UL;
    while (i < 256UL) {
        rng.fill_normal(o, (float)0.0, (float)1.0);
        rng.fill_normal(n, (float)0.0, (float)1.0);
        p.record(o, rng.below(2UL), (float)1.0, n, rng.below(20UL) == 0UL, 0);
        i = i + 1UL;
    }
    g_loss = p.update();
    p.free(); o.free(); n.free();
}

static void w_lora_(&Dataset d) {
    struct Rng rng = Rng::new(7UL);
    struct Mlp base = Mlp::new(2UL, 16UL, 2UL, 2UL, ACT_RELU, ACT_NONE, &rng);
    struct LoraConfig cfg = LoraConfig::default(2UL);
    struct LoraMlp ft = LoraMlp::attach(&base, &cfg);
    struct Optimizer opt = Optimizer::new(OPT_ADAMW, (float)0.01);
    model_add_params(&ft, &opt);
    model_to(&base, g_dev);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 2UL; tc.device = g_dev;
    struct TrainReport rep = module_fit(&ft, TRAIN_LOSS_CE, &opt, d, &tc);
    g_loss = rep.lastEpochLoss;
    ft.detach(); opt.free(); base.free();
}

static void w_conv_(&Dataset d) {
    struct ConvNet enc = ConvNet::new(1UL, 10UL, 10UL);
    struct ConvNet dec = ConvNet::new(8UL, 5UL, 5UL);
    struct Rng rng = Rng::new(8UL);
    enc.conv(8UL, 3UL, 2UL, 1UL, ACT_RELU, &rng);
    dec.deconv(1UL, 4UL, 2UL, 1UL, ACT_NONE, &rng);
    struct ConvAutoencoder ae = ConvAutoencoder::new(enc, dec, RECON_BCE);
    struct Optimizer opt = Optimizer::new(OPT_ADAM, (float)0.002);
    model_add_params(&ae, &opt);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 2UL; tc.device = g_dev;
    struct TrainReport rep = model_fit(&ae, &opt, d, &tc);
    g_loss = rep.lastEpochLoss;
    opt.free(); ae.free();
}

static void w_norm_() {
    // LayerNorm + AdaLN-style row ops and the losses, one step at a time
    struct Rng rng = Rng::new(9UL);
    struct LayerNorm ln = LayerNorm::new(6UL);
    &FloatTensor W = FloatTensor::new_2d(6UL, 3UL, 1);
    rng.fill_uniform(W, (float)-0.5, (float)0.5);
    struct Optimizer opt = Optimizer::new(OPT_LION, (float)0.003);
    model_add_params(&ln, &opt);
    opt.add(W);
    if (g_dev != DEVICE_CPU) { model_to(&ln, g_dev); W.place(g_dev); }
    &FloatTensor x = FloatTensor::new_2d(8UL, 6UL, 0);
    &FloatTensor t = FloatTensor::new_2d(8UL, 3UL, 0);
    rng.fill_normal(x, (float)0.0, (float)1.0);
    rng.fill_uniform(t, (float)0.0, (float)1.0);
    struct TensorArena ar = TensorArena::new(0UL);
    unsigned long s = 0UL;
    while (s < 20UL) {
        ar.begin();
        &FloatTensor h = tensor_matmul(ln.forward(x).gelu(), W);
        &FloatTensor l = tensor_add(tensor_bce_logits_loss(h, t),
                                    tensor_huber_loss(h.sigmoid(), t, (float)0.5).scale((float)0.5));
        g_loss = trainer_step_(&opt, l, (float)0.0);
        tensor_arena_end();
        ar.reset();
        s = s + 1UL;
    }
    ar.free(); opt.free(); ln.free(); W.free(); x.free(); t.free();
}

static &FloatTensor cn_clf_(&ConvNet n, const &FloatTensor x, const &FloatTensor y) { return tensor_cross_entropy_loss(n.forward(x), y); }

static void cn_fit_(&ConvNet n, &Dataset d) {
    struct Optimizer opt = Optimizer::new(OPT_ADAM, (float)0.003);
    model_add_params(n, &opt);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 2UL; tc.batchSize = 16UL; tc.device = g_dev;
    g_loss = trainer_fit(n, cn_clf_, &opt, d, &tc).lastEpochLoss;
    opt.free();
}

static void w_cnn_(&Dataset d) {
    // batch norm, pools, a residual block, group norm, global average pool
    struct Rng rng = Rng::new(10UL);
    struct ConvNet n = ConvNet::new(1UL, 10UL, 10UL);
    n.conv(8UL, 3UL, 1UL, 1UL, ACT_NONE, &rng);
    n.batchnorm(ACT_RELU);
    n.maxpool(2UL, 2UL, 0UL);                         // 8 x 5 x 5
    n.residual_block(8UL, 1UL, 1, ACT_RELU, &rng);
    n.conv(12UL, 3UL, 1UL, 1UL, ACT_NONE, &rng);
    n.groupnorm(4UL, ACT_SILU);
    n.avgpool(3UL, 1UL, 1UL, 0);                      // 12 x 5 x 5
    n.global_avgpool();
    n.dense(4UL, ACT_NONE, &rng);
    cn_fit_(&n, d);
    n.free();
}

static void w_spatial_(&Dataset d) {
    // upsampling, resizing, instance norm, dropout, concat, global max pool
    struct Rng rng = Rng::new(11UL);
    struct ConvNet n = ConvNet::new(1UL, 10UL, 10UL);
    n.conv(4UL, 3UL, 2UL, 1UL, ACT_RELU, &rng);       // 4 x 5 x 5
    unsigned long a = n.save();
    n.upsample(2UL);                                  // 4 x 10 x 10
    n.conv(4UL, 3UL, 1UL, 1UL, ACT_NONE, &rng);
    n.instancenorm(ACT_GELU);
    n.dropout((float)0.2, 1, &rng);
    n.resize(7UL, 7UL, RESIZE_BICUBIC, 1);
    n.resize(5UL, 5UL, RESIZE_BILINEAR, 0);
    n.concat_skip(a);                                 // 8 x 5 x 5
    n.global_maxpool();
    n.dense(4UL, ACT_NONE, &rng);
    cn_fit_(&n, d);
    n.free();
}

static float run_on_(int dev, fn void() work) {
    g_dev = dev;
    work();
    return g_loss;
}

static struct Dataset g_xor;
static struct Dataset g_bits;
static struct Dataset g_gauss;
static struct Dataset g_img;

static void wk_mlp() { w_mlp_(&g_xor); }
static void wk_vae() { w_vae_(&g_bits, 0UL); }
static void wk_cvae() { w_vae_(&g_bits, 4UL); }
static void wk_gan() { w_gan_(&g_gauss, GAN_STANDARD); }
static void wk_wgan() { w_gan_(&g_gauss, GAN_HINGE); }
static void wk_dqn() { w_dqn_(); }
static void wk_ppo() { w_ppo_(); }
static void wk_lora() { w_lora_(&g_xor); }
static void wk_conv() { w_conv_(&g_img); }
static void wk_norm() { w_norm_(); }
static void wk_cnn() { w_cnn_(&g_img); }
static void wk_spatial() { w_spatial_(&g_img); }

static int g_bad = 0;
static void check_(const &char what, int ok) {
    print(ok ? "  PASS  " : "  FAIL  "); println(what);
    if (!ok) g_bad = g_bad + 1;
}

void check_training_() {
        struct Rng rng = Rng::new(1UL);
    g_xor = Dataset::xor(256UL, (float)0.0, &rng);
    g_bits = Dataset::shapes(128UL, 10UL, &rng);
    g_gauss = Dataset::blobs(256UL, 1UL, 2UL, (float)0.5, &rng);
    g_img = Dataset::shapes(64UL, 10UL, &rng);
    const char* names[12];
    names[0] = "mlp"; names[1] = "vae"; names[2] = "cvae"; names[3] = "gan"; names[4] = "hinge gan";
    names[5] = "dqn"; names[6] = "ppo"; names[7] = "lora"; names[8] = "conv autoencoder"; names[9] = "layer norm + losses";
    names[10] = "cnn (batch / group norm, pools, residual)"; names[11] = "spatial (upsample, resize, dropout, concat)";
    fn void() works[12];
    works[0] = wk_mlp; works[1] = wk_vae; works[2] = wk_cvae; works[3] = wk_gan; works[4] = wk_wgan;
    works[5] = wk_dqn; works[6] = wk_ppo; works[7] = wk_lora; works[8] = wk_conv; works[9] = wk_norm;
    works[10] = wk_cnn; works[11] = wk_spatial;
    int k = 0;
    while (k < 12) {
        float c = run_on_(DEVICE_CPU, works[k]);
        float r = run_on_(CHECK_DEVICE, works[k]);
        unsafe {
            print("  "); print(names[k]); print(": cpu "); print_float((double)c); print(" device "); println_float((double)r);
            check_(names[k], near_(c, r, (float)2e-4));
        }
        k = k + 1;
    }
    // fp16 and bf16: every product and library convolution in 16 bits
    int pick = device_set_precision_pick(DEVICE_PICK_ALWAYS);
    int prec = DEVICE_FP16;
    while (prec <= DEVICE_BF16) {
        k = 0;
        while (k < 12) {
            float c = run_on_(DEVICE_CPU, works[k]);
            device_set_precision(CHECK_DEVICE, prec);
            float r = run_on_(CHECK_DEVICE, works[k]);
            device_set_precision(CHECK_DEVICE, DEVICE_FP32);
            unsafe {
                print("  "); print(prec == DEVICE_FP16 ? "fp16 " : "bf16 "); print(names[k]); print(": cpu ");
                print_float((double)c); print(" device "); println_float((double)r);
                check_(names[k], near_(c, r, (float)0.05));
            }
            k = k + 1;
        }
        prec = prec + 1;
    }
    device_set_precision_pick(pick);
    g_xor.free(); g_bits.free(); g_gauss.free(); g_img.free();
}

// ── Sparse tensors ───────────────────────────────────────────────────────────
static float sp_loss_(const &SparseTensor A, &FloatTensor X, &FloatTensor W) {
    return tensor_mul(tensor_spmm(A, X).tanh(), W).sum().at1(0UL);
}

void check_sparse_() {
    struct Rng rng = Rng::new(4UL);
    &FloatTensor D = FloatTensor::new_2d(12UL, 15UL, 0);
    rng.fill_uniform(D, (float)-1.0, (float)1.0);
    struct SparseTensor A = SparseTensor::from_dense(D, (float)0.6, 1);
    check_("sparse: A.nnz > 0UL && A.nnz < 180UL", A.nnz > 0UL && A.nnz < 180UL);
    // entries and the dense copy
    unsigned long i = 0UL;
    while (i < 180UL) {
        float v = D.at1(i);
        float want = (v > (float)0.6 || v < (float)-0.6) ? v : (float)0.0;
        if (A.at1(i) != want) g_bad = g_bad + 1;
        i = i + 1UL;
    }
    &FloatTensor X = FloatTensor::new_2d(15UL, 4UL, 1);
    &FloatTensor W = FloatTensor::new_2d(12UL, 4UL, 0);
    rng.fill_uniform(X, (float)-1.0, (float)1.0);
    rng.fill_uniform(W, (float)-1.0, (float)1.0);
    &FloatTensor Hs = tensor_spmm(&A, X);
    &FloatTensor Hd = tensor_matmul(A.to_dense(), X);
    i = 0UL;
    int spOk = 1;
    while (i < 48UL) { if (!near_(Hs.at1(i), Hd.at1(i), (float)1e-5)) spOk = 0; i = i + 1UL; }
    check_("sparse: spmm on the CPU = dense product", spOk);
    // gradients to the values and to X against finite differences
    A.zero_grad();
    tensor_mul(tensor_spmm(&A, X).tanh(), W).sum().backward();
    float worst = (float)0.0;
    unsafe {
        i = 0UL;
        while (i < 8UL) {
            float keep = A.values->data[i];
            float h = (float)0.002;
            A.values->data[i] = keep + h;
            float lp = sp_loss_(&A, X, W);
            A.values->data[i] = keep - h;
            float lm = sp_loss_(&A, X, W);
            A.values->data[i] = keep;
            float e = absf_((lp - lm) / ((float)2.0 * h) - A.values->grad[i]);
            if (e > worst) worst = e;
            float kx = X->data[i];
            X->data[i] = kx + h;
            lp = sp_loss_(&A, X, W);
            X->data[i] = kx - h;
            lm = sp_loss_(&A, X, W);
            X->data[i] = kx;
            e = absf_((lp - lm) / ((float)2.0 * h) - X->grad[i]);
            if (e > worst) worst = e;
            i = i + 1UL;
        }
    }
    check_("sparse: worst < (float)2e-3", worst < (float)2e-3);
    // the same on the reference device
        float gv0;
    unsafe { gv0 = A.values->grad[0]; }
    A.zero_grad(); X.zero_grad();
    check_("sparse: A.place(CHECK_DEVICE)", A.place(CHECK_DEVICE));
    check_("sparse: X.place(CHECK_DEVICE)", X.place(CHECK_DEVICE));
    tensor_mul(tensor_spmm(&A, X).tanh(), W).sum().backward();
    A.place(DEVICE_CPU); X.place(DEVICE_CPU);
    unsafe { check_("sparse: values gradient on the device = the CPU", near_(A.values->grad[0], gv0, (float)1e-5)); }
    // transpose, identity and scale
    struct SparseTensor T = A.transpose();
    check_("sparse: transpose rows", T.rows == 15UL);
    check_("sparse: T.at2(3UL, 5UL) == A.at2(5UL, 3UL)", T.at2(3UL, 5UL) == A.at2(5UL, 3UL));
    struct SparseTensor I = SparseTensor::identity(15UL, 0);
    &FloatTensor IX = tensor_spmm(&I, X);
    check_("sparse: IX.at1(7UL) == X.at1(7UL)", IX.at1(7UL) == X.at1(7UL));
    struct SparseTensor A2 = A.scale((float)2.0);
    check_("sparse: A2.at2(0UL, 0UL) == (float)2.0 * A.at2(0", A2.at2(0UL, 0UL) == (float)2.0 * A.at2(0UL, 0UL));
    A2.free(); T.free(); I.free(); A.free(); D.free(); X.free(); W.free();
}

#ifndef CHECK_DEVICE
#define CHECK_DEVICE DEVICE_ROCM
#endif

int main() {
    print("device "); print(device_name(CHECK_DEVICE)); print(": available ");
    println_int((long long)device_available(CHECK_DEVICE));
    if (!device_available(CHECK_DEVICE)) { println("  FAIL  backend not available (feature, library or GPU missing)"); return 1; }
    check_training_();
    check_sparse_();
    print("device_check: "); print_int((long long)g_bad); println(" failed");
    return g_bad == 0 ? 0 : 1;
}
