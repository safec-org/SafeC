// The GPU device backends (device.h) against the CPU: every training
// workload of test_ml_device.sc, attention's gradients (the fused kernels and
// the composed ops), and the sparse product, on CHECK_DEVICE.
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
#include <std/ml/attention.h>
#include <std/ml/attention_advanced.h>
#include <std/ml/transformer.h>

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

// ── Transformers ─────────────────────────────────────────────────────────────
// A small language model: token embedding (a row gather) plus learned
// positions, a causal multi-head self-attention block and a GELU
// feed-forward block, both pre-LN residual, then logits over the vocabulary
// and cross-entropy on the next token. A batch is one sequence: the
// dataset's rows are consecutive tokens (x the token, y the next one).
#define LM_VOCAB 8UL
#define LM_DIM   8UL
#define LM_SEQ   8UL
struct TinyLm {
    &FloatTensor E;      // [vocab, dim]
    &FloatTensor P;      // [seq, dim]
    &FloatTensor Wout;   // [dim, vocab]
    struct LayerNorm ln1;
    struct LayerNorm ln2;
    struct MultiHeadAttention mha;
    struct Mlp ffn;
};

static &FloatTensor lm_loss_(&TinyLm m, const &FloatTensor x, const &FloatTensor y) {
    &FloatTensor h = tensor_add(tensor_gather_rows(m.E, x), m.P);
    &FloatTensor h1 = tensor_add(h, m.mha.forward(m.ln1.forward(h)));
    &FloatTensor h2 = tensor_add(h1, m.ffn.forward(m.ln2.forward(h1)));
    return tensor_cross_entropy_loss(tensor_matmul(h2, m.Wout), y);
}

static struct Dataset lm_data_(unsigned long n, &Rng rng) {
    struct Dataset d;
    d.x = FloatTensor::new_2d(n, 1UL, 0);
    d.y = FloatTensor::new_2d(n, 1UL, 0);
    d.n = n; d.features = 1UL; d.targets = 1UL; d.classes = LM_VOCAB;
    // a noisy cycle: mostly the next token after the current one
    unsigned long t = 0UL;
    unsigned long i = 0UL;
    while (i <= n) {
        unsigned long nxt = rng.below(4UL) == 0UL ? rng.below(LM_VOCAB) : (t + 3UL) % LM_VOCAB;
        if (i < n) d.x.set1(i, (float)t);
        if (i > 0UL) d.y.set1(i - 1UL, (float)t);
        t = nxt;
        i = i + 1UL;
    }
    return d;
}

static void w_lm_(&Dataset d) {
    struct Rng rng = Rng::new(12UL);
    struct TinyLm m;
    m.E = FloatTensor::new_2d(LM_VOCAB, LM_DIM, 1);
    m.P = FloatTensor::new_2d(LM_SEQ, LM_DIM, 1);
    m.Wout = FloatTensor::new_2d(LM_DIM, LM_VOCAB, 1);
    rng.fill_normal(m.E, (float)0.0, (float)0.5);
    rng.fill_normal(m.P, (float)0.0, (float)0.1);
    rng.fill_uniform(m.Wout, (float)-0.35, (float)0.35);
    m.ln1 = LayerNorm::new(LM_DIM);
    m.ln2 = LayerNorm::new(LM_DIM);
    m.mha = MultiHeadAttention::new(LM_DIM, 2UL, &rng);
    m.mha.causal = 1;
    m.ffn = Mlp::new(LM_DIM, 4UL * LM_DIM, 1UL, LM_DIM, ACT_GELU, ACT_NONE, &rng);
    struct Optimizer opt = Optimizer::new(OPT_ADAMW, (float)0.01);
    opt.add(m.E); opt.add(m.P); opt.add(m.Wout);
    model_add_params(&m.ln1, &opt); model_add_params(&m.ln2, &opt);
    model_add_params(&m.mha, &opt); model_add_params(&m.ffn, &opt);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 3UL; tc.batchSize = LM_SEQ; tc.shuffle = 0; tc.device = g_dev; tc.clipNorm = (float)1.0;
    g_loss = trainer_fit(&m, lm_loss_, &opt, d, &tc).lastEpochLoss;
    opt.free(); m.ln1.free(); m.ln2.free(); m.mha.free(); m.ffn.free();
    m.E.free(); m.P.free(); m.Wout.free();
}

// A DiT block (adaLN conditioning) regressing a token sequence.
struct DitModel {
    struct DiTBlock blk;
    &FloatTensor c;   // [1, condDim], fixed
};

static &FloatTensor dit_loss_(&DitModel m, const &FloatTensor x, const &FloatTensor y) {
    return tensor_mse_loss(m.blk.forward(x, m.c), y);
}

static void w_dit_(&Dataset d) {
    struct Rng rng = Rng::new(13UL);
    struct DitModel m;
    m.blk = DiTBlock::new(8UL, 16UL, 4UL, 2UL);
    m.c = FloatTensor::new_2d(1UL, 4UL, 0);
    rng.fill_normal(m.c, (float)0.0, (float)1.0);
    if (g_dev != DEVICE_CPU) m.c.place(g_dev);
    struct Optimizer opt = Optimizer::new(OPT_ADAM, (float)0.003);
    model_add_params(&m.blk, &opt);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 2UL; tc.batchSize = 8UL; tc.shuffle = 0; tc.device = g_dev;
    g_loss = trainer_fit(&m, dit_loss_, &opt, d, &tc).lastEpochLoss;
    opt.free(); m.blk.free(); m.c.free();
}

// A JiT block stack feeding multi-head latent attention.
struct JitMla {
    struct JiT jit;
    struct Mla mla;
    &FloatTensor cond;   // [1, dModel], fixed
};

static &FloatTensor jit_mla_loss_(&JitMla m, const &FloatTensor x, const &FloatTensor y) {
    return tensor_mse_loss(m.mla.forward(m.jit.forward(x, m.cond)), y);
}

static void w_jit_mla_(&Dataset d) {
    struct Rng rng = Rng::new(14UL);
    struct JitMla m;
    m.jit = JiT::new(6UL, 8UL, 16UL, 2UL, 1UL);
    m.mla = Mla::new(8UL, 4UL, 2UL, &rng);
    m.cond = FloatTensor::new_2d(1UL, 8UL, 0);
    rng.fill_normal(m.cond, (float)0.0, (float)0.5);
    if (g_dev != DEVICE_CPU) m.cond.place(g_dev);
    struct Optimizer opt = Optimizer::new(OPT_ADAM, (float)0.003);
    model_add_params(&m.jit, &opt);
    model_add_params(&m.mla, &opt);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 2UL; tc.batchSize = 8UL; tc.shuffle = 0; tc.device = g_dev;
    g_loss = trainer_fit(&m, jit_mla_loss_, &opt, d, &tc).lastEpochLoss;
    opt.free(); m.jit.free(); m.mla.free(); m.cond.free();
}

// n tokens of dim features, regression targets of tdim.
static struct Dataset tok_data_(unsigned long n, unsigned long dim, unsigned long tdim, &Rng rng) {
    struct Dataset d;
    d.x = FloatTensor::new_2d(n, dim, 0);
    d.y = FloatTensor::new_2d(n, tdim, 0);
    rng.fill_normal(d.x, (float)0.0, (float)1.0);
    rng.fill_normal(d.y, (float)0.0, (float)0.5);
    d.n = n; d.features = dim; d.targets = tdim; d.classes = 0UL;
    return d;
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
static struct Dataset g_lm;
static struct Dataset g_tok;
static struct Dataset g_patch;

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
static void wk_lm() { w_lm_(&g_lm); }
static void wk_dit() { w_dit_(&g_tok); }
static void wk_jit_mla() { w_jit_mla_(&g_patch); }

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
    struct Rng trng = Rng::new(2UL);
    g_lm = lm_data_(64UL, &trng);
    g_tok = tok_data_(48UL, 8UL, 8UL, &trng);
    g_patch = tok_data_(48UL, 6UL, 8UL, &trng);
    const char* names[15];
    names[0] = "mlp"; names[1] = "vae"; names[2] = "cvae"; names[3] = "gan"; names[4] = "hinge gan";
    names[5] = "dqn"; names[6] = "ppo"; names[7] = "lora"; names[8] = "conv autoencoder"; names[9] = "layer norm + losses";
    names[10] = "cnn (batch / group norm, pools, residual)"; names[11] = "spatial (upsample, resize, dropout, concat)";
    names[12] = "transformer lm (causal mha, layer norm, ffn)"; names[13] = "dit block"; names[14] = "jit + mla";
    fn void() works[15];
    works[0] = wk_mlp; works[1] = wk_vae; works[2] = wk_cvae; works[3] = wk_gan; works[4] = wk_wgan;
    works[5] = wk_dqn; works[6] = wk_ppo; works[7] = wk_lora; works[8] = wk_conv; works[9] = wk_norm;
    works[10] = wk_cnn; works[11] = wk_spatial; works[12] = wk_lm; works[13] = wk_dit; works[14] = wk_jit_mla;
    int k = 0;
    while (k < 15) {
        float c = run_on_(DEVICE_CPU, works[k]);
        float r = run_on_(CHECK_DEVICE, works[k]);
        unsafe {
            print("  "); print(names[k]); print(": cpu "); print_float((double)c); print(" device "); println_float((double)r);
            check_(names[k], near_(c, r, (float)2e-4));
        }
        k = k + 1;
    }
    // the transformers through the composed attention ops instead of the fused kernels
    attention_set_fused(0);
    k = 12;
    while (k < 15) {
        float c = run_on_(DEVICE_CPU, works[k]);
        float r = run_on_(CHECK_DEVICE, works[k]);
        unsafe {
            print("  composed "); print(names[k]); print(": cpu "); print_float((double)c); print(" device "); println_float((double)r);
            check_(names[k], near_(c, r, (float)2e-4));
        }
        k = k + 1;
    }
    attention_set_fused(1);
    // fp16 and bf16: every product and library convolution in 16 bits
    int pick = device_set_precision_pick(DEVICE_PICK_ALWAYS);
    int prec = DEVICE_FP16;
    while (prec <= DEVICE_BF16) {
        k = 0;
        while (k < 15) {
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
    g_lm.free(); g_tok.free(); g_patch.free();
}

// ── Attention: the device's values and gradients = the CPU's ─────────────────
static int g_akind = 0;
static &FloatTensor at_out_(&FloatTensor Q, &FloatTensor K, &FloatTensor V) {
    if (g_akind == 0) return mha_forward(Q, K, V, 2UL);
    if (g_akind == 1) return mha_forward_mask(Q, K, V, 2UL, 1, 0UL);           // causal
    if (g_akind == 2) return mha_forward_mask(Q, K, V, 2UL, 1, 5UL);           // causal, window 5
    if (g_akind == 3) return attention_forward_mask(Q, K, V, 0, 6UL);           // window only
    if (g_akind == 4) return flash_attention_forward(Q, K, V, 8UL);
    if (g_akind == 5) return linear_attention_forward(Q, K, V);
    return gated_mha_forward(Q, K, V, 2UL, Q.scale((float)0.7));
}

static void at_grads_(&FloatTensor t, &float out) {
    unsafe { tensor_grad_host_((void*)t); unsigned long i = 0UL; while (i < t->size) { out[i] = t->grad[i]; i = i + 1UL; } }
}

static float at_maxrel_(const &float a, const &float b, unsigned long n) {
    float worst = (float)0.0;
    unsafe {
        unsigned long i = 0UL;
        while (i < n) { float e = absf_(a[i] - b[i]) / ((float)1.0 + absf_(a[i])); if (e > worst) worst = e; i = i + 1UL; }
    }
    return worst;
}

void check_attention_() {
    const char* names[7];
    names[0] = "mha"; names[1] = "mha causal"; names[2] = "mha causal window"; names[3] = "attention window";
    names[4] = "flash"; names[5] = "linear"; names[6] = "gated mha";
    struct Rng rng = Rng::new(15UL);
    // 2 heads of 32 over 24 queries and 40 keys
    &FloatTensor Q = FloatTensor::new_2d(24UL, 64UL, 1);
    &FloatTensor K = FloatTensor::new_2d(40UL, 64UL, 1);
    &FloatTensor V = FloatTensor::new_2d(40UL, 64UL, 1);
    &FloatTensor W = FloatTensor::new_2d(24UL, 64UL, 0);
    rng.fill_normal(Q, (float)0.0, (float)0.5);
    rng.fill_normal(K, (float)0.0, (float)0.5);
    rng.fill_normal(V, (float)0.0, (float)1.0);
    rng.fill_normal(W, (float)0.0, (float)1.0);
    float* cq;
    float* cv;
    unsafe { cq = (float*)malloc(4UL * 2560UL * 6UL); cv = &cq[2560UL * 3UL]; }
    g_akind = 0;
    while (g_akind < 7) {
        Q.zero_grad(); K.zero_grad(); V.zero_grad();
        &FloatTensor lc = tensor_mul(at_out_(Q, K, V), W).sum();
        float l0 = lc.at1(0UL);
        lc.backward();
        unsafe { at_grads_(Q, (&float)cq); at_grads_(K, (&float)&cq[2560UL]); at_grads_(V, (&float)&cq[5120UL]); }
        int fused = 1;
        while (fused >= 0) {
            attention_set_fused(fused);
            Q.zero_grad(); K.zero_grad(); V.zero_grad();
            Q.place(CHECK_DEVICE); K.place(CHECK_DEVICE); V.place(CHECK_DEVICE);
            &FloatTensor l = tensor_mul(at_out_(Q, K, V), W).sum();
            float ld = l.at1(0UL);
            int on = l.device() == CHECK_DEVICE;
            l.backward();
            Q.place(DEVICE_CPU); K.place(DEVICE_CPU); V.place(DEVICE_CPU);
            float e = (float)0.0;
            unsafe {
                at_grads_(Q, (&float)cv); at_grads_(K, (&float)&cv[2560UL]); at_grads_(V, (&float)&cv[5120UL]);
                e = at_maxrel_((const &float)cq, (const &float)cv, 1536UL);
                float ek = at_maxrel_((const &float)&cq[2560UL], (const &float)&cv[2560UL], 5120UL);
                if (ek > e) e = ek;
                print(fused ? "  attention fused " : "  attention composed "); print(names[g_akind]);
                print(": loss cpu "); print_float((double)l0); print(" device "); print_float((double)ld);
                print(", worst gradient difference "); println_float((double)e);
                check_(names[g_akind], on && near_(l0, ld, (float)2e-4) && e < (float)2e-4);
            }
            fused = fused - 1;
        }
        attention_set_fused(1);
        g_akind = g_akind + 1;
    }
    unsafe { free((void*)cq); }
    Q.free(); K.free(); V.free(); W.free();
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
    check_attention_();
    check_sparse_();
    // every kernel and GEMM above ran on the GPU, none on the CPU (device_ref)
    struct DeviceStats st = device_stats(CHECK_DEVICE);
    print("  "); print_int((long long)st.launches); print(" kernels, "); print_int((long long)st.gemms);
    print(" GEMMs, "); print_int((long long)st.cpuLaunches); println(" on the CPU");
    check_("no kernel fell back to the CPU", st.cpuLaunches == 0UL);
    print("device_check: "); print_int((long long)g_bad); println(" failed");
    return g_bad == 0 ? 0 : 1;
}
