// std::ml training stack: nn.h ops / losses / optimizers (finite-difference
// gradient checks), datasets.h, trainer.h, and the generative and RL
// models built on them.
#include <std/test/test.h>
#include <std/ml/nn.h>
#include <std/ml/module.h>
#include <std/ml/datasets.h>
#include <std/ml/trainer.h>
#include <std/io_file.h>
#include <std/ml/autoencoder.h>
#include <std/ml/gan.h>
#include <std/math.h>
#include <std/ml/rl.h>
#include <std/ml/convnet.h>
#include <std/ml/cnn.h>
#include <std/sync/parallel.h>

extern void* malloc(unsigned long size);
extern void free(void* ptr);

static void tensor_drop(&Tensor t) {
    t->free();
    unsafe { free((void*)t); }
}

// ── Gradient checks ──────────────────────────────────────────────────────────
// loss(x) for a family of ops; compares Tensor::backward's gradient for x
// with central differences.
static &Tensor gc_loss_(int which, &Tensor x, const &Tensor aux) {
    if (which == 0) return tensor_mul(x.exp(), aux).sum();
    if (which == 1) return (tensor_mul(x.abs().add_scalar((float)1.5).log(), aux)).sum();
    if (which == 2) return tensor_mul(x.log_softmax(), aux).sum();
    if (which == 3) return tensor_cross_entropy_loss(x, aux.argmax_rows());
    if (which == 4) return tensor_bce_logits_loss(x, aux.sigmoid());
    if (which == 5) return tensor_huber_loss(x, aux, (float)0.3);
    if (which == 6) return tensor_mul(x.sum_rows(), aux.sum_rows()).sum();
    if (which == 7) return tensor_mul(tensor_concat_cols(x, aux), tensor_concat_cols(aux, x)).sum();
    if (which == 8) return (tensor_mul(x.leaky_relu((float)0.2), aux)).sum();
    if (which == 9) return tensor_mul(tensor_minimum(x, aux), aux).sum();
    if (which == 10) return (tensor_mul(x.clamp((float)-0.4, (float)0.4), aux)).sum();
    return tensor_mse_loss(tensor_add_bias(x, aux.detach().sum_rows()), aux);
}

static double grad_check_(int which) {
    struct Rng rng = Rng::new(7UL + (unsigned long)which);
    &Tensor x = Tensor::new_2d(4UL, 4UL, 1);
    &Tensor aux = Tensor::new_2d(4UL, 4UL, 0);
    rng.fill_uniform(x, (float)-1.0, (float)1.0);
    rng.fill_uniform(aux, (float)-1.0, (float)1.0);
    // keep finite differences away from the kinks of abs / leaky / min / clamp
    unsafe {
        unsigned long i = 0UL;
        while (i < 16UL) {
            float v = x->data[i];
            if (v > (float)-0.08 && v < (float)0.08) x->data[i] = v + (float)0.2;
            float d = x->data[i] - aux->data[i];
            if (d > (float)-0.05 && d < (float)0.05) x->data[i] = x->data[i] + (float)0.15;
            float c = x->data[i];
            if ((c > (float)0.33 && c < (float)0.47) || (c < (float)-0.33 && c > (float)-0.47)) x->data[i] = c * (float)0.5;
            i = i + 1UL;
        }
    }
    struct TensorArena stepArena = TensorArena::new(0UL);
    stepArena.begin();
    &Tensor l = gc_loss_(which, x, aux);
    l.backward();
    stepArena.reset();
    double worst = 0.0;
    unsafe {
        unsigned long i = 0UL;
        while (i < 16UL) {
            float keep = x->data[i];
            float h = (float)0.002;
            x->data[i] = keep + h;
            float lp = gc_loss_(which, x, aux)->data[0];
            x->data[i] = keep - h;
            float lm = gc_loss_(which, x, aux)->data[0];
            x->data[i] = keep;
            stepArena.reset();
            double num = ((double)lp - (double)lm) / (2.0 * (double)h);
            double e = num - (double)x->grad[i];
            if (e < 0.0) e = -e;
            double sc = num < 0.0 ? -num : num;
            e = e / (sc > 1.0 ? sc : 1.0);
            if (e > worst) worst = e;
            i = i + 1UL;
        }
    }
    tensor_arena_end();
    stepArena.free();
    x->free(); aux->free();
    unsafe { free((void*)x); free((void*)aux); }
    return worst;
}

void t_grad_checks() {
    int w = 0;
    while (w < 12) {
        double e = grad_check_(w);
        if (e >= 2e-3) { ASSERT_TRUE(w < 0); }   // report which op failed
        ASSERT_TRUE(e < 2e-3);
        w = w + 1;
    }
}

void t_gather_rows_grad() {
    &Tensor table = Tensor::new_2d(3UL, 2UL, 1);
    &Tensor idx = Tensor::new_1d(4UL, 0);
    unsafe {
        unsigned long i = 0UL;
        while (i < 6UL) { table->data[i] = (float)i; i = i + 1UL; }
        idx->data[0] = (float)2.0; idx->data[1] = (float)0.0; idx->data[2] = (float)2.0; idx->data[3] = (float)1.0;
    }
    &Tensor g = tensor_gather_rows(table, idx);
    unsafe {
        ASSERT_TRUE(g->data[0] == (float)4.0 && g->data[1] == (float)5.0 && g->data[2] == (float)0.0);
    }
    g.sum().backward();
    unsafe {
        ASSERT_TRUE(table->grad[4] == (float)2.0);   // row 2 gathered twice
        ASSERT_TRUE(table->grad[0] == (float)1.0);
        ASSERT_TRUE(table->grad[2] == (float)1.0);
    }
}

// Every optimizer drives a quadratic bowl to its minimum.
void t_optimizers() {
    int kinds[6];
    kinds[0] = OPT_SGD; kinds[1] = OPT_MOMENTUM; kinds[2] = OPT_NESTEROV;
    kinds[3] = OPT_RMSPROP; kinds[4] = OPT_ADAM; kinds[5] = OPT_ADAMW;
    int k = 0;
    while (k < 6) {
        &Tensor w = Tensor::new_2d(1UL, 3UL, 1);
        &Tensor target = Tensor::new_2d(1UL, 3UL, 0);
        unsafe { target->data[0] = (float)1.0; target->data[1] = (float)-2.0; target->data[2] = (float)0.5; }
        struct Optimizer opt = Optimizer::new(kinds[k], kinds[k] == OPT_SGD ? (float)0.5 : (float)0.05);
        opt.weightDecay = (float)0.0;
        opt.add(w);
        struct TensorArena stepArena = TensorArena::new(0UL);
        unsigned long s = 0UL;
        while (s < 400UL) {
            stepArena.begin();
            &Tensor l = tensor_mse_loss(w, target);
            trainer_step_(&opt, l, (float)0.0);
            tensor_arena_end();
            stepArena.reset();
            s = s + 1UL;
        }
        unsafe {
            float e = (float)0.0;
            unsigned long i = 0UL;
            while (i < 3UL) { float d = w->data[i] - target->data[i]; e = e + d * d; i = i + 1UL; }
            ASSERT_TRUE(e < (float)1e-3);
        }
        stepArena.free();
        opt.free();
        k = k + 1;
    }
}

// ── Datasets ─────────────────────────────────────────────────────────────────
void t_datasets() {
    struct Rng rng = Rng::new(3UL);
    struct Dataset m = Dataset::moons(200UL, (float)0.05, &rng);
    ASSERT_EQ(m.n, 200UL); ASSERT_EQ(m.features, 2UL); ASSERT_EQ(m.classes, 2UL);
    struct DatasetSplit sp = m.split((float)0.25, &rng);
    ASSERT_EQ(sp.test.n, 50UL); ASSERT_EQ(sp.train.n, 150UL);
    sp.train.standardize_pair(&sp.test);
    unsafe {
        double s = 0.0;
        unsigned long i = 0UL;
        while (i < 150UL) { s = s + (double)sp.train.x->data[i * 2UL]; i = i + 1UL; }
        ASSERT_TRUE(s / 150.0 < 1e-4 && s / 150.0 > -1e-4);
    }
    // every row comes out once per epoch, in batches of 32 and a tail of 22
    struct DataLoader dl = DataLoader::new(&sp.train, 32UL, 1, 9UL);
    ASSERT_EQ(dl.batches(), 5UL);
    struct Batch b;
    unsigned long rows = 0UL;
    unsigned long batches = 0UL;
    while (dl.next(&b)) { rows = rows + b.rows; batches = batches + 1UL; }
    ASSERT_EQ(rows, 150UL); ASSERT_EQ(batches, 5UL);
    dl.free();

    struct Dataset sh = Dataset::shapes(8UL, 12UL, &rng);
    ASSERT_EQ(sh.features, 144UL);
    unsafe {
        unsigned long i = 0UL;
        while (i < 8UL) {
            float on = (float)0.0;
            unsigned long p = 0UL;
            while (p < 144UL) { on = on + sh.x->data[i * 144UL + p]; p = p + 1UL; }
            ASSERT_TRUE(on >= (float)4.0);
            i = i + 1UL;
        }
    }
    struct Dataset bl = Dataset::blobs(90UL, 3UL, 5UL, (float)0.5, &rng);
    ASSERT_EQ(bl.features, 5UL); ASSERT_EQ(bl.classes, 3UL);
    struct Dataset gr = Dataset::gaussian_ring(64UL, 8UL, (float)2.0, (float)0.02, &rng);
    unsafe {
        float x = gr.x->data[0]; float y = gr.x->data[1];
        float r = x * x + y * y;
        ASSERT_TRUE(r > (float)3.5 && r < (float)4.5);
    }
    gr.free(); bl.free(); sh.free();
    sp.train.free(); sp.test.free(); m.free();
}

void t_dataset_csv() {
    match (file_open("build/test_ml_train.csv", "w")) {
        case some(f): {
            file_puts("a,b,label\n1.5, 2,0\n-1,3.25,2\n\n4,5,1\n", f);
            file_close(f);
        }
        case null: { ASSERT_TRUE(0); }
    }
    struct Dataset d = Dataset::load_csv("build/test_ml_train.csv", 2L, 1, 1);
    ASSERT_EQ(d.n, 3UL); ASSERT_EQ(d.features, 2UL); ASSERT_EQ(d.classes, 3UL);
    unsafe {
        ASSERT_TRUE(d.x->data[0] == (float)1.5 && d.x->data[1] == (float)2.0);
        ASSERT_TRUE(d.x->data[3] == (float)3.25 && d.y->data[1] == (float)2.0);
    }
    d.free();
    // two 2x2 images and their labels in IDX format (big-endian headers)
    unsigned char img[24];
    unsigned char lab[10];
    unsafe {
        unsigned long i = 0UL;
        while (i < 24UL) { img[i] = (unsigned char)0; i = i + 1UL; }
        i = 0UL;
        while (i < 10UL) { lab[i] = (unsigned char)0; i = i + 1UL; }
        img[2] = (unsigned char)8; img[3] = (unsigned char)3;      // magic 2051
        img[7] = (unsigned char)2; img[11] = (unsigned char)2; img[15] = (unsigned char)2;
        img[16] = (unsigned char)255; img[19] = (unsigned char)51; img[23] = (unsigned char)102;
        lab[2] = (unsigned char)8; lab[3] = (unsigned char)1;      // magic 2049
        lab[7] = (unsigned char)2; lab[8] = (unsigned char)7; lab[9] = (unsigned char)3;
    }
    match (file_open("build/test_ml_train.idx3", "wb")) {
        case some(f): { unsafe { file_write(f, (const &unsigned char)&img[0], 24UL); } file_close(f); }
        case null: { ASSERT_TRUE(0); }
    }
    match (file_open("build/test_ml_train.idx1", "wb")) {
        case some(f): { unsafe { file_write(f, (const &unsigned char)&lab[0], 10UL); } file_close(f); }
        case null: { ASSERT_TRUE(0); }
    }
    struct Dataset mn = Dataset::load_mnist("build/test_ml_train.idx3", "build/test_ml_train.idx1", 0UL);
    ASSERT_EQ(mn.n, 2UL); ASSERT_EQ(mn.features, 4UL); ASSERT_EQ(mn.classes, 10UL);
    unsafe {
        ASSERT_TRUE(mn.x->data[0] == (float)1.0 && mn.x->data[3] == (float)0.2 && mn.x->data[7] == (float)0.4);
        ASSERT_TRUE(mn.y->data[0] == (float)7.0 && mn.y->data[1] == (float)3.0);
    }
    mn.free();
    struct Dataset none = Dataset::load_mnist("build/no-such-images", "build/no-such-labels", 0UL);
    ASSERT_EQ(none.n, 0UL);
    none.free();
}

// ── Trainer ──────────────────────────────────────────────────────────────────
static &Tensor clf_loss_(&Mlp m, const &Tensor x, const &Tensor y) {
    return tensor_cross_entropy_loss(m.forward(x), y);
}
static &Tensor clf_logits_(&Mlp m, const &Tensor x) { return m.forward(x); }

void t_trainer_moons() {
    struct Rng rng = Rng::new(11UL);
    struct Dataset data = Dataset::moons(400UL, (float)0.1, &rng);
    struct DatasetSplit sp = data.split((float)0.25, &rng);
    struct Mlp net = Mlp::new(2UL, 32UL, 2UL, 2UL, ACT_RELU, ACT_NONE, &rng);
    struct Optimizer opt = Optimizer::new(OPT_ADAM, (float)0.01);
    model_add_params(&net, &opt);
    struct TrainConfig cfg = TrainConfig::default();
    cfg.epochs = 60UL;
    struct TrainReport rep = trainer_fit(&net, clf_loss_, &opt, &sp.train, &cfg);
    ASSERT_EQ(rep.epochs, 60UL);
    ASSERT_TRUE(rep.lastEpochLoss < rep.firstEpochLoss * (float)0.2);
    float acc = trainer_accuracy(&net, clf_logits_, &sp.test, 64UL);
    ASSERT_TRUE(acc > (float)0.95);
    float val = trainer_evaluate(&net, clf_loss_, &sp.test, 64UL);
    ASSERT_TRUE(val < (float)0.2);
    opt.free();
    net.free();
    sp.train.free(); sp.test.free(); data.free();
}

static &Tensor reg_loss_(&Mlp m, const &Tensor x, const &Tensor y) {
    return tensor_mse_loss(m.forward(x), y);
}

void t_trainer_sine() {
    struct Rng rng = Rng::new(5UL);
    struct Dataset data = Dataset::sine(512UL, (float)0.0, &rng);
    struct Mlp net = Mlp::new(1UL, 32UL, 2UL, 1UL, ACT_TANH, ACT_NONE, &rng);
    struct Optimizer opt = Optimizer::new(OPT_ADAMW, (float)0.01);
    opt.weightDecay = (float)0.0001;
    model_add_params(&net, &opt);
    struct TrainConfig cfg = TrainConfig::default();
    cfg.epochs = 80UL;
    cfg.clipNorm = (float)1.0;
    cfg.lrDecay = (float)0.98;
    struct TrainReport rep = trainer_fit(&net, reg_loss_, &opt, &data, &cfg);
    ASSERT_TRUE(rep.lastEpochLoss < (float)0.01);
    opt.free();
    net.free();
    data.free();
}

// ── Autoencoders ─────────────────────────────────────────────────────────────
static float recon_err_(&Autoencoder ae, const &Dataset d) {
    struct GradGuard noGrad = Autograd::get().no_grad();
    &Tensor r = ae.reconstruct(d.x);
    float e;
    unsafe { e = tensor_mse_loss(r, d.x)->data[0]; }
    noGrad.restore();
    return e;
}

void t_autoencoders() {
    struct Rng rng = Rng::new(21UL);
    struct Dataset blobs = Dataset::blobs(256UL, 4UL, 8UL, (float)0.3, &rng);
    struct Dataset shapes = Dataset::shapes(256UL, 10UL, &rng);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 40UL;
    // plain, denoising (on images, BCE) and sparse
    int variant = 0;
    while (variant < 3) {
        struct AutoencoderConfig cfg = AutoencoderConfig::default(variant == 1 ? 16UL : 3UL);
        cfg.hidden = 64UL;
        &Dataset d = variant == 1 ? &shapes : &blobs;
        if (variant == 1) { cfg.recon = RECON_BCE; cfg.noiseStd = (float)0.2; cfg.dropProb = (float)0.1; }
        if (variant == 2) cfg.sparsity = (float)0.05;
        struct Autoencoder ae = Autoencoder::new(d.features, &cfg);
        float before = recon_err_(&ae, d);
        struct Optimizer opt = Optimizer::new(OPT_ADAM, (float)0.003);
        model_add_params(&ae, &opt);
        model_fit(&ae, &opt, d, &tc);
        float after = recon_err_(&ae, d);
        ASSERT_TRUE(after < before * (float)0.3);
        if (variant == 2) ASSERT_TRUE(ae.lastPenalty > (float)0.0);
        opt.free();
        ae.free();
        variant = variant + 1;
    }
    blobs.free();
    shapes.free();
}

static &Tensor shapes_logits_(&Mlp m, const &Tensor x) { return m.forward(x); }

void t_vae() {
    struct Rng rng = Rng::new(4UL);
    struct Dataset shapes = Dataset::shapes(512UL, 10UL, &rng);
    // a classifier that later judges the CVAE's samples
    struct Mlp judge = Mlp::new(100UL, 64UL, 1UL, 4UL, ACT_RELU, ACT_NONE, &rng);
    struct Optimizer jo = Optimizer::new(OPT_ADAM, (float)0.003);
    model_add_params(&judge, &jo);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 30UL;
    trainer_fit(&judge, clf_loss_, &jo, &shapes, &tc);
    ASSERT_TRUE(trainer_accuracy(&judge, shapes_logits_, &shapes, 128UL) > (float)0.95);

    int cond = 0;
    while (cond < 2) {
        struct VaeConfig cfg = VaeConfig::default(4UL);
        cfg.hidden = 128UL;
        cfg.classes = cond ? 4UL : 0UL;
        struct Vae v = Vae::new(100UL, &cfg);
        struct Optimizer opt = Optimizer::new(OPT_ADAM, (float)0.002);
        model_add_params(&v, &opt);
        tc.epochs = 60UL;
        struct TrainReport rep = model_fit(&v, &opt, &shapes, &tc);
        ASSERT_TRUE(rep.lastEpochLoss < rep.firstEpochLoss * (float)0.6);
        ASSERT_TRUE(v.lastKl > (float)0.1);
        &Tensor labels = Tensor::new_1d(64UL, 0);
        unsafe { unsigned long i = 0UL; while (i < 64UL) { labels->data[i] = (float)(i % 4UL); i = i + 1UL; } }
        struct GradGuard noGrad = Autograd::get().no_grad();
        &Tensor s = cond ? v.sample(64UL, (?&Tensor)labels) : v.sample(64UL, null);
        unsafe {
            ASSERT_EQ(s->shape[0], 64UL);
            ASSERT_EQ(s->shape[1], 100UL);
            ASSERT_TRUE(s->data[0] > (float)0.0 && s->data[0] < (float)1.0);
        }
        if (cond) {
            // conditioned samples are recognized as their class
            float acc = tensor_accuracy(judge.forward(s), labels);
            ASSERT_TRUE(acc > (float)0.7);
        }
        noGrad.restore();
        opt.free();
        v.free();
        cond = cond + 1;
    }
    jo.free();
    judge.free();
    shapes.free();
}

void t_vqvae() {
    struct Rng rng = Rng::new(8UL);
    struct Dataset blobs = Dataset::blobs(512UL, 8UL, 6UL, (float)0.2, &rng);
    struct VqVaeConfig cfg = VqVaeConfig::default(4UL);
    cfg.hidden = 64UL;
    cfg.codes = 16UL;
    struct VqVae v = VqVae::new(6UL, &cfg);
    struct Optimizer opt = Optimizer::new(OPT_ADAM, (float)0.003);
    model_add_params(&v, &opt);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 60UL;
    struct TrainReport rep = model_fit(&v, &opt, &blobs, &tc);
    ASSERT_TRUE(rep.lastEpochLoss < rep.firstEpochLoss * (float)0.2);
    ASSERT_TRUE(v.lastPerplexity > (float)4.0);     // codes in use, not collapsed to one
    struct GradGuard noGrad = Autograd::get().no_grad();
    &Tensor codes = v.encode(blobs.x);
    &Tensor a = v.decode(codes);
    &Tensor b = v.reconstruct(blobs.x);
    unsafe {
        ASSERT_TRUE(a->data[5] == b->data[5]);
        ASSERT_TRUE(tensor_mse_loss(b, blobs.x)->data[0] < (float)1.0);
    }
    noGrad.restore();
    opt.free();
    v.free();
    blobs.free();
}

// ── GANs ─────────────────────────────────────────────────────────────────────
// Mean |grad_x D| of g's critic at midpoints of real and generated rows.
static double critic_grad_norm_(&Gan g, const &Tensor real, const &Tensor fake) {
    &Tensor xh = Tensor::new_2d(256UL, 2UL, 1);
    unsafe {
        unsigned long i = 0UL;
        while (i < 512UL) { xh->data[i] = (float)0.5 * (real->data[i] + fake->data[i]); i = i + 1UL; }
    }
    g.score(xh, null).sum().backward();
    double acc = 0.0;
    unsafe {
        unsigned long r = 0UL;
        while (r < 256UL) {
            double a = (double)xh->grad[r * 2UL];
            double b = (double)xh->grad[r * 2UL + 1UL];
            acc = acc + sqrt_d(a * a + b * b);
            r = r + 1UL;
        }
    }
    tensor_drop(xh);
    return acc / 256.0;
}

// Each objective learns a 2D Gaussian at (2, -1) with std (0.5, 0.3):
// the mean tightly, the spread loosely.
void t_gans() {
    struct Rng rng = Rng::new(13UL);
    struct Dataset target = Dataset::blobs(1024UL, 1UL, 2UL, (float)0.0, &rng);
    unsafe {
        unsigned long i = 0UL;
        while (i < 1024UL) {
            target.x->data[i * 2UL] = (float)2.0 + (float)0.5 * rng.normal();
            target.x->data[i * 2UL + 1UL] = (float)-1.0 + (float)0.3 * rng.normal();
            i = i + 1UL;
        }
    }
    int k = 0;
    while (k < 5) {
        struct GanConfig cfg = GanConfig::default(4UL);
        cfg.hidden = 32UL;
        cfg.loss = k;
        cfg.emaDecay = (float)0.99;
        cfg.lrG = (float)0.0005; cfg.lrD = (float)0.0005;
        if (k == GAN_WGAN) { cfg.nCritic = 5UL; cfg.lrG = (float)0.0005; cfg.lrD = (float)0.0005; cfg.clip = (float)0.05; }
        if (k == GAN_WGAN_GP) cfg.nCritic = 5UL;
        struct Gan g = Gan::new(2UL, &cfg);
        g.train(&target, k == GAN_WGAN || k == GAN_WGAN_GP ? 100UL : 80UL, 64UL);
        &Tensor s = g.sample(2000UL, null);
        unsafe {
            double mx = 0.0; double my = 0.0; double vx = 0.0; double vy = 0.0;
            unsigned long i = 0UL;
            while (i < 2000UL) { mx = mx + (double)s->data[i * 2UL]; my = my + (double)s->data[i * 2UL + 1UL]; i = i + 1UL; }
            mx = mx / 2000.0; my = my / 2000.0;
            i = 0UL;
            while (i < 2000UL) {
                double dx = (double)s->data[i * 2UL] - mx; double dy = (double)s->data[i * 2UL + 1UL] - my;
                vx = vx + dx * dx; vy = vy + dy * dy;
                i = i + 1UL;
            }
            double sx = sqrt_d(vx / 2000.0); double sy = sqrt_d(vy / 2000.0);
            ASSERT_TRUE(mx > 1.75 && mx < 2.25);
            ASSERT_TRUE(my > -1.25 && my < -0.75);
            // GANs tend to under-cover the spread; bound it loosely
            ASSERT_TRUE(sx > 0.2 && sx < 1.0);
            ASSERT_TRUE(sy > 0.08 && sy < 0.6);
        }
        // the penalty holds the critic's input gradient near 1
        if (k == GAN_WGAN_GP) {
            double gn = critic_grad_norm_(&g, target.x, s);
            ASSERT_TRUE(gn > 0.6 && gn < 1.4);
        }
        tensor_drop(s);
        g.free();
        k = k + 1;
    }
    target.free();
}

// A conditional GAN puts each label's samples on its own mode of a ring.
void t_cgan() {
    struct Rng rng = Rng::new(17UL);
    struct Dataset ring = Dataset::gaussian_ring(2048UL, 4UL, (float)2.0, (float)0.1, &rng);
    struct GanConfig cfg = GanConfig::default(4UL);
    cfg.hidden = 64UL;
    cfg.classes = 4UL;
    cfg.lrG = (float)0.001; cfg.lrD = (float)0.001;
    struct Gan g = Gan::new(2UL, &cfg);
    g.train(&ring, 30UL, 64UL);
    &Tensor labels = Tensor::new_1d(400UL, 0);
    unsafe { unsigned long i = 0UL; while (i < 400UL) { labels->data[i] = (float)(i % 4UL); i = i + 1UL; } }
    &Tensor s = g.sample(400UL, (?&Tensor)labels);
    unsigned long hit = 0UL;
    unsafe {
        unsigned long i = 0UL;
        while (i < 400UL) {
            float t = (float)6.283185307179586 * (float)(i % 4UL) / (float)4.0;
            float dx = s->data[i * 2UL] - (float)2.0 * cos_f(t);
            float dy = s->data[i * 2UL + 1UL] - (float)2.0 * sin_f(t);
            if (dx * dx + dy * dy < (float)0.5) hit = hit + 1UL;
            i = i + 1UL;
        }
        s->free(); free((void*)s);
    }
    ASSERT_TRUE(hit > 300UL);
    g.free();
    ring.free();
}

// ── Reinforcement learning ───────────────────────────────────────────────────
void t_bandits() {
    int kind = 0;
    while (kind < 4) {
        struct Bandit b = Bandit::new(10UL, 3UL);
        float param = kind == BANDIT_EPS_GREEDY ? (float)0.1 :
                      (kind == BANDIT_UCB1 ? (float)1.414 : (kind == BANDIT_THOMPSON ? (float)1.0 : (float)0.1));
        struct BanditAgent ag = BanditAgent::new(kind, 10UL, param, 5UL);
        unsigned long best = b.best_arm();
        unsigned long hits = 0UL;
        unsigned long t = 0UL;
        while (t < 2000UL) {
            unsigned long a = ag.select();
            ag.update(a, b.pull(a));
            if (t >= 1000UL && a == best) hits = hits + 1UL;
            t = t + 1UL;
        }
        ASSERT_TRUE(hits > 850UL);
        ag.free();
        b.free();
        kind = kind + 1;
    }
}

// Greedy return on the cliff walk after training with one of the TD rules.
static float cliff_return_(int algo) {
    struct GridWorld g = GridWorld::cliff(1UL);
    struct TabularAgent ag = TabularAgent::new(g.states(), 4UL, (float)0.5, (float)1.0, (float)0.1, 2UL);
    int ep = 0;
    while (ep < 500) {
        unsigned long s = g.reset();
        unsigned long a = ag.act(s);
        while (1) {
            unsigned long s2 = 0UL;
            struct EnvStep st = g.step(a, &s2);
            unsigned long a2 = ag.act(s2);
            if (algo == 0) ag.q_learning_update(s, a, st.reward, s2, st.terminated);
            else if (algo == 1) ag.sarsa_update(s, a, st.reward, s2, a2, st.terminated);
            else ag.expected_sarsa_update(s, a, st.reward, s2, st.terminated);
            s = s2;
            a = a2;
            if (st.terminated || st.truncated) break;
        }
        ep = ep + 1;
    }
    unsigned long s = g.reset();
    float ret = (float)0.0;
    unsigned long n = 0UL;
    while (n < 100UL) {
        unsigned long s2 = 0UL;
        struct EnvStep st = g.step(ag.greedy(s), &s2);
        ret = ret + st.reward;
        s = s2;
        n = n + 1UL;
        if (st.terminated) break;
    }
    ag.free();
    g.free();
    return ret;
}

void t_tabular_cliff() {
    // Sutton & Barto Example 6.6: Q-learning learns the optimal path along
    // the cliff edge (13 steps), SARSA a safer, longer one.
    float q = cliff_return_(0);
    float sarsa = cliff_return_(1);
    float expected = cliff_return_(2);
    ASSERT_TRUE(q == (float)-13.0);
    ASSERT_TRUE(sarsa < (float)-13.0 && sarsa > (float)-25.0);
    ASSERT_TRUE(expected <= (float)-13.0 && expected > (float)-25.0);
}

void t_gridworld_env() {
    struct GridWorld g = GridWorld::new(3UL, 3UL, 1UL);
    g.set(1UL, 0UL, GRID_WALL);
    g.set(0UL, 2UL, GRID_PIT);
    g.pitTerminal = 1;
    unsigned long s = g.reset();
    ASSERT_EQ(s, 0UL);
    unsigned long n = 0UL;
    struct EnvStep st = g.step(GRID_RIGHT, &n);   // into the wall: stays
    ASSERT_EQ(n, 0UL);
    ASSERT_TRUE(st.reward == (float)-1.0 && !st.terminated);
    g.step(GRID_DOWN, &n);
    st = g.step(GRID_DOWN, &n);                    // into the pit
    ASSERT_TRUE(st.terminated && st.reward == (float)-100.0);
    &Tensor oh = Tensor::new_1d(9UL, 0);
    g.one_hot(4UL, oh);
    unsafe { ASSERT_TRUE(oh->data[4] == (float)1.0 && oh->data[3] == (float)0.0); }
    g.free();
}

void t_replay_buffer() {
    struct ReplayBuffer rb = ReplayBuffer::new(3UL, 2UL);
    &Tensor o = Tensor::new_1d(2UL, 0);
    unsigned long i = 0UL;
    while (i < 5UL) {
        unsafe { o->data[0] = (float)i; }
        rb.push(o, i, (float)i, o, 0);
        i = i + 1UL;
    }
    ASSERT_EQ(rb.size, 3UL);
    ASSERT_EQ(rb.pos, 2UL);
    unsafe { ASSERT_TRUE(rb.obs->data[0] == (float)3.0 && rb.reward->data[1] == (float)4.0); }
    rb.free();
}

static void rl_copy4_(&Tensor dst, const &Tensor src) {
    unsafe { unsigned long i = 0UL; while (i < 4UL) { dst->data[i] = src->data[i]; i = i + 1UL; } }
}

// Mean greedy return over 5 CartPole episodes.
static float eval_dqn_(&Dqn d) {
    struct CartPole env = CartPole::new(99UL);
    &Tensor obs = Tensor::new_1d(4UL, 0);
    &Tensor nx = Tensor::new_1d(4UL, 0);
    float total = (float)0.0;
    int ep = 0;
    while (ep < 5) {
        env.reset(obs);
        while (1) {
            struct EnvStep s = env.step(d.act(obs, 0), nx);
            total = total + s.reward;
            rl_copy4_(obs, nx);
            if (s.terminated || s.truncated) break;
        }
        ep = ep + 1;
    }
    tensor_drop(obs); tensor_drop(nx);
    return total / (float)5.0;
}

static float eval_pg_(&PolicyAgent p) {
    struct CartPole env = CartPole::new(99UL);
    &Tensor obs = Tensor::new_1d(4UL, 0);
    &Tensor nx = Tensor::new_1d(4UL, 0);
    float total = (float)0.0;
    int ep = 0;
    while (ep < 5) {
        env.reset(obs);
        while (1) {
            struct EnvStep s = env.step(p.act(obs, 0), nx);
            total = total + s.reward;
            rl_copy4_(obs, nx);
            if (s.terminated || s.truncated) break;
        }
        ep = ep + 1;
    }
    tensor_drop(obs); tensor_drop(nx);
    return total / (float)5.0;
}

// CartPole counts as solved at a mean greedy return of 195.
void t_dqn_cartpole() {
    struct DqnConfig cfg = DqnConfig::default();
    struct Dqn d = Dqn::new(CARTPOLE_OBS, CARTPOLE_ACTIONS, &cfg);
    struct CartPole env = CartPole::new(8UL);
    &Tensor obs = Tensor::new_1d(4UL, 0);
    &Tensor nx = Tensor::new_1d(4UL, 0);
    unsigned long steps = 0UL;
    int ep = 0;
    float best = (float)0.0;
    while (steps < 40000UL && best < (float)195.0) {
        env.reset(obs);
        while (1) {
            unsigned long a = d.act(obs, 1);
            struct EnvStep s = env.step(a, nx);
            d.observe(obs, a, s.reward, nx, s.terminated);
            steps = steps + 1UL;
            rl_copy4_(obs, nx);
            if (s.terminated || s.truncated) break;
        }
        ep = ep + 1;
        if (ep % 20 == 0) { float e = eval_dqn_(&d); if (e > best) best = e; }
    }
    ASSERT_TRUE(best >= (float)195.0);
    ASSERT_TRUE(d.updates > 0UL);
    tensor_drop(obs); tensor_drop(nx);
    d.free();
}

static float pg_cartpole_(int algo, unsigned long rolloutLen) {
    struct PgConfig cfg = PgConfig::default(algo);
    struct PolicyAgent p = PolicyAgent::new(CARTPOLE_OBS, CARTPOLE_ACTIONS, &cfg);
    struct CartPole env = CartPole::new(12UL);
    &Tensor obs = Tensor::new_1d(4UL, 0);
    &Tensor nx = Tensor::new_1d(4UL, 0);
    unsigned long steps = 0UL;
    int ep = 0;
    float best = (float)0.0;
    env.reset(obs);
    while (steps < 60000UL && best < (float)195.0) {
        unsigned long a = p.act(obs, 1);
        struct EnvStep s = env.step(a, nx);
        p.record(obs, a, s.reward, nx, s.terminated, s.truncated);
        steps = steps + 1UL;
        rl_copy4_(obs, nx);
        int done = s.terminated || s.truncated;
        if ((algo == PG_REINFORCE && done) || (algo != PG_REINFORCE && p.rollout >= rolloutLen)) p.update();
        if (done) {
            ep = ep + 1;
            env.reset(obs);
            if (ep % 25 == 0) { float e = eval_pg_(&p); if (e > best) best = e; }
        }
    }
    tensor_drop(obs); tensor_drop(nx);
    p.free();
    return best;
}

void t_reinforce_cartpole() { ASSERT_TRUE(pg_cartpole_(PG_REINFORCE, 0UL) >= (float)195.0); }
void t_a2c_cartpole() { ASSERT_TRUE(pg_cartpole_(PG_A2C, 32UL) >= (float)195.0); }
void t_ppo_cartpole() { ASSERT_TRUE(pg_cartpole_(PG_PPO, 1024UL) >= (float)195.0); }

// ── Convolutional ops ────────────────────────────────────────────────────────
// Each case: the op, x's geometry (2 images, 2 channels), and whether it
// has trained parameters (w: weight / gamma, b: bias / beta). rm / rv are
// batch norm's running statistics. Losses are sum(op(x) * aux) for a fixed
// random aux.
#define CG_CASES 32

static unsigned long cg_side_(int which) {
    int w = which;
    if (w == 2 || w == 3 || w == 6 || w == 11 || w == 12 || w == 15 || w == 16 || w == 20 || w == 21
        || w == 24 || w == 25 || w == 27 || w == 28) return 3UL;
    if (w == 4 || w == 5) return 4UL;
    if (w == 23) return 6UL;
    return 5UL;
}

static int cg_is_norm_(int which) { return which == 11 || which == 12 || which == 27 || which == 28; }

static int cg_has_params_(int which) {
    return which <= 3 || cg_is_norm_(which) || (which >= 17 && which <= 21);
}

// Weight rows / columns and bias size of a case with parameters.
static unsigned long cg_wrows_(int which) {
    if (which <= 1 || which == 18) return 3UL;
    if (which == 17) return 4UL;
    if (cg_is_norm_(which)) return 1UL;
    return 2UL;
}

static unsigned long cg_wcols_(int which) {
    if (which <= 1) return 18UL;
    if (which == 2) return 48UL;
    if (which == 17 || which == 19) return 9UL;
    if (which == 18) return 12UL;
    if (which == 20) return 18UL;
    if (cg_is_norm_(which)) return 2UL;
    return 27UL;
}

static unsigned long cg_nbias_(int which) {
    if (which == 17 || which == 20) return 4UL;
    if (which == 19 || cg_is_norm_(which)) return 2UL;
    return 3UL;
}

static &Tensor cg_forward_(int which, const &Tensor x, const &Tensor w, const &Tensor b, &Tensor rm, &Tensor rv) {
    unsigned long sd = cg_side_(which);
    if (which == 0) return tensor_conv2d(x, w, b, 2UL, sd, sd, 3UL, 3UL, 1UL, 1UL);
    if (which == 1) return tensor_conv2d(x, w, b, 2UL, sd, sd, 3UL, 3UL, 2UL, 1UL);
    if (which == 2) return tensor_conv_transpose2d(x, w, b, 2UL, sd, sd, 4UL, 4UL, 2UL, 1UL);
    if (which == 3) return tensor_conv_transpose2d(x, w, b, 2UL, sd, sd, 3UL, 3UL, 1UL, 0UL);
    if (which == 4) return x.maxpool2d(2UL, sd, sd, 2UL, 2UL, 0UL);
    if (which == 5) return x.avgpool2d(2UL, sd, sd, 2UL, 2UL, 0UL, 1);
    if (which == 6) return x.upsample_nearest(2UL, sd, sd, 2UL);
    if (which == 7) return x.maxpool2d(2UL, sd, sd, 3UL, 1UL, 0UL);   // overlapping windows
    if (which == 8) return x.maxpool2d(2UL, sd, sd, 3UL, 2UL, 1UL);   // padded
    if (which == 9) return x.avgpool2d(2UL, sd, sd, 3UL, 2UL, 1UL, 1);
    if (which == 10) return x.avgpool2d(2UL, sd, sd, 3UL, 2UL, 1UL, 0);
    if (which == 11 || which == 12)
        return tensor_batch_norm(x, w, b, rm, rv, 2UL, sd * sd, which == 11, (float)0.1, (float)0.00001);
    if (which == 13) return x.global_avgpool2d(2UL, sd, sd);
    if (which == 14) return x.global_maxpool2d(2UL, sd, sd);
    if (which == 15) return x.upsample_bilinear(2UL, sd, sd, 2UL);
    if (which == 16) return x.upsample_bilinear(2UL, sd, sd, 3UL);
    if (which == 17 || which == 19) {
        // dilated grouped (2 -> 4 channels, 2 groups) / depthwise (2 -> 2, 2 groups)
        struct Conv2dSpec c = Conv2dSpec::new(3UL, 1UL, which == 17 ? 2UL : 1UL);
        c.groups = 2UL;
        if (which == 17) { c.dilationH = 2UL; c.dilationW = 2UL; }
        return tensor_conv2d_ex(x, w, b, 2UL, sd, sd, &c);
    }
    if (which == 18) {
        struct Conv2dSpec c = Conv2dSpec::new(2UL, 2UL, 1UL);    // 2x3 kernel, strides 2 / 1, padding 1 / 0
        c.kW = 3UL; c.strideW = 1UL; c.padW = 0UL;
        return tensor_conv2d_ex(x, w, b, 2UL, sd, sd, &c);
    }
    if (which == 20) {
        struct Conv2dSpec c = Conv2dSpec::new(3UL, 2UL, 1UL);    // grouped, output padding 1
        c.groups = 2UL; c.outPadH = 1UL; c.outPadW = 1UL;
        return tensor_conv_transpose2d_ex(x, w, b, 2UL, sd, sd, &c);
    }
    if (which == 21) {
        struct Conv2dSpec c = Conv2dSpec::new(3UL, 1UL, 0UL);    // dilated
        c.dilationH = 2UL; c.dilationW = 2UL;
        return tensor_conv_transpose2d_ex(x, w, b, 2UL, sd, sd, &c);
    }
    if (which == 22) {
        struct Pool2dSpec p = Pool2dSpec::new(2UL, 2UL, 0UL);    // 2x3 window, strides 2 / 1, ceil mode
        p.kW = 3UL; p.strideW = 1UL; p.ceilMode = 1;
        return x.maxpool2d_ex(2UL, sd, sd, &p);
    }
    if (which == 23) {
        struct Pool2dSpec p = Pool2dSpec::new(3UL, 2UL, 1UL);
        p.ceilMode = 1;
        return x.avgpool2d_ex(2UL, sd, sd, &p);
    }
    if (which == 24) return x.resize2d(2UL, sd, sd, 5UL, 4UL, RESIZE_BILINEAR, 1);
    if (which == 25) return x.resize2d(2UL, sd, sd, 7UL, 5UL, RESIZE_BICUBIC, 0);
    if (which == 26) return x.resize2d(2UL, sd, sd, 3UL, 2UL, RESIZE_NEAREST, 0);
    if (which == 27 || which == 28)
        return tensor_group_norm(x, w, b, 2UL, sd * sd, which == 27 ? 1UL : 2UL, (float)0.00001);
    if (which == 29) return tensor_spectral_norm(x, rm, 1UL, 0);      // x as a [2, 50] weight
    struct Rng mask = Rng::new(99UL);                                  // the same mask every pass
    if (which == 30) return x.dropout((float)0.3, 1, &mask);
    return x.dropout2d(2UL, sd * sd, (float)0.5, 1, &mask);
}

static &Tensor cg_loss_(int which, const &Tensor x, const &Tensor w, const &Tensor b, &Tensor rm, &Tensor rv,
                        const &Tensor aux) {
    return tensor_mul(cg_forward_(which, x, w, b, rm, rv), aux).sum();
}

// Worst relative error of Tensor::backward's gradient for t against central
// differences of the loss.
static double cg_compare_(int which, &Tensor t, &Tensor x, &Tensor w, &Tensor b, &Tensor rm, &Tensor rv,
                          const &Tensor aux, &TensorArena stepArena) {
    double worst = 0.0;
    unsafe {
        unsigned long i = 0UL;
        while (i < t->size) {
            float keep = t->data[i];
            float h = (float)0.004;
            t->data[i] = keep + h;
            float lp = cg_loss_(which, x, w, b, rm, rv, aux)->data[0];
            t->data[i] = keep - h;
            float lm = cg_loss_(which, x, w, b, rm, rv, aux)->data[0];
            t->data[i] = keep;
            stepArena.reset();
            double num = ((double)lp - (double)lm) / (2.0 * (double)h);
            double e = num - (double)t->grad[i];
            if (e < 0.0) e = -e;
            double sc = num < 0.0 ? -num : num;
            e = e / (sc > 1.0 ? sc : 1.0);
            if (e > worst) worst = e;
            i = i + 1UL;
        }
    }
    return worst;
}

static double conv_grad_check_(int which) {
    struct Rng rng = Rng::new(31UL + (unsigned long)which);
    unsigned long sd = cg_side_(which);
    unsigned long n = 2UL * 2UL * sd * sd;
    &Tensor x = Tensor::new_2d(2UL, n / 2UL, 1);
    // distinct values 2/n apart, so no max-pool window has a near tie
    unsafe {
        unsigned long i = 0UL;
        while (i < n) { x->data[i] = (float)((i * 37UL) % n) * ((float)2.0 / (float)n) - (float)1.0; i = i + 1UL; }
    }
    // conv [outC, (inC/groups)*k*k], deconv [inC, (outC/groups)*k*k], norms gamma [2]
    &Tensor w = Tensor::new_2d(cg_wrows_(which), cg_wcols_(which), cg_has_params_(which));
    &Tensor b = Tensor::new_1d(cg_nbias_(which), cg_has_params_(which));
    &Tensor rm = Tensor::new_1d(2UL, 0);
    &Tensor rv = Tensor::new_1d(2UL, 0);
    rng.fill_uniform(w, (float)-0.5, (float)0.5);
    rng.fill_uniform(b, (float)-0.5, (float)0.5);
    rng.fill_uniform(rm, (float)-0.3, (float)0.3);
    rng.fill_uniform(rv, (float)0.5, (float)1.5);
    struct TensorArena stepArena = TensorArena::new(0UL);
    stepArena.begin();
    &Tensor probe = cg_forward_(which, x, w, b, rm, rv);
    tensor_arena_end();
    &Tensor aux = probe.zeros_like();
    rng.fill_uniform(aux, (float)-1.0, (float)1.0);
    stepArena.reset();
    stepArena.begin();
    cg_loss_(which, x, w, b, rm, rv, aux).backward();
    stepArena.reset();
    double worst = cg_compare_(which, x, x, w, b, rm, rv, aux, &stepArena);
    if (cg_has_params_(which)) {
        double ew = cg_compare_(which, w, x, w, b, rm, rv, aux, &stepArena);
        double eb = cg_compare_(which, b, x, w, b, rm, rv, aux, &stepArena);
        if (ew > worst) worst = ew;
        if (eb > worst) worst = eb;
    }
    tensor_arena_end();
    stepArena.free();
    tensor_drop(x); tensor_drop(w); tensor_drop(b); tensor_drop(rm); tensor_drop(rv); tensor_drop(aux);
    return worst;
}

void t_conv_grad_checks() {
    int w = 0;
    while (w < CG_CASES) {
        double e = conv_grad_check_(w);
        if (e >= 2e-3) { ASSERT_TRUE(w < 0); }   // report which op failed
        ASSERT_TRUE(e < 2e-3);
        w = w + 1;
    }
}

// Batch norm's running statistics move toward the batch's, and inference
// mode then normalizes with them.
void t_batch_norm_stats() {
    &Tensor x = Tensor::new_2d(4UL, 2UL, 0);
    &Tensor gamma = Tensor::new_1d(2UL, 0);
    &Tensor beta = Tensor::new_1d(2UL, 0);
    &Tensor rm = Tensor::new_1d(2UL, 0);
    &Tensor rv = Tensor::new_1d(2UL, 0);
    gamma.fill((float)1.0);
    rv.fill((float)1.0);
    unsafe {
        // channel 0: 1, 2, 3, 4 (mean 2.5, unbiased var 5/3); channel 1: all 7
        unsigned long i = 0UL;
        while (i < 4UL) { x->data[i * 2UL] = (float)(i + 1UL); x->data[i * 2UL + 1UL] = (float)7.0; i = i + 1UL; }
    }
    &Tensor y = tensor_batch_norm(x, gamma, beta, rm, rv, 2UL, 1UL, 1, (float)0.5, (float)0.00001);
    unsafe {
        ASSERT_TRUE(abs_f(rm->data[0] - (float)1.25) < (float)1e-5);
        ASSERT_TRUE(abs_f(rv->data[0] - (float)(0.5 + 0.5 * 5.0 / 3.0)) < (float)1e-5);
        ASSERT_TRUE(abs_f(y->data[0] + (float)1.3416355) < (float)1e-4);   // (1 - 2.5) / sqrt(1.25)
        ASSERT_TRUE(abs_f(y->data[1]) < (float)1e-3);                      // constant channel -> 0
    }
    &Tensor e = tensor_batch_norm(x, gamma, beta, rm, rv, 2UL, 1UL, 0, (float)0.5, (float)0.00001);
    unsafe {
        ASSERT_TRUE(abs_f(rm->data[0] - (float)1.25) < (float)1e-5);     // unchanged in inference
        float want = ((float)1.0 - (float)1.25) / sqrt_f(rv->data[0] + (float)0.00001);
        ASSERT_TRUE(abs_f(e->data[0] - want) < (float)1e-5);
    }
    tensor_drop(x); tensor_drop(gamma); tensor_drop(beta); tensor_drop(rm); tensor_drop(rv);
    tensor_drop(y); tensor_drop(e);
}

// Forward and backward of a batch large enough to run across the thread
// pool, against the same computation on one thread.
static void conv_run_(int deconv, const &Tensor x, const &Tensor w, const &Tensor b, const &Tensor aux,
                      &Tensor outCopy, &Tensor dx, &Tensor dw, &Tensor db) {
    x.zero_grad(); w.zero_grad(); b.zero_grad();
    &Tensor y = deconv ? tensor_conv_transpose2d(x, w, b, 4UL, 16UL, 16UL, 4UL, 4UL, 2UL, 1UL)
                       : tensor_conv2d(x, w, b, 4UL, 16UL, 16UL, 3UL, 3UL, 1UL, 1UL);
    tensor_mul(y, aux).sum().backward();
    unsafe {
        __tensor_ew(EW_COPY, (const float*)y->data, (const float*)0, (float)0.0, (float*)outCopy->data, y->size);
        __tensor_ew(EW_COPY, (const float*)x->grad, (const float*)0, (float)0.0, (float*)dx->data, x->size);
        __tensor_ew(EW_COPY, (const float*)w->grad, (const float*)0, (float)0.0, (float*)dw->data, w->size);
        __tensor_ew(EW_COPY, (const float*)b->grad, (const float*)0, (float)0.0, (float*)db->data, b->size);
    }
}

static float max_rel_diff_(const &Tensor a, const &Tensor b) {
    float worst = (float)0.0;
    unsafe {
        unsigned long i = 0UL;
        while (i < a->size) {
            float d = abs_f(a->data[i] - b->data[i]);
            float s = abs_f(a->data[i]);
            d = d / (s > (float)1.0 ? s : (float)1.0);
            if (d > worst) worst = d;
            i = i + 1UL;
        }
    }
    return worst;
}

void t_conv_parallel_batch() {
    struct Rng rng = Rng::new(77UL);
    unsigned long threads = parallel_threads();
    int deconv = 0;
    while (deconv < 2) {
        // conv 4 -> 8 channels on 16x16, deconv 4 -> 8 channels 16x16 -> 32x32
        unsigned long outSize = deconv ? 8UL * 32UL * 32UL : 8UL * 16UL * 16UL;
        &Tensor x = Tensor::new_2d(8UL, 4UL * 256UL, 1);
        &Tensor w = deconv ? Tensor::new_2d(4UL, 8UL * 16UL, 1) : Tensor::new_2d(8UL, 36UL, 1);
        &Tensor b = Tensor::new_1d(8UL, 1);
        &Tensor aux = Tensor::new_2d(8UL, outSize, 0);
        rng.fill_uniform(x, (float)-1.0, (float)1.0);
        rng.fill_uniform(w, (float)-0.3, (float)0.3);
        rng.fill_uniform(b, (float)-0.3, (float)0.3);
        rng.fill_uniform(aux, (float)-1.0, (float)1.0);
        &Tensor y1 = Tensor::new_2d(8UL, outSize, 0);
        &Tensor y2 = Tensor::new_2d(8UL, outSize, 0);
        &Tensor dx1 = x.zeros_like(); &Tensor dx2 = x.zeros_like();
        &Tensor dw1 = w.zeros_like(); &Tensor dw2 = w.zeros_like();
        &Tensor db1 = b.zeros_like(); &Tensor db2 = b.zeros_like();
        struct TensorArena stepArena = TensorArena::new(0UL);
        stepArena.begin();
        conv_run_(deconv, x, w, b, aux, y1, dx1, dw1, db1);     // threaded when the pool has threads
        stepArena.reset();
        parallel_set_threads(1UL);
        conv_run_(deconv, x, w, b, aux, y2, dx2, dw2, db2);
        parallel_set_threads(threads);
        tensor_arena_end();
        stepArena.free();
        ASSERT_TRUE(max_rel_diff_(y1, y2) < (float)1e-4);
        ASSERT_TRUE(max_rel_diff_(dx1, dx2) < (float)1e-4);
        ASSERT_TRUE(max_rel_diff_(dw1, dw2) < (float)1e-4);
        ASSERT_TRUE(max_rel_diff_(db1, db2) < (float)1e-4);
        tensor_drop(x); tensor_drop(w); tensor_drop(b); tensor_drop(aux);
        tensor_drop(y1); tensor_drop(y2); tensor_drop(dx1); tensor_drop(dx2);
        tensor_drop(dw1); tensor_drop(dw2); tensor_drop(db1); tensor_drop(db2);
        deconv = deconv + 1;
    }
}

// Batch norm, pooling and upsampling on batches large enough to split across
// the thread pool, against the same chain on one thread.
static void light_chain_(&Tensor x, &Tensor gamma, &Tensor beta, &Tensor rm, &Tensor rv,
                         const &Tensor a1, const &Tensor a2, &Tensor snap) {
    x.zero_grad(); gamma.zero_grad(); beta.zero_grad();
    rm.fill((float)0.0);
    rv.fill((float)1.0);
    &Tensor y = tensor_batch_norm(x, gamma, beta, rm, rv, 8UL, 1024UL, 1, (float)0.1, (float)0.00001);
    &Tensor mp = y.maxpool2d(8UL, 32UL, 32UL, 3UL, 2UL, 1UL);          // 8 x 16 x 16
    &Tensor ap = y.avgpool2d(8UL, 32UL, 32UL, 3UL, 2UL, 1UL, 0);
    &Tensor up = mp.upsample_nearest(8UL, 16UL, 16UL, 2UL);             // 8 x 32 x 32
    &Tensor loss = tensor_add(tensor_mul(up, a1).sum(), tensor_mul(ap, a2).sum());
    loss.backward();
    // snapshot: loss, dx, dgamma, dbeta, running mean, running variance
    unsafe {
        unsigned long n = x->size;
        snap->data[0] = loss->data[0];
        __tensor_ew(EW_COPY, (const float*)x->grad, (const float*)0, (float)0.0, (float*)&snap->data[1], n);
        unsigned long c = 0UL;
        while (c < 8UL) {
            snap->data[1UL + n + c] = gamma->grad[c];
            snap->data[9UL + n + c] = beta->grad[c];
            snap->data[17UL + n + c] = rm->data[c];
            snap->data[25UL + n + c] = rv->data[c];
            c = c + 1UL;
        }
    }
}

void t_light_ops_parallel() {
    struct Rng rng = Rng::new(41UL);
    unsigned long threads = parallel_threads();
    &Tensor x = Tensor::new_2d(16UL, 8UL * 1024UL, 1);
    &Tensor gamma = Tensor::new_1d(8UL, 1);
    &Tensor beta = Tensor::new_1d(8UL, 1);
    &Tensor rm = Tensor::new_1d(8UL, 0);
    &Tensor rv = Tensor::new_1d(8UL, 0);
    &Tensor a1 = Tensor::new_2d(16UL, 8UL * 1024UL, 0);
    &Tensor a2 = Tensor::new_2d(16UL, 8UL * 256UL, 0);
    rng.fill_uniform(x, (float)-2.0, (float)2.0);
    rng.fill_uniform(gamma, (float)0.5, (float)1.5);
    rng.fill_uniform(beta, (float)-0.5, (float)0.5);
    rng.fill_uniform(a1, (float)-1.0, (float)1.0);
    rng.fill_uniform(a2, (float)-1.0, (float)1.0);
    unsigned long snapLen = 1UL + 16UL * 8UL * 1024UL + 32UL;
    &Tensor s1 = Tensor::new_1d(snapLen, 0);
    &Tensor s2 = Tensor::new_1d(snapLen, 0);
    struct TensorArena stepArena = TensorArena::new(0UL);
    stepArena.begin();
    light_chain_(x, gamma, beta, rm, rv, a1, a2, s1);    // threaded when the pool has threads
    stepArena.reset();
    parallel_set_threads(1UL);
    light_chain_(x, gamma, beta, rm, rv, a1, a2, s2);
    parallel_set_threads(threads);
    tensor_arena_end();
    stepArena.free();
    // per-channel sums are computed the same way on either path: bitwise equal
    ASSERT_TRUE(max_rel_diff_(s1, s2) == (float)0.0);
    tensor_drop(x); tensor_drop(gamma); tensor_drop(beta); tensor_drop(rm); tensor_drop(rv);
    tensor_drop(a1); tensor_drop(a2); tensor_drop(s1); tensor_drop(s2);
}

// ── Specs, resizing values, Winograd path ────────────────────────────────────
void t_conv_specs() {
    // PyTorch's sizes
    ASSERT_EQ(pool_out_side(6UL, 3UL, 2UL, 0UL, 1), 3UL);
    ASSERT_EQ(pool_out_side(6UL, 3UL, 2UL, 1UL, 1), 4UL);
    ASSERT_EQ(pool_out_side(5UL, 2UL, 2UL, 0UL, 1), 3UL);
    ASSERT_EQ(pool_out_side(5UL, 2UL, 2UL, 0UL, 0), 2UL);
    ASSERT_EQ(pool_out_side(4UL, 2UL, 2UL, 1UL, 1), 3UL);     // last window would start in the right padding
    ASSERT_EQ(conv_out_side_ex(10UL, 3UL, 1UL, 0UL, 2UL), 6UL);
    ASSERT_EQ(conv_transpose_out_side_ex(4UL, 3UL, 2UL, 1UL, 1UL, 1UL), 8UL);
    ASSERT_EQ(conv_transpose_out_side_ex(4UL, 3UL, 1UL, 0UL, 2UL, 0UL), 8UL);

    // nearest 4 -> 2 picks 0 and 2; bilinear with aligned corners 2 -> 3 is [1, 1.5, 2];
    // bicubic keeps a constant image constant
    &Tensor a = Tensor::new_1d(4UL, 0);
    unsafe { a->data[0] = (float)1.0; a->data[1] = (float)2.0; a->data[2] = (float)3.0; a->data[3] = (float)4.0; }
    &Tensor nn = a.resize2d(1UL, 1UL, 4UL, 1UL, 2UL, RESIZE_NEAREST, 0);
    &Tensor two = Tensor::new_1d(2UL, 0);
    unsafe { two->data[0] = (float)1.0; two->data[1] = (float)2.0; }
    &Tensor al = two.resize2d(1UL, 1UL, 2UL, 1UL, 3UL, RESIZE_BILINEAR, 1);
    &Tensor flat = Tensor::new_1d(9UL, 0);
    flat.fill((float)0.7);
    &Tensor cub = flat.resize2d(1UL, 3UL, 3UL, 7UL, 5UL, RESIZE_BICUBIC, 0);
    unsafe {
        ASSERT_TRUE(nn->data[0] == (float)1.0 && nn->data[1] == (float)3.0);
        ASSERT_TRUE(al->data[0] == (float)1.0 && abs_f(al->data[1] - (float)1.5) < (float)1e-6 && al->data[2] == (float)2.0);
        unsigned long i = 0UL;
        while (i < 35UL) { ASSERT_TRUE(abs_f(cub->data[i] - (float)0.7) < (float)1e-5); i = i + 1UL; }
    }
    tensor_drop(a); tensor_drop(nn); tensor_drop(two); tensor_drop(al); tensor_drop(flat); tensor_drop(cub);

    // a 3x3 stride-1 layer with 8+ channels goes through cnn.h's Winograd
    // kernel: same result as the direct sum
    struct Rng rng = Rng::new(12UL);
    &Tensor x = Tensor::new_2d(2UL, 8UL * 100UL, 0);
    &Tensor w = Tensor::new_2d(8UL, 72UL, 0);
    &Tensor b = Tensor::new_1d(8UL, 0);
    rng.fill_uniform(x, (float)-1.0, (float)1.0);
    rng.fill_uniform(w, (float)-0.3, (float)0.3);
    rng.fill_uniform(b, (float)-0.3, (float)0.3);
    &Tensor y = tensor_conv2d(x, w, b, 8UL, 10UL, 10UL, 3UL, 3UL, 1UL, 1UL);
    unsafe {
        float worst = (float)0.0;
        unsigned long n = 0UL;
        while (n < 2UL) {
            unsigned long oc = 0UL;
            while (oc < 8UL) {
                unsigned long p = 0UL;
                while (p < 100UL) {
                    long oh = (long)(p / 10UL);
                    long ow = (long)(p % 10UL);
                    double acc = (double)b->data[oc];
                    unsigned long ic = 0UL;
                    while (ic < 8UL) {
                        long kh = 0L;
                        while (kh < 3L) {
                            long kw = 0L;
                            while (kw < 3L) {
                                long ih = oh + kh - 1L;
                                long iw = ow + kw - 1L;
                                if (ih >= 0L && ih < 10L && iw >= 0L && iw < 10L)
                                    acc = acc + (double)w->data[oc * 72UL + ic * 9UL + (unsigned long)(kh * 3L + kw)]
                                              * (double)x->data[n * 800UL + ic * 100UL + (unsigned long)(ih * 10L + iw)];
                                kw = kw + 1L;
                            }
                            kh = kh + 1L;
                        }
                        ic = ic + 1UL;
                    }
                    float d = abs_f(y->data[n * 800UL + oc * 100UL + p] - (float)acc);
                    if (d > worst) worst = d;
                    p = p + 1UL;
                }
                oc = oc + 1UL;
            }
            n = n + 1UL;
        }
        ASSERT_TRUE(worst < (float)1e-4);
    }
    tensor_drop(x); tensor_drop(w); tensor_drop(b); tensor_drop(y);
}

// ── ConvNet with skips: input gradient through add (with projection) and concat ──
static struct ConvNet skip_net_(&Rng rng) {
    struct ConvNet n = ConvNet::new(2UL, 6UL, 6UL);
    unsigned long s0 = n.save();
    n.conv(4UL, 3UL, 1UL, 1UL, ACT_TANH, rng);
    n.conv(4UL, 3UL, 2UL, 1UL, ACT_NONE, rng);          // 4 x 3 x 3
    n.add_skip(s0, ACT_TANH, rng);                      // 1x1 stride-2 projection of 2 x 6 x 6
    n.deconv(4UL, 4UL, 2UL, 1UL, ACT_NONE, rng);        // 4 x 6 x 6
    n.concat_skip(s0);                                  // 6 x 6 x 6
    n.groupnorm(2UL, ACT_TANH);
    n.dense(3UL, ACT_NONE, rng);
    return n;
}

static float skip_loss_(const &ConvNet n, const &Tensor x, const &Tensor aux) {
    float v;
    unsafe { v = tensor_mul(n.forward(x), aux).sum()->data[0]; }
    return v;
}

void t_convnet_skips() {
    struct Rng rng = Rng::new(23UL);
    struct ConvNet n = skip_net_(&rng);
    ASSERT_EQ(n.slots, 1UL);
    ASSERT_EQ(n.out_dim(), 3UL);
    // conv 2->4 (72 + 4), conv 4->4 (144 + 4), projection 2->4 (8 + 4), deconv 4->4 (256 + 4),
    // group norm (6 + 6), dense 216 -> 3 (648 + 3)
    ASSERT_EQ(model_param_count(&n), 76UL + 148UL + 12UL + 260UL + 12UL + 651UL);
    &Tensor x = Tensor::new_2d(2UL, 72UL, 1);
    &Tensor aux = Tensor::new_2d(2UL, 3UL, 0);
    rng.fill_uniform(x, (float)-1.0, (float)1.0);
    rng.fill_uniform(aux, (float)-1.0, (float)1.0);
    struct TensorArena stepArena = TensorArena::new(0UL);
    stepArena.begin();
    (tensor_mul(n.forward(x), aux).sum()).backward();
    stepArena.reset();
    double worst = 0.0;
    unsafe {
        unsigned long i = 0UL;
        while (i < 144UL) {
            float keep = x->data[i];
            float h = (float)0.004;
            x->data[i] = keep + h;
            float lp = skip_loss_(&n, x, aux);
            x->data[i] = keep - h;
            float lm = skip_loss_(&n, x, aux);
            x->data[i] = keep;
            stepArena.reset();
            double num = ((double)lp - (double)lm) / (2.0 * (double)h);
            double e = num - (double)x->grad[i];
            if (e < 0.0) e = -e;
            double sc = num < 0.0 ? -num : num;
            e = e / (sc > 1.0 ? sc : 1.0);
            if (e > worst) worst = e;
            i = i + 1UL;
        }
    }
    tensor_arena_end();
    stepArena.free();
    ASSERT_TRUE(worst < 2e-3);
    tensor_drop(x); tensor_drop(aux);
    n.free();

    // ResNet basic block: 4 x 8 x 8 -> 8 x 4 x 4 with a projection shortcut
    struct ConvNet r = ConvNet::new(4UL, 8UL, 8UL);
    r.residual_block(8UL, 2UL, 1, ACT_RELU, &rng);
    ASSERT_EQ(r.outC, 8UL);
    ASSERT_EQ(r.outH, 4UL);
    // conv 4->8 (288 + 8), bn (16), conv 8->8 (576 + 8), bn (16), projection 4->8 (32 + 8)
    ASSERT_EQ(model_param_count(&r), 296UL + 16UL + 584UL + 16UL + 40UL);
    r.residual_block(8UL, 1UL, 0, ACT_RELU, &rng);       // identity shortcut, no batch norm
    ASSERT_EQ(model_param_count(&r), 952UL + 584UL + 584UL);
    r.free();
}

// ── Batch-norm folding, save / load ──────────────────────────────────────────
static struct ConvNet fold_net_(&Rng rng) {
    struct ConvNet n = ConvNet::new(3UL, 8UL, 8UL);
    n.conv(6UL, 3UL, 1UL, 1UL, ACT_NONE, rng);
    n.batchnorm(ACT_RELU);
    struct Conv2dSpec d = Conv2dSpec::new(2UL, 2UL, 0UL);
    d.groups = 2UL;
    n.deconv_ex(4UL, &d, ACT_NONE, rng);                // grouped transposed conv, 4 x 16 x 16
    n.batchnorm(ACT_NONE);
    n.global_avgpool();
    n.dense(5UL, ACT_NONE, rng);
    n.batchnorm(ACT_NONE);
    return n;
}

static void randomize_list_(&Vec<&Tensor> v, &Rng rng, float lo, float hi) {
    unsigned long i = 0UL;
    while (i < v.length()) { rng.fill_uniform(*v.at(i), lo, hi); i = i + 1UL; }
}

void t_convnet_fold_and_files() {
    struct Rng rng = Rng::new(31UL);
    struct ConvNet n = fold_net_(&rng);
    randomize_list_(&n.buffers, &rng, (float)0.5, (float)1.5);     // running means and variances
    unsigned long i = 0UL;
    while (i < n.params.length()) {
        // gammas / betas of the norms (the 1D parameters after the first two)
        &Tensor p = *n.params.at(i);
        unsafe { if (p->ndim == 1UL && i % 4UL == 2UL) rng.fill_uniform(p, (float)0.5, (float)1.5); }
        i = i + 1UL;
    }
    n.set_training(0);
    struct GradGuard noGrad = Autograd::get().no_grad();
    &Tensor x = Tensor::new_2d(3UL, 192UL, 0);
    rng.fill_uniform(x, (float)-1.0, (float)1.0);
    &Tensor before = n.forward(x);

    // save, then load into a differently initialized net of the same layers
    ASSERT_TRUE(model_save_file(&n, "build/test_ml_train_convnet.bin"));
    struct Rng other = Rng::new(77UL);
    struct ConvNet m = fold_net_(&other);
    m.set_training(0);
    &Tensor m0 = m.forward(x);
    ASSERT_TRUE(max_rel_diff_(before, m0) > (float)1e-3);
    ASSERT_TRUE(model_load_file(&m, "build/test_ml_train_convnet.bin"));
    &Tensor m1 = m.forward(x);
    ASSERT_TRUE(max_rel_diff_(before, m1) == (float)0.0);
    struct ConvNet wrong = ConvNet::new(3UL, 8UL, 8UL);
    wrong.conv(5UL, 3UL, 1UL, 1UL, ACT_NONE, &other);
    ASSERT_TRUE(!model_load_file(&wrong, "build/test_ml_train_convnet.bin"));
    ASSERT_TRUE(!model_load_file(&wrong, "build/no_such_file_test_ml_train.bin"));
    wrong.free();

    // folding: three batch norms disappear, inference output unchanged
    ASSERT_EQ(n.fold_batchnorm(), 3UL);
    &Tensor after = n.forward(x);
    ASSERT_TRUE(max_rel_diff_(before, after) < (float)1e-4);
    ASSERT_EQ(n.fold_batchnorm(), 0UL);

    // an Mlp round trip
    struct Mlp a = Mlp::new(4UL, 8UL, 1UL, 2UL, ACT_RELU, ACT_NONE, &rng);
    struct Mlp c = Mlp::new(4UL, 8UL, 1UL, 2UL, ACT_RELU, ACT_NONE, &other);
    &Tensor mx = Tensor::new_2d(5UL, 4UL, 0);
    rng.fill_uniform(mx, (float)-1.0, (float)1.0);
    ASSERT_TRUE(model_save_file(&a, "build/test_ml_train_mlp.bin"));
    ASSERT_TRUE(model_load_file(&c, "build/test_ml_train_mlp.bin"));
    &Tensor ya = a.forward(mx);
    &Tensor yc = c.forward(mx);
    ASSERT_TRUE(max_rel_diff_(ya, yc) == (float)0.0);
    noGrad.restore();
    tensor_drop(x); tensor_drop(before); tensor_drop(after); tensor_drop(m0); tensor_drop(m1);
    tensor_drop(mx); tensor_drop(ya); tensor_drop(yc);
    a.free(); c.free();
    m.free();
    n.free();
}

// ── Dropout, spectral norm, group norm ───────────────────────────────────────
void t_dropout_and_norms() {
    struct Rng rng = Rng::new(55UL);
    &Tensor ones = Tensor::new_2d(64UL, 256UL, 0);
    ones.fill((float)1.0);
    &Tensor d = ones.dropout((float)0.25, 1, &rng);
    unsafe {
        unsigned long zeros = 0UL;
        int ok = 1;
        unsigned long i = 0UL;
        while (i < d->size) {
            if (d->data[i] == (float)0.0) zeros = zeros + 1UL;
            else if (abs_f(d->data[i] - (float)(1.0 / 0.75)) > (float)1e-6) ok = 0;
            i = i + 1UL;
        }
        ASSERT_TRUE(ok);
        float frac = (float)zeros / (float)d->size;
        ASSERT_TRUE(frac > (float)0.23 && frac < (float)0.27);
    }
    &Tensor e = ones.dropout((float)0.25, 0, &rng);           // inference: unchanged
    unsafe { ASSERT_TRUE(e->data[0] == (float)1.0 && e->size == ones->size); }
    &Tensor d2 = ones.dropout2d(16UL, 16UL, (float)0.5, 1, &rng);
    unsafe {
        int planes = 1;
        unsigned long p = 0UL;
        while (p < 64UL * 16UL) {
            unsigned long q = 1UL;
            while (q < 16UL) { if (d2->data[p * 16UL + q] != d2->data[p * 16UL]) planes = 0; q = q + 1UL; }
            p = p + 1UL;
        }
        ASSERT_TRUE(planes);
    }

    // spectral norm: after enough power iterations the result's largest
    // singular value is 1
    &Tensor w = Tensor::new_2d(6UL, 10UL, 0);
    &Tensor u = Tensor::new_1d(6UL, 0);
    &Tensor u2 = Tensor::new_1d(6UL, 0);
    rng.fill_uniform(w, (float)-2.0, (float)2.0);
    rng.fill_normal(u, (float)0.0, (float)1.0);
    rng.fill_normal(u2, (float)0.0, (float)1.0);
    &Tensor wsn = tensor_spectral_norm(w, u, 100UL, 1);
    &Tensor again = tensor_spectral_norm(wsn, u2, 100UL, 1);         // wsn / sigma(wsn)
    unsafe {
        float sigma = wsn->data[0] / again->data[0];
        ASSERT_TRUE(abs_f(sigma - (float)1.0) < (float)1e-3);
    }

    // group norm doesn't depend on the rest of the batch
    &Tensor x = Tensor::new_2d(3UL, 36UL, 0);
    &Tensor gamma = Tensor::new_1d(4UL, 0);
    &Tensor beta = Tensor::new_1d(4UL, 0);
    rng.fill_uniform(x, (float)-2.0, (float)2.0);
    rng.fill_uniform(gamma, (float)0.5, (float)1.5);
    rng.fill_uniform(beta, (float)-0.5, (float)0.5);
    &Tensor all = tensor_group_norm(x, gamma, beta, 4UL, 9UL, 2UL, (float)0.00001);
    &Tensor one = Tensor::new_1d(36UL, 0);
    unsafe { __tensor_ew(EW_COPY, (const float*)&x->data[36], (const float*)0, (float)0.0, (float*)one->data, 36UL); }
    &Tensor alone = tensor_group_norm(one, gamma, beta, 4UL, 9UL, 2UL, (float)0.00001);
    unsafe {
        float worst = (float)0.0;
        unsigned long i = 0UL;
        while (i < 36UL) { float df = abs_f(all->data[36UL + i] - alone->data[i]); if (df > worst) worst = df; i = i + 1UL; }
        ASSERT_TRUE(worst < (float)1e-6);
    }
    tensor_drop(ones); tensor_drop(d); tensor_drop(d2);
    tensor_drop(w); tensor_drop(u); tensor_drop(u2); tensor_drop(wsn); tensor_drop(again);
    tensor_drop(x); tensor_drop(gamma); tensor_drop(beta); tensor_drop(all); tensor_drop(one); tensor_drop(alone);
}

// ── WGAN-GP: the penalty's parameter gradient against finite differences ───
static float gp_value_(&Dcgan g, const &Tensor xh, &TensorArena stepArena) {
    float v = (float)0.0;
    dcgan_gradient_penalty_(g, xh, null, &v);
    stepArena.reset();
    return v;
}

void t_wgan_gp_gradient() {
    struct Rng rng = Rng::new(61UL);
    struct GanConfig cfg = GanConfig::default(4UL);
    cfg.loss = GAN_WGAN_GP;
    struct ConvNet gen = ConvNet::new(4UL, 1UL, 1UL);
    gen.dense(2UL, ACT_NONE, &rng);
    struct ConvNet disc = ConvNet::new(2UL, 1UL, 1UL);              // a smooth critic
    disc.dense(16UL, ACT_TANH, &rng);
    disc.dense(1UL, ACT_NONE, &rng);
    struct Dcgan g = Dcgan::new(gen, disc, &cfg);
    &Tensor xh = Tensor::new_2d(6UL, 2UL, 0);
    rng.fill_uniform(xh, (float)-1.5, (float)1.5);
    &Tensor w = *g.disc.params.at(0UL);                            // [2, 16]
    &Tensor ana = Tensor::new_1d(32UL, 0);                          // on the heap
    struct TensorArena stepArena = TensorArena::new(0UL);
    stepArena.begin();
    float v = (float)0.0;
    &Tensor term = dcgan_gradient_penalty_(&g, xh, null, &v);
    ASSERT_TRUE(v > (float)0.0);
    g.optD.zero_grad();
    term.backward();
    stepArena.reset();
    // snapshot: every penalty evaluation below runs its own input-gradient
    // backward pass, which adds into w's gradient
    unsafe { __tensor_ew(EW_COPY, (const float*)w->grad, (const float*)0, (float)0.0, (float*)ana->data, 32UL); }
    double worst = 0.0;
    double scale = 0.0;
    unsafe {
        unsigned long i = 0UL;
        while (i < w->size) {
            float keep = w->data[i];
            float h = (float)0.001;
            w->data[i] = keep + h;
            float pp = gp_value_(&g, xh, &stepArena);
            w->data[i] = keep - h;
            float pm = gp_value_(&g, xh, &stepArena);
            w->data[i] = keep;
            double num = ((double)pp - (double)pm) / (2.0 * (double)h);
            double e = num - (double)ana->data[i];
            if (e < 0.0) e = -e;
            if (e > worst) worst = e;
            double a = num < 0.0 ? -num : num;
            if (a > scale) scale = a;
            i = i + 1UL;
        }
    }
    tensor_arena_end();
    stepArena.free();
    ASSERT_TRUE(scale > 1e-2);
    ASSERT_TRUE(worst < 2e-2 * scale);
    tensor_drop(ana);
    tensor_drop(xh);
    g.free();
}

// ── End to end: a small residual CNN classifies 12x12 shapes ─────────────────
static &Tensor cnn_clf_loss_(&ConvNet m, const &Tensor x, const &Tensor y) {
    return tensor_cross_entropy_loss(m.forward(x), y);
}

static &Tensor cnn_clf_logits_(&ConvNet m, const &Tensor x) {
    int was = m.set_training(0);
    &Tensor l = m.forward(x);
    m.set_training(was);
    return l;
}

void t_cnn_classifier() {
    struct Rng rng = Rng::new(71UL);
    struct Dataset shapes = Dataset::shapes(768UL, 12UL, &rng);
    struct DatasetSplit sp = shapes.split((float)0.25, &rng);
    struct ConvNet net = ConvNet::new(1UL, 12UL, 12UL);
    net.conv(8UL, 3UL, 1UL, 1UL, ACT_NONE, &rng);
    net.batchnorm(ACT_RELU);
    net.residual_block(8UL, 1UL, 1, ACT_RELU, &rng);     // 8 x 12 x 12
    net.residual_block(16UL, 2UL, 1, ACT_RELU, &rng);    // 16 x 6 x 6
    net.dropout((float)0.1, 1, &rng);
    net.global_avgpool();
    net.dense(4UL, ACT_NONE, &rng);
    struct Optimizer opt = Optimizer::new(OPT_ADAM, (float)0.003);
    model_add_params(&net, &opt);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 6UL;
    struct TrainReport rep = trainer_fit(&net, cnn_clf_loss_, &opt, &sp.train, &tc);
    ASSERT_TRUE(rep.lastEpochLoss < rep.firstEpochLoss * (float)0.5);
    float acc = trainer_accuracy(&net, cnn_clf_logits_, &sp.test, 64UL);
    ASSERT_TRUE(acc > (float)0.9);
    opt.free();
    net.free();
    sp.train.free();
    sp.test.free();
    shapes.free();
}

// tensor_conv2d agrees with cnn.h's forward-only Conv2D on the same weights,
// padded pooling with cnn.h's, and the ConvNet builder tracks output geometry.
void t_conv_matches_cnn() {
    struct Conv2D layer = Conv2D::new(3UL, 4UL, 3UL, 3UL, 2UL, 1UL);
    struct FeatureMap fm = FeatureMap::new(3UL, 7UL, 7UL);
    &Tensor x = Tensor::new_2d(1UL, 147UL, 0);
    &Tensor w = Tensor::new_2d(4UL, 27UL, 0);
    &Tensor b = Tensor::new_1d(4UL, 0);
    unsafe {
        unsigned long i = 0UL;
        while (i < 147UL) { float v = sin_f((float)i * (float)0.37); fm.data[i] = v; x->data[i] = v; i = i + 1UL; }
        i = 0UL;
        while (i < 108UL) { w->data[i] = layer.weight[i]; i = i + 1UL; }
        i = 0UL;
        while (i < 4UL) { layer.bias[i] = (float)i * (float)0.1; b->data[i] = layer.bias[i]; i = i + 1UL; }
    }
    struct FeatureMap ref = layer.forward(&fm);
    &Tensor y = tensor_conv2d(x, w, b, 3UL, 7UL, 7UL, 3UL, 3UL, 2UL, 1UL);
    ASSERT_EQ(ref.height, 4UL);
    unsafe {
        ASSERT_EQ(y->size, 64UL);
        float worst = (float)0.0;
        unsigned long i = 0UL;
        while (i < 64UL) {
            float d = abs_f(y->data[i] - ref.data[i]);
            if (d > worst) worst = d;
            i = i + 1UL;
        }
        ASSERT_TRUE(worst < (float)1e-4);
    }
    // padded pooling, against cnn.h's *_forward_ex
    int kind = 0;
    while (kind < 3) {
        struct FeatureMap pr = kind == 0 ? fm.maxpool2d_ex(3UL, 2UL, 1UL)
                                         : fm.avgpool2d_ex(3UL, 2UL, 1UL, kind == 1);
        &Tensor pt = kind == 0 ? x.maxpool2d(3UL, 7UL, 7UL, 3UL, 2UL, 1UL)
                               : x.avgpool2d(3UL, 7UL, 7UL, 3UL, 2UL, 1UL, kind == 1);
        unsafe {
            ASSERT_EQ(pt->size, pr.channels * pr.height * pr.width);
            float worst = (float)0.0;
            unsigned long i = 0UL;
            while (i < pt->size) { float d = abs_f(pt->data[i] - pr.data[i]); if (d > worst) worst = d; i = i + 1UL; }
            ASSERT_TRUE(worst < (float)1e-5);
        }
        pr.free();
        tensor_drop(pt);
        kind = kind + 1;
    }
    // global pooling, against cnn.h's
    struct FeatureMap gref = fm.global_avgpool2d();
    struct FeatureMap mref = fm.global_maxpool2d();
    &Tensor ga = x.global_avgpool2d(3UL, 7UL, 7UL);
    &Tensor gm = x.global_maxpool2d(3UL, 7UL, 7UL);
    unsafe {
        ASSERT_EQ(ga->size, 3UL);
        unsigned long c = 0UL;
        while (c < 3UL) {
            ASSERT_TRUE(abs_f(ga->data[c] - gref.data[c]) < (float)1e-5);
            ASSERT_TRUE(gm->data[c] == mref.data[c]);
            c = c + 1UL;
        }
    }
    gref.free();
    mref.free();
    tensor_drop(ga);
    tensor_drop(gm);
    // bilinear 2x of [[1, 2], [3, 4]], as PyTorch's Upsample(align_corners=False) gives it
    &Tensor small = Tensor::new_1d(4UL, 0);
    unsafe { small->data[0] = (float)1.0; small->data[1] = (float)2.0; small->data[2] = (float)3.0; small->data[3] = (float)4.0; }
    &Tensor big = small.upsample_bilinear(1UL, 2UL, 2UL, 2UL);
    float want[16];
    unsafe {
        want[0] = (float)1.0;  want[1] = (float)1.25; want[2] = (float)1.75; want[3] = (float)2.0;
        want[4] = (float)1.5;  want[5] = (float)1.75; want[6] = (float)2.25; want[7] = (float)2.5;
        want[8] = (float)2.5;  want[9] = (float)2.75; want[10] = (float)3.25; want[11] = (float)3.5;
        want[12] = (float)3.0; want[13] = (float)3.25; want[14] = (float)3.75; want[15] = (float)4.0;
        ASSERT_EQ(big->size, 16UL);
        unsigned long i = 0UL;
        while (i < 16UL) { ASSERT_TRUE(abs_f(big->data[i] - want[i]) < (float)1e-6); i = i + 1UL; }
    }
    tensor_drop(small);
    tensor_drop(big);
    ref.free();
    fm.free();
    layer.free();
    tensor_drop(x); tensor_drop(w); tensor_drop(b); tensor_drop(y);

    struct Rng rng = Rng::new(3UL);
    struct ConvNet net = ConvNet::new(1UL, 16UL, 16UL);
    net.conv(8UL, 3UL, 1UL, 1UL, ACT_NONE, &rng);
    net.batchnorm(ACT_RELU);
    net.maxpool(2UL, 2UL, 0UL);
    ASSERT_EQ(net.outH, 8UL);
    net.deconv(4UL, 4UL, 2UL, 1UL, ACT_RELU, &rng);
    ASSERT_EQ(net.outH, 16UL);
    net.avgpool(3UL, 2UL, 1UL, 0);
    net.upsample(2UL);
    ASSERT_EQ(net.out_dim(), 4UL * 16UL * 16UL);
    net.dense(5UL, ACT_NONE, &rng);
    ASSERT_EQ(net.out_dim(), 5UL);
    ASSERT_EQ(model_param_count(&net), 8UL * 9UL + 8UL + 16UL + 8UL * 64UL + 4UL + 1024UL * 5UL + 5UL);
    ASSERT_EQ(net.buffers.length(), 2UL);
    &Tensor in = Tensor::new_2d(3UL, 256UL, 0);
    &Tensor out = net.forward(in);
    unsafe { ASSERT_EQ(out->shape[0], 3UL); ASSERT_EQ(out->shape[1], 5UL); }
    struct ConvNet head = ConvNet::new(4UL, 8UL, 8UL);
    head.upsample_bilinear(2UL);
    ASSERT_EQ(head.outH, 16UL);
    head.global_maxpool();
    ASSERT_EQ(head.out_dim(), 4UL);
    &Tensor hin = Tensor::new_2d(2UL, 256UL, 0);
    &Tensor hout = head.forward(hin);
    unsafe { ASSERT_EQ(hout->shape[1], 4UL); }
    struct ConvNet head2 = ConvNet::new(4UL, 8UL, 8UL);
    head2.global_avgpool();
    ASSERT_EQ(head2.out_dim(), 4UL);
    tensor_drop(hin);
    tensor_drop(hout);
    head.free();
    head2.free();
    struct ConvNet twin = net.clone();
    ASSERT_EQ(twin.buffers.length(), 2UL);
    twin.free();
    tensor_drop(in);
    net.free();
}

// Convolutional encoder (1 + extra) x 8 x 8 -> 16 x 2 x 2 with batch norm,
// and a decoder back to 1 x 8 x 8, for the shapes images.
static struct ConvNet shapes_encoder_(unsigned long extraChannels, &Rng rng) {
    struct ConvNet enc = ConvNet::new(1UL + extraChannels, 8UL, 8UL);
    enc.conv(8UL, 4UL, 2UL, 1UL, ACT_LEAKY, rng);    // 8 x 4 x 4
    enc.conv(16UL, 4UL, 2UL, 1UL, ACT_NONE, rng);    // 16 x 2 x 2
    enc.batchnorm(ACT_LEAKY);
    return enc;
}

static struct ConvNet shapes_decoder_(unsigned long inputs, &Rng rng) {
    struct ConvNet dec = ConvNet::new(inputs, 1UL, 1UL);
    dec.dense(64UL, ACT_RELU, rng);
    dec.reshape(16UL, 2UL, 2UL);
    dec.deconv(8UL, 4UL, 2UL, 1UL, ACT_NONE, rng);   // 8 x 4 x 4
    dec.batchnorm(ACT_RELU);
    dec.deconv(1UL, 4UL, 2UL, 1UL, ACT_NONE, rng);   // 1 x 8 x 8 logits
    return dec;
}

static float conv_recon_err_(&ConvAutoencoder ae, const &Dataset d) {
    struct GradGuard noGrad = Autograd::get().no_grad();
    &Tensor r = ae.reconstruct(d.x);
    float e;
    unsafe { e = tensor_mse_loss(r, d.x)->data[0]; }
    noGrad.restore();
    return e;
}

static &Tensor shapes8_logits_(&Mlp m, const &Tensor x) { return m.forward(x); }

// A classifier for the 8x8 shapes that judges conditional samples.
static struct Mlp shapes8_judge_(const &Dataset shapes, &Rng rng) {
    struct Mlp judge = Mlp::new(64UL, 64UL, 1UL, 4UL, ACT_RELU, ACT_NONE, rng);
    struct Optimizer jo = Optimizer::new(OPT_ADAM, (float)0.003);
    model_add_params(&judge, &jo);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 30UL;
    trainer_fit(&judge, clf_loss_, &jo, shapes, &tc);
    // 8x8 is coarse: the shapes are told apart about 85% of the time
    ASSERT_TRUE(trainer_accuracy(&judge, shapes8_logits_, shapes, 128UL) > (float)0.75);
    jo.free();
    return judge;
}

static &Tensor cycle_labels_(unsigned long n) {
    &Tensor labels = Tensor::new_1d(n, 0);
    unsafe { unsigned long i = 0UL; while (i < n) { labels->data[i] = (float)(i % 4UL); i = i + 1UL; } }
    return labels;
}

void t_conv_autoencoders() {
    struct Rng rng = Rng::new(5UL);
    struct Dataset shapes = Dataset::shapes(512UL, 8UL, &rng);
    struct TrainConfig tc = TrainConfig::default();
    tc.epochs = 25UL;

    // plain: the 16x2x2 feature map is the code
    struct ConvNet dec0 = ConvNet::new(16UL, 2UL, 2UL);
    dec0.deconv(8UL, 4UL, 2UL, 1UL, ACT_RELU, &rng);
    dec0.deconv(1UL, 4UL, 2UL, 1UL, ACT_NONE, &rng);
    struct ConvAutoencoder ae = ConvAutoencoder::new(shapes_encoder_(0UL, &rng), dec0, RECON_BCE);
    float before = conv_recon_err_(&ae, &shapes);
    struct Optimizer opt = Optimizer::new(OPT_ADAM, (float)0.003);
    model_add_params(&ae, &opt);
    model_fit(&ae, &opt, &shapes, &tc);
    float after = conv_recon_err_(&ae, &shapes);
    ASSERT_TRUE(after < before * (float)0.3);
    ASSERT_EQ(ae.enc.training, 1);
    opt.free();
    ae.free();

    // VAE, then conditional VAE judged by a classifier
    struct Mlp judge = shapes8_judge_(&shapes, &rng);
    &Tensor labels = cycle_labels_(64UL);
    int cond = 0;
    while (cond < 2) {
        struct VaeConfig cfg = VaeConfig::default(4UL);
        cfg.classes = cond ? 4UL : 0UL;
        struct ConvVae v = ConvVae::new(shapes_encoder_(cfg.classes, &rng),
                                        shapes_decoder_(4UL + cfg.classes, &rng), &cfg);
        struct Optimizer vo = Optimizer::new(OPT_ADAM, (float)0.003);
        model_add_params(&v, &vo);
        tc.epochs = cond ? 40UL : 25UL;
        struct TrainReport rep = model_fit(&v, &vo, &shapes, &tc);
        ASSERT_TRUE(rep.lastEpochLoss < rep.firstEpochLoss * (float)0.6);
        ASSERT_TRUE(v.lastKl > (float)0.1);
        struct GradGuard noGrad = Autograd::get().no_grad();
        &Tensor s = cond ? v.sample(64UL, (?&Tensor)labels) : v.sample(64UL, null);
        &Tensor r = v.reconstruct(shapes.x, shapes.y);
        unsafe {
            ASSERT_EQ(s->shape[0], 64UL);
            ASSERT_EQ(s->shape[1], 64UL);
            ASSERT_TRUE(s->data[0] > (float)0.0 && s->data[0] < (float)1.0);
            ASSERT_TRUE(tensor_mse_loss(r, shapes.x)->data[0] < (float)0.1);
        }
        if (cond) ASSERT_TRUE(tensor_accuracy(judge.forward(s), labels) > (float)0.7);
        noGrad.restore();
        tensor_drop(s);
        tensor_drop(r);
        vo.free();
        v.free();
        cond = cond + 1;
    }
    tensor_drop(labels);
    judge.free();
    shapes.free();
}

static struct Dcgan shapes_dcgan_(unsigned long classes, &Rng rng) {
    struct GanConfig cfg = GanConfig::default(8UL);
    cfg.lrG = (float)0.001; cfg.lrD = (float)0.001;
    cfg.emaDecay = (float)0.9;
    cfg.classes = classes;
    unsigned long w = classes > 0UL ? 2UL : 1UL;   // conditional: twice the channels
    struct ConvNet gen = ConvNet::new(8UL + classes, 1UL, 1UL);
    gen.deconv(16UL * w, 2UL, 1UL, 0UL, ACT_NONE, rng);    // 16w x 2 x 2
    gen.batchnorm(ACT_RELU);
    gen.deconv(8UL * w, 4UL, 2UL, 1UL, ACT_NONE, rng);     // 8w x 4 x 4
    gen.batchnorm(ACT_RELU);
    gen.deconv(1UL, 4UL, 2UL, 1UL, ACT_SIGMOID, rng);  // 1 x 8 x 8
    struct ConvNet disc = ConvNet::new(1UL + classes, 8UL, 8UL);
    disc.conv(8UL * w, 4UL, 2UL, 1UL, ACT_LEAKY, rng);
    disc.conv(16UL * w, 4UL, 2UL, 1UL, ACT_NONE, rng);
    disc.batchnorm(ACT_LEAKY);
    disc.conv(1UL, 2UL, 1UL, 0UL, ACT_NONE, rng);      // one score
    return Dcgan::new(gen, disc, &cfg);
}

static double tensor_mean_value_(const &Tensor t) {
    double acc = 0.0;
    unsafe {
        unsigned long i = 0UL;
        while (i < t->size) { acc = acc + (double)t->data[i]; i = i + 1UL; }
        return acc / (double)t->size;
    }
}

// A DCGAN on the shapes images moves its samples' mean intensity from the
// sigmoid's 0.5 to the data's; a conditional one draws the requested class
// (as judged by a classifier trained on the real images).
void t_dcgan() {
    struct Rng rng = Rng::new(9UL);
    struct Dataset shapes = Dataset::shapes(512UL, 8UL, &rng);
    struct Dcgan g = shapes_dcgan_(0UL, &rng);
    struct GanLosses l = g.train(&shapes, 15UL, 64UL);
    ASSERT_TRUE(l.d == l.d && l.g == l.g);   // finite, not NaN
    &Tensor s = g.sample(256UL, null);
    unsafe { ASSERT_EQ(s->shape[1], 64UL); }
    double d = tensor_mean_value_(s) - tensor_mean_value_(shapes.x);
    if (d < 0.0) d = -d;
    ASSERT_TRUE(d < 0.1);
    &Tensor sc = g.score(shapes.x, null);
    unsafe { ASSERT_EQ(sc->size, 512UL); }
    tensor_drop(sc);
    tensor_drop(s);
    g.free();

    struct Mlp judge = shapes8_judge_(&shapes, &rng);
    struct Dcgan cg = shapes_dcgan_(4UL, &rng);
    cg.train(&shapes, 60UL, 64UL);
    &Tensor labels = cycle_labels_(256UL);
    &Tensor cs = cg.sample(256UL, (?&Tensor)labels);
    struct GradGuard noGrad = Autograd::get().no_grad();
    float acc = tensor_accuracy(judge.forward(cs), labels);
    noGrad.restore();
    ASSERT_TRUE(acc > (float)0.55);   // chance is 0.25
    tensor_drop(cs);
    tensor_drop(labels);
    cg.free();
    judge.free();
    shapes.free();
}

int main() {
    struct TestSuite t = test_suite_init();
    unsafe {
        t.add("gradient checks (nn ops, losses)", t_grad_checks);
        t.add("gather rows scatter-add", t_gather_rows_grad);
        t.add("optimizers converge", t_optimizers);
        t.add("datasets + dataloader", t_datasets);
        t.add("csv / mnist loaders", t_dataset_csv);
        t.add("trainer: moons classifier", t_trainer_moons);
        t.add("trainer: sine regression", t_trainer_sine);
        t.add("autoencoders (plain / denoising / sparse)", t_autoencoders);
        t.add("VAE + conditional VAE", t_vae);
        t.add("VQ-VAE", t_vqvae);
        t.add("GAN objectives (incl. WGAN-GP)", t_gans);
        t.add("conditional GAN", t_cgan);
        t.add("gradient checks (conv / pool / resize / norms / dropout)", t_conv_grad_checks);
        t.add("batch norm running statistics", t_batch_norm_stats);
        t.add("conv batch across threads = one thread", t_conv_parallel_batch);
        t.add("pool / upsample / batch norm across threads = one thread", t_light_ops_parallel);
        t.add("conv / pool / resize specs, Winograd path", t_conv_specs);
        t.add("ConvNet skips and residual blocks", t_convnet_skips);
        t.add("ConvNet batch-norm folding, save / load", t_convnet_fold_and_files);
        t.add("dropout, spectral norm, group norm", t_dropout_and_norms);
        t.add("WGAN-GP penalty gradient", t_wgan_gp_gradient);
        t.add("residual CNN classifier", t_cnn_classifier);
        t.add("conv2d / pooling match cnn.h + ConvNet geometry", t_conv_matches_cnn);
        t.add("conv autoencoder + conv VAE / CVAE", t_conv_autoencoders);
        t.add("DCGAN + conditional DCGAN", t_dcgan);
        t.add("bandit agents", t_bandits);
        t.add("Q-learning / SARSA / expected SARSA on the cliff", t_tabular_cliff);
        t.add("gridworld walls and pits", t_gridworld_env);
        t.add("replay buffer", t_replay_buffer);
        t.add("DQN solves CartPole", t_dqn_cartpole);
        t.add("REINFORCE solves CartPole", t_reinforce_cartpole);
        t.add("A2C solves CartPole", t_a2c_cartpole);
        t.add("PPO solves CartPole", t_ppo_cartpole);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
