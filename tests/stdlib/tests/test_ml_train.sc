// std::ml training stack: nn.h ops / losses / optimizers (finite-difference
// gradient checks), datasets.h, trainer.h, and the generative and RL
// models built on them.
#include <std/test/test.h>
#include <std/ml/nn.h>
#include <std/ml/datasets.h>
#include <std/ml/trainer.h>
#include <std/io_file.h>
#include <std/ml/autoencoder.h>
#include <std/ml/gan.h>
#include <std/math.h>
#include <std/ml/rl.h>

extern void* malloc(unsigned long size);
extern void free(void* ptr);

static void tensor_drop(&Tensor t) {
    t->free();
    unsafe { free((void*)t); }
}

// ── Gradient checks ──────────────────────────────────────────────────────────
// loss(x) for a family of ops; compares tensor_backward's gradient for x
// with central differences.
static &Tensor gc_loss_(int which, &Tensor x, const &Tensor aux) {
    if (which == 0) return tensor_sum(tensor_mul(tensor_exp(x), aux));
    if (which == 1) return tensor_sum(tensor_mul(tensor_log(tensor_add_scalar(tensor_abs(x), (float)1.5)), aux));
    if (which == 2) return tensor_sum(tensor_mul(tensor_log_softmax(x), aux));
    if (which == 3) return tensor_cross_entropy_loss(x, tensor_argmax_rows(aux));
    if (which == 4) return tensor_bce_logits_loss(x, tensor_sigmoid(aux));
    if (which == 5) return tensor_huber_loss(x, aux, (float)0.3);
    if (which == 6) return tensor_sum(tensor_mul(tensor_sum_rows(x), tensor_sum_rows(aux)));
    if (which == 7) return tensor_sum(tensor_mul(tensor_concat_cols(x, aux), tensor_concat_cols(aux, x)));
    if (which == 8) return tensor_sum(tensor_mul(tensor_leaky_relu(x, (float)0.2), aux));
    if (which == 9) return tensor_sum(tensor_mul(tensor_minimum(x, aux), aux));
    if (which == 10) return tensor_sum(tensor_mul(tensor_clamp(x, (float)-0.4, (float)0.4), aux));
    return tensor_mse_loss(tensor_add_bias(x, tensor_sum_rows(tensor_detach(aux))), aux);
}

static double grad_check_(int which) {
    struct Rng rng = rng_new(7UL + (unsigned long)which);
    &Tensor x = tensor_new_2d(4UL, 4UL, 1);
    &Tensor aux = tensor_new_2d(4UL, 4UL, 0);
    rng_fill_uniform(&rng, x, (float)-1.0, (float)1.0);
    rng_fill_uniform(&rng, aux, (float)-1.0, (float)1.0);
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
    struct TensorArena stepArena = tensor_arena_new(0UL);
    tensor_arena_begin(&stepArena);
    &Tensor l = gc_loss_(which, x, aux);
    tensor_backward(l);
    tensor_arena_reset(&stepArena);
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
            tensor_arena_reset(&stepArena);
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
    tensor_arena_free(&stepArena);
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
    &Tensor table = tensor_new_2d(3UL, 2UL, 1);
    &Tensor idx = tensor_new_1d(4UL, 0);
    unsafe {
        unsigned long i = 0UL;
        while (i < 6UL) { table->data[i] = (float)i; i = i + 1UL; }
        idx->data[0] = (float)2.0; idx->data[1] = (float)0.0; idx->data[2] = (float)2.0; idx->data[3] = (float)1.0;
    }
    &Tensor g = tensor_gather_rows(table, idx);
    unsafe {
        ASSERT_TRUE(g->data[0] == (float)4.0 && g->data[1] == (float)5.0 && g->data[2] == (float)0.0);
    }
    tensor_backward(tensor_sum(g));
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
        &Tensor w = tensor_new_2d(1UL, 3UL, 1);
        &Tensor target = tensor_new_2d(1UL, 3UL, 0);
        unsafe { target->data[0] = (float)1.0; target->data[1] = (float)-2.0; target->data[2] = (float)0.5; }
        struct Optimizer opt = optimizer_new(kinds[k], kinds[k] == OPT_SGD ? (float)0.5 : (float)0.05);
        opt.weightDecay = (float)0.0;
        optimizer_add(&opt, w);
        struct TensorArena stepArena = tensor_arena_new(0UL);
        unsigned long s = 0UL;
        while (s < 400UL) {
            tensor_arena_begin(&stepArena);
            &Tensor l = tensor_mse_loss(w, target);
            trainer_step_(&opt, l, (float)0.0);
            tensor_arena_end();
            tensor_arena_reset(&stepArena);
            s = s + 1UL;
        }
        unsafe {
            float e = (float)0.0;
            unsigned long i = 0UL;
            while (i < 3UL) { float d = w->data[i] - target->data[i]; e = e + d * d; i = i + 1UL; }
            ASSERT_TRUE(e < (float)1e-3);
        }
        tensor_arena_free(&stepArena);
        optimizer_free(&opt);
        k = k + 1;
    }
}

// ── Datasets ─────────────────────────────────────────────────────────────────
void t_datasets() {
    struct Rng rng = rng_new(3UL);
    struct Dataset m = dataset_moons(200UL, (float)0.05, &rng);
    ASSERT_EQ(m.n, 200UL); ASSERT_EQ(m.features, 2UL); ASSERT_EQ(m.classes, 2UL);
    struct DatasetSplit sp = dataset_split(&m, (float)0.25, &rng);
    ASSERT_EQ(sp.test.n, 50UL); ASSERT_EQ(sp.train.n, 150UL);
    dataset_standardize_pair(&sp.train, &sp.test);
    unsafe {
        double s = 0.0;
        unsigned long i = 0UL;
        while (i < 150UL) { s = s + (double)sp.train.x->data[i * 2UL]; i = i + 1UL; }
        ASSERT_TRUE(s / 150.0 < 1e-4 && s / 150.0 > -1e-4);
    }
    // every row comes out once per epoch, in batches of 32 and a tail of 22
    struct DataLoader dl = dataloader_new(&sp.train, 32UL, 1, 9UL);
    ASSERT_EQ(dataloader_batches(&dl), 5UL);
    struct Batch b;
    unsigned long rows = 0UL;
    unsigned long batches = 0UL;
    while (dataloader_next(&dl, &b)) { rows = rows + b.rows; batches = batches + 1UL; }
    ASSERT_EQ(rows, 150UL); ASSERT_EQ(batches, 5UL);
    dataloader_free(&dl);

    struct Dataset sh = dataset_shapes(8UL, 12UL, &rng);
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
    struct Dataset bl = dataset_blobs(90UL, 3UL, 5UL, (float)0.5, &rng);
    ASSERT_EQ(bl.features, 5UL); ASSERT_EQ(bl.classes, 3UL);
    struct Dataset gr = dataset_gaussian_ring(64UL, 8UL, (float)2.0, (float)0.02, &rng);
    unsafe {
        float x = gr.x->data[0]; float y = gr.x->data[1];
        float r = x * x + y * y;
        ASSERT_TRUE(r > (float)3.5 && r < (float)4.5);
    }
    dataset_free(&gr); dataset_free(&bl); dataset_free(&sh);
    dataset_free(&sp.train); dataset_free(&sp.test); dataset_free(&m);
}

void t_dataset_csv() {
    match (file_open("build/test_ml_train.csv", "w")) {
        case some(f): {
            file_puts("a,b,label\n1.5, 2,0\n-1,3.25,2\n\n4,5,1\n", f);
            file_close(f);
        }
        case null: { ASSERT_TRUE(0); }
    }
    struct Dataset d = dataset_load_csv("build/test_ml_train.csv", 2L, 1, 1);
    ASSERT_EQ(d.n, 3UL); ASSERT_EQ(d.features, 2UL); ASSERT_EQ(d.classes, 3UL);
    unsafe {
        ASSERT_TRUE(d.x->data[0] == (float)1.5 && d.x->data[1] == (float)2.0);
        ASSERT_TRUE(d.x->data[3] == (float)3.25 && d.y->data[1] == (float)2.0);
    }
    dataset_free(&d);
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
    struct Dataset mn = dataset_load_mnist("build/test_ml_train.idx3", "build/test_ml_train.idx1", 0UL);
    ASSERT_EQ(mn.n, 2UL); ASSERT_EQ(mn.features, 4UL); ASSERT_EQ(mn.classes, 10UL);
    unsafe {
        ASSERT_TRUE(mn.x->data[0] == (float)1.0 && mn.x->data[3] == (float)0.2 && mn.x->data[7] == (float)0.4);
        ASSERT_TRUE(mn.y->data[0] == (float)7.0 && mn.y->data[1] == (float)3.0);
    }
    dataset_free(&mn);
    struct Dataset none = dataset_load_mnist("build/no-such-images", "build/no-such-labels", 0UL);
    ASSERT_EQ(none.n, 0UL);
    dataset_free(&none);
}

// ── Trainer ──────────────────────────────────────────────────────────────────
static &Tensor clf_loss_(&Mlp m, const &Tensor x, const &Tensor y) {
    return tensor_cross_entropy_loss(mlp_forward(m, x), y);
}
static &Tensor clf_logits_(&Mlp m, const &Tensor x) { return mlp_forward(m, x); }

void t_trainer_moons() {
    struct Rng rng = rng_new(11UL);
    struct Dataset data = dataset_moons(400UL, (float)0.1, &rng);
    struct DatasetSplit sp = dataset_split(&data, (float)0.25, &rng);
    struct Mlp net = mlp_new(2UL, 32UL, 2UL, 2UL, ACT_RELU, ACT_NONE, &rng);
    struct Optimizer opt = optimizer_new(OPT_ADAM, (float)0.01);
    optimizer_add_mlp(&opt, &net);
    struct TrainConfig cfg = train_config_default();
    cfg.epochs = 60UL;
    struct TrainReport rep = trainer_fit(&net, clf_loss_, &opt, &sp.train, &cfg);
    ASSERT_EQ(rep.epochs, 60UL);
    ASSERT_TRUE(rep.lastEpochLoss < rep.firstEpochLoss * (float)0.2);
    float acc = trainer_accuracy(&net, clf_logits_, &sp.test, 64UL);
    ASSERT_TRUE(acc > (float)0.95);
    float val = trainer_evaluate(&net, clf_loss_, &sp.test, 64UL);
    ASSERT_TRUE(val < (float)0.2);
    optimizer_free(&opt);
    mlp_free(&net);
    dataset_free(&sp.train); dataset_free(&sp.test); dataset_free(&data);
}

static &Tensor reg_loss_(&Mlp m, const &Tensor x, const &Tensor y) {
    return tensor_mse_loss(mlp_forward(m, x), y);
}

void t_trainer_sine() {
    struct Rng rng = rng_new(5UL);
    struct Dataset data = dataset_sine(512UL, (float)0.0, &rng);
    struct Mlp net = mlp_new(1UL, 32UL, 2UL, 1UL, ACT_TANH, ACT_NONE, &rng);
    struct Optimizer opt = optimizer_new(OPT_ADAMW, (float)0.01);
    opt.weightDecay = (float)0.0001;
    optimizer_add_mlp(&opt, &net);
    struct TrainConfig cfg = train_config_default();
    cfg.epochs = 80UL;
    cfg.clipNorm = (float)1.0;
    cfg.lrDecay = (float)0.98;
    struct TrainReport rep = trainer_fit(&net, reg_loss_, &opt, &data, &cfg);
    ASSERT_TRUE(rep.lastEpochLoss < (float)0.01);
    optimizer_free(&opt);
    mlp_free(&net);
    dataset_free(&data);
}

// ── Autoencoders ─────────────────────────────────────────────────────────────
static float recon_err_(&Autoencoder ae, const &Dataset d) {
    int was = tensor_is_grad_enabled();
    tensor_set_grad_enabled(0);
    &Tensor r = autoencoder_reconstruct(ae, d.x);
    float e;
    unsafe { e = tensor_mse_loss(r, d.x)->data[0]; }
    tensor_set_grad_enabled(was);
    return e;
}

void t_autoencoders() {
    struct Rng rng = rng_new(21UL);
    struct Dataset blobs = dataset_blobs(256UL, 4UL, 8UL, (float)0.3, &rng);
    struct Dataset shapes = dataset_shapes(256UL, 10UL, &rng);
    struct TrainConfig tc = train_config_default();
    tc.epochs = 40UL;
    // plain, denoising (on images, BCE) and sparse
    int variant = 0;
    while (variant < 3) {
        struct AutoencoderConfig cfg = autoencoder_config_default(variant == 1 ? 16UL : 3UL);
        cfg.hidden = 64UL;
        &Dataset d = variant == 1 ? &shapes : &blobs;
        if (variant == 1) { cfg.recon = RECON_BCE; cfg.noiseStd = (float)0.2; cfg.dropProb = (float)0.1; }
        if (variant == 2) cfg.sparsity = (float)0.05;
        struct Autoencoder ae = autoencoder_new(d.features, &cfg);
        float before = recon_err_(&ae, d);
        struct Optimizer opt = optimizer_new(OPT_ADAM, (float)0.003);
        autoencoder_add_params(&opt, &ae);
        trainer_fit(&ae, autoencoder_loss, &opt, d, &tc);
        float after = recon_err_(&ae, d);
        ASSERT_TRUE(after < before * (float)0.3);
        if (variant == 2) ASSERT_TRUE(ae.lastPenalty > (float)0.0);
        optimizer_free(&opt);
        autoencoder_free(&ae);
        variant = variant + 1;
    }
    dataset_free(&blobs);
    dataset_free(&shapes);
}

static &Tensor shapes_logits_(&Mlp m, const &Tensor x) { return mlp_forward(m, x); }

void t_vae() {
    struct Rng rng = rng_new(4UL);
    struct Dataset shapes = dataset_shapes(512UL, 10UL, &rng);
    // a classifier that later judges the CVAE's samples
    struct Mlp judge = mlp_new(100UL, 64UL, 1UL, 4UL, ACT_RELU, ACT_NONE, &rng);
    struct Optimizer jo = optimizer_new(OPT_ADAM, (float)0.003);
    optimizer_add_mlp(&jo, &judge);
    struct TrainConfig tc = train_config_default();
    tc.epochs = 30UL;
    trainer_fit(&judge, clf_loss_, &jo, &shapes, &tc);
    ASSERT_TRUE(trainer_accuracy(&judge, shapes_logits_, &shapes, 128UL) > (float)0.95);

    int cond = 0;
    while (cond < 2) {
        struct VaeConfig cfg = vae_config_default(4UL);
        cfg.hidden = 128UL;
        cfg.classes = cond ? 4UL : 0UL;
        struct Vae v = vae_new(100UL, &cfg);
        struct Optimizer opt = optimizer_new(OPT_ADAM, (float)0.002);
        vae_add_params(&opt, &v);
        tc.epochs = 60UL;
        struct TrainReport rep = trainer_fit(&v, vae_loss, &opt, &shapes, &tc);
        ASSERT_TRUE(rep.lastEpochLoss < rep.firstEpochLoss * (float)0.6);
        ASSERT_TRUE(v.lastKl > (float)0.1);
        &Tensor labels = tensor_new_1d(64UL, 0);
        unsafe { unsigned long i = 0UL; while (i < 64UL) { labels->data[i] = (float)(i % 4UL); i = i + 1UL; } }
        int was = tensor_is_grad_enabled();
        tensor_set_grad_enabled(0);
        &Tensor s = cond ? vae_sample(&v, 64UL, (?&Tensor)labels) : vae_sample(&v, 64UL, null);
        unsafe {
            ASSERT_EQ(s->shape[0], 64UL);
            ASSERT_EQ(s->shape[1], 100UL);
            ASSERT_TRUE(s->data[0] > (float)0.0 && s->data[0] < (float)1.0);
        }
        if (cond) {
            // conditioned samples are recognized as their class
            float acc = tensor_accuracy(mlp_forward(&judge, s), labels);
            ASSERT_TRUE(acc > (float)0.7);
        }
        tensor_set_grad_enabled(was);
        optimizer_free(&opt);
        vae_free(&v);
        cond = cond + 1;
    }
    optimizer_free(&jo);
    mlp_free(&judge);
    dataset_free(&shapes);
}

void t_vqvae() {
    struct Rng rng = rng_new(8UL);
    struct Dataset blobs = dataset_blobs(512UL, 8UL, 6UL, (float)0.2, &rng);
    struct VqVaeConfig cfg = vqvae_config_default(4UL);
    cfg.hidden = 64UL;
    cfg.codes = 16UL;
    struct VqVae v = vqvae_new(6UL, &cfg);
    struct Optimizer opt = optimizer_new(OPT_ADAM, (float)0.003);
    vqvae_add_params(&opt, &v);
    struct TrainConfig tc = train_config_default();
    tc.epochs = 60UL;
    struct TrainReport rep = trainer_fit(&v, vqvae_loss, &opt, &blobs, &tc);
    ASSERT_TRUE(rep.lastEpochLoss < rep.firstEpochLoss * (float)0.2);
    ASSERT_TRUE(v.lastPerplexity > (float)4.0);     // codes in use, not collapsed to one
    int was = tensor_is_grad_enabled();
    tensor_set_grad_enabled(0);
    &Tensor codes = vqvae_encode(&v, blobs.x);
    &Tensor a = vqvae_decode(&v, codes);
    &Tensor b = vqvae_reconstruct(&v, blobs.x);
    unsafe {
        ASSERT_TRUE(a->data[5] == b->data[5]);
        ASSERT_TRUE(tensor_mse_loss(b, blobs.x)->data[0] < (float)1.0);
    }
    tensor_set_grad_enabled(was);
    optimizer_free(&opt);
    vqvae_free(&v);
    dataset_free(&blobs);
}

// ── GANs ─────────────────────────────────────────────────────────────────────
// Each objective learns a 2D Gaussian at (2, -1) with std (0.5, 0.3):
// the mean tightly, the spread loosely.
void t_gans() {
    struct Rng rng = rng_new(13UL);
    struct Dataset target = dataset_blobs(1024UL, 1UL, 2UL, (float)0.0, &rng);
    unsafe {
        unsigned long i = 0UL;
        while (i < 1024UL) {
            target.x->data[i * 2UL] = (float)2.0 + (float)0.5 * rng_normal(&rng);
            target.x->data[i * 2UL + 1UL] = (float)-1.0 + (float)0.3 * rng_normal(&rng);
            i = i + 1UL;
        }
    }
    int k = 0;
    while (k < 4) {
        struct GanConfig cfg = gan_config_default(4UL);
        cfg.hidden = 32UL;
        cfg.loss = k;
        cfg.emaDecay = (float)0.99;
        cfg.lrG = (float)0.0005; cfg.lrD = (float)0.0005;
        if (k == GAN_WGAN) { cfg.nCritic = 5UL; cfg.lrG = (float)0.0005; cfg.lrD = (float)0.0005; cfg.clip = (float)0.05; }
        struct Gan g = gan_new(2UL, &cfg);
        gan_train(&g, &target, k == GAN_WGAN ? 100UL : 80UL, 64UL);
        &Tensor s = gan_generate(&g, 2000UL, null);
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
            s->free(); free((void*)s);
        }
        gan_free(&g);
        k = k + 1;
    }
    dataset_free(&target);
}

// A conditional GAN puts each label's samples on its own mode of a ring.
void t_cgan() {
    struct Rng rng = rng_new(17UL);
    struct Dataset ring = dataset_gaussian_ring(2048UL, 4UL, (float)2.0, (float)0.1, &rng);
    struct GanConfig cfg = gan_config_default(4UL);
    cfg.hidden = 64UL;
    cfg.classes = 4UL;
    cfg.lrG = (float)0.001; cfg.lrD = (float)0.001;
    struct Gan g = gan_new(2UL, &cfg);
    gan_train(&g, &ring, 30UL, 64UL);
    &Tensor labels = tensor_new_1d(400UL, 0);
    unsafe { unsigned long i = 0UL; while (i < 400UL) { labels->data[i] = (float)(i % 4UL); i = i + 1UL; } }
    &Tensor s = gan_generate(&g, 400UL, (?&Tensor)labels);
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
    gan_free(&g);
    dataset_free(&ring);
}

// ── Reinforcement learning ───────────────────────────────────────────────────
void t_bandits() {
    int kind = 0;
    while (kind < 4) {
        struct Bandit b = bandit_new(10UL, 3UL);
        float param = kind == BANDIT_EPS_GREEDY ? (float)0.1 :
                      (kind == BANDIT_UCB1 ? (float)1.414 : (kind == BANDIT_THOMPSON ? (float)1.0 : (float)0.1));
        struct BanditAgent ag = bandit_agent_new(kind, 10UL, param, 5UL);
        unsigned long best = bandit_best_arm(&b);
        unsigned long hits = 0UL;
        unsigned long t = 0UL;
        while (t < 2000UL) {
            unsigned long a = bandit_agent_select(&ag);
            bandit_agent_update(&ag, a, bandit_pull(&b, a));
            if (t >= 1000UL && a == best) hits = hits + 1UL;
            t = t + 1UL;
        }
        ASSERT_TRUE(hits > 850UL);
        bandit_agent_free(&ag);
        bandit_free(&b);
        kind = kind + 1;
    }
}

// Greedy return on the cliff walk after training with one of the TD rules.
static float cliff_return_(int algo) {
    struct GridWorld g = gridworld_cliff(1UL);
    struct TabularAgent ag = tabular_agent_new(gridworld_states(&g), 4UL, (float)0.5, (float)1.0, (float)0.1, 2UL);
    int ep = 0;
    while (ep < 500) {
        unsigned long s = gridworld_reset(&g);
        unsigned long a = tabular_act(&ag, s);
        while (1) {
            unsigned long s2 = 0UL;
            struct EnvStep st = gridworld_step(&g, a, &s2);
            unsigned long a2 = tabular_act(&ag, s2);
            if (algo == 0) q_learning_update(&ag, s, a, st.reward, s2, st.terminated);
            else if (algo == 1) sarsa_update(&ag, s, a, st.reward, s2, a2, st.terminated);
            else expected_sarsa_update(&ag, s, a, st.reward, s2, st.terminated);
            s = s2;
            a = a2;
            if (st.terminated || st.truncated) break;
        }
        ep = ep + 1;
    }
    unsigned long s = gridworld_reset(&g);
    float ret = (float)0.0;
    unsigned long n = 0UL;
    while (n < 100UL) {
        unsigned long s2 = 0UL;
        struct EnvStep st = gridworld_step(&g, tabular_greedy(&ag, s), &s2);
        ret = ret + st.reward;
        s = s2;
        n = n + 1UL;
        if (st.terminated) break;
    }
    tabular_agent_free(&ag);
    gridworld_free(&g);
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
    struct GridWorld g = gridworld_new(3UL, 3UL, 1UL);
    gridworld_set(&g, 1UL, 0UL, GRID_WALL);
    gridworld_set(&g, 0UL, 2UL, GRID_PIT);
    g.pitTerminal = 1;
    unsigned long s = gridworld_reset(&g);
    ASSERT_EQ(s, 0UL);
    unsigned long n = 0UL;
    struct EnvStep st = gridworld_step(&g, GRID_RIGHT, &n);   // into the wall: stays
    ASSERT_EQ(n, 0UL);
    ASSERT_TRUE(st.reward == (float)-1.0 && !st.terminated);
    gridworld_step(&g, GRID_DOWN, &n);
    st = gridworld_step(&g, GRID_DOWN, &n);                    // into the pit
    ASSERT_TRUE(st.terminated && st.reward == (float)-100.0);
    &Tensor oh = tensor_new_1d(9UL, 0);
    gridworld_one_hot(&g, 4UL, oh);
    unsafe { ASSERT_TRUE(oh->data[4] == (float)1.0 && oh->data[3] == (float)0.0); }
    gridworld_free(&g);
}

void t_replay_buffer() {
    struct ReplayBuffer rb = replay_new(3UL, 2UL);
    &Tensor o = tensor_new_1d(2UL, 0);
    unsigned long i = 0UL;
    while (i < 5UL) {
        unsafe { o->data[0] = (float)i; }
        replay_push(&rb, o, i, (float)i, o, 0);
        i = i + 1UL;
    }
    ASSERT_EQ(rb.size, 3UL);
    ASSERT_EQ(rb.pos, 2UL);
    unsafe { ASSERT_TRUE(rb.obs->data[0] == (float)3.0 && rb.reward->data[1] == (float)4.0); }
    replay_free(&rb);
}

static void rl_copy4_(&Tensor dst, const &Tensor src) {
    unsafe { unsigned long i = 0UL; while (i < 4UL) { dst->data[i] = src->data[i]; i = i + 1UL; } }
}

// Mean greedy return over 5 CartPole episodes.
static float eval_dqn_(&Dqn d) {
    struct CartPole env = cartpole_new(99UL);
    &Tensor obs = tensor_new_1d(4UL, 0);
    &Tensor nx = tensor_new_1d(4UL, 0);
    float total = (float)0.0;
    int ep = 0;
    while (ep < 5) {
        cartpole_reset(&env, obs);
        while (1) {
            struct EnvStep s = cartpole_step(&env, dqn_act(d, obs, 0), nx);
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
    struct CartPole env = cartpole_new(99UL);
    &Tensor obs = tensor_new_1d(4UL, 0);
    &Tensor nx = tensor_new_1d(4UL, 0);
    float total = (float)0.0;
    int ep = 0;
    while (ep < 5) {
        cartpole_reset(&env, obs);
        while (1) {
            struct EnvStep s = cartpole_step(&env, pg_act(p, obs, 0), nx);
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
    struct DqnConfig cfg = dqn_config_default();
    struct Dqn d = dqn_new(CARTPOLE_OBS, CARTPOLE_ACTIONS, &cfg);
    struct CartPole env = cartpole_new(8UL);
    &Tensor obs = tensor_new_1d(4UL, 0);
    &Tensor nx = tensor_new_1d(4UL, 0);
    unsigned long steps = 0UL;
    int ep = 0;
    float best = (float)0.0;
    while (steps < 40000UL && best < (float)195.0) {
        cartpole_reset(&env, obs);
        while (1) {
            unsigned long a = dqn_act(&d, obs, 1);
            struct EnvStep s = cartpole_step(&env, a, nx);
            dqn_observe(&d, obs, a, s.reward, nx, s.terminated);
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
    dqn_free(&d);
}

static float pg_cartpole_(int algo, unsigned long rolloutLen) {
    struct PgConfig cfg = pg_config_default(algo);
    struct PolicyAgent p = pg_new(CARTPOLE_OBS, CARTPOLE_ACTIONS, &cfg);
    struct CartPole env = cartpole_new(12UL);
    &Tensor obs = tensor_new_1d(4UL, 0);
    &Tensor nx = tensor_new_1d(4UL, 0);
    unsigned long steps = 0UL;
    int ep = 0;
    float best = (float)0.0;
    cartpole_reset(&env, obs);
    while (steps < 60000UL && best < (float)195.0) {
        unsigned long a = pg_act(&p, obs, 1);
        struct EnvStep s = cartpole_step(&env, a, nx);
        pg_record(&p, obs, a, s.reward, nx, s.terminated, s.truncated);
        steps = steps + 1UL;
        rl_copy4_(obs, nx);
        int done = s.terminated || s.truncated;
        if ((algo == PG_REINFORCE && done) || (algo != PG_REINFORCE && p.rollout >= rolloutLen)) pg_update(&p);
        if (done) {
            ep = ep + 1;
            cartpole_reset(&env, obs);
            if (ep % 25 == 0) { float e = eval_pg_(&p); if (e > best) best = e; }
        }
    }
    tensor_drop(obs); tensor_drop(nx);
    pg_free(&p);
    return best;
}

void t_reinforce_cartpole() { ASSERT_TRUE(pg_cartpole_(PG_REINFORCE, 0UL) >= (float)195.0); }
void t_a2c_cartpole() { ASSERT_TRUE(pg_cartpole_(PG_A2C, 32UL) >= (float)195.0); }
void t_ppo_cartpole() { ASSERT_TRUE(pg_cartpole_(PG_PPO, 1024UL) >= (float)195.0); }

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
        t.add("GAN objectives", t_gans);
        t.add("conditional GAN", t_cgan);
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
