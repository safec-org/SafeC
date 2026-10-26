#pragma once
// SafeC Standard Library — models as composable traits.
//
// Every std::ml model is a struct with methods. The traits below name what
// a model can do, one capability each, and compose into the roles generic
// code asks for (trait composition: 'trait Module : Parameterized + Forward
// {}' requires both). The generic functions after them are written once and
// apply to every model with those capabilities; each call is resolved at
// compile time.
//
//   Parameterized  collect_params / collect_state: its tensors
//   Forward        forward(x)
//   Objective      loss(x, y): its own training loss (autoencoders, VAEs)
//   Module         Parameterized + Forward: Mlp, ConvNet, LoraMlp,
//                  QLoraMlp, LayerNorm, MultiHeadAttention, Mla, EgMla,
//                  JiTBlock, Mamba
//   LossModel      Parameterized + Objective: Autoencoder, Vae, VqVae,
//                  ConvAutoencoder, ConvVae
//   CondModule     Parameterized + forward(x, c): DiTBlock, JiT
//   Recurrent      Parameterized + forward(x_t, h): RNNCell, GRUCell
//   Generative     sample(n, labels): Gan, Dcgan, Vae, ConvVae
//
//   struct Mlp net = Mlp::new(2UL, 32UL, 2UL, 2UL, ACT_RELU, ACT_NONE, &rng);
//   struct Optimizer opt = Optimizer::new(OPT_ADAM, (float)0.01);
//   model_add_params(&net, &opt);
//   struct TrainReport rep = module_fit(&net, TRAIN_LOSS_CE, &opt, &data, &cfg);
//   float acc = module_accuracy(&net, &test, 64UL);
//   model_save_file(&net, "net.bin");
//
// The same calls work on a ConvNet, a LoraMlp, a Mamba layer, ... and on a
// model of your own once it has the methods.
#include <std/ml/tensor.h>
#include <std/ml/nn.h>
#include <std/ml/datasets.h>
#include <std/ml/trainer.h>
#include <std/collections/vec.h>

namespace std {

// Built-in losses for module_fit / module_loss.
#define TRAIN_LOSS_CE  0   // softmax cross-entropy on class indices (tensor_cross_entropy_loss)
#define TRAIN_LOSS_MSE 1   // mean squared error (tensor_mse_loss)
#define TRAIN_LOSS_BCE 2   // binary cross-entropy on logits (tensor_bce_logits_loss)

trait Parameterized {
    // Appends every trained tensor, always in the same order.
    void collect_params(&Vec<&FloatTensor> out) const;
    // Appends what a copy or a file carries: the trained tensors, then the
    // non-trained buffers (batch norm running statistics, spectral norms' u).
    void collect_state(&Vec<&FloatTensor> out) const;
}

trait Forward {
    &FloatTensor forward(const &FloatTensor x);
}

trait Objective {
    // The model's own training loss on a batch (1 element).
    &FloatTensor loss(const &FloatTensor x, const &FloatTensor y);
}

trait Module : Parameterized + Forward {}

trait LossModel : Parameterized + Objective {}

trait CondModule : Parameterized {
    &FloatTensor forward(const &FloatTensor x, const &FloatTensor c);   // c: the conditioning
}

trait Recurrent : Parameterized {
    &FloatTensor forward(const &FloatTensor x_t, const &FloatTensor h_prev);   // -> h_t
}

trait Generative {
    // n new samples; labels (n class indices) for a conditional model, else null.
    &FloatTensor sample(unsigned long n, ?&FloatTensor labels);
}

// ── Parameters, files, copies: any Parameterized ────────────────────────────

// Every trained tensor into o.
generic<M: Parameterized>
void model_add_params(const &M m, &Optimizer o) {
    struct Vec<&FloatTensor> ps = vec_new();
    m.collect_params(&ps);
    unsigned long i = 0UL;
    while (i < ps.length()) { o.add(*ps.at(i)); i = i + 1UL; }
    ps.free();
}

// Trained values.
generic<M: Parameterized>
unsigned long model_param_count(const &M m) {
    struct Vec<&FloatTensor> ps = vec_new();
    m.collect_params(&ps);
    unsigned long n = 0UL;
    unsigned long i = 0UL;
    while (i < ps.length()) {
        &FloatTensor p = *ps.at(i);
        unsafe { n = n + p->size; }
        i = i + 1UL;
    }
    ps.free();
    return n;
}

// Places the whole state (parameters and buffers) on 'device' (device.h):
// the model then trains and runs there through the same calls. 1 when every
// tensor moved; an optimizer's moments follow its parameters on their next
// step. DEVICE_AUTO picks device_default().
generic<M: Parameterized>
int model_to(&M m, int device) {
    int d = device == DEVICE_AUTO ? device_default() : device;
    struct Vec<&FloatTensor> ts = vec_new();
    m.collect_state(&ts);
    int ok = 1;
    unsigned long i = 0UL;
    while (i < ts.length()) {
        if (!(*ts.at(i)).place(d)) ok = 0;
        i = i + 1UL;
    }
    ts.free();
    return ok;
}

// Where the model's first tensor lives (DEVICE_CPU for a model without any).
generic<M: Parameterized>
int model_device(const &M m) {
    struct Vec<&FloatTensor> ts = vec_new();
    m.collect_state(&ts);
    int d = DEVICE_CPU;
    if (ts.length() > 0UL) d = (*ts.at(0UL)).device();
    ts.free();
    return d;
}

// Makes every host copy of the model's state current (after device steps):
// before reading parameters directly or saving. 1 on success.
generic<M: Parameterized>
int model_sync(&M m) {
    struct Vec<&FloatTensor> ts = vec_new();
    m.collect_state(&ts);
    int ok = 1;
    unsigned long i = 0UL;
    while (i < ts.length()) {
        if (!(*ts.at(i)).sync()) ok = 0;
        i = i + 1UL;
    }
    ts.free();
    return ok;
}

// The state (collect_state) to a file in tensors_save's format, and back
// into a model built with the same shapes. 1 on success.
generic<M: Parameterized>
int model_save_file(const &M m, const &char path) {
    struct Vec<&FloatTensor> ts = vec_new();
    m.collect_state(&ts);
    int ok = tensors_save(&ts, path);
    ts.free();
    return ok;
}

generic<M: Parameterized>
int model_load_file(&M m, const &char path) {
    struct Vec<&FloatTensor> ts = vec_new();
    m.collect_state(&ts);
    int ok = tensors_load(&ts, path);
    ts.free();
    return ok;
}

// dst = tau * src + (1 - tau) * dst over the whole state (same shapes):
// target networks, averaged generators. Returns 0 on a shape mismatch.
generic<M: Parameterized>
int model_soft_update(&M dst, const &M src, float tau) {
    struct Vec<&FloatTensor> d = vec_new();
    struct Vec<&FloatTensor> s = vec_new();
    dst.collect_state(&d);
    src.collect_state(&s);
    int ok = d.length() == s.length();
    unsigned long i = 0UL;
    while (ok && i < d.length()) {
        &FloatTensor a = *d.at(i);
        &FloatTensor b = *s.at(i);
        unsafe {
            if (a->size != b->size) { ok = 0; }
            else if (a->dev != DEVICE_CPU && a->dev == b->dev && tensor_dev_raw_((void*)a, a->dev) && tensor_dev_raw_((void*)b, a->dev)) {
                // both on one device: a = (1 - tau) a + tau b there (a copy for tau 1)
                if (tau == (float)1.0) {
                    device_copy_(a->dev, a->dbuf, b->dbuf, a->size);
                } else {
                    struct DevArgs g = dev_args_();
                    g.p[0] = b->dbuf; g.p[2] = a->dbuf; g.u[0] = (unsigned int)EW_LERP; g.f[0] = tau;
                    device_launch_(a->dev, DK_EW, a->size, &g);
                }
                tensor_dev_wrote_raw_((void*)a);
            } else {
                tensor_host_write_raw_((void*)a);
                tensor_host_raw_((void*)b);
                unsigned long j = 0UL;
                while (j < a->size) { a->data[j] = tau * b->data[j] + ((float)1.0 - tau) * a->data[j]; j = j + 1UL; }
            }
        }
        i = i + 1UL;
    }
    d.free();
    s.free();
    return ok;
}

generic<M: Parameterized>
int model_copy(&M dst, const &M src) { return model_soft_update(dst, src, (float)1.0); }

// ── Training and evaluation: any Module ─────────────────────────────────────

// A built-in loss (TRAIN_LOSS_*) of m's output against y.
generic<M: Module>
&FloatTensor module_loss(&M m, int loss, const &FloatTensor x, const &FloatTensor y) {
    &FloatTensor out = m.forward(x);
    if (loss == TRAIN_LOSS_MSE) return tensor_mse_loss(out, y);
    if (loss == TRAIN_LOSS_BCE) return tensor_bce_logits_loss(out, y);
    return tensor_cross_entropy_loss(out, y);
}

// forward(x) without recording gradients.
generic<M: Module>
&FloatTensor module_predict(&M m, const &FloatTensor x) {
    struct GradGuard noGrad = Autograd::get().no_grad();
    &FloatTensor y = m.forward(x);
    noGrad.restore();
    return y;
}

// Internal: one epoch of module_fit over dl; returns the mean batch loss.
generic<M: Module>
float module_epoch_(&M m, int loss, &Optimizer opt, &DataLoader dl, &TensorArena stepArena, float clipNorm,
                    &unsigned long batches) {
    struct TrainLossSum sum = trainer_loss_sum_();
    struct Batch b;
    stepArena.begin();
    while (dl.next(&b)) {
        trainer_step_sum_(opt, module_loss(m, loss, b.x, b.y), clipNorm, &sum);
        stepArena.reset();
    }
    tensor_arena_end();
    stepArena.reset();
    *batches = sum.steps;
    return trainer_loss_mean_(&sum);
}

// trainer_fit with a built-in loss: the same batches, steps, lr decay and
// report. opt holds m's parameters (model_add_params).
generic<M: Module>
struct TrainReport module_fit(&M m, int loss, &Optimizer opt, const &Dataset data, const &TrainConfig cfg) {
    struct TrainReport rep = train_report_new_();
    trainer_place_(opt, cfg.device);
    struct DataLoader dl = DataLoader::new(data, cfg.batchSize, cfg.shuffle, cfg.seed);
    trainer_place_data_(&dl, cfg.device);
    int prevPrec = trainer_precision_begin_(cfg);
    struct TensorArena stepArena = TensorArena::new(0UL);
    unsigned long e = 0UL;
    while (e < cfg.epochs) {
        unsigned long batches = 0UL;
        float mean = module_epoch_(m, loss, opt, &dl, &stepArena, cfg.clipNorm, &batches);
        train_report_epoch_(&rep, e, mean, batches);
        opt.lr = opt.lr * cfg.lrDecay;
        e = e + 1UL;
    }
    trainer_precision_end_(cfg, prevPrec);
    stepArena.free();
    dl.free();
    return rep;
}

// Mean built-in loss over a dataset, without gradients.
generic<M: Module>
float module_evaluate(&M m, int loss, const &Dataset data, unsigned long batchSize) {
    struct DataLoader dl = DataLoader::new(data, batchSize, 0, 0UL);
    struct TensorArena stepArena = TensorArena::new(0UL);
    struct GradGuard noGrad = Autograd::get().no_grad();
    struct Batch b;
    double sum = 0.0;
    unsigned long rows = 0UL;
    stepArena.begin();
    while (dl.next(&b)) {
        &FloatTensor l = module_loss(m, loss, b.x, b.y);
        unsafe { tensor_host_raw_((void*)l); sum = sum + (double)l->data[0] * (double)b.rows; }
        rows = rows + b.rows;
        stepArena.reset();
    }
    tensor_arena_end();
    noGrad.restore();
    stepArena.free();
    dl.free();
    return rows > 0UL ? (float)(sum / (double)rows) : (float)0.0;
}

// Classification accuracy of m's logits over a dataset.
generic<M: Module>
float module_accuracy(&M m, const &Dataset data, unsigned long batchSize) {
    struct DataLoader dl = DataLoader::new(data, batchSize, 0, 0UL);
    struct TensorArena stepArena = TensorArena::new(0UL);
    struct GradGuard noGrad = Autograd::get().no_grad();
    struct Batch b;
    double hits = 0.0;
    unsigned long rows = 0UL;
    stepArena.begin();
    while (dl.next(&b)) {
        hits = hits + (double)tensor_accuracy(m.forward(b.x), b.y) * (double)b.rows;
        rows = rows + b.rows;
        stepArena.reset();
    }
    tensor_arena_end();
    noGrad.restore();
    stepArena.free();
    dl.free();
    return rows > 0UL ? (float)(hits / (double)rows) : (float)0.0;
}

// ── Training and evaluation: any LossModel ──────────────────────────────────

// Internal: one epoch of model_fit.
generic<M: LossModel>
float model_epoch_(&M m, &Optimizer opt, &DataLoader dl, &TensorArena stepArena, float clipNorm,
                   &unsigned long batches) {
    struct TrainLossSum sum = trainer_loss_sum_();
    struct Batch b;
    stepArena.begin();
    while (dl.next(&b)) {
        trainer_step_sum_(opt, m.loss(b.x, b.y), clipNorm, &sum);
        stepArena.reset();
    }
    tensor_arena_end();
    stepArena.reset();
    *batches = sum.steps;
    return trainer_loss_mean_(&sum);
}

// trainer_fit on the model's own loss (an autoencoder's reconstruction, a
// VAE's ELBO, ...).
generic<M: LossModel>
struct TrainReport model_fit(&M m, &Optimizer opt, const &Dataset data, const &TrainConfig cfg) {
    struct TrainReport rep = train_report_new_();
    trainer_place_(opt, cfg.device);
    struct DataLoader dl = DataLoader::new(data, cfg.batchSize, cfg.shuffle, cfg.seed);
    trainer_place_data_(&dl, cfg.device);
    int prevPrec = trainer_precision_begin_(cfg);
    struct TensorArena stepArena = TensorArena::new(0UL);
    unsigned long e = 0UL;
    while (e < cfg.epochs) {
        unsigned long batches = 0UL;
        float mean = model_epoch_(m, opt, &dl, &stepArena, cfg.clipNorm, &batches);
        train_report_epoch_(&rep, e, mean, batches);
        opt.lr = opt.lr * cfg.lrDecay;
        e = e + 1UL;
    }
    trainer_precision_end_(cfg, prevPrec);
    stepArena.free();
    dl.free();
    return rep;
}

// Mean loss over a dataset, without gradients.
generic<M: LossModel>
float model_evaluate(&M m, const &Dataset data, unsigned long batchSize) {
    struct DataLoader dl = DataLoader::new(data, batchSize, 0, 0UL);
    struct TensorArena stepArena = TensorArena::new(0UL);
    struct GradGuard noGrad = Autograd::get().no_grad();
    struct Batch b;
    double sum = 0.0;
    unsigned long rows = 0UL;
    stepArena.begin();
    while (dl.next(&b)) {
        &FloatTensor l = m.loss(b.x, b.y);
        unsafe { tensor_host_raw_((void*)l); sum = sum + (double)l->data[0] * (double)b.rows; }
        rows = rows + b.rows;
        stepArena.reset();
    }
    tensor_arena_end();
    noGrad.restore();
    stepArena.free();
    dl.free();
    return rows > 0UL ? (float)(sum / (double)rows) : (float)0.0;
}

// ── Recurrent cells ──────────────────────────────────────────────────────────

// Internal: row t of xs as [1, cols] (its own index tensor: the gather's
// backward pass reads it).
&FloatTensor recurrent_row_(const &FloatTensor xs, unsigned long t);

// Runs a cell over a sequence xs [T, inputSize] from h0 [1, hiddenSize] and
// returns the last hidden state [1, hiddenSize] (differentiable through every
// step).
generic<C: Recurrent>
&FloatTensor recurrent_run(&C cell, const &FloatTensor xs, const &FloatTensor h0) {
    unsigned long T = 0UL;
    unsafe { T = xs->shape[0]; }
    if (T == 0UL) return h0.add_scalar((float)0.0);
    &FloatTensor h = cell.forward(recurrent_row_(xs, 0UL), h0);
    unsigned long t = 1UL;
    while (t < T) {
        h = cell.forward(recurrent_row_(xs, t), h);
        t = t + 1UL;
    }
    return h;
}

} // namespace std
