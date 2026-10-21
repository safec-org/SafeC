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
//   struct Mlp net = mlp_new(2UL, 32UL, 2UL, 2UL, ACT_RELU, ACT_NONE, &rng);
//   struct Optimizer opt = optimizer_new(OPT_ADAM, (float)0.01);
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

// Built-in losses for module_fit / module_loss (and the net trainers of train.h).
#define TRAIN_LOSS_CE  0   // softmax cross-entropy on class indices (tensor_cross_entropy_loss)
#define TRAIN_LOSS_MSE 1   // mean squared error (tensor_mse_loss)
#define TRAIN_LOSS_BCE 2   // binary cross-entropy on logits (tensor_bce_logits_loss)

trait Parameterized {
    // Appends every trained tensor, always in the same order.
    void collect_params(&Vec<&Tensor> out) const;
    // Appends what a copy or a file carries: the trained tensors, then the
    // non-trained buffers (batch norm running statistics, spectral norms' u).
    void collect_state(&Vec<&Tensor> out) const;
}

trait Forward {
    &Tensor forward(const &Tensor x);
}

trait Objective {
    // The model's own training loss on a batch (1 element).
    &Tensor loss(const &Tensor x, const &Tensor y);
}

trait Module : Parameterized + Forward {}

trait LossModel : Parameterized + Objective {}

trait CondModule : Parameterized {
    &Tensor forward(const &Tensor x, const &Tensor c);   // c: the conditioning
}

trait Recurrent : Parameterized {
    &Tensor forward(const &Tensor x_t, const &Tensor h_prev);   // -> h_t
}

trait Generative {
    // n new samples; labels (n class indices) for a conditional model, else null.
    &Tensor sample(unsigned long n, ?&Tensor labels);
}

// ── Parameters, files, copies: any Parameterized ────────────────────────────

// Every trained tensor into o.
generic<M: Parameterized>
void model_add_params(const &M m, &Optimizer o) {
    struct Vec<&Tensor> ps = vec_new();
    m.collect_params(&ps);
    unsigned long i = 0UL;
    while (i < ps.length()) { o.add(*ps.at(i)); i = i + 1UL; }
    ps.free();
}

// Trained values.
generic<M: Parameterized>
unsigned long model_param_count(const &M m) {
    struct Vec<&Tensor> ps = vec_new();
    m.collect_params(&ps);
    unsigned long n = 0UL;
    unsigned long i = 0UL;
    while (i < ps.length()) {
        &Tensor p = *ps.at(i);
        unsafe { n = n + p->size; }
        i = i + 1UL;
    }
    ps.free();
    return n;
}

// The state (collect_state) to a file in tensors_save's format, and back
// into a model built with the same shapes. 1 on success.
generic<M: Parameterized>
int model_save_file(const &M m, const &char path) {
    struct Vec<&Tensor> ts = vec_new();
    m.collect_state(&ts);
    int ok = tensors_save(&ts, path);
    ts.free();
    return ok;
}

generic<M: Parameterized>
int model_load_file(&M m, const &char path) {
    struct Vec<&Tensor> ts = vec_new();
    m.collect_state(&ts);
    int ok = tensors_load(&ts, path);
    ts.free();
    return ok;
}

// dst = tau * src + (1 - tau) * dst over the whole state (same shapes):
// target networks, averaged generators. Returns 0 on a shape mismatch.
generic<M: Parameterized>
int model_soft_update(&M dst, const &M src, float tau) {
    struct Vec<&Tensor> d = vec_new();
    struct Vec<&Tensor> s = vec_new();
    dst.collect_state(&d);
    src.collect_state(&s);
    int ok = d.length() == s.length();
    unsigned long i = 0UL;
    while (ok && i < d.length()) {
        &Tensor a = *d.at(i);
        &Tensor b = *s.at(i);
        unsafe {
            if (a->size != b->size) { ok = 0; }
            else {
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
&Tensor module_loss(&M m, int loss, const &Tensor x, const &Tensor y) {
    &Tensor out = m.forward(x);
    if (loss == TRAIN_LOSS_MSE) return tensor_mse_loss(out, y);
    if (loss == TRAIN_LOSS_BCE) return tensor_bce_logits_loss(out, y);
    return tensor_cross_entropy_loss(out, y);
}

// forward(x) without recording gradients.
generic<M: Module>
&Tensor module_predict(&M m, const &Tensor x) {
    int was = tensor_is_grad_enabled();
    tensor_set_grad_enabled(0);
    &Tensor y = m.forward(x);
    tensor_set_grad_enabled(was);
    return y;
}

// Internal: one epoch of module_fit over dl; returns the mean batch loss.
generic<M: Module>
float module_epoch_(&M m, int loss, &Optimizer opt, &DataLoader dl, &TensorArena stepArena, float clipNorm,
                    &unsigned long batches) {
    double sum = 0.0;
    unsigned long n = 0UL;
    struct Batch b;
    stepArena.begin();
    while (dl.next(&b)) {
        sum = sum + (double)trainer_step_(opt, module_loss(m, loss, b.x, b.y), clipNorm);
        n = n + 1UL;
        stepArena.reset();
    }
    tensor_arena_end();
    stepArena.reset();
    *batches = n;
    return n > 0UL ? (float)(sum / (double)n) : (float)0.0;
}

// trainer_fit with a built-in loss: the same batches, steps, lr decay and
// report. opt holds m's parameters (model_add_params).
generic<M: Module>
struct TrainReport module_fit(&M m, int loss, &Optimizer opt, const &Dataset data, const &TrainConfig cfg) {
    struct TrainReport rep = train_report_new_();
    struct DataLoader dl = dataloader_new(data, cfg.batchSize, cfg.shuffle, cfg.seed);
    struct TensorArena stepArena = tensor_arena_new(0UL);
    unsigned long e = 0UL;
    while (e < cfg.epochs) {
        unsigned long batches = 0UL;
        float mean = module_epoch_(m, loss, opt, &dl, &stepArena, cfg.clipNorm, &batches);
        train_report_epoch_(&rep, e, mean, batches);
        opt.lr = opt.lr * cfg.lrDecay;
        e = e + 1UL;
    }
    stepArena.free();
    dl.free();
    return rep;
}

// Mean built-in loss over a dataset, without gradients.
generic<M: Module>
float module_evaluate(&M m, int loss, const &Dataset data, unsigned long batchSize) {
    struct DataLoader dl = dataloader_new(data, batchSize, 0, 0UL);
    struct TensorArena stepArena = tensor_arena_new(0UL);
    int was = tensor_is_grad_enabled();
    tensor_set_grad_enabled(0);
    struct Batch b;
    double sum = 0.0;
    unsigned long rows = 0UL;
    stepArena.begin();
    while (dl.next(&b)) {
        &Tensor l = module_loss(m, loss, b.x, b.y);
        unsafe { sum = sum + (double)l->data[0] * (double)b.rows; }
        rows = rows + b.rows;
        stepArena.reset();
    }
    tensor_arena_end();
    tensor_set_grad_enabled(was);
    stepArena.free();
    dl.free();
    return rows > 0UL ? (float)(sum / (double)rows) : (float)0.0;
}

// Classification accuracy of m's logits over a dataset.
generic<M: Module>
float module_accuracy(&M m, const &Dataset data, unsigned long batchSize) {
    struct DataLoader dl = dataloader_new(data, batchSize, 0, 0UL);
    struct TensorArena stepArena = tensor_arena_new(0UL);
    int was = tensor_is_grad_enabled();
    tensor_set_grad_enabled(0);
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
    tensor_set_grad_enabled(was);
    stepArena.free();
    dl.free();
    return rows > 0UL ? (float)(hits / (double)rows) : (float)0.0;
}

// ── Training and evaluation: any LossModel ──────────────────────────────────

// Internal: one epoch of model_fit.
generic<M: LossModel>
float model_epoch_(&M m, &Optimizer opt, &DataLoader dl, &TensorArena stepArena, float clipNorm,
                   &unsigned long batches) {
    double sum = 0.0;
    unsigned long n = 0UL;
    struct Batch b;
    stepArena.begin();
    while (dl.next(&b)) {
        sum = sum + (double)trainer_step_(opt, m.loss(b.x, b.y), clipNorm);
        n = n + 1UL;
        stepArena.reset();
    }
    tensor_arena_end();
    stepArena.reset();
    *batches = n;
    return n > 0UL ? (float)(sum / (double)n) : (float)0.0;
}

// trainer_fit on the model's own loss (an autoencoder's reconstruction, a
// VAE's ELBO, ...).
generic<M: LossModel>
struct TrainReport model_fit(&M m, &Optimizer opt, const &Dataset data, const &TrainConfig cfg) {
    struct TrainReport rep = train_report_new_();
    struct DataLoader dl = dataloader_new(data, cfg.batchSize, cfg.shuffle, cfg.seed);
    struct TensorArena stepArena = tensor_arena_new(0UL);
    unsigned long e = 0UL;
    while (e < cfg.epochs) {
        unsigned long batches = 0UL;
        float mean = model_epoch_(m, opt, &dl, &stepArena, cfg.clipNorm, &batches);
        train_report_epoch_(&rep, e, mean, batches);
        opt.lr = opt.lr * cfg.lrDecay;
        e = e + 1UL;
    }
    stepArena.free();
    dl.free();
    return rep;
}

// Mean loss over a dataset, without gradients.
generic<M: LossModel>
float model_evaluate(&M m, const &Dataset data, unsigned long batchSize) {
    struct DataLoader dl = dataloader_new(data, batchSize, 0, 0UL);
    struct TensorArena stepArena = tensor_arena_new(0UL);
    int was = tensor_is_grad_enabled();
    tensor_set_grad_enabled(0);
    struct Batch b;
    double sum = 0.0;
    unsigned long rows = 0UL;
    stepArena.begin();
    while (dl.next(&b)) {
        &Tensor l = m.loss(b.x, b.y);
        unsafe { sum = sum + (double)l->data[0] * (double)b.rows; }
        rows = rows + b.rows;
        stepArena.reset();
    }
    tensor_arena_end();
    tensor_set_grad_enabled(was);
    stepArena.free();
    dl.free();
    return rows > 0UL ? (float)(sum / (double)rows) : (float)0.0;
}

// ── Recurrent cells ──────────────────────────────────────────────────────────

// Internal: row t of xs as [1, cols] (its own index tensor: the gather's
// backward pass reads it).
&Tensor recurrent_row_(const &Tensor xs, unsigned long t);

// Runs a cell over a sequence xs [T, inputSize] from h0 [1, hiddenSize] and
// returns the last hidden state [1, hiddenSize] (differentiable through every
// step).
generic<C: Recurrent>
&Tensor recurrent_run(&C cell, const &Tensor xs, const &Tensor h0) {
    unsigned long T = 0UL;
    unsafe { T = xs->shape[0]; }
    if (T == 0UL) return tensor_add_scalar(h0, (float)0.0);
    &Tensor h = cell.forward(recurrent_row_(xs, 0UL), h0);
    unsigned long t = 1UL;
    while (t < T) {
        h = cell.forward(recurrent_row_(xs, t), h);
        t = t + 1UL;
    }
    return h;
}

} // namespace std
