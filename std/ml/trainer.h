#pragma once
// SafeC Standard Library — the training loop.
//
// trainer_fit runs epochs of mini-batch gradient descent over a Dataset:
// for each batch it calls your loss function, back-propagates, optionally
// clips the gradient norm, and steps the optimizer. Each step's tensors go
// to a TensorArena that is reset after the step, so a long run allocates
// nothing per step once the stepArena has warmed up. The model can be any
// struct: the loss function receives it, the batch inputs and the batch
// targets, and returns a 1-element loss tensor.
//
//   static &Tensor clf_loss(&Mlp m, const &Tensor x, const &Tensor y) {
//       return tensor_cross_entropy_loss(m.forward(x), y);
//   }
//   struct Optimizer opt = optimizer_new(OPT_ADAM, (float)0.01);
//   net.add_params(&opt);
//   struct TrainConfig cfg = train_config_default();
//   struct TrainReport rep = trainer_fit(&net, clf_loss, &opt, &train, &cfg);
//
// The model, the forward pass and the loss are all resolved at compile
// time (trainer_fit is generic over the model type).
//
// train_traits.h puts this loop and nn_rocm.h's rocm_trainer_fit behind one
// interface (the Trainable trait), and train.h picks the backend at build
// time.
#include <std/ml/tensor.h>
#include <std/ml/nn.h>
#include <std/ml/datasets.h>

namespace std {

struct TrainConfig {
    unsigned long epochs;
    unsigned long batchSize;
    unsigned long seed;      // shuffling
    int   shuffle;
    float clipNorm;          // > 0: clip the global gradient norm to this
    float lrDecay;           // lr *= lrDecay after every epoch (1 = constant)
};

struct TrainReport {
    float firstEpochLoss;    // mean batch loss of the first epoch
    float lastEpochLoss;     // ... and of the last
    float bestEpochLoss;
    unsigned long steps;
    unsigned long epochs;
};

// epochs 10, batch 32, seed 1, shuffled, no clipping, constant lr.
struct TrainConfig train_config_default();

// One epoch's result, as a Trainable's fit_epoch reports it (train_traits.h).
struct EpochLoss {
    float loss;              // mean batch loss
    unsigned long steps;     // batches (optimizer steps) taken
    int   ok;                // 0: the epoch failed (a device error)
};

// How a backend-neutral trainer (train.h) is built. The CPU ignores both:
// maxBatch sizes a device trainer's buffers (cfg.batchSize must not exceed
// it), precision is the activations' type on a device that supports it
// (0: fp32, TENSOR_FP16 or TENSOR_BF16, tensor_half.h's values).
struct TrainerOptions {
    unsigned long maxBatch;
    int precision;
};
// maxBatch 256, fp32.
struct TrainerOptions trainer_options_default();

// Internal: one optimizer step from a loss tensor.
float trainer_step_(&Optimizer opt, &Tensor loss, float clipNorm);

// Internal: a fresh report, and one epoch's mean loss and step count added.
struct TrainReport train_report_new_();
void train_report_epoch_(&TrainReport rep, unsigned long epoch, float mean, unsigned long steps);

// Internal: one epoch of trainer_fit over dl (its next pass), the step
// tensors in stepArena. Returns the mean batch loss and sets *batches.
generic<T>
float trainer_epoch_(&T model, fn &Tensor(&T, const &Tensor, const &Tensor) lossFn, &Optimizer opt,
                     &DataLoader dl, &TensorArena stepArena, float clipNorm, &unsigned long batches) {
    double sum = 0.0;
    unsigned long n = 0UL;
    struct Batch b;
    stepArena.begin();
    while (dl.next(&b)) {
        &Tensor loss = lossFn(model, b.x, b.y);
        sum = sum + (double)trainer_step_(opt, loss, clipNorm);
        n = n + 1UL;
        stepArena.reset();
    }
    tensor_arena_end();
    stepArena.reset();
    *batches = n;
    return n > 0UL ? (float)(sum / (double)n) : (float)0.0;
}

generic<T>
struct TrainReport trainer_fit(&T model, fn &Tensor(&T, const &Tensor, const &Tensor) lossFn,
                               &Optimizer opt, const &Dataset data, const &TrainConfig cfg) {
    struct TrainReport rep = train_report_new_();
    struct DataLoader dl = dataloader_new(data, cfg.batchSize, cfg.shuffle, cfg.seed);
    struct TensorArena stepArena = tensor_arena_new(0UL);
    unsigned long e = 0UL;
    while (e < cfg.epochs) {
        unsigned long batches = 0UL;
        float mean = trainer_epoch_(model, lossFn, opt, &dl, &stepArena, cfg.clipNorm, &batches);
        train_report_epoch_(&rep, e, mean, batches);
        opt.lr = opt.lr * cfg.lrDecay;
        e = e + 1UL;
    }
    stepArena.free();
    dl.free();
    return rep;
}

// Mean loss over a dataset without gradients (validation).
generic<T>
float trainer_evaluate(&T model, fn &Tensor(&T, const &Tensor, const &Tensor) lossFn,
                       const &Dataset data, unsigned long batchSize) {
    struct DataLoader dl = dataloader_new(data, batchSize, 0, 0UL);
    struct TensorArena stepArena = tensor_arena_new(0UL);
    int was = tensor_is_grad_enabled();
    tensor_set_grad_enabled(0);
    struct Batch b;
    double sum = 0.0;
    unsigned long rows = 0UL;
    stepArena.begin();
    while (dl.next(&b)) {
        &Tensor loss = lossFn(model, b.x, b.y);
        unsafe { sum = sum + (double)loss->data[0] * (double)b.rows; }
        rows = rows + b.rows;
        stepArena.reset();
    }
    tensor_arena_end();
    tensor_set_grad_enabled(was);
    stepArena.free();
    dl.free();
    return rows > 0UL ? (float)(sum / (double)rows) : (float)0.0;
}

// Classification accuracy over a dataset: 'logitsFn' maps a batch to
// [rows, classes] logits.
generic<T>
float trainer_accuracy(&T model, fn &Tensor(&T, const &Tensor) logitsFn,
                       const &Dataset data, unsigned long batchSize) {
    struct DataLoader dl = dataloader_new(data, batchSize, 0, 0UL);
    struct TensorArena stepArena = tensor_arena_new(0UL);
    int was = tensor_is_grad_enabled();
    tensor_set_grad_enabled(0);
    struct Batch b;
    double hits = 0.0;
    unsigned long rows = 0UL;
    stepArena.begin();
    while (dl.next(&b)) {
        &Tensor logits = logitsFn(model, b.x);
        hits = hits + (double)tensor_accuracy(logits, b.y) * (double)b.rows;
        rows = rows + b.rows;
        stepArena.reset();
    }
    tensor_arena_end();
    tensor_set_grad_enabled(was);
    stepArena.free();
    dl.free();
    return rows > 0UL ? (float)(hits / (double)rows) : (float)0.0;
}

} // namespace std
