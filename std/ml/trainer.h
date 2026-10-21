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
//       return tensor_cross_entropy_loss(mlp_forward(m, x), y);
//   }
//   struct Optimizer opt = optimizer_new(OPT_ADAM, (float)0.01);
//   optimizer_add_mlp(&opt, &net);
//   struct TrainConfig cfg = train_config_default();
//   struct TrainReport rep = trainer_fit(&net, clf_loss, &opt, &train, &cfg);
//
// The model, the forward pass and the loss are all resolved at compile
// time (trainer_fit is generic over the model type).
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

// Internal: one optimizer step from a loss tensor.
float trainer_step_(&Optimizer opt, &Tensor loss, float clipNorm);

generic<T>
struct TrainReport trainer_fit(&T model, fn &Tensor(&T, const &Tensor, const &Tensor) lossFn,
                               &Optimizer opt, const &Dataset data, const &TrainConfig cfg) {
    struct TrainReport rep;
    rep.firstEpochLoss = (float)0.0;
    rep.lastEpochLoss = (float)0.0;
    rep.bestEpochLoss = (float)3.0e38;
    rep.steps = 0UL;
    rep.epochs = 0UL;
    struct DataLoader dl = dataloader_new(data, cfg.batchSize, cfg.shuffle, cfg.seed);
    struct TensorArena stepArena = tensor_arena_new(0UL);
    struct Batch b;
    unsigned long e = 0UL;
    while (e < cfg.epochs) {
        double sum = 0.0;
        unsigned long batches = 0UL;
        tensor_arena_begin(&stepArena);
        while (dataloader_next(&dl, &b)) {
            &Tensor loss = lossFn(model, b.x, b.y);
            sum = sum + (double)trainer_step_(opt, loss, cfg.clipNorm);
            batches = batches + 1UL;
            tensor_arena_reset(&stepArena);
        }
        tensor_arena_end();
        tensor_arena_reset(&stepArena);
        float mean = batches > 0UL ? (float)(sum / (double)batches) : (float)0.0;
        if (e == 0UL) rep.firstEpochLoss = mean;
        rep.lastEpochLoss = mean;
        if (mean < rep.bestEpochLoss) rep.bestEpochLoss = mean;
        rep.steps = rep.steps + batches;
        rep.epochs = e + 1UL;
        opt.lr = opt.lr * cfg.lrDecay;
        e = e + 1UL;
    }
    tensor_arena_free(&stepArena);
    dataloader_free(&dl);
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
    tensor_arena_begin(&stepArena);
    while (dataloader_next(&dl, &b)) {
        &Tensor loss = lossFn(model, b.x, b.y);
        unsafe { sum = sum + (double)loss->data[0] * (double)b.rows; }
        rows = rows + b.rows;
        tensor_arena_reset(&stepArena);
    }
    tensor_arena_end();
    tensor_set_grad_enabled(was);
    tensor_arena_free(&stepArena);
    dataloader_free(&dl);
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
    tensor_arena_begin(&stepArena);
    while (dataloader_next(&dl, &b)) {
        &Tensor logits = logitsFn(model, b.x);
        hits = hits + (double)tensor_accuracy(logits, b.y) * (double)b.rows;
        rows = rows + b.rows;
        tensor_arena_reset(&stepArena);
    }
    tensor_arena_end();
    tensor_set_grad_enabled(was);
    tensor_arena_free(&stepArena);
    dataloader_free(&dl);
    return rows > 0UL ? (float)(hits / (double)rows) : (float)0.0;
}

} // namespace std
