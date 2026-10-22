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
//   struct Optimizer opt = Optimizer::new(OPT_ADAM, (float)0.01);
//   model_add_params(&net, &opt);   // module.h
//   struct TrainConfig cfg = TrainConfig::default();
//   struct TrainReport rep = trainer_fit(&net, clf_loss, &opt, &train, &cfg);
//
// The model, the forward pass and the loss are all resolved at compile
// time (trainer_fit is generic over the model type). For a built-in loss
// on any Module, or a model's own loss (autoencoders, VAEs), module.h's
// module_fit / model_fit need no loss function at all.
//
// The same loops train on a GPU (device.h): set TrainConfig.device, or place
// the model with model_to; every op then runs where its operands live.
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
    // Where to train (device.h): the optimizer's parameters are placed there
    // when fitting starts and stay there (model_to(&m, DEVICE_CPU) or
    // model_sync brings them back). DEVICE_CPU (the default) leaves them
    // where they are; DEVICE_AUTO picks device_default(). The dataset is
    // uploaded once and each batch gathered on the device.
    int   device;
    // What that device's convolutions and matrix products compute in while
    // fitting (device.h's DEVICE_FP32, DEVICE_FP16 with a dynamic loss scale,
    // DEVICE_BF16); the device's previous precision comes back afterwards.
    // DEVICE_PRECISION_KEEP (the default) uses the device's own setting
    // (device_set_precision).
    int   precision;

    // epochs 10, batch 32, seed 1, shuffled, no clipping, constant lr, no
    // device change, the device's precision.
    static struct TrainConfig default();
};

struct TrainReport {
    float firstEpochLoss;    // mean batch loss of the first epoch
    float lastEpochLoss;     // ... and of the last
    float bestEpochLoss;
    unsigned long steps;
    unsigned long epochs;
};


// Internal: places the optimizer's parameters on 'device' (TrainConfig.device).
void trainer_place_(&Optimizer opt, int device);
// Internal: the loader's data on 'device' too (TrainConfig.device).
void trainer_place_data_(&DataLoader dl, int device);
// Internal: TrainConfig.precision on its device for a fit; returns the
// previous precision for trainer_precision_end_.
int  trainer_precision_begin_(const &TrainConfig cfg);
void trainer_precision_end_(const &TrainConfig cfg, int previous);

// Internal: one optimizer step from a loss tensor.
float trainer_step_(&Optimizer opt, &FloatTensor loss, float clipNorm);

// Internal: a fresh report, and one epoch's mean loss and step count added.
struct TrainReport train_report_new_();
void train_report_epoch_(&TrainReport rep, unsigned long epoch, float mean, unsigned long steps);

// Internal: one epoch of trainer_fit over dl (its next pass), the step
// tensors in stepArena. Returns the mean batch loss and sets *batches.
generic<T>
float trainer_epoch_(&T model, fn &FloatTensor(&T, const &FloatTensor, const &FloatTensor) lossFn, &Optimizer opt,
                     &DataLoader dl, &TensorArena stepArena, float clipNorm, &unsigned long batches) {
    double sum = 0.0;
    unsigned long n = 0UL;
    struct Batch b;
    stepArena.begin();
    while (dl.next(&b)) {
        &FloatTensor loss = lossFn(model, b.x, b.y);
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
struct TrainReport trainer_fit(&T model, fn &FloatTensor(&T, const &FloatTensor, const &FloatTensor) lossFn,
                               &Optimizer opt, const &Dataset data, const &TrainConfig cfg) {
    struct TrainReport rep = train_report_new_();
    trainer_place_(opt, cfg.device);
    struct DataLoader dl = DataLoader::new(data, cfg.batchSize, cfg.shuffle, cfg.seed);
    trainer_place_data_(&dl, cfg.device);
    int prevPrec = trainer_precision_begin_(cfg);
    struct TensorArena stepArena = TensorArena::new(0UL);
    unsigned long e = 0UL;
    while (e < cfg.epochs) {
        unsigned long batches = 0UL;
        float mean = trainer_epoch_(model, lossFn, opt, &dl, &stepArena, cfg.clipNorm, &batches);
        train_report_epoch_(&rep, e, mean, batches);
        opt.lr = opt.lr * cfg.lrDecay;
        e = e + 1UL;
    }
    trainer_precision_end_(cfg, prevPrec);
    stepArena.free();
    dl.free();
    return rep;
}

// Mean loss over a dataset without gradients (validation).
generic<T>
float trainer_evaluate(&T model, fn &FloatTensor(&T, const &FloatTensor, const &FloatTensor) lossFn,
                       const &Dataset data, unsigned long batchSize) {
    struct DataLoader dl = DataLoader::new(data, batchSize, 0, 0UL);
    struct TensorArena stepArena = TensorArena::new(0UL);
    struct GradGuard noGrad = Autograd::get().no_grad();
    struct Batch b;
    double sum = 0.0;
    unsigned long rows = 0UL;
    stepArena.begin();
    while (dl.next(&b)) {
        &FloatTensor loss = lossFn(model, b.x, b.y);
        unsafe { tensor_host_((void*)loss); sum = sum + (double)loss->data[0] * (double)b.rows; }
        rows = rows + b.rows;
        stepArena.reset();
    }
    tensor_arena_end();
    noGrad.restore();
    stepArena.free();
    dl.free();
    return rows > 0UL ? (float)(sum / (double)rows) : (float)0.0;
}

// Classification accuracy over a dataset: 'logitsFn' maps a batch to
// [rows, classes] logits.
generic<T>
float trainer_accuracy(&T model, fn &FloatTensor(&T, const &FloatTensor) logitsFn,
                       const &Dataset data, unsigned long batchSize) {
    struct DataLoader dl = DataLoader::new(data, batchSize, 0, 0UL);
    struct TensorArena stepArena = TensorArena::new(0UL);
    struct GradGuard noGrad = Autograd::get().no_grad();
    struct Batch b;
    double hits = 0.0;
    unsigned long rows = 0UL;
    stepArena.begin();
    while (dl.next(&b)) {
        &FloatTensor logits = logitsFn(model, b.x);
        hits = hits + (double)tensor_accuracy(logits, b.y) * (double)b.rows;
        rows = rows + b.rows;
        stepArena.reset();
    }
    tensor_arena_end();
    noGrad.restore();
    stepArena.free();
    dl.free();
    return rows > 0UL ? (float)(hits / (double)rows) : (float)0.0;
}

} // namespace std
