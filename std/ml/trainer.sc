// SafeC Standard Library — the training loop (see trainer.h).
#pragma once
#include <std/ml/trainer.h>
#include <std/ml/tensor.h>
#include <std/ml/nn.h>

namespace std {

struct TrainConfig TrainConfig::default() {
    struct TrainConfig c;
    c.epochs = 10UL;
    c.batchSize = 32UL;
    c.seed = 1UL;
    c.shuffle = 1;
    c.clipNorm = (float)0.0;
    c.lrDecay = (float)1.0;
    return c;
}

struct TrainerOptions TrainerOptions::default() {
    struct TrainerOptions o;
    o.maxBatch = 256UL;
    o.precision = 0;
    return o;
}

struct TrainReport train_report_new_() {
    struct TrainReport rep;
    rep.firstEpochLoss = (float)0.0;
    rep.lastEpochLoss = (float)0.0;
    rep.bestEpochLoss = (float)3.0e38;
    rep.steps = 0UL;
    rep.epochs = 0UL;
    return rep;
}

void train_report_epoch_(&TrainReport rep, unsigned long epoch, float mean, unsigned long steps) {
    if (epoch == 0UL) rep.firstEpochLoss = mean;
    rep.lastEpochLoss = mean;
    if (mean < rep.bestEpochLoss) rep.bestEpochLoss = mean;
    rep.steps = rep.steps + steps;
    rep.epochs = epoch + 1UL;
}

float trainer_step_(&Optimizer opt, &Tensor loss, float clipNorm) {
    opt.zero_grad();
    loss.backward();
    if (clipNorm > (float)0.0) opt.clip_grad_norm(clipNorm);
    opt.step();
    unsafe { return loss->data[0]; }
}

} // namespace std
