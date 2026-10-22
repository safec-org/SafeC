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
    c.device = DEVICE_CPU;
    c.precision = DEVICE_PRECISION_KEEP;
    return c;
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

void trainer_place_(&Optimizer opt, int device) {
    if (device == DEVICE_CPU) return;
    int d = device == DEVICE_AUTO ? device_default() : device;
    unsigned long i = 0UL;
    while (i < opt.params.length()) { (*opt.params.at(i)).place(d); i = i + 1UL; }
}

static int trainer_dev_(int device) { return device == DEVICE_AUTO ? device_default() : device; }

int trainer_precision_begin_(const &TrainConfig cfg) {
    int d = trainer_dev_(cfg.device);
    int prev = device_precision(d);
    if (d != DEVICE_CPU && cfg.precision != DEVICE_PRECISION_KEEP) device_set_precision(d, cfg.precision);
    return prev;
}

void trainer_precision_end_(const &TrainConfig cfg, int previous) {
    int d = trainer_dev_(cfg.device);
    if (d != DEVICE_CPU) device_set_precision(d, previous);
}

void trainer_place_data_(&DataLoader dl, int device) {
    if (device == DEVICE_CPU) return;
    dl.to(device == DEVICE_AUTO ? device_default() : device);
}

float trainer_step_(&Optimizer opt, &FloatTensor loss, float clipNorm) {
    opt.zero_grad();
    loss.backward();
    if (clipNorm > (float)0.0) opt.clip_grad_norm(clipNorm);
    opt.step();
    unsafe { tensor_host_((void*)loss); return loss->data[0]; }
}

} // namespace std
