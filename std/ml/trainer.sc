// SafeC Standard Library — the training loop (see trainer.h).
#pragma once
#include <std/ml/trainer.h>
#include <std/ml/tensor.h>
#include <std/ml/nn.h>
#include <std/ml/device.h>
#include <std/ml/device_kernels.h>

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
    unsafe { tensor_host_raw_((FloatTensor*)loss); return loss->data[0]; }
}

struct TrainLossSum trainer_loss_sum_() {
    struct TrainLossSum s;
    s.host = 0.0; s.dev = 0UL; s.device = DEVICE_CPU; s.steps = 0UL;
    return s;
}

// The device sum's value into the host's; frees it.
static void trainer_loss_fold_(&TrainLossSum s) {
    if (s.dev == 0UL) return;
    float v = (float)0.0;
    if (device_download_(s.device, (&float)&v, s.dev, 1UL)) s.host = s.host + (double)v;
    device_mem_free_(s.device, s.dev, 4UL);
    s.dev = 0UL;
}

void trainer_step_sum_(&Optimizer opt, &FloatTensor loss, float clipNorm, &TrainLossSum s) {
    opt.zero_grad();
    loss.backward();
    if (clipNorm > (float)0.0) opt.clip_grad_norm(clipNorm);
    opt.step();
    s.steps = s.steps + 1UL;
    unsafe {
        FloatTensor* t = (FloatTensor*)loss;
        int d = t->dev;
        if (d != DEVICE_CPU && (t->coh & TS_DEV) != 0 && t->dbuf != 0UL) {
            if (s.dev != 0UL && s.device != d) trainer_loss_fold_(s);
            if (s.dev == 0UL) {
                s.dev = device_mem_alloc_(d, 4UL);
                s.device = d;
                struct DevArgs z = dev_args_();
                z.p[2] = s.dev; z.u[0] = (unsigned int)EW_FILL;
                if (s.dev != 0UL && !device_launch_(d, DK_EW, 1UL, &z)) { device_mem_free_(d, s.dev, 4UL); s.dev = 0UL; }
            }
            if (s.dev != 0UL) {
                struct DevArgs a = dev_args_();
                a.p[0] = t->dbuf; a.p[2] = s.dev; a.u[0] = (unsigned int)EW_ACC;
                if (device_launch_(d, DK_EW, 1UL, &a)) return;
            }
        }
        tensor_host_raw_((FloatTensor*)t);
        s.host = s.host + (double)t->data[0];
    }
}

float trainer_loss_mean_(&TrainLossSum s) {
    trainer_loss_fold_(s);
    return s.steps > 0UL ? (float)(s.host / (double)s.steps) : (float)0.0;
}

} // namespace std
