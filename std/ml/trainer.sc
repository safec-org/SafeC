// SafeC Standard Library — the training loop (see trainer.h).
#pragma once
#include <std/ml/trainer.h>
#include <std/ml/tensor.h>
#include <std/ml/nn.h>

namespace std {

struct TrainConfig train_config_default() {
    struct TrainConfig c;
    c.epochs = 10UL;
    c.batchSize = 32UL;
    c.seed = 1UL;
    c.shuffle = 1;
    c.clipNorm = (float)0.0;
    c.lrDecay = (float)1.0;
    return c;
}

float trainer_step_(&Optimizer opt, &Tensor loss, float clipNorm) {
    optimizer_zero_grad(opt);
    tensor_backward(loss);
    if (clipNorm > (float)0.0) optimizer_clip_grad_norm(opt, clipNorm);
    optimizer_step(opt);
    unsafe { return loss->data[0]; }
}

} // namespace std
