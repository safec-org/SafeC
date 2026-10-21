#pragma once
// SafeC Standard Library — one training interface over every backend.
//
// std::ml trains the same models on the CPU (nn.h, trainer.h, autoencoder.h,
// convnet.h, gan.h, rl.h: autograd) and on an AMD GPU (nn_rocm.h: device-
// resident nets, graph replay, optimizer state on the GPU). The traits below
// are what a backend's trainer provides, per role; the generic loops are
// written once against them. Traits are structural: a type conforms by
// having the methods, and each call is resolved at compile time (no
// dispatch at run time).
//
//   Trainable      a model with its optimizer, trained on a Dataset's
//                  batches: nets (Mlp, ConvNet) on a loss, autoencoders,
//                  VAEs. train_fit is trainer_fit over it.
//   GanTrainable   a GAN (Gan, Dcgan) and its two optimizers; train_gan.
//   ValueAgent     a value-based RL agent (Dqn).
//   PolicyLearner  a policy-gradient agent (PolicyAgent: REINFORCE, A2C, PPO).
//
// The conforming types: train_cpu.h's CpuTrainer, CpuGanTrainer,
// CpuDqnAgent, CpuPgAgent and nn_rocm.h's RocmTrainer, RocmGanTrainer,
// RocmDqnAgent, RocmPgAgent. Each owns its host model by value in a field
// named 'host' (moved in by its constructor: the model's tensors are shared
// with the struct you passed, its other fields are not, so use 'host' from
// then on). sync() makes 'host' current: a no-op on the CPU, a download from
// a device. free() releases what the trainer allocated (optimizer state,
// device buffers), not the host model.
//
// train.h picks the backend from the build's features and names the types
// and constructors without a backend prefix, so one program trains on either:
//
//   #include <std/ml/train.h>
//   struct Optimizer cfgOpt = optimizer_new(OPT_ADAM, (float)0.01);
//   MlpTrainer t = mlp_trainer_new(net, TRAIN_LOSS_CE, &cfgOpt, &opts);
//   struct TrainReport rep = train_fit(&t, &data, &cfg);   // host synced
//   ... t.host is the trained Mlp ...
//   t.free();
#include <std/ml/tensor.h>
#include <std/ml/nn.h>
#include <std/ml/datasets.h>
#include <std/ml/trainer.h>
#include <std/ml/gan.h>

namespace std {

// Losses for a net trainer (mlp_trainer_new, convnet_trainer_new).
#define TRAIN_LOSS_CE  0   // softmax cross-entropy on class indices (tensor_cross_entropy_loss)
#define TRAIN_LOSS_MSE 1   // mean squared error (tensor_mse_loss)
#define TRAIN_LOSS_BCE 2   // binary cross-entropy on logits (tensor_bce_logits_loss)

trait Trainable {
    // Prepares a run over 'data' (its batches, shuffling, gradient clipping
    // from cfg; a device uploads the set once). 0 when it can't, e.g. a batch
    // larger than the trainer's maxBatch.
    int   fit_begin(const &Dataset data, const &TrainConfig cfg);
    // One pass over the data in cfg.batchSize batches, one optimizer step each.
    struct EpochLoss fit_epoch();
    void  fit_end();
    // One optimizer step on a batch given as host tensors (x [rows, features],
    // y [rows, targets]); returns its loss. On a device this uploads the batch
    // and waits for the loss: for hand-written loops, not speed.
    float train_step(const &Tensor x, const &Tensor y);
    float learning_rate();
    void  set_learning_rate(float lr);
    int   sync();
    void  free();
}

// trainer_fit for any Trainable: the same batches, epochs, lr decay and
// report. Leaves the host model synced. A run that fails reports steps == 0.
generic<M: Trainable>
struct TrainReport train_fit(&M m, const &Dataset data, const &TrainConfig cfg) {
    struct TrainReport rep = train_report_new_();
    if (!m.fit_begin(data, cfg)) return rep;
    int ok = 1;
    unsigned long e = 0UL;
    while (ok && e < cfg.epochs) {
        struct EpochLoss el = m.fit_epoch();
        ok = el.ok;
        if (ok) train_report_epoch_(&rep, e, el.loss, el.steps);
        m.set_learning_rate(m.learning_rate() * cfg.lrDecay);
        e = e + 1UL;
    }
    m.fit_end();
    ok = m.sync() && ok;
    if (!ok) rep.steps = 0UL;
    return rep;
}

trait GanTrainable {
    // gan_train_step: cfg.nCritic critic steps and one generator step on
    // real [rows, dataDim]; labels (rows class indices) for a conditional
    // GAN, else null.
    struct GanLosses train_step(const &Tensor real, ?&Tensor labels);
    // gan_train: epochs over data in shuffled batches. The last step's losses.
    struct GanLosses train(const &Dataset data, unsigned long epochs, unsigned long batchSize);
    int  sync();
    void free();
}

// gan_train for any GanTrainable, leaving the host model synced.
generic<G: GanTrainable>
struct GanLosses train_gan(&G g, const &Dataset data, unsigned long epochs, unsigned long batchSize) {
    struct GanLosses l = g.train(data, epochs, batchSize);
    g.sync();
    return l;
}

trait ValueAgent {
    float epsilon();
    // explore 1: epsilon-greedy; 0: greedy.
    unsigned long act(const &Tensor obs, int explore);
    // Stores the transition and trains when due: 1 when a gradient step ran,
    // 0 when none was due, -1 on failure.
    int   observe(const &Tensor obs, unsigned long action, float reward, const &Tensor next, int terminal);
    float last_loss();                    // the last gradient step's loss
    unsigned long updates();              // gradient steps taken
    int   sync();
    void  free();
}

trait PolicyLearner {
    // sample 1: draw from the policy; 0: the most likely action.
    unsigned long act(const &Tensor obs, int sample);
    void  record(const &Tensor obs, unsigned long action, float reward, const &Tensor next, int terminated,
                 int truncated);
    // Trains on the recorded rollout and clears it; the last loss (-1 on failure).
    float update();
    int   sync();
    void  free();
}

} // namespace std
