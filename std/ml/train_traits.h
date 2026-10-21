#pragma once
// SafeC Standard Library — one training interface over every backend.
//
// std::ml trains the same models on the CPU (nn.h, trainer.h, autoencoder.h,
// convnet.h, gan.h, rl.h: Autograd::get), on an AMD GPU (nn_rocm.h: device-
// resident nets, graph replay, optimizer state on the GPU) and, for fully
// connected nets, on an Apple GPU (nn_mps.h). The traits below
// are what a backend's trainer provides; the generic loops are written once
// against them. Traits are structural: a type conforms by having the
// methods, and each call is resolved at compile time (no dispatch at run
// time). Each role is composed from small capabilities, and a generic
// function asks only for the ones it uses (train_fit: EpochFit + HostSynced):
//
//   HostSynced, EpochFit, BatchStep, Actor   the capabilities
//   Trainable      EpochFit + BatchStep + HostSynced: a model with its
//                  optimizer, trained on a Dataset's batches (nets on a
//                  loss, autoencoders, VAEs). train_fit is trainer_fit
//                  over it.
//   GanTrainable   HostSynced + a GAN's train / train_step; train_gan.
//   ValueAgent     Actor + HostSynced + observe ...: DQN.
//   PolicyLearner  Actor + HostSynced + record / update: REINFORCE, A2C, PPO.
//
// The conforming types: train_cpu.h's CpuTrainer, CpuGanTrainer,
// CpuDqnAgent, CpuPgAgent, nn_rocm.h's RocmTrainer, RocmGanTrainer,
// RocmDqnAgent, RocmPgAgent and nn_mps.h's MpsTrainer. Each owns its host model by value in a field
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
//   struct Optimizer cfgOpt = Optimizer::new(OPT_ADAM, (float)0.01);
//   MlpTrainer t = mlp_trainer_new(net, TRAIN_LOSS_CE, &cfgOpt, &opts);
//   struct TrainReport rep = train_fit(&t, &data, &cfg);   // host synced
//   ... t.host is the trained Mlp ...
//   t.free();
#include <std/ml/tensor.h>
#include <std/ml/nn.h>
#include <std/ml/datasets.h>
#include <std/ml/trainer.h>
#include <std/ml/gan.h>
#include <std/ml/module.h>

namespace std {

// The capabilities the roles are composed from.
trait HostSynced {
    // Makes the trainer's 'host' model current (a no-op on the CPU, a
    // download from a device). 1 on success.
    int  sync();
    // Releases what the trainer allocated, not the host model.
    void free();
}

trait EpochFit {
    // Prepares a run over 'data' (its batches, shuffling, gradient clipping
    // from cfg; a device uploads the set once). 0 when it can't, e.g. a batch
    // larger than the trainer's maxBatch.
    int   fit_begin(const &Dataset data, const &TrainConfig cfg);
    // One pass over the data in cfg.batchSize batches, one optimizer step each.
    struct EpochLoss fit_epoch();
    void  fit_end();
    float learning_rate();
    void  set_learning_rate(float lr);
}

trait BatchStep {
    // One optimizer step on a batch given as host tensors (x [rows, features],
    // y [rows, targets]); returns its loss. On a device this uploads the batch
    // and waits for the loss: for hand-written loops, not speed.
    float train_step(const &FloatTensor x, const &FloatTensor y);
}

trait Actor {
    // An action for one observation (explore / sample 1: the agent's
    // exploration; 0: the best action).
    unsigned long act(const &FloatTensor obs, int explore);
}

// ── Roles ────────────────────────────────────────────────────────────────────

trait Trainable : EpochFit + BatchStep + HostSynced {}

// trainer_fit for anything that fits by epochs and syncs its host: the same
// batches, epochs, lr decay and report. Leaves the host model synced. A run
// that fails reports steps == 0.
generic<M: EpochFit + HostSynced>
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

// A GAN trainer: Gan's training entry points (gan.h's GanModel has the same
// two) plus a host to sync.
trait GanTrainable : HostSynced {
    // Gan::train_step: cfg.nCritic critic steps and one generator step on
    // real [rows, dataDim]; labels (rows class indices) for a conditional
    // GAN, else null.
    struct GanLosses train_step(const &FloatTensor real, ?&FloatTensor labels);
    // Gan::train: epochs over data in shuffled batches. The last step's losses.
    struct GanLosses train(const &Dataset data, unsigned long epochs, unsigned long batchSize);
}

// Gan::train for any GanTrainable, leaving the host model synced.
generic<G: GanTrainable>
struct GanLosses train_gan(&G g, const &Dataset data, unsigned long epochs, unsigned long batchSize) {
    struct GanLosses l = g.train(data, epochs, batchSize);
    g.sync();
    return l;
}

trait ValueAgent : Actor + HostSynced {
    float epsilon();
    // Stores the transition and trains when due: 1 when a gradient step ran,
    // 0 when none was due, -1 on failure.
    int   observe(const &FloatTensor obs, unsigned long action, float reward, const &FloatTensor next, int terminal);
    float last_loss();                    // the last gradient step's loss
    unsigned long updates();              // gradient steps taken
}

trait PolicyLearner : Actor + HostSynced {
    void  record(const &FloatTensor obs, unsigned long action, float reward, const &FloatTensor next, int terminated,
                 int truncated);
    // Trains on the recorded rollout and clears it; the last loss (-1 on failure).
    float update();
}

} // namespace std
