#pragma once
// SafeC Standard Library — device-resident training on Apple GPUs (Metal).
//
// nn_rocm.h's training stack on gpu_mps.h's Metal backend, for the models it
// covers so far: fully connected nets (nn.h's Mlp). Everything a step
// touches — parameters, activations, gradients, optimizer state, the
// dataset and the running loss — lives in GPU buffers (memory the CPU shares
// on Apple Silicon, but nothing reads it during an epoch):
//
//   * MpsNet — a stack of fully connected layers with an activation (any
//     ACT_*) after each and a hand-written backward pass, built from an Mlp
//     or layer by layer. Matmuls are gpu_mps.h's mps_gemm (its
//     simdgroup-matrix kernels); bias, activations and their gradients are
//     kernels of gpu_mps_kernels.metal.
//   * Losses that write their gradient and add to a device loss sum:
//     softmax cross-entropy on class indices, binary cross-entropy on
//     logits, MSE.
//   * MpsOptimizer — nn.h's optimizers (OPT_SGD ... OPT_LION) with the same
//     arithmetic (the bias corrections are computed on the host as
//     Optimizer::step does), with optional global gradient-norm clipping
//     and weight clamping.
//   * mps_trainer_fit — trainer.h's trainer_fit on the GPU: the dataset is
//     uploaded once, each epoch writes only its shuffled row order (the
//     same order trainer_fit's DataLoader uses for that seed), batches are
//     gathered on the GPU, and the epoch's steps are encoded into one batch
//     (gpu_mps.h's mps_batch_begin / end), submitted every MPS_TRAIN_CHUNK
//     steps so the GPU runs while the next steps are encoded. The loss is
//     read back once per epoch.
//   * MpsTrainer — the above behind train_traits.h's Trainable, so
//     train_fit and code written against the traits run here or on the CPU
//     (train.h picks the backend from the build's mps feature).
//
// Not here (yet): nn_rocm.h's convolution, pooling, norm, resize, dropout
// and skip layers, RocmVae / RocmConvAe, RocmGan, the RL agents and 16-bit
// activations. Under the mps feature train.h trains those models on the CPU.
//
// A step is about 20 dispatches for a 3-layer net (gemm and bias each way
// per layer, the loss, the gather, one optimizer dispatch per parameter
// tensor), each costing ~5 us on an M1 Pro however small its work, so small
// nets are bound by them: 2-64-64-2 at batch 64 trains 3x faster on the CPU
// (nn.h). The GPU pays off from mid-sized layers: 784-512-512-10 at batch
// 256 trains 2.2x faster than the CPU.
//
//   struct Mlp net = Mlp::new(784UL, 256UL, 2UL, 10UL, ACT_RELU, ACT_NONE, &rng);
//   struct MpsNet dnet = MpsNet::from_mlp(&net, 128UL);
//   struct MpsOptimizer opt = MpsOptimizer::new(OPT_ADAM, (float)0.001);
//   dnet.add_params(&opt);
//   struct TrainReport rep = mps_trainer_fit(&dnet, mps_net_ce_step, &opt, &train, &cfg);
//   dnet.download(&net.params);   // back into the host Mlp
//
// Layout. Every buffer is row-major fp32 with one row per sample.
//
// Every op is encoded onto the open batch (gpu_mps.h's mps_batch_begin;
// results valid after mps_batch_end) or, outside one, runs in a batch of its
// own and waits — a step function or MpsNet::forward / backward called
// outside a batch is one such batch, not one per op. They return 1 when
// encoded, 0 on a bad shape or when there is no Metal device. Gradients are
// overwritten (not accumulated) by each backward pass. Link with -framework
// Metal -framework Foundation -lobjc.
#include <std/ml/gpu_mps.h>
#include <std/ml/nn.h>
#include <std/ml/datasets.h>
#include <std/ml/trainer.h>
#include <std/ml/train_traits.h>
#include <std/collections/vec.h>

namespace std {

// ── Layers and networks ──────────────────────────────────────────────────────

#define MPS_LAYER_LINEAR 0

struct MpsLayer {
    int kind;                 // MPS_LAYER_*
    int act;                  // ACT_* (nn.h) after the layer
    struct MpsBuf w;          // [in, out]
    struct MpsBuf b;          // [out]
    struct MpsBuf gw;         // gradients, written by each backward pass
    struct MpsBuf gb;
    unsigned long inC;        // in features
    unsigned long outC;       // out features
    struct MpsBuf z;          // pre-activation, only for GELU / SiLU (their gradient needs it)
    struct MpsBuf h;          // output [maxBatch, out]
    struct MpsBuf dh;         // gradient of the output (made the pre-activation's in place)
};

struct MpsNet {
    struct Vec<struct MpsLayer> layers;
    unsigned long maxBatch;   // every buffer is sized for this many rows
    int ok;                   // 0 once an upload, allocation or shape check failed

    // Appends a fully connected layer, uploading w [in, out] and b [out] (an
    // Mlp's shapes); in must match the previous layer's out. Return 1, or 0
    // (and ok = 0).
    int add_linear(const &FloatTensor w, const &FloatTensor b, int act);
    unsigned long in_dim() const;
    unsigned long out_dim() const;
    // Every layer's weight and bias in order (W0, b0, W1, b1, ...: an Mlp's
    // params), from or to host tensors of the same sizes. Download waits for
    // nothing: call it after the batch that trained the net has ended.
    int download(&Vec<&FloatTensor> params) const;
    int upload(const &Vec<&FloatTensor> params);
    void free();
    // Forward pass on x [batch, inDim]. Returns the output [batch, outDim]
    // (the net's own buffer, valid until the next forward), invalid on failure.
    struct MpsBuf forward(struct MpsBuf x, unsigned long batch);
    // Where a loss writes d(loss)/d(output) before MpsNet::backward.
    struct MpsBuf grad_output() const;
    // Backward pass after forward(x, batch): every layer's gw / gb, and
    // d(loss)/d(x) into dx when dx is valid.
    int backward(struct MpsBuf x, unsigned long batch, struct MpsBuf dx);
    // Every parameter and its gradient into o (MpsOptimizer::add).
    int add_params(&MpsOptimizer o) const;
    // An Mlp's parameters from its device net (MpsNet::download into m.params).
    int download_mlp(&Mlp m) const;

    static struct MpsNet new(unsigned long maxBatch);
    // An Mlp's layers (its act / outAct), for batches of up to maxBatch rows.
    static struct MpsNet from_mlp(const &Mlp m, unsigned long maxBatch);
};

// ── Losses ───────────────────────────────────────────────────────────────────
// Each writes d(loss)/d(input) into d and ADDS the loss to loss1[0] (zero it
// with MpsBuf::write to start a new sum), so a run of steps sums its losses
// on the GPU and reads them back once. 'scale' multiplies both.

// Softmax cross-entropy of logits [rows, classes] against class indices
// (rows floats), averaged over rows (tensor_cross_entropy_loss).
int mps_cross_entropy(struct MpsBuf logits, struct MpsBuf labels, struct MpsBuf d, struct MpsBuf loss1,
                      unsigned long rows, unsigned long classes);
// scale * mean binary cross-entropy of sigmoid(logits) against targets in
// [0, 1] over n values (tensor_bce_logits_loss).
int mps_bce_logits(struct MpsBuf logits, struct MpsBuf target, struct MpsBuf d, struct MpsBuf loss1,
                   unsigned long n, float scale);
// scale * mean squared error over n values (tensor_mse_loss).
int mps_mse(struct MpsBuf pred, struct MpsBuf target, struct MpsBuf d, struct MpsBuf loss1,
            unsigned long n, float scale);

// Training steps for mps_trainer_fit: forward, loss, backward on one batch.
int mps_net_ce_step(&MpsNet n, struct MpsBuf x, struct MpsBuf y, unsigned long rows, struct MpsBuf loss1);
int mps_net_mse_step(&MpsNet n, struct MpsBuf x, struct MpsBuf y, unsigned long rows, struct MpsBuf loss1);
int mps_net_bce_step(&MpsNet n, struct MpsBuf x, struct MpsBuf y, unsigned long rows, struct MpsBuf loss1);
// Autoencoders: the target is x itself (y is ignored).
int mps_net_recon_bce_step(&MpsNet n, struct MpsBuf x, struct MpsBuf y, unsigned long rows, struct MpsBuf loss1);
int mps_net_recon_mse_step(&MpsNet n, struct MpsBuf x, struct MpsBuf y, unsigned long rows, struct MpsBuf loss1);

// ── Optimizer ────────────────────────────────────────────────────────────────
// nn.h's Optimizer on the GPU. Change lr, betas, weightDecay or clipNorm
// between steps like the host fields of an Optimizer: each step is encoded
// with the values the fields hold when MpsOptimizer::step runs (so steps
// queued in one batch each keep their own lr), and t counts the steps
// encoded so far.
struct MpsOptimizer {
    struct Vec<struct MpsBuf> w;    // parameters
    struct Vec<struct MpsBuf> g;    // their gradients
    struct Vec<struct MpsBuf> m;    // first moment / velocity
    struct Vec<struct MpsBuf> v;    // second moment
    int   kind;                     // OPT_*
    float lr;
    float momentum;
    float beta1;
    float beta2;
    float eps;
    float weightDecay;
    float clipNorm;                 // > 0: scale the gradients to this global L2 norm at most
    float clampWeights;             // > 0: clamp every weight to [-c, c] after each step (WGAN's critic)
    unsigned long t;                // steps taken
    struct MpsBuf sumsq;            // clipping: each parameter's sum of squared gradients
    int   ok;

    int add(struct MpsBuf w, struct MpsBuf g);
    // One step over every parameter: a dispatch per parameter tensor, plus
    // as many for the gradient norm when clipping. Queued.
    int step();
    void free();   // frees the state, not the parameters

    // Same defaults as Optimizer::new.
    static struct MpsOptimizer new(int kind, float lr);
    // An empty optimizer with o's kind and hyperparameters (not its state).
    static struct MpsOptimizer from(const &Optimizer o);
};

// ── The training loop ────────────────────────────────────────────────────────
// A dataset in GPU memory and its batch cursor (mps_trainer_fit's
// internals, usable for hand-written loops). xb / yb hold the current batch.
struct MpsBatches {
    struct MpsBuf x;          // [n, features]
    struct MpsBuf y;          // [n, targets]
    struct MpsBuf order;      // this epoch's row order (32-bit indices)
    struct MpsBuf xb;
    struct MpsBuf yb;
    struct MpsBuf loss;       // [1] this epoch's loss sum
    unsigned long n;
    unsigned long features;
    unsigned long targets;
    unsigned long batchSize;
    unsigned long pos;        // first row of the next batch in 'order'
    int   shuffle;
    struct Rng rng;
    &heap unsigned long perm;
    &heap unsigned int permu;
    int   ok;

    // Starts an epoch: writes the row order, rewinds the cursor, zeroes the
    // loss. Call with no batch reading them in flight (after mps_batch_end).
    int epoch();
    // Queued: xb, yb = the next 'rows' rows, and the cursor moves past them.
    int gather(unsigned long rows);
    float loss_sum();                 // the epoch's loss sum (after its batch ended)
    void free();

    static struct MpsBatches new(const &Dataset d, unsigned long batchSize, int shuffle, unsigned long seed);
};

// Steps per command buffer in mps_trainer_fit's epoch batch: each
// MPS_TRAIN_CHUNK steps are submitted (mps_batch_submit) so the GPU works on
// them while the CPU encodes the next ones.
#ifndef MPS_TRAIN_CHUNK
#define MPS_TRAIN_CHUNK 4UL
#endif

// Internal: one queued step (gather, model step, optimizer).
generic<T>
int mps_trainer_step_(&T model, fn int(&T, struct MpsBuf, struct MpsBuf, unsigned long, struct MpsBuf) stepFn,
                      &MpsOptimizer opt, &MpsBatches bt, unsigned long rows) {
    if (!bt.gather(rows)) return 0;
    if (!stepFn(model, bt.xb, bt.yb, rows, bt.loss)) return 0;
    return opt.step();
}

// Internal: one epoch of mps_trainer_fit, in one batch it waits for at the
// end (the loss sum is then in bt.loss). Sets *batches; 0 on a GPU failure.
generic<T>
int mps_trainer_epoch_(&T model, fn int(&T, struct MpsBuf, struct MpsBuf, unsigned long, struct MpsBuf) stepFn,
                       &MpsOptimizer opt, &MpsBatches bt, &unsigned long batches) {
    unsigned long n = 0UL;
    int began = 0;
    int ok = !mps_batch_is_active() && bt.epoch();
    if (ok) {
        mps_batch_begin();
        began = mps_batch_is_active();
    }
    ok = ok && began;
    while (ok && bt.pos < bt.n) {
        unsigned long rows = bt.n - bt.pos;
        if (rows > bt.batchSize) rows = bt.batchSize;
        ok = mps_trainer_step_(model, stepFn, opt, bt, rows);
        n = n + 1UL;
        if (n % MPS_TRAIN_CHUNK == 0UL) mps_batch_submit();
    }
    if (began) mps_batch_end();
    *batches = n;
    return ok;
}

// trainer_fit on the GPU. stepFn queues forward, loss and backward for
// 'rows' rows of xb / yb and adds the batch's loss to loss1[0]
// (mps_net_ce_step, ...). Same batches, epochs, lr decay and report as
// trainer_fit; clipNorm sets opt.clipNorm. A run that fails on the GPU
// reports steps == 0. MpsTrainer (below) is the same loop behind
// train_traits.h's Trainable.
generic<T>
struct TrainReport mps_trainer_fit(&T model,
                                   fn int(&T, struct MpsBuf, struct MpsBuf, unsigned long, struct MpsBuf) stepFn,
                                   &MpsOptimizer opt, const &Dataset data, const &TrainConfig cfg) {
    struct TrainReport rep = train_report_new_();
    struct MpsBatches bt = MpsBatches::new(data, cfg.batchSize, cfg.shuffle, cfg.seed);
    if (!bt.ok || !opt.ok) { bt.free(); return rep; }
    opt.clipNorm = cfg.clipNorm;
    int ok = 1;
    unsigned long steps = 0UL;
    unsigned long e = 0UL;
    while (ok && e < cfg.epochs) {
        unsigned long batches = 0UL;
        ok = mps_trainer_epoch_(model, stepFn, opt, &bt, &batches);
        float sum = bt.loss_sum();
        train_report_epoch_(&rep, e, batches > 0UL ? sum / (float)batches : (float)0.0, batches);
        steps = steps + batches;
        opt.lr = opt.lr * cfg.lrDecay;
        e = e + 1UL;
    }
    rep.steps = ok ? steps : 0UL;
    bt.free();
    return rep;
}

// ── The training traits (train_traits.h) ─────────────────────────────────────
// The device models above behind the backend-neutral interface; train.h
// names them without the Mps prefix when the build enables the mps feature.
// Each owns its host model in 'host' and a device copy in 'dev'; sync()
// downloads 'dev' into 'host' (train_fit does so at the end). The GPU-side
// optimizer state and the device-resident dataset are those of
// mps_trainer_fit.

// A host model H trained as a device model D (MpsNet): Trainable. 'ok' is 0
// once building or uploading failed.
generic<H, D>
struct MpsTrainer {
    H host;
    D dev;
    struct MpsOptimizer opt;
    fn int(&D, struct MpsBuf, struct MpsBuf, unsigned long, struct MpsBuf) stepFn;
    fn int(const &D, &H) downloadFn;
    fn void(&D) freeFn;
    unsigned long maxBatch;
    struct MpsBatches bt;         // between fit_begin and fit_end
    struct MpsBuf xs;             // train_step's batch and loss
    struct MpsBuf ys;
    struct MpsBuf loss1;
    int active;
    int ok;

    int   fit_begin(const &Dataset data, const &TrainConfig cfg);
    struct EpochLoss fit_epoch();
    void  fit_end();
    float train_step(const &FloatTensor x, const &FloatTensor y);             // -1 on failure
    float learning_rate();
    void  set_learning_rate(float lr);
    int   sync();
    void  free();
};

// Internal: b holds at least n floats (reallocated when smaller).
int mps_trainer_reserve_(&MpsBuf b, unsigned long n);
// Internal: n floats from host into b, and b's first float back (no batch in flight).
int mps_trainer_write_(struct MpsBuf b, const &FloatTensor t, unsigned long n);
float mps_trainer_read1_(struct MpsBuf b);

generic<H, D>
int MpsTrainer::fit_begin(const &Dataset data, const &TrainConfig cfg) {
    self.fit_end();
    if (!self.ok || !self.opt.ok) return 0;
    self.bt = MpsBatches::new(data, cfg.batchSize, cfg.shuffle, cfg.seed);
    self.active = 1;
    if (!self.bt.ok || self.bt.batchSize > self.maxBatch) { self.fit_end(); return 0; }
    self.opt.clipNorm = cfg.clipNorm;
    return 1;
}

generic<H, D>
struct EpochLoss MpsTrainer::fit_epoch() {
    struct EpochLoss el;
    el.steps = 0UL;
    el.loss = (float)0.0;
    el.ok = 0;
    if (!self.active) return el;
    el.ok = mps_trainer_epoch_(&self.dev, self.stepFn, &self.opt, &self.bt, &el.steps);
    float sum = self.bt.loss_sum();
    if (el.steps > 0UL) el.loss = sum / (float)el.steps;
    return el;
}

generic<H, D>
void MpsTrainer::fit_end() {
    if (!self.active) return;
    self.bt.free();
    self.active = 0;
}

generic<H, D>
float MpsTrainer::train_step(const &FloatTensor x, const &FloatTensor y) {
    unsigned long rows = 0UL;
    unsigned long nx = 0UL;
    unsigned long ny = 0UL;
    unsafe { rows = x->shape[0]; nx = x->size; ny = y->size; }
    if (!self.ok || rows == 0UL || rows > self.maxBatch || mps_batch_is_active() ||
        !mps_trainer_reserve_(&self.xs, nx) || !mps_trainer_reserve_(&self.ys, ny) ||
        !mps_trainer_reserve_(&self.loss1, 1UL)) return (float)-1.0;
    float zero = (float)0.0;
    int ok = mps_trainer_write_(self.xs, x, nx) && mps_trainer_write_(self.ys, y, ny);
    unsafe { ok = ok && self.loss1.write((const &void)(const void*)&zero, 4UL); }
    int began = 0;
    if (ok) {
        mps_batch_begin();
        began = mps_batch_is_active();
    }
    ok = ok && began && self.stepFn(&self.dev, self.xs, self.ys, rows, self.loss1) && self.opt.step();
    if (began) mps_batch_end();
    return ok ? mps_trainer_read1_(self.loss1) : (float)-1.0;
}

generic<H, D>
float MpsTrainer::learning_rate() { return self.opt.lr; }

generic<H, D>
void MpsTrainer::set_learning_rate(float lr) { self.opt.lr = lr; }

generic<H, D>
int MpsTrainer::sync() { return self.ok && self.downloadFn(&self.dev, &self.host); }

generic<H, D>
void MpsTrainer::free() {
    self.fit_end();
    self.opt.free();
    self.freeFn(&self.dev);
    self.xs.free();
    self.ys.free();
    self.loss1.free();
    self.ok = 0;
}

// 'opt' supplies the optimizer's kind and hyperparameters
// (MpsOptimizer::from); o.maxBatch sizes the device buffers. loss:
// TRAIN_LOSS_* (mps_net_ce_step, _mse_step, _bce_step). There are no 16-bit
// activations here: the trainer fails to build (ok = 0) when o.precision is
// set.
struct MpsTrainer<struct Mlp, struct MpsNet> mps_mlp_trainer_new(struct Mlp m, int loss, const &Optimizer opt,
                                                                 const &TrainerOptions o);

} // namespace std
