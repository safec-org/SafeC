#pragma once
// SafeC Standard Library — backend-neutral training, the backend chosen at
// build time.
//
// The build's features pick which concrete trainer each name below is (see
// train_traits.h for the interface they share):
//
//   (none)   the CPU: train_cpu.h's CpuTrainer, CpuGanTrainer, ...
//   rocm     an AMD GPU: nn_rocm.h's RocmTrainer, RocmGanTrainer, ... (link
//            -lMIOpen -lamdhip64 -lhiprtc -lrocblas)
//   mps      an Apple GPU (Metal): nn_mps.h's MpsTrainer for MlpTrainer.
//            nn_mps.h has fully connected nets only, so the other names
//            (ConvNetTrainer, the autoencoder and VAE trainers, GanTrainer,
//            DcganTrainer, DqnLearner, PgLearner) stay the CPU's and train
//            there; an MlpTrainer built with TrainerOptions.precision set
//            fails to build (ok = 0: no 16-bit activations on Metal yet).
//            Link -framework Metal -framework Foundation -lobjc. rocm wins
//            when both are enabled.
//   cuda     not available yet: there is no device-resident CUDA training
//            stack (gpu_cuda.h / tensor_cuda.h are per-op), so the build
//            stops with an error rather than silently training on the CPU
//            (unless rocm or mps is enabled too, which wins). A CUDA backend
//            adds types satisfying train_traits.h's traits and a branch here.
//
// In Package.toml:
//
//   [features]
//   rocm = []        # safeguard build --features rocm
//   mps = []         # safeguard build --features mps
//
// which reach the sources as SAFEC_FEATURE_ROCM / SAFEC_FEATURE_MPS. Everything is resolved
// at compile time: each name is a typedef and each constructor a static
// function forwarding to the backend's, and the generic loops
// (train_fit, train_gan) are monomorphized for the chosen type.
//
//   struct Optimizer adam = Optimizer::new(OPT_ADAM, (float)0.01);    // kind + hyperparameters
//   struct TrainerOptions to = TrainerOptions::default();
//   MlpTrainer t = mlp_trainer_new(net, TRAIN_LOSS_CE, &adam, &to);
//   struct TrainReport rep = train_fit(&t, &data, &cfg);
//   float acc = trainer_accuracy(&t.host, logits_fn, &test, 64UL);   // the trained Mlp
//   t.free();
//   t.host.free();
//
// The names: MlpTrainer, ConvNetTrainer, VaeTrainer, ConvVaeTrainer and
// ConvAeTrainer (Trainable), GanTrainer and DcganTrainer (GanTrainable),
// DqnLearner (ValueAgent), PgLearner (PolicyLearner). On the CPU any other
// Module trains the same way with CpuTrainer::new, and any LossModel
// with CpuModelTrainer::new. ML_TRAIN_BACKEND is
// the backend's name as a string.
#include <std/ml/train_traits.h>

#if defined(SAFEC_FEATURE_ROCM)
#include <std/ml/nn_rocm.h>
#elif defined(SAFEC_FEATURE_MPS)
#include <std/ml/nn_mps.h>
#include <std/ml/train_cpu.h>
#elif defined(SAFEC_FEATURE_CUDA)
#error "std/ml/train.h: no CUDA training backend yet (build with the rocm or mps feature, or none)"
#else
#include <std/ml/train_cpu.h>
#endif

namespace std {

#if defined(SAFEC_FEATURE_ROCM)

#define ML_TRAIN_BACKEND "rocm"
typedef struct RocmTrainer<struct Mlp, struct RocmNet> MlpTrainer;
typedef struct RocmTrainer<struct ConvNet, struct RocmNet> ConvNetTrainer;
typedef struct RocmTrainer<struct Vae, struct RocmVae> VaeTrainer;
typedef struct RocmTrainer<struct ConvVae, struct RocmVae> ConvVaeTrainer;
typedef struct RocmTrainer<struct ConvAutoencoder, struct RocmConvAe> ConvAeTrainer;
typedef struct RocmGanTrainer<struct Gan> GanTrainer;
typedef struct RocmGanTrainer<struct Dcgan> DcganTrainer;
typedef struct RocmDqnAgent DqnLearner;
typedef struct RocmPgAgent PgLearner;

static MlpTrainer mlp_trainer_new(struct Mlp m, int loss, const &Optimizer opt, const &TrainerOptions o) {
    return rocm_mlp_trainer_new(m, loss, opt, o);
}
static ConvNetTrainer convnet_trainer_new(struct ConvNet n, int loss, const &Optimizer opt, const &TrainerOptions o) {
    return rocm_convnet_trainer_new(n, loss, opt, o);
}
static VaeTrainer vae_trainer_new(struct Vae v, const &Optimizer opt, const &TrainerOptions o) {
    return rocm_vae_trainer_new(v, opt, o);
}
static ConvVaeTrainer conv_vae_trainer_new(struct ConvVae v, const &Optimizer opt, const &TrainerOptions o) {
    return rocm_conv_vae_trainer_new(v, opt, o);
}
static ConvAeTrainer conv_ae_trainer_new(struct ConvAutoencoder ae, const &Optimizer opt, const &TrainerOptions o) {
    return rocm_conv_ae_trainer_new(ae, opt, o);
}
static GanTrainer gan_trainer_new(struct Gan g, const &TrainerOptions o) { return rocm_gan_trainer_new(g, o); }
static DcganTrainer dcgan_trainer_new(struct Dcgan g, const &TrainerOptions o) { return rocm_dcgan_trainer_new(g, o); }
static DqnLearner dqn_learner_new(struct Dqn d) { return RocmDqnAgent::new(d); }
// o.maxBatch: the longest rollout PolicyAgent::update will see.
static PgLearner pg_learner_new(struct PolicyAgent a, const &TrainerOptions o) { return RocmPgAgent::new(a, o); }

#elif defined(SAFEC_FEATURE_MPS)

// Fully connected nets on the GPU; everything else on the CPU (see the top).
#define ML_TRAIN_BACKEND "mps"
typedef struct MpsTrainer<struct Mlp, struct MpsNet> MlpTrainer;
typedef struct CpuTrainer<struct ConvNet> ConvNetTrainer;
typedef struct CpuModelTrainer<struct Vae> VaeTrainer;
typedef struct CpuModelTrainer<struct ConvVae> ConvVaeTrainer;
typedef struct CpuModelTrainer<struct ConvAutoencoder> ConvAeTrainer;
typedef struct CpuGanTrainer<struct Gan> GanTrainer;
typedef struct CpuGanTrainer<struct Dcgan> DcganTrainer;
typedef struct CpuDqnAgent DqnLearner;
typedef struct CpuPgAgent PgLearner;

static MlpTrainer mlp_trainer_new(struct Mlp m, int loss, const &Optimizer opt, const &TrainerOptions o) {
    return mps_mlp_trainer_new(m, loss, opt, o);
}
static ConvNetTrainer convnet_trainer_new(struct ConvNet n, int loss, const &Optimizer opt, const &TrainerOptions o) {
    return CpuTrainer::new(n, loss, opt);
}
static VaeTrainer vae_trainer_new(struct Vae v, const &Optimizer opt, const &TrainerOptions o) {
    return CpuModelTrainer::new(v, opt);
}
static ConvVaeTrainer conv_vae_trainer_new(struct ConvVae v, const &Optimizer opt, const &TrainerOptions o) {
    return CpuModelTrainer::new(v, opt);
}
static ConvAeTrainer conv_ae_trainer_new(struct ConvAutoencoder ae, const &Optimizer opt, const &TrainerOptions o) {
    return CpuModelTrainer::new(ae, opt);
}
static GanTrainer gan_trainer_new(struct Gan g, const &TrainerOptions o) { return CpuGanTrainer::new(g); }
static DcganTrainer dcgan_trainer_new(struct Dcgan g, const &TrainerOptions o) { return CpuGanTrainer::new(g); }
static DqnLearner dqn_learner_new(struct Dqn d) { return CpuDqnAgent::new(d); }
static PgLearner pg_learner_new(struct PolicyAgent a, const &TrainerOptions o) { return CpuPgAgent::new(a); }

#else

#define ML_TRAIN_BACKEND "cpu"
typedef struct CpuTrainer<struct Mlp> MlpTrainer;
typedef struct CpuTrainer<struct ConvNet> ConvNetTrainer;
typedef struct CpuModelTrainer<struct Vae> VaeTrainer;
typedef struct CpuModelTrainer<struct ConvVae> ConvVaeTrainer;
typedef struct CpuModelTrainer<struct ConvAutoencoder> ConvAeTrainer;
typedef struct CpuGanTrainer<struct Gan> GanTrainer;
typedef struct CpuGanTrainer<struct Dcgan> DcganTrainer;
typedef struct CpuDqnAgent DqnLearner;
typedef struct CpuPgAgent PgLearner;

static MlpTrainer mlp_trainer_new(struct Mlp m, int loss, const &Optimizer opt, const &TrainerOptions o) {
    return CpuTrainer::new(m, loss, opt);
}
static ConvNetTrainer convnet_trainer_new(struct ConvNet n, int loss, const &Optimizer opt, const &TrainerOptions o) {
    return CpuTrainer::new(n, loss, opt);
}
static VaeTrainer vae_trainer_new(struct Vae v, const &Optimizer opt, const &TrainerOptions o) {
    return CpuModelTrainer::new(v, opt);
}
static ConvVaeTrainer conv_vae_trainer_new(struct ConvVae v, const &Optimizer opt, const &TrainerOptions o) {
    return CpuModelTrainer::new(v, opt);
}
static ConvAeTrainer conv_ae_trainer_new(struct ConvAutoencoder ae, const &Optimizer opt, const &TrainerOptions o) {
    return CpuModelTrainer::new(ae, opt);
}
static GanTrainer gan_trainer_new(struct Gan g, const &TrainerOptions o) { return CpuGanTrainer::new(g); }
static DcganTrainer dcgan_trainer_new(struct Dcgan g, const &TrainerOptions o) { return CpuGanTrainer::new(g); }
static DqnLearner dqn_learner_new(struct Dqn d) { return CpuDqnAgent::new(d); }
// o.maxBatch: the longest rollout PolicyAgent::update will see (a device's limit).
static PgLearner pg_learner_new(struct PolicyAgent a, const &TrainerOptions o) { return CpuPgAgent::new(a); }

#endif

} // namespace std
