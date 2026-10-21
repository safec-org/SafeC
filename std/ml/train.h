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
//   cuda     not available yet: there is no device-resident CUDA training
//            stack (gpu_cuda.h / tensor_cuda.h are per-op), so the build
//            stops with an error rather than silently training on the CPU
//            (unless rocm is enabled too, which wins). A CUDA backend adds
//            types satisfying train_traits.h's traits and a branch here.
//
// In Package.toml:
//
//   [features]
//   rocm = []        # safeguard build --features rocm
//
// which reaches the sources as SAFEC_FEATURE_ROCM. Everything is resolved
// at compile time: each name is a typedef and each constructor a static
// function forwarding to the backend's, and the generic loops
// (train_fit, train_gan) are monomorphized for the chosen type.
//
//   struct Optimizer adam = optimizer_new(OPT_ADAM, (float)0.01);   // kind + hyperparameters
//   struct TrainerOptions to = trainer_options_default();
//   MlpTrainer t = mlp_trainer_new(net, TRAIN_LOSS_CE, &adam, &to);
//   struct TrainReport rep = train_fit(&t, &data, &cfg);
//   float acc = trainer_accuracy(&t.host, logits_fn, &test, 64UL);   // the trained Mlp
//   t.free();
//   mlp_free(&t.host);
//
// The names: MlpTrainer, ConvNetTrainer, VaeTrainer, ConvVaeTrainer and
// ConvAeTrainer (Trainable), GanTrainer and DcganTrainer (GanTrainable),
// DqnLearner (ValueAgent), PgLearner (PolicyLearner). ML_TRAIN_BACKEND is
// the backend's name as a string.
#include <std/ml/train_traits.h>

#if defined(SAFEC_FEATURE_ROCM)
#include <std/ml/nn_rocm.h>
#elif defined(SAFEC_FEATURE_CUDA)
#error "std/ml/train.h: no CUDA training backend yet (build with the rocm feature, or neither)"
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
static DqnLearner dqn_learner_new(struct Dqn d) { return rocm_dqn_agent_new(d); }
// o.maxBatch: the longest rollout pg_update will see.
static PgLearner pg_learner_new(struct PolicyAgent a, const &TrainerOptions o) { return rocm_pg_agent_new(a, o); }

#else

#define ML_TRAIN_BACKEND "cpu"
typedef struct CpuTrainer<struct Mlp> MlpTrainer;
typedef struct CpuTrainer<struct ConvNet> ConvNetTrainer;
typedef struct CpuTrainer<struct Vae> VaeTrainer;
typedef struct CpuTrainer<struct ConvVae> ConvVaeTrainer;
typedef struct CpuTrainer<struct ConvAutoencoder> ConvAeTrainer;
typedef struct CpuGanTrainer<struct Gan> GanTrainer;
typedef struct CpuGanTrainer<struct Dcgan> DcganTrainer;
typedef struct CpuDqnAgent DqnLearner;
typedef struct CpuPgAgent PgLearner;

static MlpTrainer mlp_trainer_new(struct Mlp m, int loss, const &Optimizer opt, const &TrainerOptions o) {
    return cpu_mlp_trainer_new(m, loss, opt);
}
static ConvNetTrainer convnet_trainer_new(struct ConvNet n, int loss, const &Optimizer opt, const &TrainerOptions o) {
    return cpu_convnet_trainer_new(n, loss, opt);
}
static VaeTrainer vae_trainer_new(struct Vae v, const &Optimizer opt, const &TrainerOptions o) {
    return cpu_vae_trainer_new(v, opt);
}
static ConvVaeTrainer conv_vae_trainer_new(struct ConvVae v, const &Optimizer opt, const &TrainerOptions o) {
    return cpu_conv_vae_trainer_new(v, opt);
}
static ConvAeTrainer conv_ae_trainer_new(struct ConvAutoencoder ae, const &Optimizer opt, const &TrainerOptions o) {
    return cpu_conv_ae_trainer_new(ae, opt);
}
static GanTrainer gan_trainer_new(struct Gan g, const &TrainerOptions o) { return cpu_gan_trainer_new(g); }
static DcganTrainer dcgan_trainer_new(struct Dcgan g, const &TrainerOptions o) { return cpu_dcgan_trainer_new(g); }
static DqnLearner dqn_learner_new(struct Dqn d) { return cpu_dqn_agent_new(d); }
// o.maxBatch: the longest rollout pg_update will see (a device's limit).
static PgLearner pg_learner_new(struct PolicyAgent a, const &TrainerOptions o) { return cpu_pg_agent_new(a); }

#endif

} // namespace std
