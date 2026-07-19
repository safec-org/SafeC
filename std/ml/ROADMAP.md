# std/ml roadmap

What's implemented and verified, and what's explicitly deferred.

## Implemented (verified — real compile, link, run, and numeric correctness checks)

- `tensor.h`/`tensor.sc` — CPU `Tensor` (1D/2D, f64) with reverse-mode
  autograd: add/sub/mul/scale/matmul/relu/sum, `tensor_backward()`.
  Verified against hand-computed gradients and a real finite-difference
  check on a matmul+relu+sum chain.
- `activations.h`/`activations.sc` — sigmoid/tanh/silu/gelu (elementwise),
  `tensor_layernorm_rows` (row-wise mean-0/var-1 normalization, no
  affine), `tensor_residual_add` (named `tensor_add` alias for residual/
  skip-connection call sites).
- `attention.h`/`attention.sc` — row-wise softmax, `attention_forward()`
  (single-head scaled dot-product attention), `mha_forward()` (standard
  multi-head attention), `gated_attention_forward`/`gated_mha_forward`
  (sigmoid output gating), `linear_attention_forward` (elu+1 kernel-trick
  reformulation), `flash_attention_forward` (tiled online-softmax,
  verified to match `attention_forward` within tolerance).
- `attention_advanced.h`/`attention_advanced.sc` — `mla_forward`/
  `eg_mla_forward` (Multi-head Latent Attention and a grouped-latent
  variant, verified via self-consistency against manual per-group
  matmul construction), `windowed_attention_forward` (Shifted-Window/
  Swin attention adapted to 1D sequences, verified for both window-
  locality and shift/unshift round-tripping), `gated_deltanet_forward`
  (delta-rule linear-recurrent attention replacement, verified via a
  2-step hand-computed state trace).
- `rnn.h`/`rnn.sc` — RNN/GRU/LSTM/xLSTM (sLSTM variant) cells, each
  verified via hand-computed gate values against a real forward pass.
- `cnn.h`/`cnn.sc` — `Conv2D` (stride+padding), MaxPool2D/AvgPool2D,
  `upsample2x_nearest`, `concat_channels`, all over a CHW `FeatureMap`.
  Verified via hand-computed 3x3-kernel and pooling-window checks.
- `unet.h`/`unet.sc` — a fixed 2-level U-Net (encoder/bottleneck/decoder
  with skip connections) over `FeatureMap`. Verified via output-shape
  and full self-consistency checks against a manual replication of the
  same call sequence.
- `diffusion.h`/`diffusion.sc` — `edm_karras_sigmas` (Karras et al. noise
  schedule), `ddpm_linear_schedule`, `ddpm_sampler_step`,
  `ddim_sampler_step`, `dpm_solver_1_step`/`dpm_solver_2_step` (eps-
  prediction exponential integrator, order 1 and midpoint order 2),
  `dpm_solver_pp_1_step`/`dpm_solver_pp_2_step` (data/x0-prediction
  form). Verified via hand-computed schedule values and the proven
  mathematical identity DPM-Solver++ order-1 == DDIM (both numerically
  confirmed to agree).
- `transformer.h`/`transformer.sc` — `DiTBlock` (adaLN-Zero-style
  per-block conditioning, verified via a gate=0 exact-identity check and
  full self-consistency), `JiTBlock`/`jit_forward` (plain pre-LN
  transformer with additive up-front conditioning, verified via
  self-consistency and a conditioning-sensitivity check).
- `gpu_mps.h`/`gpu_mps.sc` — Metal (Apple Silicon/macOS) backend.
  `mps_available()` fully verified on real hardware. `mps_add_f32()`'s
  pipeline is verified through pipeline/buffer/encoder setup (including a
  real runtime-compiled MSL shader) but currently segfaults at the final
  dispatch call — see that file's header comment for the precise,
  bisected failure point.
- `gpu_cuda.h`/`gpu_cuda.sc`, `gpu_rocm.h`/`gpu_rocm.sc` — written against
  the real CUDA Driver API / HIP Runtime API and type-checked, but
  UNVERIFIED: no NVIDIA/AMD GPU or toolkit in this environment (same
  status as `std/gui/gui_win32.sc`/`gui_x11.sc`).

## Deferred (explicitly out of scope, documented rather than faked)

- **DPM-Solver-v3** (Zheng et al. 2023): its distinguishing idea is an
  affine ODE reparametrization calibrated from "empirical model
  statistics" gathered by running a real trained model over real data —
  not something obtainable in this environment. A version that skipped
  the calibration step wouldn't actually implement the paper's
  contribution, so it's left out rather than faked.
- **EDM's own samplers** (Euler/Heun ancestral, etc. — the sampler half
  of the EDM paper that actually consumes `edm_karras_sigmas`): the
  schedule itself is implemented and verified above; the VE-flavored
  samplers that walk it are not, since this library's other samplers
  (DDPM/DDIM/DPM-Solver family) all use the VP convention instead (see
  `diffusion.h`'s header comment on why the two aren't unified).
- Real positional encoding (RoPE, ALiBi, etc.), patchify/depatchify for
  DiT/JiT, and decoupled rotary keys for MLA are all out of scope — this
  library's attention primitives have never included positional
  encoding (by design, see `attention.h`); callers add their own.
