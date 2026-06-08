# Module: diffusion (samplers, schedulers, Lightning learners)
> Files: src/_sampler.py, src/_learner.py  |  Part of: [codebase.md](../codebase.md)

## Purpose
Owns the diffusion process around an ε-network: the noise/β schedules, the forward
`q_sample` (corrupt) and reverse `p_sample` (denoise/integrate) steps, the training loss,
and the PyTorch-Lightning training loop. **There are two parallel implementations of the
same class names** — pick the right one:
- `src/_learner.py` → `pl.LightningModule` subclasses. **These are trained** (instantiated by
  `get_sampler_from_configs`, fed to `pl.Trainer`).
- `src/_sampler.py` → plain `object`/`nn.Module` subclasses. **Used for inference/generation**
  after a checkpoint is reloaded (e.g. by `_diffplot.reload_sampler`, `diffuse_cell`).

## Files
| File | Role |
|------|------|
| src/_sampler.py | Inference-time samplers (DDPM, DDIM, reconX) + β-schedules + eval loops |
| src/_learner.py | Lightning training modules: Equivalent/DDPM/Weight_l2/AE/ODE learners |

## Classes & Functions
### β-schedules (defined in both files)
- `linear_beta_schedule`, `quadratic_beta_schedule`, `sigmoid_beta_schedule`,
  `cosine_beta_schedule` — map `timesteps` → β array. Chosen by `configs.scheduler` (string
  `eval`'d). Configs typically use `linear_beta_schedule`, `beta_start 1e-8`, `beta_end 1e-3`.

### `src/_learner.py` (training)
- **`DiffusionSampler_base`** (LightningModule): precomputes α, ᾱ, etc. in `_compute_alphas`;
  `q_sample`/`p_sample` abstract; `p_loss` = loss between target `noise` (=ΔX) and `eps_pred`;
  `training_step`/`validation_step`/`test_step` unpack a batch as
  **`(x_0, exp_batch, c, noise, t)`** and call `forward(x_t,t,batch,c)=model(x_t,c,batch,t)`.
  Validation sweeps `t` across all timesteps and logs the worst-error time `t_MaxError`.
- **`DDPM_Sampler`** (base): standard DDPM `q_sample` (√ᾱ·x₀ + √(1−ᾱ)·ε) and ancestral
  `p_sample`. Posterior variance `sigma`.
- **`DDPM_reconX`** (DDPM_Sampler): model predicts the cleaner `x_{t}` from `x_{t+1}` (x-param,
  not ε-param); loss compares successive q-samples.
- **`Equivalent_Diffuse_Sampler`** (base): **the main learner used in configs.** Overrides
  `q_sample = x_0 − noise` and `p_sample = x_t + eps_theta`, i.e. the network directly predicts
  the displacement ΔX added each step. Training loss = `loss_fn(noise=ΔX, eps_pred)`.
- **`Weight_l2_sampler`** (Equivalent): adds L2 weight regularisation on the first encoder
  layer + optimizer `weight_decay`.
- **`AE_learner`** (LightningModule): for the CAE/CVAE epsilon nets (the `*_AE` configs).
  Unpacks `(X, batch_idx, condition_idx, Delta_X, degree)`, uses `model.get_DeltaX` and
  `model.compute_loss`; logs R², Pearson corr, and sign accuracy. Handles `variational` models.
- **`ODE_learner`** (LightningModule): wraps an `ODE_eps` net in `torchdyn`/`NeuralODE`
  (`dopri5`, adjoint), integrates over a `t_span`, fits the endpoint to `X_t1`.

### `src/_sampler.py` (inference)
- **`DiffusionSampler_base`** (object): mirror of the above without Lightning. Adds
  `q_sample_loop` (x₀→x_T), `p_sample_loop` (pure noise→x₀, used by `sample()`), and
  **`evaluation_loop`** returning `{"Q":corrupt, "S":stepwise, "R":recon, "P":de-novo}`.
- **`DDPM_Sampler`**, **`DDPM_reconX`** — ancestral sampling.
- **`DDIM_Sampler`**: deterministic/implicit sampler with `eta`, linear or quadratic
  discretisation, predicts x̂₀ then steps.

## Internal Data Flow
```
batch (x0, batch, c, ΔX, t)  ──►  q_sample(x0,t,ΔX)=x_t ──► model(x_t,c,batch,t)=ΔX̂
                                                              │
                                          loss_fn(ΔX, ΔX̂) ◄──┘   (Equivalent_Diffuse_Sampler)
inference:  x_t ──p_sample──► x_t + ΔX̂ ──► … iterate over degrees/time ──► simulated trajectory
```

## Dependencies on Other Modules
- Wraps an ε-net from [models.md](models.md) (passed as `model=`).
- Batches are produced by datasets in [data.md](data.md); the **5-tuple order matters**.
- Built by `script/main_train_with_PL_trainer.get_sampler_from_configs` (see [scripts.md](scripts.md)).

## Module-Level Gotchas
- Two implementations, same names — train with `_learner`, sample with `_sampler`.
- "noise" is **ΔX**, not Gaussian noise, in the Equivalent sampler (the one actually trained).
- Forward argument order flips between learner and model: learner `forward(x,t,b,c)` calls
  `model(x, c, b, t)`. Easy to mis-wire.
- `DDPM_reconX.__init__` passes scheduler/loss/timesteps in a different order to its parent
  (`super().__init__(model, scheduler, loss_type, timesteps,...)`) — a latent bug.
- `ODE_learner` requires `torchdyn`'s `NeuralODE` (import not shown in file header).
