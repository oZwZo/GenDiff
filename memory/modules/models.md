# Module: models (the ε / displacement networks)
> Files: src/_epsilon_module.py, src/_helper_net.py, src/_linear.py, src/my_VAE.py  |  Part of: [codebase.md](../codebase.md)

## Purpose
Defines the neural networks that, given a cell's expression vector `x`, a perturbation
condition `c` (e.g. a TF), an optional batch index and a (pseudo)time `t`, predict a
**displacement vector ΔX** — how the cell should move on the differentiation manifold.
Despite the name "epsilon" (DDPM noise predictor), in the model actually used the target
is ΔX = (neighbour cell − current cell), i.e. a **conditional velocity/vector field**, not
Gaussian noise. This module owns architecture only; the noising/denoising math and the
training loop live in [diffusion.md](diffusion.md).

## Files
| File | Role |
|------|------|
| src/_epsilon_module.py | Current ε-network zoo (Linear, CAE, CVAE, attention, ODE variants) |
| src/_epsilon_module_old.py | Legacy copy of the same zoo — NOT imported by training; reference only |
| src/_helper_net.py | Reusable NN building blocks (MLP, attention, sinusoidal time embedding, residual) |
| src/_linear.py | Standalone PL baseline models (linear, CAE, embedding/interaction, prior-injected) |
| src/my_VAE.py | scvi-tools NegativeBinomial VAE experiment (standalone, not in main pipeline) |

## Classes & Functions
### `src/_epsilon_module.py`
- **`Epsilon_base`** (nn.Module): abstract parent. Builds optional sinusoidal **time MLP**
  (`time_emb_dim`>0) and a condition **`nn.Embedding`** (`condition_emb_dim`>0, token 0 = control).
  `_get_model_input(x,c,batch,t)` assembles the model input dict and condition embedding.
  `forward`/`get_DeltaX` raise NotImplementedError.
- **`Epsilon_Linear`** (Epsilon_base): the workhorse. VAE-shaped MLP `encoder→decoder`; the
  condition embedding is **added** to the bottleneck latent (`z_c = z + c_emb`) so a TF acts
  as a vector offset in latent space. `hidden_size` list defines depth; the layer equal to
  `condition_emb_dim` is treated as the latent layer. `encode()` returns z/z_c/c.
- **`Epsilon_CAE`** (Epsilon_Linear): conditional autoencoder; non-variational; `SmoothL1`
  reconstruction loss; defines `compute_loss`.
- **`GuassianNLL_CAE`** (Epsilon_Linear): outputs (mean, var) heads, Gaussian NLL loss.
- **`Epsilon_CVAE`** (Epsilon_CAE): conditional **VAE**; reparameterised latent, learns a
  per-gene variance parameter; loss = GaussianNLL recon + `kl_weight`·KL. `variational=True`.
- **`Epsilon_VAE`** (Epsilon_CVAE): CVAE variant with explicit `fc_mu`/`fc_var` heads.
- **`Epsilon_CVAE_adv`** (Epsilon_CVAE): placeholder for an adversarial variant (stub).
- **`Epsilon_AttnCondition`** (Epsilon_Linear): replaces additive conditioning with
  `condition_attention()` — a learned weighted mix of control vs condition embeddings.
- **`Epsilon_Categorical`** (nn.Module): no embedding; concatenates one-hot condition to `x`.
- **`Epsilon_LinearAttn`** (Epsilon_base): deep model built from `LinearAttention`/`Residual`
  blocks (O(n) attention) for large/deep nets.
- **`ODE_eps` / `ODE_eps_notime`** (Epsilon_Linear): `forward(X)` signature for use as a
  Neural-ODE vector field (`c`,`t0` set as attributes); consumed by `ODE_learner`.
- **`Epsilon_Linear_token_net`** (Epsilon_Linear): TODO stub.

### `src/_helper_net.py`
- **`SinoidalPositionEmbeddings`**: standard DDPM sinusoidal time embedding.
- **`MLP`**: configurable stack (batchnorm, dropout, optional skip connections).
- **`SelfAttention`**: multi-head self-attention over the feature vector.
- **`LinearAttention`**: linear-complexity attention (Shen 2018) used by `Epsilon_LinearAttn`.
- **`Residual`**: `f(x)+x` wrapper. **`ResnetBlock`**: two attention blocks + skip.

### `src/_linear.py` (independent PL baselines — not ε-nets)
- **`base`** (LightningModule): MSE + R² baseline scaffold.
- **`Linear_model`**, **`CAE_model`**, **`LatentAdd_CAE`**, **`Embedding_model`**,
  **`Latent_Interaction`** (latent attention over multiple condition tokens; partially WIP),
  **`Pretrained_GenDiff`** (wraps a pretrained model + injects a linear regression prior
  `prior_coef`/`prior_intercept` into the output layer).

### `src/my_VAE.py`
- **`my_VAE`** (scvi `BaseModuleClass`): NB-likelihood VAE with library-size factor. Standalone
  experiment; depends on `cpa`/`scvi`; has known typos (`nb_logits`, `log_theta`, `log_like`).

## Dependencies on Other Modules
- `_epsilon_module` imports building blocks from `_helper_net`.
- Instantiated by `script/main_train_with_PL_trainer.get_model_from_config` via
  `configs.epsilon_kwargs` (see [configs.md](configs.md)). Wrapped by a sampler/learner
  from [diffusion.md](diffusion.md).

## Module-Level Gotchas
- Target is **ΔX (displacement), not noise**, despite "Epsilon" naming.
- Two parallel files: `_epsilon_module.py` (used) vs `_epsilon_module_old.py` (dead copy).
- Conditioning is **additive in latent space** (`z + c_emb`) for the Linear/CAE/CVAE family;
  `z.shape == c_emb.shape` is asserted, so a hidden layer must equal `condition_emb_dim`.
- Condition token **0 is reserved for control**; multiplexed perturbations are summed.
- `from turtle import forward` appears at the top of several files — a stray/no-op import.
