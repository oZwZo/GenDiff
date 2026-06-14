"""
gendiff.model — a high-level, scvi-tools-style entry point for training GenDiff.

The user-facing sequence mirrors scvi-tools:

    GenDiff.setup_anndata(adata, condition_key="TF", pseudotime_key="dpt_pseudotime",
                          use_rep="X_pca_harmony", sampler_config="config1")
    model = GenDiff(adata, hidden_size=(512, 512, 256, 512, 512))
    model.train(max_epochs=200)
    dx  = model.predict_delta(adata, condition="Sox2")
    model.save("runs/sox2"); GenDiff.load("runs/sox2", adata)

`setup_anndata` is where the neighbour-difference target is built: it runs the revised knn_sampler
(gendiff_dev.targets.builders.knn_sampler) and stores ΔX in `adata.obsm`. The sampler is the DEFAULT
supervision — `sampler_config="config1"` (geodesic, M=15, repeat=3, the r3 recommendation), with
`"config2"` (euclid, M=30, repeat=3) and `"others"` (your own metric/M/repeat via `sampler_kwargs`)
as the other options. This replaces the on-the-fly Path_Diffuse sampler for normal training.

The class is a thin facade over the existing pieces (src._epsilon_module epsilon net, src._learner
diffusion learner, a PyTorch-Lightning Trainer); it adds no new model maths, only an easier call.
"""
from __future__ import annotations
import os, json, warnings
import numpy as np

REGISTRY_UNS = "gendiff_setup"
TOKEN_UNS = "gendiff_token_dict"


def _densify(M):
    import scipy.sparse as sp
    return M.toarray() if sp.issparse(M) else np.asarray(M)


class GenDiff:
    # ---- data registration -------------------------------------------------
    @classmethod
    def setup_anndata(cls, adata, *, condition_key, pseudotime_key, use_rep,
                      layer="X", split_key=None, batch_key=None, control=None,
                      delimiter="|", sampler="knn", sampler_config="config1", same_condition="auto",
                      graph_k=15, alpha=1.0, seed=0, sampler_kwargs=None,
                      target_obsm="gendiff_dX", n_time_bins=200, rebuild=False, verbose=True):
        """Register data fields and build the ΔX supervision target.

        condition_key / pseudotime_key : obs columns. use_rep : obsm key for the kNN graph.
        layer : expression matrix ("X" or a layers key). split_key : obs column with train/val/test
        (a random split is created if None). control : the control condition label (token 0); inferred
        if None.

        sampler : which ΔX construction to use as supervision —
          'knn'       (default) the higher-pseudotime neighbour-difference sampler (knn_sampler); pick the
                      preset with `sampler_config` ('config1'/'config2'/'others').
          'smooth_nb' per-cluster NB/Poisson μ(t) derivative (the SAMPLER_OPT winner): removes the
                      regression-to-mean confound the neighbour difference carries and self-slows at the
                      terminal plateau. Needs counts (adata.raw or a count layer) and `cluster_key`
                      (default 'louvain'); falls back to a spline / one global cluster if absent.
          'nb_global' the same NB derivative fit as ONE global trajectory (no clustering).
          'ot'        per-cluster optimal-transport early->late displacement: the manifold-robust runner-up,
                      naturally scaled because it transports to real late cells. Needs POT.
          'ptgrad'    pseudotime-gradient field: reaches furthest forward but over-disperses off-manifold
                      and is perturbation-blind — an aggressive / ablation target, not a default.
          'smooth'    the generic smooth_deriv escape hatch; choose the mode yourself via `sampler_kwargs`
                      ({'mode': 'nb'|'gauss'|'local'|'auto', 'cluster_key': ..., 'auto_root': True}).
        same_condition : True/False or 'auto' (decide from cells-per-condition density); 'knn' only.
        sampler_kwargs : extra kwargs forwarded to the chosen sampler (overrides the defaults above).
        """
        for k in (condition_key, pseudotime_key):
            if k not in adata.obs:
                raise KeyError(f"setup_anndata: obs[{k!r}] not found")
        if use_rep not in adata.obsm:
            raise KeyError(f"setup_anndata: obsm[{use_rep!r}] not found")

        conds = adata.obs[condition_key].astype(str).values
        atoms = sorted({a for c in conds for a in c.split(delimiter)})
        if control is None:
            for guess in ("ctrl", "control", "Control", "DMSO", "GFP"):
                if guess in atoms:
                    control = guess
                    break
            if control is None:
                control = atoms[0]
                warnings.warn(f"setup_anndata: control not given; using {control!r} as token 0.")
        if control not in atoms:
            raise ValueError(f"setup_anndata: control {control!r} not among conditions")
        ordered = [control] + [a for a in atoms if a != control]
        token_dict = {a: i for i, a in enumerate(ordered)}
        adata.uns[TOKEN_UNS] = token_dict
        max_multiplexing = max(len(c.split(delimiter)) for c in conds)

        # discrete pseudotime used as the diffusion time index (must be < model timesteps)
        if "discrete_time" not in adata.obs:
            pt = np.asarray(adata.obs[pseudotime_key], float)
            lo, hi = np.nanmin(pt), np.nanmax(pt)
            ptn = (pt - lo) / (hi - lo + 1e-12)
            adata.obs["discrete_time"] = np.clip((ptn * (n_time_bins - 1)).round(), 0,
                                                 n_time_bins - 1).astype(int)

        # same_condition: auto-decide from per-condition density
        if same_condition == "auto":
            counts = adata.obs[condition_key].value_counts().values
            med = float(np.median(counts))
            same_condition = bool(len(counts) > 1 and med >= 50)
            if verbose:
                print(f"[GenDiff.setup] same_condition='auto' -> {same_condition} "
                      f"(median {med:.0f} cells/condition over {len(counts)} conditions)", flush=True)

        # build the ΔX supervision target with the chosen sampler
        valid_samplers = ("knn", "smooth", "smooth_nb", "nb_global", "ot", "ptgrad")
        if sampler not in valid_samplers:
            raise ValueError(f"setup_anndata: sampler must be one of {valid_samplers}, got {sampler!r}")
        if target_obsm in adata.obsm and not rebuild:
            if verbose:
                print(f"[GenDiff.setup] obsm[{target_obsm!r}] exists; reuse (rebuild=True to redo)",
                      flush=True)
        else:
            from gendiff_dev.targets import builders
            common = dict(use_rep=use_rep, pseudotime_key=pseudotime_key, verbose=verbose)
            if sampler == "knn":
                builder = builders.knn_sampler
                kw = dict(condition_key=condition_key, same_condition=same_condition,
                          config=sampler_config, graph_k=graph_k, alpha=alpha, seed=seed, **common)
            elif sampler == "ptgrad":
                builder = builders.pt_gradient
                kw = dict(graph_k=graph_k, **common)
            elif sampler == "ot":
                builder = builders.ot_sampler
                kw = dict(cluster_key="louvain", graph_k=graph_k, seed=seed, **common)
            else:  # smooth_nb / nb_global / smooth (generic) -> smooth_deriv
                builder = builders.smooth_deriv
                mode = dict(smooth_nb="nb", nb_global="nb").get(sampler, "auto")
                cluster_key = None if sampler == "nb_global" else "louvain"
                kw = dict(mode=mode, cluster_key=cluster_key, graph_k=graph_k, seed=seed, **common)
            kw.update(sampler_kwargs or {})
            dX = builder(adata, **kw)
            if dX.shape[1] != adata.n_vars:
                warnings.warn(f"setup_anndata: ΔX has {dX.shape[1]} genes != n_vars {adata.n_vars}")
            rms = float(np.sqrt(np.mean(np.square(np.asarray(dX, np.float64)))))
            if rms < 0.01:
                warnings.warn(f"setup_anndata: target ΔX RMS={rms:.2e} is very small (ptgrad is a "
                              f"direction field, not magnitude-calibrated); the model regresses ΔX "
                              f"magnitude, so consider scaling the target or raising lr when training.")
            adata.obsm[target_obsm] = dX

        if split_key is None:
            rng = np.random.default_rng(seed)
            r = rng.random(adata.n_obs)
            sp_ = np.where(r < 0.8, "train", np.where(r < 0.9, "val", "test"))
            adata.obs["gendiff_split"] = sp_
            split_key = "gendiff_split"

        gene_dim = adata.n_vars if layer in (None, "X") else int(adata.layers[layer].shape[1])
        adata.uns[REGISTRY_UNS] = dict(
            condition_key=condition_key, pseudotime_key=pseudotime_key, use_rep=use_rep, layer=layer,
            split_key=split_key, batch_key=batch_key, control=control, delimiter=delimiter,
            sampler=sampler, sampler_config=sampler_config, same_condition=bool(same_condition),
            target_obsm=target_obsm,
            gene_dim=gene_dim, n_base_perturbs=int(max(token_dict.values()) + 1),
            max_multiplexing=int(max_multiplexing), time_key="discrete_time")
        if verbose:
            s = adata.uns[REGISTRY_UNS]
            print(f"[GenDiff.setup] registered: {gene_dim} genes, {s['n_base_perturbs']} condition "
                  f"tokens (control={control!r}), target=obsm[{target_obsm!r}], split={split_key!r}",
                  flush=True)
        return adata

    # ---- construction ------------------------------------------------------
    def __init__(self, adata, *, hidden_size=(512, 512, 512, 256, 512, 512, 512),
                 condition_emb_dim=256, time_emb_dim=64, timesteps=200,
                 scheduler="linear_beta_schedule", loss_type="huber", beta_start=1e-8,
                 beta_end=1e-3, activation="Mish", epsilon_class="Epsilon_Linear",
                 sampler_class="Equivalent_Diffuse_Sampler", verbose=True):
        import torch  # noqa: F401  (ensures torch present early)
        from src import _epsilon_module, _learner
        if REGISTRY_UNS not in adata.uns:
            raise RuntimeError("GenDiff: call GenDiff.setup_anndata(adata, ...) before constructing.")
        self.adata = adata
        self.setup = dict(adata.uns[REGISTRY_UNS])
        s = self.setup
        if condition_emb_dim not in tuple(hidden_size):
            warnings.warn(f"GenDiff: condition_emb_dim={condition_emb_dim} not in hidden_size; the "
                          f"epsilon net will insert it as the latent layer.")
        self.module = getattr(_epsilon_module, epsilon_class)(
            gene_dim=s["gene_dim"], time_emb_dim=time_emb_dim, n_base_perturbs=s["n_base_perturbs"],
            condition_emb_dim=condition_emb_dim, use_batch_index=s["batch_key"] is not None,
            hidden_size=list(hidden_size), activation=activation)
        self._learner = getattr(_learner, sampler_class)(
            model=self.module, scheduler=scheduler, loss_type=loss_type, timesteps=timesteps,
            beta_start=beta_start, beta_end=beta_end)
        self.timesteps = timesteps
        self.is_trained_ = False
        self.init_params_ = dict(
            hidden_size=list(hidden_size), condition_emb_dim=condition_emb_dim,
            time_emb_dim=time_emb_dim, timesteps=timesteps, scheduler=scheduler, loss_type=loss_type,
            beta_start=beta_start, beta_end=beta_end, activation=activation,
            epsilon_class=epsilon_class, sampler_class=sampler_class)
        if verbose:
            n = sum(p.numel() for p in self.module.parameters())
            print(f"[GenDiff] {epsilon_class} + {sampler_class}: {n/1e6:.2f}M params, "
                  f"timesteps={timesteps}", flush=True)

    # ---- training ----------------------------------------------------------
    def _loader(self, which_set, batch_size, shuffle, num_workers, drop_last):
        import torch
        from torch.utils.data import DataLoader
        from torch.utils.data.dataloader import default_collate
        from src._reader import Precomputed_Delta_Dataset
        s = self.setup
        ds = Precomputed_Delta_Dataset(
            self.adata, condition_key=s["condition_key"], token_dict=self.adata.uns[TOKEN_UNS],
            target_obsm=s["target_obsm"], layer=s["layer"], time_key=s["time_key"],
            split_key=s["split_key"], which_set=which_set, batch_key=s["batch_key"],
            max_multiplexing=s["max_multiplexing"], delimiter=s["delimiter"])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=default_collate, drop_last=drop_last, pin_memory=False)

    def train(self, max_epochs=200, batch_size=128, lr=1e-3, accelerator="auto", devices=1,
              num_workers=4, patience=50, default_root_dir=None, early_stopping=True,
              enable_progress_bar=True, **trainer_kwargs):
        """Train the diffusion learner on the precomputed ΔX target. Wraps a PyTorch-Lightning Trainer."""
        import types, torch
        import pytorch_lightning as pl
        from pytorch_lightning import callbacks as plc

        train_loader = self._loader("train", batch_size, True, num_workers, drop_last=True)
        val_loader = self._loader("val", batch_size, False, num_workers, drop_last=False)

        # honour the requested learning rate (base learner hardcodes its optimiser otherwise)
        def _configure_optimizers(self_):
            return torch.optim.Adam(self_.parameters(), lr=lr)
        self._learner.configure_optimizers = types.MethodType(_configure_optimizers, self._learner)

        cbs = [plc.ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")]
        if early_stopping:
            cbs.append(plc.EarlyStopping(monitor="val_loss", mode="min", patience=patience))
        if accelerator == "auto":
            accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        self.trainer = pl.Trainer(
            accelerator=accelerator, devices=devices, max_epochs=max_epochs,
            default_root_dir=default_root_dir, callbacks=cbs, check_val_every_n_epoch=1,
            enable_progress_bar=enable_progress_bar, **trainer_kwargs)
        self.trainer.fit(self._learner, train_loader, val_loader)
        self.is_trained_ = True
        return self

    # ---- inference ---------------------------------------------------------
    def predict_delta(self, adata=None, condition=None, *, batch_size=1024, device=None):
        """Predict the ΔX field (cell-state displacement) for every cell under `condition`.

        With Equivalent_Diffuse_Sampler the learned net maps (X, condition, pseudotime) -> ΔX directly,
        so this is one forward pass per cell. `condition` is a token label (default: control / token 0).
        Returns an (n_obs, gene_dim) array aligned to adata.obs order.
        """
        import torch
        adata = self.adata if adata is None else adata
        s = self.setup
        tok = self.adata.uns[TOKEN_UNS]
        if condition is None:
            cid = 0
        elif condition in tok:
            cid = tok[condition]
        else:
            raise KeyError(f"predict_delta: unknown condition {condition!r}; known: {list(tok)[:8]}...")
        X = _densify(adata.layers[s["layer"]] if s["layer"] not in (None, "X") else adata.X).astype(np.float32)
        t = np.asarray(adata.obs[s["time_key"]].values).astype(np.int64) if s["time_key"] in adata.obs \
            else np.zeros(adata.n_obs, np.int64)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._learner.to(device).eval()
        mux = s["max_multiplexing"]
        out = np.empty((adata.n_obs, s["gene_dim"]), np.float32)
        with torch.no_grad():
            for a in range(0, adata.n_obs, batch_size):
                b = slice(a, a + batch_size)
                xb = torch.from_numpy(X[b]).to(device)
                cb = torch.full((xb.shape[0], mux), cid, dtype=torch.long, device=device)
                tb = torch.from_numpy(t[b]).to(device)
                out[b] = self.module(xb, cb, None, tb).cpu().numpy()
        return out

    def predict_population(self, adata=None, condition=None, **kw):
        """Predicted post-perturbation state X + ΔX(condition) for every cell."""
        adata = self.adata if adata is None else adata
        s = self.setup
        X = _densify(adata.layers[s["layer"]] if s["layer"] not in (None, "X") else adata.X).astype(np.float32)
        return X + self.predict_delta(adata, condition, **kw)

    # ---- persistence -------------------------------------------------------
    def save(self, dir_path, overwrite=True):
        import torch
        if os.path.exists(dir_path) and not overwrite:
            raise FileExistsError(dir_path)
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self._learner.state_dict(), os.path.join(dir_path, "model.pt"))
        with open(os.path.join(dir_path, "attr.json"), "w") as f:
            json.dump(dict(setup=self.setup, init_params=self.init_params_,
                           token_dict=self.adata.uns[TOKEN_UNS], is_trained=self.is_trained_), f, indent=2)
        return dir_path

    @classmethod
    def load(cls, dir_path, adata):
        import torch
        with open(os.path.join(dir_path, "attr.json")) as f:
            attr = json.load(f)
        adata.uns[REGISTRY_UNS] = attr["setup"]
        adata.uns[TOKEN_UNS] = {k: int(v) for k, v in attr["token_dict"].items()}
        model = cls(adata, verbose=False, **attr["init_params"])
        state = torch.load(os.path.join(dir_path, "model.pt"), map_location="cpu")
        model._learner.load_state_dict(state)
        model.is_trained_ = attr.get("is_trained", True)
        return model
