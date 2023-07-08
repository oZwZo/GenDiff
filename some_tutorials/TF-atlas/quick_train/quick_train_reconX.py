import os
import numpy as np
import pandas as pd
import scanpy as sc

import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import callbacks 

from src import PATH
from torch.utils.data import DataLoader, Dataset, TensorDataset
adata_diff = sc.read_h5ad(
    "/home/wergillius/Project/diffuse_differentiate/data/TFAtlas/GSE217460_210322_TFAtlas_differentiated.h5ad"
)
# tensor ds
TF_onehot = pd.get_dummies(adata_diff.obs.TF).values
gene_X = adata_diff.X
X =  np.concatenate([gene_X, TF_onehot], axis=1)

# Y = adata_diff.layers['a3_PathSampled_X'].copy()
Y = gene_X

train_idx = adata_diff.obs.split != 'test'
test_idx = adata_diff.obs.split == 'test'

x_train = torch.from_numpy(X[train_idx]).float()
x_test = torch.from_numpy(X[test_idx]).float()

Y_train = torch.from_numpy(Y[train_idx]).float()
Y_test = torch.from_numpy(Y[test_idx]).float()
train_ds = TensorDataset(x_train, Y_train)
train_dl = DataLoader(train_ds, batch_size=32, num_workers=8, shuffle=True)

test_ds = TensorDataset(x_test, Y_test)
test_dl = DataLoader(test_ds, batch_size=32,  num_workers=8, shuffle=False)
# define embedding model
from src import _linear, _helper_net
from importlib import reload
reload(_linear)
# To ensuer skip connection
    # 512 -> 512 
    # 512+32 -> 544
encoder_kwargs = dict(
    dimensions = [4806, 512, 512],
    use_batchnorm=True, skip_connection=True,
    activation_fn="ReLU", output_activation="ReLU"
)

decoder_kwargs = dict(
    dimensions = [544, 544, 4806],
    use_batchnorm=True, skip_connection=True,
    activation_fn="ReLU", output_activation=None
)

Emb_AE = _linear.Embedding_model(
                condition_dim=2535,
                encoder_kwargs=encoder_kwargs,
                decoder_kwargs=decoder_kwargs,
                lr=3e-5, 
                weight_decay=1e-9)
Emb_AE
# set up and fit
Trainer = pl.Trainer(accelerator='gpu', gpus=[0],
                    #  fast_dev_run=True,
                     default_root_dir = os.path.join(PATH.pth_dir, 'TF_atlas_quick', 'Embedding_CAE_deep_model_reconX'),
                     callbacks=[
                         callbacks.ModelCheckpoint(save_top_k=2, monitor="val_loss"),
                         callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
                         ],
                    )

Trainer.fit(Emb_AE, train_dataloaders=train_dl, 
            val_dataloaders=test_dl)
