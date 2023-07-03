import numpy as np
import pandas as pd
import scanpy as sc

import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import callbacks 

from src import PATH, _helper_net
from torch.utils.data import DataLoader, Dataset, TensorDataset

class base(pl.LightningModule):
    def __init__(self, input_dim, output_dim, lr, weight_decay):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay

        self.loss_fn = nn.MSELoss()

        self.train_r2_score = torchmetrics.R2Score(num_outputs = output_dim)
        self.val_r2_score = torchmetrics.R2Score(num_outputs = output_dim)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def forward(self,X):
        raise NotImplementedError("not defined")
    
    def training_step(self, train_batch, batch_idx):
        X,Y = train_batch
        Ypred= self.forward(X)
        loss = self.loss_fn(Y, Ypred)
        r2 = self.train_r2_score(Y, Ypred)
        self.log_dict({'train_loss':loss, 'train_r2':r2})
        return loss

    def validation_step(self, val_batch, batch_idx):
        X,Y = val_batch
        Ypred= self.forward(X)
        loss = self.loss_fn(Y, Ypred)
        r2 = self.val_r2_score(Y,Ypred)
        self.log_dict({'val_loss':loss, 'val_r2':r2})
        return loss

    def test_step(self, test_batch, batch_idx):
        X,Y = test_batch
        Ypred= self.forward(X)
        loss = self.loss_fn(Y, Ypred)
        self.log_dict({'test_loss':loss})
        return loss



class Linear_model(base):
    def __init__(self,input_dim, output_dim, lr, weight_decay):
        super().__init__(input_dim, output_dim, lr, weight_decay)
        
        self.model = nn.Linear(input_dim,output_dim)
    
    def forward(self, X):
        return self.model(X)


class CAE_model(Linear_model):
    def __init__(self, input_dim, output_dim, hidden, lr, weight_decay):
        super().__init__(input_dim, output_dim, lr, weight_decay)

        bottlenet = (len(hidden)-1)//2
        self.bottlenet = bottlenet
        self.latent = hidden[bottlenet]

        encoder_dims = [input_dim] + hidden[:bottlenet+1]
        decoder_dims = hidden[bottlenet:]
        self.encoder = _helper_net.MLP(encoder_dims, use_batchnorm=True, use_dropout=0)
        

        self.decoder = nn.Sequential(
            _helper_net.MLP(decoder_dims, use_batchnorm=True, use_dropout=0),
            nn.Linear(hidden[-1], output_dim)
            )

        self.model = nn.Sequential(
             self.encoder,
             self.decoder
        )

    def encode(self, X):
        return self.encoder(X)

    def forward(self, X):
        z = self.encode(X)
        return self.decoder(z)
    
class LatentAdd_CAE(base):
    def __init__(self, input_dim, condition_dim, output_dim, hidden, lr, weight_decay):
        super().__init__(input_dim,  output_dim, lr, weight_decay)

        self.condition_dim = condition_dim

        bottlenet = (len(hidden)-1)//2
        self.bottlenet = bottlenet
        self.latent = hidden[bottlenet]

        encoder_dims = [input_dim] + hidden[:bottlenet+1]
        decoder_dims = [self.latent +  self.condition_dim] + hidden[bottlenet+1:]

        self.encoder = _helper_net.MLP(encoder_dims, use_batchnorm=True, use_dropout=0)
        

        self.decoder = nn.Sequential(
            _helper_net.MLP(decoder_dims, use_batchnorm=True, use_dropout=0),
            nn.Linear(decoder_dims[-1], output_dim)
            )

    def forward(self, X):
        C = X[:,-1*self.condition_dim:]
        z = self.encoder(X)
        z_c = torch.cat([z,C], dim=1)
        return self.decoder(z_c)
    
class Embedding_model(base):
    def __init__(self, gene_dim, embedding_dim, output_dim, hidden, lr, weight_decay):
        super().__init__(gene_dim, output_dim, lr, weight_decay)

        bottlenet = (len(hidden)-1)//2
        self.latent = hidden[bottlenet]

        encoder_dims = [gene_dim] + hidden[:bottlenet+1]
        decoder_dims = hidden[bottlenet:]
        decoder_dims[0] = self.latent + 32

        self.encoder = _helper_net.MLP(encoder_dims, use_batchnorm=True, use_dropout=0)
        
        self.embedder = nn.Embedding(embedding_dim, 32)

        self.decoder = nn.Sequential(
            _helper_net.MLP(decoder_dims, use_batchnorm=True, use_dropout=0),
            nn.Linear(hidden[-1], output_dim)
            )

    def encode(self, X):
        X_gene = X[:,:50]
        X_condition = torch.argmax(X[:,50:], dim=1)

        Z_x = self.encoder(X_gene)
        Z_c = self.embedder(X_condition)

        return torch.concat([Z_x , Z_c], axis=1)

    def forward(self, X):
        z = self.encode(X)
        return self.decoder(z)