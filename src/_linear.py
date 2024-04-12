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


class CAE_model(base):
    def __init__(self, encoder_kwargs, decoder_kwargs, lr, weight_decay):
        super().__init__(None, decoder_kwargs['dimensions'][-1], lr, weight_decay)

        
        self.latent = encoder_kwargs['dimensions'][-1]

        self.encoder_dims = encoder_kwargs['dimensions']
        self.decoder_dims = decoder_kwargs['dimensions']

        self.encoder = _helper_net.MLP(**encoder_kwargs)
        self.decoder = _helper_net.MLP(**decoder_kwargs)


    def encode(self, X):
        return self.encoder(X)

    def forward(self, X):
        z = self.encode(X)
        return self.decoder(z)
    
class LatentAdd_CAE(base):
    def __init__(self, input_dim, condition_dim, output_dim, hidden, lr, weight_decay):
        super().__init__(input_dim,  output_dim, lr, weight_decay)

        self.gene_dim = input_dim - condition_dim
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
        x_gene, C = torch.tensor_split(X, (self.gene_dim,), dim=1)
        z = self.encoder(X)
        z_c = torch.cat([z,C], dim=1)
        return self.decoder(z_c)
    
class Embedding_model(base):
    def __init__(self, condition_dim, encoder_kwargs, decoder_kwargs, lr, weight_decay):
        super().__init__(None, decoder_kwargs['dimensions'][-1], lr, weight_decay)

        self.latent = encoder_kwargs['dimensions'][-1]
        self.condition_dim = condition_dim
        self.gene_dim = encoder_kwargs['dimensions'][0] # - condition_dim
        
        # dimensions
        self.encoder_dims = encoder_kwargs['dimensions']
        self.decoder_dims = decoder_kwargs['dimensions']

        # models
        self.embedder = nn.Embedding(condition_dim, 32)
        self.encoder = _helper_net.MLP(**encoder_kwargs)
        self.decoder = _helper_net.MLP(**decoder_kwargs)
        
    def encode(self, X):
        X_gene, X_condition = torch.tensor_split(X, (self.gene_dim,), dim=1)
        
        Z_x = self.encoder(X_gene)
        # Z_c = self.embedder(X_condition.long())
        Z_c = X_condition @ self.embedder.weight

        return torch.cat([Z_x , Z_c], axis=1)

    def forward(self, X):
        z = self.encode(X)
        return self.decoder(z)

class Latent_Interaction(Embedding_model):
    def __init__(self, condition_dim, encoder_kwargs, decoder_kwargs, embedding_dim, lr, weight_decay, control_token=-2):
        """Latent Interaction model

        Args:
            condition_dim (int): the number of unique singleton perturbations
            encoder_kwargs (dict): dict of parameters
            decoder_kwargs (dict): dict of parameters
            embedding_dim (int): the dimension of condition embeddings and also the reduced z_x
            lr (float): learning rate for optimizer
            weight_decay (float): weight decay for optimizer
        """
        super().__init__(condition_dim, encoder_kwargs, decoder_kwargs, lr, weight_decay)

        self.latent = encoder_kwargs['dimensions'][-1]
        self.condition_dim = condition_dim
        self.gene_dim = encoder_kwargs['dimensions'][0] # - condition_dim
        self.control_token = control_token if control_token > 0 else condition_dim - control_token
        
        # dimensions
        self.encoder_dims = encoder_kwargs['dimensions']
        self.decoder_dims = decoder_kwargs['dimensions']

        # models
        self.embedder = nn.Embedding(condition_dim, embedding_dim)
        self.zx_project = nn.Linear(self.encoder_dims[-1], embedding_dim)

        self.encoder = _helper_net.MLP(**encoder_kwargs)
        self.decoder = _helper_net.MLP(**decoder_kwargs)

    def process_conditions(self, X_condition):
        """collapse multiple tokens
        Args:
            X_condition (torch.Tensor): one hot encoded TF vectors
        """
        device = X_condition.device
        n_condition = X_condition.sum(axis=1)
        maxm = torch.max(n_condition) # max multiplexing

        C_list = []

        if maxm == 1:
            # all is one
            where_c = torch.where(X_condition!= 0)[1]
            control_index = torch.full_like(where_c, self.control_token)
            merge = torch.stack([where_c, control_index]).T
            C_list = self.embedder(merge)
        else:
            for i in range(n_condition.shape[0]):
                where_c = torch.where(X_condition[i])[0]

                if n_condition[i] != maxm:
                    to_pad = maxm - n_condition[i]
                    control_index = torch.full((to_pad,), self.control_token)
                    merge_i = torch.cat([where_c, control_index])
                    C_i = self.embedder(merge_i)
                    C_list.append(C_i)
            C_list = torch.stack(C_list)
        return C_list

    def interact(self, z_x, C_i):
        """latent interaction function

        Args:
            z_x (torch.Tensor): the expression represenation, output of self.encoder
            C_i (torch.Tensor): the Vector of condition single-ton embeddings

        Returns:
            tuple of Tensor: 
            z : join embeddings, [batch size, exp-rep dim + cond-emb dim]
            z_c : the merged condition embeddings, [batch size, cond-emb dim]
        """
        key_x = self.zx_project(z_x)

        # compute attention
        # prod = torch.einsum("bi, bci->bc", key_x, C_i)
        attention = torch.softmax(prod, dim=1)
        # z_c = torch.einsum("bci, bc->bi", C_i, attention)

        z = torch.cat([z_x , z_c], axis=1)
        return z, z_c, attention
        
    def encode(self, X):
        X_gene, X_condition = torch.tensor_split(X, (self.gene_dim,), dim=1)
        
        Z_x = self.encoder(X_gene)
        key_x = self.zx_project(z_x)
        C_i = self.process_conditions(X_condition)

        z, z_c, attention = self.interact(Z_x, C_i)
        return z, z_c, attention

    def forward(self, X):
        z, z_c, attention = self.encode(X)
        return self.decoder(z)
    

class Pretrained_GenDiff(base):
    def __init__(self, gene_dim, condition_dim, pretrain_model, prior_coef, prior_intercept, lr, weight_decay, update_fc=True, update_pretrain=True):
        super().__init__(gene_dim + condition_dim, prior_coef.shape[1], lr, weight_decay)
        
        # dimensions
        self.gene_dim = gene_dim
        self.condition_dim = condition_dim
        self.input_dim = gene_dim + condition_dim

        # prior
        self.update_fc = update_fc
        self.prior_coef = torch.from_numpy(prior_coef).float()
        self.prior_intercept = torch.from_numpy(prior_intercept).float()

        # models
        self.update_pretrain = update_pretrain
        self.pretrain_model = pretrain_model
        self.fc_output = nn.Linear(self.input_dim, gene_dim)
        self.insert_prior(update=self.update_fc)

        # update pretrian
        if not self.update_pretrain:
            for p in self.pretrain_model.parameters():
                p.requires_grad = False

    def insert_prior(self, update):
        self.fc_output.weight = torch.nn.Parameter(self.prior_coef, requires_grad = update)
        self.fc_output.bias = torch.nn.Parameter(self.prior_intercept, requires_grad = update)

    def configure_optimizers(self):
        return super().configure_optimizers()
    
    def encode(self,X):
        return self.pretrain_model.encode(X)

    def forward(self, X):
        X_gene, X_condition = torch.tensor_split(X, (self.gene_dim,), dim=1)

        X_upsampled = self.pretrain_model.forward(X)
        X_out = torch.cat([X_upsampled, X_condition], dim=1)
        delta_X_upsampled = self.fc_output(X_out)
        delta_X_straight = self.fc_output(X)

        return (delta_X_upsampled + delta_X_straight)/2