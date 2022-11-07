import os
import torch
from torch import optim
import os, sys, math
import numpy as np
import torch
from torch import nn, einsum
from torch.nn.modules import activation
import torch.nn.functional as F
from turtle import forward
from tqdm import tqdm
from typing import Union, Optional
from einops import rearrange
from collections import OrderedDict
import pytorch_lightning as pl
from  _sampler import linear_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule, cosine_beta_schedule



#                                               __         
#                       ___ ___ _ __ _   ___   / /___  ____
#                      (_-</ _ `//  ' \ / _ \ / // -_)/ __/
#                     /___/\_,_//_/_/_// .__//_/ \__//_/   
#                                     /_/                  


class DiffusionSampler_base(pl.LightningModule):
    r"""
    base sampler algorithm

    """
    def __init__(self, model, scheduler, timesteps=200, loss_type='huber', **scheduler_kwargs):
        r"""
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model # epsilon_theta 
        self.total_timestep = timesteps
        scheduler_class = eval(scheduler)
        self.betas = scheduler_class(timesteps, **scheduler_kwargs)
        self._compute_alphas()


        loss_fn_set = {"l1" : nn.L1Loss(),
                        "l2" : nn.MSELoss(),
                        "huber" : nn.SmoothL1Loss()}
        self.loss_fn = loss_fn_set[loss_type]

        self.metric_func = {"loss":self.loss_fn}
    
    def _compute_alphas(self):
        r"""
        useful quantities on alphas
        :math:`\alpha_t := 1 - \beta_t` and `\bar{\alpha}t := \Pi_{s=1}^{t} \alpha_s`
        """
        self.alphas = 1. - self.betas
        self.a_bar = torch.cumprod(self.alphas, dim=0) # cumulatrive product of alphas
        self.a_bar_prev = F.pad(self.a_bar[:-1], (1, 0), value=1.0)
        self.sqrt_recip_a = torch.sqrt(1.0 / self.alphas)

        # used in the mean for q_sample process
        self.sqrt_a_bar = torch.sqrt(self.a_bar)
        # used in the var for q_sample process
        self.sqrt_1_a_bar =  torch.sqrt( 1 - self.a_bar)

    
    def extract(self, a, t, x_shape):
        """
        barch-wise indexing of alpha array
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def configure_optimizers(self):
        lr = 1e-3

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def q_sample(self, x_0, t, noise=None):
        # TO be covered by child class
        raise NotImplementedError()
    
    @torch.no_grad()
    def q_sample_loop(self, x_0,  noise=None): 
        r"""
        $x_{0}$ -> $x_{t}$ ; corrupt matrix to pure noise
        """
        X_t = x_0
        b = X_t.shape[0]
        device = X_t.device
        corrupted_Xs = []
        for t_i in tqdm(range(0,self.total_timestep), "Adding noise :" , total=self.total_timestep):
            T_matrix = torch.full((b,), t_i, device=device, dtype=torch.long)
            X_t = self.q_sample(X_t, T_matrix)
            corrupted_Xs.append(X_t)
        return corrupted_Xs
    
    @torch.no_grad()
    def p_sample(self):
        raise NotImplementedError()
    
    @torch.no_grad()
    def p_sample_loop(self, shape, batch=None, c=None, *args, **kwargs):
        r"""
        $x_{t}$ -> $x_{0}$ ; generate matrix from pure noise
        """
        device = next(self.model.parameters()).device
        
        b = shape[0]

        # start from pure noise 
        X_t = torch.randn(shape, device=device)
        diffuse_Xs = []

        for t_i in tqdm(reversed(range(0,self.total_timestep)), "Z~N(0,I) → X_0 :" , total=self.total_timestep):
            
            novar = (t_i == 0)
            T_matrix = torch.full((b,), t_i, device=device, dtype=torch.long)
            X_t = self.p_sample(X_t, T_matrix, batch=batch, c=c, no_var=novar, *args, **kwargs)
            # X_{t-1}
            diffuse_Xs.append(X_t.cpu().numpy())
        
        return diffuse_Xs
    
    @torch.no_grad()
    def evaluation_loop(self, x_0, start_from=None, batch=None, c=None, *args, **kwargs):
        # sanity check
        if start_from is None:
            # can not smaller than 1
            start_from = max(1, self.total_timestep - 1)

        b = x_0.shape[0]
        device = x_0.device
        if c is not None:
            assert c.shape[0] == b, "condition should have the same batch-size as x-0"
            c = c.to(device)

        # -- 1. Q sample  --
        X_corrupted = self.q_sample_loop(x_0)

        # -- 2. One step P sample -- 
        X_t = X_corrupted[start_from]
        stepwise_P = [X_t]
        for t in tqdm(range(start_from, -1, -1), "X_t → X_{t-1}:"):
            timepoint_TS = torch.full((b,), t).long().to(device)
            X_t = self.p_sample(X_corrupted[t], timepoint_TS, batch=None, c=None, *args, **kwargs)
            stepwise_P.append(X_t)
        stepwise_P = [x.detach().cpu().numpy() for x in stepwise_P][::-1]

        # -- 3. Reconstruction -- 
        X_t = X_corrupted[start_from]
        Recon_X = [X_t]
        for t in tqdm(range(start_from, -1, -1), "X_T → X_0\t:"):
            timepoint_TS = torch.full((b,), t).long().to(device)
            X_t = self.p_sample(X_t, timepoint_TS, batch=None, c=None, *args, **kwargs)
            Recon_X.append(X_t)
        Recon_X = [x.detach().cpu().numpy() for x in Recon_X][::-1]

        # -- 4. de novo P sample -- 
        Denovo_X = self.p_sample_loop(x_0.shape, batch=None, c=None, *args, **kwargs)

        X_corrupted = [x_0.detach().cpu().numpy()] + [x.detach().cpu().numpy() for x in X_corrupted]

        return {"Q":X_corrupted, "S":stepwise_P, "R":Recon_X, "P":Denovo_X}

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        return self.p_sample_loop(*args, **kwargs)

    def forward(self, x_t, t, batch , c, *args, **kwargs):
        return self.model(x_t, t, batch , c, *args, **kwargs)
    
    def p_loss(self, x_0, t, noise=None,  batch=None, c=None, *args, **kwargs):
        r"""
        :math: $\mathbf{\epsilon} - \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \|^2$
        """
        if len(noise) == 0:
            noise = torch.randn_like(x_0)

        x_t = self.q_sample(x_0, t, noise)
        eps_pred = self.forward(x_t, t, batch , c, *args, **kwargs)

        loss = self.loss_fn(noise, eps_pred)
        return loss
    
    def training_step(self, train_batch, batch_idx):
        # get data from batch
        x_0, exp_batch, c, noise, t = train_batch
        if len(exp_batch) == 0:
            exp_batch = None

        device = x_0.device
        if len(t) == 0:
            t = torch.randint(0, self.total_timestep-1, (x_0.shape[0], ), device=device).long()

        if len(noise) == 0:
            noise = torch.randn_like(x_0)

        loss = self.p_loss(x_0, t, noise,  exp_batch, c)
        self.log('train_loss', loss)
        return loss
    
    def _shared_eval_step(self, x, t, batch , c, noise, metric_func, *args, **kwargs):

        # get data from batch
        device = x.device
        
        if len(noise) == 0:
            noise = torch.randn_like(x)

        device = x.device
        x_t = self.q_sample(x, t, noise)
        eps_pred = self.forward(x_t, t, batch , c, *args, **kwargs)

        metrics = {}
        for key,func in metric_func.items():
            metrics[key] = func(noise, eps_pred)

        return metrics
    
    
    def validation_step(self, val_batch, batch_idx):
        # loss over many time

        x_0, batch, c, noise, t = val_batch
        if len(batch) == 0:
            batch = None

        if len(t) == 0:
            metric_records = {k:[] for k in self.metric_func}

            for t in range(0, self.total_timestep, 2):
                t = torch.full((x_0.shape[0],), t, device=x_0.device).long()
                metric_t = self._shared_eval_step(x_0, t, batch , c, noise, self.metric_func)

                for k, v in metric_t.items():
                    metric_records[k].append(v.detach().cpu().numpy())
            metrics = {"val_%s"%k:np.mean(v) for k,v in metric_records.items()} # take the mean
        else:
            metrics = self._shared_eval_step(x_0, t, batch , c, noise, self.metric_func)
        
        metrics["t_MaxError"] = np.argmax(metric_records["loss"])

        self.log_dict(metrics)
        return metrics

    def test_step(self, test_batch, batch_idx):
        # loss over many time

        x_0, batch, c, noise, t = test_batch
        if len(batch) == 0:
            batch = None

        if len(t) == 0:
            #if t is an empty list, t is not specific to x, c, so we evaluate
            metric_records = {k:[] for k in self.metric_func}
            for t in range(0, self.total_timestep, 2):
                t = torch.full((x_0.shape[0],), t, device=x_0.device).long()
                metric_t = self._shared_eval_step(x_0, t, batch , c, noise, self.metric_func)

                for k, v in metric_t.items():
                    metric_records[k].append(v)
            metrics = {"test_%s"%k:torch.mean(v) for k,v in metric_records.items()} # take the mean
        else:
            metrics = self._shared_eval_step(x_0, t, batch , c, noise, self.metric_func)
        metrics["t_MaxError"] = torch.argmax(metric_records["loss"])
        
        self.log_dict(metrics)
        return metrics



#                          ___    ___    ___    __  ___
#                         / _ \  / _ \  / _ \  /  |/  /
#                        / // / / // / / ___/ / /|_/ / 
#                       /____/ /____/ /_/    /_/  /_/  

class DDPM_Sampler(DiffusionSampler_base):
    r"""
    (improved) Denoising deffusion probabilistic model 

    """
    def __init__(self, model : nn.Module,
                       scheduler: str, 
                       loss_type : str, 
                       timesteps=200, 
                       **scheduler_kwargs):
        super().__init__(model, scheduler, timesteps, loss_type,  **scheduler_kwargs)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.sigma = self.betas * (1. - self.a_bar_prev) / (1. - self.a_bar)
        
        # define loss
        
    
    def q_sample(self, x_0, t, noise=None):
        r"""
        sample corrupted samples at time t from clean matrix at one step
        
        :math:`\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{(1- \bar{\alpha}_t)  } \mathbf{\epsilon}`
        :math:`q(\mathbf{x}_t | \mathbf{x}_0) = \cal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1- \bar{\alpha}_t) \mathbf{I})`

        Params
        ------------
        :param x_0: torch.Tensor, the clean matrix
        :param t: torch.Tensor, a list of different time points to sample, time point specific to mini-batch
        :param noise: torch.Tensor, default None, the same shape as `x_0`, a specified noise
        """
        if len(noise) == 0:
            noise = torch.randn_like(x_0)

        sqrt_a_bar_t = self.extract(self.sqrt_a_bar, t , x_0.shape)
        sqrt_1_a_bar_t = self.extract(self.sqrt_1_a_bar, t, x_0.shape)
        
        # mean : $\mathbf{x}_t; \sqrt{\bar{\alpha}_t} x_0$
        # sigma^2 : $(1- \bar{\alpha}_t) \mathbf{I}$
        x_t = x_0 * sqrt_a_bar_t + noise * sqrt_1_a_bar_t
        
        return x_t

    @torch.no_grad()
    def p_sample(self, x_t, t, noise=None,  batch=None, c=None, no_var=False, *args, **kwargs):
        r"""
        $x_{t}$ -> $x_{t-1}$ ; take one reverse denoising step
        :math:`x_t-1 = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\mathcal{Z}_{\theta}(x_t,t)) + \sigma_t \mathbf{z}`
        """

        betas_t = self.extract(self.betas, t, x_t.shape)              #ß=1-∂t
        sqrt_recip_a_t = self.extract(self.sqrt_recip_a, t, x_t.shape)  #1/√∂t
        sqrt_1_a_bar_t = self.extract(self.sqrt_1_a_bar, t, x_t.shape)  #√1-åt 

        # mean
        eps_theta = self.model(x_t, t, batch, c, *args, **kwargs)
        mean = sqrt_recip_a_t * ( x_t - (betas_t/sqrt_1_a_bar_t)* eps_theta)
        # variancce
        sigma_t = self.extract( self.sigma, t, x_t.shape )
        noise = torch.randn_like(x_t)
        reparam_var = torch.sqrt(sigma_t) * noise

        # X_{t-1}
        X_t_1 = mean if no_var else mean+reparam_var # only when t = 0
        return X_t_1

class DDPM_reconX(DDPM_Sampler):
    def __init__(self,model : nn.Module,
                       scheduler: str, 
                       loss_type : str, 
                       timesteps=200, 
                       **scheduler_kwargs):
        super().__init__(model, scheduler, loss_type, timesteps,**scheduler_kwargs)

    @torch.no_grad()
    def p_sample(self, x_t, t, noise=None,  batch=None, c=None, no_var=False, *args, **kwargs):
        r"""
        the denoise process that parmeterised on $\mu$
        $\mu_{t} -> \mu_{t-1}$
        """

        betas_t = self.extract(self.betas, t, x_t.shape)              #ß=1-∂t
        sqrt_recip_a_t = self.extract(self.sqrt_recip_a, t, x_t.shape)  #1/√∂t
        sqrt_1_a_bar_t = self.extract(self.sqrt_1_a_bar, t, x_t.shape)  #√1-åt 

        # mean
        mean = self.model(x_t, t, batch, c, *args, **kwargs)

        # variancce
        sigma_t = self.extract( self.sigma, t, x_t.shape )
        noise = torch.randn_like(x_t) if len(noise) == 0 else noise
        reparam_var = torch.sqrt(sigma_t) * noise

        # X_{t-1}
        X_t_1 = mean if no_var else mean+reparam_var # only when t = 0
        return X_t_1
    
    def p_loss(self, x_0, t, noise=None,  batch=None, c=None, *args, **kwargs):
        r"""
        loss function
        :math: $\mathbf{\epsilon} - \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \|^2$
        """
        if len(noise) == 0:
            noise = torch.randn_like(x_0)

        x_t = self.q_sample(x_0, t, noise)
        x_t_plus = self.q_sample(x_0, t+1, noise)
        x_recon = self.forward(x_t_plus, t, batch , c, *args, **kwargs)

        loss = self.loss_fn(x_t, x_recon)
        return loss
    
    def _shared_eval_step(self, x, t, batch , c, noise, metric_func, *args, **kwargs):

        # get data from batch
        device = x.device
        
        if len(noise) == 0:
            noise = torch.randn_like(x)

        device = x.device
        x_t = self.q_sample(x, t, noise)
        eps_pred = self.forward(x_t, t, batch , c, *args, **kwargs)

        metrics = {}
        for key,func in metric_func.items():
            metrics[key] = func(x_t, eps_pred) # this one is different

        return metrics