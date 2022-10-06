import os, sys, math
import numpy as np
import torch
from torch import nn, einsum
from torch.nn.modules import activation
import torch.nn.functional as F
from turtle import forward
from typing import Union, Optional
from einops import rearrange
from collections import OrderedDict
from tqdm import tqdm

                            
#  ,--.            ,--.              
#  |  |-.  ,---. ,-'  '-. ,--,--.    
#  | .-. '| .-. :'-.  .-'' ,-.  |    
#  | `-' |\   --.  |  |  \ '-'  |    
#   `---'  `----'  `--'   `--`--'    
                                  
def linear_beta_schedule(timesteps,
                         beta_start : float = 0.0001 ,
                         beta_end : float = 0.02,
                        ):
    r"""
    basic beta sheduler , the scale of noise increase linearly
    :param timesteps: int, the total q_sampling steps  $T$
    """
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps,
                         beta_start : float = 0.0001 ,
                         beta_end : float = 0.02,
                        ):
    r"""
    :param timesteps: int, the total q_sampling steps  $T$
    """
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps)**2

def sigmoid_beta_schedule(timesteps,
                         beta_start : float = 0.0001 ,
                         beta_end : float = 0.02,
                        ):
    r"""
    the beta increase in a sigmoid scale
    """
    beta_range = beta_end - beta_start
    x = torch.linspace(-6, 6, timesteps)
    scaler = torch.sigmoid(x)
    return  scaler * beta_range + beta_start

def cosine_beta_schedule(timesteps, s=0.08):
    r"""
    https://arxiv.org/abs/2102.09672
    """
    n_steps = timesteps + 1
    x = torch.linspace(0, timesteps, n_steps) # [0, timesteps]
    scaled_x = x / timesteps
    alphas_cumprod = torch.cos( 0.5 * torch.pi * ( (scaled_x + s) / (1 + s) ) ) ** 2
    
    # alphas_cumprod[0] is the max value after cos
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])  

    # upper bound and lower bound to stable the values
    betas_clipped = torch.clip(betas, 0.0001, 0.9999) 
    return  betas_clipped


#                                               __         
#                       ___ ___ _ __ _   ___   / /___  ____
#                      (_-</ _ `//  ' \ / _ \ / // -_)/ __/
#                     /___/\_,_//_/_/_// .__//_/ \__//_/   
#                                     /_/                  


class DiffusionSampler_base(object):
    r"""
    base sampler algorithm

    """
    def __init__(self, model, scheduler, timesteps=200, loss_type='huber', **scheduler_kwargs):
        r"""
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        """
        super().__init__()
        self.model = model # epsilon_theta 
        self.total_timestep = timesteps
        scheduler_class = eval(scheduler)
        self.betas = scheduler_class(timesteps, **scheduler_kwargs)
        self._compute_alphas()


        loss_fn_set = {"l1" : nn.L1Loss(),
                        "l2" : nn.MSELoss(),
                        "huber" : nn.SmoothL1Loss()}
        self.loss_fn = loss_fn_set[loss_type]

    
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

    def q_sample(self, x_0, t, noise=None):
        # TO be covered by child class
        raise NotImplementedError()
    
    @torch.no_grad()
    def q_sample_loop(self, x_0,  noise=None): 
        r"""
        $x_{0}$ -> $x_{t}$ ; corrupt matrix to pure noise
        """
        X_t = X_0
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

        for t_i in tqdm(reversed(range(0,self.total_timestep)), "X_t → X_0 :" , total=self.total_timestep):
            
            novar = (t_i == 0)
            T_matrix = torch.full((b,), t_i, device=device, dtype=torch.long)
            X_t = self.p_sample(X_t, T_matrix, batch=batch, c=c, no_var=novar, *args, **kwargs)
            # X_{t-1}
            diffuse_Xs.append(X_t.cpu().numpy())
        
        return diffuse_Xs

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        return self.p_sample_loop(*args, **kwargs)

    def get_eps(self, x_t, t, batch , c, *args, **kwargs):
        return self.model(x_t, t, batch , c, *args, **kwargs)
    
    def p_loss(self, x_0, t, noise=None,  batch=None, c=None, *args, **kwargs):
        r"""
        loss function
        :math: $\mathbf{\epsilon} - \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \|^2$
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        x_t = self.q_sample(x_0, t, noise)
        eps_pred = self.get_eps(x_t, t, batch , c, *args, **kwargs)

        loss = self.loss_fn(noise, eps_pred)
        return loss

#                          ___    ___    ___    __  ___
#                         / _ \  / _ \  / _ \  /  |/  /
#                        / // / / // / / ___/ / /|_/ / 
#                       /____/ /____/ /_/    /_/  /_/  

class DDPM_Sampler(DiffusionSampler_base):
    r"""
    Denoising deffusion probabilistic model

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
        if noise is None:
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
    

#                         ___    ___    ____  __  ___
#                        / _ \  / _ \  /  _/ /  |/  /
#                       / // / / // / _/ /  / /|_/ / 
#                      /____/ /____/ /___/ /_/  /_/  

class DDIM_Sampler(DiffusionSampler_base):
    r"""
    Sampler for Denoising Diffusion Implicit Model 
    https://papers.labml.ai/paper/2010.02502

    Some of the code is editted from 
    https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html
    """
    def __init__(self, 
                 model, 
                 eta : float,
                 discretize:str='linear', 
                 timesteps=200, 
                 loss_type = 'huber',
                 **scheduler_kwargs):
        # ignore input scheduler
        super().__init__( model, "quadratic_beta_schedule", timesteps, loss_type, **scheduler_kwargs)

        self.discretize = discretize
        self.eta = eta
        self._get_discrete_time()
        self._compute_alphas()
        self._compute_ddim_alphas()

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.sigma = self.eta * \
                        torch.sqrt((1-self.a_prev)/(1-self.alphas)) * \
                        torch.sqrt(1 - (self.alphas/self.a_prev))


    def _get_discrete_time(self):
        self.total_timestep
        if self.discretize == 'linear':
            self.time_steps = np.asarray(list(range(0, self.total_timestep, 1))) + 1
        elif self.discretize == 'quadratic':
            linear_step = np.linspace(0, np.sqrt(self.total_timestep * .8), self.total_timestep)
            self.time_steps = (linear_step ** 2).astype(int) + 1
        else:
            raise ValueError("`discretize` should be either 'linear' or 'quadratic'")


    def _compute_ddim_alphas(self):
        r"""
        cover some of the alphas
        """
        self.betas = nn.parameter.Parameter(self.betas.to(torch.float32), requires_grad=False)
        alpha_bar = nn.parameter.Parameter(self.a_bar.to(torch.float32), requires_grad=False)
        
        # $a_t = \bar{\alpha}_t$ for DDIM
        DDIM_alpha = alpha_bar[self.time_steps].clone().to(torch.float32)
        self.alphas = DDIM_alpha
        self.sqrt_a  = torch.sqrt(self.alphas)
        del self.sqrt_a_bar  # remove the same item iherit from base class

        #a_{t-1}
        self.a_prev = torch.cat([alpha_bar[0:1], alpha_bar[self.time_steps[:-1]]])
        #√1-a
        self.sqrt_1_a = torch.sqrt(1 - self.alphas)

    
    def q_sample(self, x_0: torch.Tensor, t:int, noise: Optional[torch.Tensor] = None):
        r"""
        the same as DDPM q_sampling

        :math:`\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{(1- \bar{\alpha}_t)  } \mathbf{\epsilon}`
        :math:`q(\mathbf{x}_t | \mathbf{x}_0) = \cal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1- \bar{\alpha}_t) \mathbf{I})`

        Params
        ------------
        :param x_0: torch.Tensor, the clean matrix
        :param t: torch.Tensor, a list of different time points to sample, time point specific to mini-batch
        :param noise: torch.Tensor, default None, the same shape as `x_0`, a specified noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        ddim_sqrt_a_t = self.extract(self.sqrt_a, t, x_0.shape)
        ddim_sqrt_1_a_t = self.extract(self.sqrt_1_a, t, x_0.shape)

        x_t = x_0 * ddim_sqrt_a_t + noise * ddim_sqrt_1_a_t

        return x_t

    
    @torch.no_grad()
    def p_sample(self, x_t, t, noise=None,  batch=None, c=None, no_var=False, *args, **kwargs):
        r"""
        the DDIM sampling $x_{t} , \hat{x}_0 -> x_{t-1]$
        :math:`x_{t-1} = \sqrt{\alpha_t -1}\hat{x_0} + \sqrt{1 - \alpha_{t-1} - \sigma_t^2} \cdot \epsilon_{\theta}^(t)(x_t) + \sigma_t \epsilon_t`
        :math:`\hat{x}_0 = \frac{x_t - \sqrt{1-\alpha_t} \epsilon_{\theta} (x-t)}{\sqrt{\alpha_t}}`
        """
        if noise is None:
            noise = torch.randn_like(x_t) # \epsilon_t

        a_prev_t = self.extract(self.a_prev, t, x_t.shape)
        a_t = self.extract(self.alphas, t, x_t.shape)
        sqrt_1_a_t = self.extract(self.sqrt_1_a, t, x_t.shape)
        sigma_t = self.extract(self.sigma, t, x_t.shape)

        eps_pred = self.get_eps(x_t, t, batch, c)    # \epsilon_{\theta}(x, t)
        x_0_pred = (x_t - sqrt_1_a_t * eps_pred) / torch.sqrt(a_t)  # \hat{x}_0

        dift = torch.sqrt(1 - a_prev_t - sigma_t**2 ) * eps_pred 

        x_t_1 = torch.sqrt(a_prev_t) * x_0_pred + dift + sigma_t * noise # x_{t-1} 

        if no_var:
            # the last step 
            return torch.sqrt(a_prev_t) * x_0_pred + dift
        else:
            return x_t_1