#!
import os
import sys
import cpa
import scanpy as sc

import torch
from torch import nn
from torch.distributions import LogNormal, NegativeBinomial, Normal, kl_divergence

import scvi
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data

# my simple VAE
class my_VAE(BaseModuleClass):
    def __init__(self, n_input, n_latent):
        super().__init__()
        # z encoder
        self.var_encoder = FCLayers(n_in=n_input, 
                                    n_out=n_latent, 
                                    activation_fn=Exponential_act
                                   )
        self.mean_encoder = FCLayers(n_in=n_input, 
                                     n_out=n_latent, 
                                     use_activation=False
                                    )
        # size encoder
        self.size_var_encoder = FCLayers(n_in=n_input, 
                                    n_out=1, 
                                    use_activation=False
                                   )
        self.size_mean_encoder = FCLayers(n_in=n_input, 
                                     n_out=1, 
                                     use_activation=False
                                    )
        # dispersion : the dimension should be the same as the n_var
        self.log_theta = nn.Parameter(torch.randn(n_input))
        
        # decoder
        self.decoder = FCLayers(n_in=n_latent,
                                n_out=n_input,
                                activation_fn=activation.Softmax
                               )
    
    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        input_dict = dict(x=x)
        return input_dict
    
    @auto_move_data
    def inference(self,x):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        # log input !!
        x_ = torch.log(x + 1)
        
        # get mean and var
        qz_mean = self.mean_encoder(x_)
        qz_var = self.var_encoder(x_)
        # reparameterize
        prior = Normal(loc=qz_mean, scale=torch.sqrt(qz_mean))
        z_n = prior.rsample()
        
        # do the same thing for size factor
        l_mean = self.size_mean_encoder(x_)
        l_var = self.size_var_encoder(x_)
        # sample ln
        l_n = LogNormal(l_mean, torch.sqrt(l_var)).rsample()
        
        # the output dict
        outputs = dict(
            qz_m=qz_mean, qz_v=qz_var, z=z_n,
            l_m=l_mean, l_v=l_var, l=l_n
                      )
        return outputs
        
    def _get_generative_input(self, tensors, inference_output):
        z = inference_output["z"]
        l = inference_output["l"]
        
        return {"z":z, "library":l}
        
        
    @auto_move_data
    def generative(self, z, library):
        
        # get the normalized mean of negetive binomial 
        px_scale = self.decoder(z)
        
        # timing the library size
        px_rate = px_scale * library
        
        # get the dispersion
        theta = torch.exp(self.log_theta)
        
        NegBino_params = {'px_scale':px_scale,
                          'px_rate':px_rate,
                          'theta':log_theta
                         }
        return NegBino_params
    
    def loss(self, tensors, inference_output, generative_output):
        """
        # here, we would like to form the ELBO. There are two terms:
        #   1. one that pertains to the likelihood of the data
        #   2. one that pertains to the variational distribution
        """
        # so we extract all the required information
        x = tensors[REGISTRY_KEYS.X_KEY]
        qz_m = inference_output['qz_m']
        qz_v = inference_output['qz_v']
        z = inference_output['z']
        library = inference_output['l']
        
        px_scale = generative_output['px_scale']
        px_rate = generative_output['px_rate']
        theta = generative_output['theta']
        
        # term 1 : compute the likelihood
        # the pytorch NB distribution uses a different parameterization
        # so we must apply a quick transformation 
        # (included in scvi-tools, but here we use the pytorch code)
        nb_logit = (px_rate + 1e-4).log() - (theta + 1e-4).log()
        px = NegativeBinomial(total_count=theta, logits=nb_logits)
        log_lik = px.log_prob(x).sum(dim=-1)
        
        # term 2 : KL divergence 
        # standard gaussian
        prior = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
        posterior = Normal(qz_m, qz_v)
        kld = kl_divergence(prior, posterior).sum(dim=1)
        
        elbo = log_lik - kld
        loss = torch.mean(-elbo)
        return LossRecorder(loss, -log_like, kld, 0.0)