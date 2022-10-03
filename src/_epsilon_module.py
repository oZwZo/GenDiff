import os, sys, math
import numpy as np
import torch
from torch import nn, einsum
from turtle import forward
from typing import Union, Optional
from einops import rearrange
from collections import OrderedDict
sys.path.append('/ssd/users/wergillius/Project/diffuse_differentiate/src')
from _helper_net import SinoidalPositionEmbeddings, SelfAttention, LinearAttention, Residual
import warnings
warnings.simplefilter('once', UserWarning)

# def create_embed_index(list):


#                        ___               _   _              
#                       | __|  _ __   ___ (_) | |  ___   _ _  
#                       | _|  | '_ \ (_-< | | | | / _ \ | ' \ 
#                       |___| | .__/ /__/ |_| |_| \___/ |_||_|
#                             |_|                             

class Epsilon_base(nn.Module):
    r"""
    The unit step estimator for denoising diffusion probabilistic model
    https://arxiv.org/abs/2006.11239

    the reverse process is  :math:`q(\mathbf{x}_t | \mathbf{x}_0) = \cal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1- \bar{\alpha}_t) \mathbf{I})`
    While epsilon is to predict the brownian motion added to corrupt `x_0`, we thus reparameterize the mean of the list distribution by
    :math:`\mathbf{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(  \mathbf{x}_t - \frac{\beta_t}{\sqrt{1- \bar{\alpha}_t}} \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \right)`

    Parameters
    ----------
    var_dim
        int, number of gene as input
    time_emb_dim
        int, the dimension of time embedding
    condition_emb_dim
        please set to 0 if there is no additional condition
    activation_fn
        default with Mish 
    """
    def __init__(self,
                var_dim : int, 
                time_emb_dim : int,
                n_base_perturbs :int,
                condition_emb_dim : int = 0,
                use_batch_index : bool = False,
                activation : Union[str, nn.Module] = "Mish",
                pretrained_embeddings : nn.Module = None,
                ):
        super().__init__()  

        # supposed to be the number of highly var genes
        self.var_dim = var_dim
        self.input_dim = var_dim + time_emb_dim + condition_emb_dim + int(use_batch_index)

        self.use_batch_index = use_batch_index
        self.condition_emb_dim = condition_emb_dim

        # time embeddings  ( which is the same as positional embeddings)
        self.time_mlp = nn.Sequential(
                SinoidalPositionEmbeddings(64),
                nn.Linear(64, time_emb_dim),
                nn.GELU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )

        # condition embeddings
        if condition_emb_dim != 0:
            if pretrained_embeddings is not None:
                assert n_base_perturbs <= pretrained_embeddings.num_embeddings, "the number of token in pretrained embedder is fewer that of the dataset"
                self.embedder = pretrained_embeddings
            else:
                self.embedder = nn.Embedding(
                        num_embeddings = n_base_perturbs, 
                        embedding_dim = condition_emb_dim)
        else:
            print("`condition_emb_dim` is set to 0, training unconditioned models")


        if isinstance(activation, str):
            assert(activation in dir(nn.modules.activation), "invalid activation function")
            self.act_fn = eval(f"nn.modules.activation.{activation}" )
        else:
            self.act_fn = activation() # if  callable nn Module

    
    def _get_model_input(self, x, t, batch = None, c = None):
        device = x.device 
        batch_size = x.shape[0]
        # sanity check
        if self.use_batch_index and (batch is None):
            raise ValueError("Can't predict without batch input ") 
        elif not self.use_batch_index and (batch is not None):
            warnings.warn("batch info is not used with `use_batch_index` turned off")
        else:
            pass

        assert x.shape[1] == self.var_dim, "layer X should have the same dimension as `var_dim`"
        
        if c is not None:
            if self.condition_emb_dim == 0:
                raise ValueError("Can't Pass in condition embedding with `condition_emb_dim` set to 0")
            else:
                # TODO: all zeros ? or set to control embedding?
                c = torch.zeros((batch_size, self.condition_emb_dim), device=device)
        
        time_emb = self.time_mlp(t)

        if batch is None:
            if self.use_batch_index:
                batch_tensor = torch.Tensor([0]).long()
                X = torch.concat([x, time_emb, batch_tensor, c], dim=-1)
            else:
                X = torch.concat([x, time_emb, c], dim=-1)

        
        input_dict = {"full_input":X, "time_point":time_emb, "batch":None, "c":c}
        return input_dict
    
    def forward(self, x, t, batch = None, c=None):
        r"""
        Parameters
        ---------
        x
            Tensor, denoised matrix by the last q_sample process 
        t 
            Tensor, discreted sampled timepoint
        batch
            Tensor:Long, The identifier of Sequencing batch
        c
            Tensor, embedding of the condition
        """
        raise NotImplementedError("base class method `forward` not defined")

#               ___              _    _                   
#              | __|_ __ ___    | |  (_)_ _  ___ __ _ _ _ 
#              | _|| '_ (_-<    | |__| | ' \/ -_) _` | '_|
#              |___| .__/__/    |____|_|_||_\___\__,_|_|  
#                  |_|                                    

class Epsilon_Linear(Epsilon_base):
    r"""
    Gradient estimator network with all linear layers
    """
    def __init__(self, 
                var_dim: int, 
                time_emb_dim : int,
                n_base_perturbs : int,
                condition_emb_dim : int,
                use_batch_index : bool,
                hidden_size: list = [256,128,256],
                activation : Union[str, nn.Module] = "Mish",
                pretrained_embeddings : nn.Module = None,
                ) :
        super().__init__(var_dim, time_emb_dim, n_base_perturbs, condition_emb_dim, use_batch_index, activation, pretrained_embeddings)
        
        
        dimensions = [self.input_dim] + hidden_size + [self.var_dim]

        # define epsilon_theta network
        nns = []
        i = 0
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
            nns.append((f'linear_{i}', nn.Linear(input_dim, output_dim)))
            if i < len(hidden_size):
                nns.append((f'{activation}_{i}', self.act_fn()))
            else:
                # no activation for the last layer
                pass
            i+=1

        nns = OrderedDict(nns)
        self.epsilon_theta = nn.Sequential(nns)
        
        # time embeddings

    def forward(self, x, t, batch = None, c = None):
        # 
        input_dict= self._get_model_input(x, t, batch, c)
        x_ = input_dict['full_input']

        return self.epsilon_theta(x_)


#     ___                  _     _          _   _    _         
#    | __| _ __  ___      | |   (_) _ _    /_\ | |_ | |_  _ _  
#    | _| | '_ \(_-<      | |__ | || ' \  / _ \|  _||  _|| ' \ 
#    |___|| .__//__/      |____||_||_||_|/_/ \_\\__| \__||_||_|
#         |_|                                                  


class Epsilon_LinearAttn(Epsilon_base):
    
    def __init__(self, 
                var_dim: int, 
                time_emb_dim : int,
                n_base_perturbs : int,
                condition_emb_dim : int,
                use_batch_index : bool,
                hidden_size: list = [256, 256, 256],
                qk_dimension: list = 64,
                n_heads: list = 16,
                activation : Union[str, nn.Module] = "Mish",
                pretrained_embeddings : nn.Module = None,
                ) :
        super().__init__(var_dim, time_emb_dim, n_base_perturbs, condition_emb_dim, use_batch_index, activation, pretrained_embeddings)
        
        
        # define epsilon_theta network
        # start with input layer
        nns = [(f'Input_fc_0',nn.Linear(self.input_dim, hidden_size[0])), 
               (f'{activation}_0', self.act_fn())]

        # attention block
        i = 1
        for input_dim, output_dim in zip(hidden_size[:-1], hidden_size[1:]):
            if input_dim == output_dim:
                # only the transformation of the same shape can use residual connection
                nns.append(
                    (f'Res_LinAttn_{i}', Residual(LinearAttention(input_dim, output_dim, n_heads, qk_dimension)))
                    )
            else:
                nns.append(
                    (f'Res_LinAttn_{i}', LinearAttention(input_dim, output_dim, n_heads, qk_dimension))
                    )
            nns.append((f'{activation}_{i}', self.act_fn()))
            i+=1
        
        # output layer
        nns.append((f'Output_fc_{i}', nn.Linear(hidden_size[-1], self.var_dim)))
        nns = OrderedDict(nns)
        self.epsilon_theta = nn.Sequential(nns)

    def forward(self, x, t, batch, c):
        # 
        input_dict= self._get_model_input(x, t, batch, c)
        x_ = input_dict['full_input']

        assert x_.shape[1] == self.input_dim
        return self.epsilon_theta(x_)
