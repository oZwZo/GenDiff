from json import decoder
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
        default `Mish`
    pretrained_embeddings
        nn.Module, 

    Return 
    -----------
        nn.Module; This is the base parent class awaiting for inheritence. Don't call it directly.
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
        self.input_dim = var_dim + time_emb_dim + int(use_batch_index)

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
        
        # process condition info
        if c is None:
            if self.embedder is None:
                c = torch.zeros((batch_size, self.condition_emb_dim), device=device)
            else:
                control_tokens = torch.full((batch_size,), 0, device=device).long()
                c_emb = self.embedder(control_tokens)

        else:
            if self.condition_emb_dim == 0:
                raise ValueError("Can't Pass in condition embedding with `condition_emb_dim` set to 0")
            # use the embedder for conditioned input
            c_emb = self.embedder(c)
        
        time_emb = self.time_mlp(t)

        # process batch info
        if batch is None:
            if self.use_batch_index:
                batch_tensor = torch.Tensor([0]).long()
                X = torch.concat([x, time_emb, batch_tensor], dim=-1)
            else:
                X = torch.concat([x, time_emb], dim=-1)

        
        input_dict = {"full_input":X, "time_point":time_emb, "batch":None, "condition":c_emb}
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
    Gradient estimator network with all linear layers. 
    We will adapt the VAE-like architecture for the convience of applying perturbation on the latent space
    
    (what's the epsilon)
    The unit step estimator for denoising diffusion probabilistic model https://arxiv.org/abs/2006.11239.
    the reverse process is  :math:`q(\mathbf{x}_t | \mathbf{x}_0) = \cal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1- \bar{\alpha}_t) \mathbf{I})`
    While epsilon is to predict the brownian motion added to corrupt `x_0`, we thus reparameterize the mean of the list distribution by
    :math:`\mathbf{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(  \mathbf{x}_t - \frac{\beta_t}{\sqrt{1- \bar{\alpha}_t}} \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \right)`

    Parameters
    ---------------

    ---Linear specific----
    hidden_size
        list, the key parameter to the model architure.
        The length of list will determine the # of hidden layers.
        We encourage to have one layer in the middle to be `condition_emb_dim` 

    ---Base param----
    var_dim
        int, number of gene as input
    time_emb_dim
        int, the dimension of time embedding
    condition_emb_dim
        please set to 0 if there is no additional condition
    activation_fn
        default `Mish`
    pretrained_embeddings
        nn.Module, 

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

        # insert the conditional embeddings
        if self.condition_emb_dim in dimensions:
            dim_ay = np.asarray(dimensions)
            posi=np.where(dim_ay == self.condition_emb_dim)[0]
            # choose the one in the middle
            middle = len(dimensions)//2 +1
            self.latent_layer = posi[np.argmin(posi - middle)]
        else:
            self.latent_layer = len(dimensions)//2 + 1
            dimensions.insert(self.latent_layer, self.condition_emb_dim)
            warnings.warn("\n\tThe network has no shared hidden dimension as condition embeddings, inserting it into the %dth layer" %self.latent_layer)

        # define epsilon_theta network
        encoder_dims = dimensions[:self.latent_layer+1]
        self.encoder_dims = encoder_dims
        decoder_dims = dimensions[self.latent_layer:]
        self.decoder_dims = decoder_dims

        i = 0
        # define encdoer
        encoder = []
        for input_dim, output_dim in zip(encoder_dims[:-1], encoder_dims[1:]):
            encoder.append((f'linear_{i}', nn.Linear(input_dim, output_dim)))
            if i < len(encoder_dims):
                encoder.append((f'{activation}_{i}', self.act_fn()))
            else:
                # no activation for the latent layer
                pass
            i+=1
        self.latent_activator = self.act_fn()
        # define decdoer
        decoder = []
        for input_dim, output_dim in zip(decoder_dims[:-1], decoder_dims[1:]):
            decoder.append((f'linear_{i}', nn.Linear(input_dim, output_dim)))
            if i < len(decoder_dims):
                decoder.append((f'{activation}_{i}', self.act_fn()))
            else:
                # no activation for the last layer
                pass
            i+=1

        self.epsilon_theta = nn.ModuleDict(
            {'encoder': nn.Sequential(OrderedDict(encoder)), 
            'decoder': nn.Sequential(OrderedDict(decoder))}
            )
        
        # time embeddings

    def forward(self, x, t, batch = None, c = None):
        # input
        input_dict= self._get_model_input(x, t, batch, c)
        x_ = input_dict['full_input']
        c_emb = input_dict['condition']
        c_emb = c_emb.sum(dim=1) if len(c_emb.shape) == 3 else c_emb
        # encode
        z = self.epsilon_theta['encoder'](x_)
        assert z.shape == c_emb.shape, 'cell latent and condition latent is not in the same space'
        # pertub and decode
        z_c = z + c_emb
        z_c_act = self.act_fn()(z_c)
        return self.epsilon_theta['decoder'](z_c_act)


#     ___                  _     _          _   _    _         
#    | __| _ __  ___      | |   (_) _ _    /_\ | |_ | |_  _ _  
#    | _| | '_ \(_-<      | |__ | || ' \  / _ \|  _||  _|| ' \ 
#    |___|| .__//__/      |____||_||_||_|/_/ \_\\__| \__||_||_|
#         |_|                                                  


class Epsilon_LinearAttn(Epsilon_base):
    r"""
    VAE-like gradient estimator using linear attention opeartor; 
    Linear Attention by Shen et al 2018 : https://arxiv.org/abs/1812.01243
    It is encouraged to use this architecture when building a large model (large hidden/ very deep).

    Parameters
    ----------

     ---LinAttn specific----
    hidden_size
        list, the key parameter to the model architure.
        The length of list will determine the # of hidden layers.
        We encourage to have one layer in the middle to be `condition_emb_dim` 
    qk_dimension
        int: 64,  the dimension for inner query key and value vector during the inner calculatio of self-attention.\
        the actual hidden size of qkv layer is determined by `qk_dimension` * `n_heads` 
    n_heads
        int : 64, the number of mult-head attention operation. The # of heads determine the # atten matrix
        the actual hidden size of qkv layer is determined by `qk_dimension` * `n_heads` 

    ---Base param----
    var_dim
        int, number of gene as input
    time_emb_dim
        int, the dimension of time embedding
    condition_emb_dim
        please set to 0 if there is no additional condition
    activation_fn
        default `Mish`
    pretrained_embeddings
        nn.Module, 
    """
    
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

        # ---- split layers ---------
        dimensions = [self.input_dim] + hidden_size + [self.var_dim]

        # insert the conditional embeddings
        if self.condition_emb_dim in dimensions:
            dim_ay = np.asarray(dimensions)
            posi=np.where(dim_ay == self.condition_emb_dim)[0]
            # choose the one in the middle
            middle = len(dimensions)//2 
            if middle in posi:
                self.latent_layer = middle
            else:
                self.latent_layer = posi[np.argmin(posi - middle)]
        else:
            self.latent_layer = len(dimensions)//2 + 1
            dimensions.insert(self.latent_layer, self.condition_emb_dim)
            warnings.warn("\n\tThe network has no shared hidden dimension as condition embeddings, inserting it into the %dth layer" %self.latent_layer)

        # - - define epsilon_theta network - - 
        encoder_dims = dimensions[1:self.latent_layer+1]
        self.encoder_dims = encoder_dims
        decoder_dims = dimensions[self.latent_layer:-1]
        self.decoder_dims = decoder_dims

        # encoder attention block
        i = 1
        encoder = [(f'Input_fc_0',nn.Linear(self.input_dim, hidden_size[0])), 
                    (f'{activation}_0', self.act_fn())]
        
        for input_dim, output_dim in zip(encoder_dims[:-1], encoder_dims[1:]):
            if input_dim == output_dim:
                # only the transformation of the same shape can use residual connection
                encoder.append(
                    (f'Res_LinAttn_{i}', Residual(LinearAttention(input_dim, output_dim, n_heads, qk_dimension)))
                    )
            else:
                encoder.append(
                    (f'LinAttn_{i}', LinearAttention(input_dim, output_dim, n_heads, qk_dimension))
                    )
            if i != len(encoder_dims):
                encoder.append((f'{activation}_{i}', self.act_fn()))
            i+=1

        # latent
        self.latent_activator = self.act_fn()
        
        # decoder block
        decoder = []
        for input_dim, output_dim in zip(decoder_dims[:-1], decoder_dims[1:]):
            if input_dim == output_dim:
                # only the transformation of the same shape can use residual connection
                decoder.append(
                    (f'Res_LinAttn_{i}', Residual(LinearAttention(input_dim, output_dim, n_heads, qk_dimension)))
                    )
            else:
                decoder.append(
                    (f'LinAttn_{i}', LinearAttention(input_dim, output_dim, n_heads, qk_dimension))
                    )
            decoder.append((f'{activation}_{i}', self.act_fn()))

        decoder.append((f'Output_fc_{i}', nn.Linear(hidden_size[-1], self.var_dim)))
        
        self.epsilon_theta = nn.ModuleDict(
            {'encoder': nn.Sequential(OrderedDict(encoder)), 
            'decoder': nn.Sequential(OrderedDict(decoder))}
            )

    def forward(self, x, t, batch = None, c = None):
        # input
        input_dict= self._get_model_input(x, t, batch, c)
        x_ = input_dict['full_input']
        c_emb = input_dict['condition']
        c_emb = c_emb.sum(dim=1)  if len(c_emb.shape) == 3 else input_dict['condition']
        # encode
        z = self.epsilon_theta['encoder'](x_)
        assert z.shape == c_emb.shape, 'cell latent and condition latent is not in the same space'
        # pertub and decode
        z_c = z + c_emb
        z_c_act = self.act_fn()(z_c)
        return self.epsilon_theta['decoder'](z_c_act)
