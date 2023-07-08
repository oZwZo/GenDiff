from json import decoder
import os, sys, math
import numpy as np
import torch
from torch import nn, einsum
import torch.distributions as db
from turtle import forward
from typing import Union, Optional
from einops import rearrange
from collections import OrderedDict
# sys.path.append('/ssd/users/wergillius/Project/diffuse_differentiate/src')
from ._helper_net import SinoidalPositionEmbeddings, SelfAttention, LinearAttention, Residual
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
    gene_dim
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
                gene_dim : int, 
                time_emb_dim : int,
                n_base_perturbs :int,
                condition_emb_dim : int = 0,
                use_batch_index : bool = False,
                activation : Union[str, nn.Module] = "Mish",
                pretrained_embeddings : nn.Module = None,
                ):
        super().__init__()  

        # supposed to be the number of highly var genes
        self.gene_dim = gene_dim
        self.input_dim = gene_dim + time_emb_dim + int(use_batch_index)

        self.use_batch_index = use_batch_index
        self.condition_emb_dim = condition_emb_dim

        # time embeddings  ( which is the same as positional embeddings)
        if time_emb_dim == 0:
            self.time_mlp = None
        else:
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
            # assert(activation in dir(nn.modules.activation), "invalid activation function")
            self.act_fn = eval(f"nn.modules.activation.{activation}" )
        else:
            self.act_fn = activation() # if  callable nn Module

    
    def _get_model_input(self, x, c, batch = None, t = None):
        device = x.device 
        batch_size = x.shape[0]
        # sanity check
        if self.use_batch_index and (batch is None):
            raise ValueError("Can't predict without batch input ") 
        elif not self.use_batch_index and (batch is not None):
            warnings.warn("batch info is not used with `use_batch_index` turned off")
        else:
            pass

        assert x.shape[1] == self.gene_dim, "layer X should have the same dimension as `gene_dim`"
        
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
        
        if self.time_mlp is not None:
            time_emb = self.time_mlp(t)

            # process batch info
            if batch is None:
                if self.use_batch_index:
                    batch_tensor = torch.Tensor([0]).long()
                    X = torch.cat([x, time_emb, batch_tensor], dim=-1)
                else:
                    X = torch.cat([x, time_emb], dim=-1)
            else:
                X = torch.cat([x, time_emb, batch], dim=-1)
        else:
            X = x
            time_emb = None

        
        input_dict = {"full_input":X, "time_point":time_emb, "batch":None, "condition":c_emb}
        return input_dict

    def get_DeltaX(self, batch_data):
        raise NotImplementedError("base class method `get_DeltaX` not defined")
    
    def forward(self, x, c, batch = None, t=None):
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
    gene_dim
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
                gene_dim: int, 
                time_emb_dim : int,
                n_base_perturbs : int,
                condition_emb_dim : int,
                use_batch_index : bool,
                hidden_size: list = [256,128,256],
                activation : Union[str, nn.Module] = "Mish",
                pretrained_embeddings : nn.Module = None,
                last_batch_norm: nn.Module = True
                ) :
        super().__init__(gene_dim, time_emb_dim, n_base_perturbs, condition_emb_dim, use_batch_index, activation, pretrained_embeddings)
        
        
        dimensions = [self.input_dim] + hidden_size + [self.gene_dim]

        # insert the conditional embeddings
        if self.condition_emb_dim in dimensions:
            dim_ay = np.asarray(dimensions)
            posi = np.where(dim_ay == self.condition_emb_dim)[0]
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
        self.variational = False

        self.i = 0
        # define encdoer
        self.activation = activation
        self.latent_activator = self.act_fn()
        encoder = self.define_block(encoder_dims, batch_norm=False)
        decoder = self.define_block(decoder_dims, True, last_batch_norm)
        
        self.epsilon_theta = nn.ModuleDict(
            {'encoder': nn.Sequential(OrderedDict(encoder)), 
            'decoder': nn.Sequential(OrderedDict(decoder))}
            )
        
        # time embeddings

    def define_block(self,dims, batch_norm=True, last_batch_norm=False):
        encoder = []
        for input_dim, output_dim in zip(dims[:-1], dims[1:]):
            encoder.append((f'linear_{self.i}', nn.Linear(input_dim, output_dim)))
            if self.i < len(dims):
                if batch_norm:
                    encoder.append((f'BatchNorm_{self.i}', nn.BatchNorm1d(output_dim)))
                encoder.append((f'{self.activation}_{self.i}', self.act_fn()))
            else:
                if last_batch_norm:
                    encoder.append((f'BatchNorm_{self.i}', nn.BatchNorm1d(output_dim)))
                # no activation for the latent layer
            self.i+=1
        return encoder

    def encode(self, x_0, c, batch = None, t_0 = None):
        # input
        input_dict= self._get_model_input(x_0, c, batch, t_0)
        x_ = input_dict['full_input']
        c_emb = input_dict['condition']
        c_emb = c_emb.sum(dim=1) if len(c_emb.shape) == 3 else c_emb
        # encode
        z = self.epsilon_theta['encoder'](x_)
        z_c = z + c_emb
        z_c_act = self.act_fn()(z_c)
        return {"z":z, "z_c":z_c_act, "c":c_emb}

    def forward(self, x_0, c, batch = None, t_0 = None):
        # input
        input_dict= self._get_model_input(x_0, c, batch, t_0)
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

    def get_DeltaX(self, batch_data):
        return batch_data[-2]
    
class Epsilon_Categorical(nn.Module):
    def __init__(self, 
                gene_dim: int, 
                time_emb_dim : int,
                n_base_perturbs : int,
                condition_emb_dim : int,
                use_batch_index : bool,
                hidden_size: list = [256,128,256],
                activation : Union[str, nn.Module] = "Mish",
                pretrained_embeddings : nn.Module = None,
                ) :
        super().__init__()

        self.input_dim = gene_dim + n_base_perturbs
        self.gene_dim = gene_dim

        if hidden_size is None:
            self.full_model = nn.Linear(self.input_dim, self.gene_dim)
        else:
            dimensions = [self.input_dim] + hidden_size 
            layers = [nn.Sequential(nn.Linear(in_dim, out_dim), nn.Mish(), nn.BatchNorm1d(out_dim)) 
                        for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:])]
            layers += [nn.Linear(dimensions[-1], self.gene_dim)]

            self.full_model = nn.Sequential(*layers)

    def forward(self, x_0, c, batch = None, t_0 = None):

        X = torch.cat([x_0,c], axis=1).float()

        return self.full_model(X)
        
    
class Epsilon_AttnCondition(Epsilon_Linear):
    def __init__(self, 
                gene_dim: int, 
                time_emb_dim : int,
                n_base_perturbs : int,
                condition_emb_dim : int,
                use_batch_index : bool,
                hidden_size: list = [256,128,256],
                activation : Union[str, nn.Module] = "Mish",
                pretrained_embeddings : nn.Module = None,
                ) :
        super().__init__(gene_dim, time_emb_dim, n_base_perturbs, condition_emb_dim, use_batch_index, hidden_size,  activation, pretrained_embeddings)

        
        # self.encoder_dims[-1] = self.encoder_dims[-1] * 2
        
        encoder = self.define_block(self.encoder_dims)
        decoder = self.define_block(self.decoder_dims)
        
        self.epsilon_theta = nn.ModuleDict(
            {'encoder': nn.Sequential(OrderedDict(encoder)), 
            'decoder': nn.Sequential(OrderedDict(decoder))}
            )
        
        # time embeddings

    def condition_attention(self, z_emb, c_emb):
        """
        key feature of this version of model
        The joint embedding is a weighted sum of control embedding and condition embeddings
        """
        
        batch_size = z_emb.shape[0]
        control_tokens = torch.zeros((batch_size,)).long().to(z_emb.device)
        cntrl_emb = self.embedder(control_tokens).squeeze()

        cond_strength = torch.bmm(z_emb.unsqueeze(1), c_emb.unsqueeze(2)).squeeze(2)
        
        cntrl_strength = torch.bmm(z_emb.unsqueeze(1), cntrl_emb.unsqueeze(2)).squeeze(2)

        z_c = torch.multiply(cond_strength , c_emb) + torch.multiply(cntrl_strength , cntrl_emb)

        return z_c

    def encode(self, x_0, t_0, batch = None, c = None):
        # input
        input_dict= self._get_model_input(x_0, t_0, batch, c)
        x_ = input_dict['full_input']
        c_emb = input_dict['condition']
        c_emb = c_emb.sum(dim=1) if len(c_emb.shape) == 3 else c_emb
        
        # encode
        z = self.epsilon_theta['encoder'](x_)
        z_c = self.condition_attention(z, c_emb)
        z_c_act = self.act_fn()(z_c)
        return {"z":z, "z_c":z_c_act, "c":c_emb}

    def forward(self, x_0, t_0, batch = None, c = None):
        # input
        input_dict= self._get_model_input(x_0, t_0, batch, c)
        x_ = input_dict['full_input']
        c_emb = input_dict['condition']
        c_emb = c_emb.sum(dim=1) if len(c_emb.shape) == 3 else c_emb
        # encode
        z = self.epsilon_theta['encoder'](x_)
        assert z.shape == c_emb.shape, 'cell latent and condition latent is not in the same space'
        z_c = self.condition_attention(z, c_emb)
        z_c_act = self.act_fn()(z_c)
        # pertub and decode
        return self.epsilon_theta['decoder'](z_c_act)


class Epsilon_CAE(Epsilon_Linear):
    def __init__(self, 
                gene_dim: int, 
                time_emb_dim : int,
                n_base_perturbs : int,
                condition_emb_dim : int,
                use_batch_index : bool,
                hidden_size: list = [256,128,256],
                activation : Union[str, nn.Module] = "Mish",
                pretrained_embeddings : nn.Module = None,
                ) :
        super().__init__(gene_dim, time_emb_dim, n_base_perturbs, condition_emb_dim, use_batch_index, hidden_size, activation, pretrained_embeddings)
        self.variational = False
        
        self.i = 0
        encoder = self.define_block(self.encoder_dims)
        decoder = self.define_block(self.decoder_dims)
        
        self.epsilon_theta = nn.ModuleDict(
            {'encoder': nn.Sequential(OrderedDict(encoder)), 
            'decoder': nn.Sequential(OrderedDict(decoder)),
            })

        self.loss_fn = nn.SmoothL1Loss()
        # self.loss_fn = nn.MSELoss()
        # time embeddings

    def encode(self, x_0, c, batch, t_0):
        # input
        input_dict= self._get_model_input(x_0, c, batch, t_0)
        x_ = input_dict['full_input']
        c_emb = input_dict['condition']
        c_emb = c_emb.sum(dim=1) if len(c_emb.shape) == 3 else c_emb
        
        # encode
        z = self.epsilon_theta['encoder'](x_)
        z_c = z + c_emb
        z_c_act = self.act_fn()(z_c)
        return {"z":z, "z_c":z_c_act, "c":c_emb}

    def get_DeltaX(self, batch_data):
        """
        from batch data, return 
        X, batch_idx, condition_idx, Delta_X, degree
        """
        return batch_data[-2]

    def forward(self, x_0, c, batch = None, t_0 = None):
        
        z_dict = self.encode(x_0, c, batch, t_0)
        z_c_act = z_dict['z_c']

        return self.epsilon_theta['decoder'](z_c_act)

    def compute_loss(self, DeltaX, DeltaX_pred):
        return self.loss_fn(DeltaX, DeltaX_pred)

class GuassianNLL_CAE(Epsilon_Linear):
    def __init__(self, 
                gene_dim: int, 
                time_emb_dim : int,
                n_base_perturbs : int,
                condition_emb_dim : int,
                use_batch_index : bool,
                hidden_size: list = [256,128,256],
                activation : Union[str, nn.Module] = "Mish",
                pretrained_embeddings : nn.Module = None,
                ) :
        super().__init__(gene_dim, time_emb_dim, n_base_perturbs, condition_emb_dim, use_batch_index, hidden_size, activation, pretrained_embeddings)
        self.variational = True
        self.decoder_dims = self.decoder_dims[:-1]

        self.i = 0
        encoder = self.define_block(self.encoder_dims)
        decoder = self.define_block(self.decoder_dims)
        
        gene_mu = nn.Sequential(
            nn.BatchNorm1d(self.encoder_dims[-1]),
            self.act_fn(),
            nn.Linear(self.decoder_dims[-1], gene_dim)
        )
        gene_var = nn.Sequential(
            nn.BatchNorm1d(self.encoder_dims[-1]),
            self.act_fn(),
            nn.Linear(self.decoder_dims[-1], gene_dim),
            nn.Softplus()
        )

        self.epsilon_theta = nn.ModuleDict(
            {'encoder': nn.Sequential(OrderedDict(encoder)), 
            'decoder': nn.Sequential(OrderedDict(decoder)),
            'gene_mu': gene_mu,
            "gene_var": gene_var
            })

        self.loss_fn = nn.GaussianNLLLoss()
        # time embeddings

    def forward(self, x_0, c, batch = None, t_0 = None):
        
        z_dict = self.encode(x_0, c, batch, t_0)
        z_c_act = z_dict['z_c']
        hidden = self.epsilon_theta['decoder'](z_c_act)

        gene_mean = self.epsilon_theta['gene_mu'](hidden)
        gene_var = self.epsilon_theta['gene_var'](hidden)

        return gene_mean, gene_var

    def compute_loss(self, DeltaX, mu_var):
        mean, var = mu_var
        return self.loss_fn(mean, DeltaX, var)
    
class Epsilon_CVAE(Epsilon_CAE):
    def __init__(self, 
                gene_dim: int, 
                kl_weight : float = 1.0,
                time_emb_dim : int = 0,
                n_base_perturbs : int = 1,
                condition_emb_dim : int = 32,
                use_batch_index : bool = False,
                hidden_size: list = [256,128,256],
                activation : Union[str, nn.Module] = "Mish",
                pretrained_embeddings : nn.Module = None,
                ) :
        super().__init__(gene_dim, time_emb_dim, n_base_perturbs, condition_emb_dim, use_batch_index, hidden_size, activation, pretrained_embeddings)
        self.encoder_dims[-1] = self.encoder_dims[-1] * 2
        self.decoder_dims[-1] = self.decoder_dims[-1] 
        self.variational = True
        self.kl_weight = kl_weight
        

        self.i = 0
        encoder = self.define_block(self.encoder_dims)
        decoder = self.define_block(self.decoder_dims)
        # lag_predictor = self.define_block(self.decoder_dims)

        self.var = nn.parameter.Parameter(torch.ones((self.gene_dim,1)), requires_grad=True)
        
        self.epsilon_theta = nn.ModuleDict(
            {'encoder': nn.Sequential(OrderedDict(encoder)), 
            'decoder': nn.Sequential(OrderedDict(decoder)),
            })
        
        self.var_act = nn.Softplus()
        self.loss_fn = nn.GaussianNLLLoss()
        
        # important !
        self.variational = True

    def reparameterize(self, mu, logvar):
        q_v = self.var_act(logvar) + 1e-4
        dist = db.Normal(mu, q_v.sqrt())
        z = dist.rsample()
        return z

    def encode(self,x_0, c, batch = None, t_0 = None):
        # input
        input_dict= self._get_model_input(x_0, c, batch,  t_0)
        x_ = input_dict['full_input']
        c_emb = input_dict['condition']
        c_emb = c_emb.sum(dim=1) if len(c_emb.shape) == 3 else c_emb
        
        # encode
        n_latent = self.encoder_dims[-1] // 2
        inference_out = self.epsilon_theta['encoder'](x_)

        z_mean = inference_out[:, :n_latent]
        z_vars = self.var_act(inference_out[:, n_latent:])
        
        z = self.reparameterize(z_mean, z_vars) # latent

        z_c = z + c_emb
        z_c_act = self.act_fn()(z_c)
        return {"z":z, "z_c":z_c_act, "c":c_emb, 'q_m':z_mean, 'q_v':z_vars}
    
    def forward(self, x_0, c, batch = None, t_0 = None):
        """
        the forwar process for variational vae;
        Encode : inference
        Decode : generative model
        """
        # encode : q
        z_dict = self.encode(x_0, c, batch, t_0)
        z_c_act = z_dict['z_c']
        z_mean = z_dict['q_m']
        z_vars = z_dict['q_v']

        # decode : p
        n_gene = self.gene_dim
        out = self.epsilon_theta['decoder'](z_c_act)
        mu = out[:, :n_gene]
        var_ = self.var_act(self.var)
        # var = var_.pow(2)

        var_ = self.var_act(self.var)
        var = var_.T.expand(mu.shape)
        return mu, var, z_mean, z_vars

    def compute_loss(self, DeltaX, gen_infer_output):
        mu, var, z_mean, z_vars = gen_infer_output
        recon_loss = self.loss_fn(mu, DeltaX, var)

        basal_distribution = db.Normal(z_mean, z_vars.sqrt())
        dist_pz = db.Normal(
            torch.zeros_like(basal_distribution.loc), torch.ones_like(basal_distribution.scale)
        )
        kl_loss = db.kl.kl_divergence(basal_distribution, dist_pz).sum(-1)
        return recon_loss + self.kl_weight * kl_loss.mean()

class Epsilon_VAE(Epsilon_CVAE):
    def __init__(self, 
                gene_dim: int, 
                kl_weight : float = 1.0,
                time_emb_dim : int = 0,
                n_base_perturbs : int = 1,
                condition_emb_dim : int = 32,
                use_batch_index : bool = False,
                hidden_size: list = [256,128,256],
                activation : Union[str, nn.Module] = "Mish",
                pretrained_embeddings : nn.Module = None,
                ) :
        super().__init__(gene_dim, kl_weight, time_emb_dim, n_base_perturbs, condition_emb_dim, use_batch_index, hidden_size, activation, pretrained_embeddings)
        
        self.encoder_dims = self.encoder_dims[:-1]
        self.decoder_dims[-1] = self.decoder_dims[-1] 
        self.variational = True
        self.kl_weight = kl_weight
        

        self.i = 0
        encoder = self.define_block(self.encoder_dims)
        decoder = self.define_block(self.decoder_dims)
        # lag_predictor = self.define_block(self.decoder_dims)
        fc_mu = nn.Sequential(
            nn.BatchNorm1d(self.encoder_dims[-1]),
            self.act_fn(),
            nn.Linear(self.encoder_dims[-1], condition_emb_dim)
        )
        fc_var = nn.Sequential(
            nn.BatchNorm1d(self.encoder_dims[-1]),
            self.act_fn(),
            nn.Linear(self.encoder_dims[-1], condition_emb_dim)
        )

        self.var = nn.parameter.Parameter(torch.ones((self.gene_dim,1)), requires_grad=True)
        
        self.epsilon_theta = nn.ModuleDict(
            {'encoder': nn.Sequential(OrderedDict(encoder)), 
            'fc_mu' : fc_mu,
            'fc_var' : fc_var,
            'decoder': nn.Sequential(OrderedDict(decoder)),
            })

    def encode(self,x_0, c, batch = None, t_0 = None):
        # input
        input_dict= self._get_model_input(x_0, c, batch,  t_0)
        x_ = input_dict['full_input']
        c_emb = input_dict['condition']
        c_emb = c_emb.sum(dim=1) if len(c_emb.shape) == 3 else c_emb
        
        # encode
        n_latent = self.encoder_dims[-1] // 2
        hidden = self.epsilon_theta['encoder'](x_)

        z_mean = self.epsilon_theta['fc_mu'](hidden)
        z_vars = self.var_act(self.epsilon_theta['fc_var'](hidden))
        
        z = self.reparameterize(z_mean, z_vars) # latent

        return {"z":z, "z_c":z, "c":c_emb, 'q_m':z_mean, 'q_v':z_vars}
    
class Epsilon_CVAE_adv(Epsilon_CVAE):
    def __init__(self, 
                gene_dim: int, 
                time_emb_dim : int,
                n_base_perturbs : int,
                condition_emb_dim : int,
                use_batch_index : bool,
                hidden_size: list = [256,128,256],
                activation : Union[str, nn.Module] = "Mish",
                pretrained_embeddings : nn.Module = None,
                ) :
        super().__init__(gene_dim, time_emb_dim, n_base_perturbs, condition_emb_dim, use_batch_index, hidden_size, activation, pretrained_embeddings)

# TODO:
class Epsilon_Linear_token_net(Epsilon_Linear):
    def __init__(self, 
                gene_dim: int, 
                time_emb_dim : int,
                n_base_perturbs : int,
                condition_emb_dim : int,
                use_batch_index : bool,
                hidden_size: list = [256,128,256],
                activation : Union[str, nn.Module] = "Mish",
                pretrained_embeddings : nn.Module = None,
                ) :
        super().__init__(gene_dim=gene_dim, time_emb_dim=time_emb_dim, n_base_perturbs=n_base_perturbs,
                        condition_emb_dim=condition_emb_dim, use_batch_index=use_batch_index,hidden_size=hidden_size,
                        activation=activation,pretrained_embeddings=pretrained_embeddings
                        )






class ODE_eps(Epsilon_Linear):
    def __init__(self, 
                gene_dim: int, 
                time_emb_dim : int,
                n_base_perturbs : int,
                condition_emb_dim : int,
                use_batch_index : bool,
                hidden_size: list = [256,128,256],
                activation : Union[str, nn.Module] = "Mish",
                pretrained_embeddings : nn.Module = None,
                ) :
        super().__init__(gene_dim, time_emb_dim, n_base_perturbs, condition_emb_dim, use_batch_index,hidden_size, activation, pretrained_embeddings)

    def forward(self, X):
        # X contain x, t, c
        c = self.c.long()
        t0 = self.t0
        input_dict= self._get_model_input(X, t0, None, c)
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

class ODE_eps_notime(ODE_eps):
    def __init__(self, 
                gene_dim: int, 
                time_emb_dim : int,
                n_base_perturbs : int,
                condition_emb_dim : int,
                use_batch_index : bool,
                hidden_size: list = [256,128,256],
                activation : Union[str, nn.Module] = "Mish",
                pretrained_embeddings : nn.Module = None,
                ) :
        super().__init__(gene_dim, 0, n_base_perturbs, condition_emb_dim, use_batch_index,hidden_size, activation, pretrained_embeddings)

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

        assert x.shape[1] == self.gene_dim, "layer X should have the same dimension as `gene_dim`"
        
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
        
        input_dict = {"full_input":x,  "batch":None, "condition":c_emb}
        return input_dict

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
    gene_dim
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
                gene_dim: int, 
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
        super().__init__(gene_dim, time_emb_dim, n_base_perturbs, condition_emb_dim, use_batch_index, activation, pretrained_embeddings)
        
        
        # define epsilon_theta network
        # start with input layer

        # ---- split layers ---------
        dimensions = [self.input_dim] + hidden_size + [self.gene_dim]

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
                    (f'BatchNorm_0', nn.BatchNorm1d(hidden_size[0])),
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

        decoder.append((f'Output_fc_{i}', nn.Linear(hidden_size[-1], self.gene_dim)))
        
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
