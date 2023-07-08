import os, sys, math
from turtle import forward
from typing import Union, Optional
from einops import rearrange
import numpy as np
import torch
from torch import nn, einsum
from torch.nn.modules import activation
from collections import OrderedDict
import warnings

class SinoidalPositionEmbeddings(nn.Module):
    """
    position embedding for timepoints indexing
    """
    def __init__(self, time_dim:int):
        super().__init__()
        self.dim = time_dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        constant_ = math.log(10000) / (half_dim -1)
        embeddings = torch.exp(torch.arange(half_dim, device=device)* -constant_ )
        embeddings = t[:,None] * embeddings[None, :] # 
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class MLP(nn.Module):
    def __init__(self, dimensions, use_batchnorm=True, use_dropout=0, skip_connection=False, activation_fn='ReLU', output_activation=None):
        super().__init__()
        self.act_fn = eval(f"nn.{activation_fn}")
        self.dimensions = dimensions
        self.n_layer = len(dimensions) - 1

        p = use_dropout

        if skip_connection == 'auto':
            self.skip_connection = True if self.n_layer > 2 else False
        else:
            self.skip_connection = skip_connection

        # define model
        self.model = nn.ModuleList()
        for dim_in, dim_out in zip(dimensions[:-1],dimensions[1:]):

            ## define block
            block = []
            block.append( nn.Linear(dim_in, dim_out) )

            if dim_out == dimensions[-1]:
                if output_activation is not None:
                    self.out_act_fn = eval(f"nn.{output_activation}")
                    block.append( self.out_act_fn() )
            else:
                block.append( self.act_fn() )

            if use_batchnorm & (dim_out != dimensions[-1]):
                block.append( nn.BatchNorm1d(dim_out) )

            if (p != 0) & (dim_out != dimensions[-1]):
                block.append( nn.Dropout(p) )
            
            ## insert block
            self.model.append( nn.Sequential(*block) )

    def forward(self, X):
        
        for block in self.model:
            out = block(X)
            
            # skip connect
            if self.skip_connection==True:
                linear = block[0]
                if linear.in_features == linear.out_features:
                    out = X + out
            
            X = out

        return out

class SelfAttention(nn.Module):
    r"""
    Multi-head self attention module 
    https://arxiv.org/abs/1706.03762

    Parameters
    ------------------------------
    input_dim
        the input dimension, the output is the same dimension
    n_heads
        the number of output heads
    qk_dim
        the intermediate dimension for query, key and value
    """
    def __init__(self, input_dim, n_heads, qk_dim):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.n_heads = n_heads
        pooled_dimension = n_heads * qk_dim

        self.atten_fc = nn.ModuleDict(
            { key : nn.Linear(input_dim, pooled_dimension) for key in ['q', 'k', 'v'] }
                    )
        self.fc_out = nn.Sequential(
            nn.Linear(pooled_dimension, input_dim),
            nn.GroupNorm(1, input_dim)
        )
    
    def forward(self, x):
        r"""
        Parameters
        ---------------
        x 
            torch.Tensor, input tensor of shape (batch size, feature size)
        
        Return
        ---------------
        out 
            torch.Tensor, output tensor of shape (batch size, heads, feature size)
        """
        b, f = x.shape

        qkv = [self.atten_fc[key](x) for key in ['q', 'k', 'v']]

        q, k, v = map(
            lambda t: rearrange(t, "b (h f) -> b h f", h=self.n_heads), qkv
        ) 

        q = q * self.scale
        
        sim = einsum("b h i , b h j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach() #  ???
        attn = sim.softmax(dim=-1)

        v_out = einsum("b h i j , b h j -> b h i", attn , v)
        v_out = rearrange(v_out, "b h j -> b (h j)")
        return self.fc_out(v_out)


class LinearAttention(nn.Module):
    r"""
    self-attention with linear complexity
    https://arxiv.org/abs/1812.01243 

    Parameters
    ------------------------------
    input_dim
        the input dimension, the output is the same dimension
    n_heads
        the number of output heads
    qk_dim
        the intermediate dimension for query, key and value
    """
    def __init__(self, input_dim, output_dim, n_heads, qk_dim):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.n_heads = n_heads
        pooled_dimension = n_heads * qk_dim

        self.atten_fc = nn.ModuleDict(
            { key : nn.Linear(input_dim, pooled_dimension) for key in ['q', 'k', 'v'] }
                    )

        self.fc_out = nn.Sequential(
            nn.Linear(pooled_dimension, output_dim),
            nn.GroupNorm(1, output_dim)
        )

    def forward(self, x):
        b, f = x.shape
        qkv = [self.atten_fc[key](x) for key in ['q', 'k', 'v']]

        q, k, v = map(
            lambda t: rearrange(t, "b (h f) -> b h f", h=self.n_heads), qkv
        ) 

        # softmax first
        q = q.softmax(dim=-1)
        k = q.softmax(dim=-1)
        q = q * self.scale

        # get value
        context = torch.einsum("b h i, b h j -> b i j", k, v) #  i is qk_dimension  , j is v_dimension
        v_out = torch.einsum("b h i , b i j -> b h j",  q, context) #  [head * qk_dim]  @ [ qk_dim * v_dim]
        v_out = rearrange(v_out, "b h j -> b (h j)")

        return self.fc_out(v_out)

class Residual(nn.Module):
    """f(x)+x"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    def __init__(self, input_dim, output_dim, n_heads, qk_dim, time_emb_dim=None, attention="SelfAttention", **attn_kw_args):
        super().__init__()
        # why silu first ?
        # align time_emb_dim to dimout

        assert attention in ["SelfAttention", "LinearAttention"]
        Operator = eval(attention) 
         
        self.block1 = Operator(input_dim, output_dim, n_heads, qk_dim)
        self.block2 = Operator(input_dim, output_dim, n_heads, qk_dim)
        self.out_transform = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
        self.norm = nn.GroupNorm(n_heads, output_dim)

        # self.mlp = (
        #     nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, output_dim))
        #     if time_emb_dim is not None
        #     else None
        # )

    def forward(self, x, time_emb=None):
        h = self.block1(x) # output_dim
        
        # if (self.mlp is not None) and (time_emb is not None ):
        #     time_emb = self.mlp(time_emb) # output_dim
        #     h = time_emb + h
            
        h = self.block2(h)
        return h + self.out_transform(x)