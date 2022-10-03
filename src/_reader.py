import os, sys
import torch
from torch import nn
import numpy as np
import scanpy as sc


def demultiplex_token(adata, perturb_col='condition', max_multiplexing=2, 
                        delimiter="+", control_pert='ctrl'):

    delimit_fn = lambda x, i: x.split(delimiter)[i]

    for i in range(max_multiplexing):

        try:
            individual_pert = adata.obs[perturb_col].apply(delimit_fn, args=(i,))
        except IndexError:
            individual_pert = control_pert
        
        adata.obs['pert%d'%i] = individual_pert

def get_embedding_index(batch_pert1_n_2:list, unique_token):
    idx_mat = []
    for p1_p2 in batch_pert1_n_2:
        idxs = [unique_token.index(p) for p in p1_p2]
        idx_mat.append(idxs)
    idx_mat = np.array(idx_mat)
    return torch.from_numpy(idx_mat).long()
