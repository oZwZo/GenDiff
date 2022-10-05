import os, sys
import torch
from typing import Union, Optional
from torch.utils.data import Dataset, DataLoader
from anndata import AnnData
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

def get_embedding_index(batch_multipx_pert:list, unique_token):
    idx_mat = []
    for p1_p2 in batch_multipx_pert:
        idxs = [unique_token.index(p) for p in p1_p2]
        idx_mat.append(idxs)
    idx_mat = np.array(idx_mat)
    return torch.from_numpy(idx_mat).long()


#              ___           _            ___         _   
#             |   \   __ _  | |_   __ _  / __|  ___  | |_ 
#             | |) | / _` | |  _| / _` | \__ \ / -_) |  _|
#             |___/  \__,_|  \__| \__,_| |___/ \___|  \__|

class AnnDataSet(Dataset):
    """
    Torch dataset that take in Anna data

    Parameters:
    ---------------
    AnnData
        single cell AnnData loaded throgh scanpy API
    layers
        str, the layer of data to use; defualt `counts`
    split_key
        str, the AnnData.obs key that store how the data was splited
    which_set
        str, the value in the AnnData.obs[split_key], selecting cells in a specific set (i.e. train test validataion)
    
    Retrurn
    ---------------
    dataset
        torch.utils.data object
    """
    def __init__(self,AnnData:AnnData, 
                    layers:str ='counts', 
                    split_key:str = 'split', 
                    which_set: str ='train'):
        super().__init__()
        
        self.adata = AnnData[AnnData.obs[split_key]==which_set].copy()
        self.X = self.adata.layers[layers]
        
        # check matrix type
        self.sparse_input = (str(type(self.X)) == "<class 'scipy.sparse.csr.csr_matrix'>")
        if self.sparse_input:
            self.X = np.asarray(self.X.todense())
    
    def __len__(self):
        return self.adata.shape[0]
    
    def __getitem__(self, index):
        return self.X[index]

#    ___                   _   _   _     _                  ___    ___ 
#   / __|  ___   _ _    __| | (_) | |_  (_)  ___   _ _     |   \  / __|
#  | (__  / _ \ | ' \  / _` | | | |  _| | | / _ \ | ' \    | |) | \__ \
#   \___| \___/ |_||_| \__,_| |_|  \__| |_| \___/ |_||_|   |___/  |___/

class Condition_AnnDataSet(AnnDataSet):
    r"""
    Torch dataset where each cell is annotated with the condition

    Parameters
    ---------------
    - base -
    AnnData
        single cell AnnData loaded throgh scanpy API
    layers
        str, the layer of data to use; defualt `counts`
    split_key
        str, the AnnData.obs key that store how the data was splited
    which_set
        str, the value in the AnnData.obs[split_key], selecting cells in a specific set (i.e. train test validataion)

    - condition -
    unique_token_dict
        dict, required ! The dict mapping the condition tokens to its index. 
        !!! The control condition should always be 0 !!!. The default control symbol is `ctrl`
    use_batch_index
        bool, whether we return the experimental batch. 
        If set to True, then each minibatch will contain (x,condtion, batch) 
        otherwise return (x, condition)
    exp_batch_key
        str, the AnnData.obs key that record the experimental batch which cell comes from. 
        The experimental batch should be indexed by `int`.
    condition_key
        str, the AnnData.obs key that contains the experimental condition exerted to the cells.
        The condition variables is stored in str. 
        i.e.
        For control cells, we noted with `ctrl+ctrl` for `max_multiplexing=2`;
        For single perturbation, we noted with `C +ctrl` 
        For double perturbation, we noted with `C1+C2`

    Retrurn
    ---------------
    dataset
        torch.utils.data object
    """

    def __init__(self, 
                AnnData : AnnData, 
                unique_token_dict : dict,
                condition_key : str = 'condition',
                max_multiplexing : int = 2,
                use_batch_index : bool = False,
                exp_batch_key : str = 'batch', 
                delimiter : str ="+",
                layers:str ='counts', 
                split_key:str = 'split', 
                which_set: str ='train'):
        super().__init__(AnnData, layers, split_key, which_set)

        # store the attribute
        # process the token bag and check their uni-mapping
        self.unique_token_dict = unique_token_dict
        self.reverse_token = {v:k for k,v in unique_token_dict.items()}
        self.n_base_perturbs = len(self.reverse_token)
        self.null_cond_key = self.reverse_token[0]
        
        # sanity check
        self.delimiter = delimiter
        # assert delimiter in self.reverse_token[1], f"invalid delimiter {delimiter}, but the format of token is like {self.reverse_token[1]}"

        # condition pairing
        self.multipx_conditions = self.adata.obs[condition_key].values
            # if the dlimiter is correct
        n_multiplexing = [len(tokens.split(delimiter)) for tokens in self.multipx_conditions]
        assert max_multiplexing == max(n_multiplexing), "the maximal "
        self.max_multiplexing = max_multiplexing

        # exp batch
        self.use_batch_index = use_batch_index
        if use_batch_index:
            self.exp_batch = self.adata.obs[exp_batch_key].values

    
        delimit_fn = lambda x, i: x.split(self.delimiter)[i]
    
    def __getitem__(self, i):
        exp_mat = self.X[i]

        split_tokens = self.multipx_conditions[i].split(self.delimiter)
        n_tokens = len(split_tokens)

        # we pad the token list to maximal muultiplexing 
        # pad with the null key 
        if len(split_tokens) < self.max_multiplexing:
            split_tokens += [self.null_cond_key]*(self.max_multiplexing - n_tokens)

        condition_idx = np.array([self.unique_token_dict[token] for token in split_tokens])

        if self.use_batch_index:
            batch_idx = self.exp_batch[i]
            return exp_mat, batch_idx, condition_idx
        else:
            return exp_mat, condition_idx

        
        
        



    

    