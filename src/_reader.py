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

        if layers in self.adata.layers:
            self.X = self.adata.layers[layers]
        elif layers in self.adata.obsm_keys():
            self.X = self.adata.obsm[layers]
        elif layers == 'X':
             self.X = self.adata.X
        else:
            raise KeyError(f"no data was matched with key {layers}")
        
        # check matrix type
        self.sparse_input = (str(type(self.X)) == "<class 'scipy.sparse.csr.csr_matrix'>")
        if self.sparse_input:
            self.X = np.asarray(self.X.todense())
        
        self.X = torch.from_numpy(self.X).float()
    
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
                unique_token_dict : Union[dict, str],
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
        if type(unique_token_dict) == str:
            self.unique_token_dict = self.adata.uns[unique_token_dict]
        elif type(unique_token_dict) == dict:
            self.unique_token_dict = unique_token_dict
        self.reverse_token = {v:k for k,v in self.unique_token_dict.items()}
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
        
        else:
            batch_idx=[]

        return exp_mat, batch_idx, condition_idx, [], [] # noise and t is empty

        
class Diffuse_Dataset(Condition_AnnDataSet):
    def __init__(self, 
                AnnData : AnnData, 
                unique_token_dict : dict,
                condition_key : str = 'condition',
                max_multiplexing : int = 1,
                use_batch_index : bool = False,
                exp_batch_key : str = 'batch', 
                pseudotime_key : str = "dpt_pseudotime",
                neighbor_key : str = "",
                n_neighbor : int = 90,
                delimiter : str ="",
                layers:str ='counts', 
                split_key:str = 'split', 
                which_set: str ='train',
                max_degree: int = 5,
                search_strategy: str = 'traverse'
                ):
        super().__init__(AnnData, unique_token_dict, condition_key,max_multiplexing, 
                            use_batch_index, exp_batch_key, delimiter, layers, split_key, which_set)
        
        self.k = n_neighbor
        self.pseudotime_key = pseudotime_key
        self.connectivities = self.adata.obsp[neighbor_key+'connectivities']
        self.distance = self.adata.obsp[neighbor_key+'distances']
        self.max_degree = max_degree
        self.search_strategy = search_strategy

    @property
    def T(self):
        """
        generate discreted timepoint from pseudo-time
        """
        if "discrete_time" not in self.adata.obs_keys():
            def discrete_time(x):
                x = int(x*250)
                mid_err = np.random.randint(-2,2) # smooth the time
                tail_err = np.random.randint(0,10)
                return min(max(0,x+mid_err), 200-tail_err)

            self.adata.obs['discrete_time'] = self.adata.obs[self.pseudotime_key].apply(discrete_time)

        return self.adata.obs['discrete_time'].values

    def traverse_neighbor(self, knn_idx):
        k = -1*self.k 
        # random walk
        next_degree_knn = []
        for i_d in knn_idx:
            neighbor = np.argpartition(self.connectivities[i_d], k)[k:].tolist()
            next_degree_knn.extend(neighbor)
        
        # all neighbor visited to the current degree
        knn_idx_d_plus1 = knn_idx.tolist() + next_degree_knn
        knn_idx_d_plus1 = np.unique(knn_idx_d_plus1).astype(int)
        
        return knn_idx_d_plus1
    
    def expand_neighbor(self, i, n_degree):
        k = -1*self.k * (1+n_degree)
        knn_idx = np.argpartition(self.connectivities[i], k)[k:].astype(int)
        return knn_idx

    def _diffuse_neighbor(self, i, c_i, t_i):
        """
        the Key function defines the noise sampling process 
        given the starting point i
        """
        pass_1 = 0
        pass_2 = 0
        n_degree = 0
        knn_idx = np.array([i])
        while pass_1*pass_2==0 and n_degree < self.max_degree:
            
            knn_idx = self.traverse_neighbor(knn_idx) if self.search_strategy == 'traverse' else self.expand_neighbor(i, n_degree)

            # 1 : neighbor with the same condition
            knn_c = self.multipx_conditions[knn_idx]
            if c_i in knn_c:
                pass_1_idx = knn_idx[knn_c == c_i]
                pass_1 = 1
            else:
                pass_1_idx = knn_idx

            # 2 : neighbor with bigger pseudo-time
            knn_t = self.T[pass_1_idx]
            if np.any(knn_t > t_i):
                pass_2_idx = pass_1_idx[knn_t > t_i]
                pass_2 = 1
            else:
                pass_2_idx = pass_1_idx

            n_degree += 1
                
            
        # sampled by distance
        knn_p = self.connectivities[i,pass_2_idx]
        p = knn_p / knn_p.sum() if knn_p.sum() != 0 else None # normalized

        neighbor_idx = np.random.choice(pass_2_idx, p=p)
        return neighbor_idx, pass_1, pass_2


    def __getitem__(self, i):
        """
        return x , b, c, noise, t in a mini-batch
        """
        exp_mat = self.X[i]
        c_string = self.multipx_conditions[i]
        split_tokens = c_string.split(self.delimiter)
        n_tokens = len(split_tokens)

        # we pad the token list to maximal muultiplexing 
        # pad with the null key 
        if len(split_tokens) < self.max_multiplexing:
            split_tokens += [self.null_cond_key]*(self.max_multiplexing - n_tokens)

        condition_idx = np.array([self.unique_token_dict[token] for token in split_tokens])

        if self.use_batch_index:
            batch_idx = self.exp_batch[i]
        
        else:
            batch_idx=[]

        t = self.T[i]
        neighbor_idx, _, _  = self._diffuse_neighbor(i, c_string, t)
        noise = self.X[neighbor_idx] - exp_mat
        return exp_mat, batch_idx, condition_idx, noise, t
    



    

class ODE_dataset(Diffuse_Dataset):
    def __init__(self, 
                AnnData : AnnData, 
                unique_token_dict : dict,
                condition_key : str = 'condition',
                max_multiplexing : int = 1,
                use_batch_index : bool = False,
                exp_batch_key : str = 'batch', 
                pseudotime_key : str = "dpt_pseudotime",
                neighbor_key : str = "",
                n_neighbor : int = 90,
                delimiter : str ="",
                layers:str ='counts', 
                split_key:str = 'split', 
                which_set: str ='train',
                max_degree: int = 5,
                search_strategy: str = 'traverse'
                ):
        super().__init__(AnnData=AnnData,unique_token_dict=unique_token_dict,condition_key=condition_key,
                         max_multiplexing=max_multiplexing, use_batch_index=use_batch_index, exp_batch_key=exp_batch_key,
                         pseudotime_key=pseudotime_key, neighbor_key=neighbor_key, n_neighbor=n_neighbor,
                         delimiter=delimiter, layers=layers,split_key=split_key,which_set=which_set, max_degree=max_degree,search_strategy=search_strategy)

    @property
    def T(self):
        """
        discreted timepoint into several slot
        """
        if "discrete_time" not in self.adata.obs_keys():
            def discrete_time(x):
                x = int(x*250)
                mid_err = np.random.randint(-2,2) # smooth the time
                tail_err = np.random.randint(0,10)
                return min(max(0,x+mid_err), 200-tail_err)

            self.adata.obs['discrete_time'] = self.adata.obs[self.pseudotime_key].apply(discrete_time)

        if "time_slot" not in self.adata.obs_keys():
            def to_time_slot(t):
                boundary = [0,50,100,150,200]
                time_slot=200
                for lb, ub in zip(boundary[:-1], boundary[1:]):
                    if t < np.mean([ub,lb]):
                        time_slot = lb
                        break
                return time_slot
            self.adata.obs['time_slot'] = self.adata.obs['discrete_time'].apply(to_time_slot)

        return self.adata.obs['time_slot'].values
    
    def __getitem__(self, i):
        """ 
        return x , b, c, noise, t in a mini-batch
        """
        
        c_string = self.multipx_conditions[i]
        split_tokens = c_string.split(self.delimiter)
        n_tokens = len(split_tokens)

        # we pad the token list to maximal muultiplexing 
        # pad with the null key 
        if len(split_tokens) < self.max_multiplexing:
            split_tokens += [self.null_cond_key]*(self.max_multiplexing - n_tokens)

        condition_idx = np.array([self.unique_token_dict[token] for token in split_tokens])

        if self.use_batch_index:
            batch_idx = self.exp_batch[i]
        
        else:
            batch_idx=[]

        X_t0 = self.X[i]
        t0 = self.T[i]
        neighbor_idx, pass_1, pass_2 = self._diffuse_neighbor(i, c_string, t0)
        X_t1 = self.X[neighbor_idx]
        t1 = self.T[neighbor_idx]
        
        return X_t0, t0, batch_idx, condition_idx, X_t1,  t1, pass_1, pass_2