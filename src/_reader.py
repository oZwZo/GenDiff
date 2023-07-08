import os, sys
import torch
from typing import Union, Optional
from torch.utils.data import Dataset, DataLoader
from anndata import AnnData
from torch import nn
import numpy as np
import scanpy as sc
from scipy.sparse.csgraph import dijkstra

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
        if which_set == 'All':
            self.adata = AnnData.copy()
        else:
            self.adata_raw = AnnData.copy()
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
        
        self.X = self._to_dense(self.X)
        
        self.X = torch.from_numpy(self.X).float()
    
    def __len__(self):
        return self.adata.shape[0]
    
    def detect_sparse_matrix(self, x):
        return 'csr_matrix' in str(type(x)) 
    
    def _to_dense(self, x):
        if self.detect_sparse_matrix(x):
            return np.asarray(x.A)
        else:
            return x
    
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
                delimiter : str = "+",
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

class Supervised_AnnDataSet(Condition_AnnDataSet):
    def __init__(self, 
            AnnData : AnnData, 
            label_key: str,
            unique_token_dict : Union[dict, str],
            input_rep = None,
            pseudotime_key : str = 'dpt_pseudotime',
            condition_key : str = 'condition',
            max_multiplexing : int = 2,
            use_batch_index : bool = False,
            exp_batch_key : str = 'batch', 
            delimiter : str = "+",
            layers:str ='counts', 
            split_key:str = 'split', 
            which_set: str ='train'):
        super().__init__(AnnData, unique_token_dict, condition_key,max_multiplexing, 
                            use_batch_index, exp_batch_key, delimiter, layers, split_key, which_set)
        
        self.pseudotime_key = pseudotime_key
        self.label_key = label_key
        self.input_rep = input_rep

        if label_key in self.adata.obs_keys():
            self.Y = self.adata.obs[label_key].values
        elif label_key in self.adata.obsm_keys():
            self.Y = self.adata.obsm[label_key].values
        elif label_key in self.adata.layers.keys():
            self.Y = self.adata.layers[label_key].copy()
        else:
            raise KeyError("invalid label key")
        
        if input_rep is None:
            pass
        elif input_rep in self.adata.obsm_keys():
            self.X = self.adata.obsm[input_rep].values
        elif input_rep in self.adata.layers.keys():
            self.X = self.adata.layers[input_rep].copy()
        else:
            raise KeyError("invalid input representation")
        
        self.T = self.adata.obs['discrete_time'].values
        
    # @property
    # def T(self):
    #     """
    #     generate discreted timepoint from pseudo-time
    #     """
    #     if "discrete_time" not in self.adata.obs_keys():
    #         def discrete_time(x):
    #             x = int(x*250)
    #             mid_err = np.random.randint(-2,2) # smooth the time
    #             tail_err = np.random.randint(0,10)
    #             return min(max(0,x+mid_err), 200-tail_err)

    #         self.adata.obs['discrete_time'] = self.adata.obs[self.pseudotime_key].apply(discrete_time)

    #     return self.adata.obs['discrete_time'].values
    
    def __getitem__(self, i):
        exp_mat = self.X[i]
        t = self.T[i]
        deltaX = self.Y[i]

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

        return exp_mat, batch_idx, condition_idx, deltaX, t


class Dummy_condition_AnnDataSet(Condition_AnnDataSet):
    def __init__(self, 
            AnnData : AnnData, 
            label_key: str,
            unique_token_dict : Union[dict, str],
            input_rep = None,
            pseudotime_key : str = 'dpt_pseudotime',
            condition_key : str = 'condition',
            max_multiplexing : int = 2,
            use_batch_index : bool = False,
            exp_batch_key : str = 'batch', 
            delimiter : str = "+",
            layers:str ='counts', 
            split_key:str = 'split', 
            which_set: str ='train'):
        super().__init__(AnnData, unique_token_dict, condition_key,max_multiplexing, 
                            use_batch_index, exp_batch_key, delimiter, layers, split_key, which_set)
        
        self.n_token = np.max(list(self.unique_token_dict.values())) + 1
    
    def __getitem__(self, i):
        exp_mat, batch_idx, condition_idx, deltaX, t = super().__getitem__(i)

        c_onehot = np.zeros((self.n_token))
        for c in condition_idx:
            c_onehot[c] = 1
        return exp_mat, batch_idx, c_onehot, deltaX, t


class Traverse_Dataset(Condition_AnnDataSet):
    def __init__(self, 
                AnnData : AnnData, 
                unique_token_dict : dict,
                condition_key : str = 'condition',
                max_multiplexing : int = 1,
                use_batch_index : bool = False,
                exp_batch_key : str = 'batch', 
                pseudotime_key : str = "dpt_pseudotime",
                neighbor_key : str = "",
                n_neighbor : int = 15,
                delimiter : str ="",
                layers:str ='counts', 
                split_key:str = 'split', 
                which_set: str ='train',
                max_degree: int = 5,
                search_strategy: str = 'traverse',
                check_samples = False
                ):
        super().__init__(AnnData, unique_token_dict, condition_key,max_multiplexing, 
                            use_batch_index, exp_batch_key, delimiter, layers, split_key, which_set)
        
        self.k = n_neighbor
        self.pseudotime_key = pseudotime_key
        self.max_degree = max_degree
        self.search_strategy = search_strategy
        self.check_samples = check_samples
        
        ###  very important !!!!
        ##   currently set to 3
        #    
        self.free_search_degree = 3

        self.raw_X = torch.from_numpy(self._to_dense(self.adata_raw.X)).float()
        self.raw_i = self.adata_raw.obs_names
        self.raw_C = self.adata_raw.obs[condition_key].values
        self.raw_T = self.adata_raw.obs['discrete_time'].values

        self.connectivities = self.adata_raw.obsp[neighbor_key+'connectivities']
        self.connectivities = self._to_dense(self.connectivities)

        self.distance = self.adata_raw.obsp[neighbor_key+'distances']
        self.distance = self._to_dense(self.distance)


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

    def index_subset_2_raw(self,i):
        cb_i = self.adata.obs_names[i]
        return np.where(self.raw_i == cb_i)[0]
    
    def subset_2_raw_index_ls(self,indexs):
        return [self.index_subset_2_raw(i) for i in indexs]
    
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
        pass_2_idx = np.array([i])

        # dist_array = dijkstra(self.adata_raw.obsp['KNN_distances'], indices = self.index_subset_2_raw(i),, unweighted = True)[0]
        # dijkstra_knn = np.where(dist_array <= self.max_degree)[0]
        # dijkstra_c = np.array(self.raw_C[dijkstra_knn] == c_i).astype(int)
        # dijkstra_t = np.array(self.raw_T[dijkstra_knn] > t_i).astype(int)
        # print(dijkstra_c.sum(), dijkstra_t.sum(), np.multiply(dijkstra_c,dijkstra_t).sum())

        while pass_1*pass_2==0 and n_degree < self.max_degree:

            knn_idx = self.traverse_neighbor(knn_idx) if self.search_strategy == 'traverse' else self.expand_neighbor(i, n_degree)

            # 1 : neighbor with the same condition
            knn_c = self.raw_C[knn_idx]
            if c_i in knn_c:
                pass_1_idx = knn_idx[knn_c == c_i]
                pass_1 = 1
            else:
                pass_1_idx = knn_idx

            # 2 : neighbor with bigger pseudo-time
            knn_t = self.raw_T[pass_1_idx]
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
        # the sampled neighbor_idx belongs to raw idx
        neighbor_idx, pass1, pass2  = self._diffuse_neighbor(i, c_string, t)
        noise = self.raw_X[neighbor_idx] - exp_mat

        # expression matrix X , expreimental batch , perturbation [c1, c2, c3..], noise -> delta X, t: discreted t
        if self.check_samples:
            return exp_mat, batch_idx, condition_idx, noise, t, pass1, pass2
        else:
            return exp_mat, batch_idx, condition_idx, noise, t

        
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
                n_neighbor : int = 15,
                delimiter : str ="",
                layers:str ='counts', 
                split_key:str = 'split', 
                which_set: str ='train',
                max_degree: int = 5,
                search_strategy: str = 'traverse',
                check_samples = False
                ):
        super().__init__(AnnData, unique_token_dict, condition_key,max_multiplexing, 
                            use_batch_index, exp_batch_key, delimiter, layers, split_key, which_set)
        
        self.k = n_neighbor
        self.pseudotime_key = pseudotime_key
        self.max_degree = max_degree
        self.search_strategy = search_strategy
        self.check_samples = check_samples
        
        ###  very important !!!!
        ##   currently set to 3
        #    
        self.free_search_degree = 3

        self.iroot = self.adata.uns['iroot']
        self.root_x = torch.from_numpy(self.adata_raw.X[self.iroot])

        self.connectivities = self.adata.obsp[neighbor_key+'connectivities']
        self.connectivities = self._to_dense(self.connectivities)

        self.distance = self.adata.obsp[neighbor_key+'distances']
        # self.distance = self._to_dense(self.distance)


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

        knn_idx_d_plus1 = knn_idx + next_degree_knn
        # knn_idx_d_plus1 = np.unique(knn_idx_d_plus1).astype(int)
        knn_idx_d_plus1 = np.unique(next_degree_knn).astype(int).tolist()
        return knn_idx_d_plus1
    
    def _sample_by_distance(self, dist_array, candidate_idx, alpha=None, repeat=1):
        """
        based on the knn distance, sample the closest cell 
            P ~ (1 - distance)
        dist_array : ndarray, distance from all other cells to cell i
        candidate_idx : the index of knn / or cansidered neighbor cells
        alpha: parameter to adjust uncertainty, the lower the more uncertrain
        repeat: the number of cells to sample each time
        """
        alpha = 1 if alpha is None else alpha

        dist = dist_array[candidate_idx]
        where_inf = np.isinf(dist)
        if np.any(where_inf):
            try:
                next_max = dist[~where_inf].max()
                dist = np.where(where_inf, next_max, dist)
            except ValueError:
                dist = np.zeros_like(dist)

        knn_p = dist.max() - dist + 0.1*dist.min()
        if knn_p.sum() > 0:
            knn_p = knn_p**alpha
            p = knn_p / knn_p.sum() # normalized
        else:
            p = None 

        neighbor_idx = [np.random.choice(candidate_idx, p=p) for i in range(repeat)]
        if repeat == 1:
            neighbor_idx = neighbor_idx[0]
        return neighbor_idx, p
    
    def expand_neighbor(self, i, n_degree):
        k = -1*self.k * (1+n_degree)
        knn_idx = np.argpartition(self.connectivities[i], k)[k:].astype(int)
        return knn_idx
    
    def _diffuse_keepi(self, i, c_i, t_i):
        """
        the Key function defines the noise sampling process 
        given the starting point i
        """
        pass_1 = 0
        pass_2 = 0
        n_degree = 0
        knn_idx = np.array([i])


        while pass_1*pass_2==0 and n_degree < self.max_degree:
            
            # if n_degree > self.free_search_degree:   # only under this degree can we expand knn without any constraints
            #     knn_idx = pass_2_idx
        
            knn_idx = self.traverse_neighbor(knn_idx) if self.search_strategy == 'traverse' else self.expand_neighbor(i, n_degree)

            # 1 : neighbor with the same condition
            knn_c = self.multipx_conditions[knn_idx]
            if c_i in knn_c:
                pass_1_idx = np.array(knn_idx)[knn_c == c_i]
                pass_1 = 1
            else:
                pass_1_idx = knn_idx

            # 2 : neighbor with bigger pseudo-time
            knn_t = self.T[pass_1_idx]
            if np.any(knn_t > t_i):
                pass_2_idx = np.array(pass_1_idx)[knn_t > t_i]
                pass_2 = 1

            else:
                pass_2_idx = pass_1_idx
            n_degree += 1
                 
        # sampled by distance

        knn_p = self.connectivities[i,pass_2_idx]
        p = knn_p / knn_p.sum() if knn_p.sum() != 0 else None

        if len(final_index) == 0:
            final_index = knn_idx
            neighbor_idx = i
        else:
            neighbor_idx, p = self._sample_by_distance(self.distance, final_index)
        return neighbor_idx, (pass_1,pass_1_idx), (pass_2,pass_2_idx)

    def _diffuse_neighbor(self, i, c_i, t_i):
        """
        the Key function defines the noise sampling process 
        given the starting point i
        """
        pass_1 = 0
        pass_2 = 0
        n_degree = 0
        knn_idx = np.array([i])

        knn_dict = {'0':[i]}
        for d in range(self.max_degree):
            knn_next_degree = self.traverse_neighbor(knn_dict[str(d)])
            knn_dict[str(d+1)] = knn_next_degree
        searched = np.concatenate([knn_dict[str(d+1)] for d in range(self.max_degree)])
        
        knn_idx = searched
        knn_c = self.multipx_conditions[knn_idx]
        knn_t = self.T[knn_idx]

        # 1 : neighbor with the same condition
        if c_i in knn_c:
            pass_1_idx = np.array(knn_idx)[knn_c == c_i]
            pass_1 = 1
        else:
            pass_1_idx = knn_idx

        # 2 : neighbor with bigger pseudo-time
        if np.any(knn_t > t_i):
            pass_2_idx = np.array(knn_idx)[knn_t > t_i]
            pass_2 = 1
        else:
            pass_2_idx = knn_idx
        
        if pass_1 * pass_2 == 1:
            final_index = np.intersect1d(pass_1_idx, pass_2_idx)
        elif pass_2 == 0:
            final_index = []
        else:
            # pass_1 == 0 but pass_2
            final_index = pass_2_idx
        
        if len(final_index) == 0:
            final_index = knn_idx
            neighbor_idx = i
        else:
            neighbor_idx, p = self._sample_by_distance(self.distance[i].A.flatten(), final_index)
        return neighbor_idx, (pass_1,pass_1_idx), (pass_2,pass_2_idx)


    def __getitem__(self, i):
        """
        return x , b, c, noise, t in a mini-batch
        """
        X = self.X[i]
        c_string = self.multipx_conditions[i]
        t = self.T[i]
        neighbor_idx, (pass_1,pass_1_idx), (pass_2,pass_2_idx)  = self._diffuse_neighbor(i, c_string, t)

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

        X = self.X[i]
        delta_X = self.X[neighbor_idx] - X

        if self.check_samples:
            return X, batch_idx, condition_idx, delta_X, t, pass_1,pass_1_idx, pass_2,pass_2_idx
        
        # expression matrix  , expreimental batch , perturbation [c1, c2, c3..], expression change, t: discreted t
        else:
            return X, batch_idx, condition_idx, delta_X, t

class Path_Diffuse(Diffuse_Dataset):
    def __init__(self, 
                AnnData : AnnData, 
                unique_token_dict : dict,
                repeat: float = 1,
                alpha : float = None,
                condition_key : str = 'condition',
                max_multiplexing : int = 1,
                use_batch_index : bool = False,
                exp_batch_key : str = 'batch', 
                pseudotime_key : str = "dpt_pseudotime",
                neighbor_key : str = "",
                n_neighbor : int = 15,
                delimiter : str ="",
                layers:str ='counts', 
                split_key:str = 'split', 
                which_set: str ='train',
                max_degree: int = 5,
                search_strategy: str = 'traverse',
                check_samples = False
                ):
        super().__init__(AnnData, unique_token_dict=unique_token_dict, condition_key=condition_key, 
                            max_multiplexing=max_multiplexing,  pseudotime_key=pseudotime_key,
                            neighbor_key=neighbor_key, n_neighbor=n_neighbor,
                            use_batch_index=use_batch_index, exp_batch_key=use_batch_index,
                            delimiter = delimiter, layers=layers, split_key=split_key, which_set=which_set,
                            max_degree=max_degree, search_strategy=search_strategy, check_samples=check_samples
                            )
                        
        self.alpha = alpha
        self.repeat = repeat

        self.raw_X = torch.from_numpy(self.adata_raw.X).float()
        self.raw_i = self.adata_raw.obs_names
        self.raw_C = self.adata_raw.obs[condition_key].values
        self.raw_T = self.adata_raw.obs['discrete_time'].values

        self.connectivities = self.adata_raw.obsp[neighbor_key+'connectivities']
        self.connectivities = self._to_dense(self.connectivities)

        self.distance = self.adata_raw.obsp[neighbor_key+'distances']
        # self.distance = self._to_dense(self.distance)
    
    def index_subset_2_raw(self,i):
        cb_i = self.adata.obs_names[i]
        return np.where(self.raw_i == cb_i)[0]
    
    def subset_2_raw_index_ls(self,indexs):
        return [self.index_subset_2_raw(i) for i in indexs]

    def _neighbor_path(self, i, c_i, t_i):
        """
        The sampling process given cell i, we choose the other cell to construct a training pair.
        Here we define each cell can only belongs to six different scenario:
            - around_root
            - Orphan condition
            - Sample backward by condition
            - Sample forward by time
            - Single future state
            - Sample with condition and time
        """
        # free travese steps
        raw_i = self.index_subset_2_raw(i)
        knn_dict = {'0':[i]}
        Around_root = False
        for d in range(self.max_degree):
            knn_next_degree = self.traverse_neighbor(knn_dict[str(d)])
            knn_dict[str(d+1)] = knn_next_degree
            if self.iroot in knn_next_degree:
                Around_root = True


        if Around_root:
            neighbor_idx = self.iroot
            note = 'around_root'
            step = self.adata.obs['Depth_from_root'].values[i]

        else:
            # find path
            same_class_idx = np.where(self.raw_C == c_i)[0]
            gtr_t_idx = np.where(self.raw_T > t_i)[0]
            meet_both = np.intersect1d(same_class_idx, gtr_t_idx)

            dist_array = dijkstra(self.distance, indices=raw_i,
                                    limit=1000, return_predecessors=False)
            dist_array = dist_array.flatten()

            depth_from_i = dijkstra(self.distance, indices=raw_i, 
                    unweighted=True, limit=1000, return_predecessors=False)
            depth_from_i = depth_from_i.flatten()
            assert len(depth_from_i.shape) == 1, "Shape Error dist/depth array is not 1d"

            if len(meet_both) == 0:
                
                if len(same_class_idx) == 1:
                    neighbor_idx = self.iroot
                    step = self.adata.obs['Depth_from_root'].values[i]
                    note = 'Orphan condition'

                elif len(gtr_t_idx) > 0:
                    # use future cells
                    neighbor_idx, p = self._sample_by_distance(dist_array, gtr_t_idx, self.alpha, self.repeat)
                    step = depth_from_i[neighbor_idx]
                    note = 'Sample forward by time'

                else:
                    # i is the one with biggest time
                    neighbor_idx, p = self._sample_by_distance(dist_array, same_class_idx, self.alpha, self.repeat)
                    note = 'Sample backward by condition'
                    step = depth_from_i[neighbor_idx]
                

            elif len(meet_both) == 1:
                neighbor_idx = [meet_both[0]]
                note = 'Single future state'
                step = [depth_from_i[neighbor_idx]]
                
            else:
                # the idea situation
                neighbor_idx,p = self._sample_by_distance(dist_array, meet_both, self.alpha, self.repeat)

                step = depth_from_i[neighbor_idx]
                note = 'Sample with condition and time'
               
        return neighbor_idx, note, step


    def __getitem__(self, i):

        c_string = self.multipx_conditions[i]
        t = self.T[i]
        neighbor_idx, note, step  = self._neighbor_path(i, c_string, t)

        split_tokens = c_string.split(self.delimiter)
        n_tokens = len(split_tokens)

        if note in ['around_root', 'Orphan condition', 'Sample backward by condition']:
            # use root or backward
            
            X = self.root_x
            delta_X = (self.X[i] - X) 
            t = 0
            
        elif 'Orphan condition' in note:
            X = self.raw_X[neighbor_idx]
            if len(neighbor_idx) > 0:
                X = X.mean(axis=0)
            delta_X = (self.X[i] - X) 
            delta_X /= step

        elif note in ['Sample forward by time', 'Single future state', 'Sample with condition and time']:
            X = self.X[i]
            X_next = self.raw_X[neighbor_idx]
            if len(neighbor_idx) > 0:
                X_next = X_next.mean(axis=0)
            # delta_X = (X - X_next) #?????
            delta_X = (X_next - X ) #?????
        
        else:
            raise ValueError("Undefined scenario")

        # flip !!!
        # delta _X = -1 * delta_X
        
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
        
        if self.check_samples:
            return torch.from_numpy(X), batch_idx, torch.from_numpy(condition_idx), torch.from_numpy(delta_X), torch.from_numpy(t), note, step
        else:
            return X, batch_idx, condition_idx, delta_X, t

            

class Root_Diffuse(Diffuse_Dataset):
    def __init__(self, 
                AnnData : AnnData, 
                unique_token_dict : dict,
                root_cell : Union[int, str, np.ndarray] = None,
                condition_key : str = 'condition',
                max_multiplexing : int = 1,
                use_batch_index : bool = False,
                exp_batch_key : str = 'batch', 
                pseudotime_key : str = "dpt_pseudotime",
                neighbor_key : str = "",
                n_neighbor : int = 15,
                delimiter : str ="",
                layers:str ='counts', 
                split_key:str = 'split', 
                which_set: str ='train',
                max_degree: int = 5,
                search_strategy: str = 'traverse',
                check_samples = False
                ):
        super().__init__(AnnData, unique_token_dict=unique_token_dict, condition_key=condition_key, 
                            max_multiplexing=max_multiplexing,  pseudotime_key=pseudotime_key,
                            neighbor_key=neighbor_key, n_neighbor=n_neighbor,
                            use_batch_index=use_batch_index, exp_batch_key=use_batch_index,
                            delimiter = delimiter, layers=layers, split_key=split_key, which_set=which_set,
                            max_degree=max_degree, search_strategy=search_strategy, check_samples=check_samples
                            )


        self.root_cell = root_cell
        self.get_root_cell(root_cell)

        self.Degree = self.adata.obs['Depth_from_root'].values

    def get_root_cell(self, root_cell):
        """
        from the given cell index, trace the control cell expression.
        """
        if root_cell is None:
            self.iroot = self.adata_raw.uns['iroot']
            self.root_x = self.adata_raw.X[self.iroot]
        elif type(root_cell) == str:
            self.iroot = np.where(self.adata_raw.obs_names == root_cell)[0]
            self.root_x = self.adata_raw.X[self.iroot]
        elif type(root_cell) == int:
            self.iroot = root_cell
            self.root_x = self.adata_raw.X[self.iroot]
        elif type(root_cell) == np.ndarray():
            self.iroot = None
            self.root_x = root_cell
        else:
            raise ValueError("Undefined Data Type")
    
    
    def __getitem__(self, i):
        """
        return x , b, c, noise, t in a mini-batch
        """
        X = self.X[i]
        c_string = self.multipx_conditions[i]
        split_tokens = c_string.split(self.delimiter)
        n_tokens = len(split_tokens)

        # we pad the token list to maximal muultiplexing 
        # pad with the null key 
        if len(split_tokens) < self.max_multiplexing:
            split_tokens += [self.null_cond_key]*(self.max_multiplexing - n_tokens)
        condition_idx = np.array([self.unique_token_dict[token] for token in split_tokens])
        
        # batch
        if self.use_batch_index:
            batch_idx = self.exp_batch[i]
        else:
            batch_idx=[]

        # select by T
        degree = self.Degree[i]
        if degree == 0:
            degree = 1
        Delta_X = (self.X[i] - self.root_x) #/ degree

        # expression matrix X , expreimental batch , perturbation [c1, c2, c3..], noise -> delta X, t: discreted t
        return X, batch_idx, condition_idx, Delta_X, degree


class Fix_Degree_Diffuse(Diffuse_Dataset):
    def __init__(self, 
                AnnData : AnnData, 
                unique_token_dict : dict,
                condition_key : str = 'condition',
                max_multiplexing : int = 1,
                use_batch_index : bool = False,
                exp_batch_key : str = 'batch', 
                pseudotime_key : str = "dpt_pseudotime",
                neighbor_key : str = "",
                n_neighbor : int = 15,
                delimiter : str ="",
                layers:str ='counts', 
                split_key:str = 'split', 
                which_set: str ='train',
                max_degree: int = 5,
                search_strategy: str = 'traverse',
                check_samples = False
                ):
        self.super().__init__(AnnData, unique_token_dict, condition_key, max_multiplexing, 
                            use_batch_index, exp_batch_key, pseudotime_key, neighbor_key, 
                            n_neighbor, delimiter, layers, split_key, which_set, max_degree,
                            search_strategy, check_samples)
    
    def traverse_next_degree_neighbor(self, i, c_string, dpt):
        """
        
        """

        idx_ci = np.where(self.multipx_conditions == c_string )[0]
        dpt_ci = self.T[idx_ci]
        dpt_gtr = (dpt_ci > dpt)[idx_ci]

        
        return knn_idx_d_plus1

    def _diffuse_neighbor(self, i, c_i, t_i):
        """
        the Key function defines the noise sampling process 
        given the starting point i
        """
        X = self.X[i]
        c_string = self.multipx_conditions[i]
        dpt = self.T[i]
        split_tokens = c_string.split(self.delimiter)
        n_tokens = len(split_tokens)

        # we pad the token list to maximal muultiplexing 
        # pad with the null key 
        if len(split_tokens) < self.max_multiplexing:
            split_tokens += [self.null_cond_key]*(self.max_multiplexing - n_tokens)

        condition_idx = np.array([self.unique_token_dict[token] for token in split_tokens])

        # batch
        if self.use_batch_index:
            batch_idx = self.exp_batch[i]
        
        else:
            batch_idx=[]

        # select by T
        degree = self.Degree[i]

        Delta_X = (self.X[i] - self.root_x) / degree
        

        # expression matrix X , expreimental batch , perturbation [c1, c2, c3..], noise -> delta X, t: discreted t
        return X, batch_idx, condition_idx, Delta_X, degree

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
                n_neighbor : int = 15,
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