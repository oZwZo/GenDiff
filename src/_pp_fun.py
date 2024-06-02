import os
import scanpy as sc
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from scipy.sparse.csgraph import dijkstra
from sklearn.model_selection import train_test_split

def token_labels():
    all_token = pd.read_table(f'{main_dir}data/TFAtlas/tokens.txt',
            sep='\t', names=['TF'])['TF'].values
    token_dict = {token:int(token.split("-")[0].replace("TFORF","")) for token in all_token}
    idx_token_dict = {i:t for t, i in token_dict.items()}
    idx_token_dict[0] = 'TFORF3549-GFP'
    return token_dict, idx_token_dict

def discrete_time(x):
    x = int(x*250)
    mid_err = np.random.randint(-2,2) # smooth the time
    tail_err = np.random.randint(0,10)
    return min(x, 200-tail_err)

def random_split(adata):
    trainval, test = train_test_split(adata.obs.index, test_size=0.1)
    train, val = train_test_split(trainval, test_size=0.1)

    adata.obs['split'] = 'o'

    adata.obs.loc[train ,'split'] = 'train'
    adata.obs.loc[val ,'split'] = 'val'
    adata.obs.loc[test ,'split'] = 'test'
    
    
    
def infer_knn_depth(adata, distance = 'KNN_distances'):

    k = adata.uns['neighbors']['params']['n_neighbors']
    sc.pp.neighbors(adata, n_neighbors=k, knn=True, key_added="KNN")

    degree = dijkstra(adata.obsp['KNN_distances'], indices=adata.uns['iroot'],
                unweighted=True, limit=1000,
                return_predecessors=False)

    n_max = degree[~np.isinf(degree)].max()
    print('unreachable points (cells)', np.isinf(degree).sum())
    print('max degree', n_max)

    imputed_degree = np.where(np.isinf(degree), n_max+1, degree)

    adata.obs['Depth_from_root'] = imputed_degree