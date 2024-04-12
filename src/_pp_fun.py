import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.chdir('../')
import warnings
import scanpy as sc
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
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