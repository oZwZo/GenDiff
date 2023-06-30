import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.chdir('../')
import warnings
import scanpy as sc
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
import anndata as ad
from sklearn.model_selection import train_test_split

# the data path
main_dir = '/home/wergillius/Project/diffuse_differentiate/'
data_dir = '/home/wergillius/Project/diffuse_differentiate/data/Fetal_reference/'
raw_data = os.path.join(main_dir,"data/TFAtlas/GSE217460_RAW")


big_ad = sc.read_h5ad(f"{main_dir}data/TFAtlas/GSE217460_210322_TFAtlas.h5ad")

# expX = np.exp(big_ad.X) -1
# n_norm_sum = expX.sum(axis=1)
# scaler = big_ad.obs['n_counts'].values / n_norm_sum  

# expX = expX * scaler.reshape(-1,1) 
# expX_int = np.rint(expX)

# # big_ad.X = csr_matrix(big_ad.X)
# big_ad.layers['counts'] = expX_int.astype(int)
# big_ad.layers['log_1p'] = big_ad.X

expX_int = big_ad.layers['counts']
big_ad.layers['counts'] = csr_matrix(expX_int)

big_ad.write_h5ad(f"{main_dir}data/TFAtlas/GSE217460_210322_TFAtlas_raw.h5ad")