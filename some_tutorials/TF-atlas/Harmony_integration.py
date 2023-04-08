
import os
os.chdir('../')
import warnings
import scanpy as sc
import pandas as pd
import numpy as np
import scanpy.external as sce


# the data path
main_dir = '/home/wergillius/Project/diffuse_differentiate/'
data_dir = '/home/wergillius/Project/diffuse_differentiate/data/Fetal_reference/'


adata = sc.read_h5ad(f"{main_dir}/data/TFAtlas/Be4_IntFetal_hvg.h5ad")
sce.pp.harmony_integrate(adata, 'batch', max_iter_harmony = 30, max_iter_kmeans = 50)

# init
adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)

adata.write_h5ad(f"{main_dir}/data/TFAtlas/After_IntFetal_hvg.h5ad")