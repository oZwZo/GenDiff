import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.chdir('../')
import warnings
import scanpy as sc
import pandas as pd
import numpy as np

# the data path
main_dir = '/home/wergillius/Project/diffuse_differentiate/'
data_dir = '/home/wergillius/Project/diffuse_differentiate/data/Fetal_reference/'


fetal_h5 = os.path.join(data_dir, 'GSE156793_protein_coding.h5ad')
adata_fetal = sc.read_h5ad(fetal_h5)
sc.pp.filter_cells(adata_fetal, min_counts=100)
sc.pp.filter_genes(adata_fetal, min_cells=15)


adata_diff = sc.read_h5ad(f"{main_dir}/data/TFAtlas/GSE217460_210322_TFAtlas_differentiated.h5ad")

overlapped_genes = np.intersect1d(adata_fetal.var['gene_short_name'].values, adata_diff.var.index)
print('common genes ', len(overlapped_genes))

overlapped_hvg = np.intersect1d(fetal_hvg.var.query("`highly_variable` == True")['gene_short_name'].values, adata_diff.var.index)
print('common variable genes ', len(overlapped_hvg))

fvar_df = fetal_hvg.var.copy()
fvar_df = fvar_df.query("`gene_short_name` in @overlapped_genes").sort_values('Mean_expression', ascending=False)
fvar_df = fvar_df.drop_duplicates(['gene_short_name'], keep='first')

com_hvg_index = fvar_df['level_0'].values

fetal_hvg = fetal_hvg[:,overlapped_hvg]