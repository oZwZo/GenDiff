import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.linear_model import  Ridge



adata_diff = sc.read_h5ad(
    "/home/wergillius/Project/diffuse_differentiate/data/TFAtlas/GSE217460_210322_TFAtlas_differentiated.h5ad"
)

# get genes
TF_onehot = pd.get_dummies(adata_diff.obs.TF).values
gene_X = adata_diff.X
X =  np.concatenate([gene_X, TF_onehot], axis=1)

Y = adata_diff.layers['a3_PathSampled_X'].copy()

train_idx = adata_diff.obs.split != 'test'
test_idx = adata_diff.obs.split == 'test'

x_train = X[train_idx]
x_test = X[test_idx]

Y_train = Y[train_idx]
Y_test = Y[test_idx]


# fit models
model_dict = {}
params_dict = {}
r_dict = {}

# can change the gene to fit
genes = list(adata_diff.var.index)

for gene in genes:

    gene_varid = np.where(adata_diff.var.index == gene)[0].item()

    y_train = Y_train[:,gene_varid]
    y_test = Y_test[:,gene_varid]

    model = Ridge().fit(x_train,y_train)
    params = model.coef_.tolist() + [model.intercept_]
    y_pred = model.predict(x_test)
    r = pearsonr(y_pred, y_test)[0]

    model_dict[gene] = model
    r_dict[gene] = r
    params_dict[gene] = params

r_df = pd.DataFrame(np.stack(r_dict.values()), columns=['Pearson_r'], index=genes)
r_df.to_csv("ridge_model_r.csv", index=True)

params_array = np.stack(params_dict.values())
np.save('ridgemodel_params.npy', params_array)