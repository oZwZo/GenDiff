import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse.csgraph import dijkstra
import PATH
from scipy import io
from scipy.sparse import csr_matrix

TF_atlas_useful_celltype = ['Bronchiolar and alveolar epithelial cells', 
                    'Ciliated epithelial cells',
                    'Smooth muscle cells',
                    'ENS neurons',
                    'Squamous epithelial cells',
                    'Intestinal epithelial cells',
                    'Metanephric cells',
                    'PAEP_MECOM positive cells',
                    'Stromal cells',
                    'Syncytiotrophoblasts and villous cytotrophoblasts',
                    'Ureteric bud cells',
                    'Vascular endothelial cells',
                  ]

def hightlightcell(x, celltype_to_keep):
    if x in celltype_to_keep:
        anno = x.replace(" cells", "").lower()
    else:
        anno = 'others'
    return anno


def rgb_formater(rgb_list):
    rgb = np.array(rgb_list) / 255
    return rgb

def rgb_to_hex(rgb_ls): 
    return '#{:02x}{:02x}{:02x}'.format(*rgb_ls)

def TFAtlas_anno_control(adata):
    adata.obs['is_control'] = adata.obs.TF.isin(['TFORF3549-GFP', 'TFORF3550-mCherry']).astype(int)

def find_unreachable(adata, select_idex=None):
    select_idex = adata.uns['iroot'] if select_idex is None else select_idex

    Dist = dijkstra(adata.obsp['H_KNN_distances'],
        indices=select_idex, unweighted=False
                )
    adata.obs['not_reachable'] = np.isinf(Dist)[0].astype(str)

def save_pred_sampled_deltaX(adata, npy_path, key='PreSampled'):
    pj = lambda x: os.path.join(PATH.main_dir, x)
    deltaX = np.load(pj(path))
    if len(deltaX.shape) == 4:
        deltaX = deltaX[:,:,0,:]
    
    key_v = 0
    raw_key = key
    while key in adata.obsm_keys():
        key_v += 1
        key = raw_key + "_%d" %key_v

    adata.obsm[key] = deltaX.mean(axis=0)

def convert_adata_to_10x(adata, save_dir):
    """
    Convert adata to seperated obs, var, mtx and meta files that are readable for seurat.
    Noted that by default, adata.raw will be first considered to convert. 

    Input
    ----------
    adata : scanpy.anndata
    save_dir : abs path of the directoery
    """
    with open(f"{save_dir}/Int_obsname.tsv", 'w') as f:
        for item in adata.obs_names:
            f.write(item+"\n")
    
    with open(f"{save_dir}/Int_varname.tsv", 'w') as f:
        for item in ["\t".join([x,x,'Gene Expression']) for x in adata.var_names]:
            f.write(item+"\n")
    
    io.mmwrite(f"{save_dir}/matrix", csr_matrix(adata.X.T))
    adata.obs.to_csv(f"{save_dir}/metadata.csv", index=True)

    print('saved')