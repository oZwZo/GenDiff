import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse.csgraph import dijkstra

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
    select_idex = adata.uns['iroot'] is select_idex is None else select_idex

    Dist = dijkstra(adata.obsp['H_KNN_distances'],
        indices=select_idex, unweighted=False
                )
    adata.obs['not_reachable'] = np.isinf(Dist)[0].astype(str)
