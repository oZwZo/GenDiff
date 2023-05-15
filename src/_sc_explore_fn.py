import scanpy as sc
import pandas as pd
import numpy as np

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