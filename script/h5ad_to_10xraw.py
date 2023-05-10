import os
import sys
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import io

def convert_fn(adata, save_dir):
    """
    Convert adata to seperated obs, var, mtx and meta files that are readable for seurat.
    Noted that by default, adata.raw will be first considered to convert. 

    Input
    ----------
    adata : scanpy.anndata
    save_dir : abs path of the directoery
    """

    # check raw
    try:
        adata = adata.raw.to_adata()
    except:
        adata = adata

    # check path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("create directory : %s" %save_dir)
    
    with open(f"{save_dir}/barcodes.tsv", "w") as f:
        for item in adata.obs_names:
            f.write(item + '\n')
        f.close()

    with open(f"{save_dir}/features.tsv", "w") as g:
        for item in ['\t'.join([x,x,"Gene Expression"]) for x in adata.var_names]:
            g.write(item + '\n')
        g.close()

    io.mmwrite(f"{save_dir}/matrix", adata.X.T)
    adata.obs.to_csv(f"{save_dir}/metadata.csv", index=True)

    print('Done!')
    print('All things saved to %s' %save_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script to convert scanpy h5ad file into 4 10x raw files")
    parser.add_argument("-P", "--path_h5ad", type=str, required=True, help="the absolute path of the h5ad file")
    args = parser.parse_args()

    # input
    adata = sc.read_h5ad(args.path_h5ad)
    save_dir = args.path_h5ad.replace(".h5ad", "_10xmatrices")

    convert_fn(adata, save_dir)
