# %load_ext autoreload
# %autoreload 2
import os, sys, path_n_util, time, argparse

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import anndata as ad

from src import _sampler
from src import _learner
from src import _reader
from src import _configure
from src import _diffplot as dfp

import torch
from torch import nn
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script to convert scanpy h5ad file into 4 10x raw files")
    parser.add_argument("-M", "--model_config", type=str, required=True, help="the absolute path of the yaml file")
    parser.add_argument("-C", "--model_checkpoint", type=str, required=True, help="the relative path of the checkpoint file, end with version_x")
    parser.add_argument("-S", "--save_prefix", type=str, default = 'Vis&Perb', required=False, help="the prefix to replace .h5ad")
    parser.add_argument("-D", "--device", type=int, default=2, required=False, help="the cuda to use")
    parser.add_argument("-J", "--jobs", type=int, default=5, required=False, help="the number of nodes to compute velocity graph")
    args = parser.parse_args()
    #
    device = args.device if torch.cuda.is_available() else 'cpu'

    # 
    config_path = os.path.join(path_n_util.main_dir, args.model_config)
    model_config = _configure.Yaml_configurer(config_path) 
    abs_checkpoint = dfp.get_ckpt_path(args.model_checkpoint)
    assert os.path.exists(abs_checkpoint) , "invalid argument : model-checkpoint relative path"

    version = args.model_checkpoint.split("_")[-1]
    config_base = os.path.basename(config_path).split(".yaml")[0]

    adata = sc.read_h5ad(model_config.anndata_path)

    # save dir
    save_dir = os.path.join(path_n_util.main_dir, "result/TFAtlas", config_base)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    date = time.strftime("%B%d")
    h5ad_save_path = os.path.join(save_dir, f"V{version}_{date}_{args.save_prefix}.h5ad")
    png_save_path = os.path.join(save_dir, f"V{version}_{date}_{args.save_prefix}.png")

    # visualize cb order
    idxs = adata.obs.query("`Cluster Enriched TFs` == 'others'").index
    oidxs = adata.obs.query("`Cluster Enriched TFs` != 'others'").index
    vis_order = np.concatenate([idxs, oidxs], axis=0)


    # perturb
    for TF in adata.obs['Cluster Enriched TFs'].unique():
        
        if TF == 'others':
            TF = 'TFORF3549-GFP'
            
        perturb_deltaX = dfp.perturbation(
            yaml_path = config_path,
            ckpt_path = args.model_checkpoint,
            perturbation = TF,
            n_neighbors= 15,
            device = device,
            return_adata=False,
            compute_neighbor=False,
            compute_umap=False
        )

        adata.obsm['%s_deltaX'%TF] = perturb_deltaX

    
    adata.write_h5ad(h5ad_save_path)


