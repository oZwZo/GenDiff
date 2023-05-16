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
    parser.add_argument("-P", "--perturb_conditions", type=str, required=False, 
                        default = '["TFORF2078-FLI1", "TFORF0098-MSGN1", "TFORF2055-CDX1"]', 
                        help="the condition to perturb , format ['c0' , 'c1', 'c2']")
    parser.add_argument("-D", "--device", type=int, default=2, required=False, help="the cuda to use")
    parser.add_argument("-J", "--jobs", type=int, default=5, required=False, help="the number of nodes to compute velocity graph")
    args = parser.parse_args()


    # 
    device = args.device if torch.cuda.is_available() else 'cpu'
    pertrub_condition_ls = eval(perturb_conditions)

    # 
    config_path = os.path.join(path_n_util.main_dir, args.model_config)
    model_config = _configure.Yaml_configurer(config_path) 
    abs_checkpoint = dfp.get_ckpt_path(args.model_checkpoint)
    assert os.path.exists(abs_checkpoint) , "invalid argument : model-checkpoint relative path"

    version = args.model_checkpoint.split("_")[-1]
    config_base = os.path.basename(config_path).split(".yaml")[0]
    design_dir = os.path.basename(os.path.dirname(config_path))

    adata = sc.read_h5ad(model_config.anndata_path)

    # save dir
    higher_dir = os.path.join(path_n_util.main_dir, f"result/{design_dir}")
    if not os.path.exists(higher_dir):
        os.makedirs(higher_dir)

    save_dir = os.path.join(path_n_util.main_dir, f"result/TFAtlas/{design_dir}", config_base)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    date = time.strftime("%B%d")
    h5ad_save_path = os.path.join(save_dir, f"V{version}_{date}_{args.save_prefix}.h5ad")
    png_save_path = os.path.join(save_dir, f"V{version}_{date}_{args.save_prefix}.png")

    # visualize cb order
    idxs = adata.obs.query("`Cluster Enriched TFs` == 'others'").index
    oidxs = adata.obs.query("`Cluster Enriched TFs` != 'others'").index
    vis_order = np.concatenate([idxs, oidxs], axis=0)

    # compute latent
    adata_zc, model  = dfp.plot_representation(
        yaml_path = config_path,
        ckpt_path = args.model_checkpoint,
        use_rep = 'z_c',
        n_neighbors= 15,
        device = device
        )
    print("latent generated!")
    #

    n_grid = 12 + 2*len(pertrub_condition_ls)

    # visualize latent
    fig = plt.figure(tight_layout=True, dpi=500, figsize=(12,n_grid*2.5))
    gs = gridspec.GridSpec(n_grid, 5)

    root = adata.uns['iroot']
    fig1, axs = plt.subplots(3,2, figsize=(12,15), dpi=400,tight_layout=True)
    axs = axs.flatten()
    i = 0

    for color_key in ['discrete_time', 'louvain', 'Cluster Enriched TFs']:
        
        ax = fig.add_subplot(gs[i:i+1, 0:1])
        sc.pl.umap(adata_zc[vis_order], color=color_key, 
                ax=ax, return_fig=False)
        ax.scatter(adata_zc.obsm['X_umap'][root,0], 
                    adata_zc.obsm['X_umap'][root,1], 
                    color='orange', marker = '^')

        i += 2


    # velocity
    print("computing velocity graph!")
    velocity_matrix = adata_zc.obsm['delta_x']
    adata_velo = dfp.compute_velocity(
        adata=adata,
        velocity_matrix = velocity_matrix,
        n_jobs = args.jobs
    )


    ax = fig.add_subplot(gs[6:8, 0:3])
    scv.pl.velocity_embedding_stream(adata_velo,
                                    color='louvain', 
                                    legend_loc='far right', 
                                    ax = ax
                                    )


    ax = fig.add_subplot(gs[9:11, 0:3])
    scv.pl.velocity_embedding(adata_velo,
                              arrow_size = 3,
                              arrow_length = 4,
                                    color='louvain', 
                                    legend_loc='far right', 
                                    ax = ax
                                    )
    
    i = 12 
    if len(pertrub_condition_ls) > 0:
    
        for TF in pertrub_condition_ls:
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
            
            adata_perturb = dfp.compute_velocity(
                adata=adata,
                velocity_matrix = perturb_deltaX,
                n_jobs = args.jobs
            )

            ax1 = fig.add_subplot(gs[i:i+1, 0:1])
            scv.pl.velocity_embedding(adata_perturb,
                                    arrow_size=2,
                                    arrow_length=3,
                                        color='louvain', 
                                        legend_loc='none', 
                                        ax = ax1
                                        )
            ax2 = fig.add_subplot(gs[i:i+1, 2:3])
            scv.pl.velocity_embedding_stream(adata_perturb,
                                        color='louvain', 
                                        legend_loc='far right', 
                                        ax = ax2
                                        )
            ax1.set_title(TF)

            
            i += 2
    fig.savefig(png_save_path)