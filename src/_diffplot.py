import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import scanpy as sc
from src import _epsilon_module
import _reader
import _epsilon_module
import _sampler
import _configure
import scvelo
import PATH
import anndata as ad
from anndata import AnnData 
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
sys.path.append(os.path.join(PATH.main_dir, "script"))
from script.main_train_with_PL_trainer import dl_from_config, get_model_from_config, get_sampler_from_configs

mesc_marker_genes={
        "Caudal Mesodem" : ['Cdx1', 'Cdx4', 'Hoxaas3', 'Fgfbp3'],
        "Neural Crest" : ["Pax6", "Sfrp1", "Zic1",  "Ptn"],
        "Anterior Primitive Streak" : ["T", "Mixl1", "Mesp1",  "Aplnr"],
        "Surface Ectoderm" : ["Tfap2a", "Dlx5", "Bambi",  "Wnt6"],
        "Paraxial mesoderm" : ["Tbx6", "Dll1", "Aldh1a2", "Cited1"],
        "Allantois": ["Hand1", "Plac1", "Tgfb2", "Pitx1"],
        # "Anterior Primitive Streak" : ["Gsc", "Eomes", "Lhx1", "Otx2"],
        "Somitic mesoderm: Meox1": ["Meox1","Foxc2", "Gas1", "Ebf1"],
        # "Primitive hematopoietic" : ["Tal2", "Cdx4", "Itga4", "Ephb1"],
        "Notochord": ["Foxa2", "T", "Foxj1", "Slit2"]
}
mesc_marker_genes_thress={
        "Caudal Mesodem" : [2,6,6,3],
        "Neural Crest" : [8,5,8,4],
    
        "Anterior Primitive Streak" : [3, 4, 2, 2],
        "Surface Ectoderm" : [5,5,2,2.5],
        "Paraxial mesoderm" : [6, 8, 6,3],
        "Allantois": [4, 6, 4, 2],
    
        # "Anterior Primitive Streak" : [8, 6, 8, 5],
        "Somitic mesoderm: Meox1": [2,2,3,4],
        # "Primitive hematopoietic" : ["Runx3", "Cdx4", "itga4", "ephb1"],
        "Notochord": [1,2,6,4]
}

# checkpoint
def get_ckpt_path(relative_path):
    """the relative path is the name Shown in TensorBoard"""
    onemore_layer = os.path.join(PATH.pth_dir, relative_path, "checkpoints")
    ckpts = [file for file in os.listdir(onemore_layer) if file.endswith(".ckpt")]
    abs_paths = [os.path.join(onemore_layer, ckpt) for ckpt in ckpts]
    if len(abs_paths) == 1:
        return abs_paths[0]
    else: 
        return abs_paths
    
def plot_representation(yaml_path, ckpt_path):

    # config
    model_config = os.path.join(PATH.main_dir, yaml_path)
    configs = _configure.Yaml_configurer(model_config)

    v0_ckpt = get_ckpt_path(
        ckpt_path
            )

    # models
    eps_net = get_model_from_config(configs, "0")
    Sampler_pl_module = eval("_learner."+configs.sampler_class)

    v0_equi_diff = Sampler_pl_module.load_from_checkpoint(v0_ckpt, model=eps_net).to('cpu')

    v0_equi_diff.eval();


    train_iter = iter(train_dl)

    z_ls = []
    z_c_ls = []
    c_ls = []
    for X, batch_idx, c, noise, t in tqdm(train_iter):

        z_dict = v0_equi_diff.model.encode(X, t , None , c)

        z_ls.append(z_dict['z'].detach().cpu().numpy())
        z_c_ls.append(z_dict['z_c'].detach().cpu().numpy())
        c_ls.append(z_dict['c'].detach().cpu().numpy())

    z_ay = np.concatenate(z_ls, axis=0)
    z_c_ay = np.concatenate(z_c_ls, axis=0)
    c_ay = np.concatenate(c_ls, axis=0)

    adata_zc = adata.copy()
    adata_zc.obsm['Z_c'] = z_c_ay

    # sc.pp.neighbors(adata_zc, n_neighbors = 35,  metric='cosine', method='umap', key_added='Z_c' ,use_rep='Z_c', )
    sc.pp.neighbors(adata_zc, n_neighbors = 60,  key_added='Z_c' ,use_rep='Z_c', )
    sc.tl.umap(adata_zc, min_dist = 0.5, maxiter=500, spread=1, random_state=0, neighbors_key='Z_c')

    sc.pl.umap(adata_zc, color=['discrete_time', 'assignment'])

    return adata_zc

#   - device -
def reload_sampler(yaml_file):
    # define configure
    configs = _configure.Yaml_configurer(yaml_file)
    module_kw = configs.epsilon_kwargs
    Module_Class = eval("_epsilon_module.%s" %configs.epsilon_class)

    # - pretrain - 
    if configs.pretrain_embedder_pth is not None:
        pretrain_embedder = torch.load(configs.pretrain_embedder_pth)
        if configs.fix_pretrain:    
        # fix embeddings at the beginning
            for p in pretrain_embedder.parameters():
                p.required_grads = False
        module_kw['pretrained_embeddings'] = pretrain_embedder

    # - define module -
    
    # detect saved eps net
    save_path = os.path.join(PATH.pth_dir , configs.epsilon_class, 
                    os.path.basename(yaml_file).replace(".yaml",".pth"))
    if os.path.exists(save_path):
        eps_net = torch.load(save_path, map_location='cpu')
        print(f"epsilon net is loaded from \n{save_path}")
    else:
        eps_net = Module_Class(**module_kw).to(device)
    
    Samper_Class = eval("_sampler.%s" %configs.sampler_class)
    sampler_kwargs = configs.sampler_kwargs
    sampler_kwargs['model'] = eps_net

    Sampler = Samper_Class(**sampler_kwargs)
    return Sampler

def condition_on_umap(adata, c1, basis='umap', 
                                subplot_kw={"dpi":100, "figsize":(5,4)},):
    cond1_data = adata[adata.obs['pert1'] == c1].copy()

    fig, ax = plt.subplots(1,1, **subplot_kw)
    all_umap = adata.obsm[f'X_{basis}']

    ax.scatter(all_umap[:,0], all_umap[:,1], s=4, alpha=0.1, color='#808080')
    sc.pl.scatter(cond1_data, basis=basis, alpha=0.7, size=30, 
                  frameon=False, color=['pert2'], ax=ax)
    return fig, ax



def Xmat_by_time(time_series_X, time_interval=None, var_indexs=None, color_norm=None, log_transform=False):
    fig, axs = plt.subplots(1, 6, figsize=(13,2), dpi=300)
    cax = fig.add_axes([0.92,0.11,0.01, 0.75 ])
    axs = axs.flatten()
    total_time_step = len(time_series_X)
    
    if isinstance(time_series_X[0], torch.Tensor):
        time_series_X = [x.detach().cpu().numpy() for x in time_series_X]
    
    if log_transform:
        time_series_X = [np.log10(x+1e-10) for x in time_series_X]
    
    if var_indexs is not None:
        time_series_X = [x[:, var_indexs] for x in time_series_X]
    
    if time_interval is None:
        time_interval = total_time_step//6
    
    if color_norm is None:
        vmax = np.max([x.max() for x in time_series_X])
        vmin = np.min([x.min() for x in time_series_X])
        color_norm = Normalize(vmin=vmin, vmax=vmax)
        
    for i in range(6):
        ax = axs[i]
        t = max(0, i * time_interval)
        mat = time_series_X[t]
        g=ax.imshow(mat, aspect='auto', norm=color_norm)
        ax.set_title("t=%d"%t)
        ax.axis('off')
    
    plt.colorbar(g, cax=cax)
    
    return fig, axs, color_norm


def get_annDataLoader(annData, unique_token_dict=None, layers='counts',**kwargs):
    """
    Given a processed annData, return the dataloader
    Input
    -----------
    annData
        the X-0
    uniue_token_dict:
        the token look up table
    kwargs:
        dict, key word arguments will be passed to DataLoader
    
    Return
    ------------
    annDataLoader:
        dataloader, (x, c) in a mini-batch
    """
    # sanity check
    assert 'pert1' in annData.obs_keys() and 'pert2' in annData.obs_keys()
    if unique_token_dict is None:
        assert 'condition_token' in annData.uns_keys(), "please specify the token lookup dict"
        unique_token_dict=annData.uns['condition_token']
    
    
    annDataSet=_reader.Condition_AnnDataSet(annData, layers=layers,
                                unique_token_dict=unique_token_dict)
    annDataLoader = DataLoader(annDataSet, shuffle=False,**kwargs)
    return annDataLoader

def diffuse_cell(annData, Sampler, layers='counts', batch_size=400, sampled_time = range(20), discrete_time=False,
                 reuse_loading=True, plot_out=False, dpi=100):
    """
    Apply q-sample process to add noise (diffuse) to the cells, and save each time-point in a new annData object
    
    Input
    --------
    annData:
        clean data, X-0
    Sampler:
        DDPM/DDIM sampler, whose `q-sample` method will be used to add noise
    batch_size:
        int: how many cells to diffuse
    Sampled_time:
        list of int, the timepoint to saved in the annData.
    reuse_loading:
        Bool: default True. Use the PCA loading in annData to compute PCs for the diffused X, \
        so that all the PCs are stayed in the same space. This will affect the neighbors and UMAP.\
        If set to False, recompute the PCs on all noised data.
    plot_out:
        Bool: default False. Whether we visualize the noised expression matrix in PCA, UMAP and Diff-map.\
        Setting to True will triger the computation of 
    dpi:
        plot, setting
    Return
    --------
    qt_annData
        annData of shape (n-sampled * n_Cells , n_var), expanded annData.
    """
    # sample a batch of cells and diffuse (add noise)
    example_dl = get_annDataLoader(annData, layers=layers, batch_size=batch_size)
    x_0, c = next(iter(example_dl)) # only take the first batch
    if batch_size < x_0.shape[0]:
        batch_size = x_0.shape[0]
    
    ts2nday = lambda x: x.detach().cpu().numpy()
    X_noised = [ts2nday(x_0)] + [ts2nday(x) for x in Sampler.q_sample_loop(x_0)]
    
    # concatenate X and other observation
    X_qt = np.concatenate([X_noised[t] for t in sampled_time], axis=0)
    T_qt = np.concatenate([[t]*batch_size for t in sampled_time], axis=0)
    #  -  cell barcode
    base_cb = list(annData.obs.index[:batch_size])
    qt_cb = np.concatenate([[cb+f"-t{t}" for cb in base_cb] for t in sampled_time], axis=0)
    #  -  duplicate cell obs
    basedf = annData.obs.loc[base_cb]
    qt_obs = pd.concat([basedf]*len(sampled_time), axis=0)
    qt_obs.index = qt_cb
    if discrete_time:
        qt_obs['q_sample_timepoint'] = T_qt.astype(str)
    else:
        qt_obs['q_sample_timepoint'] = T_qt
        
    
    if (layers in annData.layers):
        
        # create a new annData from it
        qt_annData = AnnData(X_qt, obs=qt_obs, var=annData.var)
        if reuse_loading:
            print("Reuse the loading of annData, preserving cells in the same PC space")
            centered_X = qt_annData.X - qt_annData.X.mean(axis=0).reshape(1,-1)
            qt_annData.obsm['X_pca'] = centered_X @ annData.varm['PCs']
        else:
            print("Re-compute the PC space for the diffused cells")
            sc.tl.pca(qt_annData)
            
    elif layers == 'X_pca':
        # use the 
        X_reconX = [x@ annData.varm['PCs'].T for x in X_qt]
        X_reconX_ay = np.stack(X_reconX)
        qt_annData = AnnData(X_reconX_ay, obs=qt_obs, var=annData.var)
        qt_annData.obsm['X_pca'] = X_qt
        
    
    print("Dimension Reduction...")
    sc.pp.neighbors(qt_annData)
    print("leiden clustering...")
    sc.tl.leiden(qt_annData)
    print("computing umap...")
    sc.tl.umap(qt_annData)
    print("computing diffmap...")
    # sc.pp.neighbors(qt_annData, method='gauss')
    # sc.tl.diffmap(qt_annData)
    # sc.tl.dpt(qt_annData, n_dcs=15)

    
        
    if plot_out:
        triple_plot(qt_annData, "q_sample_timepoint", dpi=dpi);
    
    return qt_annData

def triple_plot(annData:AnnData, color_key:str, dpi:int=100, **kwargs):
    """
    Input
    --------
    annData
    color_key:
        str, must exists in 
    """
    fig, axs = plt.subplots(1,3,figsize=(15,3), dpi=dpi, gridspec_kw={"wspace":0.4})

    sc.pl.pca(annData, color=color_key,  ax=axs[0], legend_loc=None,
              show=False, annotate_var_explained=False,**kwargs)
    sc.pl.umap(annData, color=color_key, show=False, legend_loc=None, ax=axs[1],**kwargs)
    sc.pl.diffmap(annData, color=color_key,show=False, ax=axs[2],**kwargs)
    for ax in axs:
        sns.despine(ax=ax)
    return fig, axs

def compute_velocity(adata, velocity_matrix):
    adata1 = adata.copy()
    assert velocity_matrix.shape == adata1.X.shape
    adata1.layers['velocity'] = velocity_matrix
    adata1.layers['X'] = adata1.X

    del adata1.uns['neighbors']

    sc.pp.neighbors(adata1, n_neighbors=30)
    scvelo.tl.velocity_graph(adata1, xkey='X')
    return adata1


def merge_with_control(diffused_annData:AnnData, control_annData:AnnData, subsample_control:float=1):
    """
    Input
    --------
    diffused_annData
    
    control_annData
    
    subsample_control
        float default=1, the fraction of control cells to be merged. Subsampling may be useful \
        when the diffused cells are much fewer than control cells
    
    Return
    --------
    merged_bdata
        annData, of shape (n1+n2*subsampled, )
    """
    control_annData.obs['q_sample_timepoint'] = "control"
    if subsample_control <1:
        # use DataFrame.sample to sub-sample cellbarcodes
        subsampled_cbs = control_annData.obs.sample(frac=subsample_control).index
        control_annData = control_annData[subsampled_cbs].copy()

    merged_bdata = ad.concat([control_annData,diffused_annData], axis=0)
    
    if 'X_umap' in merged_bdata.obsm:
        del merged_bdata.obsm['X_umap']
    if 'X_diffmap' in merged_bdata.obsm:
        del merged_bdata.obsm['X_diffmap']

    sc.pp.neighbors(merged_bdata)
    print("computing diffmap")
    sc.tl.diffmap(merged_bdata)
    print("computing umap")
    sc.tl.umap(merged_bdata)
    sc.tl.leiden(merged_bdata)
    
    return merged_bdata
    