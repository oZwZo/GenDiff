import os, path_n_util
if __name__ == '__main__':
    parser = path_n_util.main_train_parser()
    args = parser.parse_args()
    os.environ['CUDA VISIBLE DEVICES'] = args.CUDA

import src
import time
import numpy as np
from tqdm import tqdm
import scanpy as sc
import scvelo as scv
from scipy.sparse.csgraph import dijkstra

from src import _diffplot as dfp
from src import _configure

# save dir
config_path = args.model_config
config_base = os.path.basename(config_path).split(".yaml")[0]
design_dir = os.path.basename(os.path.dirname(config_path))
higher_dir = os.path.join(path_n_util.main_dir, f"result/{design_dir}")

if not os.path.exists(higher_dir):
    os.makedirs(higher_dir)
     
save_dir = os.path.join(path_n_util.main_dir, f"result/TFAtlas/{design_dir}", config_base)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

date = time.strftime("%B%d")
save_path = os.path.join(save_dir, f"Sampled_deltaX_{date}.npy")

# sample func
Path_delta_x = dfp.sample_Delta_X(yaml_path = args.model_config,
                sampling_repeat = 30, 
                n_workers=args.n_workers, 
                return_repeat=True
                    )

np.save(save_path, Path_delta_x)