import os
import dask.array as da
import loompy
import numpy as np
import pandas as pd
import scanpy as sc

# Define the file paths
data_dir = '/home/wergillius/Project/diffuse_differentiate/data/Fetal_reference/'

loom_file = f"{data_dir}/Fetal_atlas.loom"
with loompy.connect(loom_file) as ds:
    sliced_count_matrix = ds[:, :] 
    np.save(loom_file.replace(".loom", '.npy'), sliced_count_matrix)
# Define the chunk size (number of cells per chunk)
# Connect to the loom file