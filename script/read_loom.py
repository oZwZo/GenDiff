import dask.array as da
import loompy
import numpy as np
import pandas as pd
import scanpy as sc

# Define the file paths
main_dir = '/home/wergillius/Project/diffuse_differentiate'
loom_file = f"{main_dir}/data/Fetal_reference/GSE156793_S3_gene_count.loom"
output_file = f"{main_dir}/data/Fetal_reference/GSE156793_loom_parallel.h5ad"

# Define the chunk size (number of cells per chunk)
# Connect to the loom file
fetal_ref = sc.read_loom(f"{main_dir}/data/Fetal_reference/GSE156793_S3_gene_count.loom")
# Save the AnnData object to a h5ad file
fetal_ref.write_h5ad(output_file)