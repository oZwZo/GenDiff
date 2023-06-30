# %%
import gzip
import csv
import scipy.sparse as sp
import numpy as np
import scanpy as sc
import anndata as ad
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--input", type=str, required=True, help='Directory containing the original data')
parser.add_argument("--ouput", type=str, required=True, help='Directory containing the processed data')
parser.add_argument("--chunksize", type=int, default=2000, help='Set sampling rate default 2000')
args = parser.parse_args()

matrix=[]
genes = []
sparse_matrix=np.zeros((1,1))
chunk_size = args.chunksize
cells = []

# Open the gzipped CSV file and read it line by line using csv.reader
with gzip.open(args.input, 'rt') as f:
    reader = csv.reader(f,delimiter="\t")
    # Initialize an empty list to hold your data
    counts=0
    for row in reader:
        if counts ==0:
            cells = row[1:]
            counts=counts+1
            continue
        # if counts==123:
        #     break
        if counts%chunk_size==0:
            # # Create a sparse matrix from your data
            sparse_matrix_temp = sp.csr_matrix(np.array(matrix).astype(int))
            if(sparse_matrix.shape[0]<=1):
                sparse_matrix = sparse_matrix_temp
            else:
                sparse_matrix = sp.vstack([sparse_matrix, sparse_matrix_temp])
            # # Clear the chache for the next chunk
            del matrix
            del sparse_matrix_temp
            matrix=[]
            print(counts,"rows processed")
        matrix.append(row[1:])
        genes.append(row[0])
        counts=counts+1
    
    # # Process the last chunk (which is not divisible by 2000)
    sparse_matrix_temp = sp.csr_matrix(np.array(matrix).astype(int))
    if(sparse_matrix.shape[0]<=1):
        sparse_matrix = sparse_matrix_temp
    else:
        sparse_matrix = sp.vstack([sparse_matrix, sparse_matrix_temp])

adata = ad.AnnData(X=sparse_matrix.T)
adata.var_names = genes
adata.obs_names = cells
adata.write_h5ad(args)


