import os
import sys
import pandas as pd
import dask.dataframe as dd
from scipy import io
from scipy.sparse import csr_matrix

gz_csv_path = sys.argv[1]
assert os.path.exists(gz_csv_path) , 'invalid path'
saved_path = gz_csv_path.replace(".csv.gz", ".mtx")

df = dd.read_csv(gz_csv_path, compression='gzip', sample=100000000)
print(df.shape)

sparse_m = csr_matrix(df.values)
io.mmwrite(saved_path, sparse_m)

print("Done! sparse matrix save to")
print(saved_path)
