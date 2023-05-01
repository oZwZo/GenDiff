import sys
import os
import scanpy as sc
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from joblib import dump

main_dir = '/home/wergillius/Project/diffuse_differentiate/'

print("reading integrated data...")
int_adata = sc.read_h5ad(os.path.join(main_dir,"data/TFAtlas/After_IntFetal_hvg.h5ad"))


print("subsectting X ...")

fetal = int_adata[int_adata.obs['batch'] == 'Fetal_atlas']
TF = int_adata[int_adata.obs['batch'] == 'TF_atlas']

X = fetal.X
print(f"shape of X : {X.shape}")

all_labels = fetal.obs['Organ_cell_lineage'].values
y_transform = LabelBinarizer()
Y = y_transform.fit_transform(all_labels)
print(f"shape of Y : {Y.shape}")


print("Fitting KNN classifier ...")

start_time = time.time()

model = KNeighborsClassifier(n_neighbors=20)
model.fit(X, Y)

elaspe = time.time() - start_time
print("runing time", round(elaspe/(1000*60)), "min")

print("save the model")
dump(model, 
os.path.join(main_dir,"data/TFAtlas/KNN_fit_int.joblib")
     ) 

print("transfer labels...")
x_tf = TF.X

ypred_tf = model.predict(x_tf)
labels_pred_tf = y_transform.inverse_transform(ypred_tf)
print("predicting labels")

with open(os.path.join(main_dir,"data/TFAtlas/knn_pred_gene.txt"),"w") as f:
     f.write("CB\tcell type\n")
     for cb, ct in zip(TF.obs.index, labels_pred_tf):
          f.write(f"{cb}\t{ct}\n")

     f.close()