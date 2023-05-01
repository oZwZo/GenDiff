
library("Seurat")
install.packages("SingleCellExperiment")
library(SeuratDisk)

setwd("")
load("fig1_17.Rdata")

SaveH5Seurat(sample, filename = "Fig1_17.h5Seurat")
Convert("Fig1_17.h5Seurat", dest = "h5ad")

load('Fig1_20_seurat_object.Rdata')
SaveH5Seurat(sample, filename = "Fig1_20.h5Seurat")
Convert("Fig1_20.h5Seurat", dest = "h5ad")

load('fig3_25_seurat_object.Rdata')
SaveH5Seurat(sample, filename = "Fig3_25.h5Seurat")
Convert("Fig3_25.h5Seurat", dest = "h5ad")
