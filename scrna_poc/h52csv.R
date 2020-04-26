#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

library(data.table)
library(hdf5r)
library(Seurat)

print("Reading h5")
mat = Read10X_h5(args[1])

print("Converting to matrix")
mat = as.matrix(mat)
print(dim(mat))

print("Converting to data.table")
dt = as.data.table(mat, keep.rownames="gene")
print(dim(dt))

print("Writing to file")
fwrite(dt, file=args[2], quote=F, sep=",", row.names=F)