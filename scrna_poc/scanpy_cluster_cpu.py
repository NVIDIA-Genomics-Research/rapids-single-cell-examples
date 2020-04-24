# Import requirements
import scanpy as sc
import argparse
from timeit import default_timer as timer
import os

# Parse args
parser = argparse.ArgumentParser(description='Cluster cells.')
parser.add_argument('--input', type=str, help='Input dfiltered and scaled count matrix in .h5ad format')
parser.add_argument('--out_dir', type=str, help='output directory')
parser.add_argument('--out_prefix', type=str, help='output file prefix')
parser.add_argument('--ncomps', type=int, help='Number of PCA components to calculate', default=50)
parser.add_argument('--ncomps-knn', type=int, help='Number of PCA components to use for KNN', default=50)
parser.add_argument('--neighbors', type=int, help='Number of neighbors for KNN', default=10)
parser.add_argument('--resolution', type=float, help='Number of neighbors for KNN', default=0.5)
parser.add_argument('--spread', type=float, help='Spread for UMAP', default=1.0)
parser.add_argument('--min_dist', type=float, help='Minimum distance for UMAP', default=0.5)
args = parser.parse_args()

# Load data
start = timer()
adata = sc.read(args.input)
end = timer()
print("data loading time: " + str(end - start))

# PCA
start = timer()
sc.tl.pca(adata, n_comps=args.ncomps)
end = timer()
print("PCA time: " + str(end - start))

# KNN graph
start = timer()
sc.pp.neighbors(adata, n_pcs=args.ncomps_knn, n_neighbors=args.neighbors)
end = timer()
print("KNN graph time: " + str(end - start))

# UMAP
start = timer()
sc.tl.umap(adata, min_dist=args.min_dist, spread=args.spread)
end = timer()
print("UMAP time: " + str(end - start))

# Louvain clustering
start = timer()
sc.tl.louvain(adata, resolution=args.resolution)
end = timer()
print("Louvain clustering time: " + str(end - start))

# Save
start = timer()
out_file = os.path.join(args.out_dir, args.out_prefix + "_scanpy_clustered.h5ad")
adata.write(out_file)
end = timer()
print("write time: " + str(end - start))


