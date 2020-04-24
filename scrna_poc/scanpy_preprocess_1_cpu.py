# Import requirements
import numpy as np
import scanpy as sc
from timeit import default_timer as timer
import argparse
import os

parser = argparse.ArgumentParser(description='Filter and normalize scRNA-seq count matrix')
parser.add_argument('--csv', type=str, help='Input count matrix in CSV format')
parser.add_argument('--out_dir', type=str, help='output directory')
parser.add_argument('--out_prefix', type=str, help='output file prefix')
parser.add_argument('--min_genes', type=int, help='drop cells with fewer than this number of genes', default=200)
parser.add_argument('--max_genes', type=int, help='drop cells with more than this number of genes', default=6000)
parser.add_argument('--max_mito', type=float, help='drop cells with more than this mitochondrial fraction', default=0.1)
args = parser.parse_args()

# Load data
start = timer()
adata = sc.read_csv(args.csv)
adata = adata.T
end = timer()
print("CSV loading time: " + str(end - start))

# Filtering cells in the matrix

## Filter cells with extreme number of genes
sc.pp.filter_cells(adata, min_genes=args.min_genes)
sc.pp.filter_cells(adata, max_genes=args.max_genes)

## Filter cells with high MT reads
mito_genes = adata.var_names.str.startswith('MT-')
adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
adata.obs['n_counts'] = adata.X.sum(axis=1)
adata = adata[adata.obs.percent_mito < args.max_mito, :]

## Remove zero columns
adata = adata[:,adata.X.sum(axis=0) > 0]

end = timer()
print("cell filtering time: " + str(end - start))

# Normalize data
start = timer()
sc.pp.normalize_total(adata, target_sum=1e4)
end = timer()
print("normalization time: " + str(end - start))

# Logarithmize data
start = timer()
sc.pp.log1p(adata)
end = timer()
print("log transform time: " + str(end - start))

# Save preprocessed count matrix
start = timer()
out_file = os.path.join(args.out_dir, args.out_prefix + "_scanpy_normalized_counts.h5ad")
adata.write(out_file)
end = timer()
print("write time: " + str(end - start))
