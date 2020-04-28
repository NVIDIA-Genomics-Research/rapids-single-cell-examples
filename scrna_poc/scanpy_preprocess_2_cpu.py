# Import requirements
import scanpy as sc
from timeit import default_timer as timer
import argparse
import os

# Parse args

parser = argparse.ArgumentParser(description='Create scaled HVG matrix.')
parser.add_argument('--input', type=str, help='Input normalized count matrix in .h5ad format')
parser.add_argument('--out_dir', type=str, help='output directory')
parser.add_argument('--out_prefix', type=str, help='output file prefix')
parser.add_argument('--min_disp', type=float, help='Select HVGs with greater than this threshold', default=0.5)
args = parser.parse_args()

# Read normalized count matrix
adata = sc.read(args.input)

# Filter matrix to only variable genes
if args.min_disp > 0:
    start = timer()
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=10, min_disp=args.min_disp)
    # Retain the ACE2 gene for eventual visualization
    adata.var.highly_variable['ACE2']=True
    n_genes = sum(adata.var.highly_variable)
    print("Selected " + str(n_genes) + " genes.")
    adata = adata[:, adata.var.highly_variable]
    end = timer()
    print("identifying HVG time: " + str(end - start))

# Regress out effects of total counts per cell and the percentage of mitochondrial genes expressed. 
start = timer()
sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])
end = timer()
print("regression (on filtered data) time: " + str(end - start))

# Scale each gene to unit variance. Clip values exceeding standard deviation 10.
start = timer()
sc.pp.scale(adata, max_value=10)
end = timer()
print("scaling filtered data time: " + str(end - start))

# Save output
start = timer()
out_file = os.path.join(args.out_dir, args.out_prefix + "_scanpy_scaled.h5ad")
adata.write(out_file)
end = timer()
print("write time: " + str(end - start))