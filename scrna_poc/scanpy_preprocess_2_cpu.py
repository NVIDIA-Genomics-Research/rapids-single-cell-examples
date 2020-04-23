# Import requirements
import scanpy as sc
from timeit import default_timer as timer
import sys
import os

# Inputs
in_file = sys.argv[1] # .h5ad file containing normalized count matrix, produced by scanpy_preprocess_1.py
out_dir = sys.argv[2] # Output directory
min_disp = float(sys.argv[3]) # Threshold to select highly variable genes. Lower threshold = more genes selected. Recommended 0.5. If 0, all genes used.

# Read normalized count matrix
adata = sc.read(in_file)

if min_disp > 0:

    # Filter matrix to only variable genes
    start = timer()
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=min_disp)
    
    n_genes = sum(adata.var.highly_variable)
    print("Selected " + str(n_genes) + " genes.")
    adata = adata[:, adata.var.highly_variable]
    
    end = timer()
    print("identifying HVG time: " + str(end - start))

# Regress out effects of total counts per cell and the percentage expressed. of mitochondrial genes 
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
out_file = os.path.join(out_dir, "scanpy_scaled_" + str(n_genes) + ".h5ad")
adata.write(out_file)
end = timer()
print("write time: " + str(end - start))