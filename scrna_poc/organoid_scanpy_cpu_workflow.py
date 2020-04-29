# Import requirements
import numpy as np
import scanpy as sc
from timeit import default_timer as timer
import argparse
import os

# Load data
start = timer()
adata = sc.read_csv(filename='/covid-omics/scrna_demo/data/organoid/original_matrix/GSM4447249_KidneyOrganoid_Filtered.csv')
adata = adata.T
end = timer()
print("Count matrix loading time: " + str(end - start))
print(adata)

# Filtering cells in the matrix

start = timer()

## Filter cells with extreme number of genes
sc.pp.filter_cells(adata, min_genes=200)
print(adata)

sc.pp.filter_cells(adata, max_genes=8000)
print(adata)

## Filter cells with high MT reads
mito_genes = adata.var_names.str.startswith('MT-')
adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
adata.obs['n_counts'] = adata.X.sum(axis=1)
adata = adata[adata.obs.percent_mito < 0.5, :]
print(adata)

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
out_file = os.path.join("organoid_scanpy_normalized_counts.h5ad")
adata.write(out_file)
end = timer()
print("write normalized matrix time: " + str(end - start))

# Filter matrix to only variable genes
start = timer()
sc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="cell_ranger")
# Retain the ACE2 and TMPRSS2 genes for eventual visualization
adata.var.highly_variable['ACE2']=True
adata.var.highly_variable['TMPRSS2']=True
adata = adata[:, adata.var.highly_variable]
end = timer()
print("filtering HVG time: " + str(end - start))

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
out_file = os.path.join("organoid_scanpy_scaled.h5ad")
adata.write(out_file)
end = timer()
print("write scaled and filtered matrix time: " + str(end - start))

# PCA
start = timer()
sc.tl.pca(adata, n_comps=50)
end = timer()
print("PCA time: " + str(end - start))

# t-SNE
start = timer()
sc.tl.tsne(adata, n_pcs=20)
end = timer()
print("t-SNE time: " + str(end - start))

# KNN graph
start = timer()
sc.pp.neighbors(adata, n_pcs=50, n_neighbors=15)
end = timer()
print("KNN graph time: " + str(end - start))

# UMAP
start = timer()
sc.tl.umap(adata, min_dist=0.3, spread=0.6)
end = timer()
print("UMAP time: " + str(end - start))

# K-means
start = timer()
kmeans = KMeans(n_clusters=13, random_state=0).fit(adata.obsm['X_umap'])
adata.obs['kmeans'] = kmeans.labels_.astype(str)
end = timer()
print("k-means time: " + str(end - start))

# Louvain clustering
start = timer()
sc.tl.louvain(adata, resolution=0.5)
end = timer()
print("Louvain clustering time: " + str(end - start))
print(adata.obs.louvain.value_counts())

# Save
start = timer()
adata.write("organoid_scanpy_clustered.h5ad")
end = timer()
print("write clustered annData time: " + str(end - start))

# Plot
sc.pl.umap(adata, save='_organoid_kmeans.png', color=["kmeans"])
sc.pl.umap(adata, save='_organoid_louvain.png', color=["louvain"])
sc.pl.umap(adata, save='_organoid_ace2.png', color=["ACE2"], color_map="jet")
sc.pl.tsne(adata, save='_organoid_kmeans.png', color=["kmeans"])
sc.pl.tsne(adata, save='_organoid_louvain.png', color=["louvain"])
sc.pl.tsne(adata, save='_organoid_ace2.png', color=["ACE2"], color_map="jet")


