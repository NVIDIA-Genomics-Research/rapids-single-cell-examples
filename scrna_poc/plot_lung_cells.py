import scanpy as sc
import argparse

# Parse args
parser = argparse.ArgumentParser(description='Plot cells.')
parser.add_argument('--input', type=str, help='Input clustered count matrix in .h5ad format')
parser.add_argument('--out_prefix', type=str, help='output file prefix')
args = parser.parse_args()

# Load data
adata = sc.read(args.input)

# Plot
plot_file = "_" + args.out_prefix + "_louvain.png"
sc.pl.umap(adata, save=plot_file, color=["louvain"])
plot_file = "_" + args.out_prefix + "_markers.png"
sc.pl.umap(adata, save=plot_file, color=["ACE2", "AGER", "SFTPC", "SCGB3A2", "TPPP3", "CD68", "PTPRC"])