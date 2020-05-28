# GPU-Accelerated Single-Cell Genomics Analysis with RAPIDS

This repository contains example notebooks demonstrating how to use [RAPIDS](https://www.rapids.ai) for GPU-accelerated analysis of single-cell sequencing data.

## Installation 

All dependencies for this example can be installed with conda. CUDA versions 10.1 and 10.2 are supported currently. If installing for a system running a CUDA10.1 driver, use `conda/rapidgenomics_cuda10.1.yml`

```bash
conda env create --name rapidgenomics -f conda/rapidgenomics_cuda10.2.yml
conda activate rapidgenomics
python -m ipykernel install --user --display-name "Python (rapidgenomics)"
```

After installing the necessary dependencies, you can just run `jupyter lab`.

## Human Lung Cell Atlas Example

We present an example using RAPIDS to accelerate the analysis of a ~70,000-cell single-cell RNA sequencing dataset from human lung cells. This example includes preprocessing, dimension reduction, clustering visualization and gene ranking. 

### Example Dataset

The dataset is based on [Travaglini et al. 2020](https://www.biorxiv.org/content/10.1101/742320v2). If you wish to run the notebook using the same data, use the following command to download the count matrix for this dataset and store it in the `data` folder:

```bash
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/krasnow_hlca_10x_UMIs.sparse.h5ad
```

### Example Code

Follow this [Jupyter notebook](notebooks/hlca_lung_gpu_analysis.ipynb) for RAPIDS analysis of this dataset.

We provide a second notebook with the CPU version of this analysis [here](notebooks/hlca_lung_cpu_analysis.ipynb).

### Adapting to another dataset

For our examples, we stored the count matrix in a sparse `.h5ad` format. To convert a different count matrix into this format, follow the instructions in [this notebook](notebooks/csv_to_h5ad.ipynb).

### Acceleration

All runtimes are given in seconds.

| Step                         | CPU runtime (16 core AMD EPYC 7571) | GPU runtime (Tesla V100 32 GB) | Acceleration |
|------------------------------|-------------------------------------|--------------------------------|--------------|
| Preprocessing                | 324.35                              | 68.49                          | 4.7x         |
| PCA                          | 16.2                                | 1.59                           | 10.2x        |
| t-SNE                        | 166                                 | 1.95                           | 85.1x        |
| k-means (single iteration)   | 7.3                                 | 0.11                           | 66.4x        |
| KNN                          | 23                                  | 5.18                           | 4.4x         |
| UMAP                         | 78                                  | 0.98                           | 80x          |
| Louvain clustering           | 13.6                                | 0.25                           | 54.4x        |
| Differential Gene Expression | 45.1                                | 18.9                           | 2.4x         |
