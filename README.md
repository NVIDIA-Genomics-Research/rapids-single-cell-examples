# GPU-Accelerated Single-Cell Genomics Analysis with RAPIDS

This repository contains example notebooks demonstrating how to use [RAPIDS](https://www.rapids.ai) for GPU-accelerated analysis of single-cell sequencing data.

## Installation 

All dependencies for these examples can be installed with conda. CUDA versions 10.1 and 10.2 are supported currently. If installing for a system running a CUDA10.1 driver, use `conda/rapidgenomics_cuda10.1.yml`

```bash
conda env create --name rapidgenomics -f conda/rapidgenomics_cuda10.2.yml
conda activate rapidgenomics
python -m ipykernel install --user --display-name "Python (rapidgenomics)"
```

After installing the necessary dependencies, you can just run `jupyter lab`.

## Configuration


Unified Virtual Memory (UVM) can be used to [oversubscribe](https://developer.nvidia.com/blog/beyond-gpu-memory-limits-unified-memory-pascal/) your GPU memory so that chunks of data will be automatically offloaded to main memory when necessary. This is a great way to explore data without having to worry about out of memory errors, but it does degrade performance in proportion to the amount of oversubscription. UVM is enabled by default in these examples and can be enabled/disabled in any RAPIDS workflow with the following:
```python
import cupy as cp
import rmm
rmm.reinitialize(managed_memory=True)
cp.cuda.set_allocator(rmm.rmm_cupy_allocator)
```

RAPIDS provides a [GPU Dashboard](https://medium.com/rapids-ai/gpu-dashboards-in-jupyter-lab-757b17aae1d5), which contains useful tools to monitor GPU hardware right in Jupyter. 

## Example 1: Single-cell RNA-seq of 70,000 cells from the Human Lung Cell Atlas

<img align="left" width="240" height="200" src="https://github.com/avantikalal/rapids-single-cell-examples/blob/alal/1mil/images/70k_lung.png?raw=true">

We use RAPIDS to accelerate the analysis of a ~70,000-cell single-cell RNA sequencing dataset from human lung cells. This example includes preprocessing, dimension reduction, clustering, visualization and gene ranking. 

### Example Dataset

The dataset is from [Travaglini et al. 2020](https://www.biorxiv.org/content/10.1101/742320v2). If you wish to run the example notebook using the same data, use the following command to download the count matrix for this dataset and store it in the `data` folder:

```bash
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/krasnow_hlca_10x_UMIs.sparse.h5ad
```

### Example Code

Follow this [Jupyter notebook](notebooks/hlca_lung_gpu_analysis.ipynb) for RAPIDS analysis of this dataset. In order for the notebook to run, the file [rapids_scanpy_funcs.py](notebooks/rapids_scanpy_funcs.py) needs to be in the same folder as the notebook.

We provide a second notebook with the CPU version of this analysis [here](notebooks/hlca_lung_cpu_analysis.ipynb).

### Acceleration

All runtimes are given in seconds.
Benchmarking was performed on May 28, 2020 (commit ID `1f84796fbc255baf2f997920421bd300e0c30fc0`)

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

## Example 2: Single-cell RNA-seq of 1 Million Mouse Brain Cells from 10X Genomics

<img align="left" width="240" height="200" src="https://github.com/avantikalal/rapids-single-cell-examples/blob/alal/1mil/images/1M_brain.png?raw=true">

We demonstrate the use of RAPIDS to accelerate the analysis of single-cell RNA-seq data from 1 million cells. This example includes preprocessing, dimension reduction, clustering and visualization.

This example relies heavily on UVM and a few of the operations oversubscribed a 32GB V100 GPU on a DGX1. While this example should work on any GPU built on the Pascal architecture or newer, you will want to make sure there is enough main memory available.

### Example Dataset

The dataset was made publicly available by 10X Genomics. Use the following command to download the count matrix for this dataset and store it in the `data` folder:

```bash
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/1M_brain_cells_10X.sparse.h5ad
```

### Example Code

Follow this [Jupyter notebook](notebooks/1M_brain_gpu_analysis_uvm.ipynb) for RAPIDS analysis of this dataset. In order for the notebook to run, the files [rapids_scanpy_funcs.py](notebooks/rapids_scanpy_funcs.py) and [utils.py](notebooks/utils.py) need to be in the same folder as the notebook. This notebook runs completely in under 15 minutes on a Tesla V100 GPU with 32 GB memory.

We provide a second notebook with the CPU version of this analysis [here](notebooks/1M_brain_cpu_analysis.ipynb).

### Acceleration

All runtimes are given in seconds.

| Step                         | CPU runtime <br>(2x20 core Intel Xeon E5-2698 v4) | GPU runtime (Tesla V100 32 GB) | Acceleration |
|------------------------------|-------------------------------------|--------------------------------|--------------|
| Preprocessing                | 5446                                | 244.7                          | 22.3x        |
| PCA                          | 38.2                                | 27.9                           | 1.4x         |
| t-SNE                        | 4303                                | 42.4                           | 101.5x       |
| k-means (single iteration)   | 84                                  | 1.37                           | 61.3x        |
| KNN                          | 733                                 | 45.1                           | 16.3x        |
| UMAP                         | 1537                                | 21.1                           | 72.8x        |
| Louvain clustering           | 650                                 | 2.5                            | 269x         |


## Adapting these examples to another dataset

For our examples, we stored the count matrix in a sparse `.h5ad` format. To convert a different count matrix into this format, follow the instructions in [this notebook](notebooks/csv_to_h5ad.ipynb).
