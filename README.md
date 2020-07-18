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

We report the runtime of these notebooks on various AWS instances below. All runtimes are given in seconds. Acceleration is given in parentheses. Benchmarking was performed on July 23, 2020 at commit ID `f89e71ae546fe011b9bf222ee5d70ae3fef59d25`.

| Step                         | CPU runtime <br> m5a.12xlarge <br> Intel Xeon Platinum <br> 8000, 48 vCPUs | GPU runtime <br> g4dn.xlarge <br> T4 16 GB GPU  <br> (Acceleration) | GPU runtime <br> p3.2xlarge <br> Tesla V100 16 GB GPU  <br> (Acceleration) |
|------------------------------|-------------------------------------|---------------------------------|--------------|
| Preprocessing                | 311                                 | 74       (4.2x)                 | 84   (3.7x)       |
| PCA                          | 18                                  | 3.5      (5.1x)                 | 3.4  (5.3x)       |
| t-SNE                        | 208                                 | 2.8      (74.3x)                | 2.2  (94.5x)        |
| k-means (single iteration)   | 31                                  | 0.5      (62x)                  | 0.4  (77.5x)        |
| KNN                          | 25                                  | 4.9      (5.1x)                 | 6.1  (4.1x)         |
| UMAP                         | 80                                  | 1.8      (44.4x)                | 1    (80x)          |
| Louvain clustering           | 17                                  | 0.5      (34x)                  | 0.3  (56.7x)        |
| Differential Gene Expression | 54                                  | 11.3     (4.8x)                 | 10.8 (5x)        |
| Re-analysis of subgroup      | 27                                  | 3.5      (7.7x)                 | 3.4  (7.9x)
| End-to-end notebook run<br>(steps above + data load and <br> additional processing)      | 787                              | 122                          | 134          |
| Price ($/hr)                 | 2.064                               | 0.526                           | 3.06             |
| Total cost ($)               | 0.451                               | 0.018                           | 0.114            |               



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

We report the runtime of these notebooks on various AWS instances below. All runtimes are given in seconds. Acceleration is given in parentheses. Benchmarking was performed on July 23, 2020 at commit ID `f89e71ae546fe011b9bf222ee5d70ae3fef59d25`.

| Step                         | CPU runtime <br> m5a.12xlarge <br> Intel Xeon Platinum <br> 8000, 48 vCPUs | GPU runtime <br> g4dn.16xlarge <br> T4 16 GB GPU <br> (Acceleration)  | GPU runtime <br> p3.8xlarge <br> Tesla V100 16 GB GPU <br> (Acceleration) |
|------------------------------|-------------------------------------|----------------------------|-------------------|
| Preprocessing                | 4033                                | 331  (12.2x)               | 323  (12.5x)      |
| PCA                          | 34                                  | 24.6  (1.4x)               | 20.6  (1.7x)      |
| t-SNE                        | 5417                                | 164  (33x)                 | 41  (132.1x)      |
| k-means (single iteration)   | 106                                 | 13.5  (7.9x)               | 2.1  (50.5x)      |
| KNN                          | 585                                 | 110  (5.3x)                | 53.4  (11x)       |
| UMAP                         | 1751                                | 98  (17.9x)                | 20.3  (86.3x)     |
| Louvain clustering           | 597                                 | 5  (119x)                  | 2.5  (238.8x)     |
| Re-analysis of subgroup      | 230                                 | 12.3  (18.7x)              | 10  (23x)
| End-to-end notebook run<br>(steps above + data load and <br> additional processing)      | 13002                              | 938                          | 673          |
| Price ($/hr)                 | 2.064                               | 4.352                      | 12.24             |
| Total cost ($)               | 7.455                               | 1.134                      | 2.287             |   
## Example 3: GPU-based Interactive Visualization of 70,000 cells (beta version)

<img align="left" width="240" height="200" src="https://github.com/avantikalal/rapids-single-cell-examples/blob/visualization/images/dashboard_2.png?raw=true">

We demonstrate how to use RAPIDS, Scanpy and Plotly Dash to create an interactive dashboard where we visualize a single-cell RNA-sequencing dataset. Within the interactive dashboard, we can cluster, visualize, and compare any selected groups of cells.

### Installation

Additional dependencies are needed for this example. Follow these instructions for conda installation:

```bash
conda env create --name rapidgenomics-viz -f conda/rapidgenomics_cuda10.2.viz.yml
conda activate rapidgenomics-viz
python -m ipykernel install --user --display-name "Python (rapidgenomics-viz)"
```

After installing the necessary dependencies, you can just run `jupyter lab`.

### Example Dataset

The dataset used here is the same as in example 1.

### Example Code

Follow this [Jupyter notebook](notebooks/hlca_lung_gpu_analysis-visualization.ipynb) to create the interactive visualization. In order for the notebook to run, the files [rapids_scanpy_funcs.py](notebooks/rapids_scanpy_funcs.py) and [visualize.py](notebooks/visualize.py) need to be in the same folder as the notebook.

## Example 4: Droplet single-cell ATAC-seq of 60K bone marrow cells from Lareau et al. 2019

<img align="left" width="240" height="200" src="https://github.com/rmovva/rapids-single-cell-examples/blob/scatac_gpu/images/60k_bmmc_dsciATAC.png?raw=true">

We demonstrate the use of RAPIDS to accelerate the analysis of single-cell ATAC-seq data from 60,495 cells with 25,000 peaks. We start with the peak-cell matrix from GEO, perform peak selection, normalization, dimensionality reduction, clustering, and visualization. We also visualize regulatory activity at marker genes and compute differential peaks. The notebook runs on a 16GB V100 GPU.

### Example Dataset

The dataset was made publicly available by Lareau et al., on GEO. We processed the dataset to include only cells in the 'Resting' condition and nonzero peaks. Use the following command to download (1) the processed peak-cell count matrix for this dataset (.h5ad), (2) the set of nonzero peak names (.npy), and (3) the cell metadata (.csv), and store them in the `data` folder:

```bash
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/dsci_resting_nonzeropeaks.h5ad; \
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/dsci_resting_peaknames_nonzero.npy; \
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/dsci_resting_cell_metadata.csv
```

### Example Code

Follow this [Jupyter notebook](notebooks/dsci_bmmc_60k_gpu.ipynb) for RAPIDS analysis of this dataset. In order for the notebook to run, the files [rapids_scanpy_funcs.py](notebooks/rapids_scanpy_funcs.py) and [utils.py](notebooks/utils.py) need to be in the same folder as the notebook.

We provide a second notebook with the CPU version of this analysis [here](notebooks/dsci_bmmc_60k_cpu.ipynb).

### Acceleration

## Adapting these examples to another dataset

For our examples, we stored the count matrix in a sparse `.h5ad` format. To convert a different count matrix into this format, follow the instructions in [this notebook](notebooks/csv_to_h5ad.ipynb).

