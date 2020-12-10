# GPU-Accelerated Single-Cell Genomics Analysis with RAPIDS

This repository contains example notebooks demonstrating how to use [RAPIDS](https://www.rapids.ai) for GPU-accelerated analysis of single-cell sequencing data.

## Installation 

### Launch Script
Lanuch script (./launch) can be used to start example notebooks either on a host or in a docker container. This script prepares the environment and acquires the dataset for the examples.

```bash
# rapids-single-cell-examples$ ./launch -h
usage: launch <command> [<args>]

Following commands are wrapped by this tool:
   container  : Start Jupyter notebook in a container
   host       : Start Jupyter notebook on the host
   dataset    : Download dataset
   execute    : Execute an example
   create_env : Create conda environment for an example

To execute 'hlca_lung' example in container, please execute following command:
./launch container -d /path/to/store/dataset -e hlca_lung

Example launcher

positional arguments:
  command     Subcommand to run

optional arguments:
  -h, --help  show this help message and exit
```

```./launch host``` can be used to create a conda env. for executing any of the sample. To list all supported example, please execute ```./launch host -h```.

```./launch container``` can be used to setup a container for the example.

```./launch execute```, can be used to run an example in the background. Results are saved inplace.

### conda
All dependencies for these examples can be installed with conda. CUDA versions 10.1 and 10.2 are supported currently. 

```bash
conda env create --name rapidgenomics -f conda/rapidgenomics_cuda10.2.yml
conda activate rapidgenomics
python -m ipykernel install --user --display-name "Python (rapidgenomics)"
```
If installing for a system running a CUDA 10.1 driver, use `conda/rapidgenomics_cuda10.1.yml`.

After installing the necessary dependencies, you can just run `jupyter lab`.

### Docker container
A container with all dependencies, notebooks and source code is available at https://hub.docker.com/r/claraparabricks/single-cell-examples_rapids_cuda10.2.

Please execute the following commands to start the notebook and follow the URL in the log to open Jupyter web application.

```bash
docker pull claraparabricks/single-cell-examples_rapids_cuda10.2

docker run --gpus all --rm -v /mnt/data:/data claraparabricks/single-cell-examples_rapids_cuda10.2

```

## Configuration


Unified Virtual Memory (UVM) can be used to [oversubscribe](https://developer.nvidia.com/blog/beyond-gpu-memory-limits-unified-memory-pascal/) your GPU memory so that chunks of data will be automatically offloaded to main memory when necessary. This is a great way to explore data without having to worry about out of memory errors, but it does degrade performance in proportion to the amount of oversubscription. UVM is enabled by default in these examples and can be enabled/disabled in any RAPIDS workflow with the following:
```python
import cupy as cp
import rmm
rmm.reinitialize(managed_memory=True)
cp.cuda.set_allocator(rmm.rmm_cupy_allocator)
```

RAPIDS provides a [GPU Dashboard](https://medium.com/rapids-ai/gpu-dashboards-in-jupyter-lab-757b17aae1d5), which contains useful tools to monitor GPU hardware right in Jupyter. 

## Example 1: Single-cell RNA-seq of 70,000 Human Lung Cells

<img align="left" width="240" height="200" src="https://github.com/clara-parabricks/rapids-single-cell-examples/blob/master/images/70k_lung.png?raw=true">

We use RAPIDS to accelerate the analysis of a ~70,000-cell single-cell RNA sequencing dataset from human lung cells. This example includes preprocessing, dimension reduction, clustering, visualization and gene ranking. 

### Example Dataset

The dataset is from [Travaglini et al. 2020](https://www.biorxiv.org/content/10.1101/742320v2). If you wish to run the example notebook using the same data, use the following command to download the count matrix for this dataset and store it in the `data` folder:

```bash
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/krasnow_hlca_10x.sparse.h5ad
```

### Example Code

Follow this [Jupyter notebook](notebooks/hlca_lung_gpu_analysis.ipynb) for RAPIDS analysis of this dataset. In order for the notebook to run, the file [rapids_scanpy_funcs.py](notebooks/rapids_scanpy_funcs.py) needs to be in the same folder as the notebook.

We provide a second notebook with the CPU version of this analysis [here](notebooks/hlca_lung_cpu_analysis.ipynb).

### Acceleration

We report the runtime of these notebooks on various AWS instances below. All runtimes are given in seconds. Acceleration is given in parentheses. Benchmarking was performed at commit ID `6747214a3dff2bdc016a6df2b997cc8db7173d54`.

| Step                         | AWS <br> CPU runtime <br> m5a.12xlarge <br> Intel Xeon Platinum <br> 8000, 48 vCPUs | AWS <br> GPU runtime <br> g4dn.12xlarge <br> T4 16 GB GPU <br> (Acceleration)  | AWS <br> GPU runtime <br> p3.2xlarge <br> Tesla V100 16 GB GPU <br> (Acceleration) | GCP <br> GPU runtime <br> a2-highgpu-1g (80GB) <br> Tesla A100 40GB GPU <br> (Acceleration) |
|------------------------------|-------------------------------------|---------------------------------|----------------|--------|
| Preprocessing                | 329                                 | 66       (5x)                   | 84   (3.9x)    | 91   (3.6x) |
| PCA                          | 12.2                                | 4.6      (2.7x)                 | 3.1  (3.9x)    | 2.68 (4.6x) |
| t-SNE                        | 236                                 | 3.0      (79x)                  | 1.8  (131x)    | 2.23 (105x) |
| k-means (single iteration)   | 27                                  | 0.3      (90x)                  | 0.12 (225x)    | .089 (337x) |
| KNN                          | 28                                  | 4.9      (5.7x)                 | 5.9  (4.7x)    | 5.34 (5.2x) |
| UMAP                         | 55                                  | 0.95     (58x)                  | 0.55 (100x)    | .627 (87.7x)|
| Louvain clustering           | 16                                  | 0.19     (84x)                  | 0.17 (94x)     | .145 (110x) |
| Leiden clustering            | 17                                  | 0.14     (121x)                 | 0.15 (113x)    | .123 (138x) |
| Differential Gene Expression | 99                                  | 2.9      (34x)                  | 2.4  (41x)     | 1.96 (50.5x)|
| Re-analysis of subgroup      | 21                                  | 3.7      (5.7x)                 | 3.3  (6.4x)    | 4.1  (5.12x) |
| End-to-end notebook run<br>(steps above + data load and <br> additional processing)  | 858  | 103    | 122            | 125  (6.86x) |
| Price ($/hr)                 | 2.064                               | 0.526                           | 3.06           | 4.00  |
| Total cost ($)               | 0.492                               | 0.015                           | 0.104          | 0.139           |



## Example 2: Single-cell RNA-seq of 1 Million Mouse Brain Cells

<img align="left" width="240" height="200" src="https://github.com/clara-parabricks/rapids-single-cell-examples/blob/master/images/1M_brain.png?raw=true">

We demonstrate the use of RAPIDS to accelerate the analysis of single-cell RNA-seq data from 1 million cells. This example includes preprocessing, dimension reduction, clustering and visualization.

This example relies heavily on UVM and a few of the operations oversubscribed a 32GB V100 GPU on a DGX1. While this example should work on any GPU built on the Pascal architecture or newer, you will want to make sure there is enough main memory available. Oversubscribing a GPU by more than a factor of 2x can cause thrashing in UVM, which can ultimately lead to the notebook freezing.

### Example Dataset

The dataset was made publicly available by 10X Genomics. Use the following command to download the count matrix for this dataset and store it in the `data` folder:

```bash
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/1M_brain_cells_10X.sparse.h5ad
```

### Example Code

Follow this [Jupyter notebook](notebooks/1M_brain_gpu_analysis_uvm.ipynb) for RAPIDS analysis of this dataset. In order for the notebook to run, the files [rapids_scanpy_funcs.py](notebooks/rapids_scanpy_funcs.py) and [utils.py](notebooks/utils.py) need to be in the same folder as the notebook.

We provide a second notebook with the CPU version of this analysis [here](notebooks/1M_brain_cpu_analysis.ipynb).

### Acceleration

We report the runtime of these notebooks on various AWS & GCP instances below. All runtimes are given in seconds. Acceleration is given in parentheses. Benchmarking was performed at commit ID `6747214a3dff2bdc016a6df2b997cc8db7173d54`.

| Step                         | AWS <br> CPU runtime <br> m5a.12xlarge <br> Intel Xeon Platinum <br> 8000, 48 vCPUs | AWS <br> GPU runtime <br> g4dn.12xlarge <br> T4 16 GB GPU <br> (Acceleration)  | AWS <br> GPU runtime <br> p3.8xlarge <br> Tesla V100 16 GB GPU <br> (Acceleration) | GCP <br> GPU runtime <br> a2-highgpu-1g (80GB) <br> Tesla A100 40GB GPU <br> (Acceleration) |
|------------------------------|-------------------------------------|----------------------------|-------------------|---------|
| Preprocessing                | 4337                                | 344  (13x)                 | 336  (13x)        | 201  (21.6x) |
| PCA                          | 29                                  | 28   (1.04x)               | 23   (1.3x)       | 11.4 (2.5x) |
| t-SNE                        | 5833                                | 134  (44x)                 | 38   (154x)       | 27.6 (211x)  |
| k-means (single iteration)   | 113                                 | 13.2 (8.6x)                | 2.4  (47x)        | 1.88 (60x)  |
| KNN                          | 670                                 | 106  (6.3x)                | 55.1 (12x)        | 46.3 (14.5x)  |
| UMAP                         | 1405                                | 87   (16x)                 | 19.2 (73x)        | 13.4 (105x) |
| Louvain clustering           | 573                                 | 5.2  (110x)                | 2.8  (205x)       | 1.92 (298x) |
| Leiden clustering            | 6414                                | 3.7  (1733x)               | 1.8  (3563x)      | 1.35 (4751x)  |
| Re-analysis of subgroup      | 249                                 | 10.9 (23x)                 | 8.9  (28x)        | 9.3  (26.8x) |
| End-to-end notebook run<br>(steps above + data load and <br> additional processing)      | 19908    | 912  | 702    | 502  (39.7x) |
| Price ($/hr)                 | 2.064                               | 3.912                      | 12.24             | 4.00 |
| Total cost ($)               | 11.414                              | 0.991                      | 2.388<sup>*</sup> | 0.553 |

<sup>* The amount of main memory needed to execute this notebook required an instance with 4x GPUs</sup>


## Example 3: GPU-based Interactive Visualization of 70,000 Human Lung Cells (beta version)

![Interactive browser Demo](images/viz3-2.gif)

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

## Example 4: Droplet Single-cell ATAC-seq of 60K Bone Marrow Cells

<img align="left" width="240" height="200" src="https://github.com/clara-parabricks/rapids-single-cell-examples/blob/master/images/60k_bmmc_dsciATAC.png?raw=true">

We demonstrate the use of RAPIDS to accelerate the analysis of single-cell ATAC-seq data from 60,495 cells. We start with the peak-cell matrix, then perform peak selection, normalization, dimensionality reduction, clustering, and visualization. We also visualize regulatory activity at marker genes and compute differential peaks.

### Example Dataset

The dataset is taken from [Lareau et al., Nat Biotech 2019](https://www.nature.com/articles/s41587-019-0147-6). We processed the dataset to include only cells in the 'Resting' condition and peaks with nonzero coverage. Use the following command to download (1) the processed peak-cell count matrix for this dataset (.h5ad), (2) the set of nonzero peak names (.npy), and (3) the cell metadata (.csv), and store them in the `data` folder:

```bash
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/dsci_resting_nonzeropeaks.h5ad; \
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/dsci_resting_peaknames_nonzero.npy; \
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/dsci_resting_cell_metadata.csv
```

### Example Code

Follow this [Jupyter notebook](notebooks/dsci_bmmc_60k_gpu.ipynb) for RAPIDS analysis of this dataset. In order for the notebook to run, the files [rapids_scanpy_funcs.py](notebooks/rapids_scanpy_funcs.py) and [utils.py](notebooks/utils.py) need to be in the same folder as the notebook.

We provide a second notebook with the CPU version of this analysis [here](notebooks/dsci_bmmc_60k_cpu.ipynb).

### Acceleration

We report the runtime of these notebooks on various AWS instances below. All runtimes are given in seconds. Acceleration is given in parentheses. Benchmarking was performed at commit ID `6747214a3dff2bdc016a6df2b997cc8db7173d54`.


| Step                         | AWS <br> CPU runtime <br> m5a.12xlarge <br> Intel Xeon Platinum <br> 8000, 48 vCPUs | AWS <br> GPU runtime <br> g4dn.12xlarge <br> T4 16 GB GPU <br> (Acceleration)  | AWS <br> GPU runtime <br> p3.2xlarge <br> Tesla V100 16 GB GPU <br> (Acceleration) | GCP <br> GPU runtime <br> a2-highgpu-1g (80GB) <br> Tesla A100 40GB GPU <br> (Acceleration) |
|------------------------------|-------------------------------------|----------------------------|-------------------|-------|
| PCA                          | 149                                 | 136  (1.1x)                | 64   (2.3x)       | 52.4 (2.8x) |
| KNN                          | 39                                  | 3.8  (10x)                 | 4.9  (8x)         | 4.6  (8.5x) |
| UMAP                         | 38                                  | 1.1  (35x)                 | 0.78 (49x)        | 0.68 (56x) |
| Louvain clustering           | 6.8                                 | 0.13 (52x)                 | 0.12 (57x)        | 0.10 (68x) |
| Leiden clustering            | 19                                  | 0.08 (238x)                | 0.07 (271x)       | 0.08 (238x) |
| t-SNE                        | 252                                 | 3.3  (76x)                 | 2.1  (120x)       | 2.3  (110x) |
| Differential Peak Analysis   | 1006                                | 23   (44x)                 | 20   (50x)        | 9.8  (103x) |
| End-to-end notebook run<br>(steps above + data load and <br> pre-processing)      | 1530        | 182    | 111  | 86.9 |
| Price ($/hr)                 | 2.064                               | 3.912                      | 3.06              | 4.00 | 
| Total cost ($)               | 0.877                               | 0.198                      | 0.095             | 0.096 |

## Example 5: Visualizing Chromatin Accessibility in 5,000 PBMCs with RAPIDS and AtacWorks (Beta version)

<img align="left" width="240" height="200" src="https://github.com/avantikalal/rapids-single-cell-examples/blob/rilango/mem-fix/images/atacworks_notebook_img.png?raw=true">

We analyze single-cell ATAC-seq data from 5000 PBMC cells as in example 4. Additionally, we use cuDF to calculate and visualize cluster-specific chromatin accessibility in selected marker regions. Finally, we use a deep learning model trained with [AtacWorks](https://github.com/clara-parabricks/AtacWorks), to improve the accuracy of the chromatin accessibility track and call peaks in individual clusters.

### Example Data

The dataset was made publicly available by 10X Genomics. Use the following command to download the peak x cell count matrix and the fragment file for this dataset, and store both in the `data` folder:

```bash
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/5k_pbmcs_10X.sparse.h5ad
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/5k_pbmcs_10X_fragments.tsv.gz
wget -P <path to this repository>/data https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/5k_pbmcs_10X_fragments.tsv.gz.tbi
```

### Example Model

We use a pre-trained deep learning model to denoise the chromatin accessibility track and call peaks. This model can be downloaded into the `models` folder:
```bash
wget -P <path to this repository>/models https://api.ngc.nvidia.com/v2/models/nvidia/atac_bulk_lowcov_5m_50m/versions/0.3/files/models/model.pth.tar
```

### Example Code
Follow this [Jupyter notebook](notebooks/5k_pbmc_coverage_gpu.ipynb) for GPU analysis of this dataset. In order for the notebook to run, the files [utils.py](notebooks/utils.py), and [coverage.py](notebooks/coverage.py) need to be in the same folder as the notebook.



## Adapting these examples to another dataset

For our examples, we stored the count matrix in a sparse `.h5ad` format. To convert a different count matrix into this format, follow the instructions in [this notebook](notebooks/csv_to_h5ad.ipynb).

