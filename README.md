# GPU-Accelerated Single-Cell Genomics Analysis with RAPIDS

This repository contains example notebooks demonstrating how to use [RAPIDS](https://rapids.ai) for GPU-accelerated analysis of single-cell sequencing data.

RAPIDS is a suite of open-source Python libraries that can speed up data science workflows using GPU acceleration.Starting from a single-cell count matrix, RAPIDS libraries can be used to perform data processing, dimensionality reduction, clustering, visualization, and comparison of cell clusters.

Several of our examples are inspired by the [Scanpy tutorials](https://scanpy.readthedocs.io/en/stable/tutorials.html) and based upon the [AnnData](https://anndata.readthedocs.io/en/latest/index.html) format. Currently, we provide examples for scRNA-seq and scATAC-seq, and we have scaled up to 1 million cells. We also show how to create GPU-powered interactive, in-browser visualizations to explore single-cell datasets.

Dataset sizes for single-cell genomics studies are increasing, presently reaching millions of cells. With RAPIDS, it becomes easy to analyze large datasets interactively and in real time, enabling faster scientific discoveries.

## Installation

### Docker container
A container with all dependencies, notebooks and source code is available at https://hub.docker.com/r/claraparabricks/single-cell-examples_rapids_cuda11.0.

Please execute the following commands to start the notebook and follow the URL in the log to open Jupyter web application.

```bash
docker pull claraparabricks/single-cell-examples_rapids_cuda11.0

docker run --gpus all --rm -v /mnt/data:/data claraparabricks/single-cell-examples_rapids_cuda11.0
```

### conda
All dependencies for these examples can be installed with conda.

```bash
conda env create --name rapidgenomics -f conda/rapidgenomics_cuda11.0.yml
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


## Citation

If you use this code, please cite: <a href="https://zenodo.org/badge/latestdoi/265649968"><img src="https://zenodo.org/badge/265649968.svg" alt="DOI"></a>



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

We report the runtime of these notebooks on various GCP instances below. All runtimes are given in seconds. Acceleration is given in parentheses. Benchmarking was performed on Dec 16, 2020.

| Step                         | CPU <br> n1-standard-16 <br> 16 vCPUs | GPU <br> n1-standard-16 <br> T4 16 GB GPU <br> (Acceleration)  | GPU <br> n1-highmem-8 <br> Tesla V100 16 GB GPU <br> (Acceleration) | GPU <br> a2-highgpu-1g <br> Tesla A100 40GB GPU <br> (Acceleration) |
|------------------------------|---------------------------|---------------------------|---------------|--------------|
| Preprocessing                |        58                |        (x)             |  86  (x)     |  59  (x)    |
| PCA                          |        7.6               |       (x)              | 3.3  (x)     |  2.7 (x)    |
| t-SNE                        |        206               |       (x)              | 1.4  (x)     |  2.2 (x)    |
| k-means (single iteration)   |         13               |      (x)               | 0.12 (x)     |  0.08 (x)   |
| KNN                          |         20               |      (x)               | 21 (x)       |  5.7 (x)    |
| UMAP                         |         90               |      (x)               | 0.57 (x)     |  0.53 (x)   |
| Louvain clustering           |         14.4             |      (x)               | 0.16 (x)     |  0.11 (x)   |
| Leiden clustering            |         12.5             |      (x)               | 0.10 (x)     |  0.08 (x)   |
| Differential Gene Expression |        158               |       (x)              | 7.6  (x)     |  6.3 (x)    |
| Re-analysis of subgroup      |         30               |       (x)              | 3.8  (x)     |  3.5 (x)    |
| End-to-end notebook run      |        626               |                        |  141         |  96         |
| Price ($/hr)                 |       0.760              | 1.110                  | 2.953        | 3.673       |
| Total cost ($)               |       0.132              |                        |              |             |



## Example 2: Single-cell RNA-seq of 1.3 Million Mouse Brain Cells

<img align="left" width="240" height="200" src="https://github.com/clara-parabricks/rapids-single-cell-examples/blob/master/images/1M_brain.png?raw=true">

We demonstrate the use of RAPIDS to accelerate the analysis of single-cell RNA-seq data from 1.3 million cells. This example includes preprocessing, dimension reduction, clustering and visualization.

Compared to the previous example, here we make several adjustments to handle the larger dataset. We perform most of the preprocessing operations (e.g. filtering, normalization) while reading the dataset in batches. Further, we perform a batched PCA by training on a fraction of cells and transforming the data in batches.

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

We report the runtime of these notebooks on various GCP instances below. All runtimes are given in seconds. Acceleration is given in parentheses. Benchmarking was performed on Dec 16, 2020.

| Step                         | CPU <br> n1-highmem-64 <br> 64 vCPUs | GPU <br> n1-highmem-16 <br> T4 16 GB GPU <br> (Acceleration)  | GPU <br> n1-highmem-16 <br> Tesla V100 16 GB GPU <br> (Acceleration) | GPU <br> a2-highgpu-1g <br> Tesla A100 40GB GPU <br> (Acceleration) |
|------------------------------|---------------------------|----------------------------|-------------------|--------------|
| Data load + Preprocessing    |    1120                   |   (x)                 |   (x)         |  475 (2.4x)    |
| PCA                          |      44                   |    (x)                |    (x)        |   17.8 (2.5x)  |
| t-SNE                        |    6509                   |   (x)                 |    (x)        |   37  (176x)  |
| k-means (single iteration)   |     148                   |  (x)                  |   (x)         |    2 (74x)   |
| KNN                          |     154                   |   (x)                 |    (x)        |   62 (2.5x)    |
| UMAP                         |    2571                   |    (x)                |  (x)          |   21 (122x)   |
| Louvain clustering           |    1153                   |   (x)                 |   (x)         |    2.4 (480x)  |
| Leiden clustering            |    6345                   |   (x)                 |   (x)         |    1.7 (3732x) |
| Re-analysis of subgroup      |     255                   |  (x)                  |    (x)        |   17.9 (14.2x)   |
| End-to-end notebook run      |   18338                   |                       |               |   686       |
| Price ($/hr)                 |   3.786                   | 1.296                 | 5.906         | 3.673         |
| Total cost ($)               |    19.285                 |                       |               |   0.700      |


## Example 3: GPU-based Interactive Visualization of 70,000 Human Lung Cells (beta version)

![Interactive browser Demo](images/viz3-2.gif)

We demonstrate how to use RAPIDS, Scanpy and Plotly Dash to create an interactive dashboard where we visualize a single-cell RNA-sequencing dataset. Within the interactive dashboard, we can cluster, visualize, and compare any selected groups of cells.

### Installation

Additional dependencies are needed for this example. Follow these instructions for conda installation:

```bash
conda env create --name rapidgenomics-viz -f conda/rapidgenomics_cuda11.0.viz.yml
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

We report the runtime of these notebooks on various GCP instances below. All runtimes are given in seconds. Acceleration is given in parentheses. Benchmarking was performed on Dec 16, 2020.


| Step                         | CPU <br> n1-standard-16 <br> 16 vCPUs | GPU <br> n1-standard-16 <br> T4 16 GB GPU <br> (Acceleration)  | GPU <br> n1-highmem-8 <br> Tesla V100 16 GB GPU <br> (Acceleration) | GPU <br> a2-highgpu-1g <br> Tesla A100 40GB GPU <br> (Acceleration) |
|------------------------------|-------------------------|----------------------------|-------------------|-------------|
| PCA                          | 149                     |   (x)                 |  71  (x)       |  54  (2.8x) |
| KNN                          | 19.7                      |  (x)                | 20 (x)         |  5.3 (3.7x) |
| UMAP                         | 69                      |   (x)                 | 0.76 (x)       | 0.69 (100x) |
| Louvain clustering           | 13.1                     |  (x)                 | 0.12 (x)       | 0.11 (119x) |
| Leiden clustering            | 15.7                    |  (x)                  | 0.08 (x)       | 0.06 (262x) |
| t-SNE                        | 258                     |   (x)                 | 1.5  (x)       |  2.2 (117x) |
| Differential Peak Analysis   | 135                     |    (x)                | 21 (x)         | 10.4  (13x) |
| End-to-end notebook run      | 682                    |                        | 134            |   92        |
| Price ($/hr)                 | 0.760                   | 1.110                 | 2.953          | 3.673       |
| Total cost ($)               | 0.144                   |                       | 0.110          |    0.094    |


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
