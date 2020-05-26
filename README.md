# Single cell genomics analysis examples accelerated with RAPIDS

This repository contains example notebooks demonstrating how to use RAPIDS for GPU-accelerated analysis of single-cell sequencing data.

## Installation 

All dependencies for this example can be installed with conda.

```bash
conda env create --name rapidgenomics -f conda/rapidgenomics.yml
conda activate rapidgenomics
python -m ipykernel install --user --display-name "Python (rapidgenomics)"
```

## Running the Notebooks

After installing the necessary dependencies, you can just run `jupyter lab`

## Human Lung Cell Atlas Example

We present an example using RAPIDS to accelerate the analysis of a ~70,000-cell single-cell RNA sequencing dataset from human lung cells. This example includes preprocessing, dimension reduction, clustering visualization and gene ranking. 

### Dataset

The dataset is based on [Travaglini et al. 2020](https://www.biorxiv.org/content/10.1101/742320v2) and can be downloaded following the instructions at https://github.com/krasnowlab/HLCA.

To run the examples, you will need to convert the dataset to a sparse `h5ad` format using [this notebook](notebooks/csv_to_h5ad.ipynb)

### Example

Follow this [Jupyter notebook](notebooks/hlca_lung_gpu_analysis.ipynb) for RAPIDS analysis of this dataset.

We provide a second notebook with the CPU version of this analysis [here](notebooks/hlca_lung_cpu_analysis.ipynb).

