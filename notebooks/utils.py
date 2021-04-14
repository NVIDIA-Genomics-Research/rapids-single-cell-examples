import math
import h5py
import numpy as np
import cupy as cp
import cudf
import scipy
import dask
import pandas as pd

from cuml.dask.common.part_utils import _extract_partitions
from cuml.common.memory_utils import with_cupy_rmm

from statsmodels import robust

import warnings
warnings.filterwarnings('ignore', 'Expected ')
warnings.simplefilter('ignore')


def pca(adata, n_components=50, train_ratio=0.35, n_batches=50, gpu=False):

    """
    Performs a batched PCA by training on the first `train_ratio` samples
    and transforming in `n_batches` number of batches.

    Parameters
    ----------

    adata : anndata.AnnData of shape (n_cells, n_genes)
        Annotated data object for which to perform PCA

    n_components : int
        Number of principal components to keep

    train_ratio : float
        Percentage of cells to use for training

    n_batches : int
        Number of batches to use for transform

    gpu : bool
        Uses Scikit-Learn for CPU (gpu=False) and RAPIDS cuML for GPU
        (gpu=True)
    """

    train_size = math.ceil(adata.X.shape[0] * train_ratio)

    if gpu:
        from cuml.decomposition import PCA
        import cupy as cp
    else:
        from sklearn.decomposition import PCA
        import numpy as cp

    pca = PCA(n_components=n_components).fit(adata.X[:train_size])

    embeddings = cp.zeros((adata.X.shape[0], n_components))
    batch_size = int(embeddings.shape[0] / n_batches)
    for batch in range(n_batches):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size

        if(adata.X.shape[0] - end_idx < batch_size):
            end_idx = adata.X.shape[0]

        embeddings[start_idx:end_idx,:] = cp.asarray(pca.transform(adata.X[start_idx:end_idx]))

    if gpu:
        embeddings = embeddings.get()

    adata.obsm["X_pca"] = embeddings
    return adata


def tf_idf(filtered_cells):
    '''
    Input: 2D numpy.ndarray or 2D sparse matrix with X[i, j] = (binary or continuous) read count for cell i, peak j
    Output: Normalized matrix, where Xp[i, j] = X[i, j] * (1 / sum_peaks(i)) * log(1 + N_cells/N_cells with peak j)
    Note that the 1 / sum_peaks(i) term isn't included in the standard NLP form of tf-idf, but other single-cell work uses it.
    '''
    inv_sums = 1 / np.array(filtered_cells.sum(axis=1)).ravel()

    peak_counts = np.array((filtered_cells > 0).sum(axis=0)).ravel()
    log_inv_peak_freq = np.log1p(filtered_cells.shape[0] / peak_counts)

    normalized = filtered_cells.multiply(inv_sums[:, np.newaxis])
    normalized = normalized.multiply(log_inv_peak_freq[np.newaxis, :])
    normalized = scipy.sparse.csr_matrix(normalized)

    return normalized


def logtf_idf(filtered_cells, pseudocount=10**5):
    '''
    Input: 2D numpy.ndarray or 2D sparse matrix with X[i, j] = (binary or continuous) read count for cell i, peak j
    Output: Normalized matrix, where Xp[i, j] = X[i, j] * log(1 + pseudocount/sum_peaks(i)) * log(1 + N_cells/N_cells with peak j)
    Pseudocount should be chosen as a similar order of magnitude as the mean number of reads per cell.
    '''
    log_inv_sums = np.log1p(pseudocount / np.array(filtered_cells.sum(axis=1)).ravel())

    peak_counts = np.array((filtered_cells > 0).sum(axis=0)).ravel()
    log_inv_peak_freq = np.log1p(filtered_cells.shape[0] / peak_counts)

    normalized = filtered_cells.multiply(log_inv_sums[:, np.newaxis])
    normalized = normalized.multiply(log_inv_peak_freq[np.newaxis, :])
    normalized = scipy.sparse.csr_matrix(normalized)

    return normalized


def overlap(gene, fragment, upstream=10000, downstream=0):
    '''
    Checks if a genomic interval ('fragment') overlaps a gene, or some number of bases upstream/downstream
    of that gene.
    '''
    if gene[3] == 'rev':
        t = upstream
        upstream = downstream
        downstream = t
    if gene[0] != fragment[0]: # check chromosome
        return False
    if gene[2] + downstream >= fragment[1] and gene[1] - upstream <= fragment[1]: # peak start in gene
        return True
    if gene[2] + downstream >= fragment[2] and gene[1] - upstream <= fragment[2]: # peak end in gene
        return True
    if gene[1] - upstream >= fragment[1] and gene[2] + downstream <= fragment[2]: # gene entirely within peak
        return True


def filter_peaks(adata, n_top_peaks):
    '''
    Retains the top N most frequent peaks in the count matrix.
    '''
    peak_occurrences = np.sum(adata.X > 0, axis=0)
    peak_frequency = np.array(peak_occurrences / adata.X.shape[0]).flatten()
    frequent_peak_idxs = np.argsort(peak_frequency)
    use = frequent_peak_idxs[-n_top_peaks : ]
    return adata[:, use]


@with_cupy_rmm
def sq_sum_csr_matrix(client, csr_matrix, axis=0):
    '''
    Implements sum operation for dask array when the backend is cupy sparse csr matrix
    '''
    client = dask.distributed.default_client()

    def __sq_sum(x):
        x = x.multiply(x)
        return x.sum(axis=axis)

    parts = client.sync(_extract_partitions, csr_matrix)
    futures = [client.submit(__sq_sum,
                             part,
                             workers=[w],
                             pure=False)
               for w, part in parts]
    objs = []
    for i in range(len(futures)):
        obj = dask.array.from_delayed(futures[i],
                                      shape=futures[i].result().shape,
                                      dtype=cp.float32)
        objs.append(obj)
    return dask.array.concatenate(objs, axis=axis).compute().sum(axis=axis)


@with_cupy_rmm
def sum_csr_matrix(client, csr_matrix, axis=0):
    '''
    Implements sum operation for dask array when the backend is cupy sparse csr matrix
    '''
    client = dask.distributed.default_client()

    def __sum(x):
        return x.sum(axis=axis)

    parts = client.sync(_extract_partitions, csr_matrix)
    futures = [client.submit(__sum,
                             part,
                             workers=[w],
                             pure=False)
               for w, part in parts]
    objs = []
    for i in range(len(futures)):
        obj = dask.array.from_delayed(futures[i],
                                      shape=futures[i].result().shape,
                                      dtype=cp.float32)
        objs.append(obj)
    return dask.array.concatenate(objs, axis=axis).compute().sum(axis=axis)


def read_with_filter(client,
                     sample_file,
                     min_genes_per_cell=200,
                     max_genes_per_cell=6000,
                     min_cells = 0,
                     num_cells=None,
                     batch_size=50000,
                     partial_post_processor=None):
    """
    Reads an h5ad file and applies cell and geans count filter. Dask Array is
    used allow partitioning the input file. This function supports multi-GPUs.
    """

    # Path in h5 file
    _data = '/X/data'
    _index = '/X/indices'
    _indprt = '/X/indptr'
    _genes = '/var/_index'
    _barcodes = '/obs/_index'

    @dask.delayed
    def _read_partition_to_sparse_matrix(sample_file,
                                         total_cols, batch_start, batch_end,
                                         min_genes_per_cell=200,
                                         max_genes_per_cell=6000,
                                         post_processor=None):
        with h5py.File(sample_file, 'r') as h5f:
            indptrs = h5f[_indprt]
            start_ptr = indptrs[batch_start]
            end_ptr = indptrs[batch_end]

            # Read all things data and index
            sub_data = cp.array(h5f[_data][start_ptr:end_ptr])
            sub_indices = cp.array(h5f[_index][start_ptr:end_ptr])

            # recompute the row pointer for the partial dataset
            sub_indptrs  = cp.array(indptrs[batch_start:(batch_end + 1)])
            sub_indptrs = sub_indptrs - sub_indptrs[0]

        # Reconstruct partial sparse array
        partial_sparse_array = cp.sparse.csr_matrix(
            (sub_data, sub_indices, sub_indptrs),
            shape=(batch_end - batch_start, total_cols))

        # TODO: Add barcode filtering here.
        degrees = cp.diff(partial_sparse_array.indptr)
        query = ((min_genes_per_cell <= degrees) & (degrees <= max_genes_per_cell))
        partial_sparse_array = partial_sparse_array[query]

        if post_processor is not None:
            partial_sparse_array = post_processor(partial_sparse_array)

        return partial_sparse_array


    with h5py.File(sample_file, 'r') as h5f:
        # Compute the number of cells to read
        indptr = h5f[_indprt]
        genes = cudf.Series(h5f[_genes], dtype=cp.dtype('object'))

        total_cols = genes.shape[0]
        max_cells = indptr.shape[0] - 1
        if num_cells is not None:
            max_cells = num_cells

    dls = []
    for batch_start in range(0, max_cells, batch_size):
        actual_batch_size = min(batch_size, max_cells - batch_start)
        dls.append(dask.array.from_delayed(
                   (_read_partition_to_sparse_matrix)
                   (sample_file,
                    total_cols,
                    batch_start,
                    batch_start + actual_batch_size,
                    min_genes_per_cell=min_genes_per_cell,
                    max_genes_per_cell=max_genes_per_cell,
                    post_processor=partial_post_processor),
                   dtype=cp.float32,
                   shape=(actual_batch_size, total_cols)))

    dask_sparse_arr =  dask.array.concatenate(dls)
    dask_sparse_arr = dask_sparse_arr.persist()

    # Filter by genes (i.e. cell count per gene)
    gene_wise_cell_cnt = sum_csr_matrix(client, dask_sparse_arr)
    query = gene_wise_cell_cnt > min_cells

    # Filter genes for var
    genes = genes[query]
    genes = genes.reset_index(drop=True)

    query = cp.where(query == True)[0]
    dask_sparse_arr = dask_sparse_arr[:, query.get()].persist()

    return dask_sparse_arr, genes, query


def highly_variable_genes_filter(client,
                                 data_mat,
                                 genes,
                                 n_top_genes=None):

    if n_top_genes is None:
        n_top_genes = genes.shape[0] // 10

    mean = sum_csr_matrix(client, data_mat, axis=0) / data_mat.shape[0]
    mean[mean == 0] = 1e-12

    mean_sq = sq_sum_csr_matrix(client, data_mat, axis=0) / data_mat.shape[0]
    variance = mean_sq - mean ** 2
    variance *= data_mat.shape[1] / (data_mat.shape[0] - 1)
    dispersion = variance / mean

    df = pd.DataFrame()
    df['genes'] = genes.to_array()
    df['means'] = mean.tolist()
    df['dispersions'] = dispersion.tolist()
    df['mean_bin'] = pd.cut(
        df['means'],
        np.r_[-np.inf, np.percentile(df['means'], np.arange(10, 105, 5)), np.inf],
    )

    disp_grouped = df.groupby('mean_bin')['dispersions']
    disp_median_bin = disp_grouped.median()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        disp_mad_bin = disp_grouped.apply(robust.mad)
        df['dispersions_norm'] = (
            df['dispersions'].values - disp_median_bin[df['mean_bin'].values].values
        ) / disp_mad_bin[df['mean_bin'].values].values

    dispersion_norm = df['dispersions_norm'].values

    dispersion_norm = dispersion_norm[~np.isnan(dispersion_norm)]
    dispersion_norm[::-1].sort()

    if n_top_genes > df.shape[0]:
        n_top_genes = df.shape[0]

    disp_cut_off = dispersion_norm[n_top_genes - 1]
    vaiable_genes = np.nan_to_num(df['dispersions_norm'].values) >= disp_cut_off

    return vaiable_genes
