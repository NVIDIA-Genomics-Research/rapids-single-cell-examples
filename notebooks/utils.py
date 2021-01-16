import math
import warnings
import numpy as np
import scipy
import h5py

import cudf
import cupy as cp
import pandas as pd
import dask

from statsmodels import robust

from cuml.dask.common.part_utils import _extract_partitions
from cuml.common.memory_utils import with_cupy_rmm

from bokeh.plotting import figure
from bokeh.io import push_notebook, show

from bokeh.models import ColorBar, ColumnDataSource
from bokeh.palettes import Blues9 as Blues
from bokeh.models import LinearColorMapper, ColorBar

import rapids_scanpy_funcs


COLORS = ["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941",
          "#006FA6", "#A30059", "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC",
          "#B79762", "#004D43", "#8FB0FF", "#997D87", "#5A0007", "#809693",
          "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
          "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9",
          "#B903AA", "#D16100", "#DDEFFF", "#000035", "#7B4F4B", "#A1C299",
          "#300018", "#0AA6D8", "#013349", "#00846F", "#372101", "#FFB500",
          "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
          "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68",
          "#7A87A1", "#788D66", "#885578", "#FAD09F", "#FF8A9A", "#D157A0",
          "#BEC459", "#456648", "#0086ED", "#886F4C", "#34362D", "#B4A8BD",
          "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
          "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757",
          "#C8A1A1", "#1E6E00", "#7900D7", "#A77500", "#6367A9", "#A05837",
          "#6B002C", "#772600", "#D790FF", "#9B9700", "#549E79", "#FFF69F",
          "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
          "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804",
          "#324E72", "#6A3A4C",]


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


@dask.delayed
def read_partition_to_sparse_matrix(sample_file,
                                    ds_data, ds_indices, ds_indptr, ds_genes,
                                    batch_start, rows,
                                    gene_filter, post_processor,
                                    min_genes_per_cell=200,
                                    max_genes_per_cell=6000,
                                    target_sum=1e4):
    """
    Loads a single partition into a sparse matrix from HDF5 file. Also apply the
    filter on individual partitions. This function will always be delayed and
    expected to be used with dask.

    A post processor function can be used to manipulate the paritition.
    """
    batch_end = batch_start + rows

    with h5py.File(sample_file, 'r') as h5f:
        # Read all things row pointers for one worker
        total_cols = h5f[ds_genes].shape[0]

        indptrs = h5f[ds_indptr]
        start_ptr = indptrs[batch_start]
        end_ptr = indptrs[batch_end]


        # Read all things data for one worker
        data = h5f[ds_data]
        sub_data = cp.array(data[start_ptr:end_ptr])

        # Read all things column pointers for one worker
        indices = h5f[ds_indices]
        sub_indices = cp.array(indices[start_ptr:end_ptr])

        # recompute the row pointer for the partial dataset
        sub_indptrs  = cp.array(indptrs[batch_start:(batch_end + 1)])
        first_ptr = sub_indptrs[0]
        sub_indptrs = sub_indptrs - first_ptr


    partial_sparse_array = cp.sparse.csr_matrix(
        (sub_data, sub_indices, sub_indptrs),
        shape=(batch_end - batch_start, total_cols))

    # TODO: Add barcode filtering here.
    degrees = cp.diff(partial_sparse_array.indptr)
    query = ((min_genes_per_cell <= degrees) & (degrees <= max_genes_per_cell))
    partial_sparse_array = partial_sparse_array[query]
    partial_sparse_array = rapids_scanpy_funcs.normalize_total(
        partial_sparse_array, target_sum=target_sum)

    ret_value = None

    if post_processor is not None:
        ret_value = post_processor(partial_sparse_array)

    if gene_filter is not None:
        ret_value = partial_sparse_array[:, gene_filter]

    if ret_value is None:
        ret_value = partial_sparse_array

    del partial_sparse_array
    return ret_value


def read_to_sparse_matrix(input_file,
                          max_cells, batch_size, total_cols,
                          gene_filter, post_processor,
                          min_genes_per_cell=200, max_genes_per_cell=6000):
    """
    Loads a sparse matrix from an HDF file into dask array.
    """
    dls = []
    for batch_start in range(0, max_cells, batch_size):
        batch_size = min(batch_size, max_cells - batch_start)
        dls.append(
            dask.array.from_delayed(
                (read_partition_to_sparse_matrix)(
                    input_file,
                    '/X/data', '/X/indices', '/X/indptr', '/var/_index',
                    batch_start, batch_size,
                    gene_filter, post_processor,
                    min_genes_per_cell=min_genes_per_cell,
                    max_genes_per_cell=max_genes_per_cell),
                dtype=cp.float32,
                shape=(batch_size, total_cols)))

    sum_gpu_arrays  = dask.array.concatenate(dls)
    return sum_gpu_arrays


def highly_variable_genes_filter(sum_mat,
                                 sum_sq_mat,
                                 data_mat,
                                 genes,
                                 n_top_genes=None):

    if n_top_genes is None:
        n_top_genes = genes.shape[0] // 10

    mean = sum_mat / data_mat.shape[0]
    mean[mean == 0] = 1e-12

    mean_sq = sum_sq_mat / data_mat.shape[0]
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


@with_cupy_rmm
def csr_to_csc(csr_array, client):
    def _conv_csr_to_csc(x):
        return  x.tocsc()

    parts = client.sync(_extract_partitions, csr_array)
    futures = [client.submit(_conv_csr_to_csc,
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
    return dask.array.concatenate(objs)


@with_cupy_rmm
def sum_csc(csc_array, client):

    def __sum(x):
        return x.sum(axis=1)

    parts = client.sync(_extract_partitions, csc_array)
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
    return dask.array.concatenate(objs)


@with_cupy_rmm
def clip(arr, client, max_value=10):

    def __clip(x, max_value):
        return x.clip(a_max=max_value)

    parts = client.sync(_extract_partitions, arr)
    futures = [client.submit(__clip,
                             part,
                             max_value,
                             workers=[w],
                             pure=False)
               for w, part in parts]
    objs = []
    for i in range(len(futures)):
        obj = dask.array.from_delayed(futures[i],
                                      shape=futures[i].result().shape,
                                      dtype=cp.float32)
        objs.append(obj)

    return dask.array.concatenate(objs, axis=1)


def sparse_array_to_df(sparse_dask_array, n_workers):

    @dask.delayed
    def _sparse_array_to_df(sparse_array):
        return cudf.DataFrame(sparse_array)

    num_recs = sparse_dask_array.shape[0]
    batch_size = math.ceil(num_recs / n_workers)
    # columns = genes.to_arrow().to_pylist()
    print('Number of records is', num_recs, 'and batch size is', batch_size)

    dls = []
    for start in range(0, num_recs, batch_size):
        bsize = min(num_recs - start, batch_size)
        dls.append(_sparse_array_to_df(sparse_dask_array[start:start+bsize]))


    print("Creating dask df from delays...")
    prop_meta = {i: pd.Series([], dtype='float32') for i in range(sparse_dask_array.shape[1])}
    meta_df = cudf.DataFrame(prop_meta)

    print("Creating Dataframe from futures...")
    return dask.dataframe.from_delayed(dls, meta=meta_df)


def show_tsne(df, x, y, cluster_col, title):
    tsne_fig = figure(title=title, width=800, output_backend="webgl")
    clusters = df[cluster_col].unique().values_host

    for cluster in clusters:
        cdf = df.query(cluster_col + ' == ' + str(cluster))
        if cdf.shape[0] == 0:
            continue

        tsne_fig.circle(cdf[x].to_array(),
                        cdf[y].to_array(),
                        size=2,
                        color=COLORS[cluster],
                        legend = 'Cluster ' + str(cluster))

    tsne_fig.legend.location = 'top_right'
    tsne_fig.legend.title = 'Clusters'

    tsne_fig_handle = show(tsne_fig, notebook_handle=True)
    push_notebook(handle=tsne_fig_handle)


def show_tsne_grad(df, x, y, color_col, title):

    color_array = cp.fromDlpack(df[color_col].to_dlpack())
    source = ColumnDataSource(dict(x=df[x].to_array(),
                                   y=df[y].to_array(),
                                   color_col=color_array.get()))

    mapper = LinearColorMapper(palette=Blues,
                               low=df[color_col].min(),
                               high=df[color_col].max())

    tsne_fig = figure(title=title,
                      width=800,
                      output_backend="webgl")


    tsne_fig.scatter('x', 'y',
                     color={'field': 'color_col', 'transform':mapper},
                     source=source,
                     size=2)

    color_bar = ColorBar(color_mapper=mapper, width=8,  location=(0,0))
    tsne_fig.add_layout(color_bar, 'right')

    tsne_fig_handle = show(tsne_fig, notebook_handle=True)
    push_notebook(handle=tsne_fig_handle)
