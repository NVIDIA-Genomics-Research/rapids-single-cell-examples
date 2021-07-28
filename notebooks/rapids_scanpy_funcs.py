#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cupy as cp
import cudf
import cugraph

import numpy as np
import pandas as pd
import scipy
import math
import warnings

import h5py
from statsmodels import robust

from cuml.linear_model import LinearRegression


def _regress_out_chunk(X, y):
    """
    Performs a data_chunk.shape[1] number of local linear regressions,
    replacing the data in the original chunk w/ the regressed result.

    Parameters
    ----------

    X : cupy.ndarray of shape (n_cells, 3)
        Matrix of regressors

    y : cupy.ndarray or cupy.sparse.spmatrix of shape (n_cells,)
        containing a single column of the cellxgene matrix

    Returns
    -------

    dense_mat : cupy.ndarray of shape (n_cells,)
        Adjusted column
    """
    if cp.sparse.issparse(y):
        y = y.todense()
    
    lr = LinearRegression(fit_intercept=False, output_type="cupy")
    lr.fit(X, y, convert_dtype=True)
    return y.reshape(y.shape[0],) - lr.predict(X).reshape(y.shape[0])
    

def normalize_total(csr_arr, target_sum):
    """
    Normalizes rows in matrix so they sum to `target_sum`

    Parameters
    ----------

    csr_arr : cupy.sparse.csr_matrix of shape (n_cells, n_genes)
        Matrix to normalize

    target_sum : int
        Each row will be normalized to sum to this value

    Returns
    -------

    csr_arr : cupy.sparse.csr_arr of shape (n_cells, n_genes)
        Normalized sparse matrix
    """
    
    mul_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void mul_kernel(const int *indptr, float *data, 
                    int nrows, int tsum) {
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        
        if(row >= nrows)
            return;
        
        float scale = 0.0;
        int start_idx = indptr[row];
        int stop_idx = indptr[row+1];

        for(int i = start_idx; i < stop_idx; i++)
            scale += data[i];

        if(scale > 0.0) {
            scale = tsum / scale;
            for(int i = start_idx; i < stop_idx; i++)
                data[i] *= scale;
        }
    }
    ''', 'mul_kernel')
    
    mul_kernel((math.ceil(csr_arr.shape[0] / 32.0),), (32,),
               (csr_arr.indptr,
                csr_arr.data,
                csr_arr.shape[0],
               int(target_sum)))
    
    return csr_arr


def regress_out(normalized, n_counts, percent_mito, verbose=False):

    """
    Use linear regression to adjust for the effects of unwanted noise
    and variation.

    Parameters
    ----------

    normalized : cupy.sparse.csc_matrix of shape (n_cells, n_genes)
        The matrix to adjust. The adjustment will be performed over
        the columns.

    n_counts : cupy.ndarray of shape (n_cells,)
        Number of genes for each cell

    percent_mito : cupy.ndarray of shape (n_cells,)
        Percentage of genes that each cell needs to adjust for

    verbose : bool
        Print debugging information

    Returns
    -------

    outputs : cupy.ndarray
        Adjusted matrix
    """

    regressors = cp.ones((n_counts.shape[0]*3)).reshape((n_counts.shape[0], 3), order="F")

    regressors[:, 1] = n_counts
    regressors[:, 2] = percent_mito
    
    outputs = cp.empty(normalized.shape, dtype=normalized.dtype, order="F")
    
    if n_counts.shape[0] < 1000000:
        normalized = normalized.todense()
    
    for i in range(normalized.shape[1]):
        if verbose and i % 500 == 0:
            print("Regressed %s out of %s" %(i, normalized.shape[1]))
        X = regressors
        y = normalized[:,i]
        outputs[:, i] = _regress_out_chunk(X, y)
    
    return outputs


def filter_cells(sparse_gpu_array, min_genes, max_genes, rows_per_batch=10000, barcodes=None):
    """
    Filter cells that have genes greater than a max number of genes or less than
    a minimum number of genes.

    Parameters
    ----------

    sparse_gpu_array : cupy.sparse.csr_matrix of shape (n_cells, n_genes)
        CSR matrix to filter

    min_genes : int
        Lower bound on number of genes to keep

    max_genes : int
        Upper bound on number of genes to keep

    rows_per_batch : int
        Batch size to use for filtering. This can be adjusted for performance
        to trade-off memory use.

    barcodes : series
        cudf series containing cell barcodes.

    Returns
    -------

    filtered : scipy.sparse.csr_matrix of shape (n_cells, n_genes)
        Matrix on host with filtered cells

    barcodes : If barcodes are provided, also returns a series of 
        filtered barcodes.
    """

    n_batches = math.ceil(sparse_gpu_array.shape[0] / rows_per_batch)
    filtered_list = []
    barcodes_batch = None
    for batch in range(n_batches):
        batch_size = rows_per_batch
        start_idx = batch * batch_size
        stop_idx = min(batch * batch_size + batch_size, sparse_gpu_array.shape[0])
        arr_batch = sparse_gpu_array[start_idx:stop_idx]
        if barcodes is not None:
            barcodes_batch = barcodes[start_idx:stop_idx]
        filtered_list.append(_filter_cells(arr_batch, 
                                            min_genes=min_genes, 
                                            max_genes=max_genes, 
                                            barcodes=barcodes_batch))

    if barcodes is None:
        return scipy.sparse.vstack(filtered_list)
    else:
        filtered_data = [x[0] for x in filtered_list]
        filtered_barcodes = [x[1] for x in filtered_list]
        filtered_barcodes = cudf.concat(filtered_barcodes)
        return scipy.sparse.vstack(filtered_data), filtered_barcodes.reset_index(drop=True)


def _filter_cells(sparse_gpu_array, min_genes, max_genes, barcodes=None):
    degrees = cp.diff(sparse_gpu_array.indptr)
    query = ((min_genes <= degrees) & (degrees <= max_genes)).ravel()
    query = query.get()
    if barcodes is None:
        return sparse_gpu_array.get()[query]
    else:
        return sparse_gpu_array.get()[query], barcodes[query]


def filter_genes(sparse_gpu_array, genes_idx, min_cells=0):
    """
    Filters out genes that contain less than a specified number of cells

    Parameters
    ----------

    sparse_gpu_array : scipy.sparse.csr_matrix of shape (n_cells, n_genes)
        CSR Matrix to filter

    genes_idx : cudf.Series or pandas.Series of size (n_genes,)
        Current index of genes. These must map to the indices in sparse_gpu_array

    min_cells : int
        Genes containing a number of cells below this value will be filtered
    """
    thr = np.asarray(sparse_gpu_array.sum(axis=0) >= min_cells).ravel()
    filtered_genes = cp.sparse.csr_matrix(sparse_gpu_array[:, thr])
    genes_idx = genes_idx[np.where(thr)[0]]
    
    return filtered_genes, genes_idx.reset_index(drop=True)


def select_groups(labels, groups_order_subset='all'):
    adata_obs_key = labels
    groups_order = labels.cat.categories
    groups_masks = cp.zeros(
        (len(labels.cat.categories), len(labels.cat.codes)), dtype=bool
    )
    for iname, name in enumerate(labels.cat.categories.to_pandas()):
        # if the name is not found, fallback to index retrieval
        if labels.cat.categories[iname] in labels.cat.codes:
            mask = labels.cat.categories[iname] == labels.cat.codes
        else:
            mask = iname == labels.cat.codes
        groups_masks[iname] = mask.values
    groups_ids = list(range(len(groups_order)))
    if groups_order_subset != 'all':
        groups_ids = []
        for name in groups_order_subset:
            groups_ids.append(
                cp.where(cp.array(labels.cat.categories.to_array().astype("int32")) == int(name))[0][0]
            )
        if len(groups_ids) == 0:
            # fallback to index retrieval
            groups_ids = cp.where(
                cp.in1d(
                    cp.arange(len(labels.cat.categories)).astype(str),
                    cp.array(groups_order_subset),
                )
            )[0]
            
        groups_ids = [groups_id.item() for groups_id in groups_ids]
        groups_masks = groups_masks[groups_ids]
        groups_order_subset = labels.cat.categories[groups_ids].to_array().astype(int)
    else:
        groups_order_subset = groups_order.to_array()
    return groups_order_subset, groups_masks


def rank_genes_groups(
    X,
    labels,  # louvain results
    var_names,
    groups=None,
    reference='rest',
    n_genes=100,
    **kwds,
):

    """
    Rank genes for characterizing groups.

    Parameters
    ----------

    X : cupy.ndarray of shape (n_cells, n_genes)
        The cellxgene matrix to rank genes

    labels : cudf.Series of size (n_cells,)
        Observations groupings to consider

    var_names : cudf.Series of size (n_genes,)
        Names of genes in X

    groups : Iterable[str] (default: 'all')
        Subset of groups, e.g. ['g1', 'g2', 'g3'], to which comparison
        shall be restricted, or 'all' (default), for all groups.

    reference : str (default: 'rest')
        If 'rest', compare each group to the union of the rest of the group.
        If a group identifier, compare with respect to this group.

    n_genes : int (default: 100)
        The number of genes that appear in the returned tables.
    """

    #### Wherever we see "adata.obs[groupby], we should just replace w/ the groups"
        
    # for clarity, rename variable
    if groups == 'all':
        groups_order = 'all'
    elif isinstance(groups, (str, int)):
        raise ValueError('Specify a sequence of groups')
    else:
        groups_order = list(groups)
        if isinstance(groups_order[0], int):
            groups_order = [str(n) for n in groups_order]
        if reference != 'rest' and reference not in set(groups_order):
            groups_order += [reference]
    if (
        reference != 'rest'
        and reference not in set(labels.cat.categories)
    ):
        cats = labels.cat.categories.tolist()
        raise ValueError(
            f'reference = {reference} needs to be one of groupby = {cats}.'
        )

    groups_order, groups_masks = select_groups(labels, groups_order)
    
    original_reference = reference
    
    n_vars = len(var_names)

    # for clarity, rename variable
    n_genes_user = n_genes
    # make sure indices are not OoB in case there are less genes than n_genes
    if n_genes_user > X.shape[1]:
        n_genes_user = X.shape[1]
    # in the following, n_genes is simply another name for the total number of genes
    n_genes = X.shape[1]

    n_groups = groups_masks.shape[0]
    ns = cp.zeros(n_groups, dtype=int)
    for imask, mask in enumerate(groups_masks):
        ns[imask] = cp.where(mask)[0].size
    if reference != 'rest':
        ireference = cp.where(groups_order == reference)[0][0]
    reference_indices = cp.arange(n_vars, dtype=int)

    rankings_gene_scores = []
    rankings_gene_names = []

    # Perform LogReg
        
    # if reference is not set, then the groups listed will be compared to the rest
    # if reference is set, then the groups listed will be compared only to the other groups listed
    from cuml.linear_model import LogisticRegression
    reference = groups_order[0]
    if len(groups) == 1:
        raise Exception('Cannot perform logistic regression on a single cluster.')
    grouping_mask = labels.isin(cudf.Series(groups_order))
    grouping = labels.loc[grouping_mask]
    
    X = X[grouping_mask.values, :]  # Indexing with a series causes issues, possibly segfault
    y = labels.loc[grouping]
    
    clf = LogisticRegression(**kwds)
    clf.fit(X.get(), grouping.to_array().astype('float32'))
    scores_all = cp.array(clf.coef_).T
    
    for igroup, group in enumerate(groups_order):
        if len(groups_order) <= 2:  # binary logistic regression
            scores = scores_all[0]
        else:
            scores = scores_all[igroup]

        partition = cp.argpartition(scores, -n_genes_user)[-n_genes_user:]
        partial_indices = cp.argsort(scores[partition])[::-1]
        global_indices = reference_indices[partition][partial_indices]
        rankings_gene_scores.append(scores[global_indices].get())  ## Shouldn't need to take this off device
        rankings_gene_names.append(var_names[global_indices].to_pandas())
        if len(groups_order) <= 2:
            break

    groups_order_save = [str(g) for g in groups_order]
    if (len(groups) == 2):
        groups_order_save = [g for g in groups_order if g != reference]
            
    scores = np.rec.fromarrays(
        [n for n in rankings_gene_scores],
        dtype=[(rn, 'float32') for rn in groups_order_save],
    )
    
    names = np.rec.fromarrays(
        [n for n in rankings_gene_names],
        dtype=[(rn, 'U50') for rn in groups_order_save],
    )
        
    return scores, names, original_reference


def leiden(adata, resolution=1.0):
    """
    Performs Leiden Clustering using cuGraph

    Parameters
    ----------

    adata : annData object with 'neighbors' field.
       
    resolution : float, optional (default: 1)
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.

    """
    # Adjacency graph
    adjacency = adata.obsp['connectivities']
    offsets = cudf.Series(adjacency.indptr)
    indices = cudf.Series(adjacency.indices)
    g = cugraph.Graph()
    if hasattr(g, 'add_adj_list'):
        g.add_adj_list(offsets, indices, None)
    else:
        g.from_cudf_adjlist(offsets, indices, None)
    
    # Cluster
    leiden_parts, _ = cugraph.leiden(g,resolution = resolution)
    
    # Format output
    clusters = leiden_parts.to_pandas().sort_values('vertex')[['partition']].to_numpy().ravel()
    clusters = pd.Categorical(clusters)
    
    return clusters


def _cellranger_hvg(mean, mean_sq, genes, n_cells, n_top_genes):

    mean[mean == 0] = 1e-12
    variance = mean_sq - mean ** 2
    variance *= len(genes) / (n_cells - 1)
    dispersion = variance / mean

    df = pd.DataFrame()
    # Note - can be replaced with cudf once 'cut' is added in 21.08
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

    if n_top_genes is None:
        n_top_genes = genes.shape[0] // 10

    if n_top_genes > df.shape[0]:
        n_top_genes = df.shape[0]

    disp_cut_off = dispersion_norm[n_top_genes - 1]
    variable_genes = np.nan_to_num(df['dispersions_norm'].values) >= disp_cut_off
    return variable_genes



def highly_variable_genes(sparse_gpu_array, genes, n_top_genes=None):
    """
    Identifies highly variable genes using the 'cellranger' method.
    
    Parameters
    ----------
    
    sparse_gpu_array : scipy.sparse.csr_matrix of shape (n_cells, n_genes)
    
    genes : cudf series containing genes
    
    n_top_genes : number of variable genes
    """

    n_cells = sparse_gpu_array.shape[0]
    mean = sparse_gpu_array.sum(axis=0).flatten() / n_cells
    mean_sq = sparse_gpu_array.multiply(sparse_gpu_array).sum(axis=0).flatten() / n_cells
    variable_genes = _cellranger_hvg(mean, mean_sq, genes, n_cells, n_top_genes)
    
    return variable_genes


def preprocess_in_batches(input_file, markers, min_genes_per_cell=200, max_genes_per_cell=6000, 
                          min_cells_per_gene=1, target_sum=1e4, n_top_genes=5000):

    _data = '/X/data'
    _index = '/X/indices'
    _indptr = '/X/indptr'
    _genes = '/var/_index'

    cell_batch_size = 100000
    gene_batch_size = 2000
    
    batches = []
    mean = []
    mean_sq = []
    
    # Get data from h5 file
    print("Calculating data size.")
    with h5py.File(input_file, 'r') as h5f:
        indptrs = h5f[_indptr]
        indices = cp.array(h5f[_index])
        genes = cudf.Series(h5f[_genes], dtype=cp.dtype('object'))
        n_cells = indptrs.shape[0] - 1

    # Get indices of genes to filter
    print("Identifying genes to filter.")
    gene_query = (cp.bincount(indices) >= min_cells_per_gene)
    genes_filtered = genes[gene_query].reset_index(drop=True)

    print("Filtering and normalizing data")
    # Batch by cells and filter, normalize and log transform each batch
    n_cells_filtered = 0
    for batch_start in range(0, n_cells, cell_batch_size):
        # Get batch indices
        with h5py.File(input_file, 'r') as h5f:
            indptrs = h5f[_indptr]
            actual_batch_size = min(cell_batch_size, n_cells - batch_start)
            batch_end = batch_start + actual_batch_size
            start_ptr = indptrs[batch_start]
            end_ptr = indptrs[batch_end]

            # Read data and index of batch from hdf5
            sub_data = cp.array(h5f[_data][start_ptr:end_ptr])
            sub_indices = cp.array(h5f[_index][start_ptr:end_ptr])

            # recompute the row pointer for the partial dataset
            sub_indptrs  = cp.array(indptrs[batch_start:(batch_end + 1)])
            sub_indptrs = sub_indptrs - sub_indptrs[0]

        # Reconstruct partial sparse array
        partial_sparse_array = cp.sparse.csr_matrix(
            (sub_data, sub_indices, sub_indptrs),
            shape=(batch_end - batch_start, len(genes)))

        # Filter cells in the batch
        degrees = cp.diff(partial_sparse_array.indptr)
        query = ((min_genes_per_cell <= degrees) & (degrees <= max_genes_per_cell))
        n_cells_filtered += sum(query)
        partial_sparse_array = partial_sparse_array[query]
    
        # Filter genes
        partial_sparse_array = partial_sparse_array[:, gene_query]
    
        # Normalize
        partial_sparse_array = normalize_total(partial_sparse_array, target_sum=target_sum)

        # Log transform
        batches.append(partial_sparse_array.log1p())

    print("Calculating highly variable genes.")
    # Batch across genes to calculate gene-wise dispersions
    for batch_start in range(0, len(genes_filtered), gene_batch_size):
        # Get batch indices
        actual_batch_size = min(gene_batch_size, len(genes_filtered) - batch_start)
        batch_end = batch_start + actual_batch_size
    
        partial_sparse_array = cp.sparse.vstack([x[:, batch_start:batch_end] for x in batches])

        # Calculate sum per gene
        partial_mean = partial_sparse_array.sum(axis=0) / partial_sparse_array.shape[0]
        mean.append(partial_mean)

        # Calculate sq sum per gene - can batch across genes
        partial_sparse_array = partial_sparse_array.multiply(partial_sparse_array)
        partial_mean_sq = partial_sparse_array.sum(axis=0) / partial_sparse_array.shape[0]
        mean_sq.append(partial_mean_sq)
    
    mean = cp.hstack(mean).ravel()
    mean_sq = cp.hstack(mean_sq).ravel()

    variable_genes = _cellranger_hvg(mean, mean_sq, genes_filtered, n_cells_filtered, n_top_genes)

    print("Storing raw marker gene expression.")
    marker_genes_raw = {
        ("%s_raw" % marker): cp.sparse.vstack([x[:, genes_filtered == marker] for x in batches]).todense().ravel()
        for marker in markers
    }

    print("Filtering highly variable genes.")
    sparse_gpu_array =  cp.sparse.vstack([partial_sparse_array[:, variable_genes] for partial_sparse_array in batches])
    genes_filtered = genes_filtered[variable_genes].reset_index(drop=True)
    
    return sparse_gpu_array, genes_filtered, marker_genes_raw
