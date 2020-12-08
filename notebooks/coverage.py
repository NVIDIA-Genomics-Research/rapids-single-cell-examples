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

from subprocess import Popen, PIPE

import cudf
import cupy as cp

import os
import time
import tabix
import numpy as np
import pandas as pd
from numba import cuda

from atacworks.dl4atac.models.models import DenoisingResNet
from atacworks.dl4atac.models.model_utils import load_model

import torch

def count_fragments(fragment_file):
    """
    Counts number of fragments per barcode in fragment file.
    
    Parameters
    ----------
    
    fragment_file: path to gzipped fragment file
    
    Returns
    -------
    barcode_counts: pandas DF with number of fragments per barcode.
    
    """
    fragment_barcodes = pd.read_csv(fragment_file, compression='gzip', sep='\t', header=None, usecols=[3])
    barcode_counts = fragment_barcodes.iloc[:,0].value_counts().reset_index()
    barcode_counts.columns = ['cell', 'fragments']
    return barcode_counts


def query_fragments(fragment_file, chrom, start, end):
    """
    Counts number of fragments per barcode in fragment file.
    
    Parameters
    ----------
    
    fragment_file: path to fragment file
    
    chrom: chromosome to query
    
    start: start of query region
    
    end: end of query region
    
    Returns
    -------
    
    records: fragments in given region.
    
    """
    tb = tabix.open(fragment_file)
    results = tb.querys("%s:%d-%d" % (chrom, start, end))
    records = []
    for record in results:
        records.append(record)
    return records


def tabix_query(filename, chrom, start, end):
    """
    Calls tabix and generate an array of strings for each line it returns.
    
    Parameters
    ----------
    
    filename: path to fragment file
    
    chrom: chromosome to query
    
    start: start of query region
    
    end: end of query region
    
    Returns
    -------
    
    records: fragments in given region.
    """
    query = '{}:{}-{}'.format(chrom, start, end)
    process = Popen(['tabix', '-f', filename, query], stdout=PIPE)
    records = []
    for line in process.stdout:
        record = line.decode('utf-8').strip().split('\t')
        records.append(record)
    return records


def read_fragments(chrom, start, end, fragment_file):
    """
    Creates a DF from the output of tabix_query.
    
    Parameters
    ----------
    
    filename: path to fragment file
    
    chrom: chromosome to query
    
    start: start of query region
    
    end: end of query region
    
    Returns
    -------
    
    fragments: DF containing fragments in given region.
    """
    fragments = cudf.DataFrame(
        data=tabix_query(fragment_file, chrom, start, end),
        columns=['chrom', 'start', 'end', 'cell', 'duplicate'])
    fragments.drop('duplicate', inplace=True, axis=1)
    fragments['row_num'] = fragments.index
    fragments = fragments.astype({"start": np.int32, "end": np.int32})
    fragments['len'] = fragments['end'] - fragments['start']

    return fragments


@cuda.jit

def expand_fragments(start, end, index, end_index,
                    interval_start, interval_end, interval_index, step):
    """
    Expands fragments to high resolution intervals.
    
    Parameters
    ----------
    
    start: start of fragment
    
    end: end of fragment
    
    index: index of fragment
    
    end_index: index of fragment end
    
    interval_start: array to fill start of each interval
    
    interval_end: array to fill end of each interval
    
    interval_index: array to fill index of each interval
    
    step: step size in bp
    

    """
    i = cuda.grid(1)

    # Starting position in the target frame
    first_index = end_index[i] - (end[i] - start[i])
    chrom_start = start[i]
    for j in range(first_index, end_index[i], step):
        interval_start[j] = chrom_start
        chrom_start = chrom_start + 1
        interval_end[j] = chrom_start
        interval_index[j] = index[i]

        
def get_coverages(start, end, fragments):
    """
    Calculates per-bp coverage per cluster.
    
    Parameters
    ----------
    
    start: start of selected region
    
    end: end of selected region
    
    fragments: DF containing fragments for selected region
    
    Returns:
    --------
    
    coverage_array: numpy array containing coverage for each cluster

    """
    
    # Copy fragments DF
    fragments_copy = fragments.copy()

    # Take cumulative sum of fragment lengths
    cum_sum = fragments_copy['len'].cumsum()
    expanded_size = cum_sum[len(fragments_copy) - 1].tolist()

    # Create expanded fragment dataframe
    expanded_fragments = cudf.DataFrame()
    start_arr = cp.zeros(expanded_size, dtype=cp.int32)
    end_arr = cp.zeros(expanded_size, dtype=cp.int32)
    rownum_arr = cp.zeros(expanded_size, dtype=cp.int32)

    # Expand all fragments to single-bp resolution
    expand_fragments.forall(fragments_copy.shape[0], 1)(
        fragments_copy['start'],
        fragments_copy['end'],
        fragments_copy['row_num'],
        cum_sum,
        start_arr,
        end_arr,
        rownum_arr,
        1)

    expanded_fragments['start'] = start_arr
    expanded_fragments['end'] = end_arr
    expanded_fragments['row_num'] = rownum_arr

    fragments_copy.drop(['start', 'end'], inplace=True, axis=1)
    expanded_fragments = expanded_fragments.merge(fragments_copy, on='row_num')

    # Count number of fragments at each position
    coverage_df = expanded_fragments.groupby(['chrom', 'start', 'end', 'cluster'], as_index=False).count()

    # List all clusters
    clusters = sorted(np.unique(fragments_copy['cluster'].to_array()))
    num_clusters = len(clusters)

    # Create empty array
    coverage_array = np.zeros(shape=(num_clusters, (end - start)))

    # Iterate over clusters to add coverage values
    for (i, cluster) in enumerate(clusters):
        cluster_df = coverage_df.loc[coverage_df['cluster'] == cluster]
        coords = cluster_df['start'] - start
        values = cluster_df['row_num']
        ind = (coords >= 0) & (coords < (end-start))
        coords = coords[ind].values.get()
        values = values[ind].values.get()
        coverage_array[i][coords] = values

    return coverage_array


def load_atacworks_model(weights_path, gpu, interval_size=50000):
    """
    Loads pre-trained AtacWorks resnet model.
    
    Parameters
    ----------
    
    weights_path: path to hdf5 file containing model weights.
    
    gpu: Index of GPU on which to load model.
    
    interval_size: interval size parameter for resnet model
    
    
    Returns:
    --------
    
    model: AtacWorks resnet model to be used for denoising and peak calling.

    """
    model = DenoisingResNet(interval_size=interval_size, kernel_size=51, kernel_size_class=51)
    model = load_model(model, weights_path=weights_path, rank=0)
    model = model.cuda(gpu)
    return model


def reshape_with_padding(coverage, interval_size, pad):
    """
    Reshapes array of coverage values for AtacWorks model.
    
    Parameters
    ----------
    
    coverage: array of coverage values per cluster.
    
    interval_size: interval_size parameter for AtacWorks model.
    
    pad: pad parameter for AtacWorks model
    
    
    Returns:
    --------
    
    reshaped coverage: reshaped array of coverage values.

    """
    if(len(coverage.shape)==1):
        coverage = coverage.reshape((1, coverage.shape[0]))
    
    # Calculate dimensions of empty array
    num_clusters = int(coverage.shape[0])
    n_intervals = int((coverage.shape[1] - 2*pad) / interval_size)
    padded_interval_size = int(interval_size + 2*pad)
    
    # Create empty array to fill in reshaped coverage values
    reshaped_coverage = np.zeros(shape=(num_clusters*n_intervals, padded_interval_size))
    
    if n_intervals == 1:
        interval_starts = [0]
    else:
        interval_starts = range(0, coverage.shape[1], interval_size + pad)
    
    # Fill in coverage values
    for i in range(num_clusters):
        reshaped_cluster_coverage = np.stack([coverage[i, start:start+padded_interval_size] for start in interval_starts])
        reshaped_coverage[i*n_intervals:(i+1)*n_intervals, :] = reshaped_cluster_coverage
    return reshaped_coverage


def atacworks_denoise(coverage, model, gpu, interval_size=50000, pad=0):
    """
    Denoises and calls peaks from coverage values using AtacWorks model.
    
    Parameters
    ----------
    
    coverage: array of coverage values per cluster.
    
    model: AtacWorks model object.
    
    gpu: Index of GPU for AtacWorks model.
    
    interval_size: interval_size parameter for AtacWorks model.
    
    pad: pad parameter for AtacWorks model
    
    
    Returns:
    --------
    
    pred: Predicted coverage and peaks by AtacWorks model.

    """
    # Reshape input
    input_arr = reshape_with_padding(coverage, interval_size, pad)
    with torch.no_grad():
        input_arr = torch.tensor(input_arr, dtype=float)
        input_arr = input_arr.unsqueeze(1)
        input_arr = input_arr.cuda(gpu, non_blocking=True).float()
        # Run model inference
        pred = model(input_arr)
        # Reshape output and remove padding
        pred = np.stack([x.cpu().numpy() for x in pred], axis=-1)
        center = range(pad, pred.shape[1] - pad)
        pred = pred[:, center, :]
        pred = pred.reshape((coverage.shape[0], coverage.shape[1] - 2*pad, pred.shape[2]))
        return pred
