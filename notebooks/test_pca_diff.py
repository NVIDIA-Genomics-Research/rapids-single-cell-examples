import dask.array as da
from dask_cuda.local_cuda_cluster import cuda_visible_devices
from dask_cuda.utils import get_n_gpus
from dask_cuda import initialize, LocalCUDACluster
from dask.distributed import Client

from cuml.dask.decomposition import PCA as cu_dask_PCA
from cuml.decomposition import PCA

import math
import dask
import cudf
import cupy
import h5py
import pandas as pd


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


def pca_multi_gpu(dask_df, n_components, client):
    pca_model = cu_dask_PCA(n_components=n_components, client=client)
    pca_model = pca_model(dask_df)
    dask_reduced_df = pca_model.fit_transform(dask_df)

    reduced_df = dask_reduced_df.compute()
    return pca_model, reduced_df

def pca_single_gpu(df, n_components=50, n_batches=50):
    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(df)

    return embeddings

def main():

    client, cluster = initialize_cluster()
    with h5py.File('test2.h5', 'r') as h5f:
        d = cupy.array(h5f['/data'])
        print (d.shape)

        x = da.from_array(d, chunks=(1000, 1000))
        dask_df = sparse_array_to_df(x, 7)

        pca_model, reduced_df = pca_multi_gpu(dask_df, 50, client)
        reduced_df = reduced_df.reset_index(drop=True)
        print(reduced_df.head())
        print('--------------------------')

        df = dask_df.compute()

        reduced_single = pca_single_gpu(df, n_components=50, n_batches=50)
        print(type(reduced_single))
        print(reduced_single[:5])
        print('--------------------------')

    cluster.close()
    client.close()


def initialize_cluster():
    enable_tcp_over_ucx = True
    enable_nvlink = False
    enable_infiniband = True

    initialize.initialize(create_cuda_context=True,
                            enable_tcp_over_ucx=enable_tcp_over_ucx,
                            enable_nvlink=enable_nvlink,
                            enable_infiniband=enable_infiniband)
    device_list = cuda_visible_devices(1, range(get_n_gpus())).split(',')
    CUDA_VISIBLE_DEVICES = []
    for device in device_list:
        try:
            CUDA_VISIBLE_DEVICES.append(int(device))
        except ValueError as vex:
            print(vex)

    cluster = LocalCUDACluster(protocol="ucx",
                                dashboard_address=':8787',
                                CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES,
                                enable_tcp_over_ucx=enable_tcp_over_ucx,
                                enable_nvlink=enable_nvlink,
                                enable_infiniband=enable_infiniband)
    client = Client(cluster)
    client.run(cupy.cuda.set_allocator)
    return client, cluster

if __name__ == '__main__':
    main()