import math

def pca(adata, n_components=50, train_ratio=0.35, n_batches=50, gpu=False):
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
