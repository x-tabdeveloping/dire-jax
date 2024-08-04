import faiss
import numpy as np
from scipy.sparse import csr_matrix
import gc
import time
def make_knn_adjacency(data, n_neighbors):
    #
    data = np.ascontiguousarray(data.astype(np.float32))
    n_neighbors = n_neighbors + 1  # Including the point itself
    data_dim = data.shape[1]
    n_samples = data.shape[0]

    try:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, data_dim)
    except Exception as e:
        index = faiss.IndexFlatL2(data_dim)

    index.add(data)

    distances, indices = index.search(data, n_neighbors)


    row_idx = np.repeat(np.arange(n_samples), n_neighbors)
    col_idx = indices.ravel()

    data_values = distances.ravel()
    adj_mat = csr_matrix((data_values, (row_idx, col_idx)), shape=(n_samples, n_samples))

    del index
    gc.collect()
        #
    return adj_mat


import time
from contextlib import contextmanager

@contextmanager
def timing(block_name, dict):
    start_time = time.time()
    yield
    end_time = time.time()
    dict[block_name] = end_time - start_time
