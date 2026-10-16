import numpy as np
import torch

from .tas_torch import neighborhood_overlaps

try:
    import faiss
except ImportError:
    faiss = None


def _require_faiss():
    if faiss is None:
        raise ImportError(
            "faiss is required for FAISS-based similarity. Install it with "
            "`uv sync --extra faiss` (or `pip install pointcloudsimilarity[faiss]`)."
        )


class TASFAISS:
    def __init__(self, approximate=False, metric='cosine', gpu=None):
        _require_faiss()
        assert metric in ['euclidean', 'cosine'], "Metric must be 'euclidean' or 'cosine'"

        self.approximate = approximate
        self.metric = metric
        # Default to the GPU only if this FAISS build can see one (faiss-cpu cannot).
        if gpu is None:
            gpu = hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0
        self.gpu = gpu
    
    def __create_index(self, data, gpu):
        n_samples, d = data.shape
        # A. Create Index
        if self.approximate:
            # IVF (Inverted File System) -> The O(N log N) magic
            # nlist: Number of Voronoi cells (clusters). 
            # Rule of thumb: 4 * sqrt(N)
            nlist = int(4 * np.sqrt(n_samples))
            quantizer = faiss.IndexFlatL2(d) if self.metric == 'euclidean' else faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, 
                                       faiss.METRIC_L2 if self.metric == 'euclidean' else faiss.METRIC_INNER_PRODUCT)
        else:
            # Flat (Exact) -> Brute force but highly optimized C++
            if self.metric == 'euclidean':
                index = faiss.IndexFlatL2(d)
            else:
                index = faiss.IndexFlatIP(d)

        # B. Move to GPU (if requested)
        if gpu:
            res = faiss.StandardGpuResources()
            # If multiple GPUs, use index_cpu_to_all_gpus
            index = faiss.index_cpu_to_gpu(res, 0, index)

        return index
    
    def fit(self, X, Y):
        self.n_samples = X.shape[0]
        assert self.n_samples == Y.shape[0], "X and Y must have the same number of samples"

        # Ensure float32 (FAISS requirement)
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        
        # Cosine neighbors are inner-product neighbors of L2-normalized data;
        # Euclidean (L2) preserves magnitude topology.
        if self.metric == 'cosine':
            faiss.normalize_L2(X)
            faiss.normalize_L2(Y)
        
        self.X_index = self.__create_index(X, self.gpu)
        self.Y_index = self.__create_index(Y, self.gpu)
        # C. Train (Required for IVF/Approximate)
        if self.approximate:
            # Training clusters the data to build Voronoi cells
            # We usually train on a subset if N is huge, but here we train on all
            # (you should sample X/Y before calling fit if N is large)
            self.X_index.train(X)
            self.X_index.nprobe = 100  # Number of cells to visit (Accuracy vs Speed trade-off)
            self.Y_index.train(Y)
            self.Y_index.nprobe = 100

        # D. Add Data
        self.X_index.add(X)
        self.Y_index.add(Y)
        
    def __call__(self, k : int=5, batch_size : int=5000):
        search_k = k + 1  # We need k neighbors + self

        # Search returns distances (D) and indices (I)
        # For huge N, search in batches to avoid VRAM OOM on the result matrix
        all_idx_X = []
        all_idx_Y = []

        for i in range(0, self.n_samples, batch_size):
            end = min(i + batch_size, self.n_samples)
            batch_query_X = self.X_index.reconstruct_n(i, end - i)
            batch_query_Y = self.Y_index.reconstruct_n(i, end - i)

            _, I_X = self.X_index.search(batch_query_X, search_k)
            _, I_Y = self.Y_index.search(batch_query_Y, search_k)

            all_idx_X.append(I_X)
            all_idx_Y.append(I_Y)

        idx_X = _drop_self(np.vstack(all_idx_X), k)
        idx_Y = _drop_self(np.vstack(all_idx_Y), k)

        # Compute Jaccard Similarity
        device = 'cuda' if self.gpu and torch.cuda.is_available() else 'cpu'
        t_idx_X = torch.from_numpy(idx_X).long().to(device)
        t_idx_Y = torch.from_numpy(idx_Y).long().to(device)

        total_jaccard = 0.0
        for i in range(0, self.n_samples, batch_size):
            end = min(i + batch_size, self.n_samples)
            jaccards = neighborhood_overlaps(
                t_idx_X[i:end], t_idx_Y[i:end], self.n_samples, k
            )
            total_jaccard += jaccards.sum().item()

        return total_jaccard / self.n_samples


def _drop_self(indices, k):
    """
    Remove each query point from its own (k + 1)-neighbor row, by index.

    Assuming the point comes back first fails for duplicate points. If the
    point is missing from its row (exact ties), drop the farthest instead.
    """
    is_self = indices == np.arange(len(indices))[:, None]
    keep = ~is_self
    keep[~is_self.any(axis=1), -1] = False
    return indices[keep].reshape(len(indices), k)
