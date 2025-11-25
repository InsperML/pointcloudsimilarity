import numpy as np
import faiss
import torch

def compute_nngs_faiss(X, Y, k=5, approximate=False, metric='euclidean', batch_size=5000, gpu=True):
    """
    Computes NNGS using FAISS.
    
    Args:
        X, Y (np.array): Input data (N, D).
        k (int): Number of neighbors.
        approximate (bool): 
            If False: Uses IndexFlatL2 (Exact Search, O(N^2)). 
            If True: Uses IndexIVFFlat (Inverted File, O(N log N)).
        metric (str): 'euclidean' (respects magnitude) or 'cosine' (normalized).
        batch_size (int): Size of chunks for Jaccard calculation (RAM management).
        gpu (bool): Whether to use GPU resources.
    
    Returns:
        float: NNGS score.
    """
    # 1. Validation & Prep
    n_samples, d = X.shape
    assert n_samples == Y.shape[0]
    
    # Ensure float32 (FAISS requirement)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    
    # Handle Metric
    if metric == 'cosine':
        faiss.normalize_L2(X)
        faiss.normalize_L2(Y)
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    else:
        # Euclidean (L2) preserves magnitude topology
        faiss_metric = faiss.METRIC_L2

    search_k = k + 1  # We need k neighbors + self

    # 2. Define the Index Builder Helper
    def get_neighbors(data, index_type_str):
        dim = data.shape[1]
        
        # A. Create Index
        if approximate:
            # IVF (Inverted File System) -> The O(N log N) magic
            # nlist: Number of Voronoi cells (clusters). 
            # Rule of thumb: 4 * sqrt(N)
            nlist = int(4 * np.sqrt(n_samples))
            quantizer = faiss.IndexFlatL2(dim) if metric == 'euclidean' else faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss_metric)
        else:
            # Flat (Exact) -> Brute force but highly optimized C++
            if metric == 'euclidean':
                index = faiss.IndexFlatL2(dim)
            else:
                index = faiss.IndexFlatIP(dim)

        # B. Move to GPU (if requested)
        if gpu:
            res = faiss.StandardGpuResources()
            # If multiple GPUs, use index_cpu_to_all_gpus
            index = faiss.index_cpu_to_gpu(res, 0, index)

        # C. Train (Required for IVF/Approximate)
        if approximate:
            # Training clusters the data to build Voronoi cells
            # We usually train on a subset if N is huge, but here we train on all
            index.train(data)
            index.nprobe = 10  # Number of cells to visit (Accuracy vs Speed trade-off)

        # D. Add Data and Search
        index.add(data)
        
        # Search returns distances (D) and indices (I)
        # For huge N, search in batches to avoid VRAM OOM on the result matrix
        all_indices = []
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch_query = data[i:end]
            _, I = index.search(batch_query, search_k)
            all_indices.append(I)
        
        # Concatenate and remove first column (self)
        indices = np.vstack(all_indices)
        return indices[:, 1:]

    # 3. Retrieve Neighbors
    # This is the heavy lifting
    idx_X = get_neighbors(X, "X")
    idx_Y = get_neighbors(Y, "Y")

    # 4. Compute Jaccard Similarity
    # We use PyTorch for the intersection step because it offers
    # extremely fast boolean broadcasting on GPU.
    
    # Convert indices to Torch Tensor
    t_idx_X = torch.from_numpy(idx_X).long()
    t_idx_Y = torch.from_numpy(idx_Y).long()
    
    if gpu and torch.cuda.is_available():
        t_idx_X = t_idx_X.cuda()
        t_idx_Y = t_idx_Y.cuda()
    
    total_jaccard = 0.0
    
    # Batched Jaccard Calculation
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        
        b_idx_x = t_idx_X[i:end] # (Batch, k)
        b_idx_y = t_idx_Y[i:end] # (Batch, k)
        
        # Broadcasting Trick: (Batch, k, 1) == (Batch, 1, k) -> (Batch, k, k)
        matches = (b_idx_x.unsqueeze(2) == b_idx_y.unsqueeze(1))
        
        intersection = matches.sum(dim=(1, 2)).float()
        union = (2 * k) - intersection
        
        jaccards = intersection / union.clamp(min=1e-8)
        total_jaccard += jaccards.sum().item()
        
        del matches

    return total_jaccard / n_samples