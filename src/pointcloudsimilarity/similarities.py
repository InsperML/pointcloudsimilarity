from .cka import CKASimilarity
from .gulp import GULPSimilarity
from .procrustes import ProcrustesSimilarity
from .gw_sim import GWSimilarity
from .nngs import NNGSSimilarityCPU, NNGSSimilarityFaiss, NNGSSimilarityTorch
from .pwcca import PWCCASimilarity

__all__ = [
    "CKASimilarity",
    "GULPSimilarity",
    "ProcrustesSimilarity",
    "GWSimilarity",
    "NNGSSimilarityCPU",
    "NNGSSimilarityFaiss",
    "NNGSSimilarityTorch",
    "PWCCASimilarity",
]

available_metrics = {
    "cka": CKASimilarity,
    "gulp": GULPSimilarity,
    "procrustes": ProcrustesSimilarity,
    "gw_sim": GWSimilarity,
    "nngs_cpu": NNGSSimilarityCPU,
    "nngs_faiss": NNGSSimilarityFaiss,
    "nngs_torch": NNGSSimilarityTorch,
    "pwcca": PWCCASimilarity,
}


def compute_similarity(X, Y, method='cka', **kwargs):
    """
    Calculate similarity between two point clouds using the specified method.

    Parameters:
    - X: np.ndarray, first point cloud
    - Y: np.ndarray, second point cloud
    - method: str, similarity method to use ('cka', 'gulp', 'procrustes', 'gw_sim', 'nngs', 'pwcca')
    - kwargs: additional keyword arguments for the similarity class

    Returns:
    - similarity: float, computed similarity score
    """
    if method in available_metrics:
        sim_class = available_metrics[method]
        sim = sim_class(**kwargs)
    else:
        raise ValueError(f"Unknown similarity method: {method}. Available methods are: {list(available_metrics.keys())}")

    return sim(X, Y)
