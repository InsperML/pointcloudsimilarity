from .cka import CKASimilarity
from .gulp import GULPSimilarity
from .gw_sim import GWSimilarity
from .procrustes import ProcrustesSimilarity
from .pwcca import PWCCASimilarity
from .rtd import RTDSimilarity
from .tas import (
    TASSimilarityCPU,
    TASSimilarityFaiss,
    TASSimilarityTorch,
    tas_spectrum,
)

__all__ = [
    "CKASimilarity",
    "GULPSimilarity",
    "GWSimilarity",
    "PWCCASimilarity",
    "ProcrustesSimilarity",
    "RTDSimilarity",
    "TASSimilarityCPU",
    "TASSimilarityFaiss",
    "TASSimilarityTorch",
    "tas_spectrum",
]

available_metrics = {
    "cka": CKASimilarity,
    "gulp": GULPSimilarity,
    "procrustes": ProcrustesSimilarity,
    "gw_sim": GWSimilarity,
    "tas_cpu": TASSimilarityCPU,
    "tas_faiss": TASSimilarityFaiss,
    "tas_torch": TASSimilarityTorch,
    "pwcca": PWCCASimilarity,
    "rtd": RTDSimilarity,
    # Pre-rename aliases, kept for existing scripts.
    "nngs_cpu": TASSimilarityCPU,
    "nngs_faiss": TASSimilarityFaiss,
    "nngs_torch": TASSimilarityTorch,
}


def compute_similarity(X, Y, method='cka', **kwargs):
    """
    Calculate similarity between two point clouds using the specified method.

    Parameters:
    - X: np.ndarray, first point cloud
    - Y: np.ndarray, second point cloud
    - method: str, similarity method to use (a key of available_metrics, e.g.
      'cka', 'gulp', 'procrustes', 'gw_sim', 'tas_torch', 'tas_cpu', 'tas_faiss',
      'pwcca', 'rtd')
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
