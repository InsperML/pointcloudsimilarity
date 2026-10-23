from importlib.metadata import version

from .similarities import (
    CKASimilarity,
    GULPSimilarity,
    GWSimilarity,
    PWCCASimilarity,
    ProcrustesSimilarity,
    TASSimilarityCPU,
    TASSimilarityTorch,
    available_metrics,
    compute_similarity,
    tas_spectrum,
)

__version__ = version('pointcloudsimilarity')

__all__ = [
    'CKASimilarity',
    'GULPSimilarity',
    'GWSimilarity',
    'PWCCASimilarity',
    'ProcrustesSimilarity',
    'TASSimilarityCPU',
    'TASSimilarityTorch',
    '__version__',
    'available_metrics',
    'compute_similarity',
    'tas_spectrum',
]
