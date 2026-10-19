"""
Point clouds to compare: synthetic data with known geometric relations, and
loaders for real data.

The loaders import their dependencies (Hugging Face datasets, gensim) only
when called; install them with `uv sync --extra data`.
"""

from .synthetic import (
    equidistant_blobs,
    jittered_blobs,
    mix_with_noise,
    noise_pair,
    power_distortion,
    shuffled_blobs,
    simplex_centers,
)
from .text import load_text_dataset
from .word_vectors import GLOVE_TWITTER_DIMS, aligned_glove_vectors, load_glove_twitter

__all__ = [
    'GLOVE_TWITTER_DIMS',
    'aligned_glove_vectors',
    'equidistant_blobs',
    'jittered_blobs',
    'load_glove_twitter',
    'load_text_dataset',
    'mix_with_noise',
    'noise_pair',
    'power_distortion',
    'shuffled_blobs',
    'simplex_centers',
]
