"""
Point clouds to compare: synthetic data with known geometric relations, and
loaders for real data.

The loaders import their dependencies (Hugging Face datasets, ...) only when
called; install them with `uv sync --extra data`.
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

__all__ = [
    'equidistant_blobs',
    'jittered_blobs',
    'load_text_dataset',
    'mix_with_noise',
    'noise_pair',
    'power_distortion',
    'shuffled_blobs',
    'simplex_centers',
]
