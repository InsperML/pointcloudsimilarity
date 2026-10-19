"""
Point clouds to compare: synthetic data with known geometric relations.
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

__all__ = [
    'equidistant_blobs',
    'jittered_blobs',
    'mix_with_noise',
    'noise_pair',
    'power_distortion',
    'shuffled_blobs',
    'simplex_centers',
]
