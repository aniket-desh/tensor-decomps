"""
tensor generation module.
provides utilities for creating synthetic and loading real-world tensors.
"""

from . import synthetic_tensors
from . import real_tensors

__all__ = [
    'synthetic_tensors',
    'real_tensors',
]

