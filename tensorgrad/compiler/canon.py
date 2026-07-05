"""Compositional structural hashing — thin API shim.

The implementation lives in :mod:`tensorgrad.structure`: every node class
declares its structural identity once (``Tensor.structure()``) and generic
folds derive both the networkx structural graph and these fingerprints from
it.  This module re-exports the fingerprint API under its historical import
path (tensor.py, the compiler and tests import from here).

See tensorgrad/structure.py for the contracts (invariant coarse hash, sound
refined fingerprint, (I1)/(I2) edge colors).
"""

from tensorgrad.structure import (
    CanonInfo,
    canon_info,
    refined_sort_key,
    structural_fingerprint,
    structural_hash,
)

__all__ = [
    "CanonInfo",
    "canon_info",
    "refined_sort_key",
    "structural_fingerprint",
    "structural_hash",
]
