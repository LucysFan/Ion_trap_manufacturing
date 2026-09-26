"""core/ga_v2/__init__.py

Optimized BEM and mesh helpers for GA workflows.

This package is a drop-in speedup of core.ga.fixed_bem. Public API is
identical; only the internal implementation has changed.

To use, change the import in workflow 11 (or any other workflow):

    # before
    from core.ga.fixed_bem import FixedMeshBEM, available_backend

    # after
    from core.ga_v2.fixed_bem import FixedMeshBEM, available_backend
"""

from __future__ import annotations

from . import fixed_bem

__all__ = ["fixed_bem"]