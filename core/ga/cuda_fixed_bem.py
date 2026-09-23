"""Compatibility import for projects using the former CUDA-only module."""
from .fixed_bem import (
    BackendStatus,
    CudaFixedMeshBEM,
    FixedMeshBEM,
    available_backend,
)

__all__ = [
    "BackendStatus",
    "CudaFixedMeshBEM",
    "FixedMeshBEM",
    "available_backend",
]
