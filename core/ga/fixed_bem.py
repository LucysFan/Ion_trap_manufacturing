"""Fixed-mesh rectangular-panel BEM with automatic CPU/CUDA selection.

The constructor accepts either:
- rectangular panels with shape (N, 4): x1, x2, y1, y2
- legacy panel centers with shape (N, 2): xc, yc

If legacy centers are passed, square panels are synthesized automatically using
`legacy_cell_size_m`. This keeps older mesh builders usable while the solver
internally operates on true rectangular panels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

EPS0 = 8.8541878128e-12
KE = 1.0 / (4.0 * np.pi * EPS0)
BackendName = Literal["auto", "cpu", "cuda"]


class _NumpyNamespace:
    """NumPy proxy exposing the small CuPy-compatible API used by the GA."""

    def __getattr__(self, name: str) -> Any:
        return getattr(np, name)

    @staticmethod
    def asnumpy(value: Any) -> np.ndarray:
        return np.asarray(value)


@dataclass(frozen=True)
class BackendStatus:
    requested: str
    selected: str
    available: bool
    reason: str
    device_name: str
    compute_capability: str | None = None
    total_memory_gib: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "selected": self.selected,
            "available": self.available,
            "reason": self.reason,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "total_memory_gib": self.total_memory_gib,
        }


def _probe_cuda() -> tuple[bool, str, Any | None, Any | None]:
    try:
        import cupy as cp
        import cupyx.scipy.linalg as cpx_linalg
    except (ImportError, ModuleNotFoundError) as exc:
        return False, f"CuPy is not installed: {exc}", None, None
    except Exception as exc:
        return False, f"CuPy import failed: {type(exc).__name__}: {exc}", None, None

    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        if count < 1:
            return False, "CuPy is installed, but no CUDA device is visible", None, None
        with cp.cuda.Device(0):
            cp.asarray([0.0], dtype=cp.float64)
            cp.cuda.runtime.deviceSynchronize()
    except Exception as exc:
        return False, f"CUDA runtime is unavailable: {type(exc).__name__}: {exc}", None, None

    return True, "CUDA is available", cp, cpx_linalg


def available_backend(requested: BackendName = "auto") -> BackendStatus:
    requested = str(requested).lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("backend must be one of: auto, cpu, cuda")

    if requested == "cpu":
        return BackendStatus(
            requested="cpu",
            selected="cpu",
            available=True,
            reason="CPU backend was requested explicitly",
            device_name="CPU (NumPy/SciPy)",
        )

    ok, reason, cp, _ = _probe_cuda()
    if ok:
        assert cp is not None
        device = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        return BackendStatus(
            requested=requested,
            selected="cuda",
            available=True,
            reason=reason,
            device_name=str(name),
            compute_capability=f"{props['major']}.{props['minor']}",
            total_memory_gib=float(props["totalGlobalMem"]) / 2**30,
        )

    if requested == "cuda":
        return BackendStatus(
            requested="cuda",
            selected="cuda",
            available=False,
            reason=reason,
            device_name="unavailable",
        )

    return BackendStatus(
        requested="auto",
        selected="cpu",
        available=True,
        reason=f"CUDA was not selected; falling back to CPU. {reason}",
        device_name="CPU (NumPy/SciPy)",
    )


def _coerce_panels(
    panels: np.ndarray,
    *,
    legacy_cell_size_m: float,
) -> np.ndarray:
    values = np.asarray(panels, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("panels must be a 2D array")

    if values.shape[1] == 4:
        rect = values
    elif values.shape[1] == 2:
        half = 0.5 * float(legacy_cell_size_m)
        rect = np.column_stack(
            [
                values[:, 0] - half,
                values[:, 0] + half,
                values[:, 1] - half,
                values[:, 1] + half,
            ]
        )
    else:
        raise ValueError(
            "panels must have shape (N, 4): x1,x2,y1,y2 "
            "or legacy centers shape (N, 2): xc,yc"
        )

    if rect.shape[0] == 0:
        raise ValueError("panels must not be empty")
    if np.any(rect[:, 1] <= rect[:, 0]):
        raise ValueError("each panel must satisfy x2 > x1")
    if np.any(rect[:, 3] <= rect[:, 2]):
        raise ValueError("each panel must satisfy y2 > y1")

    return rect


class FixedMeshBEM:
    """Dense fixed-mesh BEM operating on NumPy/SciPy or CuPy."""

    def __init__(
        self,
        panels: np.ndarray,
        *,
        backend: BackendName = "auto",
        z_assemble_m: float = 1e-9,
        dtype: str = "float64",
        legacy_cell_size_m: float = 8e-6,
    ) -> None:
        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")

        self.status = available_backend(backend)
        if not self.status.available:
            raise RuntimeError(
                "CUDA backend was requested but is unavailable. "
                f"{self.status.reason}. Use --backend auto or --backend cpu."
            )

        self.backend_name = self.status.selected
        self.panels_cpu = _coerce_panels(
            panels,
            legacy_cell_size_m=float(legacy_cell_size_m),
        )

        if self.backend_name == "cuda":
            ok, reason, xp, linear_algebra = _probe_cuda()
            if not ok or xp is None or linear_algebra is None:
                raise RuntimeError(f"CUDA became unavailable: {reason}")
            self.xp = xp
            self.cp = xp
            self._linalg = linear_algebra
        else:
            import scipy.linalg as scipy_linalg

            self.xp = _NumpyNamespace()
            self.cp = self.xp
            self._linalg = scipy_linalg

        self.dtype = self.xp.float64 if dtype == "float64" else self.xp.float32
        self.panels = self.xp.asarray(self.panels_cpu, dtype=self.dtype)
        self.n = int(self.panels_cpu.shape[0])
        self.z_assemble_m = float(z_assemble_m)
        self.matrix = None
        self.lu = None
        self.piv = None

    @property
    def backend_info(self) -> BackendStatus:
        return self.status

    @property
    def cuda_info(self) -> BackendStatus:
        return self.status

    @property
    def x_centers_m(self) -> np.ndarray:
        return 0.5 * (self.panels_cpu[:, 0] + self.panels_cpu[:, 1])

    @property
    def y_centers_m(self) -> np.ndarray:
        return 0.5 * (self.panels_cpu[:, 2] + self.panels_cpu[:, 3])

    @property
    def panel_centers_m(self) -> np.ndarray:
        return np.column_stack([self.x_centers_m, self.y_centers_m])

    def asnumpy(self, value: Any) -> np.ndarray:
        if self.backend_name == "cuda":
            return self.xp.asnumpy(value)
        return np.asarray(value)

    def synchronize(self) -> None:
        if self.backend_name == "cuda":
            self.xp.cuda.runtime.deviceSynchronize()

    def _antideriv(self, x, y, z):
        xp = self.xp
        a = xp.sqrt(x * x + z * z)
        b = xp.sqrt(y * y + z * z)
        sa = xp.where(a > 1e-300, a, 1.0)
        sb = xp.where(b > 1e-300, b, 1.0)
        t1 = xp.where(a > 1e-300, x * xp.arcsinh(y / sa), 0.0)
        t2 = xp.where(b > 1e-300, y * xp.arcsinh(x / sb), 0.0)
        radius = xp.sqrt(x * x + y * y + z * z)
        t3 = -z * xp.arctan2(x * y, z * radius)
        return t1 + t2 + t3

    def _rect_integral(self, x, y, z):
        p = self.panels
        x1, x2, y1, y2 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        return (
            self._antideriv(x - x1, y - y1, z)
            - self._antideriv(x - x1, y - y2, z)
            - self._antideriv(x - x2, y - y1, z)
            + self._antideriv(x - x2, y - y2, z)
        )

    def assemble(self, *, block_rows: int = 128) -> None:
        if block_rows < 1:
            raise ValueError("block_rows must be positive")
        xp = self.xp
        p = self.panels
        cx = 0.5 * (p[:, 0] + p[:, 1])
        cy = 0.5 * (p[:, 2] + p[:, 3])
        matrix = xp.empty((self.n, self.n), dtype=self.dtype)
        z = self.dtype(self.z_assemble_m)

        for start in range(0, self.n, block_rows):
            stop = min(start + block_rows, self.n)
            matrix[start:stop] = KE * self._rect_integral(
                cx[start:stop, None],
                cy[start:stop, None],
                z,
            )

        width = p[:, 1] - p[:, 0]
        height = p[:, 3] - p[:, 2]
        self_term = 2.0 * width * xp.arcsinh(height / width)
        self_term += 2.0 * height * xp.arcsinh(width / height)
        indices = xp.arange(self.n)
        matrix[indices, indices] = KE * self_term
        self.matrix = matrix

    def factorize(self) -> None:
        if self.matrix is None:
            self.assemble()
        self.lu, self.piv = self._linalg.lu_factor(
            self.matrix,
            overwrite_a=False,
            check_finite=False,
        )

    def solve_masks(self, masks: np.ndarray):
        values = np.asarray(masks, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2:
            raise ValueError("masks must have shape (batch, N) or (N,)")
        if values.shape[1] != self.n:
            raise ValueError(f"mask width {values.shape[1]} != panel count {self.n}")
        if self.lu is None:
            self.factorize()
        rhs = self.xp.asarray(values.T, dtype=self.dtype)
        sigma = self._linalg.lu_solve(
            (self.lu, self.piv),
            rhs,
            check_finite=False,
        )
        return sigma.T

    def _field_kernels(self, x, y, z):
        xp = self.xp
        p = self.panels
        x1, x2 = p[:, 0], p[:, 1]
        y1, y2 = p[:, 2], p[:, 3]
        u1, u2 = x - x2, x - x1
        v1, v2 = y - y2, y - y1

        def fx(u, v):
            return -xp.arcsinh(v / xp.sqrt(u * u + z * z))

        def fy(u, v):
            return -xp.arcsinh(u / xp.sqrt(v * v + z * z))

        gx = fx(u2, v2) - fx(u2, v1) - fx(u1, v2) + fx(u1, v1)
        gy = fy(u2, v2) - fy(u2, v1) - fy(u1, v2) + fy(u1, v1)

        def adz(u, v):
            radius = xp.sqrt(u * u + v * v + z * z)
            s1 = (u * u + z * z) * radius
            s2 = (v * v + z * z) * radius
            s3 = radius * (u * u + z * z) * (v * v + z * z)
            return (
                -u * v * z / s1
                - u * v * z / s2
                - xp.arctan2(u * v, z * radius)
                + z * u * v * (u * u + v * v + 2.0 * z * z) / s3
            )

        gz = -(
            adz(u2, v2)
            - adz(u2, v1)
            - adz(u1, v2)
            + adz(u1, v1)
        )
        return gx, gy, gz

    def field_batch(self, x, y, z, sigma, *, candidate_chunk: int = 16):
        if candidate_chunk < 1:
            raise ValueError("candidate_chunk must be positive")
        xp = self.xp
        x = xp.asarray(x, dtype=self.dtype)
        y = xp.asarray(y, dtype=self.dtype)
        z = xp.asarray(z, dtype=self.dtype)
        sigma = xp.asarray(sigma, dtype=self.dtype)

        if sigma.ndim == 1:
            sigma = sigma[None, :]
        if x.ndim == 1:
            x = xp.broadcast_to(x[None, :], (sigma.shape[0], x.size))
        if y.ndim == 1:
            y = xp.broadcast_to(y[None, :], x.shape)
        if z.ndim == 1:
            z = xp.broadcast_to(z[None, :], x.shape)

        if x.shape != y.shape or x.shape != z.shape:
            raise ValueError("x, y, z must have equal/broadcastable shapes")
        if x.shape[0] != sigma.shape[0]:
            raise ValueError("point batch and sigma batch differ")
        if sigma.shape[1] != self.n:
            raise ValueError("sigma width and panel count differ")

        outputs = []
        for start in range(0, int(sigma.shape[0]), candidate_chunk):
            stop = min(start + candidate_chunk, int(sigma.shape[0]))
            xx = x[start:stop, :, None]
            yy = y[start:stop, :, None]
            zz = z[start:stop, :, None]
            gx, gy, gz = self._field_kernels(xx, yy, zz)
            charge = sigma[start:stop, None, :]
            ex = KE * xp.sum(charge * gx, axis=2)
            ey = KE * xp.sum(charge * gy, axis=2)
            ez = KE * xp.sum(charge * gz, axis=2)
            outputs.append(xp.stack((ex, ey, ez), axis=2))
        return xp.concatenate(outputs, axis=0)

    def release_matrix(self) -> None:
        if self.lu is None:
            raise RuntimeError("factorize before releasing the matrix")
        self.matrix = None
        if self.backend_name == "cuda":
            self.xp.get_default_memory_pool().free_all_blocks()


CudaFixedMeshBEM = FixedMeshBEM

__all__ = [
    "BackendStatus",
    "BackendName",
    "FixedMeshBEM",
    "CudaFixedMeshBEM",
    "available_backend",
]