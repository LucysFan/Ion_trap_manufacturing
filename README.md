# trap_design

Robust design and control of planar Paul-trap X-junctions for a single 40Ca+ ion.

## Current implemented stage

- Analytic 2D straight surface-trap baseline
- Rectangular-panel dense BEM
- Symmetric X-junction geometry
- RF transverse-minimum tracing
- Volcano and RF pseudopotential proxy metrics
- Controlled geometry scans
- Edge-aligned BEM mesh validation

## Environment

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The main environment intentionally uses NumPy 1.26.4.