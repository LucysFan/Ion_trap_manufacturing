"""test_one_seed.py — прогнать один seed через конвейер 09e с диагностикой.

Запуск (из корня проекта):
    XJ_MAX_SLOPE=1.20 XJ_MAX_AMPLITUDE_M=45e-6 XJ_MIN_FEATURE_M=18e-6 \
        python test_one_seed.py --source legacy --file legacy_abc.json --name A

    XJ_MAX_SLOPE=1.20 XJ_MAX_AMPLITUDE_M=45e-6 XJ_MIN_FEATURE_M=18e-6 \
        python test_one_seed.py --source builtin --name baseline

    XJ_MAX_SLOPE=1.20 XJ_MAX_AMPLITUDE_M=45e-6 XJ_MIN_FEATURE_M=18e-6 \
        python test_one_seed.py --source genome --file g43.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    # запущено в Jupyter/IPython — ищем корень проекта от cwd
    ROOT = Path.cwd().resolve()
    # если мы в подпапке (workflows/, notebooks/ и т.п.) — подняться до корня
    for parent in [ROOT, *ROOT.parents]:
        if (parent / "config").is_dir() and (parent / "core").is_dir():
            ROOT = parent
            break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("ROOT =", ROOT)
# ---------------------------------------------------------------------------
# Диагностика окружения
# ---------------------------------------------------------------------------
print("=" * 78)
print("ENV")
print("=" * 78)
for k in ("XJ_MAX_SLOPE", "XJ_MAX_AMPLITUDE_M", "XJ_MIN_FEATURE_M"):
    print(f"  {k:20s} = {os.environ.get(k, '(unset -> hardcoded default)')}")

# ---------------------------------------------------------------------------
# Импорт workflow. Печатаем реальные значения модуля после патча.
# ---------------------------------------------------------------------------
from workflow_09e import (                                  # noqa: E402
    N_GENES, N_GENES_OLD, LOW, HIGH, LOCK, GAPS, INNER, OUTER,
    CENTER_IN, CENTER_OUT, START, LENGTH, POWER,
    BULGE, BULGE_CENTER, BULGE_WIDTH,
    MIN_FEATURE_M, MIN_GAP_M, Z_ION_M,
    FIXED_CORE_KNOTS_M, LOCK_MIN_M, LOCK_MAX_M,
    Settings, baseline, repair, decode, parameters_from_genome,
    build_mesh_adaptive, evaluate,
    lift_31_to_43, seed_list,
)

print()
print("=" * 78)
print("WORKFLOW CONSTANTS")
print("=" * 78)
print(f"  N_GENES           = {N_GENES}")
print(f"  N_GENES_OLD       = {N_GENES_OLD}")
print(f"  MIN_FEATURE_M     = {MIN_FEATURE_M*1e6:.2f} um")
print(f"  MIN_GAP_M         = {MIN_GAP_M*1e6:.2f} um")
print(f"  Z_ION_M           = {Z_ION_M*1e6:.2f} um")
print(f"  LOCK range        = [{LOCK_MIN_M*1e6:.0f}, {LOCK_MAX_M*1e6:.0f}] um")
print(f"  FIXED_CORE_KNOTS  = {[round(k*1e6,1) for k in FIXED_CORE_KNOTS_M]}")

# ---------------------------------------------------------------------------
# Найти patched-константы в junction_templates
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("junction_templates.py runtime limits")
print("=" * 78)
try:
    import core.geometry.junction_templates as jt
    found = []
    for name in dir(jt):
        if name.startswith("_"):
            continue
        v = getattr(jt, name)
        if isinstance(v, (int, float)) and (
            "slope" in name.lower() or "amplitude" in name.lower() or
            "slope" in str(name).lower() or "amp" in name.lower()
        ):
            found.append((name, v))
    # Явные env-константы, если патч применён
    for name in ("_MAX_SLOPE", "_MAX_AMPLITUDE_M"):
        if hasattr(jt, name):
            found.append((name, getattr(jt, name)))
    for name, v in found:
        print(f"  {name:24s} = {v}")
    if not found:
        print("  (не удалось найти; смотрите через grep по junction_templates.py)")
except Exception as err:
    print(f"  import failed: {err}")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=("builtin", "legacy", "genome"),
                   default="builtin")
    p.add_argument("--name", default="baseline",
                   help="для builtin: имя seed из seed_list(); "
                        "для legacy: ключ в JSON")
    p.add_argument("--file", type=Path, default=None,
                   help="JSON для --source legacy или --source genome")
    p.add_argument("--stage", choices=("coarse", "fine"), default="coarse")
    p.add_argument("--max-panels", type=int, default=6000)
    p.add_argument("--coarse-points", type=int, default=61)
    p.add_argument("--fine-points", type=int, default=181)
    p.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Получить геном
# ---------------------------------------------------------------------------
def load_genome(args) -> tuple[np.ndarray, str]:
    if args.source == "genome":
        if args.file is None:
            raise SystemExit("--source genome требует --file")
        arr = np.asarray(json.loads(Path(args.file).read_text()), dtype=float)
        if arr.shape == (N_GENES_OLD,):
            print(f"[genome] {N_GENES_OLD}-gene -> lifting to {N_GENES}")
            arr = lift_31_to_43(arr)
        elif arr.shape != (N_GENES,):
            raise SystemExit(f"genome must be {N_GENES_OLD} or {N_GENES}, got {arr.shape}")
        return repair(arr), "genome"

    if args.source == "legacy":
        if args.file is None:
            raise SystemExit("--source legacy требует --file")
        data = json.loads(Path(args.file).read_text())
        if args.name not in data:
            raise SystemExit(f"'{args.name}' not in {args.file} (keys: {list(data)})")
        raw = np.asarray(data[args.name], dtype=float)
        if raw.shape != (N_GENES_OLD,):
            raise SystemExit(f"legacy genome must be {N_GENES_OLD}, got {raw.shape}")
        print(f"[legacy] {args.name}: {N_GENES_OLD}-gene -> lifting to {N_GENES}")
        return repair(lift_31_to_43(raw)), f"legacy_{args.name}"

    # builtin
    seeds = dict(seed_list())
    if args.name not in seeds:
        raise SystemExit(f"'{args.name}' not in seed_list() "
                         f"(keys: {list(seeds)})")
    return repair(seeds[args.name]), args.name


# ---------------------------------------------------------------------------
# Пошаговая диагностика
# ---------------------------------------------------------------------------
def print_genome(g: np.ndarray) -> None:
    print()
    print("=" * 78)
    print("GENOME")
    print("=" * 78)
    print(f"  shape  = {g.shape}")
    print(f"  finite = {bool(np.all(np.isfinite(g)))}")
    print(f"  within bounds = {bool(np.all((g >= LOW) & (g <= HIGH)))}")
    if not np.all((g >= LOW) & (g <= HIGH)):
        bad = np.where(~((g >= LOW) & (g <= HIGH)))[0]
        print(f"  out-of-bounds indices: {bad.tolist()}")


def print_decoded(g: np.ndarray) -> None:
    knots, inner, outer = decode(g)
    print()
    print("=" * 78)
    print("DECODED")
    print("=" * 78)
    print(f"  lock_m = {g[LOCK]*1e6:.2f} um")
    print(f"  knots (um)  = {[round(k*1e6,2) for k in knots]}")
    print(f"  inner (um)  = {[round(v*1e6,2) for v in inner]}")
    print(f"  outer (um)  = {[round(v*1e6,2) for v in outer]}")
    print(f"  center_in   = {g[CENTER_IN]*1e6:+.2f} um")
    print(f"  center_out  = {g[CENTER_OUT]*1e6:+.2f} um")
    print(f"  start       = {g[START]*1e6:+.2f} um")
    print(f"  length      = {g[LENGTH]*1e6:+.2f} um")
    print(f"  power       = {g[POWER]:+.3f}")
    print(f"  bulge       = {g[BULGE]*1e6:+.2f} um")
    print(f"  bulge_c     = {g[BULGE_CENTER]*1e6:+.2f} um")
    print(f"  bulge_w     = {g[BULGE_WIDTH]*1e6:+.2f} um")

    # slope / amplitude инспекция (та же логика, что в junction_templates)
    dk = np.diff(knots)
    dk[dk == 0] = 1e-12
    inner_slope = np.abs(np.diff(inner)) / dk
    outer_slope = np.abs(np.diff(outer)) / dk
    print()
    print(f"  inner max|offset| = {np.max(np.abs(inner))*1e6:.2f} um")
    print(f"  outer max|offset| = {np.max(np.abs(outer))*1e6:.2f} um")
    print(f"  inner max|slope|  = {inner_slope.max():.3f}")
    print(f"  outer max|slope|  = {outer_slope.max():.3f}")


def print_parameters(g: np.ndarray) -> None:
    print()
    print("=" * 78)
    print("PARAMETERS (movable-knot)")
    print("=" * 78)
    p = parameters_from_genome(g)
    s = np.linspace(0.0, p.arm_length_m, 2001)
    rin, rout = p.rail_boundaries_m(s)
    print(f"  arm_length   = {p.arm_length_m*1e6:.2f} um")
    print(f"  min(rin)     = {rin.min()*1e6:.3f} um   (need >  {0.5*MIN_FEATURE_M*1e6:.1f})")
    print(f"  min(rout-rin)= {(rout-rin).min()*1e6:.3f} um   (need >= {MIN_FEATURE_M*1e6:.1f})")
    print(f"  min(rout)    = {rout.min()*1e6:.3f} um")
    print(f"  max(rout)    = {rout.max()*1e6:.3f} um")
    print()
    print(f"  START+CENTER_IN = {(g[START]+g[CENTER_IN])*1e6:+.3f} um  "
          f"(this is roughly the RF inner edge radius)")


def print_manufacturability(g: np.ndarray) -> bool:
    print()
    print("=" * 78)
    print("MANUFACTURABILITY (check_x_junction_manufacturability)")
    print("=" * 78)
    from core.geometry.manufacturability import check_x_junction_manufacturability
    p = parameters_from_genome(g)
    rep = check_x_junction_manufacturability(p, min_feature_size_m=MIN_FEATURE_M)
    print(f"  valid = {rep.valid}")
    for m in rep.messages:
        print(f"    msg: {m}")
    return bool(rep.valid)


def print_mesh(g: np.ndarray, max_panels: int) -> None:
    print()
    print("=" * 78)
    print("MESH (build_mesh_adaptive, coarse)")
    print("=" * 78)
    p = parameters_from_genome(g)
    cfg = _make_cfg(max_panels)

    # Пройтись по всей лестнице и напечатать размер каждой попытки
    lock_m = float(g[LOCK])
    central_half = max(lock_m + 30e-6, 200e-6)
    if cfg.fast_coarse:
        ladder = [
            dict(central_half_extent_m=central_half, central_max_cell_m=42e-6,
                 boundary_max_cell_m=16e-6, outer_max_cell_m=240e-6, min_cell_m=8e-6),
            dict(central_half_extent_m=central_half, central_max_cell_m=55e-6,
                 boundary_max_cell_m=22e-6, outer_max_cell_m=280e-6, min_cell_m=10e-6),
            dict(central_half_extent_m=central_half, central_max_cell_m=75e-6,
                 boundary_max_cell_m=30e-6, outer_max_cell_m=320e-6, min_cell_m=13e-6),
        ]
    else:
        ladder = [
            dict(central_half_extent_m=central_half, central_max_cell_m=24e-6,
                 boundary_max_cell_m=9e-6, outer_max_cell_m=260e-6, min_cell_m=4e-6),
            dict(central_half_extent_m=central_half, central_max_cell_m=32e-6,
                 boundary_max_cell_m=12e-6, outer_max_cell_m=280e-6, min_cell_m=6e-6),
            dict(central_half_extent_m=central_half, central_max_cell_m=45e-6,
                 boundary_max_cell_m=18e-6, outer_max_cell_m=300e-6, min_cell_m=8e-6),
            dict(central_half_extent_m=central_half, central_max_cell_m=60e-6,
                 boundary_max_cell_m=25e-6, outer_max_cell_m=340e-6, min_cell_m=10e-6),
        ]

    print(f"  central_half_extent = {central_half*1e6:.1f} um")
    from core.geometry.mask_builder import build_geometry_aware_quadtree_x_junction_bem
    for level, params in enumerate(ladder):
        t = time.time()
        try:
            model = build_geometry_aware_quadtree_x_junction_bem(p, **params)
            n = int(model.n_panels)
            dt = (time.time() - t) * 1e3
            tag = "OK" if n <= max_panels else "OVER"
            print(f"  level {level}: panels = {n:6d}  ({tag}, {dt:.0f} ms)")
            if n <= max_panels:
                print(f"  -> chosen level = {level}")
                return
        except Exception as err:
            print(f"  level {level}: FAILED {type(err).__name__}: {err}")


def _make_cfg(max_panels: int) -> Settings:
    return Settings(
        backend="auto",
        stage1_launches=1, stage1_population=1, stage1_generations=1,
        stage1_offspring=0, stage1_top_per_launch=1,
        stage1_migration_interval=1, stage1_migrants=0,
        stage1_stall_generations=1, stage1_burst_generations=1,
        stage1_burst_multiplier=1.0,
        stage2_population=1, stage2_generations=1, stage2_offspring=0,
        stage2_random_fraction=0.0, stage2_migration_interval=1,
        stage2_migrants=0, stage2_stall_generations=1,
        stage2_burst_generations=1, stage2_burst_multiplier=1.0,
        coarse_points=61, fine_points=181, fine_top=1,
        max_panels=max_panels, fast_coarse=True,
        seed=0, diversity_threshold=0.10,
        animation_stride=1, surface_points_2d=21, surface_points_3d=11,
        output_dir=Path("/tmp/09e_one_seed"),
    )


def print_evaluate(g: np.ndarray, args) -> dict:
    print()
    print("=" * 78)
    print(f"EVALUATE (stage={args.stage})")
    print("=" * 78)
    from core.ga.fixed_bem import available_backend
    bk = available_backend(args.backend)
    print(f"  backend request = {args.backend}, available = {bk.available}, selected = {bk.selected}")
    if not bk.available:
        print(f"  backend unavailable: {bk.reason}")
        return {}

    cfg = _make_cfg(args.max_panels)
    # cfg is frozen; override via replace
    from dataclasses import replace
    cfg = replace(cfg, backend=bk.selected,
                  coarse_points=args.coarse_points,
                  fine_points=args.fine_points)

    t = time.time()
    result, data = evaluate(g, cfg, stage=args.stage, retain=True)
    dt = time.time() - t
    print(f"  wall time = {dt:.2f} s")
    print()
    keys = ["valid", "verified", "reason", "stage", "backend",
            "dz_peak_m", "dz_rms_m", "barrier_ev", "lateral_peak_m",
            "barrier_reference_ev", "panels", "points",
            "converged", "n_minima", "mesh_level"]
    for k in keys:
        v = result.get(k, "<missing>")
        if isinstance(v, float):
            print(f"  {k:22s} = {v:.6g}")
        else:
            print(f"  {k:22s} = {v}")
    if data is not None:
        model, trace, pseudo_ev, field = data
        print()
        print("  extra diagnostics:")
        z = np.asarray(trace.z_m)
        y = np.asarray(trace.y_m)
        dz = z - Z_ION_M
        print(f"    trace.x  range = [{trace.x_m[0]*1e6:+.1f}, {trace.x_m[-1]*1e6:+.1f}] um")
        print(f"    dz min/max     = {dz.min()*1e6:+.3f} / {dz.max()*1e6:+.3f} um")
        print(f"    y  min/max     = {y.min()*1e6:+.3f} / {y.max()*1e6:+.3f} um")
        print(f"    U_ps min/max   = {pseudo_ev.min()*1e3:.4f} / {pseudo_ev.max()*1e3:.4f} meV")
        print(f"    U_ps span      = {(pseudo_ev.max()-pseudo_ev.min())*1e3:.4f} meV")
    return result


# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    g, label = load_genome(args)
    print()
    print("#" * 78)
    print(f"# SEED: {label}  ({g.shape[0]} genes)")
    print("#" * 78)

    print_genome(g)
    print_decoded(g)
    print_parameters(g)

    ok_mfg = print_manufacturability(g)
    if not ok_mfg:
        print()
        print("!!! manufacturability FAILED -> evaluate() will short-circuit with")
        print("    'geometry: ...' reason. Fix the geometry or lower MIN_FEATURE_M.")
        # всё равно пойдём дальше — интересно посмотреть весь путь
    print_mesh(g, args.max_panels)

    res = print_evaluate(g, args)

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    if not res:
        print("  evaluate() did not return (backend unavailable?)")
    elif res.get("valid"):
        print(f"  VALID  ({res.get('stage')})")
        print(f"  dz_peak = {res['dz_peak_m']*1e6:.3f} um  "
              f"U = {res['barrier_ev']*1e3:.3f} meV  "
              f"lat = {res['lateral_peak_m']*1e6:.3f} um")
        if res.get("verified"):
            print("  VERIFIED (fine, within all limits)")
    else:
        print(f"  INVALID: {res.get('reason')}")


if __name__ == "__main__":
    main()