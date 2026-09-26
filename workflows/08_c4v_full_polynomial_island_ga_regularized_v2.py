"""Full-genome, regularized C4v open-centre X-junction island NSGA-II.

Features
--------
* Two independently variable-degree Chebyshev polynomials: inner and outer RF edge.
* 28 fixed-length genes: degrees, 18 coefficients, and 8 central/global controls.
* Open centre only: no central RF cross is inserted.
* Three Pareto objectives: peak RF-null dz, RF barrier, and contour roughness.
* Semantic block crossover; discrete degree mutations; no hidden-coefficient revival.
* Validity-first initialization distinct on every island.
* Per-generation top-N elite persistence, rejection diagnostics, and physical
  top-5 electrode-surface GIFs per island.
* Fine BEM validation and complete top-5 lithography design records.

Fine certification only: complete trace AND dz < 3 um AND barrier < 1 meV.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.targets import RF_ANGULAR_FREQUENCY_RAD_S, TARGET_ION_HEIGHT_M
from core.analysis.barrier import compute_barrier_metrics, pseudopotential_profile_ev
from core.analysis.rf_null_trace import trace_rf_transverse_minimum
from core.ga.fixed_bem import FixedMeshBEM, available_backend
from core.geometry.junction_templates import make_house_style_x_junction
from core.geometry.manufacturability import check_x_junction_manufacturability
from core.geometry.mask_builder import build_geometry_aware_quadtree_x_junction_bem

KNOTS_M = np.array([0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 210], dtype=float) * 1e-6
N_POLY = 9
MAX_DEGREE = N_POLY - 1
DEG_IN, DEG_OUT = 0, 1
INNER = slice(2, 2 + N_POLY)
OUTER = slice(2 + N_POLY, 2 + 2 * N_POLY)
CENTER_IN = 2 + 2 * N_POLY
CENTER_OUT = CENTER_IN + 1
START = CENTER_IN + 2
LENGTH = CENTER_IN + 3
POWER = CENTER_IN + 4
BULGE = CENTER_IN + 5
BULGE_CENTER = CENTER_IN + 6
BULGE_WIDTH = CENTER_IN + 7
N_GENES = CENTER_IN + 8

LOW = np.array([0, 0] + [-35e-6]*N_POLY + [-35e-6]*N_POLY +
               [-50e-6, -8e-6, 8e-6, 60e-6, .70, -30e-6, 10e-6, 8e-6], dtype=float)
HIGH = np.array([MAX_DEGREE, MAX_DEGREE] + [35e-6]*N_POLY + [35e-6]*N_POLY +
                [-3e-6, 65e-6, 55e-6, 310e-6, 5.25, 30e-6, 140e-6, 80e-6], dtype=float)


@dataclass(frozen=True)
class Settings:
    backend: str
    islands: int
    population: int
    offspring: int
    generations: int
    migration_interval: int
    migrants: int
    retain_per_island: int
    seed: int
    coarse_points: int
    fine_points: int
    fine_top: int
    max_panels: int
    fast: bool
    animation_stride: int
    output_dir: Path
    rf_peak_v: float = 100.0
    target_z_m: float = TARGET_ION_HEIGHT_M
    dz_limit_m: float = 3e-6
    barrier_limit_ev: float = 1e-3
    lateral_limit_m: float = 5e-6
    rail_min_width_m: float = 18e-6
    roughness_hard_limit: float = 18.0


@dataclass(frozen=True)
class IslandStyle:
    name: str
    weights: tuple[float, float, float]
    explore: float
    final: float
    emphasis: str


STYLES = (
    IslandStyle("height_inner", (7, 1, .4), 1.00, .10, "inner"),
    IslandStyle("barrier_outer", (1, 7, .4), 1.00, .10, "outer"),
    IslandStyle("balanced", (2, 2, 1), .90, .09, "global"),
    IslandStyle("central", (3, 2, .7), 1.05, .10, "central"),
    IslandStyle("high_order_inner", (4, 2, 1), 1.10, .11, "high_inner"),
    IslandStyle("high_order_outer", (2, 4, 1), 1.10, .11, "high_outer"),
    IslandStyle("degree", (2, 2, 1), 1.10, .11, "degree"),
    IslandStyle("wide", (2, 2, .8), 1.25, .14, "wide"),
    IslandStyle("joint", (2, 2, .8), .95, .08, "joint"),
    IslandStyle("diverse", (2, 2, 1), 1.00, .10, "diverse"),
    IslandStyle("local", (2, 2, 1), .42, .030, "local"),
    IslandStyle("taper", (2, 3, .8), 1.00, .10, "global"),
)


@dataclass
class Individual:
    genome: np.ndarray
    result: dict[str, Any]
    island: int
    generation: int
    origin: str
    rank: int = 0
    crowding: float = 0.0

    @property
    def valid(self) -> bool:
        return bool(self.result["valid"])

    @property
    def objectives(self) -> np.ndarray:
        if not self.valid:
            return np.array([np.inf, np.inf], dtype=float)

        return np.array(
            [
                self.result["dz_peak_m"] / 3e-6,
                self.result["barrier_ev"] / 1e-3,
            ],
            dtype=float,
        )

    def copy(self):
        return Individual(self.genome.copy(), dict(self.result), self.island,
                          self.generation, self.origin, self.rank, self.crowding)


@dataclass
class Island:
    style: IslandStyle
    rng: np.random.Generator
    population: list[Individual]


class BEMField:
    def __init__(self, bem: FixedMeshBEM, charge: Any):
        self.bem, self.charge = bem, charge

    def electric_field(self, x_m: float, y_m: float, z_m: float):
        out = self.bem.field_batch(np.array([x_m]), np.array([y_m]), np.array([z_m]), self.charge)
        return tuple(map(float, self.bem.asnumpy(out)[0, 0]))


def as_degree(value: float) -> int:
    return int(np.clip(np.rint(value), 0, MAX_DEGREE))


def active_coeffs(x: np.ndarray, block: slice, order: int) -> np.ndarray:
    c = np.asarray(x[block], float).copy()
    c[order+1:] = 0.0
    return c


def profile(coeffs: np.ndarray, s_m: np.ndarray) -> np.ndarray:
    u = 2.0*(np.asarray(s_m, float)-KNOTS_M[0])/(KNOTS_M[-1]-KNOTS_M[0])-1.0
    return np.polynomial.chebyshev.chebval(np.clip(u, -1.0, 1.0), coeffs)


def contour_offsets(x: np.ndarray):
    di = as_degree(x[DEG_IN])
    do = as_degree(x[DEG_OUT])

    inner = profile(active_coeffs(x, INNER, di), KNOTS_M)
    outer = profile(active_coeffs(x, OUTER, do), KNOTS_M)

    # Existing junction template requires exact zero contour offsets
    # where the contour is attached to its central and external boundaries.
    inner[0] = 0.0
    inner[-1] = 0.0

    outer[0] = 0.0
    outer[-1] = 0.0

    return inner, outer, di, do


def contour_roughness(x: np.ndarray) -> tuple[float, float, float]:
    """RMS slope/curvature, safe for degree 0 and 1 polynomials."""
    dense = np.linspace(KNOTS_M[0], KNOTS_M[-1], 401)
    u = 2.0*(dense-KNOTS_M[0])/(KNOTS_M[-1]-KNOTS_M[0])-1.0
    ds_du = 2.0/(KNOTS_M[-1]-KNOTS_M[0])
    slopes, curvatures = [], []
    for block, degree_index in ((INNER, DEG_IN), (OUTER, DEG_OUT)):
        order = as_degree(x[degree_index])
        coeffs = active_coeffs(x, block, order)
        if order >= 1:
            d1 = np.polynomial.chebyshev.chebval(u, np.polynomial.chebyshev.chebder(coeffs))*ds_du
        else:
            d1 = np.zeros_like(u)
        if order >= 2:
            d2 = np.polynomial.chebyshev.chebval(u, np.polynomial.chebyshev.chebder(coeffs, m=2))*ds_du**2
        else:
            d2 = np.zeros_like(u)
        slopes.append(d1); curvatures.append(d2)
    slope_rms = float(np.sqrt(np.mean(np.concatenate(slopes)**2)))
    curvature_rms = float(np.sqrt(np.mean(np.concatenate(curvatures)**2))*50e-6)
    return slope_rms/.60 + curvature_rms/1.50, slope_rms, curvature_rms


def enforce_template_slope(
    coefficients: np.ndarray,
    active_degree: int,
    max_slope: float = 0.78,
) -> np.ndarray:
    """Project one active Chebyshev contour onto template-compatible slopes.

    The template rejects slopes above 0.80. We use 0.78 as a margin.
    """
    values = coefficients.copy()

    if active_degree == 0:
        return values

    samples_s = np.linspace(KNOTS_M[0], KNOTS_M[-1], 401)
    samples_u = (
        2.0 * (samples_s - KNOTS_M[0])
        / (KNOTS_M[-1] - KNOTS_M[0])
        - 1.0
    )

    slope_scale = 2.0 / (KNOTS_M[-1] - KNOTS_M[0])

    derivative = np.polynomial.chebyshev.chebder(
        values[:active_degree + 1]
    )

    slopes = (
        np.polynomial.chebyshev.chebval(samples_u, derivative)
        * slope_scale
    )

    maximum = float(np.max(np.abs(slopes)))

    if maximum <= max_slope:
        return values

    # Preserve degree and relative coefficient shape, reduce amplitude only.
    # T0 is an overall translation; the remaining terms create contour slope.
    ratio = max_slope / maximum
    values[1:active_degree + 1] *= ratio

    return values


def repair(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).copy()

    if x.shape != (N_GENES,) or not np.all(np.isfinite(x)):
        raise ValueError(f"expected {N_GENES} finite genes")

    x = np.clip(x, LOW, HIGH)

    x[DEG_IN] = as_degree(x[DEG_IN])
    x[DEG_OUT] = as_degree(x[DEG_OUT])

    degree_inner = as_degree(x[DEG_IN])
    degree_outer = as_degree(x[DEG_OUT])

    # Deactivate inactive terms permanently.
    x[INNER.start + degree_inner + 1:INNER.stop] = 0.0
    x[OUTER.start + degree_outer + 1:OUTER.stop] = 0.0

    # Enforce the same slope constraint as the project template,
    # leaving a 0.02 margin below its hard 0.80 rejection threshold.
    inner = active_coeffs(x, INNER, degree_inner)
    outer = active_coeffs(x, OUTER, degree_outer)

    inner = enforce_template_slope(
        inner,
        degree_inner,
        max_slope=0.78,
    )

    outer = enforce_template_slope(
        outer,
        degree_outer,
        max_slope=0.78,
    )

    x[INNER] = inner
    x[OUTER] = outer

    return np.clip(x, LOW, HIGH)


def baseline() -> np.ndarray:
    x = np.zeros(N_GENES)
    x[DEG_IN], x[DEG_OUT] = 2, 2
    x[CENTER_IN:] = [-25e-6, 20e-6, 30e-6, 150e-6, 2.0, 0.0, 70e-6, 25e-6]
    return x


def seed_genomes():
    base = baseline()
    a = base.copy(); a[DEG_IN] = 3; a[INNER.start:INNER.start+4] = [-4e-6, -8e-6, -7e-6, -3e-6]
    b = base.copy(); b[DEG_OUT] = 3; b[OUTER.start:OUTER.start+4] = [3e-6, 7e-6, 6e-6, 2e-6]
    c = base.copy(); c[DEG_IN] = c[DEG_OUT] = 4
    c[INNER.start:INNER.start+5] = [-3e-6, -6e-6, -8e-6, -4e-6, -2e-6]
    c[OUTER.start:OUTER.start+5] = [2e-6, 5e-6, 7e-6, 4e-6, 1e-6]
    d = base.copy(); d[START], d[LENGTH], d[POWER] = 20e-6, 130e-6, 2.4
    return [("baseline", base), ("inner", a), ("outer", b), ("coupled", c), ("central", d)]


def make_parameters(x: np.ndarray):
    inner, outer, _, _ = contour_offsets(x)
    return make_house_style_x_junction(
        ion_height_m=TARGET_ION_HEIGHT_M, arm_length_m=600e-6, outer_extent_m=900e-6,
        inner_edge_shift_at_centre_m=float(x[CENTER_IN]), outer_edge_shift_at_centre_m=float(x[CENTER_OUT]),
        rf_start_radius_override_m=float(x[START]), taper_length_m=float(x[LENGTH]), taper_power=float(x[POWER]),
        outer_bulge_amplitude_m=float(x[BULGE]), outer_bulge_center_m=float(x[BULGE_CENTER]),
        outer_bulge_sigma_m=float(x[BULGE_WIDTH]), inner_contour_knots_m=tuple(KNOTS_M),
        inner_contour_offsets_m=tuple(inner), outer_contour_knots_m=tuple(KNOTS_M),
        outer_contour_offsets_m=tuple(outer))


def rejected(reason: str, stage: str, rough: float=np.inf) -> dict[str, Any]:
    return dict(valid=False, verified=False, reason=reason, stage=stage, roughness=rough,
                dz_peak_m=np.inf, dz_rms_m=np.inf, barrier_ev=np.inf, lateral_peak_m=np.inf,
                panels=0, points=0, converged=0)


def build_mesh(p: Any, fast: bool):
    if fast:
        return build_geometry_aware_quadtree_x_junction_bem(
            p, central_half_extent_m=145e-6, central_max_cell_m=46e-6,
            boundary_max_cell_m=17e-6, outer_max_cell_m=180e-6, min_cell_m=8e-6)
    return build_geometry_aware_quadtree_x_junction_bem(
        p, central_half_extent_m=180e-6, central_max_cell_m=30e-6,
        boundary_max_cell_m=10e-6, outer_max_cell_m=180e-6, min_cell_m=5e-6)


def evaluate(genome: np.ndarray, cfg: Settings, stage="coarse", retain=False):
    x = repair(genome)
    rough, slope, curvature = contour_roughness(x)
    if rough > cfg.roughness_hard_limit:
        return rejected(
            f"roughness>{cfg.roughness_hard_limit:.1f}",
            stage,
            rough,
        ), None
    try:
        p = make_parameters(x)
        report = check_x_junction_manufacturability(p)
        if not report.valid:
            return rejected("geometry: "+"; ".join(map(str, report.messages)), stage, rough), None
        s = np.linspace(0, p.arm_length_m, 2001)
        rin, rout = p.rail_boundaries_m(s)
        if np.min(rin) <= .5e-6:
            return rejected("inner clearance", stage, rough), None
        if np.min(rout-rin) < cfg.rail_min_width_m:
            return rejected("rail width", stage, rough), None
        model = build_mesh(p, cfg.fast and stage == "coarse")
        if model.n_panels > cfg.max_panels:
            return rejected("panel limit", stage, rough), None
        bem = FixedMeshBEM(model.bem.panels_m, backend=cfg.backend)
        bem.assemble(block_rows=64); bem.factorize(); bem.release_matrix()
        charge = bem.solve_masks(model.bem.electrode_voltages_v); field = BEMField(bem, charge)
        n = cfg.fine_points if stage == "fine" else cfg.coarse_points
        trace = trace_rf_transverse_minimum(field, np.linspace(-350e-6,350e-6,n), initial_y_m=0.,
                                             initial_z_m=cfg.target_z_m, residual_tolerance_v_m=1e-3,
                                             max_transverse_shift_m=25e-6)
        ok = np.asarray(trace.converged, bool); z, y = np.asarray(trace.z_m,float), np.asarray(trace.y_m,float)
        if not trace.valid or ok.shape != (n,) or not np.all(ok):
            return rejected("incomplete trace", stage, rough), None
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(y)):
            return rejected("nonfinite trace", stage, rough), None
        pseudo = np.asarray(pseudopotential_profile_ev(field, trace, rf_voltage_peak_v=cfg.rf_peak_v,
                          rf_angular_frequency_rad_s=RF_ANGULAR_FREQUENCY_RAD_S),float)
        barrier = compute_barrier_metrics(pseudo, trace)
        if not barrier.valid or pseudo.shape != (n,) or not np.all(np.isfinite(pseudo)):
            return rejected("invalid pseudopotential", stage, rough), None
        dz = z-cfg.target_z_m; peak = float(np.max(np.abs(dz))); rms = float(np.sqrt(np.mean(dz**2)))
        lateral, U = float(np.max(np.abs(y))), float(barrier.barrier_height_ev)
        if not np.all(np.isfinite([peak,rms,lateral,U])):
            return rejected("nonfinite metrics", stage, rough), None
        return dict(valid=True, verified=bool(stage=="fine" and peak<cfg.dz_limit_m and U<cfg.barrier_limit_ev and lateral<cfg.lateral_limit_m),
                    reason="ok",stage=stage,backend=bem.backend_name,roughness=rough,slope_rms=slope,
                    curvature_rms_per_m=curvature,dz_peak_m=peak,dz_rms_m=rms,barrier_ev=U,lateral_peak_m=lateral,
                    barrier_reference_ev=float(barrier.reference_energy_ev),panels=int(model.n_panels),points=n,converged=int(np.sum(ok))), (model,trace,pseudo) if retain else None
    except Exception as err:
        if cfg.backend=="cuda" and "out of memory" in str(err).lower():
            raise RuntimeError("CUDA OOM: reduce --max-panels or use --fast") from err
        text = str(err).replace("\n"," ").strip()
        return rejected(f"BEM error: {type(err).__name__}: {text}", stage, rough), None


def dominates(a: Individual,b: Individual) -> bool:
    if a.valid != b.valid: return a.valid
    return bool(a.valid and np.all(a.objectives<=b.objectives) and np.any(a.objectives<b.objectives))


def rank_and_crowding(pop: list[Individual]) -> list[list[int]]:
    n=len(pop); beaten=[[] for _ in pop]; count=np.zeros(n,int); fronts=[[]]
    for i in range(n):
        for j in range(i+1,n):
            if dominates(pop[i],pop[j]): beaten[i].append(j); count[j]+=1
            elif dominates(pop[j],pop[i]): beaten[j].append(i); count[i]+=1
    fronts[0]=[i for i in range(n) if count[i]==0]
    while fronts[-1]:
        nxt=[]
        for i in fronts[-1]:
            for j in beaten[i]:
                count[j]-=1
                if count[j]==0: nxt.append(j)
        fronts.append(nxt)
    fronts.pop()
    for level,front in enumerate(fronts):
        for i in front: pop[i].rank,pop[i].crowding=level,0.
        valid=[i for i in front if pop[i].valid]
        if len(valid)<=2:
            for i in valid: pop[i].crowding=np.inf
            continue
        for axis in range(2):
            order=sorted(valid,key=lambda i:pop[i].objectives[axis]); lo,hi=pop[order[0]].objectives[axis],pop[order[-1]].objectives[axis]
            if hi<=lo: continue
            pop[order[0]].crowding=pop[order[-1]].crowding=np.inf
            for k in range(1,len(order)-1):
                if np.isfinite(pop[order[k]].crowding): pop[order[k]].crowding+=(pop[order[k+1]].objectives[axis]-pop[order[k-1]].objectives[axis])/(hi-lo)
    return fronts


def survivors(pool: list[Individual], size: int) -> list[Individual]:
    out=[]
    for front in rank_and_crowding(pool):
        front.sort(key=lambda i:pool[i].crowding,reverse=True); out.extend(pool[i].copy() for i in front[:size-len(out)])
        if len(out)==size: break
    return out


def score(i: Individual, style: IslandStyle) -> float:
    if not i.valid:
        return np.inf

    dz = i.result["dz_peak_m"] / 3e-6
    barrier = i.result["barrier_ev"] / 1e-3

    if style.emphasis == "joint":
        return max(
            i.result["dz_peak_m"] / 10e-6,
            i.result["barrier_ev"] / 10e-3,
        )

    return (
        style.weights[0] * dz
        + style.weights[1] * barrier
    )


def tournament(pop: list[Individual], style: IslandStyle,rng: np.random.Generator) -> Individual:
    a,b=(pop[int(k)] for k in rng.integers(0,len(pop),2))
    if rng.random()<.5: return a if score(a,style)<score(b,style) else b
    if a.rank!=b.rank: return a if a.rank<b.rank else b
    if a.crowding!=b.crowding: return a if a.crowding>b.crowding else b
    return a if score(a,style)<score(b,style) else b


def anneal(style: IslandStyle,fraction: float) -> float:
    if fraction<=.75: return style.explore
    q=(fraction-.75)/.25
    return style.final+.5*(style.explore-style.final)*(1+math.cos(math.pi*q))


def sigmas(style: IslandStyle,fraction: float) -> np.ndarray:
    scale=anneal(style,fraction); v=np.zeros(N_GENES); v[DEG_IN:DEG_OUT+1]=.60*scale
    co=np.array([10,10,8,6,4.5,3.5,2.5,2,1.5])*1e-6*scale; v[INNER]=co; v[OUTER]=co
    v[CENTER_IN:CENTER_OUT+1]=9e-6*scale; v[START],v[LENGTH],v[POWER]=10e-6*scale,24e-6*scale,.35*scale
    v[BULGE],v[BULGE_CENTER],v[BULGE_WIDTH]=10e-6*scale,18e-6*scale,12e-6*scale
    if style.emphasis=="inner": v[INNER]*=1.7
    elif style.emphasis=="outer": v[OUTER]*=1.7
    elif style.emphasis=="high_inner": v[INNER.start+4:INNER.stop]*=2.0
    elif style.emphasis=="high_outer": v[OUTER.start+4:OUTER.stop]*=2.0
    elif style.emphasis=="central": v[CENTER_IN:START+1]*=2.0
    elif style.emphasis=="degree": v[DEG_IN:DEG_OUT+1]*=2.8
    elif style.emphasis=="wide": v*=1.35
    elif style.emphasis=="local": v*=.55
    return v


def mutate(parent: np.ndarray,style: IslandStyle,rng: np.random.Generator,fraction: float) -> np.ndarray:
    x=parent.copy(); sigma=sigmas(style,fraction); chance=np.full(N_GENES,.55)
    chance[DEG_IN:DEG_OUT+1]=.13 if fraction<.75 else .025
    if style.emphasis=="degree": chance[DEG_IN:DEG_OUT+1]=.45 if fraction<.75 else .06
    if style.emphasis=="inner": chance[INNER]=.88
    elif style.emphasis=="outer": chance[OUTER]=.88
    active=rng.random(N_GENES)<chance; x[active]+=rng.normal(0,sigma[active])
    for index in (DEG_IN,DEG_OUT):
        if active[index]: x[index]=as_degree(parent[index])+int(rng.choice((-1,1)))
    for block,index in ((INNER,DEG_IN),(OUTER,DEG_OUT)):
        old,new=as_degree(parent[index]),as_degree(x[index])
        if new>old: x[block.start+old+1:block.start+new+1]=rng.normal(0,2e-6,new-old)
    return repair(x)


def block_crossover(a: np.ndarray,b: np.ndarray,rng: np.random.Generator) -> np.ndarray:
    out=np.empty(N_GENES); out[DEG_IN]=a[DEG_IN] if rng.random()<.5 else b[DEG_IN]; out[DEG_OUT]=a[DEG_OUT] if rng.random()<.5 else b[DEG_OUT]
    for block in (INNER,OUTER,slice(CENTER_IN,N_GENES)):
        alpha=rng.uniform(-.10,1.10); out[block]=alpha*a[block]+(1-alpha)*b[block]
    return repair(out)


def serialize(i: Individual) -> dict[str,Any]:
    inner,outer,di,do=contour_offsets(i.genome)
    return dict(island=i.island,generation=i.generation,origin=i.origin,rank=i.rank,crowding=i.crowding,
                genome=i.genome.tolist(),degree_inner=di,degree_outer=do,inner_offsets_at_knots_m=inner.tolist(),outer_offsets_at_knots_m=outer.tolist(),**i.result)


def save_csv(rows: list[dict[str,Any]],path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text("",encoding="utf-8"); return
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w",newline="",encoding="utf-8") as fp:
        writer=csv.DictWriter(fp,fieldnames=fields,extrasaction="ignore"); writer.writeheader()
        for r in rows: writer.writerow({k:json.dumps(v,allow_nan=True) if isinstance(v,(list,dict)) else v for k,v in r.items()})


def elite_rows(islands: list[Island],cfg: Settings) -> list[dict[str,Any]]:
    rows=[]
    for island in islands:
        rank_and_crowding(island.population)
        best=sorted(island.population,key=lambda i:(i.rank,-i.crowding,score(i,island.style)))[:cfg.retain_per_island]
        rows.extend(serialize(i) for i in best)
    return rows


def draw_geometry(ax,row: dict[str,Any],label: str) -> None:
    genome=repair(np.asarray(json.loads(row["genome"]) if isinstance(row["genome"],str) else row["genome"],float)); p=make_parameters(genome)
    s=np.linspace(0,240e-6,800); rin,rout=p.rail_boundaries_m(s)
    for theta in (0.,np.pi/2,np.pi,3*np.pi/2):
        c,q=np.cos(theta),np.sin(theta); x1,y1=s*c-rin*q,s*q+rin*c; x2,y2=s*c-rout*q,s*q+rout*c
        ax.fill(np.r_[x1,x2[::-1]]*1e6,np.r_[y1,y2[::-1]]*1e6,color="#db4f4f",ec="black",lw=.35,alpha=.90)
    ki,ko=p.rail_boundaries_m(KNOTS_M); ax.plot(KNOTS_M*1e6,ki*1e6,"ko",ms=2); ax.plot(KNOTS_M*1e6,ko*1e6,"wo",mec="black",ms=2)
    ax.axhline(0,color=".72",lw=.35); ax.axvline(0,color=".72",lw=.35); ax.set(xlim=(-240,240),ylim=(-240,240),aspect="equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{label}\ndz={float(row['dz_peak_m'])*1e6:.2f} um, U={float(row['barrier_ev'])*1e3:.2f} meV\n"
                 f"deg=({int(row['degree_inner'])},{int(row['degree_outer'])}), R={float(row['roughness']):.2f}",fontsize=8)


def animate_surfaces(cfg: Settings) -> None:
    try: import pandas as pd
    except ImportError: print("Surface GIFs skipped: pandas required."); return
    files=sorted((cfg.output_dir/"island_elites").glob("elites_gen_*.csv"))[::max(1,cfg.animation_stride)]
    if not files:return
    frames=[(int(f.stem.rsplit("_",1)[-1]),pd.read_csv(f)) for f in files]; target=cfg.output_dir/"surface_animations"; target.mkdir(exist_ok=True)
    for iid in range(cfg.islands):
        choices=[(g,t[(t["island"]==iid)&(t["valid"]==True)].sort_values(["rank","crowding","dz_peak_m","barrier_ev"],ascending=[True,False,True,True]).head(5)) for g,t in frames]
        if not any(not table.empty for _,table in choices): continue
        fig,axes=plt.subplots(1,5,figsize=(21,4.8),constrained_layout=True)
        def update(k):
            generation,table=choices[k]
            for ax in axes: ax.clear()
            rows=table.to_dict("records")
            for slot,ax in enumerate(axes):
                if slot<len(rows):
                    try: draw_geometry(ax,rows[slot],f"Top {slot+1}")
                    except Exception as err: ax.text(.5,.5,f"draw error\n{type(err).__name__}",transform=ax.transAxes,ha="center",va="center")
                else: ax.text(.5,.5,"no valid elite",transform=ax.transAxes,ha="center",va="center"); ax.set_axis_off()
            fig.suptitle(f"Island {iid}: top-5 physical RF surfaces — generation {generation}",fontsize=13); return axes
        anim=FuncAnimation(fig,update,frames=len(choices),interval=650,blit=False,repeat=True)
        try: anim.save(target/f"island_{iid:02d}_top5_surfaces.gif",writer=PillowWriter(fps=2),dpi=115)
        except Exception as err: print(f"Surface GIF skipped island {iid}: {type(err).__name__}: {err}")
        plt.close(fig)


def profile_plots(data,result,directory: Path,title: str) -> None:
    model,trace,pseudo=data; panels=np.asarray(model.bem.panels_m); rf=np.asarray(model.bem.electrode_voltages_v)>.5
    rects=[Rectangle((p[0]*1e6,p[2]*1e6),(p[1]-p[0])*1e6,(p[3]-p[2])*1e6) for p in panels]
    fig,ax=plt.subplots(figsize=(7,7)); col=PatchCollection(rects,array=rf.astype(float),cmap="RdYlBu_r"); ax.add_collection(col); ax.plot(np.asarray(trace.x_m)*1e6,np.asarray(trace.y_m)*1e6,"k-",lw=1.2); ax.set(xlim=(-230,230),ylim=(-230,230),aspect="equal",xlabel="x [um]",ylabel="y [um]",title=title); fig.colorbar(col,ax=ax); fig.tight_layout(); fig.savefig(directory/"layout.png",dpi=170); plt.close(fig)
    xx=np.asarray(trace.x_m)*1e6; fig,axs=plt.subplots(3,1,sharex=True,figsize=(9,9)); axs[0].plot(xx,(np.asarray(trace.z_m)-TARGET_ION_HEIGHT_M)*1e6); axs[0].axhline(3,color="r",ls="--"); axs[0].axhline(-3,color="r",ls="--"); axs[0].set_ylabel("z-target [um]"); axs[1].plot(xx,(pseudo-result["barrier_reference_ev"])*1e3); axs[1].axhline(1,color="r",ls="--"); axs[1].set_ylabel("RF pseudo-ref [meV]"); axs[2].plot(xx,np.asarray(trace.y_m)*1e6); axs[2].set(xlabel="x [um]",ylabel="y [um]"); [a.grid(alpha=.25) for a in axs]; fig.tight_layout(); fig.savefig(directory/"profiles.png",dpi=170); plt.close(fig)


def parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--backend",choices=("auto","cuda","cpu"),default="auto"); p.add_argument("--islands",type=int,default=12); p.add_argument("--population",type=int,default=100); p.add_argument("--offspring",type=int,default=100); p.add_argument("--generations",type=int,default=8); p.add_argument("--migration-interval",type=int,default=3); p.add_argument("--migrants",type=int,default=5); p.add_argument("--retain-per-island",type=int,default=5); p.add_argument("--coarse-points",type=int,default=61); p.add_argument("--fine-points",type=int,default=181); p.add_argument("--fine-top",type=int,default=25); p.add_argument("--max-panels",type=int,default=7000); p.add_argument("--fast",action="store_true"); p.add_argument("--animation-stride",type=int,default=1); p.add_argument("--seed",type=int,default=20260926); p.add_argument("--output-dir",type=Path,default=Path("reports/08_c4v_full_poly_regularized_v2")); return p


def main() -> None:
    a=parser().parse_args(); backend=available_backend(a.backend)
    if not backend.available: raise RuntimeError(backend.reason)
    if not 1<=a.islands<=len(STYLES): raise ValueError(f"islands must be 1..{len(STYLES)}")
    if a.population<8 or a.offspring<1 or a.generations<1: raise ValueError("population>=8, offspring/generations>=1")
    cfg=Settings(backend.selected,a.islands,a.population,a.offspring,a.generations,a.migration_interval,a.migrants,a.retain_per_island,a.seed,a.coarse_points,a.fine_points,a.fine_top,a.max_panels,a.fast,a.animation_stride,a.output_dir); cfg.output_dir.mkdir(parents=True,exist_ok=True)
    (cfg.output_dir/"settings.json").write_text(json.dumps(dict(settings=asdict(cfg),backend=backend.as_dict(),knots_m=KNOTS_M.tolist(),basis="Chebyshev",objectives=["dz","barrier","roughness"]),indent=2,default=str),encoding="utf-8")
    print("Regularized v2 full-polynomial C4v island GA | backend:",backend.selected,"| device:",backend.device_name)
    print(f"Genome={N_GENES}: two degrees, 18 polynomial coefficients, 8 global geometry genes.")
    print("Every island has distinct seeded and randomized initial geometries; high mutation through 75%, then cosine cooling.")
    master=np.random.default_rng(a.seed); seeds=seed_genomes(); cache={}; best_z=best_u=best_joint=None
    def joint(r): return max(r["dz_peak_m"]/10e-6,r["barrier_ev"]/10e-3)+.1*r["roughness"]
    def coarse(x,bar=None):
        nonlocal best_z,best_u,best_joint
        x=repair(x); key=tuple(np.round(x,12))
        if key not in cache: cache[key],_=evaluate(x,cfg)
        r=dict(cache[key])
        if r["valid"]:
            if best_z is None or r["dz_peak_m"]<best_z[1]["dz_peak_m"]: best_z=(x.copy(),r.copy())
            if best_u is None or r["barrier_ev"]<best_u[1]["barrier_ev"]: best_u=(x.copy(),r.copy())
            if best_joint is None or joint(r)<joint(best_joint[1]): best_joint=(x.copy(),r.copy())
            (cfg.output_dir/"best_live_coarse.json").write_text(json.dumps(dict(best_dz=dict(genome=best_z[0].tolist(),**best_z[1]),best_barrier=dict(genome=best_u[0].tolist(),**best_u[1]),best_joint=dict(genome=best_joint[0].tolist(),**best_joint[1])),indent=2),encoding="utf-8")
        if bar is not None:
            if best_z is None: bar.set_postfix_str("valid=0",refresh=False)
            else: bar.set_postfix(dict(dz=f'{best_z[1]["dz_peak_m"]*1e6:.2f}um/U{best_z[1]["barrier_ev"]*1e3:.1f}',U=f'{best_u[1]["barrier_ev"]*1e3:.2f}meV/dz{best_u[1]["dz_peak_m"]*1e6:.1f}',joint=f'dz{best_joint[1]["dz_peak_m"]*1e6:.2f}/U{best_joint[1]["barrier_ev"]*1e3:.2f}',R=f'{best_joint[1]["roughness"]:.2f}'),refresh=False)
        return r

    islands=[]
    with tqdm(total=cfg.islands*cfg.population,desc="Initial BEM",unit="candidate") as bar:
        for iid,style in enumerate(STYLES[:cfg.islands]):
            rng=np.random.default_rng(int(master.integers(0,2**32-1))); pop=[]
            # Rotate seed order per island, then perturb every seed. No exact
            # identical baseline is copied to all islands.
            order=np.roll(np.arange(len(seeds)),iid%len(seeds))
            for rank_seed in order:
                if len(pop)>=cfg.population: break
                name,seed=seeds[int(rank_seed)]
                perturb=rng.normal(0,(HIGH-LOW)*.012*style.explore)
                perturb[DEG_IN:DEG_OUT+1]=0.0
                x=repair(seed+perturb)
                pop.append(Individual(x,coarse(x,bar),iid,0,f"seed_{name}_island_{iid}")); bar.update()
            while len(pop)<cfg.population:
                seed=seeds[int(rng.integers(len(seeds)))][1]
                x=repair(seed+rng.normal(0,(HIGH-LOW)*.035*style.explore))
                pop.append(Individual(x,coarse(x,bar),iid,0,"initial")); bar.update()
            rank_and_crowding(pop); islands.append(Island(style,rng,pop))
    if not any(i.valid for isl in islands for i in isl.population):
        reasons=Counter(i.result["reason"] for isl in islands for i in isl.population); raise RuntimeError("No valid initial geometry: "+repr(reasons.most_common(20)))

    archive=[]; history=[]; rejection_rows=[]
    def update_archive(new):
        nonlocal archive
        pool=list({tuple(np.round(i.genome,12)):i.copy() for i in archive+new if i.valid}.values()); archive=[] if not pool else [pool[k].copy() for k in rank_and_crowding(pool)[0]]; archive.sort(key=lambda i:-i.crowding); archive[:]=archive[:1000]
    def summary(g):
        all_items=[i for isl in islands for i in isl.population]; valid=[i for i in all_items if i.valid]; z=min(valid,key=lambda i:i.result["dz_peak_m"]); u=min(valid,key=lambda i:i.result["barrier_ev"]); j=min(valid,key=lambda i:joint(i.result)); rejects=Counter(i.result["reason"] for i in all_items if not i.valid)
        for reason,count in rejects.items(): rejection_rows.append(dict(generation=g,reason=reason,count=count))
        return dict(generation=g,valid=len(valid),invalid=len(all_items)-len(valid),archive=len(archive),best_dz_um=z.result["dz_peak_m"]*1e6,U_at_best_dz_mev=z.result["barrier_ev"]*1e3,best_U_mev=u.result["barrier_ev"]*1e3,dz_at_best_U_um=u.result["dz_peak_m"]*1e6,best_joint_dz_um=j.result["dz_peak_m"]*1e6,best_joint_U_mev=j.result["barrier_ev"]*1e3,best_joint_roughness=j.result["roughness"],unique_evaluations=len(cache),rejects="; ".join(f"{k}:{v}" for k,v in rejects.most_common(4)))
    update_archive([i for isl in islands for i in isl.population]); history.append(summary(0)); save_csv(elite_rows(islands,cfg),cfg.output_dir/"island_elites"/"elites_gen_0000.csv"); print("G000",history[-1],flush=True)

    for gen in range(1,cfg.generations+1):
        next_islands=[]
        with tqdm(total=cfg.islands*cfg.offspring,desc=f"Generation {gen}",unit="candidate") as bar:
            for iid,isl in enumerate(islands):
                children=[]
                rank_and_crowding(isl.population)
                elite_parents = sorted(
                    isl.population,
                    key=lambda item: (
                        item.rank,
                        -item.crowding,
                        score(item, isl.style),
                    ),
                )[:max(2, min(5, len(isl.population)))]
                while len(children)<cfg.offspring:
                    u = isl.rng.random()

                    if u < 0.65:
                        # Exploitation: no crossover, direct local move around strong parent.
                        # 80% of these attempts attack top-1, 20% retain top-2 as a backup basin.
                        if len(elite_parents) > 1 and isl.rng.random() >= 0.80:
                            parent = elite_parents[1]
                            origin = "top2_direct_mutation"
                        else:
                            parent = elite_parents[0]
                            origin = "top1_direct_mutation"

                        child = parent.genome.copy()

                    elif u < 0.90:
                        # Standard Pareto exploration through semantically correct block crossover.
                        left = tournament(isl.population, isl.style, isl.rng)
                        right = tournament(isl.population, isl.style, isl.rng)

                        child = block_crossover(
                            left.genome,
                            right.genome,
                            isl.rng,
                        )
                        origin = "pareto_block_crossover"

                    else:
                        # Wide exploration: mix a leading solution with a random live solution.
                        left = elite_parents[
                            int(isl.rng.integers(len(elite_parents)))
                        ]
                        right = isl.population[
                            int(isl.rng.integers(len(isl.population)))
                        ]

                        child = block_crossover(
                            left.genome,
                            right.genome,
                            isl.rng,
                        )
                        origin = "elite_random_crossover"

                    child = mutate(
                        child,
                        isl.style,
                        isl.rng,
                        gen / cfg.generations,
                    )

                    children.append(
                        Individual(
                            child,
                            coarse(child, bar),
                            iid,
                            gen,
                            origin,
                        )
                    )
                    bar.update()   
                next_islands.append(Island(isl.style,isl.rng,survivors(isl.population+children,cfg.population)))
        islands=next_islands
        if cfg.migration_interval>0 and gen%cfg.migration_interval==0 and len(islands)>1:
            exported=[]
            for isl in islands:
                rank_and_crowding(isl.population); exported.append([i.copy() for i in sorted(isl.population,key=lambda i:(i.rank,-i.crowding,score(i,isl.style)))[:cfg.migrants]])
            for source,migrants in enumerate(exported):
                dest=(source+1)%len(islands); target=islands[dest]; islands[dest]=Island(target.style,target.rng,survivors(target.population+migrants,cfg.population))
        update_archive([i for isl in islands for i in isl.population]); history.append(summary(gen)); print(f"G{gen:03d}",history[-1],flush=True); save_csv(history,cfg.output_dir/"history.csv"); save_csv(rejection_rows,cfg.output_dir/"rejection_summary.csv"); save_csv(elite_rows(islands,cfg),cfg.output_dir/"island_elites"/f"elites_gen_{gen:04d}.csv"); save_csv([serialize(i) for isl in islands for i in isl.population],cfg.output_dir/f"population_gen_{gen:04d}.csv")

    animate_surfaces(cfg)
    selected=sorted(archive,key=lambda i:i.result["dz_peak_m"])[:cfg.fine_top]+sorted(archive,key=lambda i:i.result["barrier_ev"])[:cfg.fine_top]+sorted(archive,key=lambda i:joint(i.result))[:cfg.fine_top]; selected=list({tuple(np.round(i.genome,12)):i for i in selected}.values())
    fine=[]
    for num,item in enumerate(tqdm(selected,desc="Fine validation",unit="candidate"),1):
        result,data=evaluate(item.genome,cfg,"fine",True); row=dict(candidate=num,**serialize(item),**{"fine_"+k:v for k,v in result.items()}); fine.append(row)
        if data is not None:
            directory=cfg.output_dir/"fine"/f"candidate_{num:03d}"; directory.mkdir(parents=True,exist_ok=True); profile_plots(data,result,directory,f"Fine candidate {num}"); (directory/"full_candidate.json").write_text(json.dumps(row,indent=2),encoding="utf-8")
    save_csv(fine,cfg.output_dir/"fine_validation.csv"); valid=[r for r in fine if r["fine_valid"]]; strict=[r for r in valid if r["fine_verified"]]; candidates=strict or valid; top5=sorted(candidates,key=lambda r:max(r["fine_dz_peak_m"]/3e-6,r["fine_barrier_ev"]/1e-3)+.1*r["fine_roughness"])[:5]; (cfg.output_dir/"top5_lithography_candidates.json").write_text(json.dumps(top5,indent=2),encoding="utf-8"); save_csv(top5,cfg.output_dir/"top5_lithography_candidates.csv"); (cfg.output_dir/"verified_hits.json").write_text(json.dumps(strict,indent=2),encoding="utf-8"); print(f"Fine verified hits: {len(strict)} | top-5 full design records: {cfg.output_dir/'top5_lithography_candidates.json'}")


if __name__=="__main__":
    main()
