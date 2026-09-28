"""v223: run the existing v221 GA, then validate selected fine candidates.

This wrapper deliberately preserves v221 without editing it. L1/L2/L3
refer to axial SAMPLE DENSITY on v22's existing fine BEM mesh, not three
independently converged meshes. A positive result is NOT 3-D sign-off.

Example:
  python workflows/14_plus_b_spline_v223.py --smoke --output-dir reports/v223_smoke
  python workflows/14_plus_b_spline_v223.py --validate-top 3 --level L2 \
      --output-dir reports/v223_run --generations 20 --fast
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.analysis.rf_junction_validation import ValidationTargets, analyse_retained


def load_base():
    path = HERE / '14_plus_b_spline_v22.py'
    spec = importlib.util.spec_from_file_location('v22_validation_base', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validated_fine_rows(output_dir: Path, n: int) -> list[dict]:
    path = output_dir / 'fine_validation.csv'
    if not path.is_file():
        raise FileNotFoundError(f'v221 did not generate {path}')
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if row.get('genome')][:n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-top', type=int, default=3)
    parser.add_argument('--level', choices=('L1', 'L2', 'L3'), default='L2')
    parser.add_argument('--validation-budget-s', type=float, default=120.)
    parser.add_argument('--height-tolerance-um', type=float, default=2.)
    parser.add_argument('--skip-ga', action='store_true',
                        help='Reuse fine_validation.csv and settings.json from a completed run')
    args, forwarded = parser.parse_known_args()
    if args.validate_top < 1 or args.validation_budget_s <= 0 or args.height_tolerance_um <= 0:
        parser.error('validate-top, validation-budget-s and height-tolerance-um must be positive')
    if '--output-dir' in forwarded:
        index = forwarded.index('--output-dir')
        if index + 1 >= len(forwarded):
            parser.error('--output-dir needs a value')
        output_dir = Path(forwarded[index + 1])
    else:
        output_dir = Path('reports/13_seeded_zonal_islands')
    if not args.skip_ga:
        command = [sys.executable, str(HERE / '14_plus_b_spline_v221.py'), *forwarded]
        subprocess.run(command, cwd=ROOT, check=True)
    base = load_base()
    settings_path = output_dir / 'settings.json'
    if not settings_path.is_file():
        raise FileNotFoundError(f'v221 did not generate {settings_path}')
    payload = json.loads(settings_path.read_text(encoding='utf-8'))
    raw = payload['settings']
    raw['output_dir'] = Path(raw['output_dir'])
    valid_fields = base.Settings.__dataclass_fields__
    cfg = base.Settings(**{key: value for key, value in raw.items() if key in valid_fields})
    targets = ValidationTargets(height_tolerance_m=args.height_tolerance_um * 1e-6)
    report_dir = output_dir / 'multiscale_validation'
    report_dir.mkdir(parents=True, exist_ok=True)
    timing = []
    for position, row in enumerate(validated_fine_rows(output_dir, args.validate_top), 1):
        start = perf_counter()
        filename = report_dir / f'candidate_{position:03d}.json'
        try:
            genome = base.repair(np.asarray(json.loads(row['genome']), dtype=float))
            fine_result, retained = base.evaluate(genome, cfg, stage='fine', retain=True)
            if fine_result.get('valid') and retained is not None:
                diagnostic = analyse_retained(
                    fine_result, retained, cfg, base.RF_ANGULAR_FREQUENCY_RAD_S,
                    targets=targets, level=args.level,
                    max_seconds=args.validation_budget_s)
            else:
                diagnostic = {'complete': False, 'reason': fine_result.get('reason', 'fine failed')}
        except Exception as exc:
            diagnostic = {'complete': False, 'reason': f'{type(exc).__name__}: {exc}'}
        diagnostic['candidate'] = position
        diagnostic['elapsed_total_s'] = perf_counter() - start
        diagnostic['ga_fine_verified'] = str(row.get('fine_verified', '')).lower() == 'true'
        filename.write_text(json.dumps(diagnostic, indent=2, allow_nan=False), encoding='utf-8')
        timing.append({'candidate': position, 'level': args.level,
                       'panels': diagnostic.get('panels', ''),
                       'elapsed_s': diagnostic['elapsed_total_s'],
                       'complete': diagnostic['complete'],
                       'passes_axial_targets': diagnostic.get('passes_axial_targets', False)})
        print(f'candidate {position}: {diagnostic}', flush=True)
    with (output_dir / 'mesh_timing.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=('candidate', 'level', 'panels', 'elapsed_s',
                                                     'complete', 'passes_axial_targets'))
        writer.writeheader()
        writer.writerows(timing)
    return 0 if timing and all(item['complete'] for item in timing) else 2


if __name__ == '__main__':
    raise SystemExit(main())
