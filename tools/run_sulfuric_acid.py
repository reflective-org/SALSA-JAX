# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Run a configured sulfate-only trajectory and optionally compare actual Fortran."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import jax
import numpy as np

from salsa_jax.sulfuric_acid import load_config, simulate, sulfur_inventory
from tools.salsa_oracle import OUTPUTS
from tools.sulfuric_acid_validation import compare, reference_case, summarize

ROOT = Path(__file__).resolve().parents[1]


def run(config_path, output, *, compare_fortran=False):
    """Write explicit units, all-step arrays and diagnostics to a fresh directory."""
    config_path, output = Path(config_path), Path(output)
    config = load_config(config_path)
    output.mkdir(parents=True, exist_ok=False)
    jax.config.update('jax_enable_x64', True)
    initial, trajectory, mapping, grid = simulate(config)
    report = {'scope': 'Pure sulfate soluble aerosol; fixed environment; additive H2SO4 source before each step; supported SALSA core only',
              'reference_compared': bool(compare_fortran),
              'configuration': asdict(config), 'jax': summarize(config, initial, trajectory, mapping),
              'platform': platform.platform(), 'python': platform.python_version(),
              'jax_version': jax.__version__, 'numpy_version': np.__version__,
              'device': str(jax.devices()[0]), 'dtype': 'float64',
              'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
              'scientific_limits': ['No full HAM driver, radiation, transport or SO2 chemistry.',
                  'Soluble-only setup avoids the known cross-2a/2b coagulation defect; it does not repair it.',
                  'Native empty-background insertion gate and growth-option quirks are preserved.',
                  'A passed budget and native parity apply to this recorded configuration, not all possible settings.']}
    history = {key: np.asarray(getattr(trajectory, key)) for key in OUTPUTS}
    time = np.arange(config.steps+1)*config.timestep_s
    mass = np.concatenate((initial['mass'], history['mass'][:, 0]))
    gas = np.concatenate((initial['gas'], history['gas'][:, 0]))
    number = np.concatenate((initial['number'], history['number'][:, 0]))
    inventory = sulfur_inventory(mass, gas, mapping)
    target = inventory[0] + time*config.h2so4_source_molecules_m3_s
    table = np.column_stack((time, gas[:, 1], mass[:, mapping[:, 0]].sum(-1)*1e6,
                             number.sum(-1), inventory, target, inventory-target))
    np.savetxt(output/'timeseries.csv', table, delimiter=',', comments='',
               header='time_s,h2so4_molecules_m3,aerosol_sulfate_molecules_m3,number_m3,sulfur_molecules_m3,expected_sulfur_molecules_m3,sulfur_residual_molecules_m3')
    np.savetxt(output/'number-distribution.csv', np.column_stack((time, number)), delimiter=',', comments='',
               header='time_s,'+','.join(f'bin_{i+1}_number_m3' for i in range(17)))
    edges_low = np.cbrt(np.asarray(grid.vlolim)/(np.pi/6))
    edges_high = np.cbrt(np.asarray(grid.vhilim)/(np.pi/6))
    np.savetxt(output/'bins.csv', np.column_stack((np.arange(1,18), edges_low, np.asarray(grid.dpmid), edges_high)),
               delimiter=',', comments='', header='bin,dry_lower_diameter_m,dry_midpoint_diameter_m,dry_upper_diameter_m')
    np.savez_compressed(output/'trajectory.npz', time_s=time[1:], component_map=mapping,
                        **history, **{f'initial_{key}': value for key, value in initial.items()})
    accepted = report['jax']['sulfur_budget_passed'] and report['jax']['pure_sulfate_soluble_only']
    if compare_fortran:
        from tools.sulfuric_acid_oracle import build_oracle, run_oracle
        executable, provenance = build_oracle(output/'fortran-build')
        native, _ = run_oracle(executable, reference_case(config),
                              config.h2so4_source_molecules_m3_s, steps=config.steps)
        report['fortran_provenance'] = provenance
        report['fortran'] = summarize(config, initial, native, mapping)
        report['comparison'] = compare(trajectory, native)
        np.savez_compressed(output/'fortran-trajectory.npz', time_s=time[1:], **native)
        accepted = accepted and report['comparison']['passed'] and report['fortran']['sulfur_budget_passed'] and report['fortran']['pure_sulfate_soluble_only']
    report['accepted'] = bool(accepted)
    source_files = ['src/salsa_jax/sulfuric_acid.py', 'tools/run_sulfuric_acid.py', 'tools/sulfuric_acid_validation.py']
    source_files += [str(path.relative_to(ROOT)) for path in (ROOT/'src/salsa_jax').rglob('*.py') if str(path.relative_to(ROOT)) not in source_files]
    report['implementation_sha256'] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files}
    report['files'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir() if path.is_file()}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'README.md').write_text('''# Sulfuric-acid experiment output

`timeseries.csv` includes the initial state and every timestep, with units in
column names. `number-distribution.csv` contains per-bin particles/m³;
`bins.csv` supplies nominal dry bin edges and midpoints. For dN/dlog10(D), divide
each bin count by log10(upper/lower). Empty 2b bins duplicate 2a size intervals.

`trajectory.npz` retains all seven native caller fields after each step and
separate initial mass/number/gas. Mass has 69 native components: sulfate is
molecules/cm³, other materials µg/m³ (zero here). `component_map` is zero-based
SO4/OC/BC/SS/DU with -1 absent. Number is particles/m³, gas SO2/H2SO4/OC is
molecules/m³, radii are cm, density g/cm³ and water kg/m³. Sulfur budgets count
sulfur atoms; water does not contribute. No 98 g/mol acid conversion is applied.

If requested, `fortran-trajectory.npz` contains checked native reference outputs.
`report.json` records every validation gate, configuration, implementation and
output hashes. A failed comparison/budget exits nonzero; inspect its report.
''')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/sulfuric-acid.toml')
    parser.add_argument('--output', type=Path, required=True, help='Fresh output directory')
    parser.add_argument('--compare-fortran', action='store_true', help='Build checked original SALSA and compare every step')
    args = parser.parse_args()
    report = run(args.config, args.output, compare_fortran=args.compare_fortran)
    print(json.dumps({key: report[key] for key in ('accepted', 'jax', 'comparison') if key in report}, indent=2))
    return 0 if report['accepted'] else 1


if __name__ == '__main__':
    sys.exit(main())
