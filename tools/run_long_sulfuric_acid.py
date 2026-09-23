"""Run matching long JAX/Fortran sulfate experiments and save comparison data."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import jax
import numpy as np

from salsa_jax.sulfuric_acid import sulfur_inventory
from tools.long_sulfuric_acid import load_experiment, run_jax
from tools.long_sulfuric_acid_oracle import build_oracle, run_oracle
from tools.sulfuric_acid_validation import compare
from tools.salsa_oracle import OUTPUTS

ROOT = Path(__file__).resolve().parents[1]


def comparison_metrics(actual, native):
    report = compare(actual, native)
    for key in OUTPUTS:
        a, f = actual[key], native[key]
        difference = np.abs(a-f)
        relative = np.divide(difference, np.abs(f), out=np.zeros_like(f), where=f != 0)
        location = np.unravel_index(np.argmax(relative), relative.shape)
        scale = np.abs(f).sum(axis=-1)
        l1 = np.divide(difference.sum(axis=-1), scale, out=np.zeros_like(scale), where=scale != 0)
        report['fields'][key].update(max_relative_L1_error=float(l1.max()),
            worst_relative_error_index=list(map(int, location)),
            worst_relative_jax_value=float(a[location]), worst_relative_fortran_value=float(f[location]))
    return report


def summarize(initial, snapshots, monitors, mapping, experiment):
    inventory = sulfur_inventory(snapshots['mass'], snapshots['gas'], mapping)
    initial_inventory = sulfur_inventory(initial['mass'], initial['gas'], mapping)
    expected = initial_inventory+snapshots['cumulative_reservoir_added']
    error = inventory-expected
    relative = np.divide(np.abs(error), np.abs(expected), out=np.zeros_like(error), where=expected != 0)
    other = np.setdiff1d(np.arange(69), mapping[:, 0])
    pure = bool(np.all(snapshots['mass'][..., other] == 0) and
                np.all(snapshots['mass'][..., mapping[10:, 0]] == 0) and
                np.all(snapshots['number'][..., 10:] == 0) and
                np.all(snapshots['gas'][..., [0, 2]] == 0))
    return {'monitors_every_step': monitors,
        'sulfur_budget_passed': bool(np.max(monitors['max_relative_sulfur_residual']) <= 1e-10),
        'sulfur_budget_relative_tolerance': 1e-10,
        'maximum_checkpoint_sulfur_residual': float(relative.max()),
        'pure_sulfate_soluble_only': pure,
        'initial_number_cm3': float(initial['number'].sum()*1e-6),
        'final_number_cm3': float(snapshots['number'][-1].sum()*1e-6),
        'initial_sulfur_molecules_m3': float(initial_inventory[0]),
        'final_sulfur_molecules_m3': float(inventory[-1, 0]),
        'reservoir_added_molecules_m3': float(snapshots['cumulative_reservoir_added'][-1, 0]),
        'constant_vapor_at_checkpoints': bool(np.all(snapshots['gas'][..., 1] == experiment.h2so4_molecules_cm3*1e6)) if experiment.constant_h2so4 else None,
        'minimum_number': float(np.min(monitors['minimum_number'])),
        'minimum_mass': float(np.min(monitors['minimum_mass'])),
        'minimum_gas': float(np.min(monitors['minimum_gas']))}


def run(experiment, output, *, references=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    jax.config.update('jax_enable_x64', True)
    initial, actual, monitors, cases, mapping, grid = run_jax(experiment)
    if references is None:
        checked, provenance = build_oracle(output/'checked-fortran')
        optimized, optimized_provenance = build_oracle(output/'optimized-fortran', checked=False)
    else:
        (checked, provenance), (optimized, optimized_provenance) = references
    kwargs = dict(steps=experiment.steps, save_every=experiment.save_every, constant_h2so4=experiment.constant_h2so4)
    native, timing = run_oracle(checked, cases, **kwargs)
    fast, _ = run_oracle(optimized, cases, **kwargs)
    exact = all(np.array_equal(native[key], fast[key]) for key in native)
    native_monitors = {key: timing[key] for key in monitors if key != 'valid'}
    native_monitors['valid'] = [bool(timing['every_step_finite'])]
    comparisons = comparison_metrics(actual, native)
    jax_summary = summarize(initial, actual, monitors, mapping, experiment)
    native_summary = summarize(initial, native, native_monitors, mapping, experiment)
    times = np.arange(1, experiment.steps//experiment.save_every+1)*experiment.save_interval_s
    report = {'configuration': asdict(experiment), 'steps': experiment.steps,
        'snapshots': len(times), 'platform': platform.platform(), 'python': platform.python_version(),
        'jax_version': jax.__version__, 'dtype': 'float64', 'device': str(jax.devices()[0]),
        'initialization': 'Lognormal number integrated over ten native soluble-bin edges, no renormalization; each bin assigned native midpoint per-particle dry volume. This preserves binned number, not analytic lognormal third moment. Other species and seven insoluble bins zero.',
        'reservoir_convention': 'After every native/JAX step, restore H2SO4 to its initial value and account for target-minus-poststep gas as external sulfur. Native internal depletion during the timestep is retained and checked by half-timestep repeats.' if experiment.constant_h2so4 else 'No forcing; closed sulfur inventory.',
        'initial_number_cm3_by_bin': (initial['number'][0]*1e-6).tolist(),
        'checked_provenance': provenance, 'optimized_provenance': optimized_provenance,
        'checked_optimized_exact': exact, 'comparison': comparisons,
        'jax': jax_summary, 'fortran': native_summary,
        'scope': 'Two independent implementations of supported SALSA core; no original box driver/radiation. Agreement and conserved sulfur do not establish size-redistribution accuracy.',
        'comparison_gate': 'Existing 3e-11 trajectory relative tolerance retained; failures are reported, never hidden by renormalization or smoothing.'}
    report['strict_acceptance_passed'] = bool(exact and comparisons['passed'] and
        all(row['sulfur_budget_passed'] and row['pure_sulfate_soluble_only'] and
            min(row['minimum_mass'], row['minimum_number'], row['minimum_gas']) >= 0
            for row in (jax_summary, native_summary)))
    np.savez_compressed(output/'jax.npz', time_s=times, **actual)
    np.savez_compressed(output/'fortran.npz', time_s=times, **native)
    np.savez_compressed(output/'initial.npz', **initial, component_map=mapping)
    lower = np.cbrt(np.asarray(grid.vlolim)/(np.pi/6))
    upper = np.cbrt(np.asarray(grid.vhilim)/(np.pi/6))
    np.savetxt(output/'bins.csv', np.column_stack((np.arange(1,18), lower, grid.dpmid, upper)),
               delimiter=',', comments='', header='bin,lower_diameter_m,midpoint_diameter_m,upper_diameter_m')
    for label, values in (('jax', actual), ('fortran', native)):
        number = np.concatenate((initial['number'], values['number'][:, 0]))*1e-6
        fulltime = np.r_[0., times]
        np.savetxt(output/f'{label}-number-distribution.csv', np.column_stack((fulltime/3600, number)),
            delimiter=',', comments='', header='time_hours,'+','.join(f'bin{i+1}_number_cm3' for i in range(17)))
        inventory = sulfur_inventory(values['mass'], values['gas'], mapping)[:, 0]
        np.savetxt(output/f'{label}-timeseries.csv', np.column_stack((times/3600,values['number'][:,0].sum(-1)*1e-6,
            values['gas'][:,0,1]*1e-6,values['h2so4_before_restore'][:,0]*1e-6,
            values['cumulative_reservoir_added'][:,0],inventory)),delimiter=',',comments='',
            header='time_hours,total_number_cm3,h2so4_cm3,h2so4_before_restore_cm3,reservoir_added_molecules_m3,sulfur_molecules_m3')
    names = ['tools/long_sulfuric_acid.py', 'tools/long_sulfuric_acid_oracle.py', 'tools/run_long_sulfuric_acid.py',
             'tools/sulfuric_acid_oracle.py', 'tools/salsa_oracle.py', 'tools/sulfuric_acid_validation.py']
    names += [str(path.relative_to(ROOT)) for path in (ROOT/'src/salsa_jax').rglob('*.py')]
    report['implementation_sha256'] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    report['files'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir() if path.is_file()}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    result = run(load_experiment(options.config), options.output)
    print(json.dumps({key:result[key] for key in ('configuration','strict_acceptance_passed','comparison','jax','fortran')},indent=2))
    raise SystemExit(0 if result['strict_acceptance_passed'] else 1)
