"""Record all-step sulfuric-acid acceptance and timestep sensitivity evidence."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import argparse
import platform

import jax
import numpy as np

from salsa_jax.sulfuric_acid import load_config, simulate
from tools.salsa_oracle import OUTPUTS
from tools.sulfuric_acid_oracle import build_oracle, run_oracle
from tools.sulfuric_acid_validation import reference_case, summarize, compare

ROOT = Path(__file__).resolve().parents[1]


def scenarios():
    """Isolated processes, active acid schemes/growth, closed and supplied sulfur."""
    base = load_config(ROOT/'configs/sulfuric-acid.toml')
    closed = replace(base, steps=1000, h2so4_source_molecules_m3_s=0.)
    result = {'configured-source': base, 'closed-activation': closed,
              'coagulation-only': replace(closed, nucleation_scheme=0, condensation_enabled=False,
                  initial_number_m3=(1e7, 3e7, 6e7, 1e8, 5e7, 1e7, 1e6, 1e5, 1e4, 1e3)),
              'condensation-only': replace(closed, nucleation_scheme=0, coagulation_enabled=False),
              'closed-binary': replace(closed, nucleation_scheme=1, temperature_k=260.),
              'closed-kinetic': replace(closed, nucleation_scheme=3)}
    for scheme in (1, 2, 3):
        for growth in (2, 3):
            result[f'scheme{scheme}-growth{growth}'] = replace(closed, nucleation_scheme=scheme,
                growth_option=growth, temperature_k=260. if scheme == 1 else 280.)
    result['empty-background'] = replace(base, steps=100, initial_number_m3=(0.,)*10)
    return result



def conditioning_check(executable):
    """Record sparse coagulation/remapping sensitivity without relaxing parity.

    One occupied midpoint bin produces a small receiving-bin transfer through
    a cancellation-sensitive native expression. A one-ULP native input change
    demonstrates its sensitivity independently of the JAX translation.
    """
    config = replace(load_config(ROOT/'configs/sulfuric-acid.toml'), steps=1,
                     h2so4_source_molecules_m3_s=0., nucleation_scheme=0,
                     condensation_enabled=False)
    initial, history, mapping, _ = simulate(config)
    case = reference_case(config)
    native, _ = run_oracle(executable, case, steps=1)
    perturbed = {**case, 'mass': case['mass'].copy()}
    index = mapping[3, 0]
    before = float(perturbed['mass'][0, index])
    perturbed['mass'][0, index] = np.nextafter(before, 0.)
    after = float(perturbed['mass'][0, index])
    changed, _ = run_oracle(executable, perturbed, steps=1)
    fields = {}
    for key, component in (('number', 4), ('mass', mapping[4, 0]), ('water_mass', 4)):
        reference = float(native[key][0, 0, component])
        actual = float(getattr(history, key)[0, 0, component])
        disturbed = float(changed[key][0, 0, component])
        fields[key] = {'component_index': int(component), 'fortran': reference, 'jax': actual,
                       'perturbed_fortran': disturbed,
                       'jax_relative_error': abs(actual-reference)/abs(reference),
                       'native_one_ulp_relative_sensitivity': abs(disturbed-reference)/abs(reference)}
    return {'scope': 'Known relative-accuracy limitation; NOT accepted under the strict trajectory parity gate',
            'configuration': asdict(config), 'comparison': compare(history, native),
            'initial_mass_component': int(index), 'initial_mass_before': before, 'initial_mass_after': after,
            'receiving_bin_fields': fields,
            'jax': summarize(config, initial, history, mapping),
            'fortran': summarize(config, initial, native, mapping),
            'interpretation': 'Sparse receiving-bin cancellation is present in original Fortran. No global tolerance widening, clipping or scientific repair is applied.'}


def validate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    jax.config.update('jax_enable_x64', True)
    checked, provenance = build_oracle(output/'checked')
    optimized, optimized_provenance = build_oracle(output/'optimized', checked=False)
    report = {'scope': 'Every post-step field of selected pure-sulfate fixed-environment trajectories; no full HAM driver claim',
              'platform': platform.platform(), 'python_version': platform.python_version(),
              'jax_version': jax.__version__, 'numpy_version': np.__version__,
              'checked_provenance': provenance, 'optimized_provenance': optimized_provenance,
              'scenarios': {}}
    configured = None
    for label, config in scenarios().items():
        initial, trajectory, mapping, _ = simulate(config)
        native, _ = run_oracle(checked, reference_case(config), config.h2so4_source_molecules_m3_s, steps=config.steps)
        fast, _ = run_oracle(optimized, reference_case(config), config.h2so4_source_molecules_m3_s, steps=config.steps)
        exact = all(np.array_equal(native[key], fast[key]) for key in OUTPUTS)
        comparison = compare(trajectory, native)
        jax_summary, native_summary = (summarize(config, initial, values, mapping) for values in (trajectory, native))
        row = {'configuration': asdict(config), 'comparison': comparison,
               'checked_optimized_exact': exact, 'jax': jax_summary, 'fortran': native_summary,
               'initial_array_sha256': {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in initial.items()}}
        row['accepted'] = bool(exact and comparison['passed'] and all(
            value['sulfur_budget_passed'] and value['pure_sulfate_soluble_only'] for value in (jax_summary, native_summary)))
        checkpoint_steps = sorted(set((1, min(10, config.steps), min(100, config.steps), config.steps)))
        row['checkpoints'] = [{'step': step, 'time_s': step*config.timestep_s,
                              'h2so4_molecules_m3': float(trajectory.gas[step-1, 0, 1]),
                              'number_m3_by_bin': trajectory.number[step-1, 0].tolist(),
                              'aerosol_sulfate_molecules_m3_by_bin': (trajectory.mass[step-1, 0, mapping[:, 0]]*1e6).tolist()}
                             for step in checkpoint_steps]
        report['scenarios'][label] = row
        print(label, 'PASS' if row['accepted'] else 'FAIL',
              max(v['max_relative_error_nonzero'] for v in comparison['fields'].values()), flush=True)
        if label == 'configured-source':
            configured = (config, initial, trajectory, native, mapping)
    # Fixed physical duration, half timestep. Record change, do not equate it
    # with a translation error or promise convergence from a single refinement.
    config, initial, coarse, coarse_native, mapping = configured
    half = replace(config, steps=config.steps*2, timestep_s=config.timestep_s/2)
    half_initial, fine, _, _ = simulate(half)
    fine_native, _ = run_oracle(checked, reference_case(half), half.h2so4_source_molecules_m3_s, steps=half.steps)
    fine_fast, _ = run_oracle(optimized, reference_case(half), half.h2so4_source_molecules_m3_s, steps=half.steps)
    half_comparison = compare(fine, fine_native)
    def changes(coarse_fields, fine_fields):
        result = {}
        for field in ('mass', 'number', 'gas'):
            c = np.asarray(coarse_fields[field] if isinstance(coarse_fields, dict) else getattr(coarse_fields, field))[-1]
            f = np.asarray(fine_fields[field] if isinstance(fine_fields, dict) else getattr(fine_fields, field))[-1]
            result[field+'_relative_L1_change'] = float(np.abs(f-c).sum()/max(np.abs(f).sum(), np.finfo(float).tiny))
        return result
    half_summary = summarize(half, half_initial, fine, mapping)
    half_native_summary = summarize(half, half_initial, fine_native, mapping)
    half_exact = all(np.array_equal(fine_native[key], fine_fast[key]) for key in OUTPUTS)
    report['timestep_sensitivity'] = {'physical_duration_s': config.steps*config.timestep_s,
        'coarse_timestep_s': config.timestep_s, 'fine_timestep_s': half.timestep_s,
        'fine_comparison': half_comparison, 'fine_jax': half_summary, 'fine_fortran': half_native_summary,
        'checked_optimized_exact': half_exact,
        'jax_changes': changes(coarse, fine), 'fortran_changes': changes(coarse_native, fine_native),
        'interpretation': 'One refinement measures native timestep sensitivity, not an established timestep-convergence rate.'}
    report['known_limitations'] = {'sparse_coagulation_transfer': conditioning_check(checked)}
    report['accepted_scope'] = 'Named standard scenarios and half-timestep check only; known limitations are reported separately and remain unaccepted.'
    report['accepted'] = bool(all(row['accepted'] for row in report['scenarios'].values()) and
        half_comparison['passed'] and half_exact and half_summary['sulfur_budget_passed'] and half_native_summary['sulfur_budget_passed'])
    names = ['configs/sulfuric-acid.toml', 'tools/validate_sulfuric_acid.py', 'tools/sulfuric_acid_oracle.py',
             'tools/sulfuric_acid_validation.py', 'tools/salsa_oracle.py']
    names += [str(path.relative_to(ROOT)) for path in (ROOT/'src/salsa_jax').rglob('*.py')]
    report['implementation_sha256'] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    result = validate(options.output)
    raise SystemExit(0 if result['accepted'] else 1)
