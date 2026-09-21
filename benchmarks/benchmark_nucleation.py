"""CPU float64 isolated nucleation and coupled condensation/nucleation timings.

Run as ``.venv/bin/python -m benchmarks.benchmark_nucleation``. Each measured
JAX group has static scheme/growth options and the same two-cell workload as
its native Fortran oracle (one cell above and one inside the boundary layer).
"""
from __future__ import annotations
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import jax
import jaxlib
import numpy as np

from salsa_jax.processes.condensation import CondensationParameters
from salsa_jax.processes.condensation_nucleation import condensation_nucleation_step
from salsa_jax.processes.nucleation import nucleation_sources
from tools.nucleation_process_oracle import build_oracle, reference_cases, run_oracle

ROOT = Path(__file__).resolve().parents[1]
PROPERTIES = ('sulfate_molar_mass', 'sulfate_density', 'organic_molar_mass', 'organic_density', 'mass_accommodation')
SOURCE_INPUTS = ('number', 'gas', 'wet_diameter', 'temperature', 'relative_humidity', 'pressure', 'timestep')
COUPLED_INPUTS = ('number', 'volume', 'gas', 'wet_diameter', 'temperature', 'relative_humidity', 'pressure', 'timestep')
RTOL = 3e-13


def select_cases(cases, scheme, growth, single_case=False):
    indices = np.flatnonzero((cases['scheme'] == scheme) & (cases['growth'] == growth))
    if single_case:
        # Pick a nonzero binary-source case rather than benchmark only a cutoff.
        active = [i for i in indices if cases['labels'][i].startswith('active-binary-')]
        indices = np.array(active[:1]) if active else indices[:1]
    return {key: [value[i] for i in indices] if key == 'labels' else value[indices]
            for key, value in cases.items()}


def paired_cases(cases):
    """Mirror the oracle's duplicate two-level input, including its PBL boundary."""
    paired = {key: np.repeat(value[:, None], 2, axis=1) for key, value in cases.items() if key != 'labels'}
    paired['boundary_layer'] = np.broadcast_to(np.array([False, True]), (len(cases['labels']), 2))
    return paired


def error_metrics(actual, expected):
    delta = np.abs(actual - expected)
    nonzero = expected != 0
    return {'max_absolute': float(delta.max(initial=0)),
            'max_relative_nonzero': float(np.max(delta[nonzero] / np.abs(expected[nonzero]), initial=0)),
            'max_absolute_at_reference_zero': float(delta[~nonzero].max(initial=0))}


def _paired_reference(executable, cases, operation):
    outputs = []
    for boundary in (False, True):
        selected = {**cases, 'boundary_layer': np.full(len(cases['labels']), boundary)}
        result, _ = run_oracle(executable, selected, operation=operation)
        outputs.append(result)
    return {key: np.stack([value[key] for value in outputs], axis=1) for key in outputs[0]}


def benchmark(repetitions=500, trials=5, *, scheme=None, growth=None, operation='both', single_case=False):
    if min(repetitions, trials) < 1:
        raise ValueError('repetitions and trials must be positive')
    if scheme not in (None, 1, 2, 3) or growth not in (None, 1, 2, 3):
        raise ValueError('scheme and growth must be 1, 2, or 3')
    if operation not in ('both', 'isolated', 'coupled'):
        raise ValueError('operation must be isolated, coupled, or both')
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_enable_compilation_cache', False)
    device = jax.devices('cpu')[0]
    cases = reference_cases()
    measurements = []
    with tempfile.TemporaryDirectory(prefix='salsa-nucleation-benchmark-') as temporary:
        checked, checked_provenance = build_oracle(Path(temporary) / 'checked')
        optimized, optimized_provenance = build_oracle(Path(temporary) / 'optimized', checked=False)
        for selected_scheme in ((scheme,) if scheme is not None else (1, 2, 3)):
            for selected_growth in ((growth,) if growth is not None else (1, 2, 3)):
                selected = select_cases(cases, selected_scheme, selected_growth, single_case)
                paired = paired_cases(selected)
                parameters = CondensationParameters(**{key: jax.device_put(paired[key], device) for key in PROPERTIES})
                kwargs = {'parameters': parameters,
                          'organic_enabled': jax.device_put(paired['organic_enabled'], device),
                          'boundary_layer': jax.device_put(paired['boundary_layer'], device)}
                for selected_operation in (('isolated', 'coupled') if operation == 'both' else (operation,)):
                    # Check both native levels, though the oracle reports only
                    # its selected level per run. These checks are outside timing.
                    expected = _paired_reference(checked, selected, selected_operation)
                    fast = _paired_reference(optimized, selected, selected_operation)
                    for field in expected:
                        np.testing.assert_array_equal(fast[field], expected[field])
                    names = SOURCE_INPUTS if selected_operation == 'isolated' else COUPLED_INPUTS
                    inputs = tuple(jax.device_put(paired[key], device) for key in names)
                    function = nucleation_sources if selected_operation == 'isolated' else condensation_nucleation_step
                    kernel = partial(function, scheme=selected_scheme, growth=selected_growth)
                    jax.block_until_ready((inputs, kwargs))
                    started = time.perf_counter()
                    compiled = jax.jit(kernel).lower(*inputs, **kwargs).compile()
                    compile_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    actual = jax.block_until_ready(compiled(*inputs, **kwargs))
                    first_seconds = time.perf_counter() - started
                    fields = ('molecular_rate', 'fractions') if selected_operation == 'isolated' else ('number', 'volume', 'gas')
                    parity = {}
                    for field in fields:
                        value = np.asarray(getattr(actual, field))
                        np.testing.assert_allclose(value, expected[field], rtol=RTOL, atol=0)
                        parity[field] = error_metrics(value, expected[field])
                    jax_trials, fortran_trials, cpu_trials, orders = [], [], [], []
                    for trial in range(trials):
                        order = ('jax', 'fortran') if trial % 2 == 0 else ('fortran', 'jax')
                        orders.append(order)
                        for implementation in order:
                            if implementation == 'jax':
                                started = time.perf_counter()
                                for _ in range(repetitions):
                                    actual = jax.block_until_ready(compiled(*inputs, **kwargs))
                                jax_trials.append((time.perf_counter() - started) / repetitions)
                            else:
                                timed, timing = run_oracle(optimized, selected, repetitions=repetitions, operation=selected_operation)
                                fortran_trials.append(timing['kernel_wall_seconds'] / repetitions)
                                cpu_trials.append(timing['kernel_cpu_seconds'] / repetitions)
                    # Native checksums cover the selected cell, even though both
                    # cells execute. Also store full paired output checksums.
                    chosen = selected['boundary_layer'].astype(int)
                    checksums = {}
                    for field in fields:
                        value = np.asarray(getattr(actual, field))
                        selected_value = value[np.arange(len(chosen)), chosen]
                        np.testing.assert_allclose(selected_value, timed[field], rtol=RTOL, atol=0)
                        checksums[field] = {'jax_both_levels': float(value.sum()),
                                            'fortran_both_levels': float(expected[field].sum()),
                                            'jax_selected_level': float(selected_value.sum()),
                                            'fortran_selected_level': float(timed[field].sum())}
                        if selected_operation == 'coupled':
                            np.testing.assert_allclose(selected_value.sum(), timing[field + '_checksum'], rtol=RTOL, atol=0)
                    measurements.append({
                        'scheme': selected_scheme, 'growth': selected_growth, 'operation': selected_operation,
                        'case_count': len(selected['labels']), 'physical_cells_per_batch': 2 * len(selected['labels']),
                        'case_labels': selected['labels'],
                        'case_sha256': {key: hashlib.sha256(np.asarray(value, dtype='<f8').tobytes(order='C')).hexdigest()
                                        for key, value in selected.items() if key != 'labels'},
                        'checked_optimized_parity': 'bit-identical for every output at both native levels',
                        'jax_fortran_parity': parity, 'checksums': checksums,
                        'jax_compile_seconds': compile_seconds, 'jax_first_execution_seconds': first_seconds,
                        'jax_warm_wall_seconds_per_batch': jax_trials,
                        'fortran_warm_wall_seconds_per_batch': fortran_trials,
                        'fortran_cpu_seconds_per_batch': cpu_trials, 'trial_orders': orders,
                        'fortran_clock_ticks_per_second': timing['clock_ticks_per_second'],
                        'median_fortran_over_jax': statistics.median(fortran_trials) / statistics.median(jax_trials),
                    })
    return {
        'scope': 'Standalone isolated nucleation sources and/or native condensation with nucleation; 17 bins, five materials, schemes/growth 1–3, nsoa=0, no HAMMOZ',
        'scientific_limit': 'Hydrated diameter is supplied. Native Taylor-series and empty-background condensation-gate behavior are retained. No full SALSA timestep or trajectory validation is implied.',
        'platform': platform.platform(), 'device': str(device), 'dtype': 'float64',
        'versions': {'jax': jax.__version__, 'jaxlib': jaxlib.__version__, 'numpy': np.__version__},
        'thread_environment': {key: os.environ.get(key) for key in ('XLA_FLAGS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'JAX_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')},
        'implementation_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            'src/salsa_jax/constants.py', 'src/salsa_jax/processes/nucleation.py', 'src/salsa_jax/processes/nucleation_rates.py',
            'src/salsa_jax/processes/condensation.py', 'src/salsa_jax/processes/condensation_nucleation.py',
            'tools/nucleation_process_oracle.py', 'tools/condensation_process_oracle.py', 'tools/coagulation_process_oracle.py',
            'benchmarks/benchmark_nucleation.py')},
        'checked_fortran_provenance': checked_provenance, 'optimized_fortran_provenance': optimized_provenance,
        'case_hash_encoding': 'named arrays, C-order IEEE float64 little endian; native two-level expansion documented',
        'repetitions': repetitions, 'trials': trials, 'jax_persistent_compilation_cache': False,
        'measurements': measurements,
        'timing_notes': 'Alternating warmed wall-time trials. Both implementations execute two equal input cells with PBL flags False/True per case. Fortran loops over cases; JAX batches them within each static scheme/growth group. Fortran O2 uses no fast-math/LTO or runtime checks; a false diagnostic branch remains. Fortran includes mutable-input resets, species/global setup and selected-output copies. JAX uses pure fixed-input calls and includes dispatch/synchronization of all returned outputs. Compilation, transfers, process startup, I/O, diagnostic critical-state capture, and checksums are excluded from warm timing. Isolated Fortran retains the native state interface, whereas JAX returns only source rates/fractions. Thread settings are recorded, not normalized. Ratios describe these workloads, not full-model or GPU performance.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=500)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--scheme', type=int, choices=(1, 2, 3))
    parser.add_argument('--growth', type=int, choices=(1, 2, 3))
    parser.add_argument('--operation', choices=('isolated', 'coupled', 'both'), default='both')
    parser.add_argument('--single-case', action='store_true', help='one active binary case per selected scheme/growth group')
    args = parser.parse_args()
    if min(args.repetitions, args.trials) < 1:
        parser.error('repetitions and trials must be positive')
    print(json.dumps(benchmark(args.repetitions, args.trials, scheme=args.scheme, growth=args.growth,
                               operation=args.operation, single_case=args.single_case), indent=2))
