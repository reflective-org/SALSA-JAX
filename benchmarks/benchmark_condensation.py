# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""CPU float64 standalone condensation parity/timing, with nucleation disabled."""
import argparse
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

from salsa_jax.processes.condensation import CondensationParameters, condensation_step
from tools.condensation_process_oracle import build_oracle, reference_cases, run_oracle

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ('number', 'volume', 'gas', 'wet_diameter', 'temperature', 'pressure', 'timestep')
PROPERTIES = ('sulfate_molar_mass', 'sulfate_density', 'organic_molar_mass', 'organic_density', 'mass_accommodation')


def errors(actual, expected):
    delta = np.abs(actual - expected)
    nonzero = expected != 0
    return {'max_absolute': float(delta.max()),
            'max_relative_nonzero': float(np.max(delta[nonzero] / np.abs(expected[nonzero]), initial=0)),
            'max_absolute_at_reference_zero': float(np.max(delta[~nonzero], initial=0))}


def benchmark(repetitions=500, trials=5, single_case=False):
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_enable_compilation_cache', False)
    cases = reference_cases()
    if single_case:
        cases = {key: value[:1] for key, value in cases.items()}
    device = jax.devices('cpu')[0]
    inputs = tuple(jax.device_put(cases[key], device) for key in INPUTS)
    properties = CondensationParameters(**{key: jax.device_put(cases[key], device) for key in PROPERTIES})
    organic_enabled = jax.device_put(cases['organic_enabled'], device)
    kwargs = {'parameters': properties, 'organic_enabled': organic_enabled}
    jax.block_until_ready((inputs, kwargs))
    started = time.perf_counter()
    compiled = jax.jit(condensation_step).lower(*inputs, **kwargs).compile()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    actual = jax.block_until_ready(compiled(*inputs, **kwargs))
    first_execution_seconds = time.perf_counter() - started
    jax_trials, fortran_trials, cpu_trials, orders = [], [], [], []
    with tempfile.TemporaryDirectory(prefix='salsa-condensation-benchmark-') as directory:
        # Tests use checked=True. Timing uses optimized Fortran without bounds
        # checks/FPE traps, retaining the same arithmetic and no fast-math/LTO.
        executable, provenance = build_oracle(directory, checked=False)
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
                    expected, timing = run_oracle(executable, cases, repetitions=repetitions)
                    fortran_trials.append(timing['kernel_wall_seconds'] / repetitions)
                    cpu_trials.append(timing['kernel_cpu_seconds'] / repetitions)
    metrics = {}
    for name in ('number', 'volume', 'gas'):
        value = np.asarray(getattr(actual, name))
        np.testing.assert_allclose(value, expected[name], rtol=2e-13, atol=0)
        np.testing.assert_allclose(value.sum(), timing[name + '_checksum'], rtol=2e-13, atol=0)
        metrics[name] = errors(value, expected[name])
    budgets = {}
    for gas_slot, volume_slot, prefix in ((1, 0, 'sulfate'), (2, 1, 'organic')):
        molecular_volume = cases[prefix + '_molar_mass'] / 1000 / provenance['compiled_constants']['avogadro'] / cases[prefix + '_density']
        before = cases['gas'][:, gas_slot] + cases['volume'][:, :, volume_slot].sum(axis=1) / molecular_volume
        after = expected['gas'][:, gas_slot] + expected['volume'][:, :, volume_slot].sum(axis=1) / molecular_volume
        relative = np.divide(after-before, before, out=np.zeros_like(before), where=before!=0)
        budgets[prefix] = float(np.max(np.abs(relative)))
    return {
        'scope': 'standalone condensation; 17 bins, 5 materials, nsnucl=0, nsoa=0; supplied hydration, canonical SO2/H2SO4/OC gases',
        'scientific_limit': 'Registered organic cases are explicit kernel fixtures; default full driver has no organic gas registration. Nucleation, VBS and coupled trajectories excluded.',
        'platform': platform.platform(), 'device': str(device), 'dtype': str(actual.number.dtype),
        'versions': {'jax': jax.__version__, 'jaxlib': jaxlib.__version__, 'numpy': np.__version__},
        'thread_environment': {key: os.environ.get(key) for key in (
            'XLA_FLAGS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'JAX_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')},
        'implementation_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            'src/salsa_jax/constants.py', 'src/salsa_jax/processes/condensation.py',
            'tools/condensation_process_oracle.py', 'tools/coagulation_process_oracle.py',
            'benchmarks/benchmark_condensation.py')},
        'case_count': len(cases['labels']), 'case_labels': cases['labels'],
        'case_sha256': {key: hashlib.sha256(np.asarray(cases[key], dtype='<f8').tobytes(order='C')).hexdigest() for key in (*INPUTS, *PROPERTIES, 'organic_enabled')},
        'case_hash_encoding': 'named arrays, C-order IEEE float64 little endian',
        'fortran_provenance': provenance, 'parity': metrics,
        'fortran_max_relative_species_budget_error': budgets,
        'repetitions': repetitions, 'trials': trials, 'trial_orders': orders,
        'jax_compile_seconds': compile_seconds, 'jax_persistent_compilation_cache': False,
        'jax_first_execution_seconds': first_execution_seconds,
        'jax_warm_wall_seconds_per_batch': jax_trials,
        'fortran_warm_wall_seconds_per_batch': fortran_trials,
        'fortran_cpu_seconds_per_batch': cpu_trials,
        'fortran_clock_ticks_per_second': timing['clock_ticks_per_second'],
        'median_fortran_over_jax': statistics.median(fortran_trials) / statistics.median(jax_trials),
        'timing_notes': 'Alternating warmed wall-time trials. Fortran uses separately compiled original '
                        'arithmetic with O2, no fast-math/LTO, and runtime checks disabled; a false diagnostic '
                        'branch remains in the process. It includes copying mutable inputs and setting species properties '
                        'before each call; startup, I/O, diagnostic rate capture and checksums are excluded. '
                        'JAX uses the same fixed inputs each call and includes dispatch plus synchronization '
                        'of all outputs; transfer and compilation are excluded. Fortran loops over cases, '
                        'JAX batches them. Thread settings are recorded, not normalized. These measurements '
                        'are not full-model, GPU, or hydration timings.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=500)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--single-case', action='store_true', help='time only the captured default state')
    args = parser.parse_args()
    if min(args.repetitions, args.trials) < 1:
        parser.error('repetitions and trials must be positive')
    print(json.dumps(benchmark(args.repetitions, args.trials, args.single_case), indent=2))
