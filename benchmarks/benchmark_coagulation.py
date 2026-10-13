# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Scoped CPU float64 benchmark; does not measure a complete SALSA time step."""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import statistics
import tempfile
import time

import jax
import jaxlib
import numpy as np

from salsa_jax.processes.coagulation import coagulation_coefficient
from tools.coagulation_oracle import build_oracle, reference_cases, run_oracle


def benchmark(repetitions=1000, trials=5):
    jax.config.update("jax_enable_x64", True)
    # A fresh CLI process plus disabled disk cache makes compilation explicit.
    jax.config.update("jax_enable_compilation_cache", False)
    cases = reference_cases()
    device = jax.devices("cpu")[0]
    inputs = [jax.device_put(column, device) for column in cases.T]
    for value in inputs:
        value.block_until_ready()
    kernel = jax.jit(coagulation_coefficient)
    started = time.perf_counter()
    compiled = kernel.lower(*inputs).compile()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = compiled(*inputs).block_until_ready()
    first_execution_seconds = time.perf_counter() - started
    timings, fortran_wall_trials, fortran_cpu_trials, trial_orders = [], [], [], []
    with tempfile.TemporaryDirectory(prefix="salsa-coagc-benchmark-") as directory:
        executable, provenance = build_oracle(directory)
        for trial in range(trials):
            order = ("jax", "fortran") if trial % 2 == 0 else ("fortran", "jax")
            trial_orders.append(order)
            for implementation in order:
                if implementation == "jax":
                    started = time.perf_counter()
                    for _ in range(repetitions):
                        result = compiled(*inputs).block_until_ready()
                    timings.append((time.perf_counter() - started) / repetitions)
                else:
                    expected, timing = run_oracle(executable, cases, repetitions)
                    fortran_wall_trials.append(timing["kernel_wall_seconds"] / repetitions)
                    fortran_cpu_trials.append(timing["kernel_cpu_seconds"] / repetitions)
    actual = np.asarray(result)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0)
    checksum = float(actual.sum())
    np.testing.assert_allclose(timing["checksum"], checksum, rtol=1e-13, atol=0)
    return {
        "scope": "coagc collision coefficients only; not full coagulation or SALSA",
        "implementation_sha256": {
            name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
            for name in ["src/salsa_jax/constants.py", "src/salsa_jax/processes/coagulation.py",
                         "benchmarks/benchmark_coagulation.py", "tools/coagulation_oracle.py"]
        },
        "platform": platform.platform(), "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__, "numpy_version": np.__version__,
        "thread_environment": {name: os.environ.get(name) for name in (
            "XLA_FLAGS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "JAX_NUM_THREADS",
            "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
        "device": str(device), "dtype": str(actual.dtype),
        "case_count": len(cases), "repetitions": repetitions, "trials": trials,
        "case_inputs_sha256": hashlib.sha256(np.asarray(cases, dtype="<f8").tobytes()).hexdigest(),
        "case_hash_encoding": "C-order (n,6) IEEE float64 little endian",
        "trial_orders": trial_orders,
        "provenance": provenance,
        "jax_compile_seconds": compile_seconds,
        "jax_persistent_compilation_cache": False,
        "jax_first_execution_seconds": first_execution_seconds,
        "jax_warm_seconds_per_batch": timings,
        "fortran_kernel_wall_seconds_per_batch": fortran_wall_trials,
        "fortran_kernel_cpu_seconds_per_batch": fortran_cpu_trials,
        "fortran_clock_ticks_per_second": timing["clock_ticks_per_second"],
        "median_fortran_over_jax": statistics.median(fortran_wall_trials) / statistics.median(timings),
        "max_relative_error": float(np.max(np.abs(actual - expected) / expected)),
        "jax_batch_checksum": checksum,
        "fortran_batch_checksum": timing["checksum"],
        "timing_notes": "Both compared metrics are elapsed wall time. Fortran SYSTEM_CLOCK uses "
                        "int64 counters after an untimed warm pass; startup, I/O and checksum "
                        "are excluded. Separate compilation without LTO preserves repeated "
                        "non-PURE coagc calls. CPU_TIME is supplementary. JAX includes dispatch "
                        "and block_until_ready; input transfer, compilation and checksum are "
                        "outside warm timing. Trial order alternates. Thread settings are "
                        "recorded, not normalized; this is a scoped CPU leaf comparison, "
                        "not a full-model speedup or a controlled single-thread comparison.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 1 or args.trials < 1:
        parser.error("repetitions and trials must be positive")
    print(json.dumps(benchmark(args.repetitions, args.trials), indent=2))
