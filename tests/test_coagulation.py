# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Compare the first JAX leaf against the actual repository Fortran routine."""

import jax
import numpy as np
import pytest

from salsa_jax.processes.coagulation import coagulation_coefficient
from tools.coagulation_oracle import build_oracle, reference_cases, run_oracle

jax.config.update("jax_enable_x64", True)

# Measured eager/JIT errors are <1e-14 on ARM64 with gfortran 16.1/JAX 0.11.2.
# Allow headroom for compiler evaluation order, with no absolute floor hiding
# errors in the very small collision coefficients.
PARITY_RTOL = 1e-13


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    executable, provenance = build_oracle(tmp_path_factory.mktemp("coagc"))
    cases = reference_cases()
    expected, timing = run_oracle(executable, cases)
    assert timing["kernel_wall_seconds"] > 0
    assert timing["clock_ticks_per_second"] > 0
    np.testing.assert_allclose(timing["checksum"], expected.sum(), rtol=1e-13, atol=0)
    return cases, expected, provenance


def test_scalar(reference):
    cases, expected, _ = reference
    result = coagulation_coefficient(*cases[0])
    assert result.shape == ()
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected[0], rtol=PARITY_RTOL, atol=0)


@pytest.mark.parametrize("compile_kernel", [False, True])
def test_heterogeneous_fortran_parity(reference, compile_kernel):
    cases, expected, provenance = reference
    kernel = jax.jit(coagulation_coefficient) if compile_kernel else coagulation_coefficient
    result = np.asarray(kernel(*cases.T))
    assert np.all(np.isfinite(result))
    assert np.all(result > 0)
    np.testing.assert_allclose(result, expected, rtol=PARITY_RTOL, atol=0)
    assert len(provenance["source_sha256"]) == 5
    assert len(provenance["driver_sha256"]) == 64
    assert provenance["separate_compilation_without_lto"]


def test_symmetry_and_broadcast(reference):
    cases, _, _ = reference
    d1, d2, m1, m2, temperature, pressure = cases.T
    direct = coagulation_coefficient(d1, d2, m1, m2, temperature, pressure)
    swapped = coagulation_coefficient(d2, d1, m2, m1, temperature, pressure)
    np.testing.assert_allclose(direct, swapped, rtol=2e-15, atol=0)
    broadcast = coagulation_coefficient(d1[:17, None], d2[None, :13],
                                       m1[:17, None], m2[None, :13], 280.0, 101325.0)
    assert broadcast.shape == (17, 13)
    assert np.all(np.isfinite(broadcast))
    assert np.all(broadcast > 0)
