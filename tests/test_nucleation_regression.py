# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Checked actual Fortran nucleation, isolated and nested in condensation."""
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "fortran-baseline-d87dc07"
FLAGS = ["-cpp", "-O0", "-g", "-fcheck=all", "-finit-real=snan",
         "-ffpe-trap=invalid,zero,overflow", "-ffree-line-length-none"]


def compile_nucleation(directory, source):
    nuclear = directory / "nucleation.F90"
    nuclear.write_bytes(source)
    dynamics = (ROOT / "mo_ham_salsa_dynamics.F90").read_text()
    routine = re.search(r"^  SUBROUTINE condensation\(.*?^  END SUBROUTINE condensation",
                        dynamics, re.M | re.S)[0]
    condensation = directory / "condensation.F90"
    condensation.write_text("MODULE extracted_condensation\nCONTAINS\n" + routine + "\nEND MODULE\n")
    sources = [ROOT / name for name in ("parkind1.F90", "mo_kind.F90",
               "mo_math_constants.F90", "mo_physical_constants.F90")]
    sources += [ROOT / "tests/fortran/nucleation_context.F90", nuclear, condensation,
                ROOT / "tests/fortran/nucleation_driver.F90"]
    executable = directory / "nucleation"
    subprocess.run(["gfortran", *FLAGS, *map(str, sources), "-o", str(executable)],
                   cwd=directory, check=True, capture_output=True, text=True)
    return executable


@pytest.fixture(scope="module")
def original_nucleation(tmp_path_factory):
    source = subprocess.check_output(["git", "show", f"{BASELINE}:mo_ham_salsa_nucleation.F90"], cwd=ROOT)
    return compile_nucleation(tmp_path_factory.mktemp("nucleation-original"), source)


@pytest.fixture(scope="module")
def current_nucleation(tmp_path_factory):
    return compile_nucleation(tmp_path_factory.mktemp("nucleation-current"),
                              (ROOT / "mo_ham_salsa_nucleation.F90").read_bytes())


def run(executable, scheme, mode="absent", growth=1, operation="isolated", sulfate=1e14):
    return subprocess.run([str(executable), str(scheme), mode, str(growth), operation, str(sulfate)],
                          capture_output=True, text=True)


@pytest.mark.parametrize("scheme", [1, 2, 3])
def test_original_sulfate_nucleation_reads_absent_organic(original_nucleation, scheme):
    result = run(original_nucleation, scheme)
    assert result.returncode != 0
    assert "Index '0'" in result.stderr and "pc_gas" in result.stderr


def values(result, size):
    assert result.returncode == 0, result.stderr
    data = np.fromstring(result.stdout, sep=" ")
    assert data.size == size
    assert np.isfinite(data).all()
    return data


@pytest.mark.parametrize("scheme", [1, 2, 3])
@pytest.mark.parametrize("growth", [1, 2, 3])
def test_absent_organic_equals_original_zero_gas(current_nucleation, original_nucleation, scheme, growth):
    current = values(run(current_nucleation, scheme, growth=growth), 16)
    original = values(run(original_nucleation, scheme, mode="zero", growth=growth), 16)
    np.testing.assert_array_equal(current, original)
    assert np.all(current[:4] >= 0)
    assert np.any(current[:4] > 0)
    if scheme == 1:
        # Binary nucleation is inactive at 280 K but active at 260 K here.
        np.testing.assert_array_equal(current[[0, 2]], 0)
        assert np.all(current[[1, 3]] > 0)
    np.testing.assert_array_equal(current[4:8], 0)
    np.testing.assert_array_equal(current[8:12], 1)
    np.testing.assert_array_equal(current[12:], 0)


@pytest.mark.parametrize("scheme", [1, 2, 3])
@pytest.mark.parametrize("growth", [1, 2, 3])
def test_registered_organic_preserves_original_result(current_nucleation, original_nucleation, scheme, growth):
    current = values(run(current_nucleation, scheme, mode="present", growth=growth), 16)
    original = values(run(original_nucleation, scheme, mode="present", growth=growth), 16)
    np.testing.assert_array_equal(current, original)


@pytest.mark.parametrize("scheme", [1, 2, 3])
def test_coupled_nucleation_closes_sulfate_budget(current_nucleation, scheme):
    data = values(run(current_nucleation, scheme, operation="coupled"), 14)
    gas, volume, new_number = data[:4], data[4:8], data[8:12]
    molecular_volume = (98e-3 / 6.02214179e23) / 1841.0
    assert np.all((gas > 0) & (gas < 1e14))
    assert np.all(new_number >= 0)
    assert np.any(new_number > 0)
    np.testing.assert_allclose(volume / molecular_volume, 1e14 - gas, rtol=2e-14)
    assert np.all(new_number * 158.79 <= volume / molecular_volume)
    assert data[12] == data[13] == 0


@pytest.mark.parametrize("scheme", [1, 2, 3])
def test_zero_sulfate_has_no_formation(current_nucleation, scheme):
    data = values(run(current_nucleation, scheme, sulfate=0), 16)
    np.testing.assert_array_equal(data[:8], 0)


def test_missing_ammonia_remains_explicit_error(current_nucleation):
    result = run(current_nucleation, 4)
    assert result.returncode != 0
    assert "Specify NH3 mixing ratio" in result.stdout


def test_empty_background_retains_activation_source(current_nucleation):
    data = values(run(current_nucleation, 2, mode="empty"), 16)
    # With no scavenging, the activation cells retain act_coeff * [H2SO4].
    np.testing.assert_allclose(data[[0, 2, 3]], 1e-7 * 1e14 * 158.79, rtol=2e-15)
    np.testing.assert_array_equal(data[4:8], 0)
