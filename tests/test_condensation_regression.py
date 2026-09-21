"""Bounds/uninitialized regressions using the actual isolated Fortran routine.

Fixture modules provide initialized state, not alternate physics. The nucleation
stub aborts if called, so this regression deliberately isolates condensation.
"""

from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "fortran-baseline-d87dc07"
FLAGS = ["-cpp", "-O0", "-g", "-fcheck=all", "-finit-real=snan",
         "-ffpe-trap=invalid,zero,overflow", "-ffree-line-length-none"]


def compile_condensation(directory, source):
    match = re.search(r"^  SUBROUTINE condensation\(.*?^  END SUBROUTINE condensation",
                      source, re.MULTILINE | re.DOTALL)
    assert match is not None
    extracted = directory / "condensation.F90"
    extracted.write_text("MODULE extracted_condensation\nCONTAINS\n" + match[0] + "\nEND MODULE\n")
    sources = [ROOT / name for name in ["parkind1.F90", "mo_kind.F90",
                                       "mo_math_constants.F90", "mo_physical_constants.F90"]]
    sources += [ROOT / "tests/fortran/condensation_context.F90", extracted,
                ROOT / "tests/fortran/condensation_driver.F90"]
    executable = directory / "condensation"
    subprocess.run(["gfortran", *FLAGS, *map(str, sources), "-o", str(executable)],
                   cwd=directory, capture_output=True, text=True, check=True)
    return executable


@pytest.fixture(scope="module")
def original_condensation(tmp_path_factory):
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:mo_ham_salsa_dynamics.F90"], cwd=ROOT, text=True)
    return compile_condensation(tmp_path_factory.mktemp("condensation-original"), source)


def test_original_unregistered_organic_gas_reproduces_d001(original_condensation):
    result = subprocess.run([str(original_condensation), "absent"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Index '0'" in result.stderr
    assert "pc_gas" in result.stderr


@pytest.fixture(scope="module")
def current_condensation(tmp_path_factory):
    return compile_condensation(tmp_path_factory.mktemp("condensation-current"),
                                (ROOT / "mo_ham_salsa_dynamics.F90").read_text())


def run_current(executable, mode):
    import numpy as np
    result = subprocess.run([str(executable), mode], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    values = np.fromstring(result.stdout, sep=" ")
    assert values.shape == (5,)
    assert np.isfinite(values).all()
    return values


@pytest.mark.parametrize("mode", ["absent", "zero"])
def test_condensation_without_organic_preserves_sulfate_budget(current_condensation, mode):
    sulfate, organic, sulfate_volume, organic_volume, number_change = run_current(current_condensation, mode)
    assert 0 < sulfate < 1e12
    assert organic == organic_volume == number_change == 0
    molecular_volume = (98e-3 / 6.02214179e23) / 1841.0
    assert sulfate_volume / molecular_volume == pytest.approx(1e12 - sulfate, rel=2e-14)


def test_condensation_with_no_particles_has_no_sink(current_condensation):
    sulfate, organic, sulfate_volume, organic_volume, number_change = run_current(current_condensation, "empty")
    assert sulfate == 1e12
    assert organic == sulfate_volume == organic_volume == number_change == 0


def test_registered_organic_without_nucleation_closes_budget(current_condensation):
    sulfate, organic, sulfate_volume, organic_volume, number_change = run_current(current_condensation, "present")
    assert sulfate == sulfate_volume == number_change == 0
    assert 0 < organic < 2e12
    molecular_volume = (150e-3 / 6.02214179e23) / 2000.0
    assert organic_volume / molecular_volume == pytest.approx(2e12 - organic, rel=2e-14)


def test_organic_threshold_remains_inactive(current_condensation):
    _, organic, _, organic_volume, number_change = run_current(current_condensation, "threshold")
    assert organic == 1e-10
    assert organic_volume == number_change == 0


def test_registered_organic_without_particles_has_no_sink(current_condensation):
    sulfate, organic, sulfate_volume, organic_volume, number_change = run_current(current_condensation, "empty_organic")
    assert organic == 2e12
    assert sulfate == sulfate_volume == organic_volume == number_change == 0
