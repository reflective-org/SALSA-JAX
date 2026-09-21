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
