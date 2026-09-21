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


def run(executable, scheme, mode="absent", growth=1, operation="isolated", sulfate=1e13):
    return subprocess.run([str(executable), str(scheme), mode, str(growth), operation, str(sulfate)],
                          capture_output=True, text=True)


@pytest.mark.parametrize("scheme", [1, 2, 3])
def test_original_sulfate_nucleation_reads_absent_organic(original_nucleation, scheme):
    result = run(original_nucleation, scheme)
    assert result.returncode != 0
    assert "Index '0'" in result.stderr and "pc_gas" in result.stderr
