"""Actual driver wiring and unchanged activation kernels reproduce D003."""
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "fortran-baseline-d87dc07"
FLAGS = ["-cpp", "-O0", "-g", "-fcheck=all", "-finit-real=snan",
         "-ffpe-trap=invalid,zero,overflow", "-ffree-line-length-none"]


def compile_updraft(directory, driver_source):
    activ_source = (ROOT / "mo_activ.F90").read_text()
    routines = []
    for name in ("activ_updraft", "aero_activ_updraft_pdf"):
        match = re.search(r"^  SUBROUTINE " + name + r"\(.*?^  END SUBROUTINE " + name,
                          activ_source, re.MULTILINE | re.DOTALL)
        assert match is not None
        routines.append(match[0])
    context = (ROOT / "tests/fortran/updraft_context.F90").read_text()
    context_path = directory / "updraft_context.F90"
    context_path.write_text(context.replace("! ACTUAL_ROUTINES", "\n".join(routines)))
    driver = (ROOT / "tests/fortran/updraft_driver.F90").read_text()
    patterns = {
        "DECLARATION": r"^\s*REAL\(wp\), ALLOCATABLE :: zvervel[^\n]*",
        "ALLOCATION": r"^\s*ALLOCATE\(zvervel[^\n]*",
        "ASSIGNMENT": r"^\s*zvervel\([^\n]*= *[^\n]*zin\(85,25,47,1\)[^\n]*",
        "CALL": r"^\s*CALL activ_updraft\([^\n]*\n[^\n]*",
    }
    for marker, pattern in patterns.items():
        match = re.search(pattern, driver_source, re.MULTILINE)
        assert match is not None, marker
        driver = driver.replace("! DRIVER_" + marker, match[0].strip())
    driver_path = directory / "updraft_driver.F90"
    driver_path.write_text(driver)
    sources = [ROOT / name for name in ("parkind1.F90", "mo_kind.F90",
               "mo_math_constants.F90", "mo_physical_constants.F90", "mo_param_switches.F90")]
    executable = directory / "updraft"
    subprocess.run(["gfortran", *FLAGS, *map(str, sources), str(context_path),
                    str(driver_path), "-o", str(executable)], cwd=directory,
                   capture_output=True, text=True, check=True)
    return executable


@pytest.fixture(scope="module")
def original_updraft(tmp_path_factory):
    source = subprocess.check_output(["git", "show", f"{BASELINE}:driver.F90"],
                                     cwd=ROOT, text=True)
    return compile_updraft(tmp_path_factory.mktemp("updraft-original"), source)


@pytest.fixture(scope="module")
def current_updraft(tmp_path_factory):
    return compile_updraft(tmp_path_factory.mktemp("updraft-current"),
                           (ROOT / "driver.F90").read_text())


def test_original_driver_reads_poisoned_first_plane(original_updraft):
    result = subprocess.run([str(original_updraft), "0.35", "20"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert any(signal in result.stderr for signal in ("SIGFPE", "SIGILL"))
    assert "activ_updraft" in result.stderr


@pytest.mark.parametrize("velocity", [-0.6, 0.0, 0.35])
@pytest.mark.parametrize("bins", [1, 20])
def test_current_driver_preserves_velocity_and_initializes_all_cells(current_updraft, velocity, bins):
    result = subprocess.run([str(current_updraft), str(velocity), str(bins)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    values = np.fromstring(result.stdout, sep=" ")
    assert values.size == 8 * bins
    assert np.isfinite(values).all()
    actual_w = values[:4*bins].reshape((2, 2, bins), order="F")
    actual_pdf = values[4*bins:].reshape((2, 2, bins), order="F")
    widths = (np.arange(bins) + 0.5) * (4 * 0.8 / bins)
    expected_w = np.broadcast_to(widths, (2, 2, bins))
    expected_pdf = np.exp(-(expected_w - velocity)**2 / (2 * 0.8**2)) / (np.sqrt(2*np.pi)*0.8)
    np.testing.assert_allclose(actual_w, expected_w, rtol=2e-15, atol=0)
    np.testing.assert_allclose(actual_pdf, expected_pdf, rtol=5e-15, atol=0)


def test_supplied_activation_velocity_metadata_is_meters_per_second():
    header = subprocess.check_output(
        ["ncdump", "-h", str(ROOT / "input/GlobalTraj-CE_200801.01_activ_1step.nc")], text=True)
    assert re.search(r'\bW:units = "m s-1"', header)
    assert re.search(r'\bW:long_name = "total vertical velocity for activation"', header)
