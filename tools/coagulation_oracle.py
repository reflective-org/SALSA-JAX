"""Compile and run the unchanged coagc function extracted from its actual source."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("parkind1.F90", "mo_kind.F90", "mo_math_constants.F90",
           "mo_physical_constants.F90", "mo_ham_salsa_init.F90")
FLAGS = ("-O2", "-fno-fast-math", "-fno-lto", "-ffree-line-length-none")

DRIVER = """program reference_coagc
  use iso_fortran_env, only: int64
  use mo_kind, only: dp
  use extracted_coagc, only: coagc
  implicit none
  integer :: n, repetitions, i, j
  integer(int64) :: wall_started, wall_finished, clock_rate
  real(dp), allocatable :: cases(:,:), result(:)
  real(dp) :: started, finished, checksum
  read(*,*) n, repetitions
  allocate(cases(6,n), result(n))
  do i=1,n
    read(*,*) cases(:,i)
  end do
  ! Warm the code and data before timing. coagc is compiled separately without
  ! LTO and is not PURE: the caller cannot remove or hoist its repeated calls.
  do i=1,n
    result(i) = coagc(cases(1,i), cases(2,i), cases(3,i), cases(4,i), cases(5,i), cases(6,i))
  end do
  call cpu_time(started)
  call system_clock(wall_started, clock_rate)
  do j=1,repetitions
    do i=1,n
      result(i) = coagc(cases(1,i), cases(2,i), cases(3,i), cases(4,i), cases(5,i), cases(6,i))
    end do
  end do
  call system_clock(wall_finished)
  call cpu_time(finished)
  checksum = sum(result)
  write(*,'(4(ES26.17E3,1X))') real(wall_finished-wall_started,dp)/real(clock_rate,dp), &
                             finished-started, checksum, real(clock_rate,dp)
  do i=1,n
    write(*,'(ES26.17E3)') result(i)
  end do
end program reference_coagc
"""


def reference_cases():
    """Heterogeneous positive physical cases: diameter pairs, T, p and density."""
    diameters = np.geomspace(3e-9, 1e-5, 17)
    d1, d2, temperature, pressure = np.meshgrid(
        diameters, diameters, [230.0, 273.15, 310.0], [30000.0, 101325.0], indexing="ij"
    )
    # Different densities make mass/diameter indexing mistakes observable.
    m1 = np.pi / 6.0 * d1**3 * 1500.0
    m2 = np.pi / 6.0 * d2**3 * 2100.0
    return np.stack((d1, d2, m1, m2, temperature, pressure), axis=-1).reshape(-1, 6)


def build_oracle(build_directory, compiler="gfortran"):
    """Build using original constants and verbatim function text; return provenance."""
    directory = Path(build_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = (ROOT / SOURCES[-1]).read_text()
    match = re.search(r"^  FUNCTION coagc\(.*?^  END FUNCTION coagc", source, re.M | re.S)
    if match is None:
        raise ValueError("Could not find the original coagc function")
    function = match.group(0)
    extracted = directory / "extracted_coagc.f90"
    extracted.write_text("module extracted_coagc\ncontains\n" + function + "\nend module\n")
    driver = directory / "driver.f90"
    driver.write_text(DRIVER)
    executable = directory / "coagc_oracle"
    # Separate translation units and -fno-lto keep the non-PURE external call
    # opaque to the driver optimizer, so repeated identical inputs still execute.
    objects = []
    for source_path in [*[ROOT / name for name in SOURCES[:-1]], extracted, driver]:
        object_path = directory / (source_path.stem + ".o")
        command = [compiler, *FLAGS, "-c", str(source_path), "-o", str(object_path)]
        subprocess.run(command, cwd=directory, check=True, capture_output=True, text=True)
        objects.append(str(object_path))
    subprocess.run([compiler, *FLAGS, *objects, "-o", str(executable)],
                   cwd=directory, check=True, capture_output=True, text=True)
    provenance = {
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in SOURCES},
        "function_sha256": hashlib.sha256(function.encode()).hexdigest(),
        "driver_sha256": hashlib.sha256(DRIVER.encode()).hexdigest(),
        "separate_compilation_without_lto": True,
        "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        "flags": list(FLAGS),
    }
    return executable, provenance


def run_oracle(executable, cases, repetitions=1):
    """Return coefficients and warmed wall/CPU timings excluding startup and I/O."""
    cases = np.asarray(cases, dtype=np.float64)
    if cases.ndim != 2 or cases.shape[1] != 6 or len(cases) == 0 or repetitions < 1:
        raise ValueError("Expected nonempty (n, 6) cases and positive repetitions")
    payload = f"{len(cases)} {repetitions}\n" + "\n".join(
        " ".join(f"{value:.17e}" for value in case) for case in cases
    ) + "\n"
    result = subprocess.run([str(executable)], input=payload, capture_output=True,
                            text=True, check=True)
    values = np.fromstring(result.stdout, sep=" ")
    if len(values) != len(cases) + 4:
        raise ValueError("Unexpected Fortran oracle output")
    return values[4:], {
        "kernel_wall_seconds": float(values[0]), "kernel_cpu_seconds": float(values[1]),
        "checksum": float(values[2]), "clock_ticks_per_second": int(values[3]),
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="salsa-coagc-") as directory:
        executable, provenance = build_oracle(directory)
        cases = reference_cases()
        coefficients, timing = run_oracle(executable, cases)
    print(json.dumps({"provenance": provenance, "cases": cases.tolist(),
                      "coefficients": coefficients.tolist(), "timing": timing}, indent=2))
