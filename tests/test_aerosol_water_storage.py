# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Check the box driver's actual water allocation against the HAM output extent."""

from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def compile_driver_storage(directory, source):
    # Compile the actual driver storage declaration/allocation in a small context.
    # A runtime contract check prevents the historical overwrite from corrupting
    # the test process before it can produce a useful diagnostic.
    if re.search(r"zww\(\s*:\s*,\s*:\s*,\s*:\s*\)", source):
        declaration = "REAL(dp), ALLOCATABLE :: zww(:,:,:)"
        allocation = re.search(r"^\s*ALLOCATE\(zww\([^\n]+", source, re.M)[0].strip()
    else:
        declaration = "REAL(dp) :: " + re.search(r"zww\(kbdim,klev,[^)]+\)", source)[0]
        allocation = ""
    nmod = re.search(r"INTEGER, PARAMETER :: nmod = \d+", source)[0]
    program = f"""PROGRAM water_storage
  USE iso_fortran_env, ONLY: dp => real64
  IMPLICIT NONE
  INTEGER, PARAMETER :: kbdim=1,klev=1
  {nmod}
  INTEGER :: nclass,i
  CHARACTER(len=16) :: arg
  {declaration}
  CALL get_command_argument(1,arg)
  READ(arg,*) nclass
  {allocation}
  IF (SIZE(zww,3) /= nclass) ERROR STOP 'water output extent does not match nclass'
  CALL fill_water(zww,nclass)
  IF (ANY(zww(1,1,:) /= [(REAL(i,dp),i=1,nclass)])) ERROR STOP 'wrong water output'
  PRINT *, SIZE(zww,3)
CONTAINS
  SUBROUTINE fill_water(water,classes)
    INTEGER, INTENT(IN) :: classes
    REAL(dp), INTENT(OUT) :: water(kbdim,klev,classes)
    INTEGER :: j
    DO j=1,classes
      water(:,:,j)=REAL(j,dp)
    END DO
  END SUBROUTINE
END PROGRAM
"""
    path = directory / "water_storage.f90"
    path.write_text(program)
    executable = directory / "water_storage"
    subprocess.run(["gfortran", "-O0", "-fcheck=all", str(path), "-o", str(executable)],
                   cwd=directory, check=True, capture_output=True, text=True)
    return executable


def test_original_driver_water_extent_is_too_small(tmp_path):
    original = subprocess.check_output(["git", "show", "fortran-baseline-d87dc07:driver.F90"],
                                       cwd=ROOT, text=True)
    exe = compile_driver_storage(tmp_path, original)
    result = subprocess.run([str(exe), "17"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "water output extent does not match nclass" in result.stderr


@pytest.mark.parametrize("classes", [7, 17])
def test_water_output_matches_selected_model(tmp_path, classes):
    exe = compile_driver_storage(tmp_path, (ROOT / "driver.F90").read_text())
    result = subprocess.run([str(exe), str(classes)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) == classes
