# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Scientific array records must preserve precision, shape and Fortran order."""
import subprocess

import numpy as np
import pytest

from tools.array_records import HEADER, FOOTER, array_lines, parse_records


def test_fortran_binary64_roundtrip_and_column_order(tmp_path):
    expected = np.array([1.2345678901234567, 1e-200, 1e200, -2.5, 0., 1.0000000000000002])
    constants = ", &\n".join(f"{value:.17e}_real64" for value in expected)
    statements = array_lines("matrix", "values", "R", ("2", "3"))
    statements += array_lines("scalar", "values(2,3)", "R", ())
    source = f"""PROGRAM roundtrip
USE iso_fortran_env, ONLY: real64
IMPLICIT NONE
INTEGER :: state_unit
REAL(real64) :: values(2,3)
values=RESHAPE([{constants}], [2,3])
OPEN(NEWUNIT=state_unit,FILE='records.txt',STATUS='new')
WRITE(state_unit,'(A)') '{HEADER}'
{chr(10).join(statements)}
WRITE(state_unit,'(A)') '{FOOTER}'
CLOSE(state_unit)
END PROGRAM
"""
    path = tmp_path / "roundtrip.f90"
    path.write_text(source)
    executable = tmp_path / "roundtrip"
    subprocess.run(["gfortran", "-fcheck=all", str(path), "-o", str(executable)], check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], cwd=tmp_path, check=True)
    data = parse_records((tmp_path / "records.txt").read_text())
    np.testing.assert_array_equal(data["matrix"]["values"], expected)
    assert data["matrix"]["shape"] == [2, 3]
    assert data["scalar"]["shape"] == []
    assert data["scalar"]["values"] == [expected[-1]]


@pytest.mark.parametrize("body", [
    "R x 1 2\n1", "R x 1 0", "R x 2 1\n1", "R x -1", "X x 0\n1",
    "R x 0\nNaN", "R x 0\nInf", "R x 0\n1 2", "I x 0\n1.5",
    "R x 0\n1\nR x 0\n2", "R x 0\n1\nextra", "R x 0\n1\n2",
])
def test_corrupted_records_rejected(body):
    with pytest.raises(ValueError):
        parse_records(f"{HEADER}\n{body}\n{FOOTER}\n")
