"""The first-call gate must reject nonfinite state and false completion claims."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.reference_scope import (CALL_END_ANCHOR, COMPLETION_LINES, FILENAME,
                                   FINITE_FIELDS, instrument_scope, parse_completion)
from tools.runtime_metadata import instrument_driver


def completion_text():
    return "\n".join(COMPLETION_LINES) + "\n"


@pytest.mark.parametrize("text", ["", "scope full\ncompleted_calls 1\n",
    completion_text().replace("completed_calls 1", "completed_calls 0"),
    completion_text().replace("completed_calls 1", "completed_calls 2"),
    completion_text().replace("first-microphysics", "full"),
    completion_text() + "completed_calls 1\n"])
def test_invalid_completion_rejected(text):
    with pytest.raises(ValueError):
        parse_completion(text)


def test_scope_overlay_composes_after_runtime_and_preserves_checkout(tmp_path):
    source = Path(__file__).resolve().parents[1] / "driver.F90"
    original = source.read_bytes()
    copied = tmp_path / "driver.F90"
    copied.write_bytes(original)
    runtime = instrument_driver(copied)
    scope = instrument_scope(copied)
    assert scope["input_driver_sha256"] == runtime["instrumented_driver_sha256"]
    assert scope["instrumented_driver_sha256"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    assert source.read_bytes() == original
    assert scope["overlay_sha256"] == hashlib.sha256((tmp_path / scope["overlay_file"]).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="already contains"):
        instrument_scope(copied)


@pytest.mark.parametrize("anchor", ["  IMPLICIT NONE\n", "  INTEGER :: ii, jj, kk\n", CALL_END_ANCHOR])
@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_duplicate_scope_anchor_leaves_copy_unchanged(tmp_path, anchor, count):
    source = Path(__file__).resolve().parents[1] / "driver.F90"
    copied = tmp_path / "driver.F90"
    copied.write_bytes(source.read_bytes().replace(anchor.encode(), anchor.encode() * count))
    before = copied.read_bytes()
    with pytest.raises(ValueError, match="exactly one"):
        instrument_scope(copied)
    assert copied.read_bytes() == before


@pytest.mark.parametrize("poison", [None, *FINITE_FIELDS])
def test_fortran_completion_written_only_after_all_finite_checks(tmp_path, poison):
    declarations = "\n".join(f"  REAL(KIND(1.D0)) :: {name}(1,1,1)" for name in FINITE_FIELDS)
    assignments = "\n".join(f"  {name} = 1.D0" for name in FINITE_FIELDS)
    if poison:
        assignments += f"\n  {poison} = ieee_value(1.D0, ieee_quiet_nan)"
    source = (
        "PROGRAM driver\n  USE, INTRINSIC :: ieee_arithmetic, ONLY: ieee_value, ieee_quiet_nan\n"
        "  IMPLICIT NONE\n  INTEGER :: ii, jj, kk\n  INTEGER :: jn, nclass\n"
        "  REAL(KIND(1.D0)) :: paclc, pgrvolm1, zpbl\n" + declarations + "\n"
        "  nclass = 1\n  paclc = 1.D0\n  pgrvolm1 = 1.D0\n  zpbl = 1.D0\n"
        + assignments + "\n  CALL ham_subm_interface( &\n" + CALL_END_ANCHOR +
        "      END DO\n  ERROR STOP 'scope did not stop'\nCONTAINS\n"
        "  SUBROUTINE ham_subm_interface(paclc, pgrvolm1, zpbl)\n"
        "    REAL(KIND(1.D0)), INTENT(IN) :: paclc, pgrvolm1, zpbl\n"
        "  END SUBROUTINE\nEND PROGRAM\n"
    )
    copied = tmp_path / "driver.F90"
    copied.write_text(source)
    instrument_scope(copied)
    executable = tmp_path / "scope-test"
    subprocess.run(["gfortran", "-std=f2008", "-O0", "-fcheck=all", str(copied), "-o", str(executable)],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], cwd=tmp_path, capture_output=True, text=True)
    if poison:
        assert result.returncode != 0
        assert f"nonfinite {poison}" in result.stderr
        assert not (tmp_path / FILENAME).exists()
    else:
        assert result.returncode == 0
        assert parse_completion((tmp_path / FILENAME).read_text()) == {
            "schema_version": 1, "scope": "first-microphysics", "completed_calls": 1}


@pytest.mark.parametrize("scope,marker,returncode,expected_status", [
    ("full", None, 0, "run_completed_unvalidated"),
    ("first-microphysics", completion_text(), 0, "first_microphysics_completed_unvalidated"),
    ("first-microphysics", None, 0, "scope_capture_failed"),
    ("first-microphysics", "scope full\n", 0, "scope_capture_failed"),
    ("first-microphysics", None, 1, "run_failed"),
    ("first-microphysics", completion_text(), 1, "run_failed"),
])
def test_scope_failure_manifest_and_full_default(tmp_path, monkeypatch, scope, marker, returncode, expected_status):
    from tools import fortran_reference

    source = tmp_path / "source"
    source.mkdir()
    real_root = fortran_reference.ROOT
    (source / "driver.F90").write_bytes((real_root / "driver.F90").read_bytes())
    for name in ("Makefile", "Makefile.in", ".depend", "configure", "namelist.echam", "lut_kappa.nc"):
        (source / name).write_text("test input\n")
    target = tmp_path / "run"
    monkeypatch.setattr(fortran_reference, "ROOT", source)
    monkeypatch.setattr(fortran_reference, "output", lambda *args: "test-version")

    def fake_run(command, directory, log, timeout):
        if command[0] == "make":
            (directory / "ham_box").write_bytes(b"test executable")
            return {"status": "finished", "returncode": 0}
        if marker is not None:
            (directory / FILENAME).write_text(marker)
        return {"status": "finished", "returncode": returncode}

    monkeypatch.setattr(fortran_reference, "logged_run", fake_run)
    argv = ["fortran_reference", "--config", str(real_root / "configs/fortran-reference.toml"),
            "--output", str(target), "--run"]
    if scope != "full":
        argv += ["--scope", scope]
    monkeypatch.setattr(sys, "argv", argv)
    accepted = expected_status in ("run_completed_unvalidated", "first_microphysics_completed_unvalidated")
    assert fortran_reference.main() == (0 if accepted else 1)
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["scope"] == scope
    assert manifest["status"] == expected_status
    if scope == "full":
        assert "scope_capture" not in manifest
        assert (target / "driver.F90").read_bytes() == (source / "driver.F90").read_bytes()
    else:
        assert manifest["scope_capture"]["input_driver_sha256"] == manifest["source_sha256"]["driver.F90"]
        assert manifest["scope_capture"]["instrumented_driver_sha256"] == manifest["compiled_driver_sha256"]
