"""Reject incomplete runtime evidence and fail closed when source anchors drift."""

from pathlib import Path
import json
import sys

import pytest

from tools.runtime_metadata import FIELDS, instrument_driver, parse_metadata


def valid_metadata():
    integers = dict.fromkeys(FIELDS["I"], 1)
    integers.update(dp_bits=64, dp_precision=15, dp_range=307, default_real_bits=32,
                    ntrac=110, nspec=10, nclass=17, nsol=10, subm_ngasspec=2,
                    in1a=1, fn1a=3, in2a=4, fn2a=10, in2b=11, fn2b=17)
    lines = ["SALSA_RUNTIME_METADATA 1"]
    lines += [f"I {key} {integers[key]}" for key in FIELDS["I"]]
    lines += [f"L {key} T" for key in FIELDS["L"]]
    lines += [f"R {key} 1.00000000000000000E+000" for key in FIELDS["R"]]
    lines += ["G 1 4 1", "G 2 7 2", "END_SALSA_RUNTIME_METADATA"]
    return "\n".join(lines) + "\n"


def test_typed_values_and_gas_mapping():
    metadata = parse_metadata(valid_metadata())
    assert metadata["values"]["dp_bits"] == 64
    assert metadata["values"]["lscond"] is True
    assert metadata["values"]["time_step_seconds"] == 1.0
    assert metadata["gas_species"][1] == {"index": 2, "species_index": 7, "conversion_code": 2}


def test_absent_organic_gas_retains_zero_sentinel():
    metadata = parse_metadata(valid_metadata().replace("I isubm_ocnv 1", "I isubm_ocnv 0"))
    assert metadata["values"]["isubm_ocnv"] == 0


@pytest.mark.parametrize("old,new", [
    ("SALSA_RUNTIME_METADATA 1", "SALSA_RUNTIME_METADATA 2"),
    ("END_SALSA_RUNTIME_METADATA", ""),
    ("I dp_bits 64\n", ""),
    ("I dp_bits 64", "I dp_bits 64\nI dp_bits 64"),
    ("L lscond T", "L lscond yes"),
    ("I dp_bits 64", "R dp_bits 64"),
    ("I kproma 1", "I kproma 2"),
    ("I fn1a 3", "I fn1a 12"),
    ("G 2 7 2", "G 2 99 2"),
    ("I isubm_ocnv 1", "I isubm_ocnv 3"),
    ("G 2 7 2", "G 1 7 2"),
    ("G 2 7 2\n", ""),
    ("R time_step_seconds 1.00000000000000000E+000", "R time_step_seconds NaN"),
    ("R time_step_seconds 1.00000000000000000E+000", "R time_step_seconds 0"),
])
def test_malformed_metadata_fails(old, new):
    with pytest.raises(ValueError):
        parse_metadata(valid_metadata().replace(old, new))


def test_overlay_preserves_original_and_records_hashes(tmp_path):
    source = Path(__file__).resolve().parents[1] / "driver.F90"
    original = source.read_bytes()
    copied = tmp_path / "driver.F90"
    copied.write_bytes(original)
    metadata = instrument_driver(copied)
    assert source.read_bytes() == original
    assert metadata["original_driver_sha256"] != metadata["instrumented_driver_sha256"]
    assert len(metadata["overlay_sha256"]) == 64
    assert "pap(1,1)" in copied.read_text()
    assert "capture_gasspec(capture_index)" in copied.read_text()
    assert (tmp_path / metadata["overlay_file"]).exists()
    with pytest.raises(ValueError, match="already contains"):
        instrument_driver(copied)


@pytest.mark.parametrize("anchor", ["  IMPLICIT NONE\n", "  INTEGER :: ii, jj, kk\n", "   zfacn  = 1.0e-06_dp\n"])
@pytest.mark.parametrize("replacement_count", [0, 2])
def test_changed_or_ambiguous_overlay_anchor_fails_without_edit(tmp_path, anchor, replacement_count):
    source = Path(__file__).resolve().parents[1] / "driver.F90"
    copied = tmp_path / "driver.F90"
    copied.write_bytes(source.read_bytes().replace(anchor.encode(), anchor.encode() * replacement_count))
    before = copied.read_bytes()
    with pytest.raises(ValueError, match="exactly one"):
        instrument_driver(copied)
    assert copied.read_bytes() == before


@pytest.mark.parametrize("emit_metadata", [True, False])
def test_harness_requires_capture_even_when_executable_succeeds(tmp_path, monkeypatch, emit_metadata):
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
        elif emit_metadata:
            (directory / "runtime-metadata.txt").write_text(valid_metadata())
        return {"status": "finished", "returncode": 0}

    monkeypatch.setattr(fortran_reference, "logged_run", fake_run)
    monkeypatch.setattr(sys, "argv", ["fortran_reference", "--config",
                        str(real_root / "configs/fortran-reference.toml"),
                        "--output", str(target), "--run", "--capture-runtime"])
    assert fortran_reference.main() == (0 if emit_metadata else 1)
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["status"] == ("run_completed_unvalidated" if emit_metadata else "runtime_capture_failed")
    assert manifest["runtime_capture"]["status"] == ("captured" if emit_metadata else "capture_failed")
    if emit_metadata:
        assert manifest["runtime_capture"]["metadata"]["values"]["dp_bits"] == 64
        assert len(manifest["runtime_capture"]["runtime_sha256"]) == 64
