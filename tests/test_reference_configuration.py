# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Controlled first-call activation changes only an isolated driver copy."""
import hashlib
from pathlib import Path
import pytest
from tools.reference_configuration import configure_driver
from tools.runtime_metadata import instrument_driver
from tools.reference_scope import instrument_scope

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {'nsnucl': 2, 'nj3': 1, 'sulfate_molecules_m3': 1e14}


def test_profile_preserves_source_and_native_gas_conversions(tmp_path):
    original = (ROOT / 'driver.F90').read_bytes()
    path = tmp_path / 'driver.F90'
    path.write_bytes(original)
    evidence = configure_driver(path, SETTINGS)
    assert (ROOT / 'driver.F90').read_bytes() == original
    assert evidence['original_driver_sha256'] == hashlib.sha256(original).hexdigest()
    assert evidence['configured_driver_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    text = path.read_text()
    assert text.index('nsnucl = 2') < text.index('CALL start_ham')
    assert text.index('IF (ii == 1) zgas') < text.index('zgas(1:kproma,:,:) = zgas(1:kproma,:,:) * 1.e-6_dp')
    instrument_driver(path)
    instrument_scope(path)
    assert 'SALSA_RUNTIME_METADATA_V1' in path.read_text()
    with pytest.raises(ValueError, match='already'):
        configure_driver(path, SETTINGS)


@pytest.mark.parametrize('updates', [{'nsnucl': 4}, {'nsnucl': True}, {'nj3': 0},
                                    {'nj3': 1.5}, {'sulfate_molecules_m3': float('nan')},
                                    {'sulfate_molecules_m3': -1}, {'unused': 1}])
def test_invalid_profile_fails_without_writes(tmp_path, updates):
    path = tmp_path / 'driver.F90'
    original = (ROOT / path.name).read_bytes()
    path.write_bytes(original)
    with pytest.raises(ValueError, match='reference_case'):
        configure_driver(path, {**SETTINGS, **updates})
    assert path.read_bytes() == original
    assert not (tmp_path / 'reference-configuration.patch').exists()


def test_ambiguous_anchor_fails_without_writes(tmp_path):
    path = tmp_path / 'driver.F90'
    original = (ROOT / path.name).read_text() + '  CALL start_ham\n'
    path.write_text(original)
    with pytest.raises(ValueError, match='anchor'):
        configure_driver(path, SETTINGS)
    assert path.read_text() == original


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("active", [False, True])
def test_recalc_only_or_combined_profile_composes_before_initialization(tmp_path, enabled, active):
    path = tmp_path / "driver.F90"
    original = (ROOT / path.name).read_bytes()
    path.write_bytes(original)
    metadata = configure_driver(path, SETTINGS if active else None,
                                process_switches={"recalc": enabled})
    text = path.read_text()
    assignment = "recalc = " + (".TRUE." if enabled else ".FALSE.")
    assert "reference_recalc" not in text
    assert text.count("USE mo_ham_salsactl") == original.decode().count("USE mo_ham_salsactl")
    assert text.index(assignment) < text.index("CALL start_ham")
    assert ("IF (ii == 1) zgas" in text) == active
    assert metadata["process_switches"] == {"recalc": enabled}
    assert metadata["settings"] == (SETTINGS if active else None)
    assert (b"\r\n" in path.read_bytes()) == (b"\r\n" in original)
    instrument_driver(path)
    instrument_scope(path)
    assert (ROOT / path.name).read_bytes() == original


@pytest.mark.parametrize("switches", [{}, {"recalc": 1}, {"recalc": "true"},
                                    {"recalc": None}, {"recalc": True, "lscoag": False}])
def test_invalid_process_switches_refuse_all_writes(tmp_path, switches):
    path = tmp_path / "driver.F90"
    original = (ROOT / path.name).read_bytes()
    path.write_bytes(original)
    with pytest.raises(ValueError, match="process_switches"):
        configure_driver(path, process_switches=switches)
    assert path.read_bytes() == original
    assert not (tmp_path / "reference-configuration.patch").exists()


def test_recalc_initialization_anchor_must_be_unique(tmp_path):
    path = tmp_path / "driver.F90"
    original = (ROOT / path.name).read_text() + "  CALL start_ham\n"
    path.write_text(original)
    with pytest.raises(ValueError, match="anchor"):
        configure_driver(path, process_switches={"recalc": True})
    assert path.read_text() == original
