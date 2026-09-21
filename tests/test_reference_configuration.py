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
