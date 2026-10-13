"""Accept the initializer against an actual checked native driver boundary."""
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from salsa_jax.grid import size_grid
from salsa_jax.initialization import size_distribution
from tools.initialization_fixture import load_native_case
from tools.initialization_oracle import build_oracle, run_oracle
from tools.salsa_fixture import load_native_case as load_salsa

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('compiled', [False, True])
def test_checked_native_initialization_and_grid(compiled):
    args, expected, _ = load_native_case()
    actual = (jax.jit(size_distribution) if compiled else size_distribution)(*args)
    np.testing.assert_allclose(np.broadcast_to(actual, expected.shape), expected, rtol=3e-13, atol=0)
    generated = size_grid()
    for field in ('vlolim', 'vhilim', 'dpmid'):
        np.testing.assert_array_equal(getattr(args[-1], field), getattr(generated, field))


def test_original_oracle_replays_actual_initialized_modes(tmp_path):
    args, expected, _ = load_native_case()
    exe, provenance = build_oracle(tmp_path)
    n, dpg, sigma, grid = args
    cases = dict(mode_number=n, dpg=dpg, sigma=sigma, lower_volume=grid.vlolim,
                 upper_volume=grid.vhilim, dry_diameter=grid.dpmid)
    actual, _ = run_oracle(exe, cases)
    np.testing.assert_allclose(actual['number'], expected[0, 0], rtol=3e-13, atol=0)
    manifest = json.loads((ROOT/'tests/fixtures/initialization-default/manifest.json').read_text())
    for name, digest in provenance['source_sha256'].items():
        if name in manifest['source_sha256']:
            assert digest == manifest['source_sha256'][name]
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest


def test_initial_number_reaches_native_salsa_with_recorded_unit_roundtrip():
    _, initialized, _ = load_native_case()
    salsa, _, provenance = load_salsa()
    manifest = json.loads((ROOT/'tests/fixtures/initialization-default/manifest.json').read_text())
    assert manifest['input_sha256'] == provenance['input_sha256']
    assert manifest['source_sha256'] == provenance['source_sha256']
    # conc2mmr and the HAM wrapper change representation before SALSA entry.
    np.testing.assert_allclose(initialized.reshape(-1, 17), salsa['number'], rtol=3e-13, atol=0)
