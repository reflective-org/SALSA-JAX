# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Long pure-sulfate trajectories compare every step to checked original SALSA."""
from dataclasses import replace

import jax
import numpy as np
import pytest

from salsa_jax.sulfuric_acid import simulate
from tools.sulfuric_acid_oracle import build_oracle, run_oracle
from tools.sulfuric_acid_validation import compare, reference_case, summarize
from tools.validate_sulfuric_acid import conditioning_check, scenarios

jax.config.update('jax_enable_x64', True)
CASES = scenarios()
base = CASES['configured-source']
CASES['half-timestep-source'] = replace(base, steps=2*base.steps, timestep_s=base.timestep_s/2)


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    executable, provenance = build_oracle(tmp_path_factory.mktemp('sulfur-trajectories'))
    assert provenance['checked']
    return executable


@pytest.mark.parametrize('label', CASES)
def test_all_steps_match_fortran_and_close_sulfur(reference, label):
    config = CASES[label]
    initial, history, mapping, _ = simulate(config)
    expected, _ = run_oracle(reference, reference_case(config),
        config.h2so4_source_molecules_m3_s, steps=config.steps)
    metrics = compare(history, expected)
    assert metrics['passed'], metrics
    for values in (history, expected):
        report = summarize(config, initial, values, mapping)
        assert report['sulfur_budget_passed'], report
        assert report['pure_sulfate_soluble_only']
    if label == 'empty-background':
        # Reproduce, do not silently fix, the native positive-background-sink gate.
        np.testing.assert_array_equal(history.number, 0.)
        np.testing.assert_array_equal(history.mass, 0.)
        added = np.arange(1, config.steps+1)*config.timestep_s*config.h2so4_source_molecules_m3_s
        np.testing.assert_array_equal(history.gas[:, 0, 1], config.initial_h2so4_molecules_m3+added)
    elif label == 'coagulation-only':
        assert history.number[-1].sum() < initial['number'].sum()
        np.testing.assert_array_equal(history.gas[:, 0, 1], config.initial_h2so4_molecules_m3)
    elif label == 'condensation-only':
        np.testing.assert_allclose(history.number.sum(-1), initial['number'].sum(), rtol=1e-11)
        assert history.gas[-1, 0, 1] < config.initial_h2so4_molecules_m3
    else:
        # These configured temperatures/concentrations deliberately activate all
        # three nucleation schemes; a matching but inactive run is insufficient.
        assert history.number[-1].sum() > initial['number'].sum()


def test_sparse_coagulation_transfer_has_native_roundoff_sensitivity(reference):
    report = conditioning_check(reference)
    assert not report['comparison']['passed']
    for field in ('mass', 'number', 'water_mass'):
        values = report['receiving_bin_fields'][field]
        assert values['native_one_ulp_relative_sensitivity'] > 1e-10
        # The translation discrepancy is comparable to native input roundoff;
        # report it as unaccepted strict parity, never erase it with this bound.
        assert values['jax_relative_error'] <= 2*values['native_one_ulp_relative_sensitivity']
    assert report['jax']['sulfur_budget_passed']
    assert report['fortran']['sulfur_budget_passed']
