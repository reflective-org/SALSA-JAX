# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Adapt checked nested nucleation and parent condensation captures to JAX inputs.

The adapter preserves native (horizontal, vertical) ordering, flattening cells
in C order after reconstructing every Fortran-column-major record. Canonical gas
slots are SO2/H2SO4/OC; an unregistered organic gas is padded with zero.
"""
from pathlib import Path

import numpy as np

from salsa_jax import constants
from tools.condensation_capture import load_fixture as load_condensation_fixture
from tools.nucleation_capture import load_fixture as load_nucleation_fixture

ROOT = Path(__file__).resolve().parents[1]
FIXED_CONSTANTS = {
    'avo': constants.AVOGADRO, 'argas': constants.GAS_CONSTANT,
    'ak': constants.BOLTZMANN, 'p0sl_bg': constants.REFERENCE_PRESSURE,
    'pi': constants.PI, 'd_sa': constants.SULFURIC_ACID_DIAMETER,
    'n3': constants.NUCLEATION_CLUSTER_MOLECULES,
    'reglim1': constants.NUCLEATION_TARGET_DIAMETER,
    'act_coeff': constants.NUCLEATION_ACTIVATION_COEFFICIENT,
}


def record(capture, stage, name):
    item = capture[stage][name]
    return np.asarray(item['values']).reshape(item['shape'], order='F')


def adapt_captures(nucleation, condensation):
    """Return oracle cases, actual expected sources/state, and effective metadata.

    Refuse mismatched capture pairs or globals outside the fixed native/JAX
    oracle contract, including activation coefficients the oracle cannot vary.
    """
    def native(name):
        return record(nucleation, 'entry', name)
    for name, expected in FIXED_CONSTANTS.items():
        if float(native(name)) != expected:
            raise ValueError(f'Unsupported captured fixed constant: {name}')
    for name in ('pnaero', 'pvols', 'pc_gas', 'pdwet', 'ptemp', 'prh', 'ppres', 'ptstep', 'ppbl', 'krow'):
        if not np.array_equal(native(name), record(nucleation, 'exit', name)):
            raise ValueError(f'Nested nucleation modified state: {name}')
    for name in ('zj3n3', 'zxsa', 'zxocnv'):
        if np.any(native(name) != 0):
            raise ValueError(f'Nested source was not initialized to zero: {name}')
    pairs = [('pvols', 'zvols'), ('pdwet', 'zdwet')] + [(name, name) for name in (
        'pnaero', 'pc_gas', 'ptemp', 'prh', 'ppres', 'ptstep', 'ppbl', 'krow',
        'dimensions', 'boundaries', 'gas_indices', 'massacc', 'species_molar_mass',
        'species_density', 'd_sa', 'avo', 'argas', 'ak', 'p0sl_bg', 'pi')]
    for nested_name, parent_name in pairs:
        if not np.array_equal(native(nested_name), record(condensation, 'entry', parent_name)):
            raise ValueError(f'Parent/nested capture mismatch: {nested_name}')
    scheme, growth, nsoa, bits = native('options').astype(int)
    if (not np.array_equal(record(condensation, 'entry', 'options'), [scheme, nsoa, bits]) or
            int(record(condensation, 'entry', 'nj3')) != growth):
        raise ValueError('Parent/nested nucleation options mismatch')
    if scheme not in (1, 2, 3) or growth not in (1, 2, 3) or nsoa != 0 or bits != 64:
        raise ValueError('Unsupported captured nucleation configuration')
    sulfate_index, organic_index = native('gas_indices').astype(int)
    gas = native('pc_gas')
    if (gas.shape[-1], sulfate_index, organic_index) not in ((2, 2, 0), (3, 2, 3)):
        raise ValueError('Unsupported canonical SO2/H2SO4/OC gas mapping')
    number = native('pnaero')
    horizontal, vertical, bins = number.shape
    if bins != 17 or native('pvols').shape != (horizontal, vertical, 17, 5):
        raise ValueError('Unsupported captured bin/material shape')
    count = horizontal * vertical
    # Fortran levels are one-based, and the PBL includes the boundary level.
    levels = np.arange(1, vertical + 1)[None, :]
    boundary_layer = levels >= native('ppbl')[:, None]
    if native('ppbl').shape != (horizontal,) or np.any(native('ppbl') < 1) or np.any(native('ppbl') > vertical):
        raise ValueError('Invalid captured boundary-layer index')
    def canonical_gas(stage, capture):
        source = record(capture, stage, 'pc_gas')
        canonical = np.zeros((horizontal, vertical, 3), dtype=np.float64)
        canonical[..., :source.shape[-1]] = source
        return canonical.reshape(count, 3)
    masses, densities = native('species_molar_mass'), native('species_density')
    cases = {
        'labels': [f'native-h{h + 1}-level{k + 1}' for h in range(horizontal) for k in range(vertical)],
        'number': number.reshape(count, 17), 'volume': native('pvols').reshape(count, 17, 5),
        'gas': canonical_gas('entry', nucleation), 'wet_diameter': native('pdwet').reshape(count, 17),
        'temperature': native('ptemp').reshape(count), 'relative_humidity': native('prh').reshape(count),
        'pressure': native('ppres').reshape(count), 'timestep': np.full(count, float(native('ptstep'))),
        'scheme': np.full(count, scheme), 'growth': np.full(count, growth),
        'boundary_layer': boundary_layer.reshape(count), 'organic_enabled': np.full(count, organic_index > 0),
        'mass_accommodation': np.broadcast_to(native('massacc'), (count, 17)).copy(),
        'sulfate_molar_mass': np.full(count, masses[0]), 'organic_molar_mass': np.full(count, masses[1]),
        'sulfate_density': np.full(count, densities[0]), 'organic_density': np.full(count, densities[1]),
    }
    expected = {
        'molecular_rate': record(nucleation, 'exit', 'zj3n3').reshape(count, 2),
        'fractions': np.stack((record(nucleation, 'exit', 'zxsa'), record(nucleation, 'exit', 'zxocnv')), axis=-1).reshape(count, 2),
        'number': record(condensation, 'exit', 'pnaero').reshape(count, 17),
        'volume': record(condensation, 'exit', 'zvols').reshape(count, 17, 5),
        'gas': canonical_gas('exit', condensation),
    }
    metadata = {'native_cell_shape': [horizontal, vertical], 'flatten_order': 'C after Fortran record reconstruction',
                'scheme': int(scheme), 'growth': int(growth), 'organic_registered': bool(organic_index),
                'fixed_constants': {name: float(native(name)) for name in FIXED_CONSTANTS}}
    return cases, expected, metadata


def load_native_case(nucleation_directory=None, condensation_directory=None):
    """Load the hash-verified active fixture pair; never silently skip missing data."""
    nucleation_directory = Path(nucleation_directory) if nucleation_directory is not None else ROOT / 'tests/fixtures/nucleation-active'
    condensation_directory = Path(condensation_directory) if condensation_directory is not None else ROOT / 'tests/fixtures/condensation-active'
    return adapt_captures(load_nucleation_fixture(nucleation_directory), load_condensation_fixture(condensation_directory))
