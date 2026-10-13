# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Adapt actual initial-hydration captures to the standalone kernel contract."""
from pathlib import Path

import numpy as np

from salsa_jax.constants import AVOGADRO, BOLTZMANN, PI
from tools.hydration_capture import load_fixture

ROOT = Path(__file__).resolve().parents[1]


def record(capture, stage, name):
    item = capture[stage][name]
    return np.asarray(item['values']).reshape(item['shape'], order='F')


def adapt_capture(capture):
    """Return oracle cases, actual expected geometry, and initialized metadata.

    Material IDs must match SO4/OC/BC/SS/DU order. Captured configurable species
    properties and actual bin flags are inputs, not substituted defaults.
    """
    def entry(name):
        return record(capture, 'entry', name)
    for name, value in {'avo': AVOGADRO, 'ak': BOLTZMANN, 'pi': PI}.items():
        if float(entry(name)) != value:
            raise ValueError(f'Unsupported hydration fixed constant: {name}')
    species = entry('species_indices')
    canonical = entry('standard_species_indices')
    water = int(entry('water_species_index'))
    if (species.shape != (5,) or not np.array_equal(species, canonical) or
            len(set(species.tolist())) != 5 or np.any(species <= 0) or water <= 0 or water in species):
        raise ValueError('Unsupported hydration species mapping')
    if not np.array_equal(entry('boundaries'), [1, 3, 4, 10, 11, 17]):
        raise ValueError('Unsupported hydration bin boundaries')
    hydrate_insoluble, nsoa, bits = entry('options').astype(int)
    if hydrate_insoluble not in (0, 1) or nsoa != 0 or bits != 64:
        raise ValueError('Unsupported hydration configuration')
    soluble = entry('soluble')
    if soluble.shape != (17,) or not np.all(np.isin(soluble, [0, 1])):
        raise ValueError('Invalid hydration soluble flags')
    expected_mask = (np.arange(17) < 10) | bool(hydrate_insoluble)
    if not np.array_equal(soluble.astype(bool), expected_mask):
        raise ValueError('Soluble flags disagree with initialized lhydr2b mapping')
    for name in ('pnaero', 'pvols', 'prh', 'ptemp'):
        if not np.array_equal(entry(name), record(capture, 'exit', name)):
            raise ValueError(f'Hydration modified input state: {name}')
    number = entry('pnaero')
    horizontal, vertical, bins = number.shape
    count = horizontal * vertical
    if bins != 17 or entry('pvols').shape != (horizontal, vertical, 17, 5):
        raise ValueError('Unsupported hydration state shape')
    limits = entry('reglim')
    if limits.shape != (4,) or np.any(limits <= 0) or np.any(np.diff(limits) <= 0):
        raise ValueError('Invalid hydration regime limits')
    cases = {
        'labels': [f'native-h{h + 1}-level{k + 1}' for h in range(horizontal) for k in range(vertical)],
        'number': number.reshape(count, 17), 'volume': entry('pvols').reshape(count, 17, 5),
        'relative_humidity': entry('prh').reshape(count), 'temperature': entry('ptemp').reshape(count),
        'dry_diameter': np.broadcast_to(entry('dpmid'), (count, 17)).copy(),
        'soluble_mask': np.broadcast_to(soluble.astype(bool), (count, 17)).copy(),
        'species_molar_mass': np.broadcast_to(entry('species_molar_mass'), (count, 5)).copy(),
        'species_density': np.broadcast_to(entry('species_density'), (count, 5)).copy(),
        **{key: np.full(count, float(entry(native_name))) for key, native_name in (
            ('water_molar_mass', 'water_molar_mass'), ('water_density', 'water_density'),
            ('organic_soluble_fraction', 'epsoc'), ('water_surface_tension', 'surfw0'), ('number_threshold', 'nlim'))},
    }
    expected = {'core': record(capture, 'exit', 'pcore').reshape(count, 17),
                'wet_diameter': record(capture, 'exit', 'pdwet').reshape(count, 17)}
    metadata = {'native_cell_shape': [horizontal, vertical], 'flatten_order': 'C after Fortran record reconstruction',
                'material_order': ['SO4', 'OC', 'BC', 'SS', 'DU'], 'species_indices': species.tolist(),
                'water_species_index': water, 'hydrate_insoluble': bool(hydrate_insoluble),
                'recalculate': bool(entry('recalc')), 'regime_limits': limits.tolist()}
    return cases, expected, metadata


def load_native_case(directory=None):
    """Load the hash-verified first-hydration fixture; missing data is an error."""
    directory = Path(directory) if directory is not None else ROOT / 'tests/fixtures/hydration-default'
    return adapt_capture(load_fixture(directory))
