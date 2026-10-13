"""Adapt checked native remapping/correction captures to independent process APIs."""
from pathlib import Path

import numpy as np

from salsa_jax.constants import AVOGADRO, BOLTZMANN, GAS_CONSTANT, PI
from tools.remapping_capture import load_fixture

ROOT = Path(__file__).resolve().parents[1]


def record(state, name):
    item = state[name]
    return np.asarray(item['values']).reshape(item['shape'], order='F')


def adapt_capture(capture):
    context = capture['context']
    def value(name):
        return record(context, name)
    for name, expected in {'pi': PI, 'ak': BOLTZMANN, 'argas': GAS_CONSTANT, 'avo': AVOGADRO}.items():
        if float(value(name)) != expected:
            raise ValueError(f'Unsupported remapping fixed constant: {name}')
    if not np.array_equal(value('species_indices'), value('standard_species_indices')):
        raise ValueError('Unsupported remapping species mapping')
    if not np.array_equal(value('boundaries'), [1, 3, 4, 10, 11, 17]):
        raise ValueError('Unsupported remapping bin boundaries')
    transfer, hydrate, recalc, nsoa, bits = value('options').astype(int)
    if nsoa != 0 or bits != 64 or not np.all(np.isin([transfer, hydrate, recalc], [0, 1])):
        raise ValueError('Unsupported remapping options')
    soluble = value('soluble').astype(bool)
    if not np.array_equal(soluble, (np.arange(17) < 10) | bool(hydrate)):
        raise ValueError('Soluble flags disagree with native hydration option')
    sulfate, organic = value('gas_indices').astype(int)
    ngas = capture['dimensions']['ngas']
    if sulfate != 2 or (organic, ngas) not in ((0, 2), (3, 3)):
        raise ValueError('Unsupported native gas mapping')
    shape = record(capture['passes'][0]['entry'], 'pnaero').shape[:2]
    count = int(np.prod(shape))
    def field(state, name):
        array = record(state, name)
        return array.reshape((count,) + array.shape[2:])
    def scalar(name):
        return np.full(count, float(value(name)))
    def fixed(name, size):
        return np.broadcast_to(value(name), (count, size)).copy()
    def state_arrays(state):
        gas = np.zeros((count, 3))
        gas[:, :ngas] = field(state, 'pc_gas')
        return {'number': field(state, 'pnaero'), 'volume': field(state, 'zvols'), 'gas': gas}
    stages = []
    for stage in capture['passes']:
        entry, remapped = stage['entry'], stage['remap_exit']
        remap_cases = {
            'labels': [f'native-pass{stage["index"]}-cell{i}' for i in range(count)],
            'number': field(entry, 'pnaero'), 'volume': field(entry, 'zvols'),
            'temperature': field(entry, 'ptemp'), 'dry_diameter': fixed('dpmid', 17),
            'lower_volume': fixed('vlolim', 17), 'upper_volume': fixed('vhilim', 17),
            'soluble_transfer': np.full(count, bool(transfer)),
            **{key: scalar(native) for key, native in (
                ('number_threshold', 'nlim'), ('organic_soluble_fraction', 'epsoc'),
                ('water_surface_tension', 'surfw0'), ('ion_count', 'ions'), ('saturation_limit', 'slim'),
                ('water_molar_mass', 'water_molar_mass'), ('water_density', 'water_density'))},
            'sulfate_molar_mass': np.full(count, value('species_molar_mass')[0]),
            'sulfate_density': np.full(count, value('species_density')[0]),
        }
        correction_cases = {
            'labels': remap_cases['labels'], **state_arrays(remapped),
            'relative_humidity': field(remapped, 'prh'), 'temperature': field(remapped, 'ptemp'),
            'dry_diameter': fixed('dpmid', 17), 'soluble_mask': np.broadcast_to(soluble, (count, 17)).copy(),
            'organic_enabled': np.full(count, bool(organic)),
            'species_molar_mass': fixed('species_molar_mass', 5), 'species_density': fixed('species_density', 5),
            **{key: remap_cases[key] for key in ('water_molar_mass', 'water_density', 'organic_soluble_fraction',
                                               'water_surface_tension', 'number_threshold')},
        }
        corrected = {**state_arrays(stage['correction_exit']),
                     **{key: field(stage['correction_exit'], native) for key, native in (
                         ('core', 'zcore'), ('wet_diameter', 'zdwet'), ('dry_radius_cm', 'pm6dry'),
                         ('wet_radius_cm', 'pm6rp'), ('water_mass', 'pww'))}}
        stages.append({'remapping_cases': remap_cases, 'correction_cases': correction_cases,
                       'remapped': state_arrays(remapped), 'cleaned': state_arrays(stage['cleanup_exit']),
                       'corrected': corrected})
    metadata = {'native_cell_shape': list(shape), 'flatten_order': 'C after Fortran record reconstruction',
                'regime_limits': value('reglim').tolist(), 'recalculate': bool(recalc),
                'hydrate_insoluble': bool(hydrate), 'transfer_soluble': bool(transfer),
                'organic_enabled': bool(organic)}
    return stages, metadata


def load_native_case(directory=None):
    directory = Path(directory) if directory is not None else ROOT/'tests/fixtures/remapping-default'
    return adapt_capture(load_fixture(directory))
