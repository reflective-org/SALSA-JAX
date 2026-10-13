"""Adapt verified driver HAM conversion boundaries without substituting metadata."""
import hashlib
import json
from pathlib import Path

import numpy as np

from salsa_jax.constants import AVOGADRO, GAS_CONSTANT
from salsa_jax.interface import validate_component_map
from salsa_jax.processes.hydration import HydrationParameters
from tools.ham_conversion_capture import load_fixture

ROOT = Path(__file__).resolve().parents[1]
AEROSOL_INPUTS = ('number', 'core', 'tracers', 'air_density', 'component_map',
                  'component_tracer_indices', 'number_tracer_indices', 'component_unit_codes')
GAS_INPUTS = ('tracers', 'gas', 'air_density', 'pressure', 'temperature',
             'tracer_indices', 'molar_mass', 'unit_codes')


def aerosol_arguments(cases):
    return tuple(cases[key] for key in AEROSOL_INPUTS), {
        'parameters': HydrationParameters(species_molar_mass=cases['species_molar_mass'],
                                         species_density=cases['species_density']),
        'mw_so4': cases['mw_so4']}


def load_native_cases(directory=None):
    """Return aerosol case/output, gas case/output, and run provenance.

    Native cell records are reconstructed in Fortran order before flattening
    active horizontal/vertical axes in C order. All API maps become zero-based.
    Gas inputs are the actual post-driver-scaling native concentrations.
    """
    path = Path(directory) if directory is not None else ROOT/'tests/fixtures/ham-conversion-default'
    capture = load_fixture(path)
    manifest = json.loads((path/'manifest.json').read_text())
    for name in ('mo_convert_concentrations.F90', 'mo_physical_constants.F90', 'mo_ham.F90'):
        if manifest.get('source_sha256', {}).get(name) != hashlib.sha256((ROOT/name).read_bytes()).hexdigest():
            raise ValueError(f'Unsupported HAM conversion source: {name}')
    if '-fcheck=all' not in manifest.get('config', {}).get('build', {}).get('flags', []):
        raise ValueError('Checked native conversion fixture required')
    if not manifest.get('input_sha256') or not manifest.get('run_manifest_sha256'):
        raise ValueError('Missing native input/run association')
    def record(stage, name):
        item = capture[stage][name]
        return np.asarray(item['values']).reshape(item['shape'], order='F')
    def context(name):
        return record('conc_entry', name)
    if float(context('avo')) != AVOGADRO or float(context('argas')) != GAS_CONSTANT:
        raise ValueError('Unsupported native HAM physical constants')
    native_shape = record('conc_entry', 'paernl').shape[:2]
    count = int(np.prod(native_shape))
    def field(stage, name):
        value = record(stage, name)
        return value.reshape((count,)+value.shape[2:])
    def fixed(value):
        return np.broadcast_to(value, (count,)+np.shape(value)).copy()
    ids = context('material_species').astype(int)-1
    mapping = validate_component_map(context('component_map').astype(int)-1)
    aerosol = {
        'labels': [f'native-cell-{i}' for i in range(count)],
        'number': field('conc_entry', 'paernl'), 'core': fixed(context('core')),
        'tracers': field('conc_entry', 'pxtm1'), 'air_density': field('conc_entry', 'rhoa'),
        'component_map': fixed(mapping),
        'component_tracer_indices': fixed(context('component_tracer').astype(int)-1),
        'number_tracer_indices': fixed(context('number_tracer').astype(int)-1),
        'component_unit_codes': fixed(context('component_conversion').astype(int)),
        'species_molar_mass': fixed(context('species_molar_mass')[ids]),
        'species_density': fixed(context('species_density')[ids]), 'mw_so4': fixed(context('mw_so4'))}
    expected = {key: field('conc_exit', name) for key, name in
                (('number', 'paernl'), ('mass', 'paerml'), ('tracers', 'pxtm1'))}
    ids = context('gas_species').astype(int)-1
    gas = {'tracers': field('gas_entry', 'pxtm1'), 'gas': field('gas_entry', 'gas'),
           'air_density': field('gas_entry', 'rhoa'), 'pressure': field('gas_entry', 'pressure'),
           'temperature': field('gas_entry', 'temperature'),
           'tracer_indices': fixed(context('gas_tracer').astype(int)-1),
           'molar_mass': fixed(context('species_molar_mass')[ids]),
           'unit_codes': fixed(context('gas_conversion').astype(int))}
    return aerosol, expected, gas, field('gas_exit', 'pxtm1'), {
        'native_cell_shape': list(native_shape), 'source_sha256': manifest['source_sha256'],
        'input_sha256': manifest['input_sha256'], 'run_manifest_sha256': manifest['run_manifest_sha256'],
        'reference_configuration': manifest.get('reference_configuration'),
        'number_output_units': 'particles cm-3', 'gas_input_units': 'native unit-code convention after driver scaling'}
