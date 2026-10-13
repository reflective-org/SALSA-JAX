# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Driver conversion captures retain initialized state and complete tracer maps."""
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest

from tools.array_records import HEADER, FOOTER
from tools.ham_conversion_capture import (
    CONC_CALL, GAS_CALL, CONTEXT, FILES, INTEGERS, STAGES,
    export_fixture, instrument_driver_ham_conversion, load_capture, load_fixture,
)
from tools.initialization_capture import instrument_driver_initialization
from tools.runtime_metadata import instrument_driver
from tools.reference_scope import instrument_scope

ROOT = Path(__file__).resolve().parents[1]


def write_capture(directory, mutate=None):
    sizes = dict(kproma=2, kbdim=3, klev=2, nclass=17, naerocomp=69, ntrac=90, ngas=2, nspec=8, naerospec=6)
    mapping = [0] * 85
    bins, species = [], []
    for bin_index in range(17):
        for material in range(5):
            if (bin_index < 3 and material > 1) or (bin_index >= 10 and material == 3):
                continue
            bins.append(bin_index+1)
            species.append(material+1)
            mapping[bin_index+17*material] = len(bins)
    integers = {
        'dimensions': list(sizes.values()), 'options': [3, 3, 64, 32], 'unit_codes': [1, 2, 3],
        'component_map': mapping, 'material_species': [1, 2, 3, 4, 5],
        'component_bin': bins, 'component_species': species,
        'component_tracer': list(range(1, 70)), 'number_tracer': list(range(70, 87)),
        'aerosol_species_index': [1, 2, 3, 4, 5, 0, 0, 6], 'aerosol_conversion': [2, 1, 1, 1, 1, 1],
        'component_conversion': [2 if s == 1 else 1 for s in species],
        'gas_species': [6, 7], 'gas_conversion': [2, 2], 'gas_tracer': [87, 88],
    }
    for stage, fields in STAGES.items():
        records = {}
        for name, spec in {**(INTEGERS if stage.endswith('_entry') else {}), **fields}.items():
            shape = [int(dim) if dim.isdigit() else sizes[dim] for dim in spec[1]]
            records[name] = {'kind': 'I' if name in INTEGERS else 'R', 'shape': shape,
                             'values': integers[name].copy() if name in INTEGERS else [1.] * math.prod(shape)}
        if mutate:
            mutate(stage, records)
        lines = [HEADER]
        for name, record in records.items():
            lines.append(' '.join(map(str, [record['kind'], name, len(record['shape']), *record['shape']])))
            lines.extend(map(str, record['values']))
        (directory / FILES[stage]).write_text('\n'.join([*lines, FOOTER]) + '\n')


@pytest.mark.parametrize('crlf', [False, True])
def test_overlay_preserves_driver_and_wraps_actual_calls(tmp_path, crlf):
    original = (ROOT / 'driver.F90').read_bytes()
    copied = tmp_path / 'driver.F90'
    copied.write_bytes(original.replace(b'\n', b'\r\n') if crlf else original)
    before = copied.read_bytes()
    metadata = instrument_driver_ham_conversion(copied)
    assert (ROOT / 'driver.F90').read_bytes() == original
    assert metadata['original_driver_sha256'] == hashlib.sha256(before).hexdigest()
    assert metadata['instrumented_driver_sha256'] == hashlib.sha256(copied.read_bytes()).hexdigest()
    if crlf:
        assert copied.read_bytes().count(b'\r\n') == copied.read_bytes().count(b'\n')
    else:
        assert b'\r\n' not in copied.read_bytes()
    text = copied.read_text()
    assert text.index(FILES['conc_entry']) < text.index(CONC_CALL) < text.index(FILES['conc_exit'])
    assert "'R paerml 3'" not in text[:text.index(CONC_CALL)]
    gas_units = text.index('zgas(1:kproma,:,:) = zgas(1:kproma,:,:) * 1.e-6_dp')
    assert gas_units < text.index(FILES['gas_entry']) < text.index(GAS_CALL) < text.index(FILES['gas_exit'])
    assert text.index(FILES['gas_exit']) < text.index('CALL ham_subm_interface')
    assert text.count('IF (hc_gas_count == 1) THEN') == 2
    assert text.count('CALL conc2mmr') == text.count('CALL gas2mmr') == 1
    with pytest.raises(ValueError, match='already instrumented'):
        instrument_driver_ham_conversion(copied)


def test_all_driver_overlays_compose(tmp_path):
    copied = tmp_path / 'driver.F90'
    copied.write_bytes((ROOT / 'driver.F90').read_bytes())
    instrument_driver(copied)
    instrument_scope(copied)
    instrument_driver_initialization(copied)
    instrument_driver_ham_conversion(copied)
    text = copied.read_text()
    assert text.index('initialization-exit.txt') < text.index(FILES['conc_entry'])
    assert text.count('SALSA_RUNTIME_METADATA_V1') == 1
    assert text.count('hc_conc_count=hc_conc_count+1') == 1


@pytest.mark.parametrize('call', [CONC_CALL, GAS_CALL])
@pytest.mark.parametrize('copies', [0, 2])
def test_bad_anchor_rejected_without_mutation(tmp_path, call, copies):
    copied = tmp_path / 'driver.F90'
    original = (ROOT / 'driver.F90').read_text().replace(call, call * copies)
    copied.write_text(original)
    with pytest.raises(ValueError, match='anchor'):
        instrument_driver_ham_conversion(copied)
    assert copied.read_text() == original
    assert not (tmp_path / 'ham-conversion-state.patch').exists()


def test_complete_contract_maps_units_and_active_domain(tmp_path):
    write_capture(tmp_path)
    state = load_capture(tmp_path)
    assert 'paerml' not in state['conc_entry']
    assert state['conc_exit']['paerml']['shape'] == [2, 2, 69]
    assert state['gas_entry']['pxtm1']['shape'] == [2, 2, 90]
    assert state['conc_entry']['component_map']['shape'] == [17, 5]
    assert state['gas_entry']['gas_conversion']['values'] == [2, 2]
    assert state['conc_entry']['mw_so4']['shape'] == []
    assert 'particles cm-3' in state['units']['conc_exit']['paernl']
    assert state['dimensions']['kbdim'] == 3


@pytest.mark.parametrize('name,index,value,message', [
    ('dimensions', 0, 4, 'dimensions'), ('options', 0, 2, 'options'), ('unit_codes', 0, 0, 'codes'),
    ('material_species', 0, 9, 'species'), ('component_map', 0, 0, 'mapping'),
    ('component_map', 34, 1, 'absent'), ('component_bin', 0, 2, 'mapping'),
    ('component_species', 0, 2, 'mapping'), ('aerosol_species_index', 0, 0, 'index'),
    ('component_conversion', 0, 1, 'code'), ('gas_species', 0, 9, 'species'),
    ('gas_conversion', 0, 4, 'code'), ('component_tracer', 0, 0, 'tracer'),
    ('number_tracer', 0, 1, 'tracer'), ('gas_tracer', 0, 90+1, 'tracer'),
    ('species_density', 0, 0, 'property'), ('species_molar_mass', 5, 0, 'molar'),
    ('mw_so4', 0, 0, 'property'), ('core', 0, 0, 'Nonpositive'), ('gas', 0, -1, 'Negative'),
])
def test_invalid_contract_rejected(tmp_path, name, index, value, message):
    def mutate(stage, records):
        if name in records:
            records[name]['values'][index] = value
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match=message):
        load_capture(tmp_path)


@pytest.mark.parametrize('name', list(CONTEXT) + ['gas_conversion'])
def test_registry_must_stay_unchanged(tmp_path, name):
    def mutate(stage, records):
        if stage == 'gas_entry':
            records[name]['values'][0] += 1
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match='registry changed'):
        load_capture(tmp_path)


@pytest.mark.parametrize('stage,name', [('conc_exit', 'rhoa'), ('conc_exit', 'core'), ('gas_exit', 'gas'),
                                       ('gas_exit', 'pressure'), ('gas_exit', 'temperature')])
def test_pure_inputs_must_stay_unchanged(tmp_path, stage, name):
    def mutate(current, records):
        if current == stage:
            records[name]['values'][0] += 1
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match='changed pure input'):
        load_capture(tmp_path)


@pytest.mark.parametrize('stage', ['conc_exit', 'gas_exit'])
def test_unrelated_tracer_changes_rejected(tmp_path, stage):
    def mutate(current, records):
        if current == stage:
            records['pxtm1']['values'][-1] += 1
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match='unrelated tracer'):
        load_capture(tmp_path)


def test_conc_to_gas_boundary_continuity(tmp_path):
    def mutate(stage, records):
        if stage.startswith('gas'):
            records['pxtm1']['values'][0] += 1
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match='between initialization'):
        load_capture(tmp_path)


@pytest.mark.parametrize('defect', ['missing', 'kind', 'nonfinite'])
def test_bad_array_record_rejected(tmp_path, defect):
    def mutate(stage, records):
        if stage != 'gas_exit':
            return
        if defect == 'missing':
            del records['gas']
        elif defect == 'kind':
            records['gas']['kind'] = 'I'
        else:
            records['gas']['values'][0] = float('nan')
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError):
        load_capture(tmp_path)


def checked_manifest(directory):
    manifest = {'scope': 'first-microphysics', 'status': 'first_microphysics_completed_unvalidated',
                'config': {'build': {'flags': ['-fcheck=all']}},
                'ham_conversion_capture': {'status': 'captured', 'file_sha256': {
                    name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in FILES.values()}},
                'source_revision': 'test', 'working_tree_status': 'test', 'source_sha256': {},
                'input_sha256': {}, 'compiler': 'test', 'runtime_capture': {'metadata': {}},
                'scope_capture': {'state_sha256': 'test'}, 'reference_configuration': {'profile': 'test'}}
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    return manifest


def test_fixture_roundtrip_hashes_provenance_and_no_overwrite(tmp_path):
    write_capture(tmp_path)
    checked_manifest(tmp_path)
    target = tmp_path / 'fixture'
    export_fixture(tmp_path, target)
    assert load_fixture(target) == load_capture(tmp_path)
    manifest = json.loads((target / 'manifest.json').read_text())
    assert manifest['reference_configuration'] == {'profile': 'test'}
    with pytest.raises(FileExistsError):
        export_fixture(tmp_path, target)
    path = target / FILES['gas_entry']
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_fixture(target)


@pytest.mark.parametrize('defect', ['unchecked', 'failed', 'scope', 'changed'])
def test_export_rejects_unverified_run(tmp_path, defect):
    write_capture(tmp_path)
    manifest = checked_manifest(tmp_path)
    if defect == 'unchecked':
        manifest['config']['build']['flags'] = []
    elif defect == 'failed':
        manifest['ham_conversion_capture']['status'] = 'failed'
    elif defect == 'scope':
        manifest['scope'] = 'full'
    else:
        path = tmp_path / FILES['conc_exit']
        path.write_bytes(path.read_bytes() + b'\n')
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='checked|changed'):
        export_fixture(tmp_path, tmp_path / 'fixture')
    assert not (tmp_path / 'fixture').exists()


def test_runner_rejects_unscoped_capture_before_build(tmp_path, monkeypatch):
    from tools import fortran_reference
    target = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['reference', '--output', str(target), '--capture-ham-conversion-state'])
    with pytest.raises(SystemExit) as error:
        fortran_reference.main()
    assert error.value.code == 2
    assert not target.exists()
