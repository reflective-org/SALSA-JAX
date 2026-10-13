# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Native initialization capture preserves caller order, inputs and provenance."""
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest

from tools.array_records import HEADER, FOOTER
from tools.initialization_capture import (
    CALL, ENTRY, EXIT, FILES, INTEGERS, export_fixture,
    instrument_driver_initialization, load_capture, load_fixture,
)
from tools.runtime_metadata import instrument_driver
from tools.reference_scope import instrument_scope

ROOT = Path(__file__).resolve().parents[1]


def write_capture(directory, mutate=None):
    sizes = dict(kproma=2, kbdim=3, klev=2, fn2b=17, nmod=7, nclass=17)
    for stage, fields in (('entry', {**INTEGERS, **ENTRY}), ('exit', EXIT)):
        records = {}
        for name, spec in fields.items():
            shape = [int(dim) if dim.isdigit() else sizes[dim] for dim in spec[1]]
            records[name] = {'kind': 'I' if name in INTEGERS else 'R', 'shape': shape,
                             'values': [1] * math.prod(shape)}
        for name, values in {'sigmag': [1.59] * 7, 'vhilim': [2.] * 17, 'reglim': [1., 2., 3., 4.]}.items():
            records[name]['values'] = values
        if stage == 'entry':
            for name, values in {'dimensions': list(sizes.values()), 'boundaries': [1, 3, 4, 10, 11, 17],
                                 'options': [3, 3, 64, 32]}.items():
                records[name]['values'] = values
        if mutate:
            mutate(stage, records)
        lines = [HEADER]
        for name, record in records.items():
            lines.append(' '.join(map(str, [record['kind'], name, len(record['shape']), *record['shape']])))
            lines.extend(map(str, record['values']))
        (directory / FILES[stage]).write_text('\n'.join([*lines, FOOTER]) + '\n')


@pytest.mark.parametrize('crlf', [False, True])
def test_overlay_preserves_newlines_and_actual_call_order(tmp_path, crlf):
    source = ROOT / 'driver.F90'
    original = source.read_bytes()
    copied = tmp_path / source.name
    copied.write_bytes(original.replace(b'\n', b'\r\n') if crlf else original)
    before = copied.read_bytes()
    metadata = instrument_driver_initialization(copied)
    assert source.read_bytes() == original
    assert metadata['original_driver_sha256'] == hashlib.sha256(before).hexdigest()
    assert metadata['instrumented_driver_sha256'] == hashlib.sha256(copied.read_bytes()).hexdigest()
    if crlf:
        assert copied.read_bytes().count(b'\r\n') == copied.read_bytes().count(b'\n')
    else:
        assert b'\r\n' not in copied.read_bytes()
    text = copied.read_text()
    assert text.index('core(in1a:fn2b) =') < text.index(FILES['entry']) < text.index(CALL)
    assert text.index(CALL) < text.index(FILES['exit']) < text.index('  CALL conc2mmr(')
    assert text.count('CALL size_distribution') == 1
    assert "'R naero 3'" not in text[:text.index(CALL)]
    assert text.count('zaernl(1:kproma,:,:)') == 1
    with pytest.raises(ValueError, match='already instrumented'):
        instrument_driver_initialization(copied)


def test_driver_overlays_compose(tmp_path):
    copied = tmp_path / 'driver.F90'
    copied.write_bytes((ROOT / copied.name).read_bytes())
    instrument_driver(copied)
    instrument_scope(copied)
    instrument_driver_initialization(copied)
    text = copied.read_text()
    assert 'SALSA_RUNTIME_METADATA_V1' in text
    assert 'SALSA_INITIALIZATION_STATE_V1' in text
    assert text.index(FILES['exit']) < text.index('  CALL conc2mmr(')


@pytest.mark.parametrize('copies', [0, 2])
def test_bad_anchor_fails_without_mutation(tmp_path, copies):
    copied = tmp_path / 'driver.F90'
    original = (ROOT / copied.name).read_text().replace(CALL, CALL * copies)
    copied.write_text(original)
    with pytest.raises(ValueError, match='anchor'):
        instrument_driver_initialization(copied)
    assert copied.read_text() == original
    assert not (tmp_path / 'initialization-state.patch').exists()


def test_capture_contract_excludes_inactive_padding(tmp_path):
    write_capture(tmp_path)
    capture = load_capture(tmp_path)
    assert capture['dimensions']['kbdim'] == 3
    assert capture['exit']['naero']['shape'] == [2, 2, 17]
    assert capture['entry']['n']['shape'] == [7]
    assert capture['options'] == {'nham_subm': 3, 'dp_bits': 64, 'default_real_bits': 32}
    assert capture['units']['naero'] == 'particles m-3'
    assert capture['mode_order'] == ['NS', 'KS', 'AS', 'CS', 'KI', 'AI', 'CI']


@pytest.mark.parametrize('name,change,message', [
    ('dimensions', {'values': [4, 3, 2, 17, 7, 17]}, 'dimensions'),
    ('dimensions', {'values': [2, 3, 2, 17, 6, 17]}, 'dimensions'),
    ('boundaries', {'values': [1, 3, 4, 10, 12, 17]}, 'boundaries'),
    ('options', {'values': [2, 3, 64, 32]}, 'options'),
    ('options', {'values': [3, 3, 32, 32]}, 'options'),
    ('options', {'values': [3, 3, 64, 64]}, 'options'),
    ('n', {'kind': 'I'}, 'shape/type'),
    ('n', {'values': [-1] * 7}, 'Negative'),
    ('sigmag', {'values': [1.] * 7}, 'sigmag'),
    ('dpg', {'values': [0.] * 7}, 'Nonpositive'),
    ('vlolim', {'values': [2.] * 17}, 'bounds'),
    ('reglim', {'values': [1., 2., 2., 4.]}, 'limits'),
])
def test_bad_contract_rejected(tmp_path, name, change, message):
    def mutate(stage, records):
        if name in records:
            records[name].update(change)
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match=message):
        load_capture(tmp_path)


@pytest.mark.parametrize('name', list(ENTRY))
def test_changed_pure_input_rejected(tmp_path, name):
    def mutate(stage, records):
        if stage == 'exit':
            records[name]['values'][0] *= 1.1
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match='changed pure input'):
        load_capture(tmp_path)


@pytest.mark.parametrize('defect', ['missing', 'negative', 'nonfinite'])
def test_bad_output_rejected(tmp_path, defect):
    def mutate(stage, records):
        if stage != 'exit':
            return
        if defect == 'missing':
            del records['naero']
        else:
            records['naero']['values'][0] = -1 if defect == 'negative' else float('nan')
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match='Missing|Negative|Nonfinite'):
        load_capture(tmp_path)


def checked_manifest(directory):
    manifest = {'scope': 'first-microphysics', 'status': 'first_microphysics_completed_unvalidated',
                'config': {'build': {'flags': ['-fcheck=all']}},
                'initialization_capture': {'status': 'captured', 'file_sha256': {
                    name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in FILES.values()}},
                'source_revision': 'test', 'working_tree_status': 'test', 'source_sha256': {},
                'input_sha256': {}, 'compiler': 'test', 'runtime_capture': {'metadata': {}},
                'scope_capture': {'state_sha256': 'test'}, 'reference_configuration': {'profile': 'test'}}
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    return manifest


def test_fixture_export_hashes_no_overwrite_and_provenance(tmp_path):
    write_capture(tmp_path)
    checked_manifest(tmp_path)
    target = tmp_path / 'fixture'
    export_fixture(tmp_path, target)
    assert load_fixture(target) == load_capture(tmp_path)
    manifest = json.loads((target / 'manifest.json').read_text())
    assert manifest['reference_configuration'] == {'profile': 'test'}
    assert manifest['run_manifest_sha256']
    with pytest.raises(FileExistsError):
        export_fixture(tmp_path, target)
    entry = target / FILES['entry']
    entry.write_bytes(entry.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_fixture(target)


@pytest.mark.parametrize('defect', ['unchecked', 'failed', 'scope', 'changed'])
def test_export_rejects_unverified_run(tmp_path, defect):
    write_capture(tmp_path)
    manifest = checked_manifest(tmp_path)
    if defect == 'unchecked':
        manifest['config']['build']['flags'] = []
    elif defect == 'failed':
        manifest['initialization_capture']['status'] = 'failed'
    elif defect == 'scope':
        manifest['scope'] = 'full'
    else:
        path = tmp_path / FILES['exit']
        path.write_bytes(path.read_bytes() + b'\n')
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='checked|changed'):
        export_fixture(tmp_path, tmp_path / 'fixture')
    assert not (tmp_path / 'fixture').exists()


def test_runner_rejects_unscoped_initialization_before_build(tmp_path, monkeypatch):
    from tools import fortran_reference
    target = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['fortran_reference', '--output', str(target), '--capture-initialization-state'])
    with pytest.raises(SystemExit) as error:
        fortran_reference.main()
    assert error.value.code == 2
    assert not target.exists()


def test_runner_wires_capture_and_auto_runtime_without_native_build(tmp_path, monkeypatch):
    from tools import fortran_reference
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'driver.F90').write_bytes((ROOT / 'driver.F90').read_bytes())
    for name in ('Makefile', 'Makefile.in', '.depend', 'configure', 'namelist.echam', 'lut_kappa.nc'):
        (source / name).write_text('test input\n')
    monkeypatch.setattr(fortran_reference, 'ROOT', source)
    monkeypatch.setattr(fortran_reference, 'output', lambda *args: 'mock metadata')
    def build(command, directory, log, timeout):
        assert command[0] == 'make'
        (directory / 'ham_box').write_bytes(b'mocked executable')
        return {'status': 'finished', 'returncode': 0, 'command': command}
    monkeypatch.setattr(fortran_reference, 'logged_run', build)
    target = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['fortran_reference', '--config', str(ROOT / 'configs/fortran-checked.toml'),
                                    '--output', str(target), '--scope', 'first-microphysics',
                                    '--capture-initialization-state'])
    assert fortran_reference.main() == 0
    manifest = json.loads((target / 'manifest.json').read_text())
    assert manifest['status'] == 'built'
    for key in ('runtime_capture', 'scope_capture', 'initialization_capture'):
        assert manifest[key]['status'] == 'instrumented_not_run'
    driver = (target / 'driver.F90').read_bytes()
    assert manifest['compiled_driver_sha256'] == hashlib.sha256(driver).hexdigest()
    assert manifest['initialization_capture']['instrumented_driver_sha256'] == manifest['compiled_driver_sha256']
    assert (source / 'driver.F90').read_bytes() == (ROOT / 'driver.F90').read_bytes()
