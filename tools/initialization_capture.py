# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Copy-only capture of the driver's native seven-mode initialization call."""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import shutil

if __package__:
    from .array_records import HEADER, FOOTER, array_lines, parse_records
else:
    from array_records import HEADER, FOOTER, array_lines, parse_records

FILES = {'entry': 'initialization-entry.txt', 'exit': 'initialization-exit.txt'}
ENTRY = {
    'n': ('n', ('nmod',), 'particles m-3; driver already multiplies cm-3 values by 1e6'),
    'dpg': ('dpg', ('nmod',), 'm; geometric mean diameter'),
    'sigmag': ('sigmag', ('nmod',), '1; geometric standard deviation'),
    'vlolim': ('vlolim', ('fn2b',), 'm3 particle-1'),
    'vhilim': ('vhilim', ('fn2b',), 'm3 particle-1'),
    'dpmid': ('dpmid', ('fn2b',), 'm'),
    'core': ('core', ('fn2b',), 'm3 particle-1; driver midpoint-derived core'),
    'reglim': ('reglim', ('4',), 'm'),
    'pi': ('pi', (), '1'),
    'pi_6': ('pi_6', (), '1'),
}
EXIT = {**ENTRY, 'naero': ('zaernl(1:kproma,:,:)', ('kproma', 'klev', 'fn2b'), 'particles m-3')}
INTEGERS = {
    'dimensions': ('[kproma,kbdim,klev,fn2b,nmod,nclass]', ('6',)),
    'boundaries': ('[in1a,fn1a,in2a,fn2a,in2b,fn2b]', ('6',)),
    'options': ('[nham_subm,HAM_SALSA,STORAGE_SIZE(1._dp),STORAGE_SIZE(1.)]', ('4',)),
}
CALL = '''     CALL size_distribution(kproma, kbdim,  klev,   &
          n, dpg, sigmag, zaernl)
'''


def instrument_driver_initialization(path: Path) -> dict:
    """Wrap the actual SALSA branch before conc2mmr, preserving source newlines."""
    path = Path(path)
    original = path.read_bytes()
    newline = b'\r\n' if b'\r\n' in original else b'\n'
    text = original.decode().replace('\r\n', '\n')
    if 'SALSA_INITIALIZATION_STATE_V1' in text:
        raise ValueError('Initialization source already instrumented')
    declaration = '  INTEGER :: ii, jj, kk\n'
    for anchor in (declaration, CALL):
        if text.count(anchor) != 1:
            raise ValueError(f'Expected one initialization capture anchor: {anchor.strip()}')
    stages = {}
    for stage, fields in (('entry', ENTRY), ('exit', EXIT)):
        lines = []
        if stage == 'entry':
            lines += ['! SALSA_INITIALIZATION_STATE_V1: copy-only instrumentation',
                      'initialization_capture_count=initialization_capture_count+1',
                      "IF (initialization_capture_count /= 1) ERROR STOP 'initialization requires one call'",
                      'IF (nham_subm /= HAM_SALSA .OR. nmod /= 7 .OR. fn2b /= 17 .OR. nclass /= 17) &',
                      "     ERROR STOP 'unsupported initialization capture configuration'"]
        lines += [f"OPEN(NEWUNIT=initialization_capture_unit,FILE='{FILES[stage]}',STATUS='new',ACTION='write')",
                  f"WRITE(initialization_capture_unit,'(A)') '{HEADER}'"]
        records = {**INTEGERS, **fields} if stage == 'entry' else fields
        for name, spec in records.items():
            lines += array_lines(name, spec[0], 'I' if name in INTEGERS else 'R', spec[1],
                                 'initialization_capture_unit')
        lines += [f"WRITE(initialization_capture_unit,'(A)') '{FOOTER}'", 'CLOSE(initialization_capture_unit)']
        stages[stage] = ''.join('     ' + line + '\n' for line in lines)
    text = text.replace(declaration, declaration +
                        '  INTEGER :: initialization_capture_unit\n'
                        '  INTEGER, SAVE :: initialization_capture_count=0\n')
    text = text.replace(CALL, stages['entry'] + CALL + stages['exit'])
    instrumented = text.encode().replace(b'\n', newline)
    patch = ''.join(difflib.unified_diff(original.decode().splitlines(True), instrumented.decode().splitlines(True),
                                       fromfile=path.name, tofile=path.name))
    overlay = path.parent / 'initialization-state.patch'
    overlay.write_text(patch)
    path.write_bytes(instrumented)
    sha = lambda data: hashlib.sha256(data).hexdigest()
    return {'original_driver_sha256': sha(original), 'instrumented_driver_sha256': sha(instrumented),
            'overlay_file': overlay.name, 'overlay_sha256': sha(overlay.read_bytes()),
            'helper_sha256': sha(Path(__file__).read_bytes())}


def load_capture(directory: Path) -> dict:
    directory = Path(directory)
    entry = parse_records((directory / FILES['entry']).read_text())
    exit_state = parse_records((directory / FILES['exit']).read_text())
    if set(entry) != set(ENTRY) | set(INTEGERS) or set(exit_state) != set(EXIT):
        raise ValueError('Missing or unexpected initialization field')
    dimensions = entry['dimensions']
    if dimensions['kind'] != 'I' or dimensions['shape'] != [6]:
        raise ValueError('Invalid initialization dimensions record')
    sizes = dict(zip(('kproma', 'kbdim', 'klev', 'fn2b', 'nmod', 'nclass'), dimensions['values']))
    if (any(value <= 0 for value in sizes.values()) or sizes['kproma'] > sizes['kbdim'] or
            sizes['fn2b'] != 17 or sizes['nclass'] != 17 or sizes['nmod'] != 7):
        raise ValueError('Invalid initialization dimensions')
    for fields, records in ((ENTRY, entry), (EXIT, exit_state), (INTEGERS, entry)):
        for name, spec in fields.items():
            shape = [int(dim) if dim.isdigit() else sizes[dim] for dim in spec[1]]
            if records[name]['shape'] != shape or records[name]['kind'] != ('I' if fields is INTEGERS else 'R'):
                raise ValueError(f'Wrong initialization shape/type: {name}')
    if entry['boundaries']['values'] != [1, 3, 4, 10, 11, 17]:
        raise ValueError('Unsupported initialization boundaries')
    model, salsa, bits, default_bits = entry['options']['values']
    if model != salsa or salsa != 3 or bits != 64 or default_bits != 32:
        raise ValueError('Unsupported initialization options')
    for name in ENTRY:
        if entry[name] != exit_state[name]:
            raise ValueError(f'Initialization changed pure input: {name}')
        values = entry[name]['values']
        if name == 'n':
            if any(value < 0 for value in values):
                raise ValueError('Negative initialization mode number')
        elif name == 'sigmag':
            if any(value <= 1 for value in values):
                raise ValueError('Initialization sigmag must exceed one')
        elif any(value <= 0 for value in values):
            raise ValueError(f'Nonpositive initialization input: {name}')
    for lo, hi in zip(entry['vlolim']['values'], entry['vhilim']['values']):
        if lo >= hi:
            raise ValueError('Invalid initialization bin-volume bounds')
    limits = entry['reglim']['values']
    if any(lo >= hi for lo, hi in zip(limits, limits[1:])):
        raise ValueError('Invalid initialization regime limits')
    if any(value < 0 for value in exit_state['naero']['values']):
        raise ValueError('Negative initialization output')
    return {'schema_version': 1, 'order': 'Fortran column-major', 'dimensions': sizes,
            'options': {'nham_subm': model, 'dp_bits': bits, 'default_real_bits': default_bits},
            'stage': 'driver size_distribution entry/exit; before initial conc2mmr',
            'mode_order': ['NS', 'KS', 'AS', 'CS', 'KI', 'AI', 'CI'],
            'entry': entry, 'exit': exit_state,
            'units': {name: spec[2] for name, spec in EXIT.items()}}


def load_fixture(directory: Path) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or set(manifest.get('files', {})) != set(FILES.values()):
        raise ValueError('Invalid initialization fixture inventory')
    for name, expected in manifest['files'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Initialization fixture hash mismatch: {name}')
    return load_capture(directory)


def export_fixture(run_directory: Path, target: Path) -> None:
    """Retain only a successful checked first-microphysics initialization capture."""
    run_directory, target = Path(run_directory), Path(target)
    manifest_path = run_directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    capture = manifest.get('initialization_capture', {})
    if (manifest.get('scope') != 'first-microphysics' or
            manifest.get('status') != 'first_microphysics_completed_unvalidated' or
            capture.get('status') != 'captured' or
            '-fcheck=all' not in manifest.get('config', {}).get('build', {}).get('flags', [])):
        raise ValueError('A successful checked initialization capture is required')
    hashes = {name: hashlib.sha256((run_directory / name).read_bytes()).hexdigest() for name in FILES.values()}
    if hashes != capture.get('file_sha256'):
        raise ValueError('Run initialization files changed after manifest creation')
    state = load_capture(run_directory)
    fixture = {'schema_version': 1, 'scope': 'native seven-mode number initialization; not trajectory acceptance',
               'files': hashes, 'run_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               'source_revision': manifest['source_revision'], 'working_tree_status': manifest['working_tree_status'],
               'source_sha256': manifest['source_sha256'], 'input_sha256': manifest['input_sha256'],
               'compiler': manifest['compiler'], 'config': manifest['config'],
               **({'reference_configuration': manifest['reference_configuration']} if 'reference_configuration' in manifest else {}),
               'dimensions': state['dimensions'], 'options': state['options'], 'stage': state['stage'],
               'initialization_overlay': {key: value for key, value in capture.items() if key not in ('state', 'file_sha256')},
               'runtime_metadata': manifest['runtime_capture']['metadata'],
               'scope_state_sha256': manifest['scope_capture']['state_sha256']}
    target.mkdir(parents=True, exist_ok=False)
    for name in FILES.values():
        shutil.copy2(run_directory / name, target / name)
    (target / 'manifest.json').write_text(json.dumps(fixture, indent=2) + '\n')
