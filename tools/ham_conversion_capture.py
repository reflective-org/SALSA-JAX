"""Actual driver conc2mmr and first gas2mmr boundaries, copied sources only."""
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

FILES = {stage: f'ham-{stage.replace("_", "-")}.txt' for stage in ('conc_entry', 'conc_exit', 'gas_entry', 'gas_exit')}
CONTEXT = {
    'mw_so4': ('hc_mw_so4', (), 'g mol-1; legacy HAM sulfate conversion constant'),
    'species_molar_mass': ('speclist(1:nspec)%moleweight', ('nspec',), 'g mol-1'),
    'species_density': ('speclist(1:nspec)%density', ('nspec',), 'kg m-3; aerosol species only'),
    'avo': ('hc_avo', (), 'mol-1'), 'argas': ('hc_argas', (), 'J mol-1 K-1'),
}
TRACERS = {'pxtm1': ('pxtm1(1:kproma,:,:)', ('kproma', 'klev', 'ntrac'), 'tracer-specific kg/kg, number/kg or volume mixing ratio')}
AEROSOL = {
    'paernl': ('zaernl(1:kproma,:,:)', ('kproma', 'klev', 'nclass'), 'stage-dependent: entry particles m-3; exit particles cm-3'),
    'core': ('core', ('nclass',), 'm3 particle-1'),
    'rhoa': ('zrhoa(1:kproma,:)', ('kproma', 'klev'), 'kg air m-3'),
}
GAS = {
    'gas': ('zgas(1:kproma,:,:)', ('kproma', 'klev', 'ngas'), 'unit-code dependent: code1 ug m-3; codes2/3 molecules cm-3'),
    'rhoa': AEROSOL['rhoa'],
    'pressure': ('pap(1:kproma,:)', ('kproma', 'klev'), 'Pa'),
    'temperature': ('pt(1:kproma,:)', ('kproma', 'klev'), 'K'),
}
STAGES = {
    'conc_entry': {**CONTEXT, **TRACERS, **AEROSOL},
    'conc_exit': {**TRACERS, **AEROSOL, 'paerml': ('zaerml(1:kproma,:,:)', ('kproma', 'klev', 'naerocomp'), 'SO4 molecules cm-3; other materials ug m-3')},
    'gas_entry': {**CONTEXT, **TRACERS, **GAS},
    'gas_exit': {**TRACERS, **GAS},
}
INTEGERS = {
    'dimensions': ('[kproma,kbdim,klev,nclass,naerocomp,ntrac,subm_ngasspec,nspec,hc_naerospec]', ('9',)),
    'options': ('[nham_subm,HAM_SALSA,STORAGE_SIZE(1._dp),STORAGE_SIZE(1.)]', ('4',)),
    'unit_codes': ('[hc_immr2ug,hc_immr2molec,hc_ivmr2molec]', ('3',)),
    'component_map': ('hc_component_map', ('nclass', '5')),
    'material_species': ('[hc_so4,hc_oc,hc_bc,hc_ss,hc_du]', ('5',)),
    'component_bin': ('aerocomp(1:naerocomp)%iclass', ('naerocomp',)),
    'component_species': ('aerocomp(1:naerocomp)%spid', ('naerocomp',)),
    'component_tracer': ('aerocomp(1:naerocomp)%idt', ('naerocomp',)),
    'aerosol_species_index': ('hc_aero_idx(1:nspec)', ('nspec',)),
    'aerosol_conversion': ('hc_aerounitconv(1:hc_naerospec)', ('naerospec',)),
    'component_conversion': ('[(hc_aerounitconv(hc_aero_idx(aerocomp(hc_index)%spid)),hc_index=1,naerocomp)]', ('naerocomp',)),
    'number_tracer': ('sizeclass(1:nclass)%idt_no', ('nclass',)),
    'gas_species': ('hc_gasspec(1:subm_ngasspec)', ('ngas',)),
    'gas_conversion': ('hc_gasconv(1:subm_ngasspec)', ('ngas',)),
    'gas_tracer': ('[(speclist(hc_gasspec(hc_index))%idt,hc_index=1,subm_ngasspec)]', ('ngas',)),
}
CONC_CALL = '''  CALL conc2mmr(kproma,kbdim,klev,ntrac, &
       pxtm1,zaerml,zaernl,core,zrhoa)
'''
GAS_CALL = '''      CALL gas2mmr(kproma, kbdim, klev, ntrac, &
            pxtm1, zgas, zrhoa, pap, pt)
'''


def instrument_driver_ham_conversion(path: Path) -> dict:
    path = Path(path)
    original = path.read_bytes()
    newline = b'\r\n' if b'\r\n' in original else b'\n'
    text = original.decode().replace('\r\n', '\n')
    if 'HAM_CONVERSION_STATE_V1' in text:
        raise ValueError('HAM conversion source already instrumented')
    for anchor in ('  IMPLICIT NONE\n', '  INTEGER :: ii, jj, kk\n', CONC_CALL, GAS_CALL):
        if text.count(anchor) != 1:
            raise ValueError(f'Expected one HAM conversion capture anchor: {anchor.strip()}')
    imports = '''  USE mo_ham, ONLY: hc_mw_so4=>mw_so4,hc_naerospec=>subm_naerospec, &
       hc_aero_idx=>subm_aero_idx,hc_aerounitconv=>subm_aerounitconv, &
       hc_gasspec=>subm_gasspec,hc_gasconv=>subm_gasunitconv, &
       hc_immr2ug=>immr2ug,hc_immr2molec=>immr2molec,hc_ivmr2molec=>ivmr2molec
  USE mo_ham_species, ONLY: hc_so4=>id_so4,hc_oc=>id_oc,hc_bc=>id_bc,hc_ss=>id_ss,hc_du=>id_du
  USE mo_physical_constants, ONLY: hc_avo=>avo,hc_argas=>argas
'''
    declarations = '''  INTEGER :: hc_unit,hc_index,hc_component_map(17,5)
  INTEGER, SAVE :: hc_conc_count=0,hc_gas_count=0
'''
    blocks = {}
    for stage, fields in STAGES.items():
        lines = []
        if stage == 'conc_entry':
            lines += ['! HAM_CONVERSION_STATE_V1: copy-only instrumentation', 'hc_conc_count=hc_conc_count+1',
                      "IF (hc_conc_count /= 1) ERROR STOP 'conc2mmr capture requires one initialization'",
                      'IF (nham_subm /= HAM_SALSA .OR. nclass /= 17 .OR. naerocomp /= 69) &',
                      "     ERROR STOP 'unsupported HAM conversion capture configuration'",
                      'hc_component_map=0', 'hc_component_map(:,1)=iso4b', 'hc_component_map(:,2)=iocb',
                      'hc_component_map(in2a:fn2b,3)=ibcb(in2a:fn2b)',
                      'hc_component_map(in2a:fn2a,4)=issb(in2a:fn2a)',
                      'hc_component_map(in2a:fn2b,5)=idub(in2a:fn2b)']
        if stage == 'gas_entry':
            lines += ['hc_gas_count=hc_gas_count+1']
        if stage.startswith('gas_'):
            lines += ['IF (hc_gas_count == 1) THEN']
        lines += [f"OPEN(NEWUNIT=hc_unit,FILE='{FILES[stage]}',STATUS='new',ACTION='write')",
                  f"WRITE(hc_unit,'(A)') '{HEADER}'"]
        records = {**INTEGERS, **fields} if stage.endswith('_entry') else fields
        for name, spec in records.items():
            shape = tuple({'ngas': 'subm_ngasspec', 'naerospec': 'hc_naerospec'}.get(dim, dim) for dim in spec[1])
            lines += array_lines(name, spec[0], 'I' if name in INTEGERS else 'R', shape, 'hc_unit')
        lines += [f"WRITE(hc_unit,'(A)') '{FOOTER}'", 'CLOSE(hc_unit)']
        if stage.startswith('gas_'):
            lines += ['END IF']
        blocks[stage] = ''.join('  ' + line + '\n' for line in lines)
    text = text.replace('  IMPLICIT NONE\n', imports + '  IMPLICIT NONE\n')
    text = text.replace('  INTEGER :: ii, jj, kk\n', '  INTEGER :: ii, jj, kk\n' + declarations)
    text = text.replace(CONC_CALL, blocks['conc_entry'] + CONC_CALL + blocks['conc_exit'])
    text = text.replace(GAS_CALL, blocks['gas_entry'] + GAS_CALL + blocks['gas_exit'])
    instrumented = text.encode().replace(b'\n', newline)
    overlay = path.parent / 'ham-conversion-state.patch'
    overlay.write_text(''.join(difflib.unified_diff(original.decode().splitlines(True), instrumented.decode().splitlines(True),
                                                fromfile=path.name, tofile=path.name)))
    path.write_bytes(instrumented)
    sha = lambda data: hashlib.sha256(data).hexdigest()
    return {'original_driver_sha256': sha(original), 'instrumented_driver_sha256': sha(instrumented),
            'overlay_file': overlay.name, 'overlay_sha256': sha(overlay.read_bytes()),
            'helper_sha256': sha(Path(__file__).read_bytes())}


def load_capture(directory: Path) -> dict:
    directory = Path(directory)
    stages = {stage: parse_records((directory / name).read_text()) for stage, name in FILES.items()}
    entry = stages['conc_entry']
    for stage, fields in STAGES.items():
        expected = set(fields) | (set(INTEGERS) if stage.endswith('_entry') else set())
        if set(stages[stage]) != expected:
            raise ValueError(f'Missing or unexpected HAM conversion field: {stage}')
    dimensions = entry['dimensions']
    if dimensions['kind'] != 'I' or dimensions['shape'] != [9]:
        raise ValueError('Invalid HAM conversion dimensions record')
    sizes = dict(zip(('kproma', 'kbdim', 'klev', 'nclass', 'naerocomp', 'ntrac', 'ngas', 'nspec', 'naerospec'), dimensions['values']))
    if (any(value <= 0 for value in sizes.values()) or sizes['kproma'] > sizes['kbdim'] or
            sizes['nclass'] != 17 or sizes['naerocomp'] != 69 or sizes['naerospec'] > sizes['nspec']):
        raise ValueError('Invalid HAM conversion dimensions')
    for stage, fields in STAGES.items():
        for name, spec in {**(INTEGERS if stage.endswith('_entry') else {}), **fields}.items():
            shape = [int(dim) if dim.isdigit() else sizes[dim] for dim in spec[1]]
            record = stages[stage][name]
            if record['shape'] != shape or record['kind'] != ('I' if name in INTEGERS else 'R'):
                raise ValueError(f'Wrong HAM conversion shape/type: {stage}.{name}')
    for name in (*CONTEXT, *INTEGERS):
        if entry[name] != stages['gas_entry'][name]:
            raise ValueError(f'HAM conversion registry changed: {name}')
    if entry['options']['values'] != [3, 3, 64, 32] or entry['unit_codes']['values'] != [1, 2, 3]:
        raise ValueError('Unsupported HAM conversion options/unit codes')
    values = lambda name: entry[name]['values']
    material = values('material_species')
    if len(set(material)) != 5 or any(not 1 <= value <= sizes['nspec'] for value in material):
        raise ValueError('Invalid HAM material species map')
    component_map = values('component_map')
    present = []
    for species in range(5):
        for bin_index in range(17):
            component = component_map[bin_index + 17 * species]
            represented = (bin_index >= 3 or species < 2) and (bin_index < 10 or species != 3)
            if not represented:
                if component != 0:
                    raise ValueError('Unexpected absent component mapping')
                continue
            if not 1 <= component <= 69:
                raise ValueError('Invalid component mapping')
            present.append(component)
            if (values('component_bin')[component-1] != bin_index+1 or
                    values('component_species')[component-1] != material[species]):
                raise ValueError('Inconsistent component species/bin mapping')
    if sorted(present) != list(range(1, 70)):
        raise ValueError('Component mapping must contain each component once')
    for component, species in enumerate(values('component_species')):
        index = values('aerosol_species_index')[species-1]
        if not 1 <= index <= sizes['naerospec']:
            raise ValueError('Invalid aerosol species index')
        if values('component_conversion')[component] != values('aerosol_conversion')[index-1]:
            raise ValueError('Inconsistent aerosol conversion code')
    if any(code not in (1, 2) for code in values('component_conversion')):
        raise ValueError('Unsupported aerosol conversion code')
    gas_species = values('gas_species')
    if len(set(gas_species)) != sizes['ngas'] or any(not 1 <= value <= sizes['nspec'] for value in gas_species):
        raise ValueError('Invalid gas species map')
    if any(code not in (1, 2, 3) for code in values('gas_conversion')):
        raise ValueError('Unsupported gas conversion code')
    tracers = values('component_tracer') + values('number_tracer') + values('gas_tracer')
    if len(set(tracers)) != len(tracers) or any(not 1 <= value <= sizes['ntrac'] for value in tracers):
        raise ValueError('Invalid HAM tracer mapping')
    for name in ('mw_so4', 'avo', 'argas'):
        if values(name)[0] <= 0:
            raise ValueError(f'Nonpositive HAM conversion property: {name}')
    for species in material:
        if values('species_molar_mass')[species-1] <= 0 or values('species_density')[species-1] <= 0:
            raise ValueError('Nonpositive aerosol species property')
    for species in gas_species:
        if values('species_molar_mass')[species-1] <= 0:
            raise ValueError('Nonpositive gas molar mass')
    for stage, fields in STAGES.items():
        for name in ('rhoa', 'pressure', 'temperature', 'core'):
            if name in fields and any(value <= 0 for value in stages[stage][name]['values']):
                raise ValueError(f'Nonpositive HAM conversion input: {stage}.{name}')
        for name in ('pxtm1', 'paernl', 'paerml', 'gas'):
            if name in fields and any(value < 0 for value in stages[stage][name]['values']):
                raise ValueError(f'Negative HAM conversion state: {stage}.{name}')
    for before, after, names, changed in (
        ('conc_entry', 'conc_exit', ('core', 'rhoa'), values('component_tracer') + values('number_tracer')),
        ('gas_entry', 'gas_exit', tuple(GAS), values('gas_tracer')),
    ):
        for name in names:
            if stages[before][name] != stages[after][name]:
                raise ValueError(f'HAM conversion changed pure input: {name}')
        # Fortran stores one full horizontal/vertical slice per tracer.
        cells = sizes['kproma'] * sizes['klev']
        for tracer in set(range(1, sizes['ntrac']+1)) - set(changed):
            start, end = (tracer-1) * cells, tracer * cells
            if stages[before]['pxtm1']['values'][start:end] != stages[after]['pxtm1']['values'][start:end]:
                raise ValueError('HAM conversion changed an unrelated tracer')
    if stages['conc_exit']['pxtm1'] != stages['gas_entry']['pxtm1']:
        raise ValueError('Tracer state changed between initialization and first gas conversion')
    return {'schema_version': 1, 'order': 'Fortran column-major', 'dimensions': sizes,
            'options': {'nham_subm': 3, 'dp_bits': 64, 'default_real_bits': 32},
            'stage': 'driver initialization conc2mmr and first gas2mmr before HAM microphysics interface',
            **stages, 'units': {stage: {name: spec[2] for name, spec in fields.items()} for stage, fields in STAGES.items()}}


def load_fixture(directory: Path) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or set(manifest.get('files', {})) != set(FILES.values()):
        raise ValueError('Invalid HAM conversion fixture inventory')
    for name, expected in manifest['files'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'HAM conversion fixture hash mismatch: {name}')
    return load_capture(directory)


def export_fixture(run_directory: Path, target: Path) -> None:
    run_directory, target = Path(run_directory), Path(target)
    manifest_path = run_directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    capture = manifest.get('ham_conversion_capture', {})
    if (manifest.get('scope') != 'first-microphysics' or
            manifest.get('status') != 'first_microphysics_completed_unvalidated' or capture.get('status') != 'captured' or
            '-fcheck=all' not in manifest.get('config', {}).get('build', {}).get('flags', [])):
        raise ValueError('A successful checked HAM conversion capture is required')
    hashes = {name: hashlib.sha256((run_directory / name).read_bytes()).hexdigest() for name in FILES.values()}
    if hashes != capture.get('file_sha256'):
        raise ValueError('Run HAM conversion files changed after manifest creation')
    state = load_capture(run_directory)
    fixture = {'schema_version': 1, 'scope': state['stage'], 'files': hashes,
               'run_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               **{key: manifest[key] for key in ('source_revision', 'working_tree_status', 'source_sha256', 'input_sha256', 'compiler', 'config')},
               **({'reference_configuration': manifest['reference_configuration']} if 'reference_configuration' in manifest else {}),
               'dimensions': state['dimensions'], 'options': state['options'],
               'ham_conversion_overlay': {key: value for key, value in capture.items() if key not in ('state', 'file_sha256')},
               'runtime_metadata': manifest['runtime_capture']['metadata'],
               'scope_state_sha256': manifest['scope_capture']['state_sha256']}
    target.mkdir(parents=True, exist_ok=False)
    for name in FILES.values():
        shutil.copy2(run_directory / name, target / name)
    (target / 'manifest.json').write_text(json.dumps(fixture, indent=2) + '\n')
