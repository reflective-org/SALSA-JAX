"""An executable success cannot replace valid requested native capture evidence."""
import json
from pathlib import Path
import shutil
import sys

import pytest

from tools import fortran_reference as harness
from tools.reference_scope import COMPLETION_LINES, FINITE_FIELDS, STATE_FILENAME
from tools.state_capture import FILES

FIXTURE = Path(__file__).resolve().parent / 'fixtures/native-default'


@pytest.mark.parametrize("flag", ["--capture-salsa-state", "--capture-coagulation-state"])
def test_capture_requires_single_call_scope(monkeypatch, flag):
    monkeypatch.setattr(sys, 'argv', ['reference', '--output', 'unused', flag])
    with pytest.raises(SystemExit) as error:
        harness.main()
    assert error.value.code == 2


@pytest.mark.parametrize('complete', [False, True])
@pytest.mark.parametrize('process', ['salsa', 'coagulation'])
def test_requested_capture_must_exist_even_on_success(tmp_path, monkeypatch, complete, process):
    from tools.coagulation_capture import FILES as COAG_FILES
    files = FILES if process == 'salsa' else COAG_FILES
    fixture = FIXTURE if process == 'salsa' else FIXTURE.parent / 'coagulation-default'
    root, target = tmp_path / 'source', tmp_path / 'run'
    root.mkdir()
    real_root = harness.ROOT
    for name in ('driver.F90', 'mo_ham_salsa.F90'):
        shutil.copy2(real_root / name, root / name)
    for name in ('Makefile', 'Makefile.in', '.depend', 'configure', 'namelist.echam', 'lut_kappa.nc'):
        (root / name).write_text('fixture\n')
    monkeypatch.setattr(harness, 'ROOT', root)
    monkeypatch.setattr(harness, 'output', lambda *args: 'test-version')
    monkeypatch.setattr(harness, 'parse_metadata', lambda text: {'values': {'ntrac':109, 'nclass':17, 'kbdim':1, 'klev':1}})

    def fake_run(command, directory, log, timeout):
        if command[0] == 'make':
            (directory / 'ham_box').write_bytes(b'fixture executable')
        else:
            (directory / 'runtime-metadata.txt').write_text('fixture metadata\n')
            (directory / 'reference-scope.txt').write_text('\n'.join(COMPLETION_LINES) + '\n')
            lines = ['SALSA_ARRAYS 1']
            for field in FINITE_FIELDS:
                extent = 109 if field in ('pxtm1', 'pxtte') else 17
                lines += [f'R {field} 3 1 1 {extent}', *['1.0'] * extent]
            lines.append('END_SALSA_ARRAYS')
            (directory / STATE_FILENAME).write_text('\n'.join(lines) + '\n')
            if complete:
                for name in files.values():
                    shutil.copy2(fixture / name, directory / name)
        return {'status':'finished', 'returncode':0}

    monkeypatch.setattr(harness, 'logged_run', fake_run)
    monkeypatch.setattr(sys, 'argv', ['reference', '--config', str(real_root / 'configs/fortran-checked.toml'),
                        '--output', str(target), '--scope', 'first-microphysics', f'--capture-{process}-state', '--run'])
    assert harness.main() == (0 if complete else 1)
    manifest = json.loads((target / 'manifest.json').read_text())
    assert manifest['status'] == ('first_microphysics_completed_unvalidated' if complete else f'{process}_capture_failed')
    assert manifest[f'{process}_capture']['status'] == ('captured' if complete else 'capture_failed')
