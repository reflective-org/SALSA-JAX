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


@pytest.mark.parametrize("flag", ["--capture-salsa-state", "--capture-coagulation-state", "--capture-condensation-state", "--capture-nucleation-state", "--capture-hydration-state", "--capture-remapping-state"])
def test_capture_requires_single_call_scope(monkeypatch, flag):
    monkeypatch.setattr(sys, 'argv', ['reference', '--output', 'unused', flag])
    with pytest.raises(SystemExit) as error:
        harness.main()
    assert error.value.code == 2


@pytest.mark.parametrize('complete', [False, True])
@pytest.mark.parametrize('process', ['salsa', 'coagulation', 'condensation', 'nucleation', 'hydration', 'remapping'])
def test_requested_capture_must_exist_even_on_success(tmp_path, monkeypatch, complete, process):
    from tools.coagulation_capture import FILES as COAG_FILES
    from tools.condensation_capture import FILES as COND_FILES
    from tools.nucleation_capture import FILES as NUC_FILES
    from tools.hydration_capture import FILES as HYD_FILES
    from tools.remapping_capture import FILES as REMAP_FILES
    files = {'salsa': FILES, 'coagulation': COAG_FILES, 'condensation': COND_FILES, 'nucleation': NUC_FILES, 'hydration': HYD_FILES, 'remapping': REMAP_FILES}[process]
    fixture = FIXTURE if process == 'salsa' else FIXTURE.parent / f'{process}-default'
    if process == 'nucleation':
        fixture = FIXTURE.parent / 'nucleation-active'
    root, target = tmp_path / 'source', tmp_path / 'run'
    root.mkdir()
    real_root = harness.ROOT
    for name in ('driver.F90', 'mo_ham_salsa.F90', 'mo_ham_salsa_dynamics.F90'):
        shutil.copy2(real_root / name, root / name)
    for name in ('Makefile', 'Makefile.in', '.depend', 'configure', 'namelist.echam', 'lut_kappa.nc'):
        (root / name).write_text('fixture\n')
    monkeypatch.setattr(harness, 'ROOT', root)
    monkeypatch.setattr(harness, 'output', lambda *args: 'test-version')
    monkeypatch.setattr(harness, 'parse_metadata', lambda text: {'values': {'ntrac':109, 'nclass':17, 'kbdim':1, 'klev':1, 'nsnucl':2 if process == 'nucleation' else 0, 'nj3':1}})

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
            if process == 'nucleation':
                for name in COND_FILES.values():
                    shutil.copy2(FIXTURE.parent / 'condensation-active' / name, directory / name)
            if complete:
                for name in files.values():
                    shutil.copy2(fixture / name, directory / name)
        return {'status':'finished', 'returncode':0}

    monkeypatch.setattr(harness, 'logged_run', fake_run)
    profile = 'fortran-nucleation-active.toml' if process == 'nucleation' else 'fortran-checked.toml'
    monkeypatch.setattr(sys, 'argv', ['reference', '--config', str(real_root / 'configs' / profile),
                        '--output', str(target), '--scope', 'first-microphysics', f'--capture-{process}-state', '--run'])
    assert harness.main() == (0 if complete else 1)
    manifest = json.loads((target / 'manifest.json').read_text())
    assert manifest['status'] == ('first_microphysics_completed_unvalidated' if complete else f'{process}_capture_failed')
    assert manifest[f'{process}_capture']['status'] == ('captured' if complete else 'capture_failed')


def test_process_profile_requires_first_call_scope(tmp_path, monkeypatch):
    config = tmp_path / "profile.toml"
    config.write_text('[build]\nflags = []\n[process_switches]\nrecalc = true\n')
    monkeypatch.setattr(sys, "argv", ["reference", "--config", str(config), "--output", str(tmp_path / "run")])
    with pytest.raises(SystemExit) as error:
        harness.main()
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


def test_runtime_recalc_mismatch_fails_profile_capture(tmp_path, monkeypatch):
    root, target = tmp_path / "source", tmp_path / "run"
    root.mkdir()
    real_root = harness.ROOT
    shutil.copy2(real_root / "driver.F90", root / "driver.F90")
    for name in ("Makefile", "Makefile.in", ".depend", "configure", "namelist.echam", "lut_kappa.nc"):
        (root / name).write_text("fixture\n")
    monkeypatch.setattr(harness, "ROOT", root)
    monkeypatch.setattr(harness, "output", lambda *args: "test-version")
    monkeypatch.setattr(harness, "parse_metadata", lambda text: {"values": {"recalc": False}})
    def fake_run(command, directory, log, timeout):
        if command[0] == "make":
            (directory / "ham_box").write_bytes(b"executable")
        else:
            (directory / "runtime-metadata.txt").write_text("runtime\n")
        return {"status": "finished", "returncode": 0}
    monkeypatch.setattr(harness, "logged_run", fake_run)
    monkeypatch.setattr(sys, "argv", ["reference", "--config", str(real_root / "configs/fortran-recalc.toml"),
                                     "--output", str(target), "--scope", "first-microphysics", "--run"])
    assert harness.main() == 1
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["status"] == "runtime_capture_failed"
    assert "Runtime recalc differs" in manifest["runtime_capture"]["error"]
