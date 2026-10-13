"""Summarize paired long runs, timestep refinements and the remapping control."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(acid, coagulation, acid_half, coagulation_half, control):
    directories = dict(acid=acid, coagulation=coagulation, acid_half=acid_half,
                       coagulation_half=coagulation_half, control=control)
    result = {'runs': {}, 'timestep_refinement': {}, 'control_drift': {}}
    for name, directory in directories.items():
        directory = Path(directory)
        report = json.loads((directory/'report.json').read_text())
        for filename, expected in report['files'].items():
            if hashlib.sha256((directory/filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Changed input: {directory/filename}')
        result['runs'][name] = report
    for name, coarse, fine in [('acid', acid, acid_half), ('coagulation', coagulation, coagulation_half)]:
        result['timestep_refinement'][name] = {}
        for implementation in ('jax', 'fortran'):
            with np.load(Path(coarse)/f'{implementation}.npz') as a, np.load(Path(fine)/f'{implementation}.npz') as b:
                np.testing.assert_array_equal(a['time_s'], b['time_s'])
                metrics = {}
                for key in ('number', 'mass'):
                    relative = np.abs(a[key]-b[key]).sum(axis=(1, 2))/np.abs(b[key]).sum(axis=(1, 2))
                    metrics[key] = {'maximum_relative_L1': float(relative.max()),
                                    'final_relative_L1': float(relative[-1])}
                result['timestep_refinement'][name][implementation] = metrics
    with np.load(Path(control)/'initial.npz') as initial:
        for implementation in ('jax', 'fortran'):
            with np.load(Path(control)/f'{implementation}.npz') as data:
                result['control_drift'][implementation] = {}
                for key in ('number', 'mass'):
                    relative = np.abs(data[key]-initial[key]).sum(axis=(1, 2))/np.abs(initial[key]).sum()
                    result['control_drift'][implementation][key] = {
                        'maximum_relative_L1_from_initial': float(relative.max()),
                        'final_relative_L1_from_initial': float(relative[-1])}
    result['definitions'] = {
        'relative_L1': 'sum(abs(coarse-fine))/sum(abs(fine)), summed over bins/components and cells; maximum over saved times',
        'mass': 'Native aerosol sulfate molecule concentration; other aerosol species are zero in these runs',
        'control': 'No coagulation, condensation or nucleation; native hydration/remapping/correction still execute',
        'limits': 'Small port differences and budgets do not independently establish redistribution accuracy. All strict acceptance flags retained.'}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('acid', 'coagulation', 'acid-half', 'coagulation-half', 'control'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.acid, args.coagulation, args.acid_half, args.coagulation_half, args.control)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
