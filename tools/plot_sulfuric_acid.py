# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Plot saved JAX/Fortran sulfuric-acid size distributions without smoothing."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(run_directory, output):
    run_directory, output = Path(run_directory), Path(output)
    report = json.loads((run_directory/'report.json').read_text())
    required = ('trajectory.npz', 'fortran-trajectory.npz', 'bins.csv')
    for name in required:
        digest = hashlib.sha256((run_directory/name).read_bytes()).hexdigest()
        if report.get('files', {}).get(name) != digest:
            raise ValueError(f'Input hash does not match run report: {name}')
    with np.load(run_directory/'trajectory.npz') as data:
        time = data['time_s'].copy()
        number = data['number'].copy()
        initial = data['initial_number'].copy()
    with np.load(run_directory/'fortran-trajectory.npz') as data:
        np.testing.assert_array_equal(time, data['time_s'])
        native = data['number'].copy()
    if number.shape != native.shape or number.shape[1:] != (1, 17):
        raise ValueError('Expected matching single-case, 17-bin trajectories')
    if np.any(number[..., 10:] != 0) or np.any(native[..., 10:] != 0) or np.any(initial[..., 10:] != 0):
        raise ValueError('This soluble-only plot requires empty insoluble 2b bins')
    bins = np.loadtxt(run_directory/'bins.csv', delimiter=',', skiprows=1)
    np.testing.assert_allclose(bins[1:10, 1], bins[:9, 3], rtol=1e-14, atol=0)
    edges = np.r_[bins[:10, 1], bins[9, 3]]*1e9
    midpoint = bins[:10, 2]*1e9
    width = np.log10(bins[:10, 3]/bins[:10, 1])
    jax_counts = np.concatenate((initial[:, :10], number[:, 0, :10]))*1e-6
    fortran_counts = np.concatenate((initial[:, :10], native[:, 0, :10]))*1e-6
    times = np.r_[0., time]
    indices = sorted(set(int(np.argmin(abs(times-t))) for t in (0., 1., 10., 100., 300., times[-1])))
    curves = [fortran_counts/width, jax_counts/width]
    colors = plt.colormaps['viridis'](np.linspace(.1, .9, len(indices)))
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout='constrained')
        fig.suptitle('SALSA sulfuric acid: Fortran and JAX size distributions', fontsize=16)
        for ax, values, title in zip(axes[0], curves, ('Checked original Fortran', 'JAX float64')):
            for index, color in zip(indices, colors):
                ax.stairs(values[index], edges, baseline=None, color=color, lw=1.8, label=f'{times[index]:g} s')
            ax.set_title(title)
            ax.set_yscale('symlog', linthresh=1.)
            ax.set_ylim(0, max(curves[1][indices].max(), 1)*1.7)
            ax.set_ylabel(r'$dN/d\log_{10}D$ [particles cm$^{-3}$]')
            ax.legend(ncol=3, fontsize=9, loc='upper right')
        ax = axes[1, 0]
        ax.stairs(curves[0][-1], edges, baseline=None, color='#222222', lw=3, label='Fortran')
        ax.stairs(curves[1][-1], edges, baseline=None, color='#df6600', lw=1.5, ls='--', label='JAX')
        ax.scatter(midpoint, curves[1][-1], color='#df6600', s=18, zorder=3)
        ax.set_title(f'Final overlay at {times[-1]:g} s — no smoothing')
        ax.set_yscale('symlog', linthresh=1.)
        ax.set_ylim(0, max(curves[1][-1].max(), 1)*1.7)
        ax.set_ylabel(r'$dN/d\log_{10}D$ [particles cm$^{-3}$]')
        ax.legend()
        difference = np.abs(number[:, 0, :10]-native[:, 0, :10])*1e-6
        ax = axes[1, 1]
        ax.plot(midpoint, difference.max(axis=0), 'o-', color='#3355aa', lw=1)
        ax.set_title('Largest number difference over all saved times')
        ax.set_ylabel(r'$|N_{JAX}-N_{Fortran}|$ [particles cm$^{-3}$ per bin]')
        ax.set_ylim(bottom=0)
        for ax in axes.flat:
            ax.set_xscale('log')
            ax.set_xlim(edges[0], edges[-1])
            ax.set_xlabel('Nominal dry bin diameter [nm]')
            ax.grid(True, which='major', alpha=.2)
        fig.get_layout_engine().set(rect=(0, .045, 1, .96))
        fig.text(.5, .016, '10 soluble bins; 7 duplicate-size insoluble bins are empty. Distributions use true log-bin widths.\n'
                 'Distribution y-axes are linear below 1 and logarithmic above 1, so zero counts remain visible.',
                 ha='center', va='bottom', fontsize=9)
        output.mkdir(parents=True, exist_ok=False)
        fig.savefig(output/'size-distributions.png', dpi=160)
        fig.savefig(output/'size-distributions.svg')
        plt.close(fig)
    relative = np.divide(np.abs(number-native), np.abs(native), out=np.zeros_like(number), where=native != 0)
    metrics = {'scope': 'Saved sulfuric-acid example only; shared initial state and matching native timesteps; no interpolation or smoothing',
               'run_directory': str(run_directory), 'matplotlib_version': matplotlib.__version__,
               'snapshot_times_s': times[indices].tolist(), 'bin_edges_nm': edges.tolist(),
               'maximum_absolute_number_difference_cm3': float(difference.max()),
               'maximum_relative_number_difference_nonzero': float(relative.max()),
               'nonzero_jax_at_zero_fortran': int(np.count_nonzero(number[native == 0])),
               'final_number_cm3_by_soluble_bin': jax_counts[-1].tolist(),
               'final_dndlog10d_cm3': curves[1][-1].tolist(),
               'input_sha256': {name: hashlib.sha256((run_directory/name).read_bytes()).hexdigest() for name in (*required, 'report.json')},
               'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'plot-report.json').write_text(json.dumps(metrics, indent=2)+'\n')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh directory for PNG, SVG and numerical report')
    args = parser.parse_args()
    print(json.dumps(plot(args.run_directory, args.output), indent=2))
