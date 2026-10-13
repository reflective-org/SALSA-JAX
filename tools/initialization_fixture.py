# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Adapt the actual initialized-driver size distribution boundary to JAX inputs."""
from pathlib import Path
import numpy as np

from salsa_jax.constants import PI
from salsa_jax.grid import SizeGrid
from tools.initialization_capture import load_fixture

ROOT = Path(__file__).resolve().parents[1]


def load_native_case(directory=None):
    """Load hash-verified native modes/grid and active-cell number output.

    Returns (kernel arguments, expected number, capture). Modes and grid are
    global in the Fortran routine, so the expected horizontal/vertical cells
    share a single initialized population. No decimal defaults are substituted.
    """
    path = Path(directory) if directory is not None else ROOT/'tests/fixtures/initialization-default'
    capture = load_fixture(path)
    def record(stage, name):
        item = capture[stage][name]
        return np.asarray(item['values']).reshape(item['shape'], order='F')
    lower, upper, diameter = (record('entry', name) for name in ('vlolim', 'vhilim', 'dpmid'))
    middle = (PI/6.0)*diameter**3
    grid = SizeGrid(lower, upper, diameter, upper/middle, lower/middle)
    args = (*(record('entry', name) for name in ('n', 'dpg', 'sigmag')), grid)
    return args, record('exit', 'naero'), capture
