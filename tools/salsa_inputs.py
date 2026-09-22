"""Shared host-side grouping and argument adaptation for complete SALSA cases."""
import numpy as np
from salsa_jax import constants
from salsa_jax.interface import validate_component_map
from salsa_jax.model import STATIC_OPTIONS
from salsa_jax.processes.hydration import HydrationParameters

INPUT_NAMES=('mass','number','gas','temperature','relative_humidity','pressure','timestep','component_map')
GROUP_OPTIONS=tuple(key for key in STATIC_OPTIONS if key!='return_trace')


def select_cases(cases,indices):
    """Select case rows without losing labels or parameter/map alignment."""
    indices=np.asarray(indices,dtype=int)
    return {key:([value[i] for i in indices] if key=='labels' else np.asarray(value)[indices])
            for key,value in cases.items()}


def static_groups(cases):
    """Yield original row indices grouped by JAX's static process switches."""
    groups={}
    for index in range(len(cases['number'])):
        options=tuple(np.asarray(cases[key])[index].item() for key in GROUP_OPTIONS)
        groups.setdefault(options,[]).append(index)
    return [np.asarray(indices,dtype=int) for indices in groups.values()]


def arguments(cases,*,all_levels=False,return_trace=False):
    """Build salsa_step inputs; optionally duplicate each case above/within BL.

    Each group must share static switches. all_levels matches the whole Fortran
    oracle's two-level workload, including a native ppbl=2 boundary-layer mask.
    Fixed nucleation coefficients are checked when present, never silently lost.
    """
    validate_component_map(cases['component_map'])
    fixed={'cluster_molecules':constants.NUCLEATION_CLUSTER_MOLECULES,
           'acid_diameter':constants.SULFURIC_ACID_DIAMETER,
           'activation_coefficient':constants.NUCLEATION_ACTIVATION_COEFFICIENT}
    for key,expected in fixed.items():
        if key in cases and np.any(np.asarray(cases[key])!=expected):
            raise ValueError(f'Unsupported fixed nucleation constant: {key}')
    options={}
    for key in GROUP_OPTIONS:
        value=np.asarray(cases[key])
        allowed=(0,1,2,3) if key=='scheme' else (1,2,3) if key=='growth' else (0,1)
        if not np.all(np.isin(value,allowed)):
            raise ValueError(f'Unsupported static option {key}')
        if not value.size or np.any(value!=value.flat[0]):
            raise ValueError(f'Cases must share static option {key}')
        options[key]=int(value.flat[0]) if key in ('scheme','growth') else bool(value.flat[0])
    def value(key):
        array=np.asarray(cases[key])
        return np.repeat(array[:,None,...],2,axis=1) if all_levels else array
    parameters=HydrationParameters(**{key:value(key) for key in HydrationParameters._fields})
    boundary=(np.broadcast_to([False,True],(len(cases['number']),2)).copy()
              if all_levels else value('boundary_layer'))
    kwargs={'parameters':parameters,'mass_accommodation':value('mass_accommodation'),
            'regime_limits':value('regime_limits'),'ion_factor':value('ion_count'),
            'saturation_limit':value('saturation_limit'),'boundary_layer':boundary,
            'return_trace':return_trace,**options}
    return tuple(value(key) for key in INPUT_NAMES),kwargs
