"""Constants copied from the repository's Fortran modules for numerical parity."""

# mo_physical_constants.F90: ak and p0sl_bg. Do not substitute newer values.
BOLTZMANN = 1.3806504e-23  # J K-1
REFERENCE_PRESSURE = 101325.0  # Pa
# mo_math_constants.F90: pi
PI = 3.14159265358979323846264338327950288419717

# mo_ham_salsa_dynamics.F90: coagulation particle-mass approximation.
COAGULATION_DENSITY = 1500.0  # kg m-3

# mo_physical_constants.F90, preserved legacy values.
AVOGADRO = 6.02214179e23  # mol-1
GAS_CONSTANT = 8.314472  # J K-1 mol-1
# Default HAM species and mo_ham_salsactl condensation settings.
SULFATE_MOLAR_MASS = 96.0631  # g mol-1; legacy species value, not 98
ORGANIC_MOLAR_MASS = 180.0  # g mol-1
SULFATE_DENSITY = 1841.0  # kg m-3
ORGANIC_DENSITY = 2000.0  # kg m-3
SULFURIC_ACID_DIAMETER = 5.539376964394570e-10  # m
CONDENSATION_GAS_THRESHOLD = 1e-10  # molecules m-3, strict greater-than
CONDENSATION_SINK_THRESHOLD = 1e-30  # s-1, strict greater-than
