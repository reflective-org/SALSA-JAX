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

# mo_ham_salsactl.F90: supported sulfate nucleation defaults.
NUCLEATION_ACTIVATION_COEFFICIENT = 1e-7  # s-1
NUCLEATION_CLUSTER_MOLECULES = 158.79  # molecules per target particle
NUCLEATION_TARGET_DIAMETER = 3e-9  # m, reglim(1)
NUCLEATION_CRITICAL_DIAMETER = 7.9375e-10  # m, activation/kinetic schemes

# mo_ham_salsa_properties and initialized five-material hydration registry.
HYDRATION_SPECIES_MOLAR_MASS = (SULFATE_MOLAR_MASS, ORGANIC_MOLAR_MASS, 12.01, 58.443, 250.0)  # g/mol
HYDRATION_SPECIES_DENSITY = (SULFATE_DENSITY, ORGANIC_DENSITY, 2000.0, 2165.0, 2650.0)  # kg/m3
WATER_MOLAR_MASS = 18.0  # g/mol
WATER_DENSITY = 1000.0  # kg/m3
WATER_SURFACE_TENSION = 0.073  # J/m2, surfw0
ORGANIC_SOLUBLE_FRACTION = 0.15  # epsoc
HYDRATION_NUMBER_THRESHOLD = 1.0  # particles/m3, strict greater-than

# mo_ham_salsactl.F90: insoluble-to-soluble remapping criterion.
REMAPPING_SATURATION_LIMIT = 1.005  # slim, relative water saturation
REMAPPING_ION_FACTOR = 3.0  # ions, van't Hoff factor
# mo_ham_salsa.F90: strict tiny-particle cleanup comparison.
TINY_PARTICLE_DIAMETER = 1e-10  # m

# mo_ham_salsa.F90: density fallback uses a strict total-volume threshold.
DENSITY_VOLUME_THRESHOLD = 1e-30  # m3 material and water per m3 air

# gas2mmr unsuffixed 1e-3: binary32 literal promoted to double precision.
GAS_MOLECULAR_CONVERSION_FACTOR = 1.0000000474974513e-3
