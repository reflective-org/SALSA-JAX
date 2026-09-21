.PHONY: clean
PROGRAM = ham_box
F90_SOURCE_FILES = $(wildcard *.F90)
F77_SOURCE_FILES = $(wildcard *.f)
C_SOURCE_FILES = $(wildcard *.c)
SOURCE_FILES = $(F90_SOURCE_FILES) $(F77_SOURCE_FILES) $(C_SOURCE_FILES)
MOD_FILES = $(F90_SOURCE_FILES:.F90=.o) $(F77_SOURCE_FILES:.f=.o)
OBJECT_FILES = $(F90_SOURCE_FILES:.F90=.o) $(F77_SOURCE_FILES:.f=.o) $(C_SOURCE_FILES:.c=.o)
F90 = gfortran
#F90FLAGS = -cpp -g -O -DNOMPI -D__STANDALONE -D__x86_64 -I./include -I/usr/include/
F90FLAGS = -cpp -g -fbacktrace -std=legacy -ffpe-trap=invalid,zero,overflow -O -DNOMPI -DSALSA -D__x86_64 -I./include -I/usr/include/
#F90LFLAGS = -cpp -g -O -DNOMPI -D__STANDALONE -D__x86_64 -I./include -I/usr/include/
LIBS = -lnetcdf -lnetcdff

all: $(PROGRAM)

all: $(PROG)

$(PROGRAM): $(OBJECT_FILES)
	$(F90) $(LDFLAGS) -o $@ $(OBJECT_FILES) $(LIBS)

clean:
	rm -f $(PROGAM_FILES) $(OBJECT_FILES) *.mod

.SUFFIXES: $(SUFFIXES) .F90

%.o: %.F90
	$(F90) $(F90FLAGS) -c $< -o $@


driver.o : driver.F90 mo_ham_m7_trac.o mo_species.o TM5M7_OPTICS_DATA.o mo_ham_rad.o mo_hammoz_drydep.o parkind1.o mo_convert_concentrations.o mo_hammoz_sedimentation.o mo_ham_activ.o mo_read_netcdf77.o mo_hammoz_wetdep.o mo_param_switches.o mo_activ.o mo_ham_subm.o mo_tracdef.o mo_ham_salsa_cloud.o mo_filename.o mo_submodel.o mo_math_constants.o mo_physical_constants.o mo_time_control.o mo_kind.o mo_ham_subm_species.o mo_ham_salsa_sizedist.o mo_ham_salsactl.o mo_ham_salsa.o mo_ham_init.o mo_ham.o mo_ham_salsa_init.o driver_input.o 
driver_input.o : driver_input.F90 mo_kind.o 
driver_tracer.o : driver_tracer.F90 mo_ham_salsactl.o mo_ham.o mo_species.o mo_physical_constants.o mo_tracdef.o mo_kind.o 
mo_activ.o : mo_activ.F90 mo_ham.o mo_control.o mo_math_constants.o mo_param_switches.o mo_physical_constants.o mo_kind.o 
mo_advection.o : mo_advection.F90 
mo_control.o : mo_control.F90 mo_kind.o 
mo_convert_concentrations.o : mo_convert_concentrations.F90 mo_exception.o mo_physical_constants.o mo_ham_salsactl.o mo_ham_m7ctl.o mo_ham.o mo_ham_species.o mo_species.o mo_kind.o 
mo_conv.o : mo_conv.F90 mo_kind.o 
mo_exception.o : mo_exception.F90 mo_io_units.o 
mo_filename.o : mo_filename.F90 
mo_ham_activ.o : mo_ham_activ.F90 mo_activ.o mo_conv.o mo_physical_constants.o mo_math_constants.o mo_tracdef.o mo_ham_tools.o mo_ham_m7ctl.o mo_ham.o mo_kind.o 
mo_ham_drydep.o : mo_ham_drydep.F90 mo_physical_constants.o mo_math_constants.o mo_ham.o mo_ham_m7ctl.o mo_tracdef.o mo_kind.o 
mo_ham.o : mo_ham.F90 mo_tracdef.o mo_physical_constants.o mo_submodel.o mo_param_switches.o mo_exception.o mo_util_string.o mo_radiation_parameters.o mo_species.o mo_kind.o 
mo_ham_init.o : mo_ham_init.F90 mo_param_switches.o mo_activ.o mo_ham_salsa_trac.o mo_ham_kappa.o mo_ham_m7_trac.o mo_advection.o mo_physical_constants.o mo_tracer.o mo_tracdef.o mo_species.o mo_exception.o mo_ham_species.o mo_ham_subm_species.o mo_ham_salsa_init.o mo_ham_salsactl.o mo_ham_m7ctl.o mo_ham.o 
mo_ham_kappa.o : mo_ham_kappa.F90 mo_netcdf.o mo_read_netcdf77.o mo_submodel.o mo_exception.o mo_kind.o 
mo_ham_m7ctl.o : mo_ham_m7ctl.F90 mo_util_string.o mo_exception.o mo_namelist.o mo_ham_wetdep_data.o mo_ham.o mo_species.o mo_physical_constants.o mo_math_constants.o mo_kind.o 
mo_ham_m7.o : mo_ham_m7.F90 mo_ham_soa.o mo_ham_m7_nucl.o mo_time_control.o mo_ham_species.o mo_ham_kappa.o mo_species.o mo_physical_constants.o mo_math_constants.o mo_ham_subm_species.o mo_ham.o mo_ham_m7ctl.o mo_kind.o 
mo_ham_m7_nucl.o : mo_ham_m7_nucl.F90 mo_netcdf.o mo_kind.o 
mo_ham_m7_trac.o : mo_ham_m7_trac.F90 mo_exception.o mo_ham_m7ctl.o mo_ham.o mo_ham_species.o mo_physical_constants.o mo_species.o mo_tracdef.o mo_kind.o 
mo_hammoz_drydep.o : mo_hammoz_drydep.F90 mo_ham_drydep.o mo_submodel.o mo_tracdef.o mo_physical_constants.o mo_time_control.o mo_exception.o mo_ham.o mo_kind.o 
mo_hammoz_sedimentation.o : mo_hammoz_sedimentation.F90 mo_ham.o mo_ham_sedimentation.o mo_time_control.o mo_tracdef.o mo_kind.o 
mo_hammoz_wetdep.o : mo_hammoz_wetdep.F90 mo_submodel.o mo_ham_wetdep.o mo_species.o mo_tracdef.o mo_physical_constants.o mo_time_control.o mo_ham.o mo_kind.o 
mo_ham_rad_data.o : mo_ham_rad_data.F90 mo_ham.o mo_kind.o 
mo_ham_rad.o : mo_ham_rad.F90 mo_read_netcdf77.o TM5M7_DATA.o MPL_MODULE.o YOMMP0.o mo_control.o mo_ham_salsactl.o parkind1.o TM5M7_OPTICS_DATA.o mo_tracdef.o mo_exception.o mo_physical_constants.o mo_math_constants.o mo_ham_m7ctl.o mo_ham_species.o mo_species.o mo_kind.o mo_ham.o mo_ham_rad_data.o 
mo_ham_salsa_cloud.o : mo_ham_salsa_cloud.F90 mo_param_switches.o mo_tracdef.o mo_math_constants.o mo_physical_constants.o mo_activ.o mo_ham_salsactl.o mo_ham_species.o mo_species.o mo_ham.o mo_kind.o 
mo_ham_salsactl.o : mo_ham_salsactl.F90 mo_submodel.o mo_util_string.o mo_exception.o mo_species.o mo_kind.o 
mo_ham_salsa_dynamics.o : mo_ham_salsa_dynamics.F90 mo_ham_subm_species.o mo_time_control.o mo_ham_salsa_nucleation.o mo_ham_species.o mo_species.o mo_ham.o mo_physical_constants.o mo_math_constants.o mo_kind.o mo_ham_salsa_init.o mo_ham_salsactl.o 
mo_ham_salsa.o : mo_ham_salsa.F90 mo_ham_subm_species.o mo_ham_species.o mo_species.o mo_ham_salsa_trac.o mo_ham_salsactl.o mo_ham.o mo_math_constants.o mo_physical_constants.o mo_ham_salsa_init.o mo_ham_salsa_update.o mo_ham_salsa_nucleation.o mo_ham_salsa_dynamics.o mo_ham_salsa_properties.o mo_kind.o 
mo_ham_salsa_init.o : mo_ham_salsa_init.F90 mo_exception.o mo_ham_wetdep_data.o mo_ham.o mo_physical_constants.o mo_ham_salsactl.o mo_math_constants.o mo_kind.o 
mo_ham_salsa_nucleation.o : mo_ham_salsa_nucleation.F90 mo_ham_species.o mo_species.o mo_ham_subm_species.o mo_ham.o mo_exception.o mo_kind.o mo_math_constants.o mo_physical_constants.o mo_ham_salsactl.o 
mo_ham_salsa_properties.o : mo_ham_salsa_properties.F90 mo_math_constants.o mo_physical_constants.o mo_ham_salsactl.o mo_ham_species.o mo_ham.o mo_species.o mo_kind.o 
mo_ham_salsa_sizedist.o : mo_ham_salsa_sizedist.F90 mo_math_constants.o mo_kind.o mo_ham_salsactl.o 
mo_ham_salsa_trac.o : mo_ham_salsa_trac.F90 mo_exception.o mo_ham.o mo_ham_salsactl.o mo_ham_species.o mo_species.o mo_tracdef.o mo_kind.o 
mo_ham_salsa_update.o : mo_ham_salsa_update.F90 mo_ham_species.o mo_species.o mo_physical_constants.o mo_math_constants.o mo_ham.o mo_kind.o mo_ham_salsactl.o 
mo_ham_sedimentation.o : mo_ham_sedimentation.F90 mo_ham.o mo_exception.o mo_ham_m7ctl.o mo_tracdef.o mo_time_control.o mo_physical_constants.o mo_kind.o 
mo_ham_soa.o : mo_ham_soa.F90 mo_species.o mo_ham_species.o mo_tracdef.o mo_ham.o mo_kind.o 
mo_ham_species.o : mo_ham_species.F90 mo_ham_rad_data.o mo_submodel.o mo_ham.o mo_species.o mo_tracdef.o mo_kind.o 
mo_ham_subm.o : mo_ham_subm.F90 mo_ham_m7ctl.o mo_ham_m7.o mo_ham_salsa.o mo_tracer_processes.o mo_ham_subm_species.o mo_ham.o mo_species.o mo_exception.o mo_time_control.o mo_physical_constants.o mo_kind.o 
mo_ham_subm_species.o : mo_ham_subm_species.F90 mo_util_string.o mo_exception.o mo_ham_soa.o mo_ham_species.o mo_ham.o mo_species.o 
mo_ham_tools.o : mo_ham_tools.F90 mo_math_constants.o mo_ham_m7.o mo_ham_m7ctl.o mo_exception.o mo_kind.o 
mo_ham_wetdep_data.o : mo_ham_wetdep_data.F90 mo_kind.o 
mo_ham_wetdep.o : mo_ham_wetdep.F90 mo_ham_species.o mo_species.o mo_ham_salsactl.o mo_ham_salsa_trac.o mo_param_switches.o mo_ham_tools.o mo_activ.o mo_ham_m7_trac.o mo_math_constants.o mo_ham_wetdep_data.o mo_ham_m7ctl.o mo_ham.o mo_time_control.o mo_tracdef.o mo_exception.o mo_physical_constants.o mo_kind.o 
mo_io_units.o : mo_io_units.F90 
mo_kind.o : mo_kind.F90 parkind1.o 
mo_math_constants.o : mo_math_constants.F90 mo_kind.o 
mo_namelist.o : mo_namelist.F90 mo_filename.o mo_util_string.o 
mo_netcdf.o : mo_netcdf.F90 mo_control.o mo_exception.o mo_kind.o 
mo_param_switches.o : mo_param_switches.F90 
mo_physical_constants.o : mo_physical_constants.F90 mo_kind.o 
mo_radiation_parameters.o : mo_radiation_parameters.F90 mo_control.o mo_math_constants.o mo_kind.o 
mo_read_netcdf77.o : mo_read_netcdf77.F90 mo_netcdf.o mo_exception.o mo_kind.o 
mo_species.o : mo_species.F90 mo_util_string.o mo_physical_constants.o mo_exception.o mo_tracdef.o mo_kind.o 
mo_submodel.o : mo_submodel.F90 mo_kind.o mo_namelist.o mo_exception.o mo_util_string.o mo_tracdef.o 
mo_time_control.o : mo_time_control.F90 mo_kind.o 
mo_tracdef.o : mo_tracdef.F90 mo_kind.o 
mo_tracer.o : mo_tracer.F90 mo_advection.o mo_exception.o mo_tracdef.o mo_util_string.o mo_kind.o 
mo_tracer_processes.o : mo_tracer_processes.F90 mo_submodel.o mo_advection.o mo_tracdef.o mo_time_control.o mo_physical_constants.o mo_kind.o 
mo_util_string.o : mo_util_string.F90 mo_kind.o 
MPL_MODULE.o : MPL_MODULE.F90 mo_kind.o 
oifs_to_ham.o : oifs_to_ham.F90 mo_ham.o 
parkind1.o : parkind1.F90 
TM5M7_DATA.o : TM5M7_DATA.F90 
TM5M7_INIT.o : TM5M7_INIT.F90 parkind1.o TM5M7_OPTICS_DATA.o 
TM5M7_OPTICS_DATA.o : TM5M7_OPTICS_DATA.F90 parkind1.o 
yomm7ctl.o : yomm7ctl.F90 parkind1.o 
YOMMP0.o : YOMMP0.F90 

mo_ham_vbsctl.o : mo_ham_vbsctl.F90 mo_kind.o
mo_ham_vbs.o : mo_ham_vbs.F90 mo_kind.o mo_ham_vbsctl.o mo_physical_constants.o mo_tracdef.o mo_species.o mo_ham_species.o
mo_ham_vbs_partition.o : mo_ham_vbs_partition.F90 mo_kind.o mo_ham_vbsctl.o mo_math_constants.o mo_physical_constants.o mo_species.o mo_ham_species.o mo_ham_salsactl.o mo_ham.o
