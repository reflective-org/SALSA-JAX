"""Complete unmodified native SALSA call with fixture-initialized data modules.

The five scientific modules are compiled byte-for-byte. Only initialization
(set_sizebins/coagc) is extracted; no Python reconstruction chooses process order.
Repeated steps carry native outputs forward, while timed repetitions reset state.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import subprocess
import numpy as np
from tools.coagulation_process_oracle import FLAGS, CHECK_FLAGS, _extract
from tools.state_capture import load_fixture

ROOT=Path(__file__).resolve().parents[1]
BASE_SOURCES=('parkind1.F90','mo_kind.F90','mo_math_constants.F90','mo_physical_constants.F90')
PROCESS_SOURCES=('mo_ham_salsa_nucleation.F90','mo_ham_salsa_properties.F90','mo_ham_salsa_dynamics.F90','mo_ham_salsa_update.F90','mo_ham_salsa.F90')
OPTIONS=('coagulation_enabled','condensation_enabled','recalculate','transfer_soluble','hydrate_insoluble','organic_enabled','scheme','growth','boundary_layer')
PROPERTIES=('number_threshold','cluster_molecules','acid_diameter','organic_soluble_fraction','saturation_limit','ion_count','water_surface_tension','activation_coefficient')
OUTPUTS={'mass':69,'number':17,'gas':3,'wet_radius_cm':17,'dry_radius_cm':17,'density':17,'water_mass':17}
STUBS='''module mo_species
 use mo_kind,only:dp
 implicit none
 type species_type
 real(dp)::moleweight,density
 end type
 type(species_type)::speclist(6)
end module
module mo_ham_species
 integer,parameter::id_so4=1,id_oc=2,id_bc=3,id_ss=4,id_du=5,id_wat=6
end module
module mo_ham
 integer,parameter::naerocomp=69,nclass=17,nsoa=0,subm_naerospec_nowat=5,subm_aerospec_nowat(5)=[1,2,3,4,5]
 integer::subm_ngasspec=2
 logical::lscoag=.true.,lscond=.true.
 type bin_type
 logical::lsoluble
 end type
 type(bin_type)::sizeclass(17)
end module
module mo_ham_salsactl
 use mo_kind,only:dp
 integer,parameter::in1a=1,fn1a=3,in2a=4,fn2a=10,in2b=11,fn2b=17,nbin(2)=[3,7],nbin2=4,nbin3=3
 integer::nsnucl=0,nj3=1,iso4b(17),iocb(17),ibcb(17),issb(17),idub(17)
 logical::recalc=.false.,lsol2b=.false.,lhydr2b=.true.
 real(dp)::dpmid(17),vlolim(17),vhilim(17),vratiohi(17),vratiolo(17),reglim(4),massacc(17)
 real(dp)::nlim,n3,d_sa,epsoc,slim,ions,surfw0,act_coeff
end module
module mo_ham_salsa_trac
 integer::idt_ms4(17),idt_moc(17),idt_mbc(17),idt_mss(10),idt_mdu(17)
end module
module mo_ham_subm_species
 integer::isubm_so4g=2,isubm_ocnv=0
end module
module mo_time_control
 use mo_kind,only:dp
 real(dp)::time_step_len=1._dp
end module
module mo_exception
contains
 subroutine finish(where,message)
 character(len=*),intent(in)::where,message
 write(*,*) where,': ',message
 error stop 'Unsupported native configuration'
 end subroutine
end module
'''
DRIVER='''program salsa_reference
 use iso_fortran_env,only:int64
 use mo_kind,only:dp
 use mo_species
 use mo_ham
 use mo_ham_salsactl
 use mo_ham_subm_species
 use mo_time_control
 use mo_ham_salsa_init,only:set_sizebins
 use mo_ham_salsa,only:salsa
 use mo_physical_constants,only:avo,argas,ak,p0sl_bg
 use mo_math_constants,only:pi
 implicit none
 integer::n,repetitions,steps,i,j,k,level,ngas
 integer(int64)::wall_start,wall_end,clock_rate
 integer,allocatable::options(:,:)
 integer,allocatable::mapping(:,:,:)
 real(dp),allocatable::initial_mass(:,:),initial_number(:,:),initial_gas(:,:),environment(:,:),properties(:,:),masses(:,:),densities(:,:),limits(:,:),accommodation(:,:)
 real(dp),allocatable::mass_out(:,:,:),number_out(:,:,:),gas_out(:,:,:),wet_out(:,:,:),dry_out(:,:,:),rho_out(:,:,:),water_out(:,:,:)
 real(dp)::mass(1,2,69),number(1,2,17),gas(1,2,3),wet(1,2,17),dry(1,2,17),rho(1,2,17),water(1,2,17)
 real(dp)::temperature(1,2),pressure(1,2),rh(1,2),pbl(1),started,finished
 character(len=32)::command
 call get_command_argument(1,command)
 if(trim(command)=='--constants') then
 write(*,'(ES26.17E3)') avo,argas,ak,p0sl_bg,pi
 stop
 end if
 read(*,*) n,repetitions,steps
 allocate(mapping(17,5,n))
 allocate(options(9,n),initial_mass(69,n),initial_number(17,n),initial_gas(3,n),environment(4,n),properties(8,n),masses(6,n),densities(6,n),limits(4,n),accommodation(17,n))
 allocate(mass_out(69,2,n),number_out(17,2,n),gas_out(3,2,n),wet_out(17,2,n),dry_out(17,2,n),rho_out(17,2,n),water_out(17,2,n))
 do i=1,n
 read(*,*) mapping(:,:,i)
 read(*,*) options(:,i)
 read(*,*) environment(:,i)
 read(*,*) properties(:,i)
 read(*,*) masses(:,i)
 read(*,*) densities(:,i)
 read(*,*) limits(:,i)
 read(*,*) accommodation(:,i)
 read(*,*) initial_mass(:,i)
 read(*,*) initial_number(:,i)
 read(*,*) initial_gas(:,i)
 end do
 do i=1,n
 call run_case(i)
 end do
 call cpu_time(started)
 call system_clock(wall_start,clock_rate)
 do j=1,repetitions
 do i=1,n
 call run_case(i)
 end do
 end do
 call system_clock(wall_end)
 call cpu_time(finished)
 write(*,'(6ES26.17E3)') real(wall_end-wall_start,dp)/real(clock_rate,dp),finished-started,sum(mass_out),sum(number_out),sum(gas_out),real(clock_rate,dp)
 do i=1,n
 do level=1,2
 write(*,'(ES26.17E3)') mass_out(:,level,i),number_out(:,level,i),gas_out(:,level,i),wet_out(:,level,i),dry_out(:,level,i),rho_out(:,level,i),water_out(:,level,i)
 end do
 end do
contains
 subroutine run_case(index)
 integer,intent(in)::index
 integer::s,bin,lev
 iso4b=mapping(:,1,index);iocb=mapping(:,2,index);ibcb=mapping(:,3,index);issb=mapping(:,4,index);idub=mapping(:,5,index)
 lscoag=options(1,index)==1;lscond=options(2,index)==1;recalc=options(3,index)==1
 lsol2b=options(4,index)==1;lhydr2b=options(5,index)==1
 isubm_ocnv=3*options(6,index);subm_ngasspec=2+options(6,index)
 nsnucl=options(7,index);nj3=options(8,index);pbl=2;level=1+options(9,index)
 do bin=1,17
 sizeclass(bin)%lsoluble=bin<=10.or.lhydr2b
 end do
 nlim=properties(1,index);n3=properties(2,index);d_sa=properties(3,index);epsoc=properties(4,index)
 slim=properties(5,index);ions=properties(6,index);surfw0=properties(7,index);act_coeff=properties(8,index)
 do s=1,6
 speclist(s)=species_type(masses(s,index),densities(s,index))
 end do
 reglim=limits(:,index);massacc=accommodation(:,index)
 call set_sizebins()
 temperature=environment(1,index);pressure=environment(2,index);rh=environment(3,index);time_step_len=environment(4,index)
 do lev=1,2
 mass(1,lev,:)=initial_mass(:,index);number(1,lev,:)=initial_number(:,index);gas(1,lev,:)=initial_gas(:,index)
 end do
 wet=0._dp;dry=0._dp;rho=0._dp;water=0._dp
 do s=1,steps
 call salsa(1,1,2,1,pressure,rh,temperature,environment(4,index),gas,mass,number,wet,dry,rho,water,pbl)
 end do
 mass_out(:,:,index)=transpose(mass(1,:,:));number_out(:,:,index)=transpose(number(1,:,:));gas_out(:,:,index)=transpose(gas(1,:,:))
 wet_out(:,:,index)=transpose(wet(1,:,:));dry_out(:,:,index)=transpose(dry(1,:,:));rho_out(:,:,index)=transpose(rho(1,:,:));water_out(:,:,index)=transpose(water(1,:,:))
 end subroutine
end program
'''


def build_oracle(build_directory,compiler='gfortran',*,checked=True):
    directory=Path(build_directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    init=(ROOT/'mo_ham_salsa_init.F90').read_text()
    routines={name:_extract(init,kind,name) for name,kind in (('coagc','FUNCTION'),('set_sizebins','SUBROUTINE'))}
    wrappers={'stubs.f90':STUBS,'init.F90':'module mo_ham_salsa_init\ncontains\n'+'\n'.join(routines.values())+'\nend module\n','driver.f90':DRIVER}
    for name,text in wrappers.items():(directory/name).write_text(text)
    for name in PROCESS_SOURCES:(directory/name).write_bytes((ROOT/name).read_bytes())
    flags=FLAGS+CHECK_FLAGS if checked else FLAGS
    paths=[*[ROOT/name for name in BASE_SOURCES],directory/'stubs.f90',directory/'init.F90',*[directory/name for name in PROCESS_SOURCES],directory/'driver.f90']
    objects=[]
    for source in paths:
        obj=directory/(source.stem+'.o')
        subprocess.run([compiler,*flags,'-c',str(source),'-o',str(obj)],cwd=directory,check=True,capture_output=True,text=True)
        objects.append(str(obj))
    executable=directory/'salsa_oracle'
    subprocess.run([compiler,*flags,*objects,'-o',str(executable)],cwd=directory,check=True,capture_output=True,text=True)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    constants=np.fromstring(subprocess.check_output([str(executable),'--constants'],text=True),sep=' ')
    return executable,{'source_sha256':{name:sha((ROOT/name).read_bytes()) for name in (*BASE_SOURCES,*PROCESS_SOURCES,'mo_ham_salsa_init.F90')},'wrapper_sha256':{name:sha(value.encode()) for name,value in wrappers.items()},'extracted_routine_sha256':{name:sha(value.encode()) for name,value in routines.items()},'flags':list(flags),'checked':checked,'compiled_constants':dict(zip(('avogadro','gas_constant','boltzmann','reference_pressure','pi'),constants.tolist())),'compiler':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],'scientific_modules_byte_identical':list(PROCESS_SOURCES),'scope':'complete salsa call; nsoa=0, no HAMMOZ, native 69-component mapping, 17 bins, five nonwater species','separate_compilation_without_lto':True}


def record(fixture,stage,name):
    item=fixture[stage][name]
    return np.asarray(item['values']).reshape(item['shape'],order='F')


def native_case(fixture=None):
    fixture=load_fixture(ROOT/'tests/fixtures/native-default') if fixture is None else fixture
    r=lambda name:record(fixture,'entry',name)
    ids=np.r_[r('volume_species').astype(int)-1,fixture['mapping']['species_names'].index('WAT')]
    case={'mass':r('paerml')[0,0].copy(),'number':r('pnaero')[0,0].copy(),'gas':np.pad(r('pc_gas')[0,0],(0,3-len(r('pc_gas')[0,0]))),
          'temperature':float(r('ptemp')[0,0]),'pressure':float(r('ppres')[0,0]),'relative_humidity':float(r('prh')[0,0]),'timestep':float(r('ptstep')),
          'species_molar_mass':r('species_molar_mass')[ids[:5]],'species_density':r('species_density')[ids[:5]],
          'water_molar_mass':float(r('species_molar_mass')[ids[5]]),'water_density':float(r('species_density')[ids[5]]),
          'component_map':r('material_map').astype(int)-1,
          'regime_limits':r('reglim').copy(),'mass_accommodation':r('massacc').copy(),
          **dict(zip(PROPERTIES,r('constants').tolist()))}
    names=('lscoag','lscond','recalc','lsol2b','lhydr2b','isubm_ocnv','nsnucl','nj3')
    case.update({key:int(fixture['options'][name]) if key in ('scheme','growth') else bool(fixture['options'][name]) for key,name in zip(OPTIONS,names)})
    case['boundary_layer']=bool(r('ppbl')[0]<=1)
    return case,r('material_map').astype(int)


def reference_cases():
    base,mapping=native_case();rows=[];labels=[]
    def add(label,**changes):
        rows.append({**{k:np.array(v,copy=True) for k,v in base.items()},**changes});labels.append(label)
    add('captured-default')
    for coag in (False,True):
        for cond in (False,True):
            for recalc in (False,True):
                add(f'flags-coag{int(coag)}-cond{int(cond)}-recalc{int(recalc)}',coagulation_enabled=coag,condensation_enabled=cond,recalculate=recalc,gas=np.array([1e12,1e14,2e13]),organic_enabled=True)
    add('soluble-transfer',transfer_soluble=True)
    add('unhydrated-insoluble',hydrate_insoluble=False)
    add('zero-timestep',timestep=0.)
    add('empty',mass=np.zeros(69),number=np.zeros(17),gas=np.zeros(3))
    for scheme in (1,2,3):
        for growth in (1,2,3):
            add(f'active-scheme{scheme}-growth{growth}',scheme=scheme,growth=growth,gas=np.array([0.,1e14,0.]))
    add('active-above-boundary',scheme=2,gas=np.array([0.,1e14,0.]),boundary_layer=False)
    add('active-organic',scheme=2,gas=np.array([1e12,1e14,1e13]),organic_enabled=True)
    avogadro=6.02214179e23
    for bin_index in (0,3,10):
        mass=np.zeros(69);number=np.zeros(17);number[bin_index]=1e6
        total=number[bin_index]*(np.pi/6)*1e-11**3
        mass[mapping[bin_index,0]-1]=.6*total/(base['species_molar_mass'][0]/1000/avogadro/base['species_density'][0])/1e6
        mass[mapping[bin_index,1]-1]=.4*total*base['species_density'][1]*1e9
        add(f'initial-tiny-cleanup-bin{bin_index+1}',mass=mass,number=number,gas=np.zeros(3),organic_enabled=True,coagulation_enabled=False,condensation_enabled=False)
    return {'labels':labels,**{k:np.stack([row[k] for row in rows]) for k in base}}


def run_oracle(executable,cases,repetitions=1,*,steps=1,all_levels=False,timeout=30):
    number=np.asarray(cases['number'],dtype=float)
    if number.ndim!=2 or number.shape[1]!=17 or not len(number):raise ValueError('number must have shape (cases,17)')
    if any(not isinstance(x,(int,np.integer)) or x<1 for x in (repetitions,steps)):raise ValueError('repetitions and steps must be positive integers')
    n=len(number);mapping=np.asarray(cases['component_map'])
    _,native_map=native_case()
    if mapping.shape!=(n,17,5) or np.any(mapping!=mapping.astype(int)):raise ValueError('Invalid component_map')
    for row in mapping:
        if not np.array_equal(row>=0,native_map>0) or not np.array_equal(np.sort(row[row>=0]),np.arange(69)):raise ValueError('Unsupported component_map')
    base,_=native_case()
    cases={**{key:np.full(n,base[key]) for key in ('cluster_molecules','acid_diameter','activation_coefficient')},**cases}
    shapes={'mass':(n,69),'number':(n,17),'gas':(n,3),'species_molar_mass':(n,5),'species_density':(n,5),'water_molar_mass':(n,),'water_density':(n,),'regime_limits':(n,4),'mass_accommodation':(n,17),**{k:(n,) for k in (*OPTIONS,*PROPERTIES,'temperature','pressure','relative_humidity','timestep')}}
    values={}
    for key,shape in shapes.items():
        a=np.asarray(cases[key],dtype=float)
        if a.shape!=shape or not np.all(np.isfinite(a)):raise ValueError(f'Invalid {key} shape or nonfinite value')
        if np.any(a<0):raise ValueError(f'Negative {key}')
        values[key]=a
    if np.any(~np.isin(values['scheme'],[0,1,2,3])) or np.any(~np.isin(values['growth'],[1,2,3])):raise ValueError('Unsupported scheme or growth')
    for key in OPTIONS:
        if key not in ('scheme','growth') and np.any(~np.isin(values[key],[0,1])):raise ValueError(f'Invalid {key}')
    for key in ('temperature','pressure','species_molar_mass','species_density','regime_limits','acid_diameter','saturation_limit','ion_count','water_surface_tension','cluster_molecules','water_molar_mass','water_density'):
        if np.any(values[key]<=0):raise ValueError(f'Nonpositive {key}')
    lines=[f'{n} {repetitions} {steps}']
    for i in range(n):
        lines.append(' '.join(str(int(v)+1) for v in mapping[i].ravel(order='F')))
        lines.append(' '.join(str(int(values[k][i])) for k in OPTIONS))
        for row in ([values[k][i] for k in ('temperature','pressure','relative_humidity','timestep')],[values[k][i] for k in PROPERTIES],np.r_[values['species_molar_mass'][i],values['water_molar_mass'][i]],np.r_[values['species_density'][i],values['water_density'][i]],*[values[k][i] for k in ('regime_limits','mass_accommodation','mass','number','gas')]):
            lines.append(' '.join(f'{v:.17e}' for v in row))
    result=subprocess.run([str(executable)],input='\n'.join(lines)+'\n',capture_output=True,text=True,check=True,timeout=timeout)
    data=np.fromstring(result.stdout,sep=' ');width=sum(OUTPUTS.values())
    if data.size!=6+n*2*width or not np.all(np.isfinite(data)):raise ValueError('Unexpected or nonfinite SALSA output')
    rows=data[6:].reshape(n,2,width);outputs={};offset=0
    for key,size in OUTPUTS.items():outputs[key]=rows[:,:,offset:offset+size] if all_levels else rows[np.arange(n),values['boundary_layer'].astype(int),offset:offset+size];offset+=size
    return outputs,dict(zip(('kernel_wall_seconds','kernel_cpu_seconds','mass_checksum','number_checksum','gas_checksum','clock_ticks_per_second'),data[:6].tolist()),repetitions=repetitions,steps=steps,cases=n,input_reset_included=True,grid_setup_included=True,native_cells_per_case=2,boundary_layer_by_native_level=[False,True],all_levels=bool(all_levels))
