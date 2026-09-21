PROGRAM driver

  USE driver_input
  USE mo_ham_salsa_init
  USE mo_ham,          ONLY:nham_subm, naerocomp, &
       subm_ngasspec, &
       HAM_M7, HAM_SALSA,nclass, & !+alaak
       nsol, aerocomp, sizeclass
       
  USE mo_ham_init
  USE mo_ham_salsa
  USE mo_ham_salsactl
  
  USE mo_ham_salsa_sizedist
  USE mo_ham_subm_species
  USE mo_kind
  USE mo_time_control
  USE mo_physical_constants, ONLY: vtmpc1, grav, rd, rv
  USE mo_math_constants,     ONLY: pi_6, pi
  USE mo_submodel
  USE mo_filename
  !USE mo_io
  USE mo_ham_salsa_cloud
  USE mo_tracdef,             ONLY: trlist,ntrac

  !<--eehol: use statements for microphysics interface
  USE mo_ham_subm, ONLY: ham_subm_interface
  !-->eehol

  !<--eehol: use statements for wet deposition
  USE mo_activ
  USE mo_param_switches, ONLY: ncd_activ, nactivpdf
  USE mo_hammoz_wetdep, ONLY: wetdep_interface
  USE mo_read_netcdf77, ONLY: read_var_nf77_4d
  !-->eehol
  !-->HK
  USE mo_ham_activ
  USE mo_hammoz_sedimentation, ONLY: sedi_interface
  !<--HK

  !<--eehol: for converting the concentrations and mixing ratios
  USE mo_convert_concentrations
  !-->eehol

  !-->alaak
  USE mo_ham_salsa_cloud, ONLY: salsa_abdul_razzak_ghan
  !<-- 

  !-->hhalonen
  USE parkind1, ONLY: JPIM, JPRB
  USE mo_hammoz_drydep, ONLY: drydep_interface
  USE mo_ham_rad, ONLY: ham_rad
  USE TM5M7_OPTICS_DATA, ONLY : NASWBAND
  USE mo_species, ONLY : speclist, nspec
  USE mo_ham_m7_trac, ONLY : idt_cdnc_ham, idt_icnc_ham
  !<--hhalonen

  IMPLICIT NONE

  
  INTEGER :: ii, jj, kk


  !**********************************************************
  !*                                                        *
  !* I) Tracers/subroutines to be coupled with host model   *
  !*                                                        *
  !*  NB: When coupling, check units carefully!!!!          *
  !*                                                        *
  !**********************************************************

  !-----------------------------------------------------------------
  !-- Tracers (provided by/to host model) --------------------------
  !-----------------------------------------------------------------

  !-- aerosol tracers ----------------- 
  INTEGER, PARAMETER :: nmod = 7 ! number of modes

  REAL(dp), ALLOCATABLE :: pxtm1(:,:,:), pxtte(:,:,:), pxtte_wetdepstrat(:,:,:), &
      pxtte_wetdepconv(:,:,:), pxtte_drydep(:,:,:), pxtte_sedi(:,:,:), pxtte_in(:,:,:)

  !-- atmospheric conditions --------------
  REAL(dp) :: &
       
       pap(kbdim,klev),  & ! atmospheric pressure at time t+1 
                                ! for each grid point (kbdim,klev) [Pa]
       
       pt(kbdim,klev),  &       ! atmospheric temperature at time t+1 
                                ! for each grid point (kbdim,klev) [K]
       zpbl(kbdim)              ! boundary layer height

  !<--eehol: variables for submodel interface
  REAL(dp) :: &
       paph(kbdim,klev+1), & !atm pressure at half levels
       paclc(kbdim,klev),  & !cloud cover
       pqm1(kbdim,klev),   & !specific humidity
       pqsm1(kbdim,klev),   & !saturation specific humidity
       pgrvolm1(kbdim,klev) !grid box volume
  REAL(dp), ALLOCATABLE :: zww(:,:,:) ! water content for each aerosol class
  !-->eehol

  !<--eehol: variables for calculating concentrations and mixing ratios
  REAL(dp), ALLOCATABLE :: zgas(:,:,:), &
       zaerml(:,:,:), &        ! mass mixing ratio of aerosol particles for each grid point
       zaernl(:,:,:)           ! number concentration of aerosol particles
                               ! for each grid point (kbdim,klev) and each
  REAL(dp) :: &
       
       ! size bin (fn2b) [#/m3]
       zrhoa(kbdim,klev),        &   ! air mass density [kg m-3]
       core(fn2b), &
       n(nmod),sigmag(nmod),dpg(nmod)
  !-->eehol
  
  !<--eehol: for the size distribution output
  ! --> thk: for better output
  CHARACTER (len=3), dimension(17) :: column_header = [ &
       '1a1', '1a2', '1a3', &
       '2a1', '2a2', '2a3', '2a4', '2a5', '2a6', '2a7', &
       '2b1', '2b2', '2b3', '2b4', '2b5', '2b6', '2b7'  &
       ]
  ! <-- thk
  CHARACTER (len=2), dimension(7) :: column_header_m7 = [ &
       'NS', 'KS', 'AS', &
       'CS', 'KI', 'AI', 'CI' &
  ]

  INTEGER :: i1, i2, i3

  !<--eehol: variables for wet deposition
  INTEGER :: ktop = 1                     ! top level index
  LOGICAL :: lstrat = .TRUE.              !SF this switch is used to keep track in wetdep_interface
                                          !   whether the call comes from the stratiform routine or
                                          !   the convective one - that's why it's hardcoded here
                                          !eehol: we need to discuss if we do strat and conv clouds separately...
  REAL(wp) :: pclcpre  (kbdim,klev)       ! fraction of grid box covered by precip
  REAL(wp) :: pfrain   (kbdim,klev)       ! rain flux before evaporation [kg/m2/s]
  REAL(wp) :: pfsnow   (kbdim,klev)       ! snow flux before evaporation [kg/m2/s]
  REAL(wp) :: pfevapr  (kbdim,klev)       ! evaporation of rain [kg/m2/s]
  REAL(wp) :: pfsubls  (kbdim,klev)       ! sublimation of snow [kg/m2/s]
  REAL(wp) :: pmsnowacl(kbdim,klev)       ! accretion rate of snow with cloud droplets in
  REAL(wp) :: pmlwc    (kbdim,klev)       ! cloud liquid content before rain [kg/kg]
  REAL(wp) :: pmiwc    (kbdim,klev)       ! cloud ice    content before rain [kg/kg]
  REAL(wp) :: pmratepr (kbdim,klev)       ! rain formation rate in cloudy part
  REAL(wp) :: pmrateps (kbdim,klev)       ! ice  formation rate in cloudy part

  !-->hhalonen
  ! For convective wetdep
  REAL(wp) :: pclcpre_conv   (kbdim,klev)  ! fraction of grid box covered by precip
  REAL(wp) :: pmratepr_conv  (kbdim,klev)  ! rain formation rate in cloudy part
  REAL(wp) :: pmrateps_conv  (kbdim,klev)  ! ice  formation rate in cloudy part
  REAL(wp) :: pfevapr_conv   (kbdim,klev)  ! evaporation of rain [kg/m2/s]
  REAL(wp) :: pfsubls_conv   (kbdim,klev)  ! sublimation of snow [kg/m2/s]
  REAL(wp) :: pfrain_conv    (kbdim,klev)  ! rain flux before evaporation [kg/m2/s]
  REAL(wp) :: pfsnow_conv    (kbdim,klev)  
  REAL(wp) :: zdum2d_conv    (kbdim,klev)
  REAL(wp) :: PMFU           (kbdim,klev)  
  !<--hhalonen

  !local variables for wet deposition
  REAL(wp), ALLOCATABLE :: zdum3d(:,:,:)
  REAL(wp), ALLOCATABLE :: zxtp1(:,:,:)  ! updated tracer mass/number mixing ratio
  REAL(wp), ALLOCATABLE :: zxtp10(:,:,:) ! ambient tracer mass/number mixing ratio
  REAL(wp), ALLOCATABLE :: zxtp1c(:,:,:) ! in-cloud tracer mass/number mixing ratio
  REAL(wp), ALLOCATABLE :: zdummy(:,:)   ! placeholder for pxtbound, which is only necessary in the conv. case
  REAL(wp), ALLOCATABLE :: za(:,:,:),    & ! curvature parameter A of the Koehler equation
               zb(:,:,:),    & ! hygroscopicity parameter B of the Koehler equation
               zrhop(:,:,:), &
               zrwet(:,:,:),  & ! wet radius for each class
               zrdry(:,:,:),  & ! dry radius for each class
               zm6dry(:,:,:)
  REAL(wp) :: zdum2d (kbdim,klev)
  REAL(wp) :: zlfrac_so2(kbdim,klev)     ! liquid tracer fraction (SO2) -ham specific-
  REAL(wp) :: zdpg(kbdim,klev)
  !-->eehol

  !<--eehol: variables for reading input
  REAL(wp), ALLOCATABLE :: zin(:,:,:,:)    !dummy for reading input
  CHARACTER(LEN=64) :: cfile
  !-->eehol
  
  !<--eehol: local variables for indexing and concentration/mixing ratio conversion
  REAL(dp) :: zqs !eehol: for saturation specific humidity calculations
  REAL(wp), PARAMETER :: rd1    = 287.04_wp        !> [J/K/kg] gas constant
  INTEGER ::  jk, jl, jt, it                  ! for indexing
  INTEGER :: ierr                           ! error integer

  !-->eehol
 
  !-->alaak
  !INTEGER, PARAMETER:: nw    = 1! from mo_activ
    REAL(dp) :: &
         pcdncact(kbdim,klev),&        ! number of activated particles         
         pesw(kbdim,klev)   !saturation water vapour pressure
  REAL(wp), ALLOCATABLE :: zw(:,:,:)! mean or bins of updraft velocity (>0.0) [m/s]
  REAL(wp), ALLOCATABLE :: zwpdf(:,:,:)!pdf of updraft velocity [s/m]
  REAL(wp), ALLOCATABLE :: znact(:,:,:) ! number of activated particles per mode [m-3]
  REAL(wp), ALLOCATABLE :: zfracn(:,:,:) ! fraction of activated particles per mode
  REAL(wp), ALLOCATABLE :: zsc(:,:,:) ! critical supersaturation [% 0-1]
  REAL(wp), ALLOCATABLE :: zrc(:,:,:,:) ! critical radius of activation per mode [m]
  REAL(wp), ALLOCATABLE :: zsmax(:,:,:)   ! maximum supersaturation
  !<--alaak

  !-->hhalonen
  REAL(wp), ALLOCATABLE :: znacttot(:,:) ! total number of activated particles in all modes [m-3]
  !<--hhalonen
  
  REAL(dp) :: mu, sigma, mu2, sigma2
  REAL(dp) :: sulfate_pdf_value, elvoc_pdf_value
  REAL(dp) :: H2SO4_scaling_factor

  INTEGER :: i, j

  !-->hhalonen
  REAL(dp) :: pelvoc(kproma,klev), psvoc(kproma,klev), &
       new_pelvoc(kproma,klev)    ! ELVOC ans SVOC concentrations + 
                                  ! a variable for updating the elvoc concentration

  ! Distance of table knots [K]
  REAL(dp), PARAMETER :: fdeltat  =   0.001_dp
  ! Division is not sufficiently precise, have to replace 1.0_dp/fdeltat
  REAL(dp), PARAMETER :: rfdeltat = 1000.0_dp 
  
  ! Temperature evaluation bounds:
  REAL(dp), PARAMETER :: tlbound =  50.0_dp  ! lower bound [K]
  REAL(dp), PARAMETER :: tubound = 400.0_dp  ! upper bound [K]
  
  ! Derived bounds and deltas, full table:
  INTEGER,  PARAMETER :: jptlucu1 = NINT(rfdeltat*tlbound) ! lookup table lower bound
  INTEGER,  PARAMETER :: jptlucu2 = NINT(rfdeltat*tubound) ! lookup table upper bound
  
  REAL(dp) :: tlucuaw(jptlucu1-1:jptlucu2+1)    ! table - Es*Rd/Rv, water phase only

  ! Reference: Sonntag D., 1990: Important new values of the physical 
  ! constants of 1986, vapour pressure formulations based on ITS-90, 
  ! and psychrometer formulae. Z. Meteor. 70, pp 340-344. 
  REAL(dp), PARAMETER :: cavl1 = -6096.9385_dp  
  REAL(dp), PARAMETER :: cavl2 =    21.2409642_dp
  REAL(dp), PARAMETER :: cavl3 =    -2.711193_dp
  REAL(dp), PARAMETER :: cavl4 =     1.673952_dp
  REAL(dp), PARAMETER :: cavl5 =     2.433502_dp
  
  ! KROW only used in ECHAM but needed inside HAM-codes so set as 1.
  INTEGER(KIND=JPIM), parameter::zkrow=1

  REAL(dp) :: zlinner, ztt
  REAL(KIND=JPRB)    :: reffi(kbdim, klev, zkrow), reffl(kbdim, klev, zkrow)
  REAL(KIND=JPRB) :: ZRE_LIQ(kbdim, klev)               ! liquid effective radius
  REAL(KIND=JPRB) :: ZQLWP(kbdim, klev)
  REAL(KIND=JPRB) :: ZTMPA
  REAL(KIND=JPRB) :: ZAP(kbdim, klev)
  REAL(KIND=JPRB) :: ZEPSEC=1e-14_JPRB
  REAL(KIND=JPRB) :: RCLDMAX=5.E-3_JPRB                 ! max cloud water
  REAL(KIND=JPRB) :: MP9_PH(kbdim, klev)
  REAL(KIND=JPRB) :: ZMIN_CDNC=1.0_JPRB                 ! minimum CDNC 
  LOGICAL         :: LLIQCLD(kbdim, klev)               ! logical for liquid cloud
  LOGICAL         :: LICECLD(kbdim, klev)               ! logical for ice cloud
  REAL(KIND=JPRB) :: MP9PH(kbdim, klev)
  REAL(KIND=JPRB) :: ZCFML(kbdim), ZCFMW(kbdim), &
      ZCFMI(kbdim), ZCFNCL(kbdim), ZCFNCW(kbdim), &
      ZCFNCI(kbdim), ZEPDU2, ZKAP
  REAL(KIND=JPRB) :: ZGEOM1(kbdim,klev),        &
      ZRIL(kbdim), ZRIW(kbdim), ZRII(kbdim),      &
      ZTVIR1(kbdim,klev), ZTVL(kbdim), ZTVW(kbdim)
  REAL(KIND=JPRB) :: ZTVI(kbdim), ZAZ0(kbdim),   &
      ZFRL(kbdim), ZSRFL(kbdim), ZFOREST(kbdim),  &
      ZTSI(kbdim), ZAZ0L(kbdim)
  REAL(KIND=JPRB) :: ZAZ0I(kbdim), ZCDNI(kbdim)
  LOGICAL :: ZLOLAND(kbdim)             ! Land mask
  REAL(KIND=JPRB) :: PLSM(kbdim) = 0.7  ! Land-sea mask [0-1]
  REAL(KIND=JPRB) :: ZAZ0W(kbdim), ZFRW(kbdim), &
      ZCVS(kbdim), ZCVW(kbdim), ZVGRAT(kbdim)      ! rough. len. wat., wat. frac., snow cov. frac., wet skin frac., veg. ratio
  REAL(KIND=JPRB) :: ZCDNL(kbdim), ZCDNW(kbdim)    ! ustar (in not used variable), aerodynamic resis. on surface (in not used variable)
  REAL(KIND=JPRB) :: PZ0M(kbdim) = 1.   ! added roughness length for momentum for dry deposition
  REAL(KIND=JPRB) :: PCI(kbdim) = 0.5   ! added fraction of sea-ice for dry deposition
  REAL(KIND=JPRB) :: PFRTI(kbdim) = 0.5 ! Tile fractions
  REAL(KIND=JPRB) :: zdp(kbdim,klev)
  REAL(KIND=JPRB) :: RG = 9.80665_JPRB             ! Gravity constant
  REAL(KIND=JPRB) :: PUP(kbdim,klev) = 0.0 ! added u component of wind
  REAL(KIND=JPRB) :: PVP(kbdim,klev) = 0.0 ! added u component of wind
  REAL(KIND=JPRB), ALLOCATABLE :: ZDDEPFLUX(:,:)
  REAL(KIND=JPRB), ALLOCATABLE :: ZXTMD1(:,:,:)      ! tracer mixing ratios for HAM drydep (updated with tend)
  REAL(KIND=JPRB), ALLOCATABLE :: ZVDEP(:,:)         ! ddep velocity for diagnostics from ham
  REAL(KIND=JPRB), ALLOCATABLE :: ZTENCIH(:,:,:)     ! for HAM tendencies
  REAL(KIND=JPRB), ALLOCATABLE :: ZXTEMS(:,:)        ! surface emissions modified by dry deposition
  REAL(KIND=JPRB) :: ZTKEM1(kbdim,klev)        ! turbulent kinetic energy
  REAL(KIND=JPRB) :: ZTUNPAR = 0.8164965_JPRB  ! tuning parameter for sigma_w derived from TKE (square root of 2/3 (isotropy assumption))
  REAL(KIND=JPRB) :: ZWCAPE(kbdim) = 0._JPRB   ! CAPE as zero as it is not used
  REAL(wp), ALLOCATABLE :: zvervel(:,:) ! pressure vertical velocity [Pa s-1]
  REAL(KIND=JPRB) :: ppd_hl(kbdim, klev)
  REAL(KIND=JPRB) :: ZAER_TAU(1,kbdim,14,1), &
      ZAER_SSA(1,kbdim,14),ZAER_ASYM(1,kbdim,14),ZAER_TAU_LW(1,kbdim,16)
  INTEGER, PARAMETER :: kb_diag = 14
  REAL(dp)        :: lambda_diag(kb_diag)
  REAL(dp)        :: zaer_tau_diag(kbdim,klev,kb_diag)
  REAL(dp)        :: zaer_ssa_diag(kbdim,klev,kb_diag)
  REAL(dp)        :: zaer_asym_diag(kbdim,klev,kb_diag)
  REAL(dp)        :: zfacn
  CHARACTER(LEN=64) :: cfile2
  INTEGER         :: kpband
  LOGICAL         :: ldiag_aeropt
  INTEGER         :: ntype_diaf
  INTEGER         :: jn

  ! For test plotting
  character(len=*), parameter :: datfile = "zaerml.dat"
  character(len=3000) :: cmd, plotline
  real, ALLOCATABLE :: zaernl_series(:,:)
  real, ALLOCATABLE :: zdrydepflux_series(:,:)
  real, ALLOCATABLE :: zsediflux_series(:,:)
  real, ALLOCATABLE :: zwetdepstratflux_series(:,:)
  real, ALLOCATABLE :: zwetdepconvflux_series(:,:)
  character(len=100) :: filename
  character(len=10)  :: istr
  integer :: lu, ifrac, inact, irdry, inum
  !<--hhalonen
 
!>>>>

  !  External subroutines 
  EXTERNAL :: inictl
  
  !  Executable statements

  !-->hhalonen
  ! Initialize variables for the normal distribution
  mu = 2000.0_dp       ! Mean of the sulfate distribution (seconds)
  sigma = 200.0_dp     ! Standard deviation of the sulfate distribution (seconds)
  mu2 = 2000.0_dp      ! Mean of the ELVOC distribution (seconds)
  sigma2 = 200.0_dp    ! Standard deviation of the ELVOC distribution (seconds)
  !<--hhalonen

  !-->hhalonen:
  ! ELVOC concentration: 1.6E7 cm^-3
  ! SVOC concentration:  2E8 cm^-3
  ! Molar mass:          300 g/mol
  ! Source: https://acp.copernicus.org/articles/18/12085/2018/acp-18-12085-2018.pdf
  do i = 1, kbdim
    do j = 1, klev
      pelvoc(i, j) = 1.6E7_dp
      psvoc(i, j) = 2E8_dp
    end do
  end do
  !<--hhalonen

  H2SO4_scaling_factor = 1.0E15_dp

  !<--eehol: initialize ncd_activ to be 2
  ncd_activ = 2
  nactivpdf = 1
  !-->eehol
 
  !-- 1. Set control variables

  !<--eehol: initialization for ham_subm_interface
  !CALL init_convect_tables !eehol: this is needed only for saturation specific humidity calculations.. Strongly related to ECHAM!! (sat. spec. hum. should come from host model!)
  !-->eehol
  
  ! read submodel name list and register submodels
  CALL setsubmodel
  CALL start_ham
  
  !<--eehol: define tracer numbers, idt, etc. with ham_define_tracer
  IF (lham) THEN
     CALL starttracdef(id_ham)
     CALL ham_define_tracer
     CALL endtracdef(id_ham)
  END IF
  !-->eehol

  !<--eehol: initialize HAM
  ! -- HAM aerosol module
  IF (lham) THEN
    CALL ham_initialize
  END IF
  !-->eehol
  
  !<--eehol: activ initialize
  CALL activ_initialize
  !--> HK:  CALL construct_activ_stream
  !-->eehol

  !--------------------------------------------------------------------------------
  !
  !  Calculate coagulation coefficients for particles:
  !  different set for each vertical (pressure) level
  !
  !  The values are calculated for bin mid-diameters and scaled each
  !  time step according to actual particle wet size 
  !
  !  NB: This must be done somewhere in the host model -
  !  but only for one time i1.e. before any aerosol calculations are started
  !

  ! call set_coagc(klev,pap,pt)

  !*************************************************
  !*                                               *
  !*  III) REAL STUFF BEGINS HERE !!!!             *
  !*                                               *
  !*************************************************

  !-->eehol: init cdnc and icnc index
  idt_cdnc = idt_cdnc_ham
  idt_icnc = idt_icnc_ham
  !<--eehol

  !--->hhalonen
  ALLOCATE(zrhop(kbdim,klev,nclass))
  ALLOCATE(zww(kbdim,klev,nclass))
  ALLOCATE(ZXTMD1(kbdim,klev,ntrac))
  ALLOCATE(ZVDEP(kbdim,ntrac))
  ALLOCATE(ZTENCIH(kbdim,klev,ntrac))
  ALLOCATE(ZXTEMS(kbdim,ntrac))
  ALLOCATE(ZDDEPFLUX(kbdim,ntrac))
  ALLOCATE(zaernl_series(1000, nclass))
  ALLOCATE(zdrydepflux_series(1000, ntrac))
  ALLOCATE(zsediflux_series(1000, ntrac))
  ALLOCATE(zwetdepstratflux_series(1000,ntrac))
  ALLOCATE(zwetdepconvflux_series(1000,ntrac))
  ALLOCATE(zrwet(kbdim,klev,nclass))    ! mean mode actual radius (wet for soluble and dry for insoluble modes) [cm]
  !<---hhalonen

  !<--eehol: Allocate tracer mixing ratio + tendency
  ALLOCATE(pxtm1(kbdim,klev,ntrac))
  ALLOCATE(pxtte(kbdim,klev,ntrac))
  ALLOCATE(pxtte_wetdepstrat(kbdim,klev,ntrac))
  ALLOCATE(pxtte_wetdepconv(kbdim,klev,ntrac))
  ALLOCATE(pxtte_drydep(kbdim,klev,ntrac))
  ALLOCATE(pxtte_sedi(kbdim,klev,ntrac))
  ALLOCATE(pxtte_in(kbdim,klev,ntrac))
  
  !-->eehol
  
  !<--eehol: initialize tracer mixing ratio and tendency
  pxtm1(:,:,:) = 0.0_dp
  pxtte(:,:,:) = 0.0_dp
  pxtte_wetdepstrat(:,:,:) = 0.0_dp
  pxtte_wetdepconv(:,:,:) = 0.0_dp
  pxtte_drydep(:,:,:) = 0.0_dp
  pxtte_sedi(:,:,:) = 0.0_dp
  pxtte_in(:,:,:) = 0.0_dp
  !-->eehol

  !-->hhalonen
  DO it = jptlucu1-1, jptlucu2+1
	ztt = fdeltat*REAL(it,dp)
	zlinner  = (cavl1/ztt+cavl2+cavl3*0.01_dp*ztt+cavl4*ztt*ztt*1.e-5_dp+cavl5*LOG(ztt))
	tlucuaw(it) = EXP(zlinner)*rd/rv
  END DO
  !<--hhalonen
  
  !<--eehol: define variables for ham_subm_interface
  zpbl = 1                               !boundary layer top level
  pap = 101325.                          !Ambient pressure (Pa)
  pt = 298._dp                           !Ambient temperature (K)
  pqm1(:,:) = 0.0058535_dp!0.01_dp       !specific humidity
  !eehol: calculate saturation specific humidity
  DO jk = 1,klev
     DO jl = 1,kproma
        it    = NINT(pt(jl,jk)*1000._dp)
        it    = MAX(MIN(it,jptlucu2),jptlucu1)
        zqs = tlucuaw(it)/pap(jl,jk)
        zqs = MIN(zqs,0.5_dp)
        zqs = zqs/(1._dp-vtmpc1*zqs)
        pqsm1(jl,jk) = zqs      !saturation specific humidity
     END DO
  END DO
  paclc(:,:) = 1._dp                     !cloud cover
  pgrvolm1(:,:) = 1.7964E12_dp           !grid box volume [m3] used in m7 diagn
  paph(:,:) = 0._dp                      !define half level pressure as zero
  paph(1:kproma,1) = pap(1:kproma,1)-100 !some value for 1st half level
  DO ii = 2,klev+1                       !calculate the half level pressure values from mean
     paph(1:kproma,ii) = 2*pap(1:kproma,ii-1)-paph(1:kproma,ii-1)
  END DO
  time_step_len = 1.0_dp
  !-->eehol

  !<--eehol: initialization for wet deposition
  zlfrac_so2(1:kproma,:) = 0._wp                   ! liquid fraction of SO2 for HAM
  zdpg(:,:) = 0._wp

  !-- (re-) compute geopotential height
  DO jk=1,klev
     DO jl=1,kproma
        zdpg(jl,jk)=(paph(jl,jk+1)-paph(jl,jk))/grav
     END DO
  END DO
  !-->eehol

  !<--eehol: Calculate air density 
  DO jk = 1,klev
     DO  jl = 1,kproma       
        !--- 2.1) Calculate air density:
        !         (currently neglects volume occupied by liquid and ice water  = > physc)
        zrhoa(jl,jk) = pap(jl,jk)/(pt(jl,jk)*rd1*(1._dp+vtmpc1*pqm1(jl,jk)))       
     ENDDO
  ENDDO
  !-->eehol

  !<--eehol: allocate and initialize variables
  !allocate array for gases 
  ALLOCATE(zgas(kbdim,klev,subm_ngasspec))
  !allocate mass mixing ratio
  ALLOCATE(zaerml(kbdim,klev,naerocomp))

  !allocate number concentration
  ALLOCATE(zaernl(kbdim,klev,nclass))

  WRITE(*,*) 'eehol: subm_ngasspec =', subm_ngasspec
  
  zgas = 0._dp    ! gases

  !-->eehol

  !<--eehol: allocate variables for wet deposition
  ALLOCATE(zdum3d(kbdim,klev,ntrac))
  ALLOCATE(zxtp1(kbdim,klev,ntrac))
  ALLOCATE(zxtp10(kbdim,klev,ntrac))
  ALLOCATE(zxtp1c(kbdim,klev,ntrac))
  ALLOCATE(zdummy(kbdim,ntrac))
  !allocate activ variables (why doesnt this work from construct stream?)
  ALLOCATE(za(kbdim,klev,nclass)) ! curvature parameter A of the Koehler equation
  ALLOCATE(zb(kbdim,klev,nclass)) ! hygroscopicity parameter B of the Koehler equation
  ALLOCATE(zrdry(kbdim,klev,nclass))    ! dry radius for each classe
  ALLOCATE(zm6dry(kbdim,klev,nsol))     ! dry radius for soluble modes
  !<---eehol

  !-->alaak needed for abdul razzak ghan:
  ALLOCATE(zw(kbdim,klev,nw))! mean or bins of updraft velocity (>0.0) [m/s]
  ALLOCATE(zwpdf(kbdim,klev,nw))!pdf of updraft velocity [s/m]
  ALLOCATE(znact(kbdim,klev,nclass)) ! number of activated particles per mode [m-3]
  ALLOCATE(zfracn(kbdim,klev,nclass)) ! fraction of activated particles per mode
  ALLOCATE(zsc(kbdim,klev,nclass)) ! critical supersaturation [% 0-1]
  ALLOCATE(zrc(kbdim,klev,nclass,nw)) ! critical radius of activation per mode [m]
  ALLOCATE(zsmax(kbdim,klev,nw))   ! maximum supersaturation
  !<--alaak

  !-->hhalonen
  ALLOCATE(zvervel(kbdim,klev))
  ALLOCATE(znacttot(kbdim,klev)) ! total number of activated particles in all modes [m-3]
  !<--hhalonen

  !<--eehol: read input variables for cloud activation and wet deposition
  ALLOCATE (zin(192,96,47,1))
  cfile = 'input/HAM_box_inp_200007.01_activ.nc'
  CALL read_var_nf77_4d (cfile, "lon", "lat", "lev", "time", "CLC_PRE", zin, ierr)
  pclcpre(1:kproma,:) = zin(85,25,47,1)
  CALL read_var_nf77_4d (cfile, "lon", "lat", "lev", "time", "F_RAIN", zin, ierr)
  pfrain(1:kproma,:) = zin(85,25,47,1)
  CALL read_var_nf77_4d (cfile, "lon", "lat", "lev", "time", "F_SNOW", zin, ierr)
  pfsnow(1:kproma,:) = zin(85,25,47,1)
  CALL read_var_nf77_4d (cfile, "lon", "lat", "lev", "time", "F_EVAPR", zin, ierr)
  pfevapr(1:kproma,:) = zin(85,25,47,1)
  CALL read_var_nf77_4d (cfile, "lon", "lat", "lev", "time", "F_SUBLS", zin, ierr)
  pfsubls(1:kproma,:) = zin(85,25,47,1)
  CALL read_var_nf77_4d (cfile, "lon", "lat", "lev", "time", "M_SNOW_ACL", zin, ierr)
  pmsnowacl(1:kproma,:) = zin(85,25,47,1)
  CALL read_var_nf77_4d (cfile, "lon", "lat", "lev", "time", "M_LWC", zin, ierr)
  pmlwc(1:kproma,:) = zin(85,25,47,1)
  CALL read_var_nf77_4d (cfile, "lon", "lat", "lev", "time", "M_IWC", zin, ierr)
  pmiwc(1:kproma,:) = zin(85,25,47,1)
  CALL read_var_nf77_4d (cfile, "lon", "lat", "lev", "time", "M_RATE_PR", zin, ierr)
  pmratepr(1:kproma,:) = zin(85,25,47,1)
  CALL read_var_nf77_4d (cfile, "lon", "lat", "lev", "time", "M_RATE_PS", zin, ierr)
  pmrateps(1:kproma,:) = zin(85,25,47,1)
  CALL read_var_nf77_4d (cfile, "lon", "lat", "lev", "time", "ESW", zin, ierr)
  pesw(1:kproma,:) = zin(85,25,47,1)
  CALL read_var_nf77_4d (cfile, "lon", "lat", "lev", "time", "ZETW", zin, ierr)
  !zw(1:kproma,:,nw) = zin(85,25,47,1)
  CALL read_var_nf77_4d (cfile, "lon", "lat", "lev", "time", "ZETWPDF", zin, ierr)
  zwpdf(1:kproma,:,nw) = zin(85,25,47,1)

  !-->hhalonen
  ! Reading total vertical velocity of activation from the input
  cfile2 = 'input/GlobalTraj-CE_200801.01_activ_1step.nc'
  CALL read_var_nf77_4d (cfile2, "lon", "lat", "lev", "time", "W", zin, ierr)
  ! Input W is [m s-1]; activ_updraft expects pressure velocity [Pa s-1].
  zvervel(1:kproma,:) = -grav*zrhoa(1:kproma,:)*zin(85,25,47,1)
  !<--hhalonen

  !<--eehol: testing wet deposition
  ! pclcpre(1:kproma,:) = 0.5_wp
  ! pfrain(1:kproma,:) = 0.4_wp
  ! pfsnow(1:kproma,:) = 0.5_wp
  ! pfevapr(1:kproma,:) = 0.001_wp
  ! pfsubls(1:kproma,:) = 0.0004_wp
  ! pmsnowacl(1:kproma,:) = 0._wp
  ! pmlwc(1:kproma,:) = 0.03_wp
  ! pmiwc(1:kproma,:) = 0.01_wp
  ! pmratepr(1:kproma,:) = 0.4_wp
  ! pmrateps(1:kproma,:) = 0.1_wp
  WRITE(*,*) 'eehol: frac grid box covered by precip =', pclcpre(1:kproma,:)
  WRITE(*,*) 'eehol: rain flux before evaporation =', pfrain(1:kproma,:)    
  WRITE(*,*) 'eehol: snow flux before evaporation =', pfsnow(1:kproma,:)
  WRITE(*,*) 'eehol: evaporation of rain =', pfevapr(1:kproma,:)     
  WRITE(*,*) 'eehol: sublimation of snow =', pfsubls(1:kproma,:)
  WRITE(*,*) 'eehol: accretion rate of snow with cloud droplets in =', pmsnowacl(1:kproma,:) 
  WRITE(*,*) 'eehol: cloud liquid content before rain =', pmlwc(1:kproma,:)    
  WRITE(*,*) 'eehol: cloud ice content before rain =', pmiwc(1:kproma,:)   
  WRITE(*,*) 'eehol: rain formation rate in cloudy part =', pmratepr(1:kproma,:)     
  WRITE(*,*) 'eehol: ice formation rate in cloudy part =', pmrateps(1:kproma,:)
  WRITE(*,*) 'eehol: saturation water vapour pressure =', pesw(1:kproma,:)
  WRITE(*,*) 'eehol: mean or bins of updraft velocity =', zw(1:kproma,:,nw)     
  WRITE(*,*) 'eehol: pdf of updraft velocity =', zwpdf(1:kproma,:,nw)
  WRITE(*,*) 'nham_subm =', nham_subm
  !-->eehol

  za(:,:,:) = 0
  zb(:,:,:) = 0
  zsc(:,:,:) = 0

  !N=nucleation mode, K=Aitken, A=Accumulation, C=Coarse
  !(NS, KS, AS, CS, KI, AI, CS)
  ! Stdev of modes
  sigmag = (/ 1.59, 1.59, 1.59, 2.0, 1.59, 1.59, 2.0 /)
  ! Mean diameter of the modes (m)
  dpg = (/0.01, 0.3, 1.0, 3.0, 0.03, 0.3, 3.0/)*1.e-6_dp
  ! Number concentration of modes (#/cm3)
  n = (/1000.0, 100.0, 10.0, 0.001, 100.0, 10.0, 0.001/)*1.e6_dp
 
  !<--eehol: calculate initial size distribution (zaernl) depending on nham_subm
  SELECT CASE(nham_subm)
  CASE(HAM_M7)
     core(1:nclass) = pi_6 * dpg(1:nclass)**3
     DO kk = 1,nclass
        DO jj = 1,klev
           DO ii = 1,kproma
              zaernl(ii,jj,kk) = n(kk)
           END DO
        END DO
     END DO

  CASE(HAM_SALSA)
     !<--eehol: calculating the initial size distribution (zaernl)
     core(in1a:fn2b) = pi_6 * dpmid(in1a:fn2b)**3
     CALL size_distribution(kproma, kbdim,  klev,   &
          n, dpg, sigmag, zaernl)

  END SELECT
  !-->eehol

  !<--eehol: converting initial concentrations to mixing ratios
  CALL conc2mmr(kproma,kbdim,klev,ntrac, &
       pxtm1,zaerml,zaernl,core,zrhoa)
  !-->eehol

  !-->hhalonen
  ! Liquid and ice effective radius for wet deposition.

   ! Threshold for cloud cover
   DO JK=1,klev
      DO JL=1, kproma
         ZAP(JL,JK)=MIN(1.0_JPRB,MAX(0.0_JPRB,paclc(JL,JK)))
      ENDDO
   ENDDO

   ! LWP
   DO JK=1,klev
      DO JL=1,kproma
         IF (ZAP(JL,JK) >=0.001_JPRB) THEN
            ZTMPA = 1.0_JPRB/ZAP(JL,JK)
            LLIQCLD(JL,JK) = (pmlwc(JL,JK)*ZTMPA) > ZEPSEC ! logical for liquid cloud
            LICECLD(JL,JK) = (pmiwc(JL,JK)*ZTMPA) > ZEPSEC ! logical for ice cloud
            ZQLWP(JL,JK) = MIN(MAX(0._JPRB, pmlwc(JL,JK)*ZTMPA), RCLDMAX) ! lwp
         ELSE
            LLIQCLD(JL,JK) = .FALSE.
            LICECLD(JL,JK) = .FALSE.
            ZQLWP(JL,JK) = 0.0_JPRB
         END IF
      END DO
   END DO
   !<--hhalonen

   !-->hhalonen: Variables not needed for aerosol dry deposition
   ZCFML(:) = 0._JPRB
   ZCFMW(:) = 0._JPRB
   ZCFMI(:) = 0._JPRB
   ZCFNCL(:) = 0._JPRB
   ZCFNCW(:) = 0._JPRB
   ZCFNCI(:) = 0._JPRB
   ZEPDU2 = 0._JPRB
   ZKAP = 0._JPRB
   ZGEOM1(:,:) = 0._JPRB
   ZRIL(:) = 0._JPRB
   ZRIW(:) = 0._JPRB
   ZRII(:) = 0._JPRB
   ZTVIR1(:,:) = 0._JPRB
   ZTVL(:) = 0._JPRB
   ZTVW(:) = 0._JPRB
   ZTVI(:) = 0._JPRB
   ZAZ0(:) = 0._JPRB
   ZFRL(:) = 0._JPRB
   ZSRFL(:) = 0._JPRB
   ZFOREST(:) = 0._JPRB
   ZTSI(:) = 0._JPRB
   ZAZ0L(:) = 0._JPRB
   ZAZ0I(:) = 0._JPRB
   ZCDNI(:) = 0._JPRB
   ZDDEPFLUX(1:kbdim,:) = 0._JPRB
   !<--hhalonen

   !-->hhalonen: Variables for ham_rad
   kpband     = 16
   ntype_diaf = 1
   ppd_hl(1:kproma, :) = paph(1:kproma, 2:klev+1) - paph(1:kproma, 1:klev) ! Pressure diff between half levels [Pa]
   ZAER_TAU(1:kproma,:,:,:)     = 1.0_JPRB  ! Aerosol optical thickness
   ZAER_SSA(1:kproma,:,:)       = 1.0_JPRB  ! Aerosol single scattering albedo
   ZAER_ASYM(1:kproma,:,:)      = 1.0_JPRB  ! Aerosol asymmetry factor
   ZAER_TAU_LW(1:kproma,:,:)    = 1.0_JPRB  ! LW optical thickness of aerosols
   ldiag_aeropt                 = .TRUE.    ! Logical for aerosol optics
   zaer_tau_diag(1:kproma,:,:)  = 0.0_JPRB  ! Can be 0 in input
   zaer_ssa_diag(1:kproma,:,:)  = 0.0_JPRB  ! Can be 0 in input
   zaer_asym_diag(1:kproma,:,:) = 0.0_JPRB  ! Can be 0 in input
   lambda_diag(:)               = 550.E-9_JPRB
   !<--hhalonen
   !--- Factor to transform N/m**3 into N/cm**3:
   zfacn  = 1.0e-06_dp

   !-----------------------------------------------------------------------------------
  ! open and initialize some output files
  inum = find_next_free_unit(20,99)
  OPEN(inum, FILE='data/num.dat', STATUS='unknown')
  CALL write_header(inum)

  irdry = find_next_free_unit(20,99)
  OPEN(irdry, FILE='data/rdry.dat', STATUS='unknown')
  CALL write_header(irdry)

  inact = find_next_free_unit(20,99)
  OPEN(inact, FILE='data/nact.dat', STATUS='unknown')
  CALL write_header(inact)
  
  ifrac = find_next_free_unit(20,99)
  OPEN(ifrac, FILE='data/fracn.dat', STATUS='unknown')
  CALL write_header(ifrac)

  ! writing out initial size distribution
  CALL write_row(irdry, dpmid(in1a:fn2b)/2._dp)  ! all zero at this time
  CALL write_row(inum, zaernl(1,1,in1a:fn2b))
  CALL write_row(ifrac, zfracn(1,1,in1a:fn2b))
  CALL write_row(inact, znact(1,1,in1a:fn2b))
  write(6,*) dpmid(in1a:fn2b)/2._dp
   ! Time loop
   DO ii = 1, 1000

      !-->hhalonen:
      ! Normal distribution pdf value for sulfate at time ii
      sulfate_pdf_value = (1.0_dp / (sigma * SQRT(2.0_dp * pi))) * &
                  EXP(-((REAL(ii, dp) - mu)**2) / (2.0_dp * sigma**2))

      ! Normal distribution pdf value for ELVOC at time ii
      elvoc_pdf_value = (1.0_dp / (sigma2 * SQRT(2.0_dp * pi))) * &
                  EXP(-((REAL(ii, dp) - mu2)**2) / (2.0_dp * sigma2**2))
      
      ! Sulfate concentration from the distribution
      zgas(:,:,isubm_so4g) = sulfate_pdf_value * H2SO4_scaling_factor

      ! ELVOC concentration from the distribution
      new_pelvoc = elvoc_pdf_value * pelvoc
      !<--hhalonen

      ! Gas phase concentrations converted from m-3 to cm-3 for compatibility with M7
      zgas(1:kproma,:,:) = zgas(1:kproma,:,:) * 1.e-6_dp
      
      ! Convert gas concentration to mixing ratio
      CALL gas2mmr(kproma, kbdim, klev, ntrac, &
            pxtm1, zgas, zrhoa, pap, pt)
            !-->eehol
      
      !CALL set_nsnucl_nonucl(1, 3)
      
      !<--eehol: call microphysics interface
      CALL ham_subm_interface(kproma,  kbdim,   klev,    krow,&  ! ECHAM indices
          ntrac, pap, paph,                                   &  ! number of tracers, pressure full levels, pressure half levels
          pt,    pqm1, pqsm1,                                 &  ! temperature, specific humidity, saturation specific humidity
          pxtm1, pxtte,                                       &  ! tracer mass/number mr, tendencies
          zrwet, zm6dry, zrhop, zww,                          &  ! mean mode actual radius [m], dry radius for soluble modes [m] 
          paclc, pgrvolm1, zpbl)                                 ! cloud cover, grid box volume, boundary layer top level
      DO jn=1, nclass
         jt = sizeclass(jn)%idt_no
         zaernl(1:kproma,:,jn) = zfacn*zrhoa(1:kproma,:)*(pxtm1(1:kproma,:,jt)+pxtte(1:kproma,:,jt)*time_step_len)
         !zout3(kidia:kproma,1:klev,jn) = zaernl(1:kproma,1:klev,jn)
         !     write(3334,*)jt,'NUM',sizeclass(jn)%shortname,jn
         
      END DO
      !-->alaak call cloud activation

      !-->hhalonen:
      DO kk = 1,nclass
         IF (sizeclass(kk)%LSOLUBLE) THEN 
            zrdry(1:kproma,1:klev,kk) = zm6dry(1:kproma,1:klev,kk) !soluble modes rdry from rdry_m7
         ELSE
            zrdry(1:kproma,1:klev,kk) = zrwet(1:kproma,1:klev,kk)  !insoluble modes rdry from rwet_m7
         END IF
      END DO
      !<--hhalonen

      !-->hhalonen:
      ZTKEM1(1:kproma,1:klev) = ((1/ZTUNPAR)**2)*((0.8_JPRB)**2)
      CALL activ_updraft(kproma, kbdim, klev, krow, &
         ZTKEM1, ZWCAPE, zvervel, zrhoa, zw, zwpdf)
      
      CALL ham_rad(kproma, kbdim, klev, krow, kpband, NASWBAND, &
         pxtm1, ppd_hl, &
         ZAER_TAU(:,:,:,1), ZAER_SSA, ZAER_ASYM, ZAER_TAU_LW, zrwet, &
         & ldiag_aeropt, kb_diag, ntype_diaf, &
         & lambda_diag, zaer_tau_diag, zaer_ssa_diag, zaer_asym_diag)
      !<--hhalonen

      SELECT CASE(nham_subm)
         
      CASE(HAM_M7)

         !CALL radii(kproma, kbdim, klev, krow, zrdry)

         !-->hhalonen
         ! Getting parameters A and B of the Koehler eq.
         CALL ham_activ_koehler_ab(kproma, kbdim, klev, krow, ktdia, &
                                  pxtm1, pt, za, zb)
         !<--hhalonen

         CALL ham_activ_abdulrazzak_ghan(kproma, kbdim, klev, krow, ktdia, &
               pcdncact, pesw, zrhoa,             &
               pxtm1, pt, pap, pqm1,         &
               zw, zwpdf, za, zb, zrdry,         &
               znact, zfracn, zsc, zrc, zsmax, znacttot)

      CASE(HAM_SALSA)
         !>> thk #511: AR&G scheme for SALSA
         
         ! for now we decided to not use diagnostics routines
         ! in order to cut down on output

         !-->hhalonen
         ! Getting parameters A and B of the Koehler eq.
         CALL ham_activ_koehler_ab(kproma, kbdim, klev, krow, ktdia, &
                                  pxtm1, pt, za, zb)
         !<--hhalonen
         !CALL radii(kproma, kbdim, klev, krow, zrdry)

         CALL salsa_abdul_razzak_ghan(&
               kproma,   kbdim, klev,  krow, ktop, &
               pcdncact, pesw,  zrhoa,               &
               pxtm1,    pt,  pap, pqm1,        &
               zw,       zwpdf,                     &
               znact,    zfracn,zsc,   zrc, &
               zsmax  )
         ! ECHAM indices
         ! n of act p (o), saturation water vapour press  ! writing out initial size distribution
         CALL write_row(irdry, zrdry(1,1,in1a:fn2b))  ! all zero at this time
         CALL write_row(inum, zaernl(1,1,in1a:fn2b))
         CALL write_row(ifrac, zfracn(1,1,in1a:fn2b))
         CALL write_row(inact, znact(1,1,in1a:fn2b))
         ! tracer mixing ratios at t-d (i1), temperature(i1),pressure(i1), specific humidity(i1)
         ! mean or bins of updraft velocity (i1), pdf of updraft velocity (i1)
         ! number of activated p per mode (o), fraction of act. p (o), critical supersat. (o) critical r of act per mode(o)
         ! maximum supersaturation (o)
         
         ! pesw calculated by sat_spec_hum module in ECHAM (uses lookuptables)
         ! zw and zwpdf calculated by activ_updraft module  

         !<< thk #511
         
      END SELECT
      !<--alaak: call cloud activation

      !-->eehol: add cdnc to tracers
      pxtm1(1:kproma,:,idt_cdnc) = (MAX(pcdncact(1:kproma,:),((1.0E6_dp)*ZMIN_CDNC)))/zrhoa(1:kproma,:)
      !<--eehol

      !-->hhalonen
      ! Convert from #/m3 to #/cm3 and threshold minimum value to 1 cm-3
      MP9PH(1:kproma, 1:klev) = MAX((1.0E-6_JPRB)*pcdncact(1:kproma, 1:klev),ZMIN_CDNC)

      DO JK=1,klev
         DO JL=1,kproma
            ! Effective radius (in um). 
            ! 2.387e-10 is 3/(4*pi*rho_liq*10^6)  [10^6 for N in right units]
            ZRE_LIQ(JL,JK) = 1.E+06_JPRB*(2.387e-10_JPRB* &
               zrhoa(JL,JK)*ZQLWP(JL,JK)/MP9PH(JL,JK))**0.333_JPRB
         END DO
      END DO

      ! Add liq. eff. rad. to HAM variables (only if there is liquid cloud 
      ! else minimum value)
      reffl(1:kproma,1:klev,zkrow) = MERGE(ZRE_LIQ(1:kproma,1:klev), &
            4._JPRB, LLIQCLD(1:kproma,1:klev))    ! [um]
      ! only if there is ice cloud else minimum value
      reffi(1:kproma,1:klev,zkrow) = MERGE(reffi(1:kproma,1:klev,zkrow), &
            20._JPRB, LICECLD(1:kproma,1:klev))   ! [um]
      !<--hhalonen

      !-->eehol: initialize mixing ratios for wet deposition
      !-- initialise in-cloud and interstitial mixing ratios
      !   set both equal to tracer mixing ratio as starting point
      !   ham_wet_chemistry will re-compute these values if lham=true
      DO jt = 1,ntrac
         zxtp1(1:kproma,:,jt)  = pxtm1(1:kproma,:,jt) + &
               pxtte(1:kproma,:,jt) * time_step_len
         zxtp1c(1:kproma,:,jt) = zxtp1(1:kproma,:,jt)
         zxtp10(1:kproma,:,jt) = zxtp1(1:kproma,:,jt)
      END DO
      !<--eehol
         
      !<--eehol: call wetdep interface for wet deposition
      !-- interface to wet deposition routine (also from cuflx_subm)
      IF ( lwetdep .AND. ANY(trlist%ti(:)%nwetdep > 0) ) THEN

         zdummy(1:kproma,:) = 0._dp !eehol: initialize dummy variables (is this necessary?)
         zdum2d(1:kproma,:) = 0._dp !eehol: initialize dummy variables (is this necessary?)
         zdum3d(1:kproma,:,:) = 0._dp !eehol: initialize dummy variables (is this necessary?)

         !pmratepr = 0.001_dp
         !pmrateps = 1.0e-8_dp
         !pfevapr = 1.0e-5_dp
         !pfsubls = 0.0001_dp
         !pmlwc = 1.0_dp
         !pmiwc = 1.0_dp
         lstrat = .TRUE.

         pxtte_in = pxtte
         
         CALL wetdep_interface(kproma, kbdim, klev, ktop, krow, lstrat, &
             zdpg,   pmratepr, pmrateps,   pmsnowacl,         &
             pmlwc,  pmiwc,                                   &
             zrwet,  zrdry,                                   &
             reffi,  reffl,                                   &
             znact, zfracn,                                   &
             pt, pxtm1, zlfrac_so2, pxtte, zxtp10, zxtp1c,    &
             pfrain, pfsnow, pfevapr, pfsubls,                &
             zdum2d, zdum3d,                                  &
             paclc,  pclcpre, zrhoa, zdummy, znacttot)

         pxtte_wetdepstrat(1:kproma,:,:) = pxtte_in(1:kproma,:,:) - pxtte(1:kproma,:,:)

         !-->hhalonen
         !Convective case for wet deposition
         lstrat = .FALSE.
         pclcpre_conv(1:kproma,:) = 0.0_dp
         pmratepr_conv(1:kproma,:) = 0.002_dp
         pmrateps_conv(1:kproma,:) = 0.0002_dp
         pfevapr_conv(1:kproma,:) = 8.0E-8_dp
         pfsubls_conv(1:kproma,:) = 4.0E-5_dp
         zdum2d_conv(1:kproma,:) = 0.1_dp
         pfrain_conv(1:kproma,:) = 0.0004_dp
         pfsnow_conv(1:kproma,:) = 0.0001_dp

         ! initialize mixing ratios for wet deposition
         ! -- initialise in-cloud and interstitial mixing ratios
         ! set both equal to tracer mixing ratio as starting point
         ! ham_wet_chemistry will re-compute these values if lham=true
         DO jt = 1,ntrac
            zxtp1(1:kproma,:,jt)  = pxtm1(1:kproma,:,jt) + &
                  pxtte(1:kproma,:,jt) * time_step_len
            zxtp1c(1:kproma,:,jt) = zxtp1(1:kproma,:,jt)
            zxtp10(1:kproma,:,jt) = zxtp1(1:kproma,:,jt)
         END DO

         pxtte_in = pxtte

         CALL wetdep_interface(kproma, kbdim, klev, ktop, krow, lstrat, &
             zdpg,   pmratepr_conv, pmrateps_conv,   pmsnowacl,         &
             pmlwc,  pmiwc,                                   &
             zrwet,  zrdry,                                   &
             reffi,  reffl,                                   &
             znact, zfracn,                                   &
             pt, pxtm1, zlfrac_so2, pxtte, zxtp10, zxtp1c,    &
             pfrain_conv, pfsnow_conv, pfevapr_conv, pfsubls_conv,                &
             zdum2d_conv, zdum3d,                                  &
             paclc,  pclcpre_conv, zrhoa, zdummy, znacttot)

         pxtte_wetdepconv(1:kproma,:,:) = pxtte_in(1:kproma,:,:) - pxtte(1:kproma,:,:)
         !<--hhalonen
         
      END IF

      !-->hhalonen
      ! Dry deposition
      !--> variables for dry deposition
      DO JL = 1,kproma
         DO JK = 1,klev
         ZDP(JL,JK) = paph(JL,JK+1) - paph(JL,JK)
         END DO
         IF ( PLSM(JL) < 0.99_JPRB ) THEN
         ZLOLAND(JL) = .FALSE.
         ZAZ0W(JL) = PZ0M(JL)
         ELSE
         ZLOLAND(JL) = .TRUE.
         ZAZ0W(JL) = 0._JPRB
         END IF
         ZAZ0W(JL)  = MAX(1.0E-5_JPRB,ZAZ0W(JL))  ! threshold roughness length to min value
         ZFRW(JL)   = MAX(0.,1.-PLSM(JL)-PCI(JL)) ! water fraction = 1 - land mask - sea ice fraction
         ! Random values for now
         ZCVS(JL) = 0.5
         ZCVW(JL) = 0.5
         ZVGRAT(JL) = 0.5
         ZCDNL(JL) = 0.5
         ZCDNW(JL) = 0
         !ZCVS(JL)   = PFRTI(JL,5)+PFRTI(JL,7)    ! snow cover fraction = Snow on low-veg + snow on bare-soil + snow under high-veg
         !ZCVW(JL)   = PFRTI(JL,3)                 ! wet skin fraction
         !ZVGRAT(JL) = PCVL(JL)+PCVH(JL)           ! vegetation ratio = low veg. cover + high veg. cover
         !ZCDNL(JL)  = PAERUST(JL)                 ! adding ustar to not used variable
         !ZCDNW(JL) = LOG(ZDZ(JL,KLEV)/PZ0M(JL))/(VKARMAN*PAERUST(JL)) ! calculate aerodyn. resistance on surface to not used variable
      END DO

      !--> init values
      ZTENCIH(1:kproma,1:klev,:) = pxtte(1:kproma,1:klev,:) ! init tendency before drydep
      ZXTEMS(1:kproma,:)         = 0._JPRB                  ! surface emissions as zero for input
      ZXTMD1(1:kproma,1:klev,:)  = 0._JPRB
      ZXTMD1(1:kproma,1:klev,:)  = pxtm1(1:kproma,1:klev,:) + (ZTENCIH(1:kproma,1:klev,:) * time_step_len) ! update mixrat with tendency
      ZVDEP(1:kproma,:)          = 0._JPRB                  ! ddep velocity as zero

      pxtte_in = pxtte

      ! RCHG: Recommendation, those subroutines specific of m7 should have 
      !       m7 in the name not sure if this is specific or general/common 
      !       but adapted to m7. Like m7_simple_sulfur_drydep below. 
      CALL drydep_interface(kbdim, kproma,  klev, zkrow,                    &
         & pqm1(:,klev), pqsm1(:,klev), pt(:,klev), ZCFML, ZCFMW, ZCFMI, &
         & ZCFNCL, ZCFNCW, ZCFNCI,                                       &
         & ZEPDU2, ZKAP, PUP, PVP, ZGEOM1, ZRIL, ZRIW,                   &
         & ZRII,                                                         &
         & ZTVIR1, ZTVL, ZTVW, ZTVI, ZAZ0,                               &
         & pt(:,klev), ZLOLAND,                                         &
         & zrwet, ZRHOP,                                                 & ! M7
         & ZFRL,   ZFRW,  PCI,     ZCVS,   ZCVW,     ZVGRAT,             &
         & ZSRFL,  PUP(:,klev),     PVP(:,klev),                         & !eehol: FIXME 10m u and v wind from lowest level.. needs to be revised in future!!
         & ZXTEMS, ZXTMD1, zrhoa(:,klev), paph, ZFOREST, ZTSI,            & !air dens lowest, air press at int.
         & ZAZ0L, ZAZ0W, ZAZ0I, ZCDNL, ZCDNW, ZCDNI, ZDDEPFLUX, ZVDEP)     !ZCDNL and ZCDNW used for ustar and aerodyn. resist.
      
      !--> modify tendency at surface according to changes in surface emissions
      DO JT = 1,ntrac
         DO JL = 1,kproma
         pxtte(JL,klev,JT) = ZTENCIH(JL,klev,JT) + ((ZXTEMS(JL,JT)*RG)/(zdp(JL,klev)))
         END DO
      END DO

      pxtte_drydep(1:kproma,:,:) = pxtte_in(1:kproma,:,:) - pxtte(1:kproma,:,:)

      pxtte_in = pxtte
      !<--hhalonen
      
      !-->eehol
      IF (lsedimentation .AND. ANY(trlist%ti(:)%nsedi > 0)) THEN
         
          CALL sedi_interface(kbdim, kproma, klev, krow,   &
             pt,    pqm1,     pap,  paph, zrwet, zrhop, &
             pxtm1, pxtte)
      pxtte_sedi(1:kproma,:,:) = pxtte_in(1:kproma,:,:) - pxtte(1:kproma,:,:)

      END IF
      !<--eehol: updating pxtm1 according to pxtte and time step and nullify pxtte
      pxtm1(1:kproma,:,:) = pxtm1(1:kproma,:,:)+(pxtte(1:kproma,:,:)*time_step_len)
      pxtte(1:kproma,:,:) = 0._dp
      !-->eehol

      !<--eehol: calculate number concentration and mixing ratios from pxtm1
      CALL mmr2conc(kproma,kbdim,klev,ntrac, &
            pxtm1,zaerml,zaernl,zrhoa)
      !-->eehol

      !-->hhalonen: For test plotting
      do i = 1, ntrac
         zwetdepstratflux_series(ii,i) = pxtte_wetdepstrat(1,1,i)
      end do
      do i = 1, ntrac
         zwetdepconvflux_series(ii,i) = pxtte_wetdepconv(1,1,i)
      end do
      do i = 1, ntrac
         zdrydepflux_series(ii, i) = pxtte_drydep(1,1,i)
      end do
      do i = 1, ntrac
         zsediflux_series(ii, i) = pxtte_sedi(1,1,i)
      end do
      do i = 1, nclass
         zaernl_series(ii, i) = zaernl(1,1,i)
      end do
      !<--hhalonen

      !<--eehol: write number concentration to output data file
      SELECT CASE(nham_subm)
      CASE(HAM_M7)
         WRITE(15,665) zaernl(1,1,1:nclass)
         ! Dry mean darius
         WRITE(16,665) zrdry(1,1,1:nclass)
      CASE(HAM_SALSA)
         WRITE(15,665) zaernl(1,1,in1a:fn2b)
      END SELECT
      !-->eehol

   END DO
   !-->eehol

   !-->hhalonen
   ! Test plotting
   open(unit=13, file='zpxtte_wetdepstrat.dat', status='unknown')
   do ii = 1, 1000
      !write(13,*) (zwetdepflux_series(ii,i), i = 1, ntrac)
      write(13,*) (zwetdepstratflux_series(ii,i), i = 22, 28)
   end do
   close(13)

   open(unit=13, file='zpxtte_wetdepconv.dat', status='unknown')
   do ii = 1, 1000
      !write(13,*) (zwetdepflux_series(ii,i), i = 1, ntrac)
      write(13,*) (zwetdepconvflux_series(ii,i), i = 22, 28)
   end do
   close(13)

   open(unit=11, file='zpxtte_drydep.dat', status='unknown')
   do ii = 1, 1000
      !write(11,*) (zdrydepflux_series(ii, i), i = 1, ntrac)
      write(11,*) (zdrydepflux_series(ii, i), i = 22, 28)
   end do
   close(11)

   open(unit=12, file='zpxtte_sedi.dat', status='unknown')
   do ii = 1, 1000
      !write(12,*) (zsediflux_series(ii, i), i = 1, ntrac)
      write(12,*) (zsediflux_series(ii, i), i = 22, 28)
   end do
   close(12)

   open(unit=10, file='zaernl_series.dat', status='unknown')
   do ii = 1, 1000
      write(10,*) (zaernl_series(ii, i), i = 1, nclass)
   end do
   close(10)
   !<--hhalonen

  !-----------------------------------------------------------------------------------

  !<--eehol: for writing the size distribution from ham_subm_interface
  ! <-- mirfan: fixing output issues for very small numbers  
665 FORMAT(99(E14.4E5,1X))
  ! --> mirfan
   !-->eehol
 CONTAINS
   
   SUBROUTINE write_header(filenum)
     ! Write a comma-separated column header for easier 
     ! readability into the given output file
     ! hard-coded for 17-bin SALSA
     INTEGER, INTENT(IN) :: filenum
     CHARACTER (len=3), DIMENSION(17) :: column_header = [ &
          '1a1', '1a2', '1a3', &
          '2a1', '2a2', '2a3', '2a4', '2a5', '2a6', '2a7', &
          '2b1', '2b2', '2b3', '2b4', '2b5', '2b6', '2b7'  &
    ]
     
     WRITE(filenum,'(16(A3,", "),A3)') column_header
   END SUBROUTINE write_header
   
   
   SUBROUTINE write_row(filenum, row)
     ! Write the next row of data comma-separated into the
    ! given output file
     ! hard-coded for 17-bin SALSA
     INTEGER,  INTENT(IN) :: filenum
     REAL(dp), INTENT(IN), DIMENSION(17) :: row
     CHARACTER(LEN=32) :: fmt = "(16(E14.4E5,','),E14.4E5)"
     
     WRITE(filenum,fmt) row
   END SUBROUTINE write_row
   
END PROGRAM driver
