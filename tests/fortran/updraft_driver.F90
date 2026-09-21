PROGRAM updraft_regression
  USE mo_kind, ONLY: dp, wp
  USE mo_physical_constants, ONLY: grav
  USE mo_param_switches, ONLY: nactivpdf, ncd_activ
  USE extracted_updraft, ONLY: nw, activ_updraft
  USE, INTRINSIC :: ieee_arithmetic, ONLY: ieee_value, ieee_signaling_nan
  IMPLICIT NONE
  INTEGER, PARAMETER :: kproma=2, kbdim=2, klev=2, krow=1
! DRIVER_DECLARATION
  REAL(dp) :: zin(85,25,47,1), zrhoa(kbdim,klev)
  REAL(dp) :: ZTKEM1(kbdim,klev), ZWCAPE(kbdim)
  REAL(dp), ALLOCATABLE :: zw(:,:,:), zwpdf(:,:,:)
  REAL(dp) :: input_velocity
  CHARACTER(len=32) :: argument
  CALL get_command_argument(1, argument)
  READ(argument,*) input_velocity
  CALL get_command_argument(2, argument)
  READ(argument,*) nw
  nactivpdf=1
  ncd_activ=2
  zrhoa=RESHAPE([0.8_dp,1.1_dp,1.3_dp,1.5_dp],SHAPE(zrhoa))
  ZTKEM1=0.96_dp
  ZWCAPE=0._dp
  ALLOCATE(zw(kbdim,klev,nw),zwpdf(kbdim,klev,nw))
! DRIVER_ALLOCATION
  ! Explicit poisoning is essential: compiler initialization does not cover
  ! newly allocated storage. Old driver wiring fills only the final plane.
  zvervel=ieee_value(0._dp,ieee_signaling_nan)
  zin(85,25,47,1)=input_velocity
! DRIVER_ASSIGNMENT
! DRIVER_CALL
  WRITE(*,'(*(ES26.17E3,1X))') zw
  WRITE(*,'(*(ES26.17E3,1X))') zwpdf
END PROGRAM
