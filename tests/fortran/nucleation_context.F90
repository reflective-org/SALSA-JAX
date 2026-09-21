! Explicit fixture state for actual nucleation and condensation routines.
! Defaults below match mo_ham_salsactl; no scientific routines are stubbed.
MODULE mo_species
  USE mo_kind, ONLY: dp
  IMPLICIT NONE
  TYPE species_type
    REAL(dp) :: moleweight, density
  END TYPE
  TYPE(species_type) :: speclist(4) = [ &
    species_type(98._dp,1841._dp), species_type(150._dp,2000._dp), &
    species_type(58.44_dp,2165._dp), species_type(18._dp,1000._dp)]
END MODULE
MODULE mo_ham_species
  IMPLICIT NONE
  INTEGER, PARAMETER :: id_so4=1, id_oc=2, id_ss=3, id_wat=4
END MODULE
MODULE mo_ham_salsactl
  USE mo_kind, ONLY: dp
  IMPLICIT NONE
  INTEGER, PARAMETER :: in1a=1, in2a=4, fn2b=17, nbin(2)=[3,7]
  INTEGER :: nsnucl=0, nj3=1
  REAL(dp), PARAMETER :: d_sa=5.539376964394570e-10_dp, epsoc=0.15_dp, n3=158.79_dp
  REAL(dp) :: act_coeff=1.e-7_dp
  REAL(dp), PARAMETER :: reglim(4)=[3.e-9_dp,5.e-8_dp,7.e-7_dp,1.e-5_dp]
  REAL(dp) :: massacc(fn2b)=1._dp, dpmid(fn2b)=1.e-8_dp
END MODULE
MODULE mo_ham
  IMPLICIT NONE
  INTEGER :: subm_ngasspec=2, subm_naerospec_nowat=5, nsoa=0
END MODULE
MODULE mo_ham_subm_species
  IMPLICIT NONE
  INTEGER :: isubm_so4g=2, isubm_ocnv=0
END MODULE
MODULE mo_time_control
  USE mo_kind, ONLY: dp
  IMPLICIT NONE
  REAL(dp) :: time_step_len=1._dp
END MODULE
MODULE mo_exception
  IMPLICIT NONE
CONTAINS
  SUBROUTINE finish(where,message)
    CHARACTER(len=*), INTENT(IN) :: where,message
    WRITE(*,*) where, ': ', message
    ERROR STOP 'Unsupported nucleation configuration'
  END SUBROUTINE
END MODULE
