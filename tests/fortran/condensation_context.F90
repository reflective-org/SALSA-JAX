! Minimal explicit state for the unchanged condensation routine, with nucleation
! disabled. These fixture modules supply data only; no transport formula is stubbed.
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
  INTEGER :: nsnucl=0
  REAL(dp), PARAMETER :: d_sa=5.5393769643945706e-10_dp, epsoc=1._dp, n3=100._dp
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
MODULE mo_ham_salsa_nucleation
  USE mo_kind, ONLY: dp
  USE mo_ham_salsactl, ONLY: fn2b
  USE mo_ham, ONLY: subm_ngasspec
  IMPLICIT NONE
CONTAINS
  SUBROUTINE nucleation(kproma,kbdim,klev,krow,number,diameter,temp,rh,pres,dt,gas,rate,sa,oc,pbl)
    INTEGER :: kproma,kbdim,klev,krow,pbl(kbdim)
    REAL(dp) :: number(kbdim,klev,fn2b),diameter(kbdim,klev,fn2b)
    REAL(dp) :: temp(kbdim,klev),rh(kbdim,klev),pres(kbdim,klev),dt
    REAL(dp) :: gas(kbdim,klev,subm_ngasspec),rate(kbdim,klev,2),sa(kbdim,klev),oc(kbdim,klev)
    ERROR STOP 'Nucleation must remain disabled in this condensation regression'
  END SUBROUTINE
END MODULE
