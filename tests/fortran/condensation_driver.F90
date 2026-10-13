! Copyright 2026 Reflective
! SPDX-License-Identifier: Apache-2.0
PROGRAM condensation_regression
  USE mo_kind, ONLY: dp
  USE mo_ham, ONLY: subm_ngasspec
  USE mo_ham_salsactl, ONLY: fn2b
  USE mo_ham_subm_species, ONLY: isubm_ocnv
  USE extracted_condensation, ONLY: condensation
  IMPLICIT NONE
  REAL(dp) :: number(1,1,fn2b),before(1,1,fn2b),volumes(1,1,fn2b,5),diameter(1,1,fn2b)
  REAL(dp) :: temp(1,1),rh(1,1),pres(1,1),organic
  REAL(dp), ALLOCATABLE :: gas(:,:,:)
  INTEGER :: pbl(1),i
  CHARACTER(len=24) :: mode
  CALL get_command_argument(1,mode)
  IF (TRIM(mode)=='present' .OR. TRIM(mode)=='zero' .OR. &
      TRIM(mode)=='threshold' .OR. TRIM(mode)=='empty_organic') THEN
    subm_ngasspec=3
    isubm_ocnv=3
  END IF
  ALLOCATE(gas(1,1,subm_ngasspec))
  gas=0._dp
  gas(1,1,2)=1.e12_dp
  IF (TRIM(mode)=='present' .OR. TRIM(mode)=='empty_organic') THEN
    gas(1,1,2)=0._dp
    gas(1,1,3)=2.e12_dp
  END IF
  IF (TRIM(mode)=='threshold') gas(1,1,3)=1.e-10_dp
  number=1.e8_dp
  IF (TRIM(mode)=='empty' .OR. TRIM(mode)=='empty_organic') number=0._dp
  before=number
  volumes=0._dp
  DO i=1,fn2b
    diameter(1,1,i)=3.e-9_dp*(1.e-5_dp/3.e-9_dp)**(REAL(i-1,dp)/REAL(fn2b-1,dp))
  END DO
  temp=280._dp
  rh=0.5_dp
  pres=101325._dp
  pbl=1
  CALL condensation(1,1,1,1,number,volumes,diameter,gas,rh,temp,pres,1._dp,pbl)
  organic=0._dp
  IF (isubm_ocnv>0) organic=gas(1,1,isubm_ocnv)
  WRITE(*,'(5(ES26.17E3,1X))') gas(1,1,2),organic,SUM(volumes(:,:,:,1)), &
                             SUM(volumes(:,:,:,2)),MAXVAL(ABS(number-before))
END PROGRAM
