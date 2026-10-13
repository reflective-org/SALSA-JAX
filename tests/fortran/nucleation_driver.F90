! Copyright 2026 Reflective
! SPDX-License-Identifier: Apache-2.0
PROGRAM nucleation_regression
  USE mo_kind, ONLY: dp
  USE mo_ham, ONLY: subm_ngasspec
  USE mo_ham_salsactl, ONLY: fn2b,nsnucl,nj3
  USE mo_ham_subm_species, ONLY: isubm_ocnv
  USE mo_ham_salsa_nucleation, ONLY: nucleation
  USE extracted_condensation, ONLY: condensation
  IMPLICIT NONE
  REAL(dp) :: number(2,2,fn2b),before(2,2,fn2b),diameter(2,2,fn2b),volumes(2,2,fn2b,5)
  REAL(dp) :: temp(2,2),rh(2,2),pres(2,2),rate(2,2,2),sa(2,2),oc(2,2),sulfate
  REAL(dp), ALLOCATABLE :: gas(:,:,:),gas_before(:,:,:)
  INTEGER :: pbl(2),i
  CHARACTER(len=24) :: mode,arg,operation
  CALL get_command_argument(1,arg)
  READ(arg,*) nsnucl
  CALL get_command_argument(2,mode)
  CALL get_command_argument(3,arg)
  READ(arg,*) nj3
  CALL get_command_argument(4,operation)
  CALL get_command_argument(5,arg)
  READ(arg,*) sulfate
  IF (TRIM(mode)=='zero' .OR. TRIM(mode)=='present') THEN
    subm_ngasspec=3
    isubm_ocnv=3
  END IF
  ALLOCATE(gas(2,2,subm_ngasspec),gas_before(2,2,subm_ngasspec))
  gas=0._dp
  gas(:,:,2)=sulfate
  IF (TRIM(mode)=='present') gas(:,:,3)=2.e12_dp
  gas_before=gas
  number=1.e6_dp
  IF (TRIM(mode)=='empty') number=0._dp
  before=number
  volumes=0._dp
  DO i=1,fn2b
    diameter(:,:,i)=3.e-9_dp*(1.e-5_dp/3.e-9_dp)**(REAL(i-1,dp)/REAL(fn2b-1,dp))
  END DO
  temp=280._dp
  temp(2,:)=260._dp
  rh=0.5_dp
  pres=101325._dp
  pbl=[1,2]
  rate=0._dp
  IF (TRIM(operation)=='coupled') THEN
    CALL condensation(2,2,2,1,number,volumes,diameter,gas,rh,temp,pres,1._dp,pbl)
    WRITE(*,'(*(ES26.17E3,1X))') gas(:,:,2),SUM(volumes(:,:,:,1),DIM=3), &
      number(:,:,1)-before(:,:,1),MAXVAL(ABS(number(:,:,2:)-before(:,:,2:))),SUM(volumes(:,:,:,2))
  ELSE
    CALL nucleation(2,2,2,1,number,diameter,temp,rh,pres,1._dp,gas,rate,sa,oc,pbl)
    IF (ANY(gas/=gas_before)) ERROR STOP 'Nucleation modified input gases'
    IF (ANY(number/=before)) ERROR STOP 'Nucleation modified background particles'
    WRITE(*,'(*(ES26.17E3,1X))') rate,sa,oc
  END IF
END PROGRAM
