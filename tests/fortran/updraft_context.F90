! Copyright 2026 Reflective
! SPDX-License-Identifier: Apache-2.0
! Data context only: the test inserts unchanged production updraft routines.
MODULE extracted_updraft
  USE mo_kind, ONLY: dp
  IMPLICIT NONE
  INTEGER :: nw=20
  REAL(dp) :: w_min=0._dp
  TYPE vmem3d
    REAL(dp), POINTER :: ptr(:,:,:)
  END TYPE
  TYPE(vmem3d), ALLOCATABLE :: w(:), w_pdf(:)
CONTAINS
! ACTUAL_ROUTINES
END MODULE
