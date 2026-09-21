!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!>
!! \filename 
!! mo_convert_concentrations.f90
!!
!! \brief
!! Contains subroutines and functions to convert initial concentrations to pxtm1 and then pxtm1 back
!! to concentrations and mixing ratios. In addition, the gas conversion to mixing ratio
!! is done in a different routine as it is calculated for different time steps.
!!         
!!
!! \author Eemeli Holopainen (FMI)
!!
!! \responsible_coder
!! Eemeli Holopainen, eemeli.holopainen@fmi.fi
!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODULE mo_convert_concentrations

CONTAINS

  SUBROUTINE conc2mmr(kproma,kbdim,klev, &
       ktrac,pxtm1,paerml,paernl,core,prhoa)

    USE mo_kind, ONLY : dp
    
    USE mo_species, ONLY: &
       speclist

    USE mo_ham_species, ONLY: &
       id_so4, id_oc, id_ss, id_bc, id_du

    USE mo_ham,          ONLY: nham_subm, naerocomp, aerocomp, subm_aerounitconv, subm_aero_idx, &
         sizeclass, mw_so4, nclass, immr2molec, HAM_M7, HAM_SALSA

    USE mo_ham_m7ctl,    ONLY: inucs, iaits, iaccs, icoas,          &
                               iaiti, iacci, icoai,                 &
                               iso4ns, iso4ks, iso4as, iso4cs,      &
                               ibcks, ibcas, ibccs, ibcki,          &
                               iocks, iocas, ioccs, iocki,          &
                               issas, isscs,                        &
                               iduas, iducs, iduai, iduci
    
    USE mo_ham_salsactl,  ONLY: in1a,in2a,in2b,fn1a,fn2a,fn2b, &
         iso4b, iocb, ibcb, issb, idub
    
    USE mo_physical_constants, ONLY: avo

    USE mo_exception,          ONLY: finish, message, message_text

    IMPLICIT NONE

    INTEGER, PARAMETER :: nmod = 7
    
    !-- input output variables --------
    INTEGER, INTENT(IN) :: &
         kproma, &
         kbdim,  &
         klev,   &    ! number of vertical levels
         ktrac

    REAL(dp), INTENT(IN) :: core(fn2b), &
         prhoa(kbdim,klev)  ! air mass density [kg m-3]
    
    REAL(dp), INTENT(INOUT) :: &
         paerml(kbdim,klev,naerocomp), &
         paernl(kbdim,klev,nclass), &
         pxtm1(kbdim,klev,ktrac) ! tracer mass/number mixing ratio
       
    
    !-- local variables --------

    INTEGER :: ii, kk
    INTEGER :: jt, jn ,jl, jk, jspec                    ! for indexing
         
    REAL(dp):: zfac,          &
             zfacm,           &
             zfacn,           &
             zfac1, &
             zqunitfac, &
             zmvsu
    
    
    zmvsu = (speclist(id_so4)%moleweight/1000.)/avo/speclist(id_so4)%density

    ! Set up mass concentrations (for compatibility with ECHAM6) depending on nham_subm
    SELECT CASE(nham_subm)
    CASE(HAM_M7)
       !--- Sulfate mass
       paerml(1:kproma,:,iso4ns) = 1.0*paernl(1:kproma,:,inucs)*core(inucs) /(1.e6_dp *zmvsu)
       paerml(1:kproma,:,iso4ks) = 0.5*paernl(1:kproma,:,iaits)*core(iaits) /(1.e6_dp *zmvsu)
       paerml(1:kproma,:,iso4as) = 0.5*paernl(1:kproma,:,iaccs)*core(iaccs) /(1.e6_dp *zmvsu)
       paerml(1:kproma,:,iso4cs) = 0.5*paernl(1:kproma,:,icoas)*core(icoas) /(1.e6_dp *zmvsu)
       !--- Organic carbon mass
       paerml(1:kproma,:,iocks) = 0.05*paernl(1:kproma,:,iaits)*core(iaits) * speclist(id_oc)%density / 1.e-9_dp
       paerml(1:kproma,:,iocas) = 0.05*paernl(1:kproma,:,iaccs)*core(iaccs) * speclist(id_oc)%density / 1.e-9_dp
       paerml(1:kproma,:,ioccs) = 0.05*paernl(1:kproma,:,icoas)*core(icoas) * speclist(id_oc)%density / 1.e-9_dp
       paerml(1:kproma,:,iocki) = 0.01*paernl(1:kproma,:,iaiti)*core(iaiti) * speclist(id_oc)%density / 1.e-9_dp
       !--- Black carbon mass
       paerml(1:kproma,:,ibcks) = 0.05*paernl(1:kproma,:,iaits)*core(iaits) * speclist(id_bc)%density / 1.e-9_dp
       paerml(1:kproma,:,ibcas) = 0.05*paernl(1:kproma,:,iaccs)*core(iaccs) * speclist(id_bc)%density / 1.e-9_dp
       paerml(1:kproma,:,ibccs) = 0.05*paernl(1:kproma,:,icoas)*core(icoas) * speclist(id_bc)%density / 1.e-9_dp
       paerml(1:kproma,:,ibcki) = 0.99*paernl(1:kproma,:,iaiti)*core(iaiti) * speclist(id_bc)%density / 1.e-9_dp
       !--- Sea salt mass
       paerml(1:kproma,:,issas) =  0.1_dp*paernl(1:kproma,:,iaccs)*core(iaccs) * speclist(id_ss)%density / 1.e-9_dp
       paerml(1:kproma,:,isscs) =  0.1_dp*paernl(1:kproma,:,icoas)*core(icoas) * speclist(id_ss)%density / 1.e-9_dp
       !--- Mineral dust mass
       paerml(1:kproma,:,iduas) =  0.3_dp*paernl(1:kproma,:,iaccs)*core(iaccs) * speclist(id_du)%density / 1.e-9_dp
       paerml(1:kproma,:,iducs) =  0.3_dp*paernl(1:kproma,:,icoas)*core(icoas) * speclist(id_du)%density / 1.e-9_dp
       paerml(1:kproma,:,iduai)  = 1.0_dp*paernl(1:kproma,:,iacci)*core(iacci) * speclist(id_du)%density / 1.e-9_dp
       paerml(1:kproma,:,iduci)  = 1.0_dp*paernl(1:kproma,:,icoai)*core(icoai) * speclist(id_du)%density / 1.e-9_dp
    CASE(HAM_SALSA)
       
       DO ii = in1a, fn1a
          !--- Sulfate mass
          paerml(1:kproma,:,iso4b(ii)) = 0.2_dp*paernl(1:kproma,:,ii)*core(ii) /(1.e6_dp *zmvsu)
       END DO
       
       DO ii = in1a, fn1a
          !--- Organic carbon
          paerml(1:kproma,:,iocb(ii))  = 0.8_dp*paernl(1:kproma,:,ii)*core(ii) * speclist(id_oc)%density / 1.e-9_dp
       END DO
       
       DO ii = in2a, fn2a     
          !--- Sulfate volume
          paerml(1:kproma,:,iso4b(ii)) = 0.5_dp*paernl(1:kproma,:,ii)*core(ii) / (1.e6_dp *zmvsu)
       END DO
       
       DO ii = in2a, fn2a     
          !--- Organic carbon
          paerml(1:kproma,:,iocb(ii)) =  0.05_dp*paernl(1:kproma,:,ii)*core(ii) * speclist(id_oc)%density / 1.e-9_dp 
       END DO
       
       DO ii = in2a, fn2a     
          !--- Black carbon
          paerml(1:kproma,:,ibcb(ii)) =  0.05_dp*paernl(1:kproma,:,ii)*core(ii) * speclist(id_bc)%density / 1.e-9_dp     
       END DO
       
       DO ii = in2a, fn2a     
          !--- Sea salt
          ! <-- thk: bugfix
          paerml(1:kproma,:,issb(ii)) =  0.1_dp*paernl(1:kproma,:,ii)*core(ii) * speclist(id_ss)%density / 1.e-9_dp
          ! --> thk
       END DO
       
       DO ii = in2a, fn2a     
          !--- Mineral dust
          paerml(1:kproma,:,idub(ii)) =  0.3_dp*paernl(1:kproma,:,ii)*core(ii) * speclist(id_du)%density / 1.e-9_dp     
       END DO
       
       DO ii = in2b, fn2b     
          !--- Sulfate volume
          paerml(1:kproma,:,iso4b(ii)) = 0.01_dp*paernl(1:kproma,:,ii)*core(ii) / (1.e6_dp *zmvsu)     
       END DO
       
       DO ii = in2b, fn2b     
          !--- Organic carbon
          paerml(1:kproma,:,iocb(ii))  = 0.01_dp*paernl(1:kproma,:,ii)*core(ii) * speclist(id_oc)%density / 1.e-9_dp
       END DO
       
       DO ii = in2b, fn2b   
          !--- Black carbon
          paerml(1:kproma,:,ibcb(ii))  = 0.01_dp*paernl(1:kproma,:,ii)*core(ii) * speclist(id_bc)%density / 1.e-9_dp
       END DO
       
       DO ii = in2b, fn2b
          !--- Mineral dust
          paerml(1:kproma,:,idub(ii))  = 0.97_dp*paernl(1:kproma,:,ii)*core(ii) * speclist(id_du)%density / 1.e-9_dp
       END DO
    END SELECT
    
    !--- Factor to transform mass SO4 in kg into molecules per kg:
    zfacm  = 6.022e+20_dp/mw_so4
    
    !--- Factor to transform kg into micro gram:
    
    zfac   = 1.e09_dp
    
    !--- Factor to transform N/m**3 into N/cm**3:
    
    zfacn  = 1.0e-06_dp
    
    !--- Calculate mixing ratios from paerml
    DO jn=1, naerocomp
       jt    = aerocomp(jn)%idt       ! get tracer id
       jspec = aerocomp(jn)%spid      ! get species id
       jl    = subm_aero_idx(jspec)     ! get index to subm_aerospec list
       !!mgs=old code!!     IF (aerocomp(jn)%species%m7unitconv == immr2molec) THEN
       IF (jl <= 0) THEN
          WRITE(message_text,*) 'SUBM_AERO_IDX Mapping error !! No index for jspec=',jspec
          CALL finish('ham_subm_interface', message_text)
       END IF
       IF (subm_aerounitconv(jl) == immr2molec) THEN
          zfac1 = zfacm
       ELSE
          zfac1 = zfac
       END IF
       
       pxtm1(1:kproma,:,jt) = paerml(1:kproma,:,jn)/ (zfac1*prhoa(1:kproma,:))  
       
    END DO
    
    !--- Calculate particle numbers from paernl
    
    DO jn=1, nclass
       jt = sizeclass(jn)%idt_no
       paernl(1:kproma,:,jn) = paernl(1:kproma,:,jn)*1.e-6_dp !eehol: convert 1/m3 to 1/cm3
       pxtm1(1:kproma,:,jt) = paernl(1:kproma,:,jn) / (zfacn*prhoa(1:kproma,:))
    END DO
    
  END SUBROUTINE conc2mmr


  SUBROUTINE gas2mmr(kproma,kbdim,klev,ktrac, &
       pxtm1,pgas,prhoa,pap,pt)

    USE mo_kind, ONLY : dp
    
    USE mo_species, ONLY: &
         speclist
    
    USE mo_ham,   ONLY: subm_ngasspec, &
         immr2ug, immr2molec, ivmr2molec, &
         subm_gasspec, subm_gasunitconv
    
    USE mo_physical_constants, ONLY: avo, argas
    
    !-- input output variables --------
    INTEGER, INTENT(in) :: &
         kproma, &
         kbdim,  &
         klev,   &    ! number of vertical levels
         ktrac
    
    REAL(dp), INTENT(IN) :: &
         pgas(kbdim,klev,subm_ngasspec), &
         prhoa(kbdim,klev), & ! air mass density [kg m-3]
         pap(kbdim,klev),pt(kbdim,klev)
    
    REAL(dp), INTENT(INOUT) ::  &
         pxtm1(kbdim,klev,ktrac) ! tracer mass/number mixing ratio
    
    
    !-- local variables --------
    INTEGER :: jn, jt, jk, jl  ! loop indices


    REAL(dp) :: &
         zunitfac, &
         zfac, &
         zfac_vmr

    !--- Factor to transform kg into micro gram:    
    zfac   = 1.e09_dp
  
    !--- Prefactor used when converting VMR into molec cm-3
    zfac_vmr = 1.E-6_dp*avo/argas
    
    !--- Convert gas phase concentrations to mixing ratios
    DO jn=1,subm_ngasspec
       jt = speclist(subm_gasspec(jn))%idt
       
       SELECT CASE(subm_gasunitconv(jn))
       CASE(immr2ug)
          DO jk=1,klev
             DO jl=1,kproma
                zunitfac = zfac*prhoa(jl,jk)
                pxtm1(jl,jk,jt) = pgas(jl,jk,jn)/zunitfac
             END DO
          END DO
          
       CASE(immr2molec)
          DO jk=1,klev
             DO jl=1,kproma
                zunitfac = 1e-3*prhoa(jl,jk)*avo/speclist(subm_gasspec(jn))%moleweight
                pxtm1(jl,jk,jt) = pgas(jl,jk,jn)/zunitfac
             END DO
          END DO
          
       CASE(ivmr2molec)
          DO jk=1,klev
             DO jl=1,kproma
                zunitfac = zfac_vmr*pap(jl,jk)/pt(jl,jk)
                pxtm1(jl,jk,jt) = pgas(jl,jk,jn)/zunitfac
             END DO
          END DO
       END SELECT
            
    END DO
    
  END SUBROUTINE gas2mmr

  SUBROUTINE mmr2conc(kproma,kbdim,klev, &
       ktrac,pxtm1,paerml,paernl,prhoa)

    USE mo_kind, ONLY : dp

    USE mo_ham_salsactl,  ONLY: fn2b
    
    USE mo_ham,          ONLY: naerocomp, aerocomp, subm_aerounitconv, subm_aero_idx, &
         sizeclass, mw_so4, nclass, immr2molec
    
    USE mo_exception,          ONLY: finish, message, message_text

    !-- input output variables --------

    INTEGER, INTENT(in) :: &
         kproma, &
         kbdim,  &
         klev,   &    ! number of vertical levels
         ktrac
    
    REAL(dp), INTENT(INOUT) :: &
         paerml(kbdim,klev,naerocomp), &
         paernl(kbdim,klev,nclass)

    REAL(dp), INTENT(IN) ::  &
         pxtm1(kbdim,klev,ktrac), & ! tracer mass/number mixing ratio
         prhoa(kbdim,klev)   ! air mass density [kg m-3]
    
    !-- local variables --------

    REAL(dp):: zfac,            &
             zfacm,           &
             zfacn,           &
             zfac1, &
             zqunitfac, &
             zmvsu

    INTEGER :: jt, jn ,jl, jk, jspec                    ! for indexing


    !--- Factor to transform mass SO4 in kg into molecules per kg:
    zfacm  = 6.022e+20_dp/mw_so4
    
    !--- Factor to transform kg into micro gram:
    
    zfac   = 1.e09_dp
    
    !--- Factor to transform N/m**3 into N/cm**3:
    
    zfacn  = 1.0e-06_dp
    
    !--- Calculate mixing ratios from pxtm1
    DO jn=1, naerocomp
       jt    = aerocomp(jn)%idt       ! get tracer id
       jspec = aerocomp(jn)%spid      ! get species id
       jl    = subm_aero_idx(jspec)     ! get index to subm_aerospec list
       !!mgs=old code!!     IF (aerocomp(jn)%species%m7unitconv == immr2molec) THEN
       IF (jl <= 0) THEN
          WRITE(message_text,*) 'SUBM_AERO_IDX Mapping error !! No index for jspec=',jspec
          CALL finish('ham_subm_interface', message_text)
       END IF
       IF (subm_aerounitconv(jl) == immr2molec) THEN
          zfac1 = zfacm
       ELSE
          zfac1 = zfac
       END IF
       
       paerml(1:kproma,:,jn) = pxtm1(1:kproma,:,jt)*(zfac1*prhoa(1:kproma,:))
       
    END DO
    
    !--- Calculate particle numbers from pxtm1
    
    DO jn=1, nclass
       jt = sizeclass(jn)%idt_no
       paernl(1:kproma,:,jn) = zfacn*prhoa(1:kproma,:)*pxtm1(1:kproma,:,jt)*1.e6_dp !eehol: convert 1/cm3 to 1/m3
    END DO

  END SUBROUTINE mmr2conc
  
END MODULE mo_convert_concentrations
