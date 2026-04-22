! SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
! affiliates <open-source-office@arm.com></text>
!
! SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

program test_fortran_sprot_exec
    use perflibs_sparse
    implicit none

    ! Use compiler flags to switch between 4, 8 byte reals and preprocessor to select FTYPE=real/complex type
    FTYPE, target :: dummy
    integer, parameter :: rk = kind(dummy)
    class(*), pointer :: dummy_ptr

    integer, parameter :: index_base = 1
    integer, parameter :: n = 10
    integer, parameter :: nnz = 6
    FTYPE, dimension(nnz) :: vals
    FTYPE, dimension(n) :: y, y_orig, y_ref
    FTYPE, dimension(n) :: x_orig, x_ref
    real(kind=rk) :: eps, op_tol_mod_add, op_tol_mod_mul
    real(kind=rk) :: tolsx, tolsy, diffx, diffy
    integer, dimension(nnz) :: indx = (/ 1, 2, 4, 7, 8, 9 /)
    FTYPE :: s, temp
    real(kind=rk) :: c
    FTYPE, dimension(nnz) :: vals_export
    integer, dimension(nnz) :: indx_export
    integer :: index_base_export, n_export, nnz_export

    integer :: stat, i, idx
    integer :: flags = 0
    integer(kind=perflibs_i8) :: perflibs_vec

    logical :: is_complex

    interface populate
        subroutine populate_real(arr1, arr2, sca1, sca2)
            import rk
            implicit none
            real(kind=rk), dimension(:), intent(out) :: arr1, arr2
            real(kind=rk), intent(out) :: sca1, sca2
        end subroutine populate_real

        subroutine populate_complex(arr1, arr2, sca1, sca2)
            import rk
            implicit none
            complex(kind=rk), dimension(:), intent(out) :: arr1, arr2
            real(kind=rk), intent(out) :: sca1
            complex(kind=rk), intent(out) :: sca2
        end subroutine populate_complex
    end interface populate

    interface perflibs_conj
        function perflibs_conj_real(a) result (b)
            import rk
            implicit none
            real(kind=rk), intent(in) :: a
            real(kind=rk) :: b
        end function perflibs_conj_real

        function perflibs_conj_cmplx(a) result (b)
            import rk
            implicit none
            complex(kind=rk), intent(in) :: a
            complex(kind=rk) :: b
        end function perflibs_conj_cmplx
    end interface perflibs_conj

    dummy_ptr => dummy
    select type(dummy_ptr)
    type is (real)
        is_complex = .false.
    type is (complex)
        is_complex = .true.
    class default
        stop 'Tests for this datatype not implemented'
    end select

    ! Populate arrays and set some scalars
    call populate(vals, y_orig, c, s)
    y = y_orig
    y_ref = y_orig

    ! Populate original dense vector with non-zero values
    x_orig = 0.0_rk
    do i = 1, nnz
        x_orig(indx(i)) = vals(i)
    end do
    x_ref = x_orig

    ! TEST 1: Run sprot exec (s,d,c,z)
    call perflibs_spvec_create(perflibs_vec, index_base, n, nnz, indx, vals, flags, stat)
    call check_stat(perflibs_status_success, 'Create perflibs_vec')

    call perflibs_sprot_exec(perflibs_vec, y, c, s, stat)
    call check_stat(perflibs_status_success, 'sprot_exec')

    call perflibs_spvec_export(perflibs_vec, index_base_export, n_export, nnz_export, indx_export, vals_export, stat)
    call check_stat(perflibs_status_success, 'spvec_export')

    ! Calculate reference solution using dense vectors
    do i = 1, n
        temp = c*x_ref(i) + s*y_ref(i)
        y_ref(i) = c*y_ref(i) - perflibs_conj(s)*x_ref(i)
        x_ref(i) = temp
    end do

    ! Compare results
    eps = epsilon(c)
    if (is_complex) then
        op_tol_mod_add = 2.0_rk
        op_tol_mod_mul = 4.0_rk
    else
        op_tol_mod_add = 1.0_rk
        op_tol_mod_mul = 1.0_rk
    end if
    call compare_results
    call check_stat(perflibs_status_success, 'TEST 1 result')

    ! Destroy perflibs_vec
    call perflibs_spvec_destroy(perflibs_vec, stat)
    call check_stat(perflibs_status_success, 'Destroy perflibs_vec')

    ! TEST 2: Run sprot exec (cs, zd)
    if (is_complex) then
        y = y_orig
        y_ref = y_orig
        x_ref = x_orig
        call perflibs_spvec_create(perflibs_vec, index_base, n, nnz, indx, vals, flags, stat)
        call check_stat(perflibs_status_success, 'Create perflibs_vec')

        call perflibs_sprot_exec(perflibs_vec, y, c, real(s,kind=rk), stat)
        call check_stat(perflibs_status_success, 'sprot_exec')

        call perflibs_spvec_export(perflibs_vec, index_base_export, n_export, nnz_export, indx_export, vals_export, stat)
        call check_stat(perflibs_status_success, 'spvec_export')

        ! Calculate reference solution using dense vectors
        do i = 1, n
            temp = c*x_ref(i) + real(s,kind=rk)*y_ref(i)
            y_ref(i) = c*y_ref(i) - real(s,kind=rk)*x_ref(i)
            x_ref(i) = temp
        end do

        ! Compare results
        op_tol_mod_add = 2.0_rk
        op_tol_mod_mul = 2.0_rk
        call compare_results
        call check_stat(perflibs_status_success, 'TEST 2 result')

        ! Destroy perflibs_vec
        call perflibs_spvec_destroy(perflibs_vec, stat)
        call check_stat(perflibs_status_success, 'Destroy perflibs_vec')
    end if

    write (*,'(A,I1,A,I3)') "Success - test_fortran_sprot_exec, kind = ", rk, " ftype nbits: ", storage_size(dummy)

contains

    subroutine check_stat(stat_to_check, test_text)
        integer(kind=perflibs_i4), intent(in) :: stat_to_check
        character(len=*), intent(in) :: test_text

        if (stat .ne. stat_to_check) then
            write (*,*) 'Failed ', test_text(1:len_trim(test_text)), ' in test_fortran_sprot_exec'
            stop 1
        end if

        return
    end subroutine check_stat

    subroutine compare_results
        do i = 1, nnz
            idx = indx(i)
            tolsx = (2 * op_tol_mod_mul + op_tol_mod_add) * max(abs(x_ref(idx)),1.0_rk) * eps
            tolsy = (2 * op_tol_mod_mul + op_tol_mod_add) * max(abs(y_ref(idx)),1.0_rk) * eps
            diffx = abs(vals_export(i)-x_ref(idx))
            diffy = abs(y(idx)-y_ref(idx))
            if (diffx > tolsx .or. diffy > tolsy) then
                stat = perflibs_status_execution_failure
                exit
            end if
        end do
    end subroutine compare_results

end program test_fortran_sprot_exec

subroutine populate_real(arr1, arr2, sca1, sca2)

    implicit none

    FTYPE, target :: dummy
    integer, parameter :: rk = kind(dummy)
    real(kind=rk), dimension(:), intent(out) :: arr1, arr2
    real(kind=rk), intent(out) :: sca1, sca2
    integer :: i

    do i = 1, size(arr1, 1)
        arr1(i) = real(i, kind=rk)
    end do
    do i = 1, size(arr2, 1)
        arr2(i) = real(2*i, kind=rk)
    end do
    sca1 = real(0.052335956242944, kind=rk)
    sca2 = real(sin(acos(sca1)), kind=rk)

end subroutine populate_real

subroutine populate_complex(arr1, arr2, sca1, sca2)

    implicit none

    FTYPE, target :: dummy
    integer, parameter :: rk = kind(dummy)
    complex(kind=rk), dimension(:), intent(out) :: arr1, arr2
    real(kind=rk), intent(out) :: sca1
    complex(kind=rk), intent(out) :: sca2
    integer :: i

    do i = 1, size(arr1, 1)
        arr1(i) = cmplx(i, i+1, kind=rk)
    end do
    do i = 1, size(arr2, 1)
        arr2(i) = cmplx(2*i, 2*i+1, kind=rk)
    end do
    sca1 = real(0.052335956242944, kind=rk)
    sca2 = cmplx(sin(acos(sca1)), sca1, kind=rk)

end subroutine populate_complex

function perflibs_conj_real(a) result (b)
    implicit none
    FTYPE, target :: dummy
    integer, parameter :: rk = kind(dummy)
    real(kind=rk), intent(in) :: a
    real(kind=rk) :: b

    b = a
end function perflibs_conj_real

function perflibs_conj_cmplx(a) result (b)
    implicit none
    FTYPE, target :: dummy
    integer, parameter :: rk = kind(dummy)
    complex(kind=rk), intent(in) :: a
    complex(kind=rk) :: b

    b = conjg(a)
end function perflibs_conj_cmplx
