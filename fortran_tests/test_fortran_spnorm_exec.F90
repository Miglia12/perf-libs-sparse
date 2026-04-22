! SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
! affiliates <open-source-office@arm.com></text>
!
! SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

program test_fortran_spnorm_exec
    use perflibs_sparse
    implicit none

    ! Use compiler flags to switch between 4, 8 byte reals and preprocessor to select FTYPE=real/complex type
    FTYPE, target :: dummy
    integer, parameter :: rk = kind(dummy)
    class(*), pointer :: dummy_ptr

    integer, parameter :: m = 5
    integer, parameter :: n = 4
    integer, parameter :: nnz = 6
    FTYPE, dimension(nnz) :: vals;
    integer, dimension(nnz) :: col_indx = (/ 1, 2, 3, 4, 3, 4 /)
    integer, dimension(m + 1) :: row_ptr = (/ 1, 2, 3, 4, 5, 7 /)
    integer(kind=perflibs_i8) :: perflibs_mat
    real(kind=rk) :: result
    real(kind=rk) :: reference_result = -1
    real(kind=rk) :: op_tol_mod, eps, tol, diff

    integer :: stat
    integer :: flags = 0
    logical :: is_complex

    interface populate
        subroutine populate_real(arr)
            import rk
            implicit none
            real(kind=rk), dimension(:), intent(out) :: arr
        end subroutine populate_real

        subroutine populate_complex(arr)
            import rk
            implicit none
            complex(kind=rk), dimension(:), intent(out) :: arr
        end subroutine populate_complex
    end interface populate

    dummy_ptr => dummy
    select type(dummy_ptr)
    type is (real)
        is_complex = .false.
    type is (complex)
        is_complex = .true.
    class default
        stop 'Tests for this datatype not implemented'
    end select

    ! Populate vals array with real or complex values
    call populate(vals)

    ! Set Frobenius norm reference_result and op_tol_mod depending on the input type
    if (is_complex) then
        reference_result = sqrt(230.0_rk)
        op_tol_mod = 4.0_rk
    else
        reference_result = sqrt(91.0_rk)
        op_tol_mod = 1.0_rk
    end if

    ! Calculate the tolerance
    eps = epsilon(result)
    tol = op_tol_mod * nnz * reference_result * eps

    ! Create a CSR matrix
    call perflibs_spmat_create_csr(perflibs_mat, m, n, row_ptr, col_indx, vals, flags, stat)
    call check_stat('perflibs_spmat_create_csr')

    ! Call the spnorm_exec function for the Frobenius norm
    call perflibs_spnorm_exec(perflibs_mat, perflibs_sparse_norm_frb, result, stat)
    call check_stat('perflibs_spnorm_exec')
    ! Compare the result with the reference and report any errors
    call compare_results('Frobenius norm')

    ! Set the infinity norm reference result
    if (is_complex) then
        reference_result = sqrt(25.0_rk + 36.0_rk) + sqrt(36.0_rk + 49.0_rk)
    else
        reference_result = 11.0_rk
    end if

    ! Call the spnorm_exec function for the infinity norm
    call perflibs_spnorm_exec(perflibs_mat, perflibs_sparse_norm_inf, result, stat)
    call check_stat('perflibs_spnorm_exec')
    ! Compare the result with the reference and report any errors
    call compare_results('Infinity norm')

    ! Destroy perflibs_mat
    call perflibs_spmat_destroy(perflibs_mat, stat)
    call check_stat('perflibs_spmat_destroy')

    write (*, '(A,I1,A,I3)') "Success - test_fortran_spnorm_exec, kind = ", rk, " ftype nbits: ", storage_size(dummy)

contains

    subroutine compare_results(err_msg)
        character(len=*), intent(in) :: err_msg

        diff = abs(reference_result - result)
        if (diff >= tol) then
            stat = perflibs_status_execution_failure
        end if
        call check_stat(err_msg)
    end subroutine compare_results

    subroutine check_stat(err_msg)
        character(len=*), intent(in) :: err_msg

        if (stat .ne. perflibs_status_success) then
            write (*,*) err_msg, ' failed in test_fortran_spnorm_exec'
            stop 1
        end if
    end subroutine check_stat

end program test_fortran_spnorm_exec

subroutine populate_real(arr)

    implicit none

    FTYPE, target :: dummy
    integer, parameter :: rk = kind(dummy)
    real(kind=rk), dimension(:), intent(out) :: arr
    integer :: i

    do i = 1, size(arr, 1)
        arr(i) = real(i, kind=rk)
    end do

end subroutine populate_real

subroutine populate_complex(arr)

    implicit none

    FTYPE, target :: dummy
    integer, parameter :: rk = kind(dummy)
    complex(kind=rk), dimension(:), intent(out) :: arr
    integer :: i

    do i = 1, size(arr, 1)
        arr(i) = cmplx(i, i+1, kind=rk)
    end do

end subroutine populate_complex
