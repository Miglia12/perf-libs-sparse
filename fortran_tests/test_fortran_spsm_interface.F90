! SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
! affiliates <open-source-office@arm.com></text>
!
! SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

program test_fortran_spsm_interface
    use perflibs_sparse
    use iso_c_binding
    implicit none

    ! Use compiler flags to switch between 4, 8 byte reals and preprocessor to select FTYPE=real/complex type
    FTYPE :: dummy
    integer, parameter :: rk = kind(dummy)

    integer, parameter :: n = 4
    integer, parameter :: nrhs = 2
    integer, parameter :: n_rect = 3
    integer :: m_out, n_out
    integer :: stat
    integer :: flags = 0
    integer(kind=perflibs_i4) :: fifty_five = 55
    real(kind=rk) :: tol
    FTYPE :: alpha
    integer :: libgfortran_mismatch

    integer(kind=perflibs_i8) :: mat_a, mat_x, mat_y
    integer(kind=perflibs_i8) :: mat_rect_a, mat_xc, mat_yc
    integer(kind=perflibs_i8) :: mat_xr, mat_yr
    integer(kind=perflibs_i8) :: mat_xa, mat_ya
    integer(kind=perflibs_i8) :: mat_null_a
    integer(kind=perflibs_i8) :: mat_zero_diag_a

    FTYPE, dimension(n*n) :: a_vals
    FTYPE, dimension(n*nrhs) :: y_vals
    FTYPE, dimension(n*nrhs) :: x_init
    FTYPE, dimension(n*nrhs) :: x_expected
    FTYPE, dimension(n*1) :: x_col1_vals
    FTYPE, dimension(n*nrhs) :: y_col2_vals
    FTYPE, dimension((n-1)*nrhs) :: x_row3_vals, y_row3_vals
    FTYPE, dimension(n*n) :: a_zero_diag_vals
    FTYPE, allocatable :: x_out(:)
    type(c_ptr) :: c_vals
    FTYPE, pointer :: p_vals(:)

    external :: query_mismatch
    external :: perflibs_spmat_export_dense_wrapper_s
    external :: perflibs_spmat_export_dense_wrapper_d
    external :: perflibs_spmat_export_dense_wrapper_c
    external :: perflibs_spmat_export_dense_wrapper_z
    external :: perflibs_spmat_export_deallocate_wrapper

    include 'type_query.h'

    alpha = 1.0_rk
    tol = 100.0_rk * epsilon(1.0_rk)

    ! Lower-triangular A (column-major):
    ! [1 0 0 0
    !  2 1 0 0
    !  0 3 1 0
    !  0 0 4 1]
    a_vals = (/ &
        1.0_rk, 2.0_rk, 0.0_rk, 0.0_rk, &
        0.0_rk, 1.0_rk, 3.0_rk, 0.0_rk, &
        0.0_rk, 0.0_rk, 1.0_rk, 4.0_rk, &
        0.0_rk, 0.0_rk, 0.0_rk, 1.0_rk &
    /)

    ! Y = A * X_expected, with X_expected columns [1,2,3,4] and [4,3,2,1]
    y_vals = (/ &
        1.0_rk, 4.0_rk, 9.0_rk, 16.0_rk, &
        4.0_rk, 11.0_rk, 11.0_rk, 9.0_rk &
    /)

    x_expected = (/ &
        1.0_rk, 2.0_rk, 3.0_rk, 4.0_rk, &
        4.0_rk, 3.0_rk, 2.0_rk, 1.0_rk &
    /)
    x_init = 0.0_rk

    ! Setup baseline matrices for positive test
    call perflibs_spmat_create_dense(mat_a, perflibs_col_major, n, n, n, a_vals, flags, stat)
    call check_stat(perflibs_status_success, stat, 'Setup A')
    call perflibs_spmat_create_dense(mat_x, perflibs_col_major, n, nrhs, n, x_init, flags, stat)
    call check_stat(perflibs_status_success, stat, 'Setup X')
    call perflibs_spmat_create_dense(mat_y, perflibs_col_major, n, nrhs, n, y_vals, flags, stat)
    call check_stat(perflibs_status_success, stat, 'Setup Y')

    ! Test 1: Positive path: optimize + exec + value check
    call perflibs_spsm_optimize(perflibs_sparse_operation_notrans, mat_a, mat_x, &
                                perflibs_sparse_scalar_any, mat_y, stat)
    call check_stat(perflibs_status_success, stat, 'Test 1.0 optimize')

    call perflibs_spsm_exec(perflibs_sparse_operation_notrans, mat_a, mat_x, alpha, mat_y, stat)
    call check_stat(perflibs_status_success, stat, 'Test 1.1 exec')

    call query_mismatch(libgfortran_mismatch)
    if (libgfortran_mismatch .eq. 1) then
        if (typeid .eq. 1) then
            call perflibs_spmat_export_dense_wrapper_s(mat_x, perflibs_col_major, m_out, n_out, c_vals, stat)
        else if (typeid .eq. 2) then
            call perflibs_spmat_export_dense_wrapper_d(mat_x, perflibs_col_major, m_out, n_out, c_vals, stat)
        else if (typeid .eq. 3) then
            call perflibs_spmat_export_dense_wrapper_c(mat_x, perflibs_col_major, m_out, n_out, c_vals, stat)
        else if (typeid .eq. 4) then
            call perflibs_spmat_export_dense_wrapper_z(mat_x, perflibs_col_major, m_out, n_out, c_vals, stat)
        end if
        call check_stat(perflibs_status_success, stat, 'Test 1.2w export')

        call c_f_pointer(c_vals, p_vals, [m_out*n_out])
        allocate(x_out(m_out*n_out))
        x_out(:) = p_vals(:)
        call perflibs_spmat_export_deallocate_wrapper(c_vals)
    else
        call perflibs_spmat_export_dense(mat_x, perflibs_col_major, m_out, n_out, x_out, stat)
        call check_stat(perflibs_status_success, stat, 'Test 1.2 export')
    end if

    if (m_out .ne. n .or. n_out .ne. nrhs) then
        write (*,*) 'Test 1.3 failed in test_fortran_spsm_interface'
        stop 1
    end if
    call check_close_vec(x_out, x_expected, tol, 'Test 1.4 values')
    deallocate(x_out)

    ! Test 2: invalid trans in exec
    ! Param under test: TRANSA in perflibs_spsm_exec.
    ! All matrix dimensions/properties are valid.
    call perflibs_spsm_exec(fifty_five, mat_a, mat_x, alpha, mat_y, stat)
    call check_stat(perflibs_status_input_parameter_error, stat, 'Test 2 invalid trans')

    ! Test 3: A not square
    ! Param under test: A dimensions in perflibs_spsm_exec (A must be square).
    ! Use triangular-valid/non-zero-diagonal values so only non-square dims are invalid.
    call perflibs_spmat_create_dense(mat_rect_a, perflibs_col_major, n, n_rect, n, a_vals, flags, stat)
    call check_stat(perflibs_status_success, stat, 'Setup rect A')
    call perflibs_spsm_exec(perflibs_sparse_operation_notrans, mat_rect_a, mat_x, alpha, mat_y, stat)
    call check_stat(perflibs_status_input_parameter_error, stat, 'Test 3 A not square')
    call destroy_mat(mat_rect_a, 'Cleanup rect A')

    ! Test 4: X and Y have different number of columns
    ! Param under test: ncols(X) == ncols(Y).
    ! A is valid square triangular and row dimensions match.
    x_col1_vals = 0.0_rk
    y_col2_vals = 1.0_rk
    call perflibs_spmat_create_dense(mat_xc, perflibs_col_major, n, 1, n, x_col1_vals, flags, stat)
    call check_stat(perflibs_status_success, stat, 'Setup X col mismatch')
    call perflibs_spmat_create_dense(mat_yc, perflibs_col_major, n, nrhs, n, y_col2_vals, flags, stat)
    call check_stat(perflibs_status_success, stat, 'Setup Y col mismatch')
    call perflibs_spsm_exec(perflibs_sparse_operation_notrans, mat_a, mat_xc, alpha, mat_yc, stat)
    call check_stat(perflibs_status_input_parameter_error, stat, 'Test 4 column mismatch')
    call destroy_mat(mat_xc, 'Cleanup X col mismatch')
    call destroy_mat(mat_yc, 'Cleanup Y col mismatch')

    ! Test 5: X and Y have different number of rows
    ! Param under test: nrows(X) == nrows(Y).
    ! A is valid square triangular and ncols(X)==ncols(Y).
    x_init = 0.0_rk
    y_row3_vals = 1.0_rk
    call perflibs_spmat_create_dense(mat_xr, perflibs_col_major, n, nrhs, n, x_init, flags, stat)
    call check_stat(perflibs_status_success, stat, 'Setup X row mismatch')
    call perflibs_spmat_create_dense(mat_yr, perflibs_col_major, n-1, nrhs, n-1, y_row3_vals, flags, stat)
    call check_stat(perflibs_status_success, stat, 'Setup Y row mismatch')
    call perflibs_spsm_exec(perflibs_sparse_operation_notrans, mat_a, mat_xr, alpha, mat_yr, stat)
    call check_stat(perflibs_status_input_parameter_error, stat, 'Test 5 row mismatch XY')
    call destroy_mat(mat_xr, 'Cleanup X row mismatch')
    call destroy_mat(mat_yr, 'Cleanup Y row mismatch')

    ! Test 6: A row dimension mismatch with X and Y
    ! Param under test: nrows(A) == nrows(X) == nrows(Y).
    ! X and Y are mutually dimension-consistent.
    x_row3_vals = 0.0_rk
    y_row3_vals = 1.0_rk
    call perflibs_spmat_create_dense(mat_xa, perflibs_col_major, n-1, nrhs, n-1, x_row3_vals, flags, stat)
    call check_stat(perflibs_status_success, stat, 'Setup X A mismatch')
    call perflibs_spmat_create_dense(mat_ya, perflibs_col_major, n-1, nrhs, n-1, y_row3_vals, flags, stat)
    call check_stat(perflibs_status_success, stat, 'Setup Y A mismatch')
    call perflibs_spsm_exec(perflibs_sparse_operation_notrans, mat_a, mat_xa, alpha, mat_ya, stat)
    call check_stat(perflibs_status_input_parameter_error, stat, 'Test 6 row mismatch A vs XY')
    call destroy_mat(mat_xa, 'Cleanup X A mismatch')
    call destroy_mat(mat_ya, 'Cleanup Y A mismatch')

    ! Test 7: null A
    ! Param under test: A must not be null-matrix format.
    ! X, Y and TRANSA are valid.
    mat_null_a = perflibs_spmat_create_null(n, n)
    if (mat_null_a .le. 0) then
        write (*,*) 'Setup null A failed in test_fortran_spsm_interface'
        stop 1
    end if
    call perflibs_spsm_exec(perflibs_sparse_operation_notrans, mat_null_a, mat_x, alpha, mat_y, stat)
    call check_stat(perflibs_status_input_parameter_error, stat, 'Test 7 null A')
    call destroy_mat(mat_null_a, 'Cleanup null A')

    ! Test 8: A has at least one zero diagonal entry
    ! Param under test: diagonal validity of A (no zero diagonal entries).
    ! A remains square/triangular; only diagonal validity is broken.
    a_zero_diag_vals = a_vals
    a_zero_diag_vals(1) = 0.0_rk
    call perflibs_spmat_create_dense(mat_zero_diag_a, perflibs_col_major, n, n, n, a_zero_diag_vals, flags, stat)
    call check_stat(perflibs_status_success, stat, 'Setup zero-diag A')
    call perflibs_spsm_exec(perflibs_sparse_operation_notrans, mat_zero_diag_a, mat_x, alpha, mat_y, stat)
    call check_stat(perflibs_status_input_parameter_error, stat, 'Test 8 zero diagonal A')
    call destroy_mat(mat_zero_diag_a, 'Cleanup zero-diag A')

    ! Final cleanup
    call destroy_mat(mat_a, 'Cleanup A')
    call destroy_mat(mat_x, 'Cleanup X')
    call destroy_mat(mat_y, 'Cleanup Y')

    write (*,'(A,I1,A,I3)') "Success - test_fortran_spsm_interface, kind = ", rk, " ftype nbits: ", storage_size(dummy)

contains

    subroutine check_stat(expected, actual, label)
        integer(kind=perflibs_i4), intent(in) :: expected
        integer, intent(in) :: actual
        character(len=*), intent(in) :: label
        if (actual .ne. expected) then
            write (*,*) trim(label), ' failed in test_fortran_spsm_interface'
            stop 1
        end if
    end subroutine check_stat

    subroutine destroy_mat(mat, label)
        integer(kind=perflibs_i8), intent(inout) :: mat
        character(len=*), intent(in) :: label
        if (mat .gt. 0) then
            call perflibs_spmat_destroy(mat, stat)
            call check_stat(perflibs_status_success, stat, label)
            mat = 0
        end if
    end subroutine destroy_mat

    subroutine check_close_vec(got, expected, tolerance, label)
        FTYPE, intent(in), dimension(:) :: got, expected
        real(kind=rk), intent(in) :: tolerance
        character(len=*), intent(in) :: label
        integer :: i
        if (size(got) .ne. size(expected)) then
            write (*,*) trim(label), ' size mismatch in test_fortran_spsm_interface'
            stop 1
        end if
        do i = 1, size(got)
            if (abs(got(i) - expected(i)) .gt. tolerance) then
                write (*,*) trim(label), ' value mismatch at index ', i
                stop 1
            end if
        end do
    end subroutine check_close_vec

end program test_fortran_spsm_interface
