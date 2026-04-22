! SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
! affiliates <open-source-office@arm.com></text>
!
! SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

program test_fortran_spsv_exec_interface
    use perflibs_sparse
    use iso_c_binding
    implicit none

    ! Use compiler flags to switch between 4, 8 byte reals and preprocessor to select FTYPE=real/complex type
    FTYPE :: dummy
    integer, parameter :: rk = kind(dummy)

    integer, parameter :: n_orig = 4
    integer, parameter :: nnz_orig = 8
    integer :: m, n, index_base, nnz, block_size
    FTYPE, dimension(nnz_orig) :: vals = (/ 1.0_rk, 2.0_rk, 3.0_rk, 4.0_rk, 5.0_rk, 6.0_rk, 7.0_rk, 8.0_rk /)
    integer, dimension(nnz_orig) :: col_indx_orig = (/ 1, 1, 2, 1, 2, 3, 2, 4 /)
    integer, dimension(n_orig + 1) :: row_ptr_orig = (/ 1, 2, 4, 7, 9 /)
    FTYPE, dimension(n_orig) :: x
    FTYPE, dimension(n_orig) :: y = (/ 9.0_rk, 10.0_rk, 11.0_rk, 12.0_rk /)
    FTYPE, allocatable :: vals_csc(:), vals_coo(:), vals_bsr(:), vals_dense(:)
    integer, allocatable :: row_indx(:), row_ptr(:), col_indx(:), col_ptr(:)

    FTYPE:: alpha = 1.1_rk
    integer :: stat
    integer :: flags = 0
    integer(kind=perflibs_i4) :: fake_trans = 7777
    integer(kind=perflibs_i8) :: perflibs_mat_csr, perflibs_mat_csc, perflibs_mat_coo, perflibs_mat_bsr, perflibs_mat_dense
    integer(kind=perflibs_i8) :: perflibs_mat_identity, perflibs_mat_null

    integer :: libgfortran_mismatch
    external :: query_mismatch
    external :: perflibs_spmat_export_dense_wrapper_s
    external :: perflibs_spmat_export_dense_wrapper_d
    external :: perflibs_spmat_export_dense_wrapper_c
    external :: perflibs_spmat_export_dense_wrapper_z
    external :: perflibs_spmat_export_csc_wrapper_s
    external :: perflibs_spmat_export_csc_wrapper_d
    external :: perflibs_spmat_export_csc_wrapper_c
    external :: perflibs_spmat_export_csc_wrapper_z
    external :: perflibs_spmat_export_coo_wrapper_s
    external :: perflibs_spmat_export_coo_wrapper_d
    external :: perflibs_spmat_export_coo_wrapper_c
    external :: perflibs_spmat_export_coo_wrapper_z
    external :: perflibs_spmat_export_bsr_wrapper_s
    external :: perflibs_spmat_export_bsr_wrapper_d
    external :: perflibs_spmat_export_bsr_wrapper_c
    external :: perflibs_spmat_export_bsr_wrapper_z
    external :: perflibs_spmat_export_deallocate_wrapper
    type(c_ptr) :: c_row_indx, c_col_ptr, c_vals, c_col_indx, c_row_ptr
    integer, pointer :: p_row_indx(:), p_col_ptr(:), p_col_indx(:), p_row_ptr(:)
    FTYPE, pointer :: p_vals(:)

    include 'type_query.h'

    n = n_orig
    nnz = nnz_orig
    ! Setup 1: create csr matrix
    call perflibs_spmat_create_csr(perflibs_mat_csr, n, n, row_ptr_orig, col_indx_orig, vals, flags, stat)
    call check_stat(perflibs_status_success, 'Setup 1')

    ! Test 1: Run exec with csr format with invalid trans
    call perflibs_spsv_exec(fake_trans, perflibs_mat_csr, x, alpha, y, stat)
    call check_stat(perflibs_status_input_parameter_error, 'Test 1')

    ! Test 1.1: Pass a SpSV hint
    call perflibs_spmat_hint(perflibs_mat_csr, perflibs_sparse_hint_spsv_invocations, perflibs_sparse_invocations_few, stat)
    call check_stat(perflibs_status_success, 'Test 1.1')

    ! Test 1.2: Run optimize with csr format
    call perflibs_spsv_optimize(perflibs_mat_csr, stat)
    call check_stat(perflibs_status_success, 'Test 1.2')

    ! Test 1.3: Run exec with csr format with correct parameters
    call perflibs_spsv_exec(perflibs_sparse_operation_notrans, perflibs_mat_csr, x, alpha, y, stat)
    call check_stat(perflibs_status_success, 'Test 1.3')

    ! Setup 2: Query parameters of csr matrix
    call perflibs_spmat_query(perflibs_mat_csr, index_base, m, n, nnz, stat)
    call check_stat(perflibs_status_success, 'Setup 2')
    if ((m .ne. n_orig) .or. (n .ne. n_orig) .or. (nnz .ne. nnz_orig)) then
        stat = perflibs_status_execution_failure
    else
        stat = perflibs_status_success
    end if
    call check_stat(perflibs_status_success, 'Setup 2.1')

    ! Setup 3: Export csc format
    call query_mismatch(libgfortran_mismatch)
    if (libgfortran_mismatch .eq. 1) then

        if (typeid .eq. 1) then
            call perflibs_spmat_export_csc_wrapper_s(perflibs_mat_csr, index_base, m, n, &
                                                     c_row_indx, c_col_ptr, c_vals, stat)
        else if (typeid .eq. 2) then
            call perflibs_spmat_export_csc_wrapper_d(perflibs_mat_csr, index_base, m, n, &
                                                     c_row_indx, c_col_ptr, c_vals, stat)
        else if (typeid .eq. 3) then
            call perflibs_spmat_export_csc_wrapper_c(perflibs_mat_csr, index_base, m, n, &
                                                     c_row_indx, c_col_ptr, c_vals, stat)
        else if (typeid .eq. 4) then
            call perflibs_spmat_export_csc_wrapper_z(perflibs_mat_csr, index_base, m, n, &
                                                     c_row_indx, c_col_ptr, c_vals, stat)
        end if
        call check_stat(perflibs_status_success, 'Setup 3w')

        call c_f_pointer(c_row_indx, p_row_indx, [nnz])
        call c_f_pointer(c_col_ptr, p_col_ptr, [n + 1])
        call c_f_pointer(c_vals, p_vals, [nnz])

        ! Allocate and copy into the arrays that are used later on
        allocate(row_indx(nnz))
        row_indx(:) = p_row_indx(:)
        call perflibs_spmat_export_deallocate_wrapper(c_row_indx)
        allocate(col_ptr(n + 1))
        col_ptr(:) = p_col_ptr(:)
        call perflibs_spmat_export_deallocate_wrapper(c_col_ptr)
        allocate(vals_csc(nnz))
        vals_csc(:) = p_vals(:)
        call perflibs_spmat_export_deallocate_wrapper(c_vals)

    else

        call perflibs_spmat_export_csc(perflibs_mat_csr, index_base, m, n, row_indx, col_ptr, vals_csc, stat)
        call check_stat(perflibs_status_success, 'Setup 3')

    end if

    ! Setup 4: create csc matrix
    call perflibs_spmat_create_csc(perflibs_mat_csc, n, n, row_indx, col_ptr, vals_csc, flags, stat)
    call check_stat(perflibs_status_success, 'Setup 4')

    ! Test 2: Run exec with csc format with correct parameters
    call perflibs_spsv_optimize(perflibs_mat_csc, stat)
    call perflibs_spsv_exec(perflibs_sparse_operation_notrans, perflibs_mat_csc, x, alpha, y, stat)
    call check_stat(perflibs_status_success, 'Test 2')
    deallocate(vals_csc)
    deallocate(row_indx)
    deallocate(col_ptr)

    ! Setup 5: Export coo format
    if (libgfortran_mismatch .eq. 1) then

        if (typeid .eq. 1) then
            call perflibs_spmat_export_coo_wrapper_s(perflibs_mat_csr, m, n, nnz, &
                                                     c_row_indx, c_col_indx, c_vals, stat)
        else if (typeid .eq. 2) then
            call perflibs_spmat_export_coo_wrapper_d(perflibs_mat_csr, m, n, nnz, &
                                                     c_row_indx, c_col_indx, c_vals, stat)
        else if (typeid .eq. 3) then
            call perflibs_spmat_export_coo_wrapper_c(perflibs_mat_csr, m, n, nnz, &
                                                     c_row_indx, c_col_indx, c_vals, stat)
        else if (typeid .eq. 4) then
            call perflibs_spmat_export_coo_wrapper_z(perflibs_mat_csr, m, n, nnz, &
                                                     c_row_indx, c_col_indx, c_vals, stat)
        end if
        call check_stat(perflibs_status_success, 'Setup 5w')

        call c_f_pointer(c_row_indx, p_row_indx, [nnz])
        call c_f_pointer(c_col_indx, p_col_indx, [nnz])
        call c_f_pointer(c_vals, p_vals, [nnz])

        ! Allocate and copy into the arrays that are used later on
        allocate(row_indx(nnz))
        row_indx(:) = p_row_indx(:)
        call perflibs_spmat_export_deallocate_wrapper(c_row_indx)
        allocate(col_indx(nnz))
        col_indx(:) = p_col_indx(:)
        call perflibs_spmat_export_deallocate_wrapper(c_col_indx)
        allocate(vals_coo(nnz))
        vals_coo(:) = p_vals(:)
        call perflibs_spmat_export_deallocate_wrapper(c_vals)

    else

        call perflibs_spmat_export_coo(perflibs_mat_csr, m, n, nnz, row_indx, col_indx, vals_coo, stat)
        call check_stat(perflibs_status_success, 'Setup 5')

    end if

    ! Setup 6: create coo matrix
    call perflibs_spmat_create_coo(perflibs_mat_coo, n, n, nnz, row_indx, col_indx, vals_coo, flags, stat)
    call check_stat(perflibs_status_success, 'Setup 6')

    ! Test 3: Run exec with coo format with correct parameters
    call perflibs_spsv_optimize(perflibs_mat_coo, stat)
    call perflibs_spsv_exec(perflibs_sparse_operation_notrans, perflibs_mat_coo, x, alpha, y, stat)
    call check_stat(perflibs_status_success, 'Test 3')
    deallocate(vals_coo)
    deallocate(row_indx)
    deallocate(col_indx)

    ! Setup 7: Export bsr format
    if (libgfortran_mismatch .eq. 1) then

        if (typeid .eq. 1) then
            call perflibs_spmat_export_bsr_wrapper_s(perflibs_mat_csr, perflibs_col_major, index_base, m, n, block_size, &
                                                     c_row_ptr, c_col_indx, c_vals, stat)
        else if (typeid .eq. 2) then
            call perflibs_spmat_export_bsr_wrapper_d(perflibs_mat_csr, perflibs_col_major, index_base, m, n, block_size, &
                                                     c_row_ptr, c_col_indx, c_vals, stat)
        else if (typeid .eq. 3) then
            call perflibs_spmat_export_bsr_wrapper_c(perflibs_mat_csr, perflibs_col_major, index_base, m, n, block_size, &
                                                     c_row_ptr, c_col_indx, c_vals, stat)
        else if (typeid .eq. 4) then
            call perflibs_spmat_export_bsr_wrapper_z(perflibs_mat_csr, perflibs_col_major, index_base, m, n, block_size, &
                                                     c_row_ptr, c_col_indx, c_vals, stat)
        end if
        call check_stat(perflibs_status_success, 'Setup 7w')

        call c_f_pointer(c_row_ptr, p_row_ptr, [m + 1])
        call c_f_pointer(c_col_indx, p_col_indx, [nnz])
        call c_f_pointer(c_vals, p_vals, [nnz])

        ! Allocate and copy into the arrays that are used later on
        allocate(row_ptr(m + 1))
        row_ptr(:) = p_row_ptr(:)
        call perflibs_spmat_export_deallocate_wrapper(c_row_ptr)
        allocate(col_indx(nnz))
        col_indx(:) = p_col_indx(:)
        call perflibs_spmat_export_deallocate_wrapper(c_col_indx)
        allocate(vals_bsr(nnz))
        vals_bsr(:) = p_vals(:)
        call perflibs_spmat_export_deallocate_wrapper(c_vals)

    else

        call perflibs_spmat_export_bsr(perflibs_mat_csr, perflibs_col_major, index_base, &
                                       m, n, block_size, row_ptr, col_indx, vals_bsr, stat)
        call check_stat(perflibs_status_success, 'Setup 7')

    end if

    ! Setup 8: create bsr matrix
    call perflibs_spmat_create_bsr(perflibs_mat_bsr, perflibs_col_major, n, n, block_size, row_ptr, col_indx, vals_bsr, flags, stat)
    call check_stat(perflibs_status_success, 'Setup 8')

    ! Test 4: Run exec with bsr format with correct parameters
    call perflibs_spsv_optimize(perflibs_mat_bsr, stat)
    call perflibs_spsv_exec(perflibs_sparse_operation_notrans, perflibs_mat_bsr, x, alpha, y, stat)
    call check_stat(perflibs_status_success, 'Test 4')
    deallocate(vals_bsr)
    deallocate(row_ptr)
    deallocate(col_indx)

    ! Setup 10: Export dense format
    if (libgfortran_mismatch .eq. 1) then

        if (typeid .eq. 1) then
            call perflibs_spmat_export_dense_wrapper_s(perflibs_mat_csr, perflibs_col_major, m, n, c_vals, stat)
        else if (typeid .eq. 2) then
            call perflibs_spmat_export_dense_wrapper_d(perflibs_mat_csr, perflibs_col_major, m, n, c_vals, stat)
        else if (typeid .eq. 3) then
            call perflibs_spmat_export_dense_wrapper_c(perflibs_mat_csr, perflibs_col_major, m, n, c_vals, stat)
        else if (typeid .eq. 4) then
            call perflibs_spmat_export_dense_wrapper_z(perflibs_mat_csr, perflibs_col_major, m, n, c_vals, stat)
        end if
        call check_stat(perflibs_status_success, 'Setup 10w')

        call c_f_pointer(c_vals, p_vals, [m*n])

        ! Allocate and copy into the arrays that are used later on
        allocate(vals_dense(m*n))
        vals_dense(:) = p_vals(:)
        call perflibs_spmat_export_deallocate_wrapper(c_vals)

    else

        call perflibs_spmat_export_dense(perflibs_mat_csr, perflibs_col_major, m, n, vals_dense, stat)
        call check_stat(perflibs_status_success, 'Setup 10')

    end if

    ! Setup 11: create dense matrix
    call perflibs_spmat_create_dense(perflibs_mat_dense, perflibs_col_major, n, n, n, vals_dense, flags, stat)
    call check_stat(perflibs_status_success, 'Setup 11')

    ! Test 5: Run exec with dense format with correct parameters
    call perflibs_spsv_optimize(perflibs_mat_dense, stat)
    call perflibs_spsv_exec(perflibs_sparse_operation_notrans, perflibs_mat_dense, x, alpha, y, stat)
    call check_stat(perflibs_status_success, 'Test 5')
    deallocate(vals_dense)

    ! Test 6: Run exec with identity format
    perflibs_mat_identity = perflibs_spmat_create_identity(n)
    call perflibs_spsv_exec(perflibs_sparse_operation_notrans, perflibs_mat_identity, x, alpha, y, stat)
    call check_stat(perflibs_status_success, 'Test 6')

    ! Test 7: Run exec with null format
    perflibs_mat_null = perflibs_spmat_create_null(n, n)
    call perflibs_spsv_exec(perflibs_sparse_operation_notrans, perflibs_mat_null, x, alpha, y, stat)
    call check_stat(perflibs_status_input_parameter_error, 'Test 7')

    ! Destroy csr:
    call perflibs_spmat_destroy(perflibs_mat_csr, stat)
    call check_stat(perflibs_status_success, 'Destroy csr')

    ! Destroy csc:
    call perflibs_spmat_destroy(perflibs_mat_csc, stat)
    call check_stat(perflibs_status_success, 'Destroy csc')

    ! Destroy coo:
    call perflibs_spmat_destroy(perflibs_mat_coo, stat)
    call check_stat(perflibs_status_success, 'Destroy coo')

    ! Destroy bsr:
    call perflibs_spmat_destroy(perflibs_mat_bsr, stat)
    call check_stat(perflibs_status_success, 'Destroy bsr')

    ! Destroy dense:
    call perflibs_spmat_destroy(perflibs_mat_dense, stat)
    call check_stat(perflibs_status_success, 'Destroy dense')

    ! Destroy identity:
    call perflibs_spmat_destroy(perflibs_mat_identity, stat)
    call check_stat(perflibs_status_success, 'Destroy identity')

    ! Destroy null:
    call perflibs_spmat_destroy(perflibs_mat_null, stat)
    call check_stat(perflibs_status_success, 'Destroy null')

    write (*,'(A,I1,A,I3)') "Success - test_fortran_spsv_exec_interface, kind = ", rk, " ftype nbits: ", storage_size(dummy)

contains

    subroutine check_stat(stat_to_check, test_text)
        integer(kind=perflibs_i4), intent(in) :: stat_to_check
        character(len=*), intent(in) :: test_text

        if (stat .ne. stat_to_check) then
            write (*,*) 'Failed ', test_text(1:len_trim(test_text)), ' in test_fortran_spsv_exec_interface'
            stop 1
        end if

        return
    end subroutine check_stat

end program test_fortran_spsv_exec_interface
