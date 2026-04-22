! SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
! affiliates <open-source-office@arm.com></text>
!
! SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

program test_fortran_spmm_interface
    use perflibs_sparse
    use iso_c_binding
    implicit none

    ! Use compiler flags to switch between 4, 8 byte reals and preprocessor to select FTYPE=real/complex type
    FTYPE :: dummy
    integer, parameter :: rk = kind(dummy)

    integer :: m = 5
    integer, parameter :: m_orig = 5
    integer :: n = 4
    integer, parameter :: n_orig = 4
    integer :: nnz = 6
    integer, parameter :: nnz_orig = 6
    integer :: lda = 6
    integer, parameter :: lda_orig = 6
    integer :: index_base = 0
    FTYPE, dimension(nnz_orig) :: vals = (/ 1.0_rk, 2.0_rk, 3.0_rk, 4.0_rk, 5.0_rk, 6.0_rk /)
    integer, dimension(nnz_orig) :: col_indx = (/ 0, 1, 2, 3, 2, 3 /)
    integer, dimension(m_orig + 1) :: row_ptr = (/ 0, 1, 2, 3, 4, 6 /)
    FTYPE, dimension(lda_orig*n_orig) :: dense_vals = (/ 1.0_rk, 2.0_rk, 3.0_rk, 4.0_rk, 5.0_rk, 6.0_rk, &
        7.0_rk, 8.0_rk, 9.0_rk, 10.0_rk, 11.0_rk, 12.0_rk, &
        13.0_rk, 14.0_rk, 15.0_rk, 16.0_rk, 17.0_rk, 18.0_rk, &
        19.0_rk, 20.0_rk, 21.0_rk, 22.0_rk, 23.0_rk, 24.0_rk /)
    integer, parameter :: bsr_m_orig = 9
    integer, parameter :: bsr_n_orig = 6
    integer, parameter :: nnzb = 2
    integer, parameter :: nrowsb = 3
    integer, parameter :: bsr_nnz = 18
    integer :: block_size = 3
    integer, parameter :: block_size_orig = 3
    integer(kind=perflibs_i4) :: block_layout = PERFLIBS_COL_MAJOR
    integer, dimension(nnzb) :: bsr_col_indx = (/ 0, 1 /)
    integer, dimension(nrowsb + 1) :: bsr_row_ptr = (/ 0, 1, 1, 2 /)
    FTYPE, dimension(bsr_nnz) :: bsr_vals = (/ 1.0_rk, 2.0_rk, 3.0_rk, 4.0_rk, 5.0_rk, 6.0_rk, &
                                               7.0_rk, 8.0_rk, 9.0_rk, 10.0_rk, 11.0_rk, 12.0_rk, &
                                               13.0_rk, 14.0_rk, 15.0_rk, 16.0_rk, 17.0_rk, 18.0_rk /)

    integer, allocatable :: row_alloc(:), col_alloc(:)
    FTYPE, allocatable :: vals_alloc(:)
    integer(kind=perflibs_i4) :: layout = PERFLIBS_COL_MAJOR
    integer(kind=perflibs_i4) :: bad_layout = -999

    FTYPE:: alpha = 1.0_rk, beta = 1.0_rk
    integer :: stat
    integer :: flags = 0
    integer(kind=perflibs_i8) :: perflibs_mat_a, perflibs_mat_b, &
                                 perflibs_mat_c, perflibs_null_mat, &
                                 perflibs_identity_mat, perflibs_mat_bsr
    integer(kind=perflibs_i4) :: fifty_five = 55

    integer :: libgfortran_mismatch
    external :: query_mismatch
    external :: perflibs_spmat_export_bsr_wrapper_s
    external :: perflibs_spmat_export_bsr_wrapper_d
    external :: perflibs_spmat_export_bsr_wrapper_c
    external :: perflibs_spmat_export_bsr_wrapper_z
    external :: perflibs_spmat_export_deallocate_wrapper
    type(c_ptr) :: c_row_alloc, c_col_alloc
    type(c_ptr) :: c_vals_alloc

    include 'type_query.h'

    ! Test 1: M < 0 in create_dense
    m = -1
    call perflibs_spmat_create_dense(perflibs_mat_a, layout, m, n, lda, dense_vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 1 failed in test_fortran_spmm_interface'
        stop 1
    end if
    call perflibs_spmat_destroy(perflibs_mat_a, stat)
    m = m_orig

    ! Test 2: N < 0 in create_dense
    n = -1
    call perflibs_spmat_create_dense(perflibs_mat_a, layout, m, n, lda, dense_vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 2 failed in test_fortran_spmm_interface'
        stop 1
    end if
    call perflibs_spmat_destroy(perflibs_mat_a, stat)
    n = n_orig

    ! Test 3: LDA < M in create_dense
    lda = m-1
    call perflibs_spmat_create_dense(perflibs_mat_a, layout, m, n, lda, dense_vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 3 failed in test_fortran_spmm_interface'
        stop 1
    end if
    call perflibs_spmat_destroy(perflibs_mat_a, stat)
    lda = lda_orig

    ! Test 4: LAYOUT type not valid in create_dense
    call perflibs_spmat_create_dense(perflibs_mat_a, fifty_five, m, n, lda, dense_vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 4 failed in test_fortran_spmm_interface'
        stop 1
    end if
    call perflibs_spmat_destroy(perflibs_mat_a, stat)

    ! Test 5: M < 0 in create_null
    ! Testing for NULL assumes a valid pointer value is always > 0
    m = -1
    perflibs_null_mat = perflibs_spmat_create_null(m, n)
    if (perflibs_null_mat .gt. 0) then
        write (*,*) 'Test 5 failed in test_fortran_spmm_interface'
        stop 1
    end if
    m = m_orig

    ! Test 6: N < 0 in create_null
    ! Testing for NULL assumes a valid pointer value is always > 0
    n = -1
    perflibs_null_mat = perflibs_spmat_create_null(m, n)
    if (perflibs_null_mat .gt. 0) then
        write (*,*) 'Test 6 failed in test_fortran_spmm_interface'
        stop 1
    end if
    n = n_orig

    ! Test 7: N < 0 in create_identity
    ! Testing for NULL assumes a valid pointer value is always > 0
    n = -1
    perflibs_identity_mat = perflibs_spmat_create_identity(n)
    if (perflibs_identity_mat .gt. 0) then
        write (*,*) 'Test 7 failed in test_fortran_spmm_interface'
        stop 1
    end if
    n = n_orig

    ! Setup testing for other routines
    call perflibs_spmat_create_csr(perflibs_mat_a, m, n, row_ptr, col_indx, vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Failed setup in test_fortran_spmm_interface'
        stop 1
    end if
    call perflibs_spmat_create_csr(perflibs_mat_b, m, n, row_ptr, col_indx, vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Failed setup in test_fortran_spmm_interface'
        stop 1
    end if
    call perflibs_spmat_create_csr(perflibs_mat_c, m, n, row_ptr, col_indx, vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Failed setup in test_fortran_spmm_interface'
        stop 1
    end if

    ! Test 8: perflibs_spmat_query always returns PERFLIBS_STATUS_SUCCESS, even with invalid A
    call perflibs_spmat_query(perflibs_mat_a, index_base, m, n, nnz, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Test 8 failed in test_fortran_spmm_interface'
        stop 1
    end if
    m = n_orig
    n = n_orig
    nnz = nnz_orig

    ! Test 9: invalid TRANSA in perflibs_spmm_optimize
    call perflibs_spmm_optimize(fifty_five, PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_SCALAR_ANY, &
                                perflibs_mat_a, perflibs_mat_b, PERFLIBS_SPARSE_SCALAR_ANY, perflibs_mat_c, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 9 failed in test_fortran_spmm_interface'
        stop 1
    end if

    ! Test 10: invalid TRANSB in perflibs_spmm_optimize
    call perflibs_spmm_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, fifty_five, PERFLIBS_SPARSE_SCALAR_ANY, &
                                perflibs_mat_a, perflibs_mat_b, PERFLIBS_SPARSE_SCALAR_ANY, perflibs_mat_c, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 10 failed in test_fortran_spmm_interface'
        stop 1
    end if

    ! Test 11: invalid ALPHA in perflibs_spmm_optimize
    call perflibs_spmm_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS, &
                                fifty_five, perflibs_mat_a, perflibs_mat_b, PERFLIBS_SPARSE_SCALAR_ANY,  &
                                perflibs_mat_c, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 11 failed in test_fortran_spmm_interface'
        stop 1
    end if

    ! Test 12: invalid BETA in perflibs_spmm_optimize
    call perflibs_spmm_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS, &
                                PERFLIBS_SPARSE_SCALAR_ANY, perflibs_mat_a, perflibs_mat_b, fifty_five,  &
                                perflibs_mat_c, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 12 failed in test_fortran_spmm_interface'
        stop 1
    end if

    ! Test 13: invalid TRANSA in perflibs_spadd_optimize
    call perflibs_spadd_optimize(fifty_five, PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_SCALAR_ANY, &
                                 perflibs_mat_a, PERFLIBS_SPARSE_SCALAR_ANY, perflibs_mat_b, perflibs_mat_c, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 13 failed in test_fortran_spmm_interface'
        stop 1
    end if

    ! Test 14: invalid TRANSB in perflibs_spadd_optimize
    call perflibs_spadd_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, fifty_five, PERFLIBS_SPARSE_SCALAR_ANY, &
                                 perflibs_mat_a, PERFLIBS_SPARSE_SCALAR_ANY, perflibs_mat_b, perflibs_mat_c, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 14 failed in test_fortran_spmm_interface'
        stop 1
    end if

    ! Test 15: invalid ALPHA in perflibs_spadd_optimize
    call perflibs_spadd_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS, &
                                 fifty_five, perflibs_mat_a, PERFLIBS_SPARSE_SCALAR_ANY, perflibs_mat_b,  &
                                 perflibs_mat_c, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 15 failed in test_fortran_spmm_interface'
        stop 1
    end if

    ! Test 16: invalid BETA in perflibs_spadd_optimize
    call perflibs_spadd_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS, &
                                 PERFLIBS_SPARSE_SCALAR_ANY, perflibs_mat_a, fifty_five, perflibs_mat_b,  &
                                 perflibs_mat_c, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 16 failed in test_fortran_spmm_interface'
        stop 1
    end if

    ! Test 17: TRANSA not valid in perflibs_spadd_exec
    call perflibs_spadd_exec(fifty_five, PERFLIBS_SPARSE_OPERATION_NOTRANS, alpha, perflibs_mat_a, beta, &
                             perflibs_mat_b, perflibs_mat_c, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 17 failed in test_fortran_spmm_interface'
        stop 1
    end if

    ! Test 18: TRANSB not valid in perflibs_spadd_exec
    call perflibs_spadd_exec(PERFLIBS_SPARSE_OPERATION_NOTRANS, fifty_five, alpha, perflibs_mat_a, beta, &
                             perflibs_mat_b, perflibs_mat_c, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 18 failed in test_fortran_spmm_interface'
        stop 1
    end if

    ! Test 19: INDEX_BASE not valid in perflibs_spmat_export_csr
    call perflibs_spmat_export_csr(perflibs_mat_a, -1, m, n, row_alloc, col_alloc, vals_alloc, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 19 failed in test_fortran_spmm_interface'
        stop 1
    end if

    ! Test 20: INDEX_BASE not valid in perflibs_spmat_export_csc
    call perflibs_spmat_export_csc(perflibs_mat_a, -1, m, n, row_alloc, col_alloc, vals_alloc, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 20 failed in test_fortran_spmm_interface'
        stop 1
    end if

    ! No error checking on user input parameters to perflibs_spmat_export_coo
    ! so no test written

    ! Test 21: Invalid LAYOUT to perflibs_spmat_export_dense
    call perflibs_spmat_export_dense(perflibs_mat_a, bad_layout, m, n, vals_alloc, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 21 failed in test_fortran_spmm_interface'
        stop 1
    end if

    ! Test 22: m < 0 in create_bsr
    m = -1
    n = bsr_n_orig
    call perflibs_spmat_create_bsr(perflibs_mat_bsr, block_layout, m, n, block_size, bsr_row_ptr, &
                                   bsr_col_indx, bsr_vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 22 failed in test_fortran_spmm_interface'
        stop 1
    end if
    call perflibs_spmat_destroy(perflibs_mat_bsr, stat)
    m = bsr_m_orig

    ! Test 23: n < 0 in create_bsr
    n = -1
    call perflibs_spmat_create_bsr(perflibs_mat_bsr, block_layout, m, n, block_size, bsr_row_ptr, &
                                   bsr_col_indx, bsr_vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 23 failed in test_fortran_spmm_interface'
        stop 1
    end if
    call perflibs_spmat_destroy(perflibs_mat_bsr, stat)
    n = bsr_n_orig

    ! Test 24: block_size < 0 in create_bsr
    block_size = -1
    call perflibs_spmat_create_bsr(perflibs_mat_bsr, block_layout, m, n, block_size, bsr_row_ptr, &
                                   bsr_col_indx, bsr_vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 24 failed in test_fortran_spmm_interface'
        stop 1
    end if
    call perflibs_spmat_destroy(perflibs_mat_bsr, stat)

    ! Test 25: block_size = 0 in create_bsr
    block_size = 0
    call perflibs_spmat_create_bsr(perflibs_mat_bsr, block_layout, m, n, block_size, bsr_row_ptr, &
                                   bsr_col_indx, bsr_vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 25 failed in test_fortran_spmm_interface'
        stop 1
    end if
    call perflibs_spmat_destroy(perflibs_mat_bsr, stat)

    ! Test 26: bad multiple block_size of m in create_bsr
    block_size = 2
    call perflibs_spmat_create_bsr(perflibs_mat_bsr, block_layout, m, n, block_size, bsr_row_ptr, &
                                   bsr_col_indx, bsr_vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 26 failed in test_fortran_spmm_interface'
        stop 1
    end if
    call perflibs_spmat_destroy(perflibs_mat_bsr, stat)

    ! Test 27: bad multiple block_size of n in create_bsr
    block_size = 9
    call perflibs_spmat_create_bsr(perflibs_mat_bsr, block_layout, m, n, block_size, bsr_row_ptr, &
                                   bsr_col_indx, bsr_vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 27 failed in test_fortran_spmm_interface'
        stop 1
    end if
    call perflibs_spmat_destroy(perflibs_mat_bsr, stat)
    block_size = block_size_orig

    ! Test 28: bad block_layout in create_bsr
    call perflibs_spmat_create_bsr(perflibs_mat_bsr, bad_layout, m, n, block_size, bsr_row_ptr, &
                                   bsr_col_indx, bsr_vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 28 failed in test_fortran_spmm_interface'
        stop 1
    end if
    call perflibs_spmat_destroy(perflibs_mat_bsr, stat)

    ! Test 29: bad index_base in create_bsr
    bsr_row_ptr(1) = 55
    call perflibs_spmat_create_bsr(perflibs_mat_bsr, block_layout, m, n, block_size, bsr_row_ptr, &
                                   bsr_col_indx, bsr_vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 29 failed in test_fortran_spmm_interface'
        stop 1
    end if
    call perflibs_spmat_destroy(perflibs_mat_bsr, stat)
    bsr_row_ptr(1) = 0

    ! Test 30: all variables valid in create_bsr
    ! Also sets up perflibs_mat_bsr object for the export tests
    call perflibs_spmat_create_bsr(perflibs_mat_bsr, block_layout, m, n, block_size, bsr_row_ptr, &
                                   bsr_col_indx, bsr_vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Test 30 failed in test_fortran_spmm_interface'
        stop 1
    end if

    ! Test 31: bad index_base in export_bsr
    call perflibs_spmat_export_bsr(perflibs_mat_bsr, block_layout, -1, m, n, block_size, row_alloc, col_alloc, vals_alloc, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 31 failed in test_fortran_spmm_interface'
        stop 1
    end if

    ! Test 32: bad block_layout in export_bsr
    call perflibs_spmat_export_bsr(perflibs_mat_bsr, bad_layout, index_base, &
                                   m, n, block_size, row_alloc, col_alloc, vals_alloc, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 32 failed in test_fortran_spmm_interface'
        stop 1
    end if

    ! Test 33: all variables valid in export_bsr
    call query_mismatch(libgfortran_mismatch)
    if (libgfortran_mismatch .eq. 1) then

        if (typeid .eq. 1) then
            call perflibs_spmat_export_bsr_wrapper_s(perflibs_mat_bsr, block_layout, index_base, m, n, block_size, &
                                                     c_row_alloc, c_col_alloc, c_vals_alloc, stat)
        else if (typeid .eq. 2) then
            call perflibs_spmat_export_bsr_wrapper_d(perflibs_mat_bsr, block_layout, index_base, m, n, block_size, &
                                                     c_row_alloc, c_col_alloc, c_vals_alloc, stat)
        else if (typeid .eq. 3) then
            call perflibs_spmat_export_bsr_wrapper_c(perflibs_mat_bsr, block_layout, index_base, m, n, block_size, &
                                                     c_row_alloc, c_col_alloc, c_vals_alloc, stat)
        else if (typeid .eq. 4) then
            call perflibs_spmat_export_bsr_wrapper_z(perflibs_mat_bsr, block_layout, index_base, m, n, block_size, &
                                                     c_row_alloc, c_col_alloc, c_vals_alloc, stat)
        end if

        if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
            write (*,*) 'Test 35w failed in test_fortran_spmm_interface'
            stop 1
        end if

        call perflibs_spmat_export_deallocate_wrapper(c_row_alloc)
        call perflibs_spmat_export_deallocate_wrapper(c_col_alloc)
        call perflibs_spmat_export_deallocate_wrapper(c_vals_alloc)

    else

        call perflibs_spmat_export_bsr(perflibs_mat_bsr, block_layout, index_base, m, n, block_size, &
                                       row_alloc, col_alloc, vals_alloc, stat)
        if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
            write (*,*) 'Test 35 failed in test_fortran_spmm_interface'
            stop 1
        end if
        deallocate(row_alloc)
        deallocate(col_alloc)
        deallocate(vals_alloc)

    end if

    ! Destroy A
    call perflibs_spmat_destroy(perflibs_mat_a, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Failed to destroy A in test_fortran_spmm_interface'
        stop 1
    end if

    ! Destroy B
    call perflibs_spmat_destroy(perflibs_mat_b, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Failed to destroy B in test_fortran_spmm_interface'
        stop 1
    end if

    ! Destroy C
    call perflibs_spmat_destroy(perflibs_mat_c, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Failed to destroy C in test_fortran_spmm_interface'
        stop 1
    end if

    ! Destroy the bsr matrix
    call perflibs_spmat_destroy(perflibs_mat_bsr, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Failed BSR destroy in test_fortran_spmm_interface'
        stop 1
    end if

    write (*,'(A,I1,A,I3)') "Success - test_fortran_spmm_interface, kind = ", rk, &
                            " ftype nbits: ", storage_size(dummy)

end program test_fortran_spmm_interface
