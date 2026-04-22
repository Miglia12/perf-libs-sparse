! SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
! affiliates <open-source-office@arm.com></text>
!
! SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

program test_fortran_spmv_interface
    use perflibs_sparse
    implicit none

    ! Use compiler flags to switch between 4, 8 byte reals and preprocessor to select FTYPE=real/complex type
    FTYPE :: dummy
    integer, parameter :: rk = kind(dummy)

    integer :: m = 5
    integer, parameter :: m_orig = 5
    integer :: n = 4
    integer, parameter :: n_orig = 4
    integer, parameter :: nnz = 6
    FTYPE, dimension(nnz) :: vals = (/ 1.0_rk, 2.0_rk, 3.0_rk, 4.0_rk, 5.0_rk, 6.0_rk /)
    integer, dimension(nnz) :: col_indx = (/ 0, 1, 2, 3, 2, 3 /)
    integer, dimension(m_orig + 1) :: row_ptr = (/ 0, 1, 2, 3, 4, 6 /)
    FTYPE, dimension(n_orig) :: x
    FTYPE, dimension(m_orig) :: y

    integer, parameter :: n_updates_size = 1
    integer, dimension(n_updates_size) :: col_indx_up = (/ 0 /)
    integer, dimension(n_updates_size) :: row_indx_up = (/ 3 /)
    FTYPE, dimension(n_updates_size) :: vals_up = (/ 55.0_rk /)
    integer :: n_updates = 1

    FTYPE:: alpha = 1.0_rk, beta = 1.0_rk
    integer :: stat
    integer :: flags = 0
    integer(kind=perflibs_i8) :: perflibs_mat
    integer(kind=perflibs_i4) :: fifty_five = 55

    ! Test 1: m < 0 in create_csr
    m = -1
    call perflibs_spmat_create_csr(perflibs_mat, m, n, row_ptr, col_indx, vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 1 failed in test_fortran_spmv_interface'
        stop 1
    end if
    call cleanup()
    m = m_orig

    ! Test 2: n < 0 in create_csr
    n = -1
    call perflibs_spmat_create_csr(perflibs_mat, m, n, row_ptr, col_indx, vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 2 failed in test_fortran_spmv_interface'
        stop 1
    end if
    call cleanup()
    n = n_orig

    ! Test 3: row_ptr(1) not 0 or 1 in create_csr
    row_ptr(1) = 55
    call perflibs_spmat_create_csr(perflibs_mat, m, n, row_ptr, col_indx, vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 3 failed in test_fortran_spmv_interface'
        stop 1
    end if
    call cleanup()
    row_ptr(1) = 0

    ! Test 4: m < 0 in create_csc
    m = -1
    call perflibs_spmat_create_csc(perflibs_mat, m, n, col_indx, row_ptr, vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 4 failed in test_fortran_spmv_interface'
        stop 1
    end if
    call cleanup()
    m = m_orig

    ! Test 5: n < 0 in create_csc
    n = -1
    call perflibs_spmat_create_csc(perflibs_mat, m, n, col_indx, row_ptr, vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 5 failed in test_fortran_spmv_interface'
        stop 1
    end if
    call cleanup()
    n = n_orig

    ! Test 6: col_ptr(1) not 0 or 1 in create_csc
    row_ptr(1) = 55
    call perflibs_spmat_create_csc(perflibs_mat, m, n, col_indx, row_ptr, vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 6 failed in test_fortran_spmv_interface'
        stop 1
    end if
    call cleanup()
    row_ptr(1) = 0

    ! Test 7: m < 0 in create_coo
    m = -1
    call perflibs_spmat_create_coo(perflibs_mat, m, n, nnz, col_indx, col_indx, vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 7 failed in test_fortran_spmv_interface'
        stop 1
    end if
    m = m_orig
    call cleanup()

    ! Test 8: n < 0 in create_coo
    n = -1
    call perflibs_spmat_create_coo(perflibs_mat, m, n, nnz, col_indx, col_indx, vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 8 failed in test_fortran_spmv_interface'
        stop 1
    end if
    n = n_orig
    call cleanup()


    ! Setup testing for other routines
    call perflibs_spmat_create_csr(perflibs_mat, m, n, row_ptr, col_indx, vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Failed setup in test_fortran_spmv_interface'
        stop 1
    end if

    ! Test 9: n_updates < 0 in update
    n_updates = -1
    call perflibs_spmat_update(perflibs_mat, n_updates, row_indx_up, col_indx_up, vals_up, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 9 failed in test_fortran_spmv_interface'
        stop 1
    end if
    n_updates = 1

    ! Test 10: update not valid
    call perflibs_spmat_update(perflibs_mat, n_updates, row_indx_up, col_indx_up, vals_up, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 10 failed in test_fortran_spmv_interface'
        stop 1
    end if
    call perflibs_spmat_print_err(perflibs_mat) ! Print the error (for coverage; not checked)

    ! Test 11: hint type not valid
    call perflibs_spmat_hint(perflibs_mat, fifty_five, PERFLIBS_SPARSE_OPERATION_NOTRANS, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 11 failed in test_fortran_spmv_interface'
        stop 1
    end if

    ! Test 12: hint value not valid
    call perflibs_spmat_hint(perflibs_mat, PERFLIBS_SPARSE_HINT_SPMV_OPERATION, fifty_five, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 12 failed in test_fortran_spmv_interface'
        stop 1
    end if

    ! Test 13: trans not valid
    call perflibs_spmv_exec(fifty_five, alpha, perflibs_mat, x, beta, y, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 13 failed in test_fortran_spmv_interface'
        stop 1
    end if

    ! Final cleanup
    call cleanup()

    write (*,'(A,I1,A,I3)') "Success - test_fortran_spmv_interface, kind = ", rk, " ftype nbits: ", storage_size(dummy)

contains

    subroutine cleanup()
        ! Destroy the thing
        call perflibs_spmat_destroy(perflibs_mat, stat)
        if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
            write (*,*) 'Failed destroy in test_fortran_spmv_interface'
            stop 1
        end if
    end subroutine cleanup

end program test_fortran_spmv_interface
