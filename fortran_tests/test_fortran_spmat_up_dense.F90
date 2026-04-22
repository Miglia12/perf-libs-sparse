! SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
! affiliates <open-source-office@arm.com></text>
!
! SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

program test_fortran_spmat_update_dense
    use perflibs_sparse
    use iso_c_binding
    implicit none

    ! Use compiler flags to switch between 4, 8 byte reals and preprocessor to select FTYPE=real/complex type
    FTYPE :: dummy
    integer, parameter :: rk = kind(dummy)

    integer, parameter :: nnz = 6
    integer :: m = 3
    integer :: n = 2
    integer :: lda
    FTYPE, dimension(nnz) :: vals = (/ 1.0_rk, 2.0_rk, 3.0_rk, 4.0_rk, 5.0_rk, 6.0_rk /)
    FTYPE, allocatable, dimension(:) :: vals_exp1
    FTYPE, allocatable, dimension(:) :: vals_exp2

    integer, parameter :: n_updates_size = 1
    integer, dimension(n_updates_size) :: col_indx_up = (/ 2 /)
    integer, dimension(n_updates_size) :: row_indx_up = (/ 1 /)
    FTYPE, dimension(n_updates_size) :: vals_up = (/ 55.0_rk /)
    integer :: n_updates = 1

    integer :: stat
    integer :: flags = 0
    integer(kind=perflibs_i8) :: perflibs_mat

    integer :: libgfortran_mismatch
    external :: query_mismatch
    external :: perflibs_spmat_export_dense_wrapper_s
    external :: perflibs_spmat_export_dense_wrapper_d
    external :: perflibs_spmat_export_dense_wrapper_c
    external :: perflibs_spmat_export_dense_wrapper_z
    external :: perflibs_spmat_export_deallocate_wrapper
    type(c_ptr) :: c_vals_exp1, c_vals_exp2
    FTYPE, pointer :: p_vals_exp1(:), p_vals_exp2(:)

    include 'type_query.h'

    ! First test with a column major matrix
    lda = m

    call perflibs_spmat_create_dense(perflibs_mat, PERFLIBS_COL_MAJOR, m, n, lda, vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Test 1.0 failed in test_fortran_spmat_update_dense'
        stop 1
    end if

    ! Test 1.1: update element
    call perflibs_spmat_update(perflibs_mat, n_updates, row_indx_up, col_indx_up, vals_up, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Test 1.1 failed in test_fortran_spmat_update_dense'
        stop 1
    end if

    ! Test 1.2: export matrix
    call query_mismatch(libgfortran_mismatch)
    if (libgfortran_mismatch .eq. 1) then

        if (typeid .eq. 1) then
            call perflibs_spmat_export_dense_wrapper_s(perflibs_mat, PERFLIBS_COL_MAJOR, m, n, c_vals_exp1, stat)
        else if (typeid .eq. 2) then
            call perflibs_spmat_export_dense_wrapper_d(perflibs_mat, PERFLIBS_COL_MAJOR, m, n, c_vals_exp1, stat)
        else if (typeid .eq. 3) then
            call perflibs_spmat_export_dense_wrapper_c(perflibs_mat, PERFLIBS_COL_MAJOR, m, n, c_vals_exp1, stat)
        else if (typeid .eq. 4) then
            call perflibs_spmat_export_dense_wrapper_z(perflibs_mat, PERFLIBS_COL_MAJOR, m, n, c_vals_exp1, stat)
        end if

        if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
          write (*,*) 'Test 1.2w failed in test_fortran_spmat_update_dense'
          stop 1
        end if

        call c_f_pointer(c_vals_exp1, p_vals_exp1, [nnz])

        ! Check the value was updated properly
        if (p_vals_exp1(4) /= vals_up(1)) then
            write (*,*) 'Test 1.3w failed in test_fortran_spmat_update_dense'
            stop 1
        end if

        call perflibs_spmat_export_deallocate_wrapper(c_vals_exp1)

    else

        call perflibs_spmat_export_dense(perflibs_mat, PERFLIBS_COL_MAJOR, m, n, vals_exp1, stat)
        if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
          write (*,*) 'Test 1.2 failed in test_fortran_spmat_update_dense'
          stop 1
        end if

        ! Check the value was updated properly
        if (vals_exp1(4) /= vals_up(1)) then
            write (*,*) 'Test 1.3 failed in test_fortran_spmat_update_dense'
            stop 1
        end if

        deallocate(vals_exp1)

    end if

    ! Destroy the thing
    call perflibs_spmat_destroy(perflibs_mat, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Failed destroy in test_fortran_spmat_update_dense'
        stop 1
    end if

!   Repeat all of the above for a matrix input as row major

    lda = n

    call perflibs_spmat_create_dense(perflibs_mat, PERFLIBS_ROW_MAJOR, m, n, lda, vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Test 2.0 failed in test_fortran_spmat_update_dense'
        stop 1
    end if

    ! Test 2.1: update element
    call perflibs_spmat_update(perflibs_mat, n_updates, row_indx_up, col_indx_up, vals_up, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Test 2.1 failed in test_fortran_spmat_update_dense'
        stop 1
    end if

    call query_mismatch(libgfortran_mismatch)
    if (libgfortran_mismatch .eq. 1) then

        if (typeid .eq. 1) then
            call perflibs_spmat_export_dense_wrapper_s(perflibs_mat, PERFLIBS_COL_MAJOR, m, n, c_vals_exp2, stat)
        else if (typeid .eq. 2) then
            call perflibs_spmat_export_dense_wrapper_d(perflibs_mat, PERFLIBS_COL_MAJOR, m, n, c_vals_exp2, stat)
        else if (typeid .eq. 3) then
            call perflibs_spmat_export_dense_wrapper_c(perflibs_mat, PERFLIBS_COL_MAJOR, m, n, c_vals_exp2, stat)
        else if (typeid .eq. 4) then
            call perflibs_spmat_export_dense_wrapper_z(perflibs_mat, PERFLIBS_COL_MAJOR, m, n, c_vals_exp2, stat)
        end if

        if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
          write (*,*) 'Test 2.2w failed in test_fortran_spmat_update_dense'
          stop 1
        end if

        call c_f_pointer(c_vals_exp2, p_vals_exp2, [m*n])

        ! Check the value was updated properly
        if (p_vals_exp2(4) /= vals_up(1)) then
            write (*,*) 'Test 2.3w failed in test_fortran_spmat_update_dense'
            stop 1
        end if

        call perflibs_spmat_export_deallocate_wrapper(c_vals_exp2)

    else

        ! Test 2.2: export matrix - export as col major just for fun
        call perflibs_spmat_export_dense(perflibs_mat, PERFLIBS_COL_MAJOR, m, n, vals_exp2, stat)
        if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
            write (*,*) 'Test 2.2 failed in test_fortran_spmat_update_dense'
            stop 1
        end if

        ! Check the value was updated properly
        if (vals_exp2(4) /= vals_up(1)) then
            write (*,*) 'Test 2.3 failed in test_fortran_spmat_update_dense'
            stop 1
        end if

        deallocate(vals_exp2)

    end if

    ! Destroy the thing
    call perflibs_spmat_destroy(perflibs_mat, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Failed destroy in test_fortran_spmat_update_dense'
        stop 1
    end if


    write (*,'(A,I1,A,I3)') "Success - test_fortran_spmat_update_dense, kind = ", rk, " ftype nbits: ", storage_size(dummy)

end program test_fortran_spmat_update_dense
