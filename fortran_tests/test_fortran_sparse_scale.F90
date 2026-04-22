! SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
! affiliates <open-source-office@arm.com></text>
!
! SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

program test_fortran_sparse_scale_exec
    use perflibs_sparse
    use iso_c_binding
    implicit none

    FTYPE :: dummy
    integer, parameter :: rk = kind(dummy)

    integer, parameter :: m_orig = 5
    integer, parameter :: n_orig = 4
    integer :: m
    integer :: n
    integer, parameter :: nnz = 6
    FTYPE, dimension(nnz) :: vals_a = (/ &
        1.0_rk, 2.0_rk, 3.0_rk, 4.0_rk, 5.0_rk, 6.0_rk /)
    integer, dimension(nnz) :: col_indx = (/ 1, 1, 1, 2, 3, 4 /)
    integer, dimension(m_orig + 1) :: row_ptr = (/ 1, 2, 3, 4, 5, 7 /)
    FTYPE, allocatable :: dense_a_alloc(:), result_alloc(:)
    FTYPE, dimension(m_orig * n_orig) :: reference
    real(kind=rk) :: op_tol_mod, eps, tol, diff

    FTYPE:: alpha = 5.0_rk
    integer :: stat, i
    integer :: flags = 0
    integer(kind=perflibs_i8) :: perflibs_mat_a
    logical :: is_complex

    integer :: libgfortran_mismatch
    external :: query_mismatch
    external :: perflibs_spmat_export_dense_wrapper_s
    external :: perflibs_spmat_export_dense_wrapper_d
    external :: perflibs_spmat_export_dense_wrapper_c
    external :: perflibs_spmat_export_dense_wrapper_z
    external :: perflibs_spmat_export_deallocate_wrapper
    type(c_ptr) :: dense_a_vals
    FTYPE, pointer :: dense_a_ptr(:)

    include 'type_query.h'

    if (typeid .eq. 1 .or. typeid .eq. 2) then
        is_complex = .false.
    else
        is_complex = .true.
    end if

    ! Create CSR matrix A
    call perflibs_spmat_create_csr(perflibs_mat_a, m_orig, n_orig, row_ptr, col_indx, vals_a, flags, stat)
    call check_stat("perflibs_spmat_create_csr")

    ! Calculate the reference result
    call get_reference()

    ! Do scaling
    call perflibs_spscale_exec(alpha, perflibs_mat_a, stat)
    call check_stat("perflibs_spscale_exec")

    ! Compare the result to the reference
    call compare_results()

    ! Free memory
    call perflibs_spmat_destroy(perflibs_mat_a, stat)

    write (*,'(A,I1,A,I3)') "Success - test_fortran_sparse_scale_exec, kind = ", rk, &
                            " ftype nbits: ", storage_size(dummy)

contains

    subroutine get_reference()
        ! Get dense vals arrays for input A
        call query_mismatch(libgfortran_mismatch)
        if (libgfortran_mismatch .eq. 1) then

            if (typeid .eq. 1) then
                call perflibs_spmat_export_dense_wrapper_s(perflibs_mat_a, perflibs_row_major, m, n, dense_a_vals, stat)
                call check_stat("perflibs_export_dense_s")
            else if (typeid .eq. 2) then
                call perflibs_spmat_export_dense_wrapper_d(perflibs_mat_a, perflibs_row_major, m, n, dense_a_vals, stat)
                call check_stat("perflibs_export_dense_d")
            else if (typeid .eq. 3) then
                call perflibs_spmat_export_dense_wrapper_c(perflibs_mat_a, perflibs_row_major, m, n, dense_a_vals, stat)
                call check_stat("perflibs_export_dense_c")
            else if (typeid .eq. 4) then
                call perflibs_spmat_export_dense_wrapper_z(perflibs_mat_a, perflibs_row_major, m, n, dense_a_vals, stat)
                call check_stat("perflibs_export_dense_z")
            end if

            call c_f_pointer(dense_a_vals, dense_a_ptr, [m*n])

            allocate(dense_a_alloc(m*n))
            dense_a_alloc(:) = dense_a_ptr(:)
            call perflibs_spmat_export_deallocate_wrapper(dense_a_vals)

        else

            call perflibs_spmat_export_dense(perflibs_mat_a, perflibs_row_major, m, n, dense_a_alloc, stat)
            call check_stat("perflibs_spmat_export_dense")

        end if

        ! Scale A by alpha
        do i = 1, m * n
            reference(i) = dense_a_alloc(i) * alpha
        end do

        deallocate(dense_a_alloc)
    end subroutine

    subroutine compare_results()
        call query_mismatch(libgfortran_mismatch)
        if (libgfortran_mismatch .eq. 1) then

            if (typeid .eq. 1) then
                call perflibs_spmat_export_dense_wrapper_s(perflibs_mat_a, perflibs_row_major, m, n, dense_a_vals, stat)
                call check_stat("perflibs_export_dense_s")
            else if (typeid .eq. 2) then
                call perflibs_spmat_export_dense_wrapper_d(perflibs_mat_a, perflibs_row_major, m, n, dense_a_vals, stat)
                call check_stat("perflibs_export_dense_d")
            else if (typeid .eq. 3) then
                call perflibs_spmat_export_dense_wrapper_c(perflibs_mat_a, perflibs_row_major, m, n, dense_a_vals, stat)
                call check_stat("perflibs_export_dense_c")
            else if (typeid .eq. 4) then
                call perflibs_spmat_export_dense_wrapper_z(perflibs_mat_a, perflibs_row_major, m, n, dense_a_vals, stat)
                call check_stat("perflibs_export_dense_z")
            end if

            call c_f_pointer(dense_a_vals, dense_a_ptr, [m*n])

            allocate(result_alloc(m*n))
            result_alloc(:) = dense_a_ptr(:)
            call perflibs_spmat_export_deallocate_wrapper(dense_a_vals)

        else
            call perflibs_spmat_export_dense(perflibs_mat_a, perflibs_row_major, m, n, result_alloc, stat)
            call check_stat("perflibs_export_dense")

        end if

        eps = epsilon(real(result_alloc(1)))
        if (is_complex) then
             op_tol_mod = 4.0_rk
        else
             op_tol_mod = 1.0_rk
        end if

        do i = 1, m * n
            tol = op_tol_mod * abs(reference(i)) * eps
            diff = abs(reference(i) - result_alloc(i))
            if (diff > tol) then
                stat = perflibs_status_execution_failure
                call check_stat("sparse scale failed tolerance check")
            end if
        end do

        deallocate(result_alloc)
    end subroutine


    subroutine check_stat(err_msg)
        character(len=*), intent(in) :: err_msg

        if (stat .ne. perflibs_status_success) then
            write (*,*) err_msg, ' failed in test_fortran_sparse_scale_exec'
            stop 1
        end if
    end subroutine check_stat

end program test_fortran_sparse_scale_exec
