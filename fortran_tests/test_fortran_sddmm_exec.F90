! SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
! affiliates <open-source-office@arm.com></text>
!
! SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

program test_fortran_sddmm_exec
    use perflibs_sparse
    use iso_c_binding
    implicit none

    FTYPE :: dummy
    integer, parameter :: rk = kind(dummy)

    integer, parameter :: m_orig = 5
    integer, parameter :: n_orig = 4
    integer, parameter :: k_orig = 4
    integer :: m
    integer :: n
    integer, parameter :: nnz = 6
    FTYPE, dimension(m_orig * k_orig) :: vals_a = (/ &
        1.0_rk, 2.0_rk, 3.0_rk, 4.0_rk, 5.0_rk, &
        6.0_rk, 7.0_rk, 8.0_rk, 9.0_rk, 10.0_rk, &
        11.0_rk, 12.0_rk, 13.0_rk, 14.0_rk, 15.0_rk, &
        16.0_rk, 17.0_rk, 18.0_rk, 19.0_rk, 20.0_rk /)
    FTYPE, dimension(k_orig * n_orig) :: vals_b = (/ &
        1.0_rk, 2.0_rk, 3.0_rk, 4.0_rk, 5.0_rk, &
        6.0_rk, 7.0_rk, 8.0_rk, 9.0_rk, 10.0_rk, &
        11.0_rk, 12.0_rk, 13.0_rk, 14.0_rk, 15.0_rk, &
        16.0_rk /)
    FTYPE, dimension(nnz) :: vals_c = (/ 1.0_rk, 2.0_rk, 3.0_rk, 4.0_rk, 5.0_rk, 6.0_rk /)
    integer, dimension(nnz) :: col_indx = (/ 1, 1, 1, 2, 3, 4 /)
    integer, dimension(m_orig + 1) :: row_ptr = (/ 1, 2, 3, 4, 5, 7 /)
    FTYPE, allocatable :: dense_ref_alloc(:), dense_c_alloc(:), result_alloc(:)
    FTYPE, dimension(m_orig * n_orig) :: reference
    real(kind=rk) :: op_tol_mod, eps, tol, diff

    FTYPE:: alpha = 1.0_rk, beta = 1.0_rk
    integer :: stat, i
    integer :: flags = 0
    integer(kind=perflibs_i8) :: perflibs_mat_a, perflibs_mat_b, perflibs_mat_c, perflibs_mat_ref
    logical :: is_complex

    integer :: libgfortran_mismatch
    external :: query_mismatch
    external :: perflibs_spmat_export_dense_wrapper_s
    external :: perflibs_spmat_export_dense_wrapper_d
    external :: perflibs_spmat_export_dense_wrapper_c
    external :: perflibs_spmat_export_dense_wrapper_z
    external :: perflibs_spmat_export_deallocate_wrapper
    type(c_ptr) :: dense_ref_vals, dense_c_vals
    FTYPE, pointer :: dense_ref_ptr(:), dense_c_ptr(:)

    include 'type_query.h'

    if (typeid .eq. 1 .or. typeid .eq. 2) then
        is_complex = .false.
    else
        is_complex = .true.
    end if

    ! Create dense and CSR matrices
    call perflibs_spmat_create_dense(perflibs_mat_a, perflibs_col_major, m_orig, k_orig, m_orig, vals_a, flags, stat)
    call check_stat("perflibs_spmat_create_dense")
    call perflibs_spmat_create_dense(perflibs_mat_b, perflibs_col_major, k_orig, n_orig, k_orig, vals_b, flags, stat)
    call check_stat("perflibs_spmat_create_dense")
    call perflibs_spmat_create_csr(perflibs_mat_c, m_orig, n_orig, row_ptr, col_indx, vals_c, flags, stat)
    call check_stat("perflibs_spmat_create_csr")
    call perflibs_spmat_create_csr(perflibs_mat_ref, m_orig, n_orig, row_ptr, col_indx, vals_c, flags, stat)
    call check_stat("perflibs_spmat_create_csr")

    ! Calculate the reference result
    call get_reference()

    ! Call SDDMM optimize function
    call perflibs_sddmm_optimize(perflibs_sparse_operation_notrans, perflibs_sparse_operation_notrans, &
                                 perflibs_sparse_scalar_one, perflibs_mat_a, perflibs_mat_b, &
                                 perflibs_sparse_scalar_one, perflibs_mat_c, stat)
    call check_stat("perflibs_sddmm_optimize")

    ! Do sampled dense-dense multiplication
    call perflibs_sddmm_exec(perflibs_sparse_operation_notrans, perflibs_sparse_operation_notrans, &
                             alpha, perflibs_mat_a, perflibs_mat_b, beta, perflibs_mat_c, stat)
    call check_stat("perflibs_sddmm_exec")

    ! Compare the result to the reference
    call compare_results()

    ! Free any memory
    call perflibs_spmat_destroy(perflibs_mat_a, stat)
    call perflibs_spmat_destroy(perflibs_mat_b, stat)
    call perflibs_spmat_destroy(perflibs_mat_c, stat)
    call perflibs_spmat_destroy(perflibs_mat_ref, stat)

    write (*,'(A,I1,A,I3)') "Success - test_fortran_sddmm_exec, kind = ", rk, &
                            " ftype nbits: ", storage_size(dummy)

contains

    subroutine get_reference()
        ! Compute alpha * (AB) + beta * C
        call perflibs_spmm_exec(perflibs_sparse_operation_notrans, perflibs_sparse_operation_notrans, &
                                alpha, perflibs_mat_a, perflibs_mat_b, beta, perflibs_mat_ref, stat)
        call check_stat("perflibs_spmm_exec")

        ! Get dense vals arrays for reference result and input C
        call query_mismatch(libgfortran_mismatch)
        if (libgfortran_mismatch .eq. 1) then

            if (typeid .eq. 1) then
                call perflibs_spmat_export_dense_wrapper_s(perflibs_mat_ref, perflibs_row_major, m, n, dense_ref_vals, stat)
                call check_stat("perflibs_spmat_export_dense_s")
                call perflibs_spmat_export_dense_wrapper_s(perflibs_mat_c, perflibs_row_major, m, n, dense_c_vals, stat)
                call check_stat("perflibs_export_dense_s")
            else if (typeid .eq. 2) then
                call perflibs_spmat_export_dense_wrapper_d(perflibs_mat_ref, perflibs_row_major, m, n, dense_ref_vals, stat)
                call check_stat("perflibs_spmat_export_dense_d")
                call perflibs_spmat_export_dense_wrapper_d(perflibs_mat_c, perflibs_row_major, m, n, dense_c_vals, stat)
                call check_stat("perflibs_export_dense_d")
            else if (typeid .eq. 3) then
                call perflibs_spmat_export_dense_wrapper_c(perflibs_mat_ref, perflibs_row_major, m, n, dense_ref_vals, stat)
                call check_stat("perflibs_spmat_export_dense_c")
                call perflibs_spmat_export_dense_wrapper_c(perflibs_mat_c, perflibs_row_major, m, n, dense_c_vals, stat)
                call check_stat("perflibs_export_dense_c")
            else if (typeid .eq. 4) then
                call perflibs_spmat_export_dense_wrapper_z(perflibs_mat_ref, perflibs_row_major, m, n, dense_ref_vals, stat)
                call check_stat("perflibs_spmat_export_dense_z")
                call perflibs_spmat_export_dense_wrapper_z(perflibs_mat_c, perflibs_row_major, m, n, dense_c_vals, stat)
                call check_stat("perflibs_export_dense_z")
            end if

            call c_f_pointer(dense_ref_vals, dense_ref_ptr, [m*n])
            call c_f_pointer(dense_c_vals, dense_c_ptr, [m*n])

            allocate(dense_ref_alloc(m*n))
            allocate(dense_c_alloc(m*n))
            dense_ref_alloc(:) = dense_ref_ptr(:)
            dense_c_alloc(:) = dense_c_ptr(:)
            call perflibs_spmat_export_deallocate_wrapper(dense_ref_vals)
            call perflibs_spmat_export_deallocate_wrapper(dense_c_vals)

        else

            call perflibs_spmat_export_dense(perflibs_mat_ref, perflibs_row_major, m, n, dense_ref_alloc, stat)
            call check_stat("perflibs_spmat_export_dense")
            call perflibs_spmat_export_dense(perflibs_mat_c, perflibs_row_major, m, n, dense_c_alloc, stat)
            call check_stat("perflibs_spmat_export_dense")

        end if

        ! Apply the sparsity structure of C to the reference
        do i = 1, m * n
            if (dense_c_alloc(i) .eq. 0) then
                reference(i) = 0
            else
                reference(i) = dense_ref_alloc(i)
            end if
        end do

        deallocate(dense_ref_alloc)
        deallocate(dense_c_alloc)
    end subroutine

    subroutine compare_results()
        call query_mismatch(libgfortran_mismatch)
        if (libgfortran_mismatch .eq. 1) then

            if (typeid .eq. 1) then
                call perflibs_spmat_export_dense_wrapper_s(perflibs_mat_c, perflibs_row_major, m, n, dense_c_vals, stat)
                call check_stat("perflibs_export_dense_s")
            else if (typeid .eq. 2) then
                call perflibs_spmat_export_dense_wrapper_d(perflibs_mat_c, perflibs_row_major, m, n, dense_c_vals, stat)
                call check_stat("perflibs_export_dense_d")
            else if (typeid .eq. 3) then
                call perflibs_spmat_export_dense_wrapper_c(perflibs_mat_c, perflibs_row_major, m, n, dense_c_vals, stat)
                call check_stat("perflibs_export_dense_c")
            else if (typeid .eq. 4) then
                call perflibs_spmat_export_dense_wrapper_z(perflibs_mat_c, perflibs_row_major, m, n, dense_c_vals, stat)
                call check_stat("perflibs_export_dense_z")
            end if

            call c_f_pointer(dense_c_vals, dense_c_ptr, [m*n])

            allocate(result_alloc(m*n))
            result_alloc(:) = dense_c_ptr(:)
            call perflibs_spmat_export_deallocate_wrapper(dense_c_vals)

        else
            call perflibs_spmat_export_dense(perflibs_mat_c, perflibs_row_major, m, n, result_alloc, stat)
            call check_stat("perflibs_export_dense")

        end if

        eps = epsilon(real(result_alloc(1)))
        if (is_complex) then
             op_tol_mod = 4.0_rk
        else
             op_tol_mod = 1.0_rk
        end if

        do i = 1, m * n
            tol = op_tol_mod * real(m) * real(n) * abs(reference(i)) * eps
            diff = abs(reference(i) - result_alloc(i))
            if (diff > tol) then
                stat = perflibs_status_execution_failure
                call check_stat("sddmm failed tolerance check")
            end if
        end do

        deallocate(result_alloc)
    end subroutine


    subroutine check_stat(err_msg)
        character(len=*), intent(in) :: err_msg

        if (stat .ne. perflibs_status_success) then
            write (*,*) err_msg, ' failed in test_fortran_sddmm_exec'
            stop 1
        end if
    end subroutine check_stat

end program test_fortran_sddmm_exec
