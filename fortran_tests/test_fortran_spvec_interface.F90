! SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
! affiliates <open-source-office@arm.com></text>
!
! SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

program test_fortran_spvec_interface
    use perflibs_sparse
    implicit none

    ! Use compiler flags to switch between 4, 8 byte reals and preprocessor to select FTYPE=real/complex type
    FTYPE, target :: dummy
    integer, parameter :: rk = kind(dummy)
    class(*), pointer :: dummy_ptr

    integer :: index_base = 1
    integer, parameter :: index_base_orig = 1
    integer :: n = 10
    integer, parameter :: n_orig = 10
    integer :: nnz = 6
    integer, parameter :: nnz_orig = 6
    integer :: n_updates = 2
    integer, parameter :: n_updates_orig = 2
    FTYPE, dimension(nnz_orig) :: vals
    FTYPE, dimension(n_updates_orig) :: update_vals
    FTYPE, dimension(n_orig) :: y, y_orig, y_ref
    real(kind=rk) :: eps, tol, op_tol_mod
    real(kind=rk), dimension(n_orig) :: tols
    integer, dimension(nnz_orig) :: indx = (/ 1, 2, 4, 7, 8, 9 /)
    integer, dimension(n_updates_orig) :: update_indx = (/ 2, 8 /)
    FTYPE, allocatable :: vals_alloc(:), x_d_orig(:), x_d_orig_updated(:), x_d(:)
    integer, allocatable :: indx_alloc(:)
    FTYPE :: alpha, beta

    integer :: stat, i, j, idx
    integer :: flags = 0
    integer(kind=perflibs_i8) :: perflibs_vec, perflibs_vec_orig

    logical :: failed, is_complex

    type, abstract :: dot_problem
    end type dot_problem

    type, extends(dot_problem) :: dot_problem_real
        real(kind=rk), dimension(n_orig) :: x, y
        real(kind=rk) :: res, res_ref
    end type dot_problem_real

    type, extends(dot_problem) :: dot_problem_complex
        complex(kind=rk), dimension(n_orig) :: x, y
        complex(kind=rk) :: res, res_ref
    end type dot_problem_complex

    class(dot_problem), allocatable :: dot

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
            complex(kind=rk), intent(out) :: sca1, sca2
        end subroutine populate_complex
    end interface populate

    dummy_ptr => dummy
    select type(dummy_ptr)
    type is (real)
        is_complex = .false.
        allocate(dot_problem_real::dot)
    type is (complex)
        is_complex = .true.
        allocate(dot_problem_complex::dot)
    class default
        stop 'Tests for this datatype not implemented'
    end select

    ! Populate arrays and set some scalars
    call populate(vals, y_orig, alpha, beta)
    y = y_orig
    do i = 1, n_updates_orig
        idx = 0
        do j = 1, nnz_orig
            if (indx(j) .eq. update_indx(i)) then
                idx = j
                exit
            end if
        end do
        update_vals(i) = 2.0_rk * vals(idx)
    end do

    allocate(x_d_orig(n))
    allocate(x_d_orig_updated(n))
    allocate(x_d(n))

    ! Populate original dense vector with non-zero values
    x_d_orig = 0.0_rk
    do i = 1, nnz
        x_d_orig(indx(i)) = vals(i)
    end do

    ! Update original dense vector and store in a separate array for use later
    x_d_orig_updated = x_d_orig
    do i = 1, n_updates_orig
        x_d_orig_updated(update_indx(i)) = update_vals(i)
    end do

    ! Setup testing for other routines
    call perflibs_spvec_create(perflibs_vec_orig, index_base, n, nnz, indx, vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Failed setup in test_fortran_spvec_interface'
        stop 1
    end if

    ! Test 1: n < 0 in spvec_create
    n = -1
    call perflibs_spvec_create(perflibs_vec, index_base, n, nnz, indx, vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 1 failed in test_fortran_spvec_interface'
        stop 1
    end if
    call cleanup(perflibs_vec)
    n = n_orig

    ! Test 2: nnz < 0 in spvec_create
    nnz = -1
    call perflibs_spvec_create(perflibs_vec, index_base, n, nnz, indx, vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 2 failed in test_fortran_spvec_interface'
        stop 1
    end if
    call cleanup(perflibs_vec)
    nnz = nnz_orig

    ! Test 3: index_base not 0 or 1 in spvec_create
    index_base = 55
    call perflibs_spvec_create(perflibs_vec, index_base, n, nnz, indx, vals, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 3 failed in test_fortran_spvec_interface'
        stop 1
    end if
    call cleanup(perflibs_vec)
    index_base = index_base_orig

    ! Test 4: perflibs_spvec_query always returns PERFLIBS_STATUS_SUCCESS, even with invalid vec
    call perflibs_spvec_query(perflibs_vec_orig, index_base, n, nnz, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Test 4 failed in test_fortran_spvec_interface'
        stop 1
    end if
    index_base = index_base_orig
    n = n_orig
    nnz = nnz_orig

    ! Test 5: perflibs_spvec_export returns expected results
    allocate(indx_alloc(nnz_orig))
    allocate(vals_alloc(nnz_orig))
    call perflibs_spvec_export(perflibs_vec_orig, index_base, n, nnz, indx_alloc, vals_alloc, stat)
    call check_params_and_vectors(failed)
    if (failed) then
        write (*,*) 'Test 5 failed in test_fortran_spvec_interface'
        stop 1
    end if
    deallocate(indx_alloc)
    deallocate(vals_alloc)

    ! Test 6: test invalid index_base in perflibs_spvec_gather
    index_base = -1
    call perflibs_spvec_gather(x_d_orig, index_base, n, perflibs_vec, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 6 failed in test_fortran_spvec_interface'
        stop 1
    end if
    call cleanup(perflibs_vec)
    index_base = index_base_orig

    ! Test 7: test n < 0 in perflibs_spvec_gather
    n = -1
    call perflibs_spvec_gather(x_d_orig, index_base, n, perflibs_vec, flags, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 7 failed in test_fortran_spvec_interface'
        stop 1
    end if
    call cleanup(perflibs_vec)
    n = n_orig

    ! Test 8: perflibs_spvec_gather returns expected results
    allocate(indx_alloc(nnz_orig))
    allocate(vals_alloc(nnz_orig))
    call perflibs_spvec_gather(x_d_orig, index_base, n, perflibs_vec, flags, stat)
    call perflibs_spvec_export(perflibs_vec, index_base, n, nnz, indx_alloc, vals_alloc, stat)
    call check_params_and_vectors(failed)
    if (failed) then
        write (*,*) 'Test 8 failed in test_fortran_spvec_interface'
        stop 1
    end if
    call cleanup(perflibs_vec)
    deallocate(indx_alloc)
    deallocate(vals_alloc)

    ! Test 9: perflibs_spvec_scatter returns expected results
    call perflibs_spvec_scatter(perflibs_vec_orig, x_d, stat)
    failed = stat .ne. PERFLIBS_STATUS_SUCCESS
    failed = failed .or. any(x_d .ne. x_d_orig)
    if (failed) then
        write (*,*) 'Test 9 failed in test_fortran_spvec_interface'
        stop 1
    end if

    ! Test 10: test n_updates < 0 in perflibs_spvec_update
    perflibs_vec = perflibs_vec_orig
    n_updates = -1
    call perflibs_spvec_update(perflibs_vec_orig, n_updates, update_indx, update_vals, stat)
    if (stat .ne. PERFLIBS_STATUS_INPUT_PARAMETER_ERROR) then
        write (*,*) 'Test 10 failed in test_fortran_spvec_interface'
        stop 1
    end if
    call perflibs_spvec_print_err(perflibs_vec_orig) ! Print the error (for coverage; not checked)
    n_updates = n_updates_orig

    ! Test 11: perflibs_spvec_update returns expected results
    call perflibs_spvec_gather(x_d_orig, index_base_orig, n_orig, perflibs_vec, flags, stat)
    call perflibs_spvec_update(perflibs_vec, n_updates, update_indx, update_vals, stat)
    failed = stat .ne. PERFLIBS_STATUS_SUCCESS
    call perflibs_spvec_scatter(perflibs_vec, x_d, stat)
    failed = failed .or. any(x_d .ne. x_d_orig_updated)
    if (failed) then
        write (*,*) 'Test 11 failed in test_fortran_spvec_interface'
        stop 1
    end if
    call cleanup(perflibs_vec)

    select type(dot)
    type is(dot_problem_real)
        ! TEST 12: perflibs_spdot_exec returns expected results
        dot%x = real(x_d_orig, kind=rk)
        dot%y = real(y, kind=rk)
        eps = epsilon(dot%res)
        dot%res_ref = sum(dot%x * dot%y)
        call perflibs_spdot_exec(perflibs_vec_orig, dot%y, dot%res, stat)
        tol = 2.0_rk * nnz_orig * eps * abs(dot%res_ref)
        failed = (stat .ne. PERFLIBS_STATUS_SUCCESS) .or. (abs(dot%res - dot%res_ref) > tol)
        if (failed) then
            write (*,*) 'Test 12 failed in test_fortran_spvec_interface'
            stop 1
        end if
    type is(dot_problem_complex)
        ! TEST 12.1: perflibs_spdotu_exec returns expected results
        dot%x = x_d_orig
        dot%y = y
        eps = epsilon(real(dot%res))
        dot%res_ref = sum(dot%x * dot%y)
        call perflibs_spdotu_exec(perflibs_vec_orig, dot%y, dot%res, stat)
        tol = 8.0_rk * nnz_orig * eps * abs(dot%res_ref)
        failed = (stat .ne. PERFLIBS_STATUS_SUCCESS) .or. (abs(dot%res - dot%res_ref) > tol)
        if (failed) then
            write (*,*) 'Test 12.1 failed in test_fortran_spvec_interface'
            stop 1
        end if

        ! TEST 12.2: perflibs_spdotc_exec returns expected results
        dot%res_ref = sum(conjg(dot%x) * dot%y)
        call perflibs_spdotc_exec(perflibs_vec_orig, dot%y, dot%res, stat)
        tol = 8.0_rk * nnz_orig * eps * abs(dot%res_ref)
        failed = (stat .ne. PERFLIBS_STATUS_SUCCESS) .or. (abs(dot%res - dot%res_ref) > tol)
        if (failed) then
            write (*,*) 'Test 12.2 failed in test_fortran_spvec_interface'
            stop 1
        end if
    class default
        stop 'Dot problem is neither dot_problem_real or dot_problem_complex'
    end select
    deallocate(dot)

    ! TEST 13: perflibs_spaxpby_exec returns expected results
    call perflibs_spaxpby_exec(alpha, perflibs_vec_orig, beta, y, stat)
    y_ref = alpha * x_d_orig + beta * y_orig
    if (is_complex) then
        op_tol_mod = 14.0_rk
    else
        op_tol_mod = 3.0_rk
    end if
    tols = op_tol_mod * eps * abs(y_ref)
    failed = (stat .ne. PERFLIBS_STATUS_SUCCESS) .or. any(abs(y-y_ref) > tols)
    if (failed) then
        write (*,*) 'Test 13 failed in test_fortran_spvec_interface'
        stop 1
    end if
    y = y_orig

    ! TEST 14: perflibs_spwaxpby_exec returns expected results
    call perflibs_spwaxpby_exec(alpha, perflibs_vec_orig, beta, y_orig, y, stat)
    y_ref = alpha * x_d_orig + beta * y_orig
    if (is_complex) then
        op_tol_mod = 14.0_rk
    else
        op_tol_mod = 3.0_rk
    end if
    tols = op_tol_mod * eps * abs(y_ref)
    failed = (stat .ne. PERFLIBS_STATUS_SUCCESS) .or. any(abs(y-y_ref) > tols)
    if (failed) then
        write (*,*) 'Test 14 failed in test_fortran_spvec_interface'
        stop 1
    end if
    y = y_orig

    ! Final cleanup
    call cleanup(perflibs_vec_orig)

    deallocate(x_d_orig)
    deallocate(x_d_orig_updated)
    deallocate(x_d)

    write (*,'(A,I1,A,I3)') "Success - test_fortran_spvec_interface, kind = ", rk, " ftype nbits: ", storage_size(dummy)

contains

    subroutine check_params_and_vectors(failed)

        logical, intent(out) :: failed

        failed = stat .ne. PERFLIBS_STATUS_SUCCESS
        failed = failed .or. (index_base .ne. index_base_orig)
        failed = failed .or. (n .ne. n_orig)
        failed = failed .or. (nnz .ne. nnz_orig)
        failed = failed .or. any(indx_alloc .ne. indx)
        failed = failed .or. any(vals_alloc .ne. vals)

    end subroutine check_params_and_vectors


end program test_fortran_spvec_interface

subroutine cleanup(perflibs_mat)
    use perflibs_sparse
    integer(kind=perflibs_i8) :: perflibs_mat
    integer :: stat
    ! Destroy the thing
    call perflibs_spvec_destroy(perflibs_mat, stat)
    if (stat .ne. PERFLIBS_STATUS_SUCCESS) then
        write (*,*) 'Failed destroy in test_fortran_spvec_interface'
        stop 1
    end if
end subroutine cleanup

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
    sca1 = real(1.1, kind=rk)
    sca2 = real(1.3, kind=rk)

end subroutine populate_real

subroutine populate_complex(arr1, arr2, sca1, sca2)

    implicit none

    FTYPE, target :: dummy
    integer, parameter :: rk = kind(dummy)
    complex(kind=rk), dimension(:), intent(out) :: arr1, arr2
    complex(kind=rk), intent(out) :: sca1, sca2
    integer :: i

    do i = 1, size(arr1, 1)
        arr1(i) = cmplx(i, i+1, kind=rk)
    end do
    do i = 1, size(arr2, 1)
        arr2(i) = cmplx(2*i, 2*i+1, kind=rk)
    end do
    sca1 = cmplx(1.1, 1.2, kind=rk)
    sca2 = cmplx(1.3, 1.4, kind=rk)

end subroutine populate_complex
