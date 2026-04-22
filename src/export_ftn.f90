! SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
! affiliates <open-source-office@arm.com></text>
!
! SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

module perflibs_csr_functions
    interface
        subroutine perflibs_csr_param_check_s(A, impl_A, index_base, m, n, nnz, index_base_a, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: index_base_a
            integer, intent(out) :: info
        end subroutine perflibs_csr_param_check_s
    end interface

    interface
        subroutine perflibs_csr_populate_arrays_s(A, impl_A, nrows, nnz, index_base_a, index_base, row_ptr, col_indx, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer, intent(in) :: nrows
            integer, intent(in) :: nnz
            integer, intent(in) :: index_base_a
            integer, intent(in) :: index_base
            integer, intent(out) :: row_ptr(nrows+1)
            integer, intent(out) :: col_indx(nnz)
            real(kind=perflibs_r32), intent(out) :: vals(nnz)
        end subroutine perflibs_csr_populate_arrays_s
    end interface

    interface
        subroutine perflibs_csr_param_check_d(A, impl_A, index_base, m, n, nnz, index_base_a, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: index_base_a
            integer, intent(out) :: info
        end subroutine perflibs_csr_param_check_d
    end interface

    interface
        subroutine perflibs_csr_populate_arrays_d(A, impl_A, nrows, nnz, index_base_a, index_base, row_ptr, col_indx, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer, intent(in) :: nrows
            integer, intent(in) :: nnz
            integer, intent(in) :: index_base_a
            integer, intent(in) :: index_base
            integer, intent(out) :: row_ptr(nrows+1)
            integer, intent(out) :: col_indx(nnz)
            real(kind=perflibs_r64), intent(out) :: vals(nnz)
        end subroutine perflibs_csr_populate_arrays_d
    end interface

    interface
        subroutine perflibs_csr_param_check_c(A, impl_A, index_base, m, n, nnz, index_base_a, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: index_base_a
            integer, intent(out) :: info
        end subroutine perflibs_csr_param_check_c
    end interface

    interface
        subroutine perflibs_csr_populate_arrays_c(A, impl_A, nrows, nnz, index_base_a, index_base, row_ptr, col_indx, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer, intent(in) :: nrows
            integer, intent(in) :: nnz
            integer, intent(in) :: index_base_a
            integer, intent(in) :: index_base
            integer, intent(out) :: row_ptr(nrows+1)
            integer, intent(out) :: col_indx(nnz)
            complex(kind=perflibs_r32), intent(out) :: vals(nnz)
        end subroutine perflibs_csr_populate_arrays_c
    end interface

    interface
        subroutine perflibs_csr_param_check_z(A, impl_A, index_base, m, n, nnz, index_base_a, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: index_base_a
            integer, intent(out) :: info
        end subroutine perflibs_csr_param_check_z
    end interface

    interface
        subroutine perflibs_csr_populate_arrays_z(A, impl_A, nrows, nnz, index_base_a, index_base, row_ptr, col_indx, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer, intent(in) :: nrows
            integer, intent(in) :: nnz
            integer, intent(in) :: index_base_a
            integer, intent(in) :: index_base
            integer, intent(out) :: row_ptr(nrows+1)
            integer, intent(out) :: col_indx(nnz)
            complex(kind=perflibs_r64), intent(out) :: vals(nnz)
        end subroutine perflibs_csr_populate_arrays_z
    end interface
end module perflibs_csr_functions

module perflibs_csc_functions
    interface
        subroutine perflibs_csc_param_check_s(A, impl_A, index_base, m, n, nnz, index_base_a, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: index_base_a
            integer, intent(out) :: info
        end subroutine perflibs_csc_param_check_s
    end interface

    interface
        subroutine perflibs_csc_populate_arrays_s(A, impl_A, ncols, nnz, index_base_a, index_base, row_indx, col_ptr, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer, intent(in) :: ncols
            integer, intent(in) :: nnz
            integer, intent(in) :: index_base_a
            integer, intent(in) :: index_base
            integer, intent(out) :: row_indx(nnz)
            integer, intent(out) :: col_ptr(ncols+1)
            real(kind=perflibs_r32), intent(out) :: vals(nnz)
        end subroutine perflibs_csc_populate_arrays_s
    end interface

    interface
        subroutine perflibs_csc_param_check_d(A, impl_A, index_base, m, n, nnz, index_base_a, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: index_base_a
            integer, intent(out) :: info
        end subroutine perflibs_csc_param_check_d
    end interface

    interface
        subroutine perflibs_csc_populate_arrays_d(A, impl_A, ncols, nnz, index_base_a, index_base, row_indx, col_ptr, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer, intent(in) :: ncols
            integer, intent(in) :: nnz
            integer, intent(in) :: index_base_a
            integer, intent(in) :: index_base
            integer, intent(out) :: row_indx(nnz)
            integer, intent(out) :: col_ptr(ncols+1)
            real(kind=perflibs_r64), intent(out) :: vals(nnz)
        end subroutine perflibs_csc_populate_arrays_d
    end interface

    interface
        subroutine perflibs_csc_param_check_c(A, impl_A, index_base, m, n, nnz, index_base_a, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: index_base_a
            integer, intent(out) :: info
        end subroutine perflibs_csc_param_check_c
    end interface

    interface
        subroutine perflibs_csc_populate_arrays_c(A, impl_A, ncols, nnz, index_base_a, index_base, row_indx, col_ptr, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer, intent(in) :: ncols
            integer, intent(in) :: nnz
            integer, intent(in) :: index_base_a
            integer, intent(in) :: index_base
            integer, intent(out) :: row_indx(nnz)
            integer, intent(out) :: col_ptr(ncols+1)
            complex(kind=perflibs_r32), intent(out) :: vals(nnz)
        end subroutine perflibs_csc_populate_arrays_c
    end interface

    interface
        subroutine perflibs_csc_param_check_z(A, impl_A, index_base, m, n, nnz, index_base_a, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: index_base_a
            integer, intent(out) :: info
        end subroutine perflibs_csc_param_check_z
    end interface

    interface
        subroutine perflibs_csc_populate_arrays_z(A, impl_A, ncols, nnz, index_base_a, index_base, row_indx, col_ptr, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer, intent(in) :: ncols
            integer, intent(in) :: nnz
            integer, intent(in) :: index_base_a
            integer, intent(in) :: index_base
            integer, intent(out) :: row_indx(nnz)
            integer, intent(out) :: col_ptr(ncols+1)
            complex(kind=perflibs_r64), intent(out) :: vals(nnz)
        end subroutine perflibs_csc_populate_arrays_z
    end interface
end module perflibs_csc_functions

module perflibs_coo_functions
    interface
        subroutine perflibs_coo_param_check_s(A, impl_A, m, n, nnz, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: info
        end subroutine perflibs_coo_param_check_s
    end interface

    interface
        subroutine perflibs_coo_populate_arrays_s(A, impl_A, nnz, row_indx, col_indx, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer, intent(in) :: nnz
            integer, intent(out) :: row_indx(nnz)
            integer, intent(out) :: col_indx(nnz)
            real(kind=perflibs_r32), intent(out) :: vals(nnz)
        end subroutine perflibs_coo_populate_arrays_s
    end interface

    interface
        subroutine perflibs_coo_param_check_d(A, impl_A, m, n, nnz, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: info
        end subroutine perflibs_coo_param_check_d
    end interface

    interface
        subroutine perflibs_coo_populate_arrays_d(A, impl_A, nnz, row_indx, col_indx, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer, intent(in) :: nnz
            integer, intent(out) :: row_indx(nnz)
            integer, intent(out) :: col_indx(nnz)
            real(kind=perflibs_r64), intent(out) :: vals(nnz)
        end subroutine perflibs_coo_populate_arrays_d
    end interface

    interface
        subroutine perflibs_coo_param_check_c(A, impl_A, m, n, nnz, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: info
        end subroutine perflibs_coo_param_check_c
    end interface

    interface
        subroutine perflibs_coo_populate_arrays_c(A, impl_A, nnz, row_indx, col_indx, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer, intent(in) :: nnz
            integer, intent(out) :: row_indx(nnz)
            integer, intent(out) :: col_indx(nnz)
            complex(kind=perflibs_r32), intent(out) :: vals(nnz)
        end subroutine perflibs_coo_populate_arrays_c
    end interface

    interface
        subroutine perflibs_coo_param_check_z(A, impl_A, m, n, nnz, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: info
        end subroutine perflibs_coo_param_check_z
    end interface

    interface
        subroutine perflibs_coo_populate_arrays_z(A, impl_A, nnz, row_indx, col_indx, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer, intent(in) :: nnz
            integer, intent(out) :: row_indx(nnz)
            integer, intent(out) :: col_indx(nnz)
            complex(kind=perflibs_r64), intent(out) :: vals(nnz)
        end subroutine perflibs_coo_populate_arrays_z
    end interface
end module perflibs_coo_functions

module perflibs_dense_functions
    interface
        subroutine perflibs_dense_param_check_s(A, impl_A, layout, m, n, lda, size, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer(kind=perflibs_i4), intent(in) :: layout
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: lda
            integer, intent(out) :: size
            integer, intent(out) :: info
        end subroutine perflibs_dense_param_check_s
    end interface

    interface
        subroutine perflibs_dense_populate_arrays_s(A, impl_A, layout, nrows, ncols, lda, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer(kind=perflibs_i4), intent(in) :: layout
            integer, intent(in) :: nrows
            integer, intent(in) :: ncols
            integer, intent(in) :: lda
            real(kind=perflibs_r32), intent(out) :: vals(*)
        end subroutine perflibs_dense_populate_arrays_s
    end interface

    interface
        subroutine perflibs_dense_param_check_d(A, impl_A, layout, m, n, lda, size, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer(kind=perflibs_i4), intent(in) :: layout
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: lda
            integer, intent(out) :: size
            integer, intent(out) :: info
        end subroutine perflibs_dense_param_check_d
    end interface

    interface
        subroutine perflibs_dense_populate_arrays_d(A, impl_A, layout, nrows, ncols, lda, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer(kind=perflibs_i4), intent(in) :: layout
            integer, intent(in) :: nrows
            integer, intent(in) :: ncols
            integer, intent(in) :: lda
            real(kind=perflibs_r64), intent(out) :: vals(*)
        end subroutine perflibs_dense_populate_arrays_d
    end interface

    interface
        subroutine perflibs_dense_param_check_c(A, impl_A, layout, m, n, lda, size, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer(kind=perflibs_i4), intent(in) :: layout
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: lda
            integer, intent(out) :: size
            integer, intent(out) :: info
        end subroutine perflibs_dense_param_check_c
    end interface

    interface
        subroutine perflibs_dense_populate_arrays_c(A, impl_A, layout, nrows, ncols, lda, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer(kind=perflibs_i4), intent(in) :: layout
            integer, intent(in) :: nrows
            integer, intent(in) :: ncols
            integer, intent(in) :: lda
            complex(kind=perflibs_r32), intent(out) :: vals(*)
        end subroutine perflibs_dense_populate_arrays_c
    end interface

    interface
        subroutine perflibs_dense_param_check_z(A, impl_A, layout, m, n, lda, size, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer(kind=perflibs_i4), intent(in) :: layout
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: lda
            integer, intent(out) :: size
            integer, intent(out) :: info
        end subroutine perflibs_dense_param_check_z
    end interface

    interface
        subroutine perflibs_dense_populate_arrays_z(A, impl_A, layout, nrows, ncols, lda, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer(kind=perflibs_i4), intent(in) :: layout
            integer, intent(in) :: nrows
            integer, intent(in) :: ncols
            integer, intent(in) :: lda
            complex(kind=perflibs_r64), intent(out) :: vals(*)
        end subroutine perflibs_dense_populate_arrays_z
    end interface
end module perflibs_dense_functions

module perflibs_bsr_functions
    interface
        subroutine perflibs_bsr_param_check_s(A, impl_A, block_layout, index_base, m, n, block_size, &
                                              nnzb, nrowsb, nnz, index_base_A, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer(kind=perflibs_i4), intent(in) :: block_layout
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: block_size
            integer, intent(out) :: nnzb
            integer, intent(out) :: nrowsb
            integer, intent(out) :: nnz
            integer, intent(out) :: index_base_A
            integer, intent(out) :: info
        end subroutine perflibs_bsr_param_check_s
    end interface

    interface
        subroutine perflibs_bsr_populate_arrays_s(A, impl_A, block_layout, index_base_A, index_base, &
                                                  block_size, nnz, nnzb, nrowsb, row_ptr, col_indx, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer(kind=perflibs_i4), intent(in) :: block_layout
            integer, intent(in) :: index_base_A
            integer, intent(in) :: index_base
            integer, intent(in) :: block_size
            integer, intent(in) :: nnz
            integer, intent(in) :: nnzb
            integer, intent(in) :: nrowsb
            integer, intent(out) :: row_ptr(nrowsb+1)
            integer, intent(out) :: col_indx(nnzb)
            real(kind=perflibs_r32), intent(out) :: vals(nnz)
        end subroutine perflibs_bsr_populate_arrays_s
    end interface

    interface
        subroutine perflibs_bsr_param_check_d(A, impl_A, block_layout, index_base, m, n, block_size, &
                                              nnzb, nrowsb, nnz, index_base_A, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer(kind=perflibs_i4), intent(in) :: block_layout
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: block_size
            integer, intent(out) :: nnzb
            integer, intent(out) :: nrowsb
            integer, intent(out) :: nnz
            integer, intent(out) :: index_base_A
            integer, intent(out) :: info
        end subroutine perflibs_bsr_param_check_d
    end interface

    interface
        subroutine perflibs_bsr_populate_arrays_d(A, impl_A, block_layout, index_base_A, index_base, &
                                                  block_size, nnz, nnzb, nrowsb, row_ptr, col_indx, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer(kind=perflibs_i4), intent(in) :: block_layout
            integer, intent(in) :: index_base_A
            integer, intent(in) :: index_base
            integer, intent(in) :: block_size
            integer, intent(in) :: nnz
            integer, intent(in) :: nnzb
            integer, intent(in) :: nrowsb
            integer, intent(out) :: row_ptr(nrowsb+1)
            integer, intent(out) :: col_indx(nnzb)
            real(kind=perflibs_r64), intent(out) :: vals(nnz)
        end subroutine perflibs_bsr_populate_arrays_d
    end interface

    interface
        subroutine perflibs_bsr_param_check_c(A, impl_A, block_layout, index_base, m, n, block_size, &
                                              nnzb, nrowsb, nnz, index_base_A, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer(kind=perflibs_i4), intent(in) :: block_layout
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: block_size
            integer, intent(out) :: nnzb
            integer, intent(out) :: nrowsb
            integer, intent(out) :: nnz
            integer, intent(out) :: index_base_A
            integer, intent(out) :: info
        end subroutine perflibs_bsr_param_check_c
    end interface

    interface
        subroutine perflibs_bsr_populate_arrays_c(A, impl_A, block_layout, index_base_A, index_base, &
                                                  block_size, nnz, nnzb, nrowsb, row_ptr, col_indx, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer(kind=perflibs_i4), intent(in) :: block_layout
            integer, intent(in) :: index_base_A
            integer, intent(in) :: index_base
            integer, intent(in) :: block_size
            integer, intent(in) :: nnz
            integer, intent(in) :: nnzb
            integer, intent(in) :: nrowsb
            integer, intent(out) :: row_ptr(nrowsb+1)
            integer, intent(out) :: col_indx(nnzb)
            complex(kind=perflibs_r32), intent(out) :: vals(nnz)
        end subroutine perflibs_bsr_populate_arrays_c
    end interface

    interface
        subroutine perflibs_bsr_param_check_z(A, impl_A, block_layout, index_base, m, n, block_size, &
                                              nnzb, nrowsb, nnz, index_base_A, info)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(out) :: impl_A
            integer(kind=perflibs_i4), intent(in) :: block_layout
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: block_size
            integer, intent(out) :: nnzb
            integer, intent(out) :: nrowsb
            integer, intent(out) :: nnz
            integer, intent(out) :: index_base_A
            integer, intent(out) :: info
        end subroutine perflibs_bsr_param_check_z
    end interface

    interface
        subroutine perflibs_bsr_populate_arrays_z(A, impl_A, block_layout, index_base_A, index_base, &
                                                  block_size, nnz, nnzb, nrowsb, row_ptr, col_indx, vals)
            use perflibs_kinds
            implicit none
            integer(kind=perflibs_i8), intent(in) :: A
            integer(kind=perflibs_i8), intent(in) :: impl_A
            integer(kind=perflibs_i4), intent(in) :: block_layout
            integer, intent(in) :: index_base_A
            integer, intent(in) :: index_base
            integer, intent(in) :: block_size
            integer, intent(in) :: nnz
            integer, intent(in) :: nnzb
            integer, intent(in) :: nrowsb
            integer, intent(out) :: row_ptr(nrowsb+1)
            integer, intent(out) :: col_indx(nnzb)
            complex(kind=perflibs_r64), intent(out) :: vals(nnz)
        end subroutine perflibs_bsr_populate_arrays_z
    end interface
end module

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!  CSR Functions
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine perflibs_spmat_export_csr_s(A, index_base, m, n, row_ptr, col_indx, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_csr_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer, intent(in) :: index_base
    integer, intent(out) :: m
    integer, intent(out) :: n
    integer, allocatable, intent(out) :: row_ptr(:)
    integer, allocatable, intent(out) :: col_indx(:)
    real(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A
    integer :: index_base_a, nnz

    call perflibs_csr_param_check_s(A, impl_A, index_base, m, n, nnz, index_base_a, info)

    if  (allocated(row_ptr) .or. allocated(col_indx) .or. allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(row_ptr(m+1))
    allocate(col_indx(nnz))
    allocate(vals(nnz))

    call perflibs_csr_populate_arrays_s(A, impl_A, m, nnz, index_base_a, index_base, row_ptr, col_indx, vals)
end subroutine perflibs_spmat_export_csr_s

subroutine perflibs_spmat_export_csr_d(A, index_base, m, n, row_ptr, col_indx, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_csr_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer, intent(in) :: index_base
    integer, intent(out) :: m
    integer, intent(out) :: n
    integer, allocatable, intent(out) :: row_ptr(:)
    integer, allocatable, intent(out) :: col_indx(:)
    real(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A
    integer :: index_base_a, nnz

    call perflibs_csr_param_check_d(A, impl_A, index_base, m, n, nnz, index_base_a, info)

    if  (allocated(row_ptr) .or. allocated(col_indx) .or. allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(row_ptr(m+1))
    allocate(col_indx(nnz))
    allocate(vals(nnz))

    call perflibs_csr_populate_arrays_d(A, impl_A, m, nnz, index_base_a, index_base, row_ptr, col_indx, vals)
end subroutine perflibs_spmat_export_csr_d

subroutine perflibs_spmat_export_csr_c(A, index_base, m, n, row_ptr, col_indx, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_csr_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer, intent(in) :: index_base
    integer, intent(out) :: m
    integer, intent(out) :: n
    integer, allocatable, intent(out) :: row_ptr(:)
    integer, allocatable, intent(out) :: col_indx(:)
    complex(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A
    integer :: index_base_a, nnz

    call perflibs_csr_param_check_c(A, impl_A, index_base, m, n, nnz, index_base_a, info)

    if  (allocated(row_ptr) .or. allocated(col_indx) .or. allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(row_ptr(m+1))
    allocate(col_indx(nnz))
    allocate(vals(nnz))

    call perflibs_csr_populate_arrays_c(A, impl_A, m, nnz, index_base_a, index_base, row_ptr, col_indx, vals)
end subroutine perflibs_spmat_export_csr_c

subroutine perflibs_spmat_export_csr_z(A, index_base, m, n, row_ptr, col_indx, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_csr_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer, intent(in) :: index_base
    integer, intent(out) :: m
    integer, intent(out) :: n
    integer, allocatable, intent(out) :: row_ptr(:)
    integer, allocatable, intent(out) :: col_indx(:)
    complex(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A
    integer :: index_base_a, nnz

    call perflibs_csr_param_check_z(A, impl_A, index_base, m, n, nnz, index_base_a, info)

    if  (allocated(row_ptr) .or. allocated(col_indx) .or. allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(row_ptr(m+1))
    allocate(col_indx(nnz))
    allocate(vals(nnz))

    call perflibs_csr_populate_arrays_z(A, impl_A, m, nnz, index_base_a, index_base, row_ptr, col_indx, vals)
end subroutine perflibs_spmat_export_csr_z

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!  CSC Functions
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine perflibs_spmat_export_csc_s(A, index_base, m, n, row_indx, col_ptr, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_csc_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer, intent(in) :: index_base
    integer, intent(out) :: m
    integer, intent(out) :: n
    integer, allocatable, intent(out) :: row_indx(:)
    integer, allocatable, intent(out) :: col_ptr(:)
    real(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A
    integer :: index_base_a, nnz

    call perflibs_csc_param_check_s(A, impl_A, index_base, m, n, nnz, index_base_a, info)

    if  (allocated(row_indx) .or. allocated(col_ptr) .or. allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(row_indx(nnz))
    allocate(col_ptr(n+1))
    allocate(vals(nnz))

    call perflibs_csc_populate_arrays_s(A, impl_A, n, nnz, index_base_a, index_base, row_indx, col_ptr, vals)
end subroutine perflibs_spmat_export_csc_s

subroutine perflibs_spmat_export_csc_d(A, index_base, m, n, row_indx, col_ptr, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_csc_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer, intent(in) :: index_base
    integer, intent(out) :: m
    integer, intent(out) :: n
    integer, allocatable, intent(out) :: row_indx(:)
    integer, allocatable, intent(out) :: col_ptr(:)
    real(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A
    integer :: index_base_a, nnz

    call perflibs_csc_param_check_d(A, impl_A, index_base, m, n, nnz, index_base_a, info)

    if  (allocated(row_indx) .or. allocated(col_ptr) .or. allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(row_indx(nnz))
    allocate(col_ptr(n+1))
    allocate(vals(nnz))

    call perflibs_csc_populate_arrays_d(A, impl_A, n, nnz, index_base_a, index_base, row_indx, col_ptr, vals)
end subroutine perflibs_spmat_export_csc_d

subroutine perflibs_spmat_export_csc_c(A, index_base, m, n, row_indx, col_ptr, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_csc_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer, intent(in) :: index_base
    integer, intent(out) :: m
    integer, intent(out) :: n
    integer, allocatable, intent(out) :: row_indx(:)
    integer, allocatable, intent(out) :: col_ptr(:)
    complex(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A
    integer :: index_base_a, nnz

    call perflibs_csc_param_check_c(A, impl_A, index_base, m, n, nnz, index_base_a, info)

    if  (allocated(row_indx) .or. allocated(col_ptr) .or. allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(row_indx(nnz))
    allocate(col_ptr(n+1))
    allocate(vals(nnz))

    call perflibs_csc_populate_arrays_c(A, impl_A, n, nnz, index_base_a, index_base, row_indx, col_ptr, vals)
end subroutine perflibs_spmat_export_csc_c

subroutine perflibs_spmat_export_csc_z(A, index_base, m, n, row_indx, col_ptr, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_csc_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer, intent(in) :: index_base
    integer, intent(out) :: m
    integer, intent(out) :: n
    integer, allocatable, intent(out) :: row_indx(:)
    integer, allocatable, intent(out) :: col_ptr(:)
    complex(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A
    integer :: index_base_a, nnz

    call perflibs_csc_param_check_z(A, impl_A, index_base, m, n, nnz, index_base_a, info)

    if  (allocated(row_indx) .or. allocated(col_ptr) .or. allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(row_indx(nnz))
    allocate(col_ptr(n+1))
    allocate(vals(nnz))

    call perflibs_csc_populate_arrays_z(A, impl_A, n, nnz, index_base_a, index_base, row_indx, col_ptr, vals)
end subroutine perflibs_spmat_export_csc_z

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!  COO Functions
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine perflibs_spmat_export_coo_s(A, m, n, nnz, row_indx, col_indx, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_coo_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer, intent(out) :: m
    integer, intent(out) :: n
    integer, intent(out) :: nnz
    integer, allocatable, intent(out) :: row_indx(:)
    integer, allocatable, intent(out) :: col_indx(:)
    real(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A

    call perflibs_coo_param_check_s(A, impl_A, m, n, nnz, info)

    if  (allocated(row_indx) .or. allocated(col_indx) .or. allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(row_indx(nnz))
    allocate(col_indx(nnz))
    allocate(vals(nnz))

    call perflibs_coo_populate_arrays_s(A, impl_A, nnz, row_indx, col_indx, vals)
end subroutine perflibs_spmat_export_coo_s

subroutine perflibs_spmat_export_coo_d(A, m, n, nnz, row_indx, col_indx, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_coo_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer, intent(out) :: m
    integer, intent(out) :: n
    integer, intent(out) :: nnz
    integer, allocatable, intent(out) :: row_indx(:)
    integer, allocatable, intent(out) :: col_indx(:)
    real(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A

    call perflibs_coo_param_check_d(A, impl_A, m, n, nnz, info)

    if  (allocated(row_indx) .or. allocated(col_indx) .or. allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(row_indx(nnz))
    allocate(col_indx(nnz))
    allocate(vals(nnz))

    call perflibs_coo_populate_arrays_d(A, impl_A, nnz, row_indx, col_indx, vals)
end subroutine perflibs_spmat_export_coo_d

subroutine perflibs_spmat_export_coo_c(A, m, n, nnz, row_indx, col_indx, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_coo_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer, intent(out) :: m
    integer, intent(out) :: n
    integer, intent(out) :: nnz
    integer, allocatable, intent(out) :: row_indx(:)
    integer, allocatable, intent(out) :: col_indx(:)
    complex(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A

    call perflibs_coo_param_check_c(A, impl_A, m, n, nnz, info)

    if  (allocated(row_indx) .or. allocated(col_indx) .or. allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(row_indx(nnz))
    allocate(col_indx(nnz))
    allocate(vals(nnz))

    call perflibs_coo_populate_arrays_c(A, impl_A, nnz, row_indx, col_indx, vals)
end subroutine perflibs_spmat_export_coo_c

subroutine perflibs_spmat_export_coo_z(A, m, n, nnz, row_indx, col_indx, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_coo_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer, intent(out) :: m
    integer, intent(out) :: n
    integer, intent(out) :: nnz
    integer, allocatable, intent(out) :: row_indx(:)
    integer, allocatable, intent(out) :: col_indx(:)
    complex(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A

    call perflibs_coo_param_check_z(A, impl_A, m, n, nnz, info)

    if  (allocated(row_indx) .or. allocated(col_indx) .or. allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(row_indx(nnz))
    allocate(col_indx(nnz))
    allocate(vals(nnz))

    call perflibs_coo_populate_arrays_z(A, impl_A, nnz, row_indx, col_indx, vals)
end subroutine perflibs_spmat_export_coo_z


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!  Dense Functions
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine perflibs_spmat_export_dense_s(A, layout, m, n, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_dense_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer(kind=perflibs_i4), intent(in) :: layout
    integer, intent(out) :: m
    integer, intent(out) :: n
    real(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A
    integer :: lda, size

    call perflibs_dense_param_check_s(A, impl_A, layout, m, n, lda, size, info)

    if  (allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(vals(size))

    call perflibs_dense_populate_arrays_s(A, impl_A, layout, m, n, lda, vals)
end subroutine perflibs_spmat_export_dense_s

subroutine perflibs_spmat_export_dense_d(A, layout, m, n, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_dense_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer(kind=perflibs_i4), intent(in) :: layout
    integer, intent(out) :: m
    integer, intent(out) :: n
    real(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A
    integer :: lda, size

    call perflibs_dense_param_check_d(A, impl_A, layout, m, n, lda, size, info)

    if  (allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(vals(size))

    call perflibs_dense_populate_arrays_d(A, impl_A, layout, m, n, lda, vals)
end subroutine perflibs_spmat_export_dense_d

subroutine perflibs_spmat_export_dense_c(A, layout, m, n, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_dense_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer(kind=perflibs_i4), intent(in) :: layout
    integer, intent(out) :: m
    integer, intent(out) :: n
    complex(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A
    integer :: lda, size

    call perflibs_dense_param_check_c(A, impl_A, layout, m, n, lda, size, info)

    if  (allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(vals(size))

    call perflibs_dense_populate_arrays_c(A, impl_A, layout, m, n, lda, vals)
end subroutine perflibs_spmat_export_dense_c

subroutine perflibs_spmat_export_dense_z(A, layout, m, n, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_dense_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer(kind=perflibs_i4), intent(in) :: layout
    integer, intent(out) :: m
    integer, intent(out) :: n
    complex(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A
    integer :: lda, size

    call perflibs_dense_param_check_z(A, impl_A, layout, m, n, lda, size, info)

    if  (allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(vals(size))

    call perflibs_dense_populate_arrays_z(A, impl_A, layout, m, n, lda, vals)
end subroutine perflibs_spmat_export_dense_z

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!  BSR Functions
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine perflibs_spmat_export_bsr_s(A, block_layout, index_base, m, n, block_size, &
                                       row_ptr, col_indx, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_bsr_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer(kind=perflibs_i4), intent(in) :: block_layout
    integer, intent(in) :: index_base
    integer, intent(out) :: m
    integer, intent(out) :: n
    integer, intent(out) :: block_size
    integer, allocatable, intent(out) :: row_ptr(:)
    integer, allocatable, intent(out) :: col_indx(:)
    real(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A
    integer :: nnzb, nrowsb, nnz, index_base_A

    call perflibs_bsr_param_check_s(A, impl_A, block_layout, index_base, m, n, block_size, &
                                    nnzb, nrowsb, nnz, index_base_A, info)

    if (allocated(row_ptr) .or. allocated(col_indx) .or. allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(row_ptr(nrowsb+1))
    allocate(col_indx(nnzb))
    allocate(vals(nnz))

    call perflibs_bsr_populate_arrays_s(A, impl_A, block_layout, index_base_A, index_base, block_size, &
                                        nnz, nnzb, nrowsb, row_ptr, col_indx, vals)
end subroutine perflibs_spmat_export_bsr_s


subroutine perflibs_spmat_export_bsr_d(A, block_layout, index_base, m, n, block_size, &
                                       row_ptr, col_indx, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_bsr_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer(kind=perflibs_i4), intent(in) :: block_layout
    integer, intent(in) :: index_base
    integer, intent(out) :: m
    integer, intent(out) :: n
    integer, intent(out) :: block_size
    integer, allocatable, intent(out) :: row_ptr(:)
    integer, allocatable, intent(out) :: col_indx(:)
    real(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A
    integer :: nnzb, nrowsb, nnz, index_base_A

    call perflibs_bsr_param_check_d(A, impl_A, block_layout, index_base, m, n, block_size, &
                                    nnzb, nrowsb, nnz, index_base_A, info)

    if (allocated(row_ptr) .or. allocated(col_indx) .or. allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(row_ptr(nrowsb+1))
    allocate(col_indx(nnzb))
    allocate(vals(nnz))

    call perflibs_bsr_populate_arrays_d(A, impl_A, block_layout, index_base_A, index_base, block_size, &
                                        nnz, nnzb, nrowsb, row_ptr, col_indx, vals)
end subroutine perflibs_spmat_export_bsr_d

subroutine perflibs_spmat_export_bsr_c(A, block_layout, index_base, m, n, block_size, &
                                       row_ptr, col_indx, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_bsr_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer(kind=perflibs_i4), intent(in) :: block_layout
    integer, intent(in) :: index_base
    integer, intent(out) :: m
    integer, intent(out) :: n
    integer, intent(out) :: block_size
    integer, allocatable, intent(out) :: row_ptr(:)
    integer, allocatable, intent(out) :: col_indx(:)
    complex(kind=perflibs_r32), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A
    integer :: nnzb, nrowsb, nnz, index_base_A

    call perflibs_bsr_param_check_c(A, impl_A, block_layout, index_base, m, n, block_size, &
                                    nnzb, nrowsb, nnz, index_base_A, info)

    if (allocated(row_ptr) .or. allocated(col_indx) .or. allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(row_ptr(nrowsb+1))
    allocate(col_indx(nnzb))
    allocate(vals(nnz))

    call perflibs_bsr_populate_arrays_c(A, impl_A, block_layout, index_base_A, index_base, block_size, &
                                        nnz, nnzb, nrowsb, row_ptr, col_indx, vals)
end subroutine perflibs_spmat_export_bsr_c

subroutine perflibs_spmat_export_bsr_z(A, block_layout, index_base, m, n, block_size, &
                                       row_ptr, col_indx, vals, info)
    use perflibs_kinds
    use perflibs_sparse_params
    use perflibs_bsr_functions
    implicit none
    integer(kind=perflibs_i8), intent(in) :: A
    integer(kind=perflibs_i4), intent(in) :: block_layout
    integer, intent(in) :: index_base
    integer, intent(out) :: m
    integer, intent(out) :: n
    integer, intent(out) :: block_size
    integer, allocatable, intent(out) :: row_ptr(:)
    integer, allocatable, intent(out) :: col_indx(:)
    complex(kind=perflibs_r64), allocatable, intent(out) :: vals(:)
    integer, intent(out) :: info

    integer(kind=perflibs_i8) :: impl_A
    integer :: nnzb, nrowsb, nnz, index_base_A

    call perflibs_bsr_param_check_z(A, impl_A, block_layout, index_base, m, n, block_size, &
                                    nnzb, nrowsb, nnz, index_base_A, info)

    if (allocated(row_ptr) .or. allocated(col_indx) .or. allocated(vals)) then
        info = perflibs_status_input_parameter_error
    end if

    if (info .ne. perflibs_status_success) then
        return
    end if

    allocate(row_ptr(nrowsb+1))
    allocate(col_indx(nnzb))
    allocate(vals(nnz))

    call perflibs_bsr_populate_arrays_z(A, impl_A, block_layout, index_base_A, index_base, block_size, &
                                        nnz, nnzb, nrowsb, row_ptr, col_indx, vals)
end subroutine perflibs_spmat_export_bsr_z

