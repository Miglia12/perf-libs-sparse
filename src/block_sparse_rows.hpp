/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "pod_vector.hpp"
#include "util.hpp"
#include <vector>

template <typename T> struct perflibs_spmat_impl_t;

namespace perflibs::sparse {

template <typename T> struct perflibs_bsr {
  /// Layout (column or row major) within a dense block
  perflibs_dense_layout block_layout;
  /// Number of rows
  int64_t m;
  /// Number of columns
  int64_t n;
  /// Size of the square block
  int64_t block_size;
  /// The number of blocks in the sparse matrix
  int64_t nnzb;
  /// The number of blocked rows
  int64_t nrowsb;

  /// The non-zero values in the matrix
  perflibs::sparse::pod_vector<T> vals;
  /// The indices of the start of the blocked rows
  perflibs::sparse::pod_vector<perflibs_int_t> row_ptr;
  /// The indices of the blocked columns in the matrix
  perflibs::sparse::pod_vector<perflibs_int_t> col_indx;

  // Pointers to either user data or the data under the vector copies
  const T *vals_ptr;
  const perflibs_int_t *row_ptr_ptr;
  const perflibs_int_t *col_indx_ptr;

  int nthreads = perflibs::sparse::omp::get_max_threads();

  bool use_vanilla = false;

  // Constructors
  // Set m, n negative so that we know when to copy data in the copy operator
  perflibs_bsr()
      : block_layout(PERFLIBS_COL_MAJOR), m(-1), n(-1), block_size(-1),
        nnzb(-1), nrowsb(-1), vals_ptr(nullptr), row_ptr_ptr(nullptr),
        col_indx_ptr(nullptr) {};

  // Copy version
  perflibs_bsr(perflibs_dense_layout block_layout, int64_t rows, int64_t cols,
               int64_t block_size, int64_t nnzb, int64_t nrowsb, int64_t nnz,
               const T *vals, const perflibs_int_t *row_ptr,
               const perflibs_int_t *col_indx)
      : block_layout(block_layout), m(rows), n(cols), block_size(block_size),
        nnzb(nnzb), nrowsb(nrowsb), vals(vals, vals + nnz),
        row_ptr(row_ptr, row_ptr + nrowsb + 1),
        col_indx(col_indx, col_indx + nnzb), vals_ptr(this->vals.data()),
        row_ptr_ptr(this->row_ptr.data()), col_indx_ptr(this->col_indx.data()) {
  }

  // No copy version
  perflibs_bsr(perflibs_dense_layout block_layout, int64_t rows, int64_t cols,
               int64_t block_size, int64_t nnzb, int64_t nrowsb, const T *vals,
               const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx)
      : block_layout(block_layout), m(rows), n(cols), block_size(block_size),
        nnzb(nnzb), nrowsb(nrowsb), vals_ptr(vals), row_ptr_ptr(row_ptr),
        col_indx_ptr(col_indx) {}

  perflibs_bsr &operator=(const perflibs_bsr &other);
  perflibs_bsr(const perflibs_bsr &other) { *this = other; }
  perflibs_bsr &operator=(perflibs_bsr &&other) = default;
  perflibs_bsr(perflibs_bsr &&other) = default;

  void scale_matrix(T alpha);
};

/**
 * Populates the @p bsr member of the matrix @p A with the data provided. Some
 * basic checks are performed that the data represents a block sparse row
 * matrix.
 * @param [in,out] A         The sparse matrix for which to populate data
 * @param [in] block_layout  The layout (column or row major) within a dense
 * block
 * @param [in] m             The number of rows in the matrix
 * @param [in] n             The number of columns in the matrix
 * @param [in] block_size    The dimension of a square block
 * @param [in] row_ptr       Indices to the start of a blocked row
 * @param [in] col_indx      Indices of the blocked column
 * @param [in] vals          The data to copy
 * @param [in] no_copy       Indicates whether it is valid to take a copy of the
 * data or not
 */
template <typename T>
perflibs_status_t fill_initial_data_bsr(
    perflibs_spmat_t A, perflibs_dense_layout block_layout, perflibs_int_t m,
    perflibs_int_t n, perflibs_int_t block_size, const perflibs_int_t *row_ptr,
    const perflibs_int_t *col_indx, const T *vals, bool no_copy);

/**
 * Returns a BSR structure from CSR input data.
 * By default, it sets the block_size to 1, and the block_layout to col_major
 * although this does not matter for unity block size.
 * This functions preserves the base index of the matrix.
 * @param [in] m           The number of rows in the matrix
 * @param [in] n           The number of columns in the matrix
 * @param [in] vals        The values in the CSR representation of the matrix
 * @param [in] row_ptr     The pointer to the start of the rows in the CSR
 * matrix
 * @param [in] col_indx    The column indices of non-zero values in the CSR
 * matrix
 * @return A matrix in internal BSR format, with a block size of one.
 */
template <typename T>
perflibs_bsr<T> csr2bsr(perflibs_int_t m, perflibs_int_t n, const T *vals,
                        const perflibs_int_t *row_ptr,
                        const perflibs_int_t *col_indx);

/**
 * Updates the data structures in the block sparse row matrix using the same
 * first touch policy as will be used in the SpMV execute phase. This is not
 * a member function of perflibs_bsr as the data layout may be specific for
 * SpMV, and so is defined with the rest of the SpMV functions.
 *
 * @param [in,out] bsr   The matrix for which to reallocate the data
 */
template <typename T> void spmv_bsr_realloc_data(perflibs_bsr<T> &bsr);

/**
 * Performs the sparse matrix-vector multiplication of the form @f$y = \alpha A
 * x + \beta y@f$ where matrix
 * @f$A@f$ is in block sparse row format.
 * @param [in] bsr    The matrix to use in the SpMV operation
 * @param [in] trans  Hint as to whether the transpose of the matrix should be
 * taken. At the moment this is not supported
 * @param [in] x      The vector @f$x@f$ to use in the SpMV operation
 * @param [in,out] y  The output vector for the SpMV operation. For non-zero @p
 * beta, this is also an input variable
 * @param [in] alpha  Scalar @f$\alpha@f$ used in the SpMV operation
 * @param [in] beta   Scalar @f$\beta@f$ used in the SpMV operation
 */
template <typename T>
void spmv_bsr(const perflibs_bsr<T> &bsr, sparse_hint_value_internal trans,
              const T *x, T *y, T alpha, T beta);

/**
 * Performs a triangular solve of a system of linear equations supplied as a
 * sparse matrix and a dense RHS vector. The solution is written into a separate
 * dense vector.
 * @f[ op(A) x = \alpha y @f]
 * The matrix @f$ A @f$ is supplied in block sparse row (BSR) format and @f$ op
 * @f$ either the transpose or identity transform
 * @param [in]  bsr    The matrix in BSR format
 * @param [in]  trans  Hint as to whether the matrix is to be transposed before
 * doing the triangular solve
 * @param [in]  uplo   Hint as to whether the matrix is to be treated as an
 * upper or lower triangular matrix
 * @param [in]  diag   Hint as to whether the diagonals of the matrix A is unit
 * or non-unit
 * @param [out] x      The output vector @f$ x @f$ to be solved for
 * @param [in]  y      The input RHS vector.
 * @param [in]  alpha  Scalar @f$ \alpha @f$ to be multiplied to the RHS vector
 * y
 */
template <typename T>
void spsv_bsr(const perflibs_bsr<T> &bsr, sparse_hint_value_internal trans,
              sparse_hint_value_internal uplo, sparse_hint_value_internal diag,
              T *x, const T *y, T alpha);

template <typename T>
void spnorm_inf_bsr(const perflibs_bsr<T> &bsr,
                    perflibs::sparse::remove_complex_t<T> *result);

template <typename T>
perflibs_status_t
spmat_update_bsr(perflibs_spmat_impl_t<T> *impl, perflibs_int_t n_updates,
                 const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
                 const T *vals);

} // namespace perflibs::sparse
