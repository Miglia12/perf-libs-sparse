/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "pod_vector.hpp"
#include "solve_parallel.hpp"
#include "util.hpp"

#include <cstdint>
#include <utility>
#include <vector>

// Declarations of functions required to support CSR

// Forward declaration
template <typename T> struct perflibs_spmat_impl_t;

namespace perflibs::sparse {

/// This defines SpMV parallelism
struct par_mv_t {
  /// The number of threads to use for SpMV operations
  int nthreads = perflibs::sparse::omp::get_max_threads();

  /// Stores the index into the values which are to be used by a thread
  std::vector<perflibs_int_t> thread_inds;
  /// Stores the start / end row of each of the threads in a parallel
  /// decomposition
  std::vector<perflibs_int_t> thread_rows;
};

// An internal type for Compressed Sparse Rows (CSR) structures
template <typename T> struct perflibs_csr {
  /// The number of rows in the matrix
  int64_t m;
  /// The number of columns in the matrix
  int64_t n;

  /// Non-zero values in the matrix. This is populated when user data
  /// has been copied, otherwise it is empty.
  perflibs::sparse::pod_vector<T> vals;
  /// Indices of the start of the rows in the matrix. This is populated
  /// when user data has been copied, otherwise it is empty.
  std::vector<perflibs_int_t> row_ptr;
  /// Indices of the columns of the non-zero values in the matrix. This
  /// is populated when user data has been copied, otherwise it is empty.
  perflibs::sparse::pod_vector<perflibs_int_t> col_indx;

  /// Pointer to either an array of values in the matrix provided by a user,
  /// or to the base of #vals if user data has been copied.
  const T *vals_ptr;
  /// Pointer to either an array of row start indices provided by a user,
  /// or to the base of #row_ptr if user data has been copied.
  const perflibs_int_t *row_ptr_ptr;
  /// Pointer to either an array of column indices for non-zero values in the
  /// matrix provided by a user, or to the base of #col_indx if user data
  /// has been copied.
  const perflibs_int_t *col_indx_ptr;

  // A flag to indicate that the vanilla/standard CSR kernel is to be used
  bool use_vanilla = false;

  /// A type containing details of the sparse matrix-vector parallel setup
  par_mv_t par_mv;

  /// A type containing details of the sparse triangular solve parallel setup
  par_sv_t par_sv;

  /**
   * Constructor of an empty matrix
   */
  perflibs_csr()
      : m(-1), n(-1), vals_ptr(nullptr), row_ptr_ptr(nullptr),
        col_indx_ptr(nullptr), par_mv(), par_sv() {}

  /**
   * Data copying constructor: we take copies of user arrays
   * @param [in] m	Number of rows
   * @param [in] n	Number of rows
   * @param [in] nnz	Number of non-zeros
   * @param [in] vals	Array of non-zero values of the matrix
   * @param [in] row_ptr Array of pointers to the start of each row in vals,
   * col_indx
   * @param [in] col_indx Array of column indices for non-zero values
   */
  perflibs_csr(int64_t m, int64_t n, int64_t nnz, const T *vals,
               const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx)
      : m(m), n(n), vals(vals, vals + nnz), row_ptr(row_ptr, row_ptr + m + 1),
        col_indx(col_indx, col_indx + nnz), vals_ptr(this->vals.data()),
        row_ptr_ptr(this->row_ptr.data()), col_indx_ptr(this->col_indx.data()),
        par_mv(), par_sv() {}

  /**
   * Non-data-copying constructor: pointers to user arrays are assigned
   * @param [in] m	Number of rows
   * @param [in] n	Number of rows
   * @param [in] vals	Array of non-zero values of the matrix
   * @param [in] row_ptr Array of pointers to the start of each row in vals,
   * col_indx
   * @param [in] col_indx Array of column indices for non-zero values
   * @param [in] par_sv Parallel solve setup structure
   */
  perflibs_csr(int64_t m, int64_t n, const T *vals,
               const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
               par_sv_t par_sv)
      : m(m), n(n), vals_ptr(vals), row_ptr_ptr(row_ptr),
        col_indx_ptr(col_indx), par_mv(), par_sv(std::move(par_sv)) {}

  /**
   * In-place constructor: vector and pod_vectors are moved into the
   * perflibs_csr type
   * @param [in] m	Number of rows
   * @param [in] n	Number of rows
   * @param [in] vals	Array of non-zero values of the matrix
   * @param [in] row_ptr Array of pointers to the start of each row in vals,
   * col_indx
   * @param [in] col_indx Array of column indices for non-zero values
   */
  perflibs_csr(int64_t m, int64_t n, perflibs::sparse::pod_vector<T> vals,
               std::vector<perflibs_int_t> row_ptr,
               perflibs::sparse::pod_vector<perflibs_int_t> col_indx)
      : m(m), n(n), vals(std::move(vals)), row_ptr(std::move(row_ptr)),
        col_indx(std::move(col_indx)), vals_ptr(this->vals.data()),
        row_ptr_ptr(this->row_ptr.data()), col_indx_ptr(this->col_indx.data()),
        par_mv(), par_sv() {}

  perflibs_csr &operator=(const perflibs_csr &other);
  perflibs_csr(const perflibs_csr &other) { *this = other; };
  perflibs_csr &operator=(perflibs_csr &&other) = default;
  perflibs_csr(perflibs_csr &&other) = default;

  void scale_matrix(enum perflibs_sparse_hint_value trans, T alpha);

  /**
   * Generates the structure required for a SpMV parallel decomposition of the
   * matrix. This method does not assign data, though. For that call
   * gen_data_vectors
   * @param [in] strat    The parallel decomposition strategy to use
   */
  void gen_parallel_decomp_mv(perflibs_parallel_decomp_strategy strat);

  /**
   * Generates the data vectors to fit in with the parallel execution strategy.
   */
  void gen_data_vectors_mv();

  /**
   * Generates the structure required for a SpTRSV parallel decomposition of the
   * matrix. Template parameter is the integer type to use in tracking row
   * dependencies. We optimistically try uint8, then fall back to uint16, etc.
   * if the dependency count is too large. We use the CSR format in parallel if
   * we're solving the non-transpose of the matrix. CSC structure arrays are
   * passed in because they contain column-wise dependencies.
   * @param [in] shape    Whether the matrix is upper or lower triangular
   * @param [in] col_ptr  The column pointers, taken from the CSC representation
   * of the matrix
   * @param [in] row_indx The row indices, taken from the CSC representation of
   * the matrix
   */
  template <typename T2>
  void gen_parallel_decomp_sv(perflibs_sparse_matrix_shape_t shape,
                              const perflibs_int_t *col_ptr,
                              const perflibs_int_t *row_indx);

  /// decide whether the diagonal elements are in the expected positions for all
  /// rows
  bool diag_in_place(bool lower) const noexcept {
    const perflibs_int_t base = row_ptr_ptr[0];

    for (int64_t i = 0; i < m; ++i) {
      const auto start = row_ptr_ptr[i] - base;
      const auto end = row_ptr_ptr[i + 1] - base;
      const perflibs_int_t diag = i + base;

      if (lower) {
        if (col_indx_ptr[end - 1] != diag)
          return false;
      } else {
        if (col_indx_ptr[start] != diag)
          return false;
      }
    }
    return true;
  }
};

/**
 * Populate A with new details after checking params. This version takes raw
 * pointers, which may be pointers to user data.
 * @param [inout] A		Sparse matrix type to populate
 * @param [in] m		Number of rows
 * @param [in] n		Number of columns
 * @param [in] row_ptr	Array of row pointers
 * @param [in] col_indx	Array of column indices
 * @param [in] vals 	Non-zero values
 * @param [in] no_copy	Do we want to take a copy of the arrays, or just set
 * pointers?
 */
template <typename T>
perflibs_status_t fill_initial_data_csr(perflibs_spmat_t A, perflibs_int_t m,
                                        perflibs_int_t n,
                                        const perflibs_int_t *row_ptr,
                                        const perflibs_int_t *col_indx,
                                        const T *vals, bool no_copy);

/**
 * Populate A with new details after checking params. This version takes vector
 * types which are moved into A, rather than incurring an expensive copy. This
 * is performance critical in SpMM.
 * @param [inout] A		Sparse matrix type to populate
 * @param [in] m		Number of rows
 * @param [in] n		Number of columns
 * @param [in] row_ptr	Array of row pointers
 * @param [in] col_indx	Array of column indices
 * @param [in] vals 	Non-zero values
 * @param [in] no_copy	Do we want to take a copy of the arrays, or just set
 * pointers?
 */
template <typename T>
perflibs_status_t
fill_initial_data_csr(perflibs_spmat_t A, perflibs_int_t m, perflibs_int_t n,
                      std::vector<perflibs_int_t> &&row_ptr,
                      perflibs::sparse::pod_vector<perflibs_int_t> &&col_indx,
                      perflibs::sparse::pod_vector<T> &&vals);

template <typename T>
perflibs_csr<T> coo2csr(perflibs_int_t m, perflibs_int_t n, perflibs_int_t nnz,
                        const T *vals, const perflibs_int_t *col_indx,
                        const perflibs_int_t *row_indx,
                        perflibs_int_t index_base);

template <typename T>
perflibs_csr<T> dense2csr(perflibs_dense_layout layout, perflibs_int_t m,
                          perflibs_int_t n, perflibs_int_t lda, const T *A,
                          perflibs_int_t index_base);

/// Returns pair containing row_ptr and col_indx only
std::pair<std::vector<perflibs_int_t>, std::vector<perflibs_int_t>>
csc2csr_struct(int64_t rows, int64_t cols, const perflibs_int_t *row_indx,
               const perflibs_int_t *col_ptr);

template <typename T>
perflibs_csr<T> csc2csr(enum sparse_hint_value_internal trans, int64_t m,
                        int64_t n, const T *vals,
                        const perflibs_int_t *row_indx,
                        const perflibs_int_t *col_ptr);

template <typename T>
perflibs_csr<T>
scs2csr(perflibs_int_t m, perflibs_int_t n, perflibs_int_t C,
        perflibs::sparse::pod_vector<T> &scs_vals,
        perflibs::sparse::pod_vector<perflibs_int_t> &scs_col_indx_offsets,
        std::vector<int64_t> &col_indx_min, perflibs_int_t col_indx_bytes,
        std::vector<int64_t> &scs_cl, std::vector<int64_t> &scs_cs,
        std::vector<int64_t> scs_row_permd2in);

template <typename T>
perflibs_csr<T> bsr2csr(perflibs_dense_layout block_layout, perflibs_int_t m,
                        perflibs_int_t n, perflibs_int_t block_size,
                        perflibs_int_t nnzb, perflibs_int_t nrowsb,
                        const T *vals, const perflibs_int_t *row_ptr,
                        const perflibs_int_t *col_indx);

template <typename T>
perflibs_csr<T> null2csr(perflibs_int_t m, perflibs_int_t n);

template <typename T> perflibs_csr<T> identity2csr(perflibs_int_t n);

void make_index_base_equal(
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t nnzA,
    perflibs_int_t nnzB, const perflibs_int_t **row_ptrA_ref,
    const perflibs_int_t **col_ptrB_ref, const perflibs_int_t **col_indxA_ref,
    const perflibs_int_t **row_indxB_ref,
    perflibs::sparse::pod_vector<perflibs_int_t> &index_copy,
    perflibs::sparse::pod_vector<perflibs_int_t> &ptr_copy);

/**
 * Performs a sparse matrix-vector multiplication of the form
 * @f[ y = \alpha op(A) x + \beta y @f]
 * for matrix @f$ A @f$ in compressed-sparse row (CSR) format and @f$ op @f$
 * either the transpose or identity transform
 * @param [in] csr    The matrix in CSR format
 * @param [in] trans  Hint as to whether the matrix is to be transposed before
 * multiplication
 * @param [in] x      Vector @f$ x @f$ with which to multiply the matrix
 * @param [in,out] y  The output vector. This is updated with the result of the
 * SpMV operation
 * @param [in] alpha  Scalar @f$ \alpha @f$ in the SpMV operation
 * @param [in] beta   Scalar @f$ \beta @f$ in the SpMV operation
 */
template <typename T>
void spmv_csr(const perflibs_csr<T> &csr, sparse_hint_value_internal trans,
              const T *x, T *y, T alpha, T beta);

/**
 * Performs a sparse matrix - dense matrix multiplication using a CSR sparse
 * matrix and a row-major dense matrix
 * @param [in] csr    The sparse matrix A in CSR format
 * @param [in] B      The dense input matrix B pointer
 * @param [in] ldb    Leading dimension of B
 * @param [in,out] C  The dense output matrix C pointer
 * @param [in] ldc    Leading dimension of C
 * @param [in] n      Number of columns of C
 * @param [in] alpha  Scalar @f$ \alpha @f$ in the SpMM operation
 * @param [in] beta   Scalar @f$ \beta @f$ in the SpMM operation
 */
template <typename T, bool conjA_flag, bool conjB_flag>
void spmm_rowwise_csr_blocked_m(const perflibs_csr<T> &csr, const T *B,
                                perflibs_int_t ldb, T *C, perflibs_int_t ldc,
                                perflibs_int_t n, T alpha, T beta);

/**
 * Performs a triangular solve of a system of linear equations supplied as a
 * sparse matrix and a dense RHS vector. The solution is written into a separate
 * dense vector.
 * @f[ op(A) x = \alpha y @f]
 * The matrix @f$ A @f$ is supplied in the compressed-sparse row (CSR) format
 * and @f$ op @f$ either the transpose or identity transform
 * @param [in]  csr    The matrix in CSR format
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
void spsv_csr(const perflibs_csr<T> &csr, sparse_hint_value_internal trans,
              sparse_hint_value_internal uplo, sparse_hint_value_internal diag,
              T *x, const T *y, T alpha);

template <typename T>
perflibs_status_t
spmat_update_csr(perflibs_spmat_impl_t<T> *impl, perflibs_int_t n_updates,
                 const perflibs_int_t *row_indx, const perflibs_int_t *col_indx,
                 const T *vals);

template <typename T>
perflibs_status_t
spelmm_csr(perflibs_sparse_hint_value transA, perflibs_spmat_impl_t<T> *impl_A,
           perflibs_sparse_hint_value transB, perflibs_spmat_impl_t<T> *impl_B,
           perflibs_spmat_t AB);

template <typename T>
void spelmm_csr_kernel(perflibs_sparse_hint_value transA,
                       const perflibs_int_t *row_ptrA,
                       const perflibs_int_t *col_indxA, const T *valsA,
                       perflibs_sparse_hint_value transB,
                       const perflibs_int_t *row_ptrB,
                       const perflibs_int_t *col_indxB, const T *valsB,
                       perflibs_int_t m, perflibs_int_t index_base,
                       perflibs::sparse::pod_vector<perflibs_int_t> &row_ptrAB,
                       std::vector<perflibs_int_t> &col_indxAB,
                       std::vector<T> &valsAB);

template <typename T>
perflibs_status_t
sddmm_csr(perflibs_sparse_hint_value transA, perflibs_sparse_hint_value transB,
          T alpha, perflibs_spmat_impl_t<T> *impl_A,
          perflibs_spmat_impl_t<T> *impl_B, T beta,
          perflibs_spmat_impl_t<T> *impl_C, perflibs_spmat_t AB);

/// A helper function to use the inline fp64 ACLE spsv_dot function for sparse
/// vector dot product as well as SpSV
void spsv_dot_fp64_helper(double &sum, const perflibs_int_t *col_indx,
                          const double *vals, const double *x,
                          perflibs_int_t off, perflibs_int_t row_start_indx,
                          perflibs_int_t row_end_indx);

} // namespace perflibs::sparse
