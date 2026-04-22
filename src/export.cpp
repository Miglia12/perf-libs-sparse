/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "export.hpp"
#include "convert.hpp"
#include "util.hpp"

// Macro to produce platform-specific Fortran symbols
// Default to Linux-style single trailing underscore
#define FTN_SYMB(FN_NAME) FN_NAME##_

namespace perflibs::sparse {

/// Used in export_{CSR, CSC, BSR}. Converts the index/pointer arrays into the
/// requested user index_base. ie
/// +/- 1.
void adjust_exported_indexes(int mat_base, int user_base,
                             const perflibs_int_t *mat_ptr_arr,
                             perflibs_int_t *user_ptr_arr,
                             const perflibs_int_t *mat_indx_arr,
                             perflibs_int_t *user_indx_arr,
                             perflibs_int_t size_of_ptr_arr,
                             perflibs_int_t size_of_indx_arr) {
  int adjust = user_base - mat_base;
  for (perflibs_int_t i = 0; i < size_of_ptr_arr; i++) {
    user_ptr_arr[i] = mat_ptr_arr[i] + adjust;
  }
  for (perflibs_int_t i = 0; i < size_of_indx_arr; i++) {
    user_indx_arr[i] = mat_indx_arr[i] + adjust;
  }
}

/* -------------------------------------------------
 * CSR
 * ------------------------------------------------*/

template <typename T>
std::pair<perflibs_status_t, std::unique_ptr<perflibs_spmat_impl_t<T>>>
csr_param_check(perflibs_const_spmat_t A, const perflibs_int_t index_base,
                perflibs_int_t *m, perflibs_int_t *n, perflibs_int_t *nnz,
                perflibs_int_t *index_base_A) {

  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);

  // basic dimension check of input matrix
  if (impl_A->m <= 0 || impl_A->n <= 0) {
    return {PERFLIBS_STATUS_INPUT_PARAMETER_ERROR, nullptr};
  }
  // Check for valid index_base integer input
  if (index_base != 0 && index_base != 1) {
    return {PERFLIBS_STATUS_INPUT_PARAMETER_ERROR, nullptr};
  }

  // Conversion Cases
  // If matrix object is not already in CSR, then convert.
  std::unique_ptr<perflibs_spmat_impl_t<T>> A_copy(nullptr);
  if (impl_A->spmat_format != perflibs_format_csr) {
    A_copy = std::make_unique<perflibs_spmat_impl_t<T>>(
        *reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl));
    impl_A = A_copy.get();
    auto info = convert<T>(perflibs_format_csr, impl_A);
    if (info != PERFLIBS_STATUS_SUCCESS) {
      return {info, nullptr};
    }
  }

  assert(impl_A->spmat_format == perflibs_format_csr);

  *m = impl_A->m;
  *n = impl_A->n;
  *nnz = impl_A->nnz;
  *index_base_A = impl_A->index_base;

  // Check validity for matrix index_base input
  if (*index_base_A != 0 && *index_base_A != 1) {
    return {PERFLIBS_STATUS_INPUT_PARAMETER_ERROR, nullptr};
  }

  return {PERFLIBS_STATUS_SUCCESS, std::move(A_copy)};
}

template <typename T>
void csr_populate_arrays(const perflibs_spmat_impl_t<T> *impl_A,
                         const perflibs_int_t nrows, const perflibs_int_t nnz,
                         const perflibs_int_t index_base_A,
                         const perflibs_int_t index_base,
                         perflibs_int_t *row_ptr_out,
                         perflibs_int_t *col_indx_out, T *vals_out) {
  // Vals array can always be copied
  std::memcpy((void *)vals_out, (const void *)impl_A->csr.vals_ptr,
              sizeof(T) * nnz);

  // If input and output have same indexing, then copy
  if (index_base_A == index_base) {
    std::memcpy((void *)row_ptr_out, (const void *)impl_A->csr.row_ptr_ptr,
                sizeof(perflibs_int_t) * (nrows + 1));
    std::memcpy((void *)col_indx_out, (const void *)impl_A->csr.col_indx_ptr,
                sizeof(perflibs_int_t) * nnz);
  }
  // Input and output matrix have different bases then adjust row_ptr and
  // col_indx arrays
  else {
    adjust_exported_indexes(index_base_A, index_base, impl_A->csr.row_ptr_ptr,
                            row_ptr_out, impl_A->csr.col_indx_ptr, col_indx_out,
                            nrows + 1, nnz);
  }
}

template <typename T>
perflibs_status_t spmat_export_csr(perflibs_const_spmat_t A,
                                   perflibs_int_t index_base, perflibs_int_t *m,
                                   perflibs_int_t *n, perflibs_int_t **row_ptr,
                                   perflibs_int_t **col_indx, T **vals) {

  // Null Pointer Check
  if (!m || !n || !row_ptr || !col_indx || !vals) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  perflibs_int_t nnz = -1;
  perflibs_int_t index_base_A = -1;

  auto [param_status, A_copy] =
      csr_param_check<T>(A, index_base, m, n, &nnz, &index_base_A);
  if (param_status != PERFLIBS_STATUS_SUCCESS) {
    return param_status;
  }

  // A deep copy may have been made, so use the pointer to the copy if it's not
  // empty
  auto impl_A =
      A_copy.get()
          ? A_copy.get()
          : reinterpret_cast<const perflibs_spmat_impl_t<T> *>(A->impl);

  // Assigning new name for readability
  perflibs_int_t nrows = *m;

  // Allocate memory for arrays
  perflibs_int_t *row_ptr_out =
      (perflibs_int_t *)std::malloc(sizeof(perflibs_int_t) * (nrows + 1));
  perflibs_int_t *col_indx_out =
      (perflibs_int_t *)std::malloc(sizeof(perflibs_int_t) * nnz);
  T *vals_out = (T *)std::malloc(sizeof(T) * nnz);

  csr_populate_arrays<T>(impl_A, nrows, nnz, index_base_A, index_base,
                         row_ptr_out, col_indx_out, vals_out);

  // Point to user arrays
  *row_ptr = row_ptr_out;
  *col_indx = col_indx_out;
  *vals = vals_out;

  return PERFLIBS_STATUS_SUCCESS;
}

template perflibs_status_t
spmat_export_csr<float>(perflibs_const_spmat_t A, perflibs_int_t index_base,
                        perflibs_int_t *m, perflibs_int_t *n,
                        perflibs_int_t **row_ptr, perflibs_int_t **col_indx,
                        float **vals);
template perflibs_status_t
spmat_export_csr<double>(perflibs_const_spmat_t A, perflibs_int_t index_base,
                         perflibs_int_t *m, perflibs_int_t *n,
                         perflibs_int_t **row_ptr, perflibs_int_t **col_indx,
                         double **vals);
template perflibs_status_t spmat_export_csr<std::complex<float>>(
    perflibs_const_spmat_t A, perflibs_int_t index_base, perflibs_int_t *m,
    perflibs_int_t *n, perflibs_int_t **row_ptr, perflibs_int_t **col_indx,
    std::complex<float> **vals);
template perflibs_status_t spmat_export_csr<std::complex<double>>(
    perflibs_const_spmat_t A, perflibs_int_t index_base, perflibs_int_t *m,
    perflibs_int_t *n, perflibs_int_t **row_ptr, perflibs_int_t **col_indx,
    std::complex<double> **vals);

/* -------------------------------------------------
 * CSC
 * ------------------------------------------------*/

template <typename T>
std::pair<perflibs_status_t, std::unique_ptr<perflibs_spmat_impl_t<T>>>
csc_param_check(perflibs_const_spmat_t A, const perflibs_int_t index_base,
                perflibs_int_t *m, perflibs_int_t *n, perflibs_int_t *nnz,
                perflibs_int_t *index_base_A) {

  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);

  // basic dimension checks of matrix
  if (impl_A->m <= 0 || impl_A->n <= 0) {
    return {PERFLIBS_STATUS_INPUT_PARAMETER_ERROR, nullptr};
  }
  // Check for valid index_base integer input
  if (index_base != 0 && index_base != 1) {
    return {PERFLIBS_STATUS_INPUT_PARAMETER_ERROR, nullptr};
  }
  // Conversion Cases
  // if matrix object is not already in CSC, then convert.
  std::unique_ptr<perflibs_spmat_impl_t<T>> A_copy(nullptr);
  if (impl_A->spmat_format != perflibs_format_csc) {
    A_copy = std::make_unique<perflibs_spmat_impl_t<T>>(
        *reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl));
    impl_A = A_copy.get();
    auto info = convert<T>(perflibs_format_csc, impl_A);
    if (info != PERFLIBS_STATUS_SUCCESS) {
      return {info, nullptr};
    }
  }

  assert(impl_A->spmat_format == perflibs_format_csc);

  *m = impl_A->m;
  *n = impl_A->n;
  *nnz = impl_A->nnz;
  *index_base_A = impl_A->index_base;

  // Check validity for matrix index_base input
  if (*index_base_A != 0 && *index_base_A != 1) {
    return {PERFLIBS_STATUS_INPUT_PARAMETER_ERROR, nullptr};
  }

  return {PERFLIBS_STATUS_SUCCESS, std::move(A_copy)};
}

template <typename T>
void csc_populate_arrays(const perflibs_spmat_impl_t<T> *impl_A,
                         const perflibs_int_t ncols, const perflibs_int_t nnz,
                         const perflibs_int_t index_base_A,
                         const perflibs_int_t index_base,
                         perflibs_int_t *row_indx_out,
                         perflibs_int_t *col_ptr_out, T *vals_out) {
  // Vals array can always be copied
  std::memcpy((void *)vals_out, (const void *)impl_A->csc.vals_ptr,
              sizeof(T) * nnz);

  if (index_base_A == index_base) {
    std::memcpy((void *)col_ptr_out, (const void *)impl_A->csc.col_ptr_ptr,
                sizeof(perflibs_int_t) * (ncols + 1));
    std::memcpy((void *)row_indx_out, (const void *)impl_A->csc.row_indx_ptr,
                sizeof(perflibs_int_t) * nnz);
  } else {
    adjust_exported_indexes(index_base_A, index_base, impl_A->csc.col_ptr_ptr,
                            col_ptr_out, impl_A->csc.row_indx_ptr, row_indx_out,
                            ncols + 1, nnz);
  }
}

template <typename T>
perflibs_status_t spmat_export_csc(perflibs_const_spmat_t A,
                                   perflibs_int_t index_base, perflibs_int_t *m,
                                   perflibs_int_t *n, perflibs_int_t **row_indx,
                                   perflibs_int_t **col_ptr, T **vals) {

  // Null Pointer Check
  if (!m || !n || !row_indx || !col_ptr || !vals) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  perflibs_int_t nnz = -1;
  perflibs_int_t index_base_A = -1;

  auto [param_status, A_copy] =
      csc_param_check<T>(A, index_base, m, n, &nnz, &index_base_A);
  if (param_status != PERFLIBS_STATUS_SUCCESS) {
    return param_status;
  }

  // A deep copy may have been made, so use the pointer to the copy if it's not
  // empty
  auto impl_A =
      A_copy.get()
          ? A_copy.get()
          : reinterpret_cast<const perflibs_spmat_impl_t<T> *>(A->impl);

  // Assigning new name for readability
  perflibs_int_t ncols = *n;

  // Allocate memory for arrays
  perflibs_int_t *col_ptr_out =
      (perflibs_int_t *)std::malloc(sizeof(perflibs_int_t) * (ncols + 1));
  perflibs_int_t *row_indx_out =
      (perflibs_int_t *)std::malloc(sizeof(perflibs_int_t) * nnz);
  T *vals_out = (T *)std::malloc(sizeof(T) * nnz);

  csc_populate_arrays<T>(impl_A, ncols, nnz, index_base_A, index_base,
                         row_indx_out, col_ptr_out, vals_out);

  // Point to user arrays
  *col_ptr = col_ptr_out;
  *row_indx = row_indx_out;
  *vals = vals_out;

  return PERFLIBS_STATUS_SUCCESS;
}

template perflibs_status_t
spmat_export_csc<float>(perflibs_const_spmat_t A, perflibs_int_t index_base,
                        perflibs_int_t *m, perflibs_int_t *n,
                        perflibs_int_t **row_indx, perflibs_int_t **col_ptr,
                        float **vals);
template perflibs_status_t
spmat_export_csc<double>(perflibs_const_spmat_t A, perflibs_int_t index_base,
                         perflibs_int_t *m, perflibs_int_t *n,
                         perflibs_int_t **row_indx, perflibs_int_t **col_ptr,
                         double **vals);
template perflibs_status_t spmat_export_csc<std::complex<float>>(
    perflibs_const_spmat_t A, perflibs_int_t index_base, perflibs_int_t *m,
    perflibs_int_t *n, perflibs_int_t **row_indx, perflibs_int_t **col_ptr,
    std::complex<float> **vals);
template perflibs_status_t spmat_export_csc<std::complex<double>>(
    perflibs_const_spmat_t A, perflibs_int_t index_base, perflibs_int_t *m,
    perflibs_int_t *n, perflibs_int_t **row_indx, perflibs_int_t **col_ptr,
    std::complex<double> **vals);

/* -------------------------------------------------
 * COO
 * ------------------------------------------------*/

template <typename T>
std::pair<perflibs_status_t, std::unique_ptr<perflibs_spmat_impl_t<T>>>
coo_param_check(perflibs_const_spmat_t A, perflibs_int_t *m, perflibs_int_t *n,
                perflibs_int_t *nnz) {

  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);

  // basic matrix dimension check
  if (impl_A->m <= 0 || impl_A->n <= 0) {
    return {PERFLIBS_STATUS_INPUT_PARAMETER_ERROR, nullptr};
  }
  // Conversion Cases
  // if matrix object is not already in COO, then convert.
  std::unique_ptr<perflibs_spmat_impl_t<T>> A_copy(nullptr);
  if (impl_A->spmat_format != perflibs_format_coo) {
    A_copy = std::make_unique<perflibs_spmat_impl_t<T>>(
        *reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl));
    impl_A = A_copy.get();
    auto info = convert<T>(perflibs_format_coo, impl_A);
    if (info != PERFLIBS_STATUS_SUCCESS) {
      return {info, nullptr};
    }
  }
  assert(impl_A->spmat_format == perflibs_format_coo);

  *m = impl_A->m;
  *n = impl_A->n;
  *nnz = impl_A->nnz;

  return {PERFLIBS_STATUS_SUCCESS, std::move(A_copy)};
}

template <typename T>
void coo_populate_arrays(const perflibs_spmat_impl_t<T> *impl_A,
                         const perflibs_int_t *nnz,
                         perflibs_int_t *row_indx_out,
                         perflibs_int_t *col_indx_out, T *vals_out) {
  // All arrays of size nnz
  std::memcpy((void *)col_indx_out, (const void *)impl_A->coo.col_indx_ptr,
              sizeof(perflibs_int_t) * (*nnz));
  std::memcpy((void *)row_indx_out, (const void *)impl_A->coo.row_indx_ptr,
              sizeof(perflibs_int_t) * (*nnz));
  std::memcpy((void *)vals_out, (const void *)impl_A->coo.vals_ptr,
              sizeof(T) * (*nnz));
}

template <typename T>
perflibs_status_t spmat_export_coo(perflibs_const_spmat_t A, perflibs_int_t *m,
                                   perflibs_int_t *n, perflibs_int_t *nnz,
                                   perflibs_int_t **row_indx,
                                   perflibs_int_t **col_indx, T **vals) {

  // Null pointer check
  if (!m || !n || !nnz || !row_indx || !col_indx || !vals) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  auto [param_status, A_copy] = coo_param_check<T>(A, m, n, nnz);
  if (param_status != PERFLIBS_STATUS_SUCCESS) {
    return param_status;
  }

  // A deep copy may have been made, so use the pointer to the copy if it's not
  // empty
  auto impl_A =
      A_copy.get()
          ? A_copy.get()
          : reinterpret_cast<const perflibs_spmat_impl_t<T> *>(A->impl);

  perflibs_int_t *col_indx_out =
      (perflibs_int_t *)std::malloc(sizeof(perflibs_int_t) * (*nnz));
  perflibs_int_t *row_indx_out =
      (perflibs_int_t *)std::malloc(sizeof(perflibs_int_t) * (*nnz));
  T *vals_out = (T *)std::malloc(sizeof(T) * (*nnz));

  coo_populate_arrays(impl_A, nnz, row_indx_out, col_indx_out, vals_out);

  // Point to user arrays
  *col_indx = col_indx_out;
  *row_indx = row_indx_out;
  *vals = vals_out;

  return PERFLIBS_STATUS_SUCCESS;
}

template perflibs_status_t
spmat_export_coo<float>(perflibs_const_spmat_t A, perflibs_int_t *m,
                        perflibs_int_t *n, perflibs_int_t *nnz,
                        perflibs_int_t **row_indx, perflibs_int_t **col_indx,
                        float **vals);
template perflibs_status_t
spmat_export_coo<double>(perflibs_const_spmat_t A, perflibs_int_t *m,
                         perflibs_int_t *n, perflibs_int_t *nnz,
                         perflibs_int_t **row_indx, perflibs_int_t **col_indx,
                         double **vals);
template perflibs_status_t spmat_export_coo<std::complex<float>>(
    perflibs_const_spmat_t A, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *nnz, perflibs_int_t **row_indx, perflibs_int_t **col_indx,
    std::complex<float> **vals);
template perflibs_status_t spmat_export_coo<std::complex<double>>(
    perflibs_const_spmat_t A, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *nnz, perflibs_int_t **row_indx, perflibs_int_t **col_indx,
    std::complex<double> **vals);

/* -------------------------------------------------
 * Dense
 * ------------------------------------------------*/

/// Handles all 4 types of conversion between ROW_MAJOR and COL_MAJOR formats.
template <typename T>
void set_dense_arrays(perflibs_int_t m, perflibs_int_t n, perflibs_int_t lda,
                      const T *matrix_array, T *user_array, bool transpose) {
  for (perflibs_int_t i = 0; i < m; i++) {
    for (perflibs_int_t j = 0; j < n; j++) {
      perflibs_int_t index = transpose ? i + lda * j : i * lda + j;
      user_array[i * n + j] = matrix_array[index];
    }
  }
}

template <typename T>
std::pair<perflibs_status_t, std::unique_ptr<perflibs_spmat_impl_t<T>>>
dense_param_check(perflibs_const_spmat_t A, enum perflibs_dense_layout layout,
                  perflibs_int_t *m, perflibs_int_t *n, perflibs_int_t *lda,
                  perflibs_int_t *size) {

  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);

  // matrix dimension check
  if (impl_A->m <= 0 || impl_A->n <= 0) {
    return {PERFLIBS_STATUS_INPUT_PARAMETER_ERROR, nullptr};
  }
  // Bad layout check
  if (layout != PERFLIBS_COL_MAJOR && layout != PERFLIBS_ROW_MAJOR) {
    return {PERFLIBS_STATUS_INPUT_PARAMETER_ERROR, nullptr};
  }
  // Conversion Cases
  // if matrix object is not already in DENSE, then convert.
  std::unique_ptr<perflibs_spmat_impl_t<T>> A_copy(nullptr);
  if (impl_A->spmat_format != perflibs_format_dense) {
    A_copy = std::make_unique<perflibs_spmat_impl_t<T>>(
        *reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl));
    impl_A = A_copy.get();
    auto info = convert<T>(perflibs_format_dense, impl_A);
    if (info != PERFLIBS_STATUS_SUCCESS) {
      return {info, nullptr};
    }
  }

  assert(impl_A->spmat_format == perflibs_format_dense);

  *m = impl_A->m;
  *n = impl_A->n;
  *lda = impl_A->dense.lda;
  *size = (*m) * (*n);

  return {PERFLIBS_STATUS_SUCCESS, std::move(A_copy)};
}

template <typename T>
void dense_populate_arrays(const perflibs_spmat_impl_t<T> *impl_A,
                           const enum perflibs_dense_layout layout,
                           const perflibs_int_t nrows,
                           const perflibs_int_t ncols, const perflibs_int_t lda,
                           T *vals_out) {
  // col2col
  if (layout == PERFLIBS_COL_MAJOR &&
      impl_A->dense.layout == PERFLIBS_COL_MAJOR) {
    set_dense_arrays(ncols, nrows, lda, impl_A->dense.vals_ptr, vals_out,
                     false);
  }
  // row2row
  else if (layout == PERFLIBS_ROW_MAJOR &&
           impl_A->dense.layout == PERFLIBS_ROW_MAJOR) {
    set_dense_arrays(nrows, ncols, lda, impl_A->dense.vals_ptr, vals_out,
                     false);
  }
  // col2row
  else if (layout == PERFLIBS_ROW_MAJOR &&
           impl_A->dense.layout == PERFLIBS_COL_MAJOR) {
    set_dense_arrays(nrows, ncols, lda, impl_A->dense.vals_ptr, vals_out, true);
  }
  // row2col
  else {
    set_dense_arrays(ncols, nrows, lda, impl_A->dense.vals_ptr, vals_out, true);
  }
}

template <typename T>
perflibs_status_t
spmat_export_dense(perflibs_const_spmat_t A, enum perflibs_dense_layout layout,
                   perflibs_int_t *m, perflibs_int_t *n, T **vals) {

  // Null pointer check
  if (!m || !n || !vals) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  perflibs_int_t lda = -1;
  perflibs_int_t size = -1;

  auto [param_status, A_copy] =
      dense_param_check<T>(A, layout, m, n, &lda, &size);
  if (param_status != PERFLIBS_STATUS_SUCCESS) {
    return param_status;
  }

  // A deep copy may have been made, so use the pointer to the copy if it's not
  // empty
  auto impl_A =
      A_copy.get()
          ? A_copy.get()
          : reinterpret_cast<const perflibs_spmat_impl_t<T> *>(A->impl);

  perflibs_int_t nrows = *m;
  perflibs_int_t ncols = *n;

  // allocate memory depending on size/layout of dense matrix
  T *vals_out = (T *)std::malloc(sizeof(T) * size);

  dense_populate_arrays(impl_A, layout, nrows, ncols, lda, vals_out);

  // Point to user array
  *vals = vals_out;

  return PERFLIBS_STATUS_SUCCESS;
}

template perflibs_status_t
spmat_export_dense<float>(perflibs_const_spmat_t A,
                          enum perflibs_dense_layout layout, perflibs_int_t *m,
                          perflibs_int_t *n, float **vals);
template perflibs_status_t
spmat_export_dense<double>(perflibs_const_spmat_t A,
                           enum perflibs_dense_layout layout, perflibs_int_t *m,
                           perflibs_int_t *n, double **vals);
template perflibs_status_t spmat_export_dense<std::complex<float>>(
    perflibs_const_spmat_t A, enum perflibs_dense_layout layout,
    perflibs_int_t *m, perflibs_int_t *n, std::complex<float> **vals);
template perflibs_status_t spmat_export_dense<std::complex<double>>(
    perflibs_const_spmat_t A, enum perflibs_dense_layout layout,
    perflibs_int_t *m, perflibs_int_t *n, std::complex<double> **vals);

/* -------------------------------------------------
 * BSR
 * ------------------------------------------------*/

template <typename T>
void transpose_bsr_arrays(perflibs_int_t block_size, perflibs_int_t nnzb,
                          const T *matrix_array, T *user_array) {
  for (perflibs_int_t block = 0; block < nnzb; block++) {
    auto block_offset = block_size * block_size * block;
    for (perflibs_int_t i = 0; i < block_size; i++) {
      for (perflibs_int_t j = 0; j < block_size; j++) {
        auto index = block_offset + i * block_size + j;
        auto reindex = block_offset + i + block_size * j;
        user_array[index] = matrix_array[reindex];
      }
    }
  }
}

template <typename T>
std::pair<perflibs_status_t, std::unique_ptr<perflibs_spmat_impl_t<T>>>
bsr_param_check(perflibs_const_spmat_t A,
                const perflibs_dense_layout block_layout,
                const perflibs_int_t index_base, perflibs_int_t *m,
                perflibs_int_t *n, perflibs_int_t *block_size,
                perflibs_int_t *nnzb, perflibs_int_t *nrowsb,
                perflibs_int_t *nnz, perflibs_int_t *index_base_A) {

  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl);

  // Basic Matrix dimension check
  if (impl_A->m <= 0 || impl_A->n <= 0) {
    return {PERFLIBS_STATUS_INPUT_PARAMETER_ERROR, nullptr};
  }
  // Check for valid index_base input
  if (index_base != 0 && index_base != 1) {
    return {PERFLIBS_STATUS_INPUT_PARAMETER_ERROR, nullptr};
  }
  // Input layout request check
  if (block_layout != PERFLIBS_COL_MAJOR &&
      block_layout != PERFLIBS_ROW_MAJOR) {
    return {PERFLIBS_STATUS_INPUT_PARAMETER_ERROR, nullptr};
  }
  // ------------------------------------
  // Conversion cases
  //-------------------------------------
  std::unique_ptr<perflibs_spmat_impl_t<T>> A_copy(nullptr);
  if (impl_A->spmat_format != perflibs_format_bsr) {
    A_copy = std::make_unique<perflibs_spmat_impl_t<T>>(
        *reinterpret_cast<perflibs_spmat_impl_t<T> *>(A->impl));
    impl_A = A_copy.get();
    auto info = convert<T>(perflibs_format_bsr, impl_A);
    if (info != PERFLIBS_STATUS_SUCCESS) {
      return {info, nullptr};
    }
  }

  assert(impl_A->spmat_format == perflibs_format_bsr);

  *m = impl_A->m;
  *n = impl_A->n;
  *block_size = impl_A->bsr.block_size;
  *nnzb = impl_A->bsr.nnzb;
  *nrowsb = impl_A->bsr.nrowsb;
  *nnz = impl_A->nnz;
  *index_base_A = impl_A->index_base;

  return {PERFLIBS_STATUS_SUCCESS, std::move(A_copy)};
}

template <typename T>
void bsr_populate_arrays(const perflibs_spmat_impl_t<T> *impl_A,
                         const perflibs_dense_layout block_layout,
                         const perflibs_int_t index_base_A,
                         const perflibs_int_t index_base,
                         const perflibs_int_t block_size,
                         const perflibs_int_t nnz, const perflibs_int_t nnzb,
                         const perflibs_int_t nrowsb,
                         perflibs_int_t *row_ptr_out,
                         perflibs_int_t *col_indx_out, T *vals_out) {

  /* -----------------------------------------------------
   * Index base adjustments for row and column index arrays
   * ----------------------------------------------------*/

  // If input and output have same indexing, then copy
  if (index_base_A == index_base) {
    std::memcpy((void *)row_ptr_out, (const void *)impl_A->bsr.row_ptr_ptr,
                sizeof(perflibs_int_t) * (nrowsb + 1));
    std::memcpy((void *)col_indx_out, (const void *)impl_A->bsr.col_indx_ptr,
                sizeof(perflibs_int_t) * nnzb);
  }
  // Input and output matrix have different bases then adjust row_ptr and
  // col_indx arrays
  else {
    adjust_exported_indexes(index_base_A, index_base, impl_A->bsr.row_ptr_ptr,
                            row_ptr_out, impl_A->bsr.col_indx_ptr, col_indx_out,
                            nrowsb + 1, nnzb);
  }

  /* -----------------------------------
   * Transpose adjustments for vals array
   * -----------------------------------*/

  // col2col and row2row
  if ((block_layout == PERFLIBS_COL_MAJOR &&
       impl_A->bsr.block_layout == PERFLIBS_COL_MAJOR) ||
      (block_layout == PERFLIBS_ROW_MAJOR &&
       impl_A->bsr.block_layout == PERFLIBS_ROW_MAJOR)) {
    std::memcpy((void *)vals_out, (const void *)impl_A->bsr.vals_ptr,
                sizeof(T) * nnz);
  }
  // col2row and row2col - symmetric as dense blocks are square
  else {
    transpose_bsr_arrays<T>(block_size, nnzb, impl_A->bsr.vals_ptr, vals_out);
  }
}

template <typename T>
perflibs_status_t
spmat_export_bsr(perflibs_const_spmat_t A, perflibs_dense_layout block_layout,
                 perflibs_int_t index_base, perflibs_int_t *m,
                 perflibs_int_t *n, perflibs_int_t *block_size,
                 perflibs_int_t **row_ptr, perflibs_int_t **col_indx,
                 T **vals) {

  // Null Pointer Check
  if (!m || !n || !block_size || !row_ptr || !col_indx || !vals) {
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }

  perflibs_int_t nnzb = -1;
  perflibs_int_t nrowsb = -1;
  perflibs_int_t nnz = -1;
  perflibs_int_t index_base_A = -1;

  auto [param_status, A_copy] =
      bsr_param_check<T>(A, block_layout, index_base, m, n, block_size, &nnzb,
                         &nrowsb, &nnz, &index_base_A);
  if (param_status != PERFLIBS_STATUS_SUCCESS) {
    return param_status;
  }

  // A deep copy may have been made, so use the pointer to the copy if it's not
  // empty
  auto impl_A =
      A_copy.get()
          ? A_copy.get()
          : reinterpret_cast<const perflibs_spmat_impl_t<T> *>(A->impl);

  // Assign new name for readability
  perflibs_int_t sizeb = *block_size;

  // Memory allocation
  perflibs_int_t *row_ptr_out =
      (perflibs_int_t *)std::malloc(sizeof(perflibs_int_t) * (nrowsb + 1));
  perflibs_int_t *col_indx_out =
      (perflibs_int_t *)std::malloc(sizeof(perflibs_int_t) * nnzb);
  T *vals_out = (T *)std::malloc(sizeof(T) * nnz);

  bsr_populate_arrays<T>(impl_A, block_layout, index_base_A, index_base, sizeb,
                         nnz, nnzb, nrowsb, row_ptr_out, col_indx_out,
                         vals_out);

  // Point to user arrays
  *row_ptr = row_ptr_out;
  *col_indx = col_indx_out;
  *vals = vals_out;

  return PERFLIBS_STATUS_SUCCESS;
}

template perflibs_status_t spmat_export_bsr<float>(
    perflibs_const_spmat_t A, perflibs_dense_layout block_layout,
    perflibs_int_t index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *block_size, perflibs_int_t **row_ptr,
    perflibs_int_t **col_indx, float **vals);
template perflibs_status_t spmat_export_bsr<double>(
    perflibs_const_spmat_t A, perflibs_dense_layout block_layout,
    perflibs_int_t index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *block_size, perflibs_int_t **row_ptr,
    perflibs_int_t **col_indx, double **vals);
template perflibs_status_t spmat_export_bsr<std::complex<float>>(
    perflibs_const_spmat_t A, perflibs_dense_layout block_layout,
    perflibs_int_t index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *block_size, perflibs_int_t **row_ptr,
    perflibs_int_t **col_indx, std::complex<float> **vals);
template perflibs_status_t spmat_export_bsr<std::complex<double>>(
    perflibs_const_spmat_t A, perflibs_dense_layout block_layout,
    perflibs_int_t index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *block_size, perflibs_int_t **row_ptr,
    perflibs_int_t **col_indx, std::complex<double> **vals);

} // end of namespace perflibs::sparse

// C interfaces for Fortran-callable subroutines
extern "C" {
void FTN_SYMB(perflibs_csr_param_check_s)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<float> **impl_A,
    const perflibs_int_t *index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *nnz, perflibs_int_t *index_base_A, perflibs_int_t *info) {
  auto [ret, A_copy] = perflibs::sparse::csr_param_check<float>(
      *A, *index_base, m, n, nnz, index_base_A);
  *info = ret;
  // We release the unique_ptr here since it needs to exist in the Fortran
  // subroutine which allocates arrays passed back to the user. We will manually
  // delete in the *_populate_arrays_*_ functions below, which is the last thing
  // called by the Fortran subroutines.
  *impl_A =
      A_copy.get()
          ? A_copy.release()
          : reinterpret_cast<const perflibs_spmat_impl_t<float> *>((*A)->impl);
}

void FTN_SYMB(perflibs_csr_populate_arrays_s)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<float> **impl_A,
    const perflibs_int_t *nrows, const perflibs_int_t *nnz,
    const perflibs_int_t *index_base_A, const perflibs_int_t *index_base,
    perflibs_int_t *row_ptr_out, perflibs_int_t *col_indx_out,
    float *vals_out) {
  perflibs::sparse::csr_populate_arrays<float>(
      *impl_A, *nrows, *nnz, *index_base_A, *index_base, row_ptr_out,
      col_indx_out, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_csr_param_check_d)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<double> **impl_A,
    const perflibs_int_t *index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *nnz, perflibs_int_t *index_base_A, perflibs_int_t *info) {
  auto [ret, A_copy] = perflibs::sparse::csr_param_check<double>(
      *A, *index_base, m, n, nnz, index_base_A);
  *info = ret;
  *impl_A =
      A_copy.get()
          ? A_copy.release()
          : reinterpret_cast<const perflibs_spmat_impl_t<double> *>((*A)->impl);
}

void FTN_SYMB(perflibs_csr_populate_arrays_d)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<double> **impl_A,
    const perflibs_int_t *nrows, const perflibs_int_t *nnz,
    const perflibs_int_t *index_base_A, const perflibs_int_t *index_base,
    perflibs_int_t *row_ptr_out, perflibs_int_t *col_indx_out,
    double *vals_out) {
  perflibs::sparse::csr_populate_arrays<double>(
      *impl_A, *nrows, *nnz, *index_base_A, *index_base, row_ptr_out,
      col_indx_out, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_csr_param_check_c)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<float>> **impl_A,
    const perflibs_int_t *index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *nnz, perflibs_int_t *index_base_A, perflibs_int_t *info) {
  auto [ret, A_copy] = perflibs::sparse::csr_param_check<std::complex<float>>(
      *A, *index_base, m, n, nnz, index_base_A);
  *info = ret;
  *impl_A =
      A_copy.get()
          ? A_copy.release()
          : reinterpret_cast<
                const perflibs_spmat_impl_t<std::complex<float>> *>((*A)->impl);
}

void FTN_SYMB(perflibs_csr_populate_arrays_c)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<float>> **impl_A,
    const perflibs_int_t *nrows, const perflibs_int_t *nnz,
    const perflibs_int_t *index_base_A, const perflibs_int_t *index_base,
    perflibs_int_t *row_ptr_out, perflibs_int_t *col_indx_out,
    std::complex<float> *vals_out) {
  perflibs::sparse::csr_populate_arrays<std::complex<float>>(
      *impl_A, *nrows, *nnz, *index_base_A, *index_base, row_ptr_out,
      col_indx_out, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_csr_param_check_z)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<double>> **impl_A,
    const perflibs_int_t *index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *nnz, perflibs_int_t *index_base_A, perflibs_int_t *info) {
  auto [ret, A_copy] = perflibs::sparse::csr_param_check<std::complex<double>>(
      *A, *index_base, m, n, nnz, index_base_A);
  *info = ret;
  *impl_A = A_copy.get()
                ? A_copy.release()
                : reinterpret_cast<
                      const perflibs_spmat_impl_t<std::complex<double>> *>(
                      (*A)->impl);
}

void FTN_SYMB(perflibs_csr_populate_arrays_z)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<double>> **impl_A,
    const perflibs_int_t *nrows, const perflibs_int_t *nnz,
    const perflibs_int_t *index_base_A, const perflibs_int_t *index_base,
    perflibs_int_t *row_ptr_out, perflibs_int_t *col_indx_out,
    std::complex<double> *vals_out) {
  perflibs::sparse::csr_populate_arrays<std::complex<double>>(
      *impl_A, *nrows, *nnz, *index_base_A, *index_base, row_ptr_out,
      col_indx_out, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_csc_param_check_s)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<float> **impl_A,
    const perflibs_int_t *index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *nnz, perflibs_int_t *index_base_A, perflibs_int_t *info) {
  auto [ret, A_copy] = perflibs::sparse::csc_param_check<float>(
      *A, *index_base, m, n, nnz, index_base_A);
  *info = ret;
  *impl_A =
      A_copy.get()
          ? A_copy.release()
          : reinterpret_cast<const perflibs_spmat_impl_t<float> *>((*A)->impl);
}

void FTN_SYMB(perflibs_csc_populate_arrays_s)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<float> **impl_A,
    const perflibs_int_t *ncols, const perflibs_int_t *nnz,
    const perflibs_int_t *index_base_A, const perflibs_int_t *index_base,
    perflibs_int_t *row_indx_out, perflibs_int_t *col_ptr_out,
    float *vals_out) {
  perflibs::sparse::csc_populate_arrays<float>(
      *impl_A, *ncols, *nnz, *index_base_A, *index_base, row_indx_out,
      col_ptr_out, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_csc_param_check_d)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<double> **impl_A,
    const perflibs_int_t *index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *nnz, perflibs_int_t *index_base_A, perflibs_int_t *info) {
  auto [ret, A_copy] = perflibs::sparse::csc_param_check<double>(
      *A, *index_base, m, n, nnz, index_base_A);
  *info = ret;
  *impl_A =
      A_copy.get()
          ? A_copy.release()
          : reinterpret_cast<const perflibs_spmat_impl_t<double> *>((*A)->impl);
}

void FTN_SYMB(perflibs_csc_populate_arrays_d)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<double> **impl_A,
    const perflibs_int_t *ncols, const perflibs_int_t *nnz,
    const perflibs_int_t *index_base_A, const perflibs_int_t *index_base,
    perflibs_int_t *row_indx_out, perflibs_int_t *col_ptr_out,
    double *vals_out) {
  perflibs::sparse::csc_populate_arrays<double>(
      *impl_A, *ncols, *nnz, *index_base_A, *index_base, row_indx_out,
      col_ptr_out, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_csc_param_check_c)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<float>> **impl_A,
    const perflibs_int_t *index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *nnz, perflibs_int_t *index_base_A, perflibs_int_t *info) {
  auto [ret, A_copy] = perflibs::sparse::csc_param_check<std::complex<float>>(
      *A, *index_base, m, n, nnz, index_base_A);
  *info = ret;
  *impl_A =
      A_copy.get()
          ? A_copy.release()
          : reinterpret_cast<
                const perflibs_spmat_impl_t<std::complex<float>> *>((*A)->impl);
}

void FTN_SYMB(perflibs_csc_populate_arrays_c)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<float>> **impl_A,
    const perflibs_int_t *ncols, const perflibs_int_t *nnz,
    const perflibs_int_t *index_base_A, const perflibs_int_t *index_base,
    perflibs_int_t *row_indx_out, perflibs_int_t *col_ptr_out,
    std::complex<float> *vals_out) {
  perflibs::sparse::csc_populate_arrays<std::complex<float>>(
      *impl_A, *ncols, *nnz, *index_base_A, *index_base, row_indx_out,
      col_ptr_out, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_csc_param_check_z)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<double>> **impl_A,
    const perflibs_int_t *index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *nnz, perflibs_int_t *index_base_A, perflibs_int_t *info) {
  auto [ret, A_copy] = perflibs::sparse::csc_param_check<std::complex<double>>(
      *A, *index_base, m, n, nnz, index_base_A);
  *info = ret;
  *impl_A = A_copy.get()
                ? A_copy.release()
                : reinterpret_cast<
                      const perflibs_spmat_impl_t<std::complex<double>> *>(
                      (*A)->impl);
}

void FTN_SYMB(perflibs_csc_populate_arrays_z)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<double>> **impl_A,
    const perflibs_int_t *ncols, const perflibs_int_t *nnz,
    const perflibs_int_t *index_base_A, const perflibs_int_t *index_base,
    perflibs_int_t *row_indx_out, perflibs_int_t *col_ptr_out,
    std::complex<double> *vals_out) {
  perflibs::sparse::csc_populate_arrays<std::complex<double>>(
      *impl_A, *ncols, *nnz, *index_base_A, *index_base, row_indx_out,
      col_ptr_out, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_coo_param_check_s)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<float> **impl_A,
    perflibs_int_t *m, perflibs_int_t *n, perflibs_int_t *nnz,
    perflibs_int_t *info) {
  auto [ret, A_copy] = perflibs::sparse::coo_param_check<float>(*A, m, n, nnz);
  *info = ret;
  *impl_A =
      A_copy.get()
          ? A_copy.release()
          : reinterpret_cast<const perflibs_spmat_impl_t<float> *>((*A)->impl);
}

void FTN_SYMB(perflibs_coo_populate_arrays_s)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<float> **impl_A,
    const perflibs_int_t *nnz, perflibs_int_t *row_indx_out,
    perflibs_int_t *col_indx_out, float *vals_out) {
  perflibs::sparse::coo_populate_arrays<float>(*impl_A, nnz, row_indx_out,
                                               col_indx_out, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_coo_param_check_d)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<double> **impl_A,
    perflibs_int_t *m, perflibs_int_t *n, perflibs_int_t *nnz,
    perflibs_int_t *info) {
  auto [ret, A_copy] = perflibs::sparse::coo_param_check<double>(*A, m, n, nnz);
  *info = ret;
  *impl_A =
      A_copy.get()
          ? A_copy.release()
          : reinterpret_cast<const perflibs_spmat_impl_t<double> *>((*A)->impl);
}

void FTN_SYMB(perflibs_coo_populate_arrays_d)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<double> **impl_A,
    const perflibs_int_t *nnz, perflibs_int_t *row_indx_out,
    perflibs_int_t *col_indx_out, double *vals_out) {
  perflibs::sparse::coo_populate_arrays<double>(*impl_A, nnz, row_indx_out,
                                                col_indx_out, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_coo_param_check_c)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<float>> **impl_A,
    perflibs_int_t *m, perflibs_int_t *n, perflibs_int_t *nnz,
    perflibs_int_t *info) {
  auto [ret, A_copy] =
      perflibs::sparse::coo_param_check<std::complex<float>>(*A, m, n, nnz);
  *info = ret;
  *impl_A =
      A_copy.get()
          ? A_copy.release()
          : reinterpret_cast<
                const perflibs_spmat_impl_t<std::complex<float>> *>((*A)->impl);
}

void FTN_SYMB(perflibs_coo_populate_arrays_c)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<float>> **impl_A,
    const perflibs_int_t *nnz, perflibs_int_t *row_indx_out,
    perflibs_int_t *col_indx_out, std::complex<float> *vals_out) {
  perflibs::sparse::coo_populate_arrays<std::complex<float>>(
      *impl_A, nnz, row_indx_out, col_indx_out, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_coo_param_check_z)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<double>> **impl_A,
    perflibs_int_t *m, perflibs_int_t *n, perflibs_int_t *nnz,
    perflibs_int_t *info) {
  auto [ret, A_copy] =
      perflibs::sparse::coo_param_check<std::complex<double>>(*A, m, n, nnz);
  *info = ret;
  *impl_A = A_copy.get()
                ? A_copy.release()
                : reinterpret_cast<
                      const perflibs_spmat_impl_t<std::complex<double>> *>(
                      (*A)->impl);
}

void FTN_SYMB(perflibs_coo_populate_arrays_z)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<double>> **impl_A,
    const perflibs_int_t *nnz, perflibs_int_t *row_indx_out,
    perflibs_int_t *col_indx_out, std::complex<double> *vals_out) {
  perflibs::sparse::coo_populate_arrays<std::complex<double>>(
      *impl_A, nnz, row_indx_out, col_indx_out, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_dense_param_check_s)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<float> **impl_A,
    enum perflibs_dense_layout *layout, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *lda, perflibs_int_t *size, perflibs_int_t *info) {
  auto [ret, A_copy] =
      perflibs::sparse::dense_param_check<float>(*A, *layout, m, n, lda, size);
  *info = ret;
  *impl_A =
      A_copy.get()
          ? A_copy.release()
          : reinterpret_cast<const perflibs_spmat_impl_t<float> *>((*A)->impl);
}

void FTN_SYMB(perflibs_dense_populate_arrays_s)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<float> **impl_A,
    const enum perflibs_dense_layout *layout, const perflibs_int_t *nrows,
    const perflibs_int_t *ncols, const perflibs_int_t *lda, float *vals_out) {
  perflibs::sparse::dense_populate_arrays<float>(*impl_A, *layout, *nrows,
                                                 *ncols, *lda, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_dense_param_check_d)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<double> **impl_A,
    enum perflibs_dense_layout *layout, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *lda, perflibs_int_t *size, perflibs_int_t *info) {
  auto [ret, A_copy] =
      perflibs::sparse::dense_param_check<double>(*A, *layout, m, n, lda, size);
  *info = ret;
  *impl_A =
      A_copy.get()
          ? A_copy.release()
          : reinterpret_cast<const perflibs_spmat_impl_t<double> *>((*A)->impl);
}

void FTN_SYMB(perflibs_dense_populate_arrays_d)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<double> **impl_A,
    const enum perflibs_dense_layout *layout, const perflibs_int_t *nrows,
    const perflibs_int_t *ncols, const perflibs_int_t *lda, double *vals_out) {
  perflibs::sparse::dense_populate_arrays<double>(*impl_A, *layout, *nrows,
                                                  *ncols, *lda, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_dense_param_check_c)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<float>> **impl_A,
    enum perflibs_dense_layout *layout, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *lda, perflibs_int_t *size, perflibs_int_t *info) {
  auto [ret, A_copy] = perflibs::sparse::dense_param_check<std::complex<float>>(
      *A, *layout, m, n, lda, size);
  *info = ret;
  *impl_A =
      A_copy.get()
          ? A_copy.release()
          : reinterpret_cast<
                const perflibs_spmat_impl_t<std::complex<float>> *>((*A)->impl);
}

void FTN_SYMB(perflibs_dense_populate_arrays_c)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<float>> **impl_A,
    const enum perflibs_dense_layout *layout, const perflibs_int_t *nrows,
    const perflibs_int_t *ncols, const perflibs_int_t *lda,
    std::complex<float> *vals_out) {
  perflibs::sparse::dense_populate_arrays<std::complex<float>>(
      *impl_A, *layout, *nrows, *ncols, *lda, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_dense_param_check_z)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<double>> **impl_A,
    enum perflibs_dense_layout *layout, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *lda, perflibs_int_t *size, perflibs_int_t *info) {
  auto [ret, A_copy] =
      perflibs::sparse::dense_param_check<std::complex<double>>(*A, *layout, m,
                                                                n, lda, size);
  *info = ret;
  *impl_A = A_copy.get()
                ? A_copy.release()
                : reinterpret_cast<
                      const perflibs_spmat_impl_t<std::complex<double>> *>(
                      (*A)->impl);
}

void FTN_SYMB(perflibs_dense_populate_arrays_z)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<double>> **impl_A,
    const enum perflibs_dense_layout *layout, const perflibs_int_t *nrows,
    const perflibs_int_t *ncols, const perflibs_int_t *lda,
    std::complex<double> *vals_out) {
  perflibs::sparse::dense_populate_arrays<std::complex<double>>(
      *impl_A, *layout, *nrows, *ncols, *lda, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_bsr_param_check_s)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<float> **impl_A,
    const perflibs_dense_layout *block_layout, const perflibs_int_t *index_base,
    perflibs_int_t *m, perflibs_int_t *n, perflibs_int_t *block_size,
    perflibs_int_t *nnzb, perflibs_int_t *nrowsb, perflibs_int_t *nnz,
    perflibs_int_t *index_base_A, perflibs_int_t *info) {

  auto [ret, A_copy] = perflibs::sparse::bsr_param_check<float>(
      *A, *block_layout, *index_base, m, n, block_size, nnzb, nrowsb, nnz,
      index_base_A);
  *info = ret;
  *impl_A =
      A_copy.get()
          ? A_copy.release()
          : reinterpret_cast<const perflibs_spmat_impl_t<float> *>((*A)->impl);
}

void FTN_SYMB(perflibs_bsr_populate_arrays_s)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<float> **impl_A,
    const perflibs_dense_layout *block_layout,
    const perflibs_int_t *index_base_A, const perflibs_int_t *index_base,
    const perflibs_int_t *block_size, const perflibs_int_t *nnz,
    const perflibs_int_t *nnzb, const perflibs_int_t *nrowsb,
    perflibs_int_t *row_ptr_out, perflibs_int_t *col_indx_out,
    float *vals_out) {

  perflibs::sparse::bsr_populate_arrays<float>(
      *impl_A, *block_layout, *index_base_A, *index_base, *block_size, *nnz,
      *nnzb, *nrowsb, row_ptr_out, col_indx_out, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_bsr_param_check_d)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<double> **impl_A,
    const perflibs_dense_layout *block_layout, const perflibs_int_t *index_base,
    perflibs_int_t *m, perflibs_int_t *n, perflibs_int_t *block_size,
    perflibs_int_t *nnzb, perflibs_int_t *nrowsb, perflibs_int_t *nnz,
    perflibs_int_t *index_base_A, perflibs_int_t *info) {

  auto [ret, A_copy] = perflibs::sparse::bsr_param_check<double>(
      *A, *block_layout, *index_base, m, n, block_size, nnzb, nrowsb, nnz,
      index_base_A);
  *info = ret;
  *impl_A =
      A_copy.get()
          ? A_copy.release()
          : reinterpret_cast<const perflibs_spmat_impl_t<double> *>((*A)->impl);
}

void FTN_SYMB(perflibs_bsr_populate_arrays_d)(
    perflibs_const_spmat_t *A, const perflibs_spmat_impl_t<double> **impl_A,
    const perflibs_dense_layout *block_layout,
    const perflibs_int_t *index_base_A, const perflibs_int_t *index_base,
    const perflibs_int_t *block_size, const perflibs_int_t *nnz,
    const perflibs_int_t *nnzb, const perflibs_int_t *nrowsb,
    perflibs_int_t *row_ptr_out, perflibs_int_t *col_indx_out,
    double *vals_out) {

  perflibs::sparse::bsr_populate_arrays<double>(
      *impl_A, *block_layout, *index_base_A, *index_base, *block_size, *nnz,
      *nnzb, *nrowsb, row_ptr_out, col_indx_out, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_bsr_param_check_c)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<float>> **impl_A,
    const perflibs_dense_layout *block_layout, const perflibs_int_t *index_base,
    perflibs_int_t *m, perflibs_int_t *n, perflibs_int_t *block_size,
    perflibs_int_t *nnzb, perflibs_int_t *nrowsb, perflibs_int_t *nnz,
    perflibs_int_t *index_base_A, perflibs_int_t *info) {

  auto [ret, A_copy] = perflibs::sparse::bsr_param_check<std::complex<float>>(
      *A, *block_layout, *index_base, m, n, block_size, nnzb, nrowsb, nnz,
      index_base_A);
  *info = ret;
  *impl_A =
      A_copy.get()
          ? A_copy.release()
          : reinterpret_cast<
                const perflibs_spmat_impl_t<std::complex<float>> *>((*A)->impl);
}

void FTN_SYMB(perflibs_bsr_populate_arrays_c)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<float>> **impl_A,
    const perflibs_dense_layout *block_layout,
    const perflibs_int_t *index_base_A, const perflibs_int_t *index_base,
    const perflibs_int_t *block_size, const perflibs_int_t *nnz,
    const perflibs_int_t *nnzb, const perflibs_int_t *nrowsb,
    perflibs_int_t *row_ptr_out, perflibs_int_t *col_indx_out,
    std::complex<float> *vals_out) {

  perflibs::sparse::bsr_populate_arrays<std::complex<float>>(
      *impl_A, *block_layout, *index_base_A, *index_base, *block_size, *nnz,
      *nnzb, *nrowsb, row_ptr_out, col_indx_out, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}

void FTN_SYMB(perflibs_bsr_param_check_z)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<double>> **impl_A,
    const perflibs_dense_layout *block_layout, const perflibs_int_t *index_base,
    perflibs_int_t *m, perflibs_int_t *n, perflibs_int_t *block_size,
    perflibs_int_t *nnzb, perflibs_int_t *nrowsb, perflibs_int_t *nnz,
    perflibs_int_t *index_base_A, perflibs_int_t *info) {

  auto [ret, A_copy] = perflibs::sparse::bsr_param_check<std::complex<double>>(
      *A, *block_layout, *index_base, m, n, block_size, nnzb, nrowsb, nnz,
      index_base_A);
  *info = ret;
  *impl_A = A_copy.get()
                ? A_copy.release()
                : reinterpret_cast<
                      const perflibs_spmat_impl_t<std::complex<double>> *>(
                      (*A)->impl);
}

void FTN_SYMB(perflibs_bsr_populate_arrays_z)(
    perflibs_const_spmat_t *A,
    const perflibs_spmat_impl_t<std::complex<double>> **impl_A,
    const perflibs_dense_layout *block_layout,
    const perflibs_int_t *index_base_A, const perflibs_int_t *index_base,
    const perflibs_int_t *block_size, const perflibs_int_t *nnz,
    const perflibs_int_t *nnzb, const perflibs_int_t *nrowsb,
    perflibs_int_t *row_ptr_out, perflibs_int_t *col_indx_out,
    std::complex<double> *vals_out) {

  perflibs::sparse::bsr_populate_arrays<std::complex<double>>(
      *impl_A, *block_layout, *index_base_A, *index_base, *block_size, *nnz,
      *nnzb, *nrowsb, row_ptr_out, col_indx_out, vals_out);
  if ((*A)->impl != *impl_A) {
    delete *impl_A;
  }
}
}
