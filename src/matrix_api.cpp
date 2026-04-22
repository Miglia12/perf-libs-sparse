/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "c_api_complex_abi.hpp"
#include "export.hpp"
#include "matrix_state.hpp"
#include "matvec.hpp"
#include "object_helpers.hpp"
#include "util.hpp"

perflibs_status_t
perflibs_spmat_create_csr_s(perflibs_spmat_top_t **A, perflibs_int_t m,
                            perflibs_int_t n, const perflibs_int_t *row_ptr,
                            const perflibs_int_t *col_indx, const float *vals,
                            perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_csr(A, m, n, row_ptr, col_indx,
                                                vals, flags);
}

perflibs_status_t
perflibs_spmat_create_csr_d(perflibs_spmat_top_t **A, perflibs_int_t m,
                            perflibs_int_t n, const perflibs_int_t *row_ptr,
                            const perflibs_int_t *col_indx, const double *vals,
                            perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_csr(A, m, n, row_ptr, col_indx,
                                                vals, flags);
}

perflibs_status_t perflibs_spmat_create_csr_c(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const perflibs_singlecomplex_t *vals, perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_csr(
      A, m, n, row_ptr, col_indx,
      reinterpret_cast<const std::complex<float> *>(vals), flags);
}

perflibs_status_t perflibs_spmat_create_csr_z(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const perflibs_doublecomplex_t *vals, perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_csr(
      A, m, n, row_ptr, col_indx,
      reinterpret_cast<const std::complex<double> *>(vals), flags);
}

perflibs_status_t
perflibs_spmat_create_csc_s(perflibs_spmat_top_t **A, perflibs_int_t m,
                            perflibs_int_t n, const perflibs_int_t *row_indx,
                            const perflibs_int_t *col_ptr, const float *vals,
                            perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_csc(A, m, n, row_indx, col_ptr,
                                                vals, flags);
}

perflibs_status_t
perflibs_spmat_create_csc_d(perflibs_spmat_top_t **A, perflibs_int_t m,
                            perflibs_int_t n, const perflibs_int_t *row_indx,
                            const perflibs_int_t *col_ptr, const double *vals,
                            perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_csc(A, m, n, row_indx, col_ptr,
                                                vals, flags);
}

perflibs_status_t perflibs_spmat_create_csc_c(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const perflibs_singlecomplex_t *vals, perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_csc(
      A, m, n, row_indx, col_ptr,
      reinterpret_cast<const std::complex<float> *>(vals), flags);
}

perflibs_status_t perflibs_spmat_create_csc_z(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const perflibs_doublecomplex_t *vals, perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_csc(
      A, m, n, row_indx, col_ptr,
      reinterpret_cast<const std::complex<double> *>(vals), flags);
}

perflibs_status_t perflibs_spmat_create_coo_s(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nnz, const perflibs_int_t *row_indx,
    const perflibs_int_t *col_indx, const float *vals, perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_coo(A, m, n, nnz, 0, row_indx,
                                                col_indx, vals, flags);
}

perflibs_status_t perflibs_spmat_create_coo_d(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nnz, const perflibs_int_t *row_indx,
    const perflibs_int_t *col_indx, const double *vals, perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_coo(A, m, n, nnz, 0, row_indx,
                                                col_indx, vals, flags);
}

perflibs_status_t perflibs_spmat_create_coo_c(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nnz, const perflibs_int_t *row_indx,
    const perflibs_int_t *col_indx, const perflibs_singlecomplex_t *vals,
    perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_coo(
      A, m, n, nnz, 0, row_indx, col_indx,
      reinterpret_cast<const std::complex<float> *>(vals), flags);
}

perflibs_status_t perflibs_spmat_create_coo_z(
    perflibs_spmat_top_t **A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nnz, const perflibs_int_t *row_indx,
    const perflibs_int_t *col_indx, const perflibs_doublecomplex_t *vals,
    perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_coo(
      A, m, n, nnz, 0, row_indx, col_indx,
      reinterpret_cast<const std::complex<double> *>(vals), flags);
}

perflibs_status_t perflibs_spmat_create_dense_s(
    perflibs_spmat_t *A, enum perflibs_dense_layout layout, perflibs_int_t m,
    perflibs_int_t n, perflibs_int_t lda, const float *vals,
    perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_dense(A, layout, m, n, lda, 0, vals,
                                                  flags);
}

perflibs_status_t perflibs_spmat_create_dense_d(
    perflibs_spmat_t *A, enum perflibs_dense_layout layout, perflibs_int_t m,
    perflibs_int_t n, perflibs_int_t lda, const double *vals,
    perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_dense(A, layout, m, n, lda, 0, vals,
                                                  flags);
}

perflibs_status_t perflibs_spmat_create_dense_c(
    perflibs_spmat_t *A, enum perflibs_dense_layout layout, perflibs_int_t m,
    perflibs_int_t n, perflibs_int_t lda, const perflibs_singlecomplex_t *vals,
    perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_dense(
      A, layout, m, n, lda, 0,
      reinterpret_cast<const std::complex<float> *>(vals), flags);
}

perflibs_status_t perflibs_spmat_create_dense_z(
    perflibs_spmat_t *A, enum perflibs_dense_layout layout, perflibs_int_t m,
    perflibs_int_t n, perflibs_int_t lda, const perflibs_doublecomplex_t *vals,
    perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_dense(
      A, layout, m, n, lda, 0,
      reinterpret_cast<const std::complex<double> *>(vals), flags);
}

perflibs_status_t perflibs_spmat_create_bsr_s(
    perflibs_spmat_t *A, enum perflibs_dense_layout block_layout,
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t block_size,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const float *vals, perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_bsr(
      A, block_layout, m, n, block_size, row_ptr, col_indx, vals, flags);
}

perflibs_status_t perflibs_spmat_create_bsr_d(
    perflibs_spmat_t *A, enum perflibs_dense_layout block_layout,
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t block_size,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const double *vals, perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_bsr(
      A, block_layout, m, n, block_size, row_ptr, col_indx, vals, flags);
}

perflibs_status_t perflibs_spmat_create_bsr_c(
    perflibs_spmat_t *A, enum perflibs_dense_layout block_layout,
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t block_size,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const perflibs_singlecomplex_t *vals, perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_bsr(
      A, block_layout, m, n, block_size, row_ptr, col_indx,
      reinterpret_cast<const std::complex<float> *>(vals), flags);
}

perflibs_status_t perflibs_spmat_create_bsr_z(
    perflibs_spmat_t *A, enum perflibs_dense_layout block_layout,
    perflibs_int_t m, perflibs_int_t n, perflibs_int_t block_size,
    const perflibs_int_t *row_ptr, const perflibs_int_t *col_indx,
    const perflibs_doublecomplex_t *vals, perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_bsr(
      A, block_layout, m, n, block_size, row_ptr, col_indx,
      reinterpret_cast<const std::complex<double> *>(vals), flags);
}

perflibs_status_t perflibs_spmat_create_supernodal_s(
    perflibs_spmat_t *A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const float *vals, const perflibs_int_t *part_indx, perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_supernodal(
      A, m, n, nsuper, nparts, super_row_ptr, super_col_indx, row_indx, col_ptr,
      vals, part_indx, flags);
}

perflibs_status_t perflibs_spmat_create_supernodal_d(
    perflibs_spmat_t *A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const double *vals, const perflibs_int_t *part_indx, perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_supernodal(
      A, m, n, nsuper, nparts, super_row_ptr, super_col_indx, row_indx, col_ptr,
      vals, part_indx, flags);
}

perflibs_status_t perflibs_spmat_create_supernodal_c(
    perflibs_spmat_t *A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const perflibs_singlecomplex_t *vals, const perflibs_int_t *part_indx,
    perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_supernodal(
      A, m, n, nsuper, nparts, super_row_ptr, super_col_indx, row_indx, col_ptr,
      reinterpret_cast<const std::complex<float> *>(vals), part_indx, flags);
}

perflibs_status_t perflibs_spmat_create_supernodal_z(
    perflibs_spmat_t *A, perflibs_int_t m, perflibs_int_t n,
    perflibs_int_t nsuper, perflibs_int_t nparts,
    const perflibs_int_t *super_row_ptr, const perflibs_int_t *super_col_indx,
    const perflibs_int_t *row_indx, const perflibs_int_t *col_ptr,
    const perflibs_doublecomplex_t *vals, const perflibs_int_t *part_indx,
    perflibs_int_t flags) {
  return perflibs::sparse::create_spmat_top_supernodal(
      A, m, n, nsuper, nparts, super_row_ptr, super_col_indx, row_indx, col_ptr,
      reinterpret_cast<const std::complex<double> *>(vals), part_indx, flags);
}

perflibs_spmat_t perflibs_spmat_create_null(perflibs_int_t m,
                                            perflibs_int_t n) {
  return perflibs::sparse::null_matrix(m, n).release();
}

perflibs_spmat_t perflibs_spmat_create_identity(perflibs_int_t n) {
  return perflibs::sparse::identity_matrix(n).release();
}

perflibs_status_t perflibs_spmat_query(perflibs_spmat_t A,
                                       perflibs_int_t *index_base,
                                       perflibs_int_t *m, perflibs_int_t *n,
                                       perflibs_int_t *nnz) {
  return perflibs::sparse::basic_query(A, index_base, m, n, nnz);
}

perflibs_status_t perflibs_spmat_destroy(perflibs_spmat_top_t *A) {
  delete A;
  return PERFLIBS_STATUS_SUCCESS;
}

perflibs_status_t perflibs_spmat_hint(perflibs_spmat_top_t *A,
                                      perflibs_sparse_hint_type hint,
                                      perflibs_sparse_hint_value value) {
  if (A->datatype == PERFLIBS_DATATYPE_SINGLE) {
    return perflibs::sparse::set_hint(
        reinterpret_cast<perflibs_spmat_impl_t<float> *>(A->impl), hint, value);
  } else if (A->datatype == PERFLIBS_DATATYPE_DOUBLE) {
    return perflibs::sparse::set_hint(
        reinterpret_cast<perflibs_spmat_impl_t<double> *>(A->impl), hint,
        value);
  } else if (A->datatype == PERFLIBS_DATATYPE_CPLXSINGLE) {
    return perflibs::sparse::set_hint(
        reinterpret_cast<perflibs_spmat_impl_t<std::complex<float>> *>(A->impl),
        hint, value);
  } else if (A->datatype == PERFLIBS_DATATYPE_CPLXDOUBLE) {
    return perflibs::sparse::set_hint(
        reinterpret_cast<perflibs_spmat_impl_t<std::complex<double>> *>(
            A->impl),
        hint, value);
  } else {
    return PERFLIBS_STATUS_EXECUTION_FAILURE;
  }
}

perflibs_status_t perflibs_spmat_update_s(perflibs_spmat_t A,
                                          perflibs_int_t n_updates,
                                          const perflibs_int_t *row_indx,
                                          const perflibs_int_t *col_indx,
                                          const float *vals) {
  return perflibs::sparse::spmat_update(A, n_updates, row_indx, col_indx, vals);
}

perflibs_status_t perflibs_spmat_update_d(perflibs_spmat_t A,
                                          perflibs_int_t n_updates,
                                          const perflibs_int_t *row_indx,
                                          const perflibs_int_t *col_indx,
                                          const double *vals) {
  return perflibs::sparse::spmat_update(A, n_updates, row_indx, col_indx, vals);
}

perflibs_status_t
perflibs_spmat_update_c(perflibs_spmat_t A, perflibs_int_t n_updates,
                        const perflibs_int_t *row_indx,
                        const perflibs_int_t *col_indx,
                        const perflibs_singlecomplex_t *vals) {
  return perflibs::sparse::spmat_update(
      A, n_updates, row_indx, col_indx,
      reinterpret_cast<const std::complex<float> *>(vals));
}

perflibs_status_t
perflibs_spmat_update_z(perflibs_spmat_t A, perflibs_int_t n_updates,
                        const perflibs_int_t *row_indx,
                        const perflibs_int_t *col_indx,
                        const perflibs_doublecomplex_t *vals) {
  return perflibs::sparse::spmat_update(
      A, n_updates, row_indx, col_indx,
      reinterpret_cast<const std::complex<double> *>(vals));
}

perflibs_status_t
perflibs_spmat_export_csr_s(perflibs_const_spmat_t A, perflibs_int_t index_base,
                            perflibs_int_t *m, perflibs_int_t *n,
                            perflibs_int_t **row_ptr, perflibs_int_t **col_indx,
                            float **vals) {
  return perflibs::sparse::spmat_export_csr(A, index_base, m, n, row_ptr,
                                            col_indx, vals);
}

perflibs_status_t
perflibs_spmat_export_csr_d(perflibs_const_spmat_t A, perflibs_int_t index_base,
                            perflibs_int_t *m, perflibs_int_t *n,
                            perflibs_int_t **row_ptr, perflibs_int_t **col_indx,
                            double **vals) {
  return perflibs::sparse::spmat_export_csr(A, index_base, m, n, row_ptr,
                                            col_indx, vals);
}

perflibs_status_t
perflibs_spmat_export_csr_c(perflibs_const_spmat_t A, perflibs_int_t index_base,
                            perflibs_int_t *m, perflibs_int_t *n,
                            perflibs_int_t **row_ptr, perflibs_int_t **col_indx,
                            perflibs_singlecomplex_t **vals) {
  return perflibs::sparse::spmat_export_csr(
      A, index_base, m, n, row_ptr, col_indx,
      reinterpret_cast<std::complex<float> **>(vals));
}

perflibs_status_t
perflibs_spmat_export_csr_z(perflibs_const_spmat_t A, perflibs_int_t index_base,
                            perflibs_int_t *m, perflibs_int_t *n,
                            perflibs_int_t **row_ptr, perflibs_int_t **col_indx,
                            perflibs_doublecomplex_t **vals) {
  return perflibs::sparse::spmat_export_csr(
      A, index_base, m, n, row_ptr, col_indx,
      reinterpret_cast<std::complex<double> **>(vals));
}

perflibs_status_t
perflibs_spmat_export_csc_s(perflibs_const_spmat_t A, perflibs_int_t index_base,
                            perflibs_int_t *m, perflibs_int_t *n,
                            perflibs_int_t **row_indx, perflibs_int_t **col_ptr,
                            float **vals) {
  return perflibs::sparse::spmat_export_csc(A, index_base, m, n, row_indx,
                                            col_ptr, vals);
}

perflibs_status_t
perflibs_spmat_export_csc_d(perflibs_const_spmat_t A, perflibs_int_t index_base,
                            perflibs_int_t *m, perflibs_int_t *n,
                            perflibs_int_t **row_indx, perflibs_int_t **col_ptr,
                            double **vals) {
  return perflibs::sparse::spmat_export_csc(A, index_base, m, n, row_indx,
                                            col_ptr, vals);
}

perflibs_status_t
perflibs_spmat_export_csc_c(perflibs_const_spmat_t A, perflibs_int_t index_base,
                            perflibs_int_t *m, perflibs_int_t *n,
                            perflibs_int_t **row_indx, perflibs_int_t **col_ptr,
                            perflibs_singlecomplex_t **vals) {
  return perflibs::sparse::spmat_export_csc(
      A, index_base, m, n, row_indx, col_ptr,
      reinterpret_cast<std::complex<float> **>(vals));
}

perflibs_status_t
perflibs_spmat_export_csc_z(perflibs_const_spmat_t A, perflibs_int_t index_base,
                            perflibs_int_t *m, perflibs_int_t *n,
                            perflibs_int_t **row_indx, perflibs_int_t **col_ptr,
                            perflibs_doublecomplex_t **vals) {
  return perflibs::sparse::spmat_export_csc(
      A, index_base, m, n, row_indx, col_ptr,
      reinterpret_cast<std::complex<double> **>(vals));
}

perflibs_status_t
perflibs_spmat_export_coo_s(perflibs_const_spmat_t A, perflibs_int_t *m,
                            perflibs_int_t *n, perflibs_int_t *nnz,
                            perflibs_int_t **row_indx,
                            perflibs_int_t **col_indx, float **vals) {
  return perflibs::sparse::spmat_export_coo(A, m, n, nnz, row_indx, col_indx,
                                            vals);
}

perflibs_status_t
perflibs_spmat_export_coo_d(perflibs_const_spmat_t A, perflibs_int_t *m,
                            perflibs_int_t *n, perflibs_int_t *nnz,
                            perflibs_int_t **row_indx,
                            perflibs_int_t **col_indx, double **vals) {
  return perflibs::sparse::spmat_export_coo(A, m, n, nnz, row_indx, col_indx,
                                            vals);
}

perflibs_status_t perflibs_spmat_export_coo_c(
    perflibs_const_spmat_t A, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *nnz, perflibs_int_t **row_indx, perflibs_int_t **col_indx,
    perflibs_singlecomplex_t **vals) {
  return perflibs::sparse::spmat_export_coo(
      A, m, n, nnz, row_indx, col_indx,
      reinterpret_cast<std::complex<float> **>(vals));
}

perflibs_status_t perflibs_spmat_export_coo_z(
    perflibs_const_spmat_t A, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *nnz, perflibs_int_t **row_indx, perflibs_int_t **col_indx,
    perflibs_doublecomplex_t **vals) {
  return perflibs::sparse::spmat_export_coo(
      A, m, n, nnz, row_indx, col_indx,
      reinterpret_cast<std::complex<double> **>(vals));
}

perflibs_status_t perflibs_spmat_export_dense_s(
    perflibs_const_spmat_t A, enum perflibs_dense_layout layout,
    perflibs_int_t *m, perflibs_int_t *n, float **vals) {
  return perflibs::sparse::spmat_export_dense(A, layout, m, n, vals);
}

perflibs_status_t perflibs_spmat_export_dense_d(
    perflibs_const_spmat_t A, enum perflibs_dense_layout layout,
    perflibs_int_t *m, perflibs_int_t *n, double **vals) {
  return perflibs::sparse::spmat_export_dense(A, layout, m, n, vals);
}

perflibs_status_t perflibs_spmat_export_dense_c(
    perflibs_const_spmat_t A, enum perflibs_dense_layout layout,
    perflibs_int_t *m, perflibs_int_t *n, perflibs_singlecomplex_t **vals) {
  return perflibs::sparse::spmat_export_dense(
      A, layout, m, n, reinterpret_cast<std::complex<float> **>(vals));
}

perflibs_status_t perflibs_spmat_export_dense_z(
    perflibs_const_spmat_t A, enum perflibs_dense_layout layout,
    perflibs_int_t *m, perflibs_int_t *n, perflibs_doublecomplex_t **vals) {
  return perflibs::sparse::spmat_export_dense(
      A, layout, m, n, reinterpret_cast<std::complex<double> **>(vals));
}

perflibs_status_t perflibs_spmat_export_bsr_s(
    perflibs_const_spmat_t A, enum perflibs_dense_layout block_layout,
    perflibs_int_t index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *block_size, perflibs_int_t **row_ptr,
    perflibs_int_t **col_indx, float **vals) {
  return perflibs::sparse::spmat_export_bsr(
      A, block_layout, index_base, m, n, block_size, row_ptr, col_indx, vals);
}

perflibs_status_t perflibs_spmat_export_bsr_d(
    perflibs_const_spmat_t A, enum perflibs_dense_layout block_layout,
    perflibs_int_t index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *block_size, perflibs_int_t **row_ptr,
    perflibs_int_t **col_indx, double **vals) {
  return perflibs::sparse::spmat_export_bsr(
      A, block_layout, index_base, m, n, block_size, row_ptr, col_indx, vals);
}

perflibs_status_t perflibs_spmat_export_bsr_c(
    perflibs_const_spmat_t A, enum perflibs_dense_layout block_layout,
    perflibs_int_t index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *block_size, perflibs_int_t **row_ptr,
    perflibs_int_t **col_indx, perflibs_singlecomplex_t **vals) {
  return perflibs::sparse::spmat_export_bsr(
      A, block_layout, index_base, m, n, block_size, row_ptr, col_indx,
      reinterpret_cast<std::complex<float> **>(vals));
}

perflibs_status_t perflibs_spmat_export_bsr_z(
    perflibs_const_spmat_t A, enum perflibs_dense_layout block_layout,
    perflibs_int_t index_base, perflibs_int_t *m, perflibs_int_t *n,
    perflibs_int_t *block_size, perflibs_int_t **row_ptr,
    perflibs_int_t **col_indx, perflibs_doublecomplex_t **vals) {
  return perflibs::sparse::spmat_export_bsr(
      A, block_layout, index_base, m, n, block_size, row_ptr, col_indx,
      reinterpret_cast<std::complex<double> **>(vals));
}

perflibs_status_t perflibs_spscale_exec_s(float alpha, perflibs_spmat_t A) {
  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<float> *>(A->impl);
  if (impl_A->no_copy) {
    impl_A->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl_A->error_handle.err_msg = "In-place scaling requires the "
                                   "PERFLIBS_SPARSE_NO_COPY flag to be false.";
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  return perflibs::sparse::scale_matrix(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                        alpha, A);
}

perflibs_status_t perflibs_spscale_exec_d(double alpha, perflibs_spmat_t A) {
  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<double> *>(A->impl);
  if (impl_A->no_copy) {
    impl_A->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl_A->error_handle.err_msg = "In-place scaling requires the "
                                   "PERFLIBS_SPARSE_NO_COPY flag to be false.";
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  return perflibs::sparse::scale_matrix(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                        alpha, A);
}

perflibs_status_t perflibs_spscale_exec_c(perflibs_singlecomplex_t alpha,
                                          perflibs_spmat_t A) {
  auto impl_A =
      reinterpret_cast<perflibs_spmat_impl_t<std::complex<float>> *>(A->impl);
  if (impl_A->no_copy) {
    impl_A->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl_A->error_handle.err_msg = "In-place scaling requires the "
                                   "PERFLIBS_SPARSE_NO_COPY flag to be false.";
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  return perflibs::sparse::scale_matrix(
      PERFLIBS_SPARSE_OPERATION_NOTRANS,
      perflibs::sparse::c_api::to_cpp_scalar(alpha), A);
}

perflibs_status_t perflibs_spscale_exec_z(perflibs_doublecomplex_t alpha,
                                          perflibs_spmat_t A) {
  auto impl_A =
      reinterpret_cast<perflibs_spmat_impl_t<std::complex<double>> *>(A->impl);
  if (impl_A->no_copy) {
    impl_A->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl_A->error_handle.err_msg = "In-place scaling requires the "
                                   "PERFLIBS_SPARSE_NO_COPY flag to be false.";
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  return perflibs::sparse::scale_matrix(
      PERFLIBS_SPARSE_OPERATION_NOTRANS,
      perflibs::sparse::c_api::to_cpp_scalar(alpha), A);
}

perflibs_status_t
perflibs_sptranspose_exec_s(enum perflibs_sparse_hint_value transA,
                            perflibs_spmat_t A) {
  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<float> *>(A->impl);
  if (impl_A->no_copy) {
    impl_A->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl_A->error_handle.err_msg = "In-place scaling requires the "
                                   "PERFLIBS_SPARSE_NO_COPY flag to be false.";
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  return perflibs::sparse::scale_matrix(transA, (float)1.0, A);
}

perflibs_status_t
perflibs_sptranspose_exec_d(enum perflibs_sparse_hint_value transA,
                            perflibs_spmat_t A) {
  auto impl_A = reinterpret_cast<perflibs_spmat_impl_t<double> *>(A->impl);
  if (impl_A->no_copy) {
    impl_A->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl_A->error_handle.err_msg = "In-place scaling requires the "
                                   "PERFLIBS_SPARSE_NO_COPY flag to be false.";
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  return perflibs::sparse::scale_matrix(transA, (double)1.0, A);
}

perflibs_status_t
perflibs_sptranspose_exec_c(enum perflibs_sparse_hint_value transA,
                            perflibs_spmat_t A) {
  auto impl_A =
      reinterpret_cast<perflibs_spmat_impl_t<std::complex<float>> *>(A->impl);
  if (impl_A->no_copy) {
    impl_A->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl_A->error_handle.err_msg = "In-place scaling requires the "
                                   "PERFLIBS_SPARSE_NO_COPY flag to be false.";
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  return perflibs::sparse::scale_matrix(transA, (std::complex<float>)1.0, A);
}

perflibs_status_t
perflibs_sptranspose_exec_z(enum perflibs_sparse_hint_value transA,
                            perflibs_spmat_t A) {
  auto impl_A =
      reinterpret_cast<perflibs_spmat_impl_t<std::complex<double>> *>(A->impl);
  if (impl_A->no_copy) {
    impl_A->error_handle.perflibs_error_type =
        PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
    impl_A->error_handle.err_msg = "In-place scaling requires the "
                                   "PERFLIBS_SPARSE_NO_COPY flag to be false.";
    return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  }
  return perflibs::sparse::scale_matrix(transA, (std::complex<double>)1.0, A);
}

void perflibs_spmat_print_err(perflibs_spmat_t A) {
  perflibs::sparse::spmat_print_err(A);
}
