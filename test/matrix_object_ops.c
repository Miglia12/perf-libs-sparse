/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <math.h>
#include <stdlib.h>

static int test_null_and_identity() {
  perflibs_spmat_t null_mat = perflibs_spmat_create_null(4, 4);
  perflibs_spmat_t identity = perflibs_spmat_create_identity(4);
  perflibs_int_t index_base = -1, m = -1, n = -1, nnz = -1;
  double *dense = NULL;
  perflibs_int_t *row_ptr = NULL;
  perflibs_int_t *col_indx = NULL;
  double *vals = NULL;
  double x[4] = {2.0, -1.0, 0.5, 3.0};
  double y[4] = {0.0, 0.0, 0.0, 0.0};
  const double zero_dense[16] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const perflibs_int_t identity_row_ptr[5] = {1, 2, 3, 4, 5};
  const perflibs_int_t identity_col_indx[4] = {1, 2, 3, 4};
  const double identity_vals[4] = {1.0, 1.0, 1.0, 1.0};

  CHECK_TRUE(null_mat != NULL, "null matrix creation returned NULL");
  CHECK_TRUE(identity != NULL, "identity matrix creation returned NULL");

  CHECK_STATUS(perflibs_spmat_query(null_mat, &index_base, &m, &n, &nnz));
  CHECK_INT_EQ(index_base, 0);
  CHECK_INT_EQ(m, 4);
  CHECK_INT_EQ(n, 4);
  CHECK_INT_EQ(nnz, 0);

  CHECK_STATUS(perflibs_spmat_export_dense_d(null_mat, PERFLIBS_ROW_MAJOR, &m,
                                             &n, &dense));
  CHECK_INT_EQ(m, 4);
  CHECK_INT_EQ(n, 4);
  CHECK_DOUBLE_ARRAY(dense, zero_dense, 16, 1e-12);
  free(dense);
  dense = NULL;

  CHECK_STATUS(perflibs_spmat_query(identity, &index_base, &m, &n, &nnz));
  CHECK_INT_EQ(index_base, 0);
  CHECK_INT_EQ(m, 4);
  CHECK_INT_EQ(n, 4);
  CHECK_INT_EQ(nnz, 4);

  CHECK_STATUS(perflibs_spmat_export_csr_d(identity, 1, &m, &n, &row_ptr,
                                           &col_indx, &vals));
  CHECK_INT_ARRAY(row_ptr, identity_row_ptr, 5);
  CHECK_INT_ARRAY(col_indx, identity_col_indx, 4);
  CHECK_DOUBLE_ARRAY(vals, identity_vals, 4, 1e-12);
  free(row_ptr);
  free(col_indx);
  free(vals);
  row_ptr = NULL;
  col_indx = NULL;
  vals = NULL;

  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0,
                                    identity, x, 0.0, y));
  CHECK_DOUBLE_ARRAY(y, x, 4, 1e-12);

  CHECK_STATUS(perflibs_spmat_destroy(null_mat));
  CHECK_STATUS(perflibs_spmat_destroy(identity));
  return EXIT_SUCCESS;
}

static int test_create_export_update_and_transform() {
  perflibs_spmat_t csr = NULL;
  perflibs_spmat_t csc = NULL;
  perflibs_spmat_t coo = NULL;
  perflibs_spmat_t bsr = NULL;
  perflibs_spmat_t nocopy_dense = NULL;
  perflibs_int_t m = -1, n = -1, nnz = -1, index_base = -1;
  perflibs_int_t *row_ptr = NULL;
  perflibs_int_t *col_indx = NULL;
  perflibs_int_t *row_indx = NULL;
  perflibs_int_t *col_ptr = NULL;
  double *vals = NULL;
  double *dense = NULL;
  perflibs_int_t block_size = -1;
  double inf_norm = 0.0, fro_norm = 0.0;
  const perflibs_int_t csr_row_ptr[5] = {0, 2, 3, 5, 6};
  const perflibs_int_t csr_col_indx[6] = {0, 2, 1, 0, 2, 3};
  const double csr_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  const perflibs_int_t csc_row_indx[6] = {0, 2, 1, 0, 2, 3};
  const perflibs_int_t csc_col_ptr[5] = {0, 2, 3, 5, 6};
  const perflibs_int_t coo_row_indx[6] = {0, 0, 1, 2, 2, 3};
  const perflibs_int_t coo_col_indx[6] = {0, 2, 1, 0, 2, 3};
  const double csc_vals[6] = {1.0, 4.0, 3.0, 2.0, 5.0, 6.0};
  const double expected_dense_row[16] = {1.0, 0.0, 2.0, 0.0, 0.0, 3.0,
                                         0.0, 0.0, 4.0, 0.0, 5.0, 0.0,
                                         0.0, 0.0, 0.0, 6.0};
  const double expected_dense_col[16] = {1.0, 0.0, 4.0, 0.0, 0.0, 3.0,
                                         0.0, 0.0, 2.0, 0.0, 5.0, 0.0,
                                         0.0, 0.0, 0.0, 6.0};
  const perflibs_int_t expected_csr_row_ptr_base1[5] = {1, 3, 4, 6, 7};
  const perflibs_int_t expected_csr_col_indx_base1[6] = {1, 3, 2, 1, 3, 4};
  const double expected_updated_dense[16] = {1.0, 0.0, 7.0, 0.0, 0.0, 3.0,
                                             0.0, 0.0, 6.0, 0.0, 5.0, 0.0,
                                             0.0, 0.0, 0.0, 6.0};
  const double expected_scaled_transposed[16] = {0.5, 0.0, 3.0, 0.0, 0.0, 1.5,
                                                 0.0, 0.0, 3.5, 0.0, 2.5, 0.0,
                                                 0.0, 0.0, 0.0, 3.0};
  const perflibs_int_t update_rows[2] = {0, 2};
  const perflibs_int_t update_cols[2] = {2, 0};
  const double update_vals[2] = {7.0, 6.0};
  const perflibs_int_t bsr_row_ptr[3] = {0, 1, 2};
  const perflibs_int_t bsr_col_indx[2] = {0, 1};
  const double bsr_vals[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const perflibs_int_t expected_bsr_row_ptr_base1[3] = {1, 2, 3};
  const perflibs_int_t expected_bsr_col_indx_base1[2] = {1, 2};
  const double expected_bsr_dense[16] = {1.0, 2.0, 0.0, 0.0, 3.0, 4.0,
                                         0.0, 0.0, 0.0, 0.0, 5.0, 6.0,
                                         0.0, 0.0, 7.0, 8.0};
  double nocopy_vals[16] = {1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};

  CHECK_STATUS(perflibs_spmat_create_csr_d(&csr, 4, 4, csr_row_ptr,
                                           csr_col_indx, csr_vals, 0));
  CHECK_STATUS(perflibs_spmat_query(csr, &index_base, &m, &n, &nnz));
  CHECK_INT_EQ(index_base, 0);
  CHECK_INT_EQ(m, 4);
  CHECK_INT_EQ(n, 4);
  CHECK_INT_EQ(nnz, 6);

  CHECK_STATUS(perflibs_spmat_hint(csr, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_UNSTRUCTURED));
  CHECK_STATUS(perflibs_spmat_hint(csr, PERFLIBS_SPARSE_HINT_MEMORY,
                                   PERFLIBS_SPARSE_MEMORY_ALLOCS));
  CHECK_STATUS(perflibs_spmat_hint(csr, PERFLIBS_SPARSE_HINT_SPMM_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spmat_hint(csr, PERFLIBS_SPARSE_HINT_SPMM_STRATEGY,
                                   PERFLIBS_SPARSE_SPMM_STRAT_OPT_NO_STRUCT));

  CHECK_STATUS(
      perflibs_spmat_export_dense_d(csr, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
  CHECK_DOUBLE_ARRAY(dense, expected_dense_row, 16, 1e-12);
  free(dense);
  dense = NULL;

  CHECK_STATUS(
      perflibs_spmat_export_dense_d(csr, PERFLIBS_COL_MAJOR, &m, &n, &dense));
  CHECK_DOUBLE_ARRAY(dense, expected_dense_col, 16, 1e-12);
  free(dense);
  dense = NULL;

  CHECK_STATUS(
      perflibs_spmat_export_csr_d(csr, 1, &m, &n, &row_ptr, &col_indx, &vals));
  CHECK_INT_ARRAY(row_ptr, expected_csr_row_ptr_base1, 5);
  CHECK_INT_ARRAY(col_indx, expected_csr_col_indx_base1, 6);
  CHECK_DOUBLE_ARRAY(vals, csr_vals, 6, 1e-12);
  free(row_ptr);
  free(col_indx);
  free(vals);
  row_ptr = NULL;
  col_indx = NULL;
  vals = NULL;

  CHECK_STATUS(
      perflibs_spmat_export_csc_d(csr, 0, &m, &n, &row_indx, &col_ptr, &vals));
  CHECK_INT_ARRAY(row_indx, csc_row_indx, 6);
  CHECK_INT_ARRAY(col_ptr, csc_col_ptr, 5);
  CHECK_DOUBLE_ARRAY(vals, csc_vals, 6, 1e-12);
  free(row_indx);
  free(col_ptr);
  free(vals);
  row_indx = NULL;
  col_ptr = NULL;
  vals = NULL;

  CHECK_STATUS(perflibs_spmat_export_coo_d(csr, &m, &n, &nnz, &row_indx,
                                           &col_indx, &vals));
  CHECK_INT_ARRAY(row_indx, coo_row_indx, 6);
  CHECK_INT_ARRAY(col_indx, coo_col_indx, 6);
  CHECK_DOUBLE_ARRAY(vals, csr_vals, 6, 1e-12);
  free(row_indx);
  free(col_indx);
  free(vals);
  row_indx = NULL;
  col_indx = NULL;
  vals = NULL;

  CHECK_STATUS(
      perflibs_spmat_update_d(csr, 2, update_rows, update_cols, update_vals));
  CHECK_STATUS(
      perflibs_spmat_export_dense_d(csr, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
  CHECK_DOUBLE_ARRAY(dense, expected_updated_dense, 16, 1e-12);
  free(dense);
  dense = NULL;

  CHECK_STATUS(perflibs_spscale_exec_d(0.5, csr));
  CHECK_STATUS(
      perflibs_spnorm_exec_d(csr, PERFLIBS_SPARSE_NORM_INF, &inf_norm));
  CHECK_STATUS(
      perflibs_spnorm_exec_d(csr, PERFLIBS_SPARSE_NORM_FRB, &fro_norm));
  CHECK_DOUBLE_NEAR(inf_norm, 5.5, 1e-12);
  CHECK_DOUBLE_NEAR(fro_norm, sqrt(39.0), 1e-12);

  CHECK_STATUS(
      perflibs_sptranspose_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS, csr));
  CHECK_STATUS(
      perflibs_spmat_export_dense_d(csr, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
  CHECK_DOUBLE_ARRAY(dense, expected_scaled_transposed, 16, 1e-12);
  free(dense);
  dense = NULL;

  CHECK_STATUS(perflibs_spmat_create_csc_d(&csc, 4, 4, csc_row_indx,
                                           csc_col_ptr, csc_vals, 0));
  CHECK_STATUS(
      perflibs_spmat_export_csr_d(csc, 0, &m, &n, &row_ptr, &col_indx, &vals));
  CHECK_INT_ARRAY(row_ptr, csr_row_ptr, 5);
  CHECK_INT_ARRAY(col_indx, csr_col_indx, 6);
  CHECK_DOUBLE_ARRAY(vals, csr_vals, 6, 1e-12);
  free(row_ptr);
  free(col_indx);
  free(vals);
  row_ptr = NULL;
  col_indx = NULL;
  vals = NULL;

  CHECK_STATUS(perflibs_spmat_create_coo_d(&coo, 4, 4, 6, coo_row_indx,
                                           coo_col_indx, csr_vals, 0));
  CHECK_STATUS(
      perflibs_spmat_export_csc_d(coo, 0, &m, &n, &row_indx, &col_ptr, &vals));
  CHECK_INT_ARRAY(row_indx, csc_row_indx, 6);
  CHECK_INT_ARRAY(col_ptr, csc_col_ptr, 5);
  CHECK_DOUBLE_ARRAY(vals, csc_vals, 6, 1e-12);
  free(row_indx);
  free(col_ptr);
  free(vals);
  row_indx = NULL;
  col_ptr = NULL;
  vals = NULL;

  CHECK_STATUS(perflibs_spmat_create_bsr_d(&bsr, PERFLIBS_ROW_MAJOR, 4, 4, 2,
                                           bsr_row_ptr, bsr_col_indx, bsr_vals,
                                           0));
  CHECK_STATUS(perflibs_spmat_export_bsr_d(bsr, PERFLIBS_ROW_MAJOR, 1, &m, &n,
                                           &block_size, &row_ptr, &col_indx,
                                           &vals));
  CHECK_INT_EQ(block_size, 2);
  CHECK_INT_ARRAY(row_ptr, expected_bsr_row_ptr_base1, 3);
  CHECK_INT_ARRAY(col_indx, expected_bsr_col_indx_base1, 2);
  CHECK_DOUBLE_ARRAY(vals, bsr_vals, 8, 1e-12);
  free(row_ptr);
  free(col_indx);
  free(vals);
  row_ptr = NULL;
  col_indx = NULL;
  vals = NULL;

  CHECK_STATUS(
      perflibs_spmat_export_dense_d(bsr, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
  CHECK_DOUBLE_ARRAY(dense, expected_bsr_dense, 16, 1e-12);
  free(dense);
  dense = NULL;

  CHECK_STATUS(perflibs_spmat_create_dense_d(&nocopy_dense, PERFLIBS_ROW_MAJOR,
                                             4, 4, 4, nocopy_vals,
                                             PERFLIBS_SPARSE_CREATE_NOCOPY));
  CHECK_STATUS_EQ(perflibs_spscale_exec_d(2.0, nocopy_dense),
                  PERFLIBS_STATUS_INPUT_PARAMETER_ERROR);
  CHECK_STATUS_EQ(perflibs_sptranspose_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS,
                                              nocopy_dense),
                  PERFLIBS_STATUS_INPUT_PARAMETER_ERROR);
  perflibs_spmat_print_err(nocopy_dense);

  CHECK_STATUS(perflibs_spmat_destroy(csr));
  CHECK_STATUS(perflibs_spmat_destroy(csc));
  CHECK_STATUS(perflibs_spmat_destroy(coo));
  CHECK_STATUS(perflibs_spmat_destroy(bsr));
  CHECK_STATUS(perflibs_spmat_destroy(nocopy_dense));
  return EXIT_SUCCESS;
}

int main() {
  if (test_null_and_identity() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_create_export_update_and_transform() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
