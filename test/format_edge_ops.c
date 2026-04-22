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

static int test_complex_coo_norm_and_conjtranspose() {
  const perflibs_int_t row_indx[] = {0, 1, 1};
  const perflibs_int_t col_indx[] = {0, 0, 1};
  const perflibs_singlecomplex_t vals[] = {
      TEST_C(1.0f, 1.0f), TEST_C(2.0f, -1.0f), TEST_C(3.0f, 0.0f)};
  const perflibs_singlecomplex_t expected_dense[] = {
      TEST_C(1.0f, -1.0f), TEST_C(2.0f, 1.0f), TEST_C(0.0f, 0.0f),
      TEST_C(0.0f, 0.0f),  TEST_C(0.0f, 0.0f), TEST_C(3.0f, 0.0f),
      TEST_C(0.0f, 0.0f),  TEST_C(0.0f, 0.0f), TEST_C(0.0f, 0.0f),
      TEST_C(0.0f, 0.0f),  TEST_C(0.0f, 0.0f), TEST_C(0.0f, 0.0f),
      TEST_C(0.0f, 0.0f),  TEST_C(0.0f, 0.0f), TEST_C(0.0f, 0.0f),
      TEST_C(0.0f, 0.0f)};
  perflibs_spmat_t coo = NULL;
  perflibs_int_t index_base = -1;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  perflibs_int_t nnz = -1;
  float inf_norm = -1.0f;
  float fro_norm = -1.0f;
  perflibs_singlecomplex_t *dense = NULL;

  CHECK_STATUS(
      perflibs_spmat_create_coo_c(&coo, 4, 4, 3, row_indx, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_query(coo, &index_base, &m, &n, &nnz));
  CHECK_INT_EQ(index_base, 0);
  CHECK_INT_EQ(m, 4);
  CHECK_INT_EQ(n, 4);
  CHECK_INT_EQ(nnz, 3);

  CHECK_STATUS(
      perflibs_spnorm_exec_c(coo, PERFLIBS_SPARSE_NORM_INF, &inf_norm));
  CHECK_STATUS(
      perflibs_spnorm_exec_c(coo, PERFLIBS_SPARSE_NORM_FRB, &fro_norm));
  CHECK_DOUBLE_NEAR(inf_norm, 3.0 + sqrt(5.0), 1e-5);
  CHECK_DOUBLE_NEAR(fro_norm, 4.0, 1e-5);

  CHECK_STATUS(
      perflibs_sptranspose_exec_c(PERFLIBS_SPARSE_OPERATION_CONJTRANS, coo));
  CHECK_STATUS(
      perflibs_spmat_export_dense_c(coo, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
  CHECK_SINGLECOMPLEX_ARRAY(dense, expected_dense, 16, 1e-6f);
  free(dense);

  CHECK_STATUS(perflibs_spmat_destroy(coo));
  return EXIT_SUCCESS;
}

static int test_dense_col_major_conjtranspose() {
  const perflibs_doublecomplex_t input[] = {
      TEST_Z(1.0, 1.0), TEST_Z(2.0, 0.0),  TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0), TEST_Z(3.0, -1.0), TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),  TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),  TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0)};
  const perflibs_doublecomplex_t expected[] = {
      TEST_Z(1.0, -1.0), TEST_Z(2.0, 0.0), TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0),  TEST_Z(3.0, 1.0), TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0),  TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0),  TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0)};
  perflibs_spmat_t dense_mat = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  double fro_norm = -1.0;
  perflibs_doublecomplex_t *dense = NULL;

  CHECK_STATUS(perflibs_spmat_create_dense_z(&dense_mat, PERFLIBS_COL_MAJOR, 4,
                                             4, 4, input, 0));
  CHECK_STATUS(
      perflibs_spnorm_exec_z(dense_mat, PERFLIBS_SPARSE_NORM_FRB, &fro_norm));
  CHECK_DOUBLE_NEAR(fro_norm, sqrt(16.0), 1e-12);

  CHECK_STATUS(perflibs_sptranspose_exec_z(PERFLIBS_SPARSE_OPERATION_CONJTRANS,
                                           dense_mat));
  CHECK_STATUS(perflibs_spmat_export_dense_z(dense_mat, PERFLIBS_ROW_MAJOR, &m,
                                             &n, &dense));
  CHECK_DOUBLECOMPLEX_ARRAY(dense, expected, 16, 1e-12);
  free(dense);

  CHECK_STATUS(perflibs_spmat_destroy(dense_mat));
  return EXIT_SUCCESS;
}

static int test_col_major_bsr_transpose_and_scale() {
  const perflibs_int_t row_ptr[] = {1, 2, 4};
  const perflibs_int_t col_indx[] = {1, 1, 2};
  const double vals[] = {1.0, 0.0, 2.0, 3.0, 4.0, 5.0,
                         0.0, 6.0, 7.0, 0.0, 0.0, 8.0};
  const double expected_dense[] = {1.0, 0.0, 4.0, 5.0, 2.0, 3.0, 0.0, 6.0,
                                   0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 8.0};
  const double expected_scaled[] = {0.5, 0.0, 2.0, 2.5, 1.0, 1.5, 0.0, 3.0,
                                    0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 4.0};
  perflibs_spmat_t bsr = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  perflibs_int_t block_size = -1;
  perflibs_int_t *row_ptr_out = NULL;
  perflibs_int_t *col_indx_out = NULL;
  double *vals_out = NULL;
  double *dense = NULL;

  CHECK_STATUS(perflibs_spmat_create_bsr_d(&bsr, PERFLIBS_COL_MAJOR, 4, 4, 2,
                                           row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_export_bsr_d(bsr, PERFLIBS_COL_MAJOR, 1, &m, &n,
                                           &block_size, &row_ptr_out,
                                           &col_indx_out, &vals_out));
  CHECK_INT_EQ(m, 4);
  CHECK_INT_EQ(n, 4);
  CHECK_INT_EQ(block_size, 2);
  CHECK_INT_ARRAY(row_ptr_out, row_ptr, 3);
  CHECK_INT_ARRAY(col_indx_out, col_indx, 3);
  CHECK_DOUBLE_ARRAY(vals_out, vals, 12, 1e-12);
  free(row_ptr_out);
  free(col_indx_out);
  free(vals_out);

  CHECK_STATUS(
      perflibs_sptranspose_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS, bsr));
  CHECK_STATUS(
      perflibs_spmat_export_dense_d(bsr, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
  CHECK_DOUBLE_ARRAY(dense, expected_dense, 16, 1e-12);
  free(dense);
  dense = NULL;

  CHECK_STATUS(perflibs_spscale_exec_d(0.5, bsr));
  CHECK_STATUS(
      perflibs_spmat_export_dense_d(bsr, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
  CHECK_DOUBLE_ARRAY(dense, expected_scaled, 16, 1e-12);
  free(dense);

  CHECK_STATUS(perflibs_spmat_destroy(bsr));
  return EXIT_SUCCESS;
}

int main() {
  if (test_complex_coo_norm_and_conjtranspose() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_dense_col_major_conjtranspose() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_col_major_bsr_transpose_and_scale() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
