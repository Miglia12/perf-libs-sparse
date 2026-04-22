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

static int test_dense_row_major_real_update_scale_norm_roundtrip() {
  const double initial_row_major[] = {1.0,  2.0, 0.0,  4.0, 5.0, 6.0,
                                      7.0,  0.0, 0.0,  8.0, 9.0, 10.0,
                                      11.0, 0.0, 12.0, 13.0};
  const perflibs_int_t update_rows[] = {0, 1};
  const perflibs_int_t update_cols[] = {2, 3};
  const double update_vals[] = {3.0, -1.0};
  const double expected_after_update_row_major[] = {
      1.0, 2.0, 3.0, 4.0,  5.0,  6.0, 7.0,  -1.0,
      0.0, 8.0, 9.0, 10.0, 11.0, 0.0, 12.0, 13.0};
  const double expected_after_update_col_major[] = {
      1.0, 5.0, 0.0, 11.0, 2.0, 6.0,  8.0,  0.0,
      3.0, 7.0, 9.0, 12.0, 4.0, -1.0, 10.0, 13.0};
  const double expected_after_scale_row_major[] = {
      0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, -0.5,
      0.0, 4.0, 4.5, 5.0, 5.5, 0.0, 6.0, 6.5};
  const double expected_after_transpose_row_major[] = {
      0.5, 2.5, 0.0, 5.5, 1.0, 3.0,  4.0, 0.0,
      1.5, 3.5, 4.5, 6.0, 2.0, -0.5, 5.0, 6.5};
  perflibs_spmat_t dense = NULL;
  perflibs_spmat_t roundtrip = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  double *dense_export = NULL;
  double inf_norm = -1.0;
  double frb_norm = -1.0;
  perflibs_int_t *row_ptr = NULL;
  perflibs_int_t *col_indx = NULL;
  double *vals = NULL;

  CHECK_STATUS(perflibs_spmat_create_dense_d(&dense, PERFLIBS_ROW_MAJOR, 4, 4,
                                             4, initial_row_major, 0));
  CHECK_STATUS(perflibs_spmat_export_dense_d(dense, PERFLIBS_ROW_MAJOR, &m, &n,
                                             &dense_export));
  CHECK_INT_EQ(m, 4);
  CHECK_INT_EQ(n, 4);
  CHECK_DOUBLE_ARRAY(dense_export, initial_row_major, 16, 1e-12);
  free(dense_export);

  CHECK_STATUS(
      perflibs_spmat_update_d(dense, 2, update_rows, update_cols, update_vals));
  CHECK_STATUS(perflibs_spmat_export_dense_d(dense, PERFLIBS_ROW_MAJOR, &m, &n,
                                             &dense_export));
  CHECK_DOUBLE_ARRAY(dense_export, expected_after_update_row_major, 16, 1e-12);
  free(dense_export);

  CHECK_STATUS(perflibs_spmat_export_dense_d(dense, PERFLIBS_COL_MAJOR, &m, &n,
                                             &dense_export));
  CHECK_DOUBLE_ARRAY(dense_export, expected_after_update_col_major, 16, 1e-12);
  free(dense_export);

  CHECK_STATUS(
      perflibs_spnorm_exec_d(dense, PERFLIBS_SPARSE_NORM_INF, &inf_norm));
  CHECK_STATUS(
      perflibs_spnorm_exec_d(dense, PERFLIBS_SPARSE_NORM_FRB, &frb_norm));
  CHECK_DOUBLE_NEAR(inf_norm, 36.0, 1e-12);
  CHECK_DOUBLE_NEAR(frb_norm, sqrt(820.0), 1e-12);

  CHECK_STATUS(perflibs_spscale_exec_d(0.5, dense));
  CHECK_STATUS(perflibs_spmat_export_dense_d(dense, PERFLIBS_ROW_MAJOR, &m, &n,
                                             &dense_export));
  CHECK_DOUBLE_ARRAY(dense_export, expected_after_scale_row_major, 16, 1e-12);
  free(dense_export);

  CHECK_STATUS(
      perflibs_sptranspose_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS, dense));
  CHECK_STATUS(perflibs_spmat_export_dense_d(dense, PERFLIBS_ROW_MAJOR, &m, &n,
                                             &dense_export));
  CHECK_INT_EQ(m, 4);
  CHECK_INT_EQ(n, 4);
  CHECK_DOUBLE_ARRAY(dense_export, expected_after_transpose_row_major, 16,
                     1e-12);
  free(dense_export);

  CHECK_STATUS(perflibs_spmat_export_csr_d(dense, 0, &m, &n, &row_ptr,
                                           &col_indx, &vals));
  CHECK_STATUS(perflibs_spmat_create_csr_d(&roundtrip, m, n, row_ptr, col_indx,
                                           vals, 0));
  free(row_ptr);
  free(col_indx);
  free(vals);

  CHECK_STATUS(perflibs_spmat_export_dense_d(roundtrip, PERFLIBS_ROW_MAJOR, &m,
                                             &n, &dense_export));
  CHECK_DOUBLE_ARRAY(dense_export, expected_after_transpose_row_major, 16,
                     1e-12);
  free(dense_export);

  CHECK_STATUS(perflibs_spmat_destroy(dense));
  CHECK_STATUS(perflibs_spmat_destroy(roundtrip));
  return EXIT_SUCCESS;
}

static int test_dense_col_major_complex_update_scale_norm_roundtrip() {
  const perflibs_doublecomplex_t initial_col_major[] = {
      TEST_Z(1.0, 1.0), TEST_Z(3.0, -1.0), TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),
      TEST_Z(2.0, 0.0), TEST_Z(4.0, 2.0),  TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),  TEST_Z(5.0, 1.0), TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),  TEST_Z(0.0, 0.0), TEST_Z(6.0, -2.0)};
  const perflibs_int_t update_row = 0;
  const perflibs_int_t update_col = 1;
  const perflibs_doublecomplex_t update_val = TEST_Z(-2.0, 1.0);
  const perflibs_doublecomplex_t expected_after_update_row_major[] = {
      TEST_Z(1.0, 1.0), TEST_Z(-2.0, 1.0), TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0), TEST_Z(3.0, -1.0), TEST_Z(4.0, 2.0),
      TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),  TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0), TEST_Z(5.0, 1.0),  TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),  TEST_Z(0.0, 0.0),
      TEST_Z(6.0, -2.0)};
  const perflibs_doublecomplex_t scale_alpha = TEST_Z(0.5, 0.5);
  const perflibs_doublecomplex_t expected_after_scale_row_major[] = {
      TEST_Z(0.0, 1.0), TEST_Z(-1.5, -0.5), TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),
      TEST_Z(2.0, 1.0), TEST_Z(1.0, 3.0),   TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),   TEST_Z(2.0, 3.0), TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),   TEST_Z(0.0, 0.0), TEST_Z(4.0, 2.0)};
  const perflibs_doublecomplex_t expected_after_conjtrans_row_major[] = {
      TEST_Z(0.0, -1.0), TEST_Z(2.0, -1.0), TEST_Z(0.0, -0.0),
      TEST_Z(0.0, -0.0), TEST_Z(-1.5, 0.5), TEST_Z(1.0, -3.0),
      TEST_Z(0.0, -0.0), TEST_Z(0.0, -0.0), TEST_Z(0.0, -0.0),
      TEST_Z(0.0, -0.0), TEST_Z(2.0, -3.0), TEST_Z(0.0, -0.0),
      TEST_Z(0.0, -0.0), TEST_Z(0.0, -0.0), TEST_Z(0.0, -0.0),
      TEST_Z(4.0, -2.0)};
  perflibs_spmat_t dense = NULL;
  perflibs_spmat_t dense_roundtrip = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  perflibs_doublecomplex_t *dense_export = NULL;
  perflibs_doublecomplex_t *dense_col_major = NULL;
  double inf_norm = -1.0;
  double frb_norm = -1.0;

  CHECK_STATUS(perflibs_spmat_create_dense_z(&dense, PERFLIBS_COL_MAJOR, 4, 4,
                                             4, initial_col_major, 0));
  CHECK_STATUS(
      perflibs_spmat_update_z(dense, 1, &update_row, &update_col, &update_val));
  CHECK_STATUS(perflibs_spmat_export_dense_z(dense, PERFLIBS_ROW_MAJOR, &m, &n,
                                             &dense_export));
  CHECK_DOUBLECOMPLEX_ARRAY(dense_export, expected_after_update_row_major, 16,
                            1e-12);
  free(dense_export);

  CHECK_STATUS(
      perflibs_spnorm_exec_z(dense, PERFLIBS_SPARSE_NORM_INF, &inf_norm));
  CHECK_STATUS(
      perflibs_spnorm_exec_z(dense, PERFLIBS_SPARSE_NORM_FRB, &frb_norm));
  CHECK_DOUBLE_NEAR(inf_norm, sqrt(10.0) + sqrt(20.0), 1e-12);
  CHECK_DOUBLE_NEAR(frb_norm, sqrt(103.0), 1e-12);

  CHECK_STATUS(perflibs_spscale_exec_z(scale_alpha, dense));
  CHECK_STATUS(perflibs_spmat_export_dense_z(dense, PERFLIBS_ROW_MAJOR, &m, &n,
                                             &dense_export));
  CHECK_DOUBLECOMPLEX_ARRAY(dense_export, expected_after_scale_row_major, 16,
                            1e-12);
  free(dense_export);

  CHECK_STATUS(
      perflibs_sptranspose_exec_z(PERFLIBS_SPARSE_OPERATION_CONJTRANS, dense));
  CHECK_STATUS(perflibs_spmat_export_dense_z(dense, PERFLIBS_ROW_MAJOR, &m, &n,
                                             &dense_export));
  CHECK_DOUBLECOMPLEX_ARRAY(dense_export, expected_after_conjtrans_row_major,
                            16, 1e-12);
  free(dense_export);

  CHECK_STATUS(perflibs_spmat_export_dense_z(dense, PERFLIBS_COL_MAJOR, &m, &n,
                                             &dense_col_major));
  CHECK_STATUS(perflibs_spmat_create_dense_z(
      &dense_roundtrip, PERFLIBS_COL_MAJOR, m, n, m, dense_col_major, 0));
  free(dense_col_major);

  CHECK_STATUS(perflibs_spmat_export_dense_z(
      dense_roundtrip, PERFLIBS_ROW_MAJOR, &m, &n, &dense_export));
  CHECK_DOUBLECOMPLEX_ARRAY(dense_export, expected_after_conjtrans_row_major,
                            16, 1e-12);
  free(dense_export);

  CHECK_STATUS(perflibs_spmat_destroy(dense));
  CHECK_STATUS(perflibs_spmat_destroy(dense_roundtrip));
  return EXIT_SUCCESS;
}

int main() {
  if (test_dense_row_major_real_update_scale_norm_roundtrip() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_dense_col_major_complex_update_scale_norm_roundtrip() !=
      EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
