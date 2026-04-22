/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <stdlib.h>

static int test_csc_spmv_real_and_updates() {
  const perflibs_int_t row_indx[] = {0, 1, 1, 2, 0, 2};
  const perflibs_int_t col_ptr[] = {0, 2, 4, 6, 6};
  const double vals[] = {2.0, 1.0, 3.0, 4.0, 1.0, 5.0};
  const double x[] = {1.0, 2.0, 3.0, 0.0};
  double y[4] = {0.0, 0.0, 0.0, 0.0};
  double dense_expected[16] = {2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0,
                               0.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  perflibs_spmat_t csc = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  double *dense = NULL;

  CHECK_STATUS(
      perflibs_spmat_create_csc_d(&csc, 4, 4, row_indx, col_ptr, vals, 0));
  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0, csc,
                                    x, 0.0, y));
  CHECK_DOUBLE_NEAR(y[0], 5.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[1], 7.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[2], 23.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[3], 0.0, 1e-12);

  y[0] = y[1] = y[2] = y[3] = 0.0;
  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS, 1.0, csc,
                                    x, 0.0, y));
  CHECK_DOUBLE_NEAR(y[0], 4.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[1], 18.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[2], 16.0, 1e-12);
  CHECK_DOUBLE_NEAR(y[3], 0.0, 1e-12);

  const perflibs_int_t upd_row = 0;
  const perflibs_int_t upd_col = 2;
  const double upd_val = 2.0;
  CHECK_STATUS(perflibs_spmat_update_d(csc, 1, &upd_row, &upd_col, &upd_val));
  CHECK_STATUS(perflibs_spscale_exec_d(0.5, csc));
  dense_expected[2] = 1.0;
  dense_expected[0] = 1.0;
  dense_expected[4] = 0.5;
  dense_expected[5] = 1.5;
  dense_expected[9] = 2.0;
  dense_expected[10] = 2.5;
  CHECK_STATUS(
      perflibs_spmat_export_dense_d(csc, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
  CHECK_DOUBLE_ARRAY(dense, dense_expected, 16, 1e-12);
  free(dense);

  CHECK_STATUS(perflibs_spmat_destroy(csc));
  return EXIT_SUCCESS;
}

static int test_csc_spsv_real_notrans_and_trans() {
  const perflibs_int_t row_indx[] = {0, 1, 1, 2, 2, 3};
  const perflibs_int_t col_ptr[] = {0, 2, 4, 5, 6};
  const double vals[] = {2.0, 1.0, 3.0, 4.0, 5.0, 1.0};
  const double rhs_notrans[] = {2.0, 7.0, 23.0, 4.0};
  const double rhs_trans[] = {4.0, 18.0, 15.0, 4.0};
  perflibs_spmat_t csc = NULL;
  double x[4] = {0.0, 0.0, 0.0, 0.0};

  CHECK_STATUS(
      perflibs_spmat_create_csc_d(&csc, 4, 4, row_indx, col_ptr, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(csc, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_hint(csc, PERFLIBS_SPARSE_HINT_SPSV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spsv_optimize(csc));

  CHECK_STATUS(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, csc, x,
                                    1.0, rhs_notrans));
  CHECK_DOUBLE_NEAR(x[0], 1.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[1], 2.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[2], 3.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[3], 4.0, 1e-12);

  x[0] = x[1] = x[2] = x[3] = 0.0;
  CHECK_STATUS(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS, csc, x,
                                    1.0, rhs_trans));
  CHECK_DOUBLE_NEAR(x[0], 1.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[1], 2.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[2], 3.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[3], 4.0, 1e-12);

  CHECK_STATUS(perflibs_spmat_destroy(csc));
  return EXIT_SUCCESS;
}

static int test_csc_spmv_complex_conj_paths() {
  const perflibs_int_t row_indx[] = {0, 0, 1};
  const perflibs_int_t col_ptr[] = {0, 1, 3, 3, 3};
  const perflibs_doublecomplex_t vals[] = {TEST_Z(1.0, 1.0), TEST_Z(2.0, -1.0),
                                           TEST_Z(3.0, 2.0)};
  const perflibs_doublecomplex_t x[] = {TEST_Z(1.0, -1.0), TEST_Z(2.0, 1.0),
                                        TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0)};
  perflibs_doublecomplex_t y[4] = {TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),
                                   TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0)};
  perflibs_spmat_t csc = NULL;

  CHECK_STATUS(
      perflibs_spmat_create_csc_z(&csc, 4, 4, row_indx, col_ptr, vals, 0));
  CHECK_STATUS(perflibs_spmv_exec_z(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    TEST_Z(1.0, 0.0), csc, x, TEST_Z(0.0, 0.0),
                                    y));
  CHECK_DOUBLECOMPLEX_NEAR(y[0], TEST_Z(7.0, 0.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(y[1], TEST_Z(4.0, 7.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(y[2], TEST_Z(0.0, 0.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(y[3], TEST_Z(0.0, 0.0), 1e-12);

  y[0] = y[1] = y[2] = y[3] = TEST_Z(0.0, 0.0);
  CHECK_STATUS(perflibs_spmv_exec_z(PERFLIBS_SPARSE_OPERATION_CONJTRANS,
                                    TEST_Z(1.0, 0.0), csc, x, TEST_Z(0.0, 0.0),
                                    y));
  CHECK_DOUBLECOMPLEX_NEAR(y[0], TEST_Z(0.0, -2.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(y[1], TEST_Z(11.0, -2.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(y[2], TEST_Z(0.0, 0.0), 1e-12);
  CHECK_DOUBLECOMPLEX_NEAR(y[3], TEST_Z(0.0, 0.0), 1e-12);

  CHECK_STATUS(perflibs_spmat_destroy(csc));
  return EXIT_SUCCESS;
}

int main() {
  if (test_csc_spmv_real_and_updates() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_csc_spsv_real_notrans_and_trans() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_csc_spmv_complex_conj_paths() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
