/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <stdlib.h>

static int test_bsr_spmv_row_major() {
  const perflibs_int_t row_ptr[] = {0, 2, 4};
  const perflibs_int_t col_indx[] = {0, 1, 0, 1};
  const double vals[] = {/* row block 0, col block 0 */ 1.0,
                         2.0,
                         3.0,
                         4.0,
                         /* row block 0, col block 1 */ 5.0,
                         6.0,
                         7.0,
                         8.0,
                         /* row block 1, col block 0 */ 9.0,
                         10.0,
                         11.0,
                         12.0,
                         /* row block 1, col block 1 */ 13.0,
                         14.0,
                         15.0,
                         16.0};
  const double x[] = {1.0, 2.0, 3.0, 4.0};
  const double expected_notrans[] = {44.0, 64.0, 124.0, 144.0};
  const double expected_trans[] = {78.0, 88.0, 118.0, 128.0};
  double y[4] = {0.0, 0.0, 0.0, 0.0};
  perflibs_spmat_t bsr = NULL;

  CHECK_STATUS(perflibs_spmat_create_bsr_d(&bsr, PERFLIBS_ROW_MAJOR, 4, 4, 2,
                                           row_ptr, col_indx, vals, 0));

  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0, bsr,
                                    x, 0.0, y));
  CHECK_DOUBLE_ARRAY(y, expected_notrans, 4, 1e-12);

  y[0] = y[1] = y[2] = y[3] = 0.0;
  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS, 1.0, bsr,
                                    x, 0.0, y));
  CHECK_DOUBLE_ARRAY(y, expected_trans, 4, 1e-12);

  CHECK_STATUS(perflibs_spmat_destroy(bsr));
  return EXIT_SUCCESS;
}

static int test_bsr_spsv_triangular() {
  const perflibs_int_t row_ptr[] = {0, 1, 2};
  const perflibs_int_t col_indx[] = {0, 1};
  const double vals[] = {
      /* block (0,0) */ 2.0, 0.0, 1.0, 3.0,
      /* block (1,1) */ 4.0, 0.0, 2.0, 5.0};
  const double expected_x[] = {1.0, 2.0, 3.0, 4.0};
  const double y_notrans[] = {2.0, 7.0, 12.0, 26.0};
  const double y_trans[] = {4.0, 6.0, 20.0, 20.0};
  double x[4] = {0.0, 0.0, 0.0, 0.0};
  perflibs_spmat_t bsr = NULL;

  CHECK_STATUS(perflibs_spmat_create_bsr_d(&bsr, PERFLIBS_ROW_MAJOR, 4, 4, 2,
                                           row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(bsr, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_hint(bsr, PERFLIBS_SPARSE_HINT_SPSV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spsv_optimize(bsr));

  CHECK_STATUS(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, bsr, x,
                                    1.0, y_notrans));
  CHECK_DOUBLE_ARRAY(x, expected_x, 4, 1e-12);

  x[0] = x[1] = x[2] = x[3] = 0.0;
  CHECK_STATUS(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS, bsr, x,
                                    1.0, y_trans));
  CHECK_DOUBLE_ARRAY(x, expected_x, 4, 1e-12);

  CHECK_STATUS(perflibs_spmat_destroy(bsr));
  return EXIT_SUCCESS;
}

static int test_bsr_col_major_transpose_scale_norm_and_spmv() {
  const perflibs_int_t row_ptr[] = {0, 1, 3};
  const perflibs_int_t col_indx[] = {0, 0, 1};
  const double vals[] = {/* block (0,0), col-major */ 1.0,
                         2.0,
                         3.0,
                         4.0,
                         /* block (1,0), col-major */ 5.0,
                         6.0,
                         7.0,
                         8.0,
                         /* block (1,1), col-major */ 9.0,
                         10.0,
                         11.0,
                         12.0};
  const double x[] = {1.0, 1.0, 1.0, 1.0};
  const double expected_y[] = {7.0, 11.0, 9.5, 11.5};
  double y[4] = {0.0, 0.0, 0.0, 0.0};
  double inf_norm = -1.0;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  double *dense = NULL;
  perflibs_spmat_t bsr = NULL;

  CHECK_STATUS(perflibs_spmat_create_bsr_d(&bsr, PERFLIBS_COL_MAJOR, 4, 4, 2,
                                           row_ptr, col_indx, vals, 0));
  CHECK_STATUS(
      perflibs_spnorm_exec_d(bsr, PERFLIBS_SPARSE_NORM_INF, &inf_norm));
  CHECK_DOUBLE_NEAR(inf_norm, 36.0, 1e-12);

  CHECK_STATUS(
      perflibs_sptranspose_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS, bsr));
  CHECK_STATUS(perflibs_spscale_exec_d(0.5, bsr));
  CHECK_STATUS(
      perflibs_spmat_export_dense_d(bsr, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
  CHECK_INT_EQ(m, 4);
  CHECK_INT_EQ(n, 4);
  free(dense);

  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0, bsr,
                                    x, 0.0, y));
  CHECK_DOUBLE_ARRAY(y, expected_y, 4, 1e-12);

  CHECK_STATUS(perflibs_spmat_destroy(bsr));
  return EXIT_SUCCESS;
}

int main() {
  if (test_bsr_spmv_row_major() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_bsr_spsv_triangular() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_bsr_col_major_transpose_scale_norm_and_spmv() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
