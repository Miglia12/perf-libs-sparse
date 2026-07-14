/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <stdlib.h>

static int test_identity_and_null_paths() {
  perflibs_spmat_t identity = perflibs_spmat_create_identity(4);
  perflibs_spmat_t null_mat = perflibs_spmat_create_null(4, 4);
  double x[4] = {-3.0, 9.0, 1.0, -2.0};
  const double y[4] = {2.0, -1.0, 3.0, 4.0};

  CHECK_TRUE(identity != NULL, "identity creation failed");
  CHECK_TRUE(null_mat != NULL, "null creation failed");

  CHECK_STATUS(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, identity,
                                    x, 1.5, y));
  CHECK_DOUBLE_NEAR(x[0], 3.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[1], -1.5, 1e-12);
  CHECK_DOUBLE_NEAR(x[2], 4.5, 1e-12);
  CHECK_DOUBLE_NEAR(x[3], 6.0, 1e-12);

  x[0] = 7.0;
  x[1] = -2.0;
  CHECK_STATUS(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, identity,
                                    x, 0.0, y));
  CHECK_DOUBLE_NEAR(x[0], 0.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[1], 0.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[2], 0.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[3], 0.0, 1e-12);

  CHECK_STATUS_EQ(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                       null_mat, x, 1.0, y),
                  PERFLIBS_STATUS_INPUT_PARAMETER_ERROR);

  CHECK_STATUS(perflibs_spmat_destroy(identity));
  CHECK_STATUS(perflibs_spmat_destroy(null_mat));
  return EXIT_SUCCESS;
}

static int test_dense_and_csc_triangular_paths() {
  const double dense_vals[] = {2.0, 1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  const perflibs_int_t row_indx[] = {0, 0, 1, 2, 3};
  const perflibs_int_t col_ptr[] = {0, 1, 3, 4, 5};
  const double csc_vals[] = {2.0, 1.0, 3.0, 1.0, 1.0};
  const double rhs_notrans[] = {2.0, 7.0, 3.0, 4.0};
  const double rhs_trans[] = {4.0, 6.0, 3.0, 4.0};
  perflibs_spmat_t dense = NULL;
  perflibs_spmat_t csc = NULL;
  double x[4];

  CHECK_STATUS(perflibs_spmat_create_dense_d(&dense, PERFLIBS_COL_MAJOR, 4, 4,
                                             4, dense_vals, 0));
  CHECK_STATUS(
      perflibs_spmat_create_csc_d(&csc, 4, 4, row_indx, col_ptr, csc_vals, 0));

  CHECK_STATUS(perflibs_spmat_hint(dense, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_hint(csc, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_hint(dense, PERFLIBS_SPARSE_HINT_SPSV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spmat_hint(csc, PERFLIBS_SPARSE_HINT_SPSV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spsv_optimize(dense));
  CHECK_STATUS(perflibs_spsv_optimize(csc));

  x[0] = x[1] = x[2] = x[3] = 0.0;
  CHECK_STATUS(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, dense, x,
                                    1.0, rhs_notrans));
  CHECK_DOUBLE_NEAR(x[0], 1.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[1], 2.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[2], 3.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[3], 4.0, 1e-12);

  x[0] = x[1] = x[2] = x[3] = 0.0;
  CHECK_STATUS(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS, dense, x,
                                    1.0, rhs_trans));
  CHECK_DOUBLE_NEAR(x[0], 1.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[1], 2.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[2], 3.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[3], 4.0, 1e-12);

  x[0] = x[1] = x[2] = x[3] = 0.0;
  CHECK_STATUS(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, csc, x,
                                    1.0, rhs_trans));
  CHECK_DOUBLE_NEAR(x[0], 1.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[1], 2.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[2], 3.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[3], 4.0, 1e-12);

  x[0] = x[1] = x[2] = x[3] = 0.0;
  CHECK_STATUS(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS, csc, x,
                                    1.0, rhs_notrans));
  CHECK_DOUBLE_NEAR(x[0], 1.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[1], 2.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[2], 3.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[3], 4.0, 1e-12);

  CHECK_STATUS(perflibs_spmat_destroy(dense));
  CHECK_STATUS(perflibs_spmat_destroy(csc));
  return EXIT_SUCCESS;
}

static int test_hint_switch_and_rectangular_error() {
  const perflibs_int_t tri_row_ptr[] = {0, 1, 3, 4, 5};
  const perflibs_int_t tri_col_indx[] = {0, 0, 1, 2, 3};
  const double tri_vals[] = {2.0, 1.0, 3.0, 1.0, 1.0};
  const double rhs_notrans[] = {2.0, 7.0, 3.0, 4.0};
  const perflibs_int_t full_row_ptr[] = {0, 2, 4, 6, 8};
  const perflibs_int_t full_col_indx[] = {0, 1, 0, 1, 2, 3, 2, 3};
  const double full_vals[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  perflibs_spmat_t triangular = NULL;
  perflibs_spmat_t full = NULL;
  double x[4] = {0.0, 0.0, 0.0, 0.0};

  CHECK_STATUS(perflibs_spmat_create_csr_d(&triangular, 4, 4, tri_row_ptr,
                                           tri_col_indx, tri_vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(triangular, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_hint(triangular,
                                   PERFLIBS_SPARSE_HINT_SPSV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_TRANS));
  CHECK_STATUS(perflibs_spsv_optimize(triangular));
  CHECK_STATUS(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    triangular, x, 1.0, rhs_notrans));
  CHECK_DOUBLE_NEAR(x[0], 1.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[1], 2.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[2], 3.0, 1e-12);
  CHECK_DOUBLE_NEAR(x[3], 4.0, 1e-12);

  CHECK_STATUS(perflibs_spmat_create_csr_d(&full, 4, 4, full_row_ptr,
                                           full_col_indx, full_vals, 0));
  CHECK_STATUS_EQ(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, full,
                                       x, 1.0, rhs_notrans),
                  PERFLIBS_STATUS_INPUT_PARAMETER_ERROR);

  CHECK_STATUS(perflibs_spmat_destroy(triangular));
  CHECK_STATUS(perflibs_spmat_destroy(full));
  return EXIT_SUCCESS;
}

static int test_spsm_hint_switch() {
  const perflibs_int_t tri_row_ptr[] = {0, 1, 3, 4, 5};
  const perflibs_int_t tri_col_indx[] = {0, 0, 1, 2, 3};
  const double tri_vals[] = {2.0, 1.0, 3.0, 1.0, 1.0};
  const double rhs_notrans[] = {2.0, 7.0, 3.0, 4.0, 4.0, 11.0, 5.0, 6.0};
  const double expected_x[] = {1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 5.0, 6.0};
  double x_init[8] = {0.0};
  perflibs_spmat_t triangular = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  double *x_out = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;

  CHECK_STATUS(perflibs_spmat_create_csr_d(&triangular, 4, 4, tri_row_ptr,
                                           tri_col_indx, tri_vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(triangular, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_create_dense_d(&X, PERFLIBS_COL_MAJOR, 4, 2, 4,
                                             x_init, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_d(&Y, PERFLIBS_COL_MAJOR, 4, 2, 4,
                                             rhs_notrans, 0));
  CHECK_STATUS(perflibs_spsm_optimize(PERFLIBS_SPARSE_OPERATION_TRANS,
                                      triangular, X, PERFLIBS_SPARSE_SCALAR_ANY,
                                      Y));
  CHECK_STATUS(perflibs_spsm_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    triangular, X, 1.0, Y));

  CHECK_STATUS(
      perflibs_spmat_export_dense_d(X, PERFLIBS_COL_MAJOR, &m, &n, &x_out));
  CHECK_TRUE(m == 4 && n == 2,
             "unexpected hint-switch SpSM solution shape %lld x %lld",
             test_i64(m), test_i64(n));
  CHECK_DOUBLE_ARRAY(x_out, expected_x, 8, 1e-12);

  free(x_out);
  CHECK_STATUS(perflibs_spmat_destroy(triangular));
  CHECK_STATUS(perflibs_spmat_destroy(X));
  CHECK_STATUS(perflibs_spmat_destroy(Y));
  return EXIT_SUCCESS;
}

int main() {
  if (test_identity_and_null_paths() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_dense_and_csc_triangular_paths() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_hint_switch_and_rectangular_error() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_spsm_hint_switch() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
