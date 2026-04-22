/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <stdlib.h>

static int test_basic_spsm_exec() {
  const perflibs_int_t n = 5;
  const perflibs_int_t nrhs = 2;
  const double alpha = 1.0;

  /*
   * Lower-triangular A (column-major):
   * [1 0 0 0 0
   *  2 1 0 0 0
   *  0 3 1 0 0
   *  0 0 4 1 0
   *  0 0 0 5 1]
   */
  const double A_vals[] = {1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0,
                           0.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0,
                           1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  const double Y_vals[] = {1.0, 4.0,  9.0,  16.0, 25.0,
                           5.0, 14.0, 15.0, 14.0, 11.0};
  const double expected_X[] = {1.0, 2.0, 3.0, 4.0, 5.0,
                               5.0, 4.0, 3.0, 2.0, 1.0};
  double X_init[10] = {0.0};

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_spmat_t Y = NULL;
  double *X_out = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t out_n = -1;

  CHECK_STATUS(perflibs_spmat_create_dense_d(&A, PERFLIBS_COL_MAJOR, n, n, n,
                                             A_vals, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_d(&X, PERFLIBS_COL_MAJOR, n, nrhs, n,
                                             X_init, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_d(&Y, PERFLIBS_COL_MAJOR, n, nrhs, n,
                                             Y_vals, 0));

  CHECK_STATUS(perflibs_spsm_optimize(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X,
                                      PERFLIBS_SPARSE_SCALAR_ANY, Y));

  CHECK_STATUS(
      perflibs_spsm_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, X, alpha, Y));

  CHECK_STATUS(
      perflibs_spmat_export_dense_d(X, PERFLIBS_COL_MAJOR, &m, &out_n, &X_out));
  CHECK_TRUE(m == n && out_n == nrhs,
             "unexpected exported SpSM solution shape %lld x %lld", test_i64(m),
             test_i64(out_n));
  CHECK_DOUBLE_ARRAY(X_out, expected_X, n * nrhs, 1e-12);

  free(X_out);
  CHECK_STATUS(perflibs_spmat_destroy(A));
  CHECK_STATUS(perflibs_spmat_destroy(X));
  CHECK_STATUS(perflibs_spmat_destroy(Y));
  return EXIT_SUCCESS;
}

int main() {
  CHECK_STATUS(test_basic_spsm_exec());
  return EXIT_SUCCESS;
}
