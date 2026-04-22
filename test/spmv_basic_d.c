/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <stdlib.h>

static int test_basic_spmv_exec() {
  const perflibs_int_t m = 5;
  const perflibs_int_t n = 5;
  const double alpha = 1.0;
  const double beta = 0.0;
  const double vals[] = {1.0, 2.0, 3.0, 4.0,  5.0,  6.0,
                         7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  const perflibs_int_t row_ptr[] = {0, 2, 4, 7, 9, 12};
  const perflibs_int_t col_indx[] = {0, 2, 1, 3, 1, 2, 3, 2, 3, 2, 3, 4};
  const double expected[] = {3.0, 7.0, 18.0, 17.0, 33.0};

  perflibs_spmat_t A = NULL;
  double *x = (double *)malloc(sizeof(double) * n);
  double *y = (double *)malloc(sizeof(double) * m);

  CHECK_TRUE(x && y, "allocation failure in spmv test");

  for (perflibs_int_t i = 0; i < n; ++i) {
    x[i] = 1.0;
  }

  CHECK_STATUS(
      perflibs_spmat_create_csr_d(&A, m, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_UNSTRUCTURED));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_MANY));
  CHECK_STATUS(perflibs_spmv_optimize(A));

  CHECK_STATUS(perflibs_spmv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, alpha, A,
                                    x, beta, y));

  CHECK_DOUBLE_ARRAY(y, expected, m, 1e-12);

  CHECK_STATUS(perflibs_spmat_destroy(A));
  free(x);
  free(y);
  return EXIT_SUCCESS;
}

int main() {
  CHECK_STATUS(test_basic_spmv_exec());
  return EXIT_SUCCESS;
}
