/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <stdlib.h>

static int test_lower_triangular_transforms() {
  const perflibs_int_t row_ptr[] = {0, 1, 3, 5, 6};
  const perflibs_int_t col_indx[] = {0, 0, 1, 1, 2, 3};
  const double vals[] = {2.0, 1.0, 3.0, 1.0, 4.0, 1.0};
  perflibs_spmat_t triangular = NULL;

  CHECK_STATUS(perflibs_spmat_create_csr_d(&triangular, 4, 4, row_ptr, col_indx,
                                           vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(triangular, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_hint(triangular,
                                   PERFLIBS_SPARSE_HINT_SPSV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spsv_optimize(triangular));

  double x_notrans[4] = {0.0, 0.0, 0.0, 0.0};
  double rhs_notrans[4] = {2.0, 10.0, 11.0, 4.0};
  const double expected[4] = {1.0, 3.0, 2.0, 4.0};
  CHECK_STATUS(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    triangular, x_notrans, 1.0, rhs_notrans));
  CHECK_DOUBLE_ARRAY(x_notrans, expected, 4, 1e-12);

  double x_trans[4] = {0.0, 0.0, 0.0, 0.0};
  double rhs_trans[4] = {5.0, 11.0, 8.0, 4.0};
  CHECK_STATUS(perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_TRANS, triangular,
                                    x_trans, 1.0, rhs_trans));
  CHECK_DOUBLE_ARRAY(x_trans, expected, 4, 1e-12);

  CHECK_STATUS(perflibs_spmat_destroy(triangular));
  return EXIT_SUCCESS;
}

static int test_complex_conj_transpose() {
  const perflibs_int_t row_ptr[] = {0, 1, 3, 4, 5};
  const perflibs_int_t col_indx[] = {0, 0, 1, 2, 3};
  const perflibs_singlecomplex_t vals[] = {
      TEST_C(2.0f, 0.0f), TEST_C(1.0f, -1.0f), TEST_C(4.0f, 2.0f),
      TEST_C(1.0f, 0.0f), TEST_C(1.0f, 0.0f)};
  perflibs_spmat_t triangular = NULL;

  CHECK_STATUS(perflibs_spmat_create_csr_c(&triangular, 4, 4, row_ptr, col_indx,
                                           vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(triangular, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_hint(triangular,
                                   PERFLIBS_SPARSE_HINT_SPSV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_CONJTRANS));
  CHECK_STATUS(perflibs_spsv_optimize(triangular));

  const perflibs_singlecomplex_t rhs_conj[] = {
      TEST_C(5.0f, 1.0f), TEST_C(6.0f, -8.0f), TEST_C(3.0f, 0.0f),
      TEST_C(4.0f, 0.0f)};
  const perflibs_singlecomplex_t rhs_notrans[] = {
      TEST_C(2.0f, 0.0f), TEST_C(11.0f, -1.0f), TEST_C(3.0f, 0.0f),
      TEST_C(4.0f, 0.0f)};
  const perflibs_singlecomplex_t alpha =
      TEST_C(1.0f, 0.0f); // For both solves we scale by one
  const perflibs_singlecomplex_t expected[4] = {
      TEST_C(1.0f, 0.0f), TEST_C(2.0f, -1.0f), TEST_C(3.0f, 0.0f),
      TEST_C(4.0f, 0.0f)};

  perflibs_singlecomplex_t x_conj[4] = {0};
  perflibs_singlecomplex_t x_notrans[4] = {0};

  CHECK_STATUS(perflibs_spsv_exec_c(PERFLIBS_SPARSE_OPERATION_CONJTRANS,
                                    triangular, x_conj, alpha, rhs_conj));
  CHECK_STATUS(perflibs_spsv_exec_c(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    triangular, x_notrans, alpha, rhs_notrans));
  CHECK_SINGLECOMPLEX_ARRAY(x_conj, expected, 4, 1e-6f);
  CHECK_SINGLECOMPLEX_ARRAY(x_notrans, expected, 4, 1e-6f);

  CHECK_STATUS(perflibs_spmat_destroy(triangular));
  return EXIT_SUCCESS;
}

static int test_diagonal_zero_error() {
  const perflibs_int_t row_ptr[] = {0, 1, 2, 3, 4};
  const perflibs_int_t col_indx[] = {0, 1, 2, 3};
  const double vals[] = {0.0, 1.0, 1.0, 1.0};
  perflibs_spmat_t diag = NULL;

  CHECK_STATUS(
      perflibs_spmat_create_csr_d(&diag, 4, 4, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(diag, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_hint(diag, PERFLIBS_SPARSE_HINT_SPSV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS_EQ(perflibs_spsv_optimize(diag),
                  PERFLIBS_STATUS_INPUT_PARAMETER_ERROR);

  CHECK_STATUS(perflibs_spmat_destroy(diag));
  return EXIT_SUCCESS;
}

int main() {
  CHECK_STATUS(test_lower_triangular_transforms());
  CHECK_STATUS(test_complex_conj_transpose());
  CHECK_STATUS(test_diagonal_zero_error());
  return EXIT_SUCCESS;
}
