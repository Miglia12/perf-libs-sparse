/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <stdlib.h>

static int test_complex_double_triangular_variants() {
  const perflibs_int_t row_ptr[] = {0, 1, 3, 4, 5};
  const perflibs_int_t col_indx[] = {0, 0, 1, 2, 3};
  const perflibs_doublecomplex_t vals[] = {TEST_Z(2.0, 0.0), TEST_Z(1.0, -1.0),
                                           TEST_Z(4.0, 2.0), TEST_Z(1.0, 0.0),
                                           TEST_Z(1.0, 0.0)};
  const perflibs_doublecomplex_t alpha = TEST_Z(1.0, 0.0);
  const perflibs_doublecomplex_t expected[4] = {
      TEST_Z(1.0, 0.0), TEST_Z(2.0, -1.0), TEST_Z(3.0, 0.0), TEST_Z(4.0, 0.0)};
  const perflibs_doublecomplex_t rhs_notrans[] = {
      TEST_Z(2.0, 0.0), TEST_Z(11.0, -1.0), TEST_Z(3.0, 0.0), TEST_Z(4.0, 0.0)};
  const perflibs_doublecomplex_t rhs_trans[] = {
      TEST_Z(3.0, -3.0), TEST_Z(10.0, 0.0), TEST_Z(3.0, 0.0), TEST_Z(4.0, 0.0)};
  const perflibs_doublecomplex_t rhs_conj[] = {
      TEST_Z(5.0, 1.0), TEST_Z(6.0, -8.0), TEST_Z(3.0, 0.0), TEST_Z(4.0, 0.0)};
  perflibs_spmat_t triangular = NULL;

  CHECK_STATUS(perflibs_spmat_create_csr_z(&triangular, 4, 4, row_ptr, col_indx,
                                           vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(triangular, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_hint(triangular,
                                   PERFLIBS_SPARSE_HINT_SPSV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spsv_optimize(triangular));

  perflibs_doublecomplex_t x_notrans[4] = {0};
  perflibs_doublecomplex_t x_trans[4] = {0};
  perflibs_doublecomplex_t x_conj[4] = {0};

  CHECK_STATUS(perflibs_spsv_exec_z(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    triangular, x_notrans, alpha, rhs_notrans));
  CHECK_DOUBLECOMPLEX_ARRAY(x_notrans, expected, 4, 1e-12);

  CHECK_STATUS(perflibs_spsv_exec_z(PERFLIBS_SPARSE_OPERATION_TRANS, triangular,
                                    x_trans, alpha, rhs_trans));
  CHECK_DOUBLECOMPLEX_ARRAY(x_trans, expected, 4, 1e-12);

  CHECK_STATUS(perflibs_spsv_exec_z(PERFLIBS_SPARSE_OPERATION_CONJTRANS,
                                    triangular, x_conj, alpha, rhs_conj));
  CHECK_DOUBLECOMPLEX_ARRAY(x_conj, expected, 4, 1e-12);

  CHECK_STATUS(perflibs_spmat_destroy(triangular));
  return EXIT_SUCCESS;
}

static int test_complex_double_diagonal_zero_error() {
  const perflibs_int_t row_ptr[] = {0, 1, 2, 3, 4};
  const perflibs_int_t col_indx[] = {0, 1, 2, 3};
  const perflibs_doublecomplex_t vals[] = {TEST_Z(0.0, 0.0), TEST_Z(1.0, 0.0),
                                           TEST_Z(1.0, 0.0), TEST_Z(1.0, 0.0)};
  perflibs_spmat_t diag = NULL;

  CHECK_STATUS(
      perflibs_spmat_create_csr_z(&diag, 4, 4, row_ptr, col_indx, vals, 0));
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
  if (test_complex_double_triangular_variants() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_complex_double_diagonal_zero_error() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
