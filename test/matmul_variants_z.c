/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <cblas.h>
#include <stdlib.h>

static void compute_expected_mm_complex(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_doublecomplex_t alpha,
    perflibs_doublecomplex_t beta, const perflibs_doublecomplex_t *denseA,
    perflibs_int_t A_rows, perflibs_int_t A_cols,
    const perflibs_doublecomplex_t *denseB, perflibs_int_t B_rows,
    perflibs_int_t B_cols, const perflibs_doublecomplex_t *c_init,
    perflibs_doublecomplex_t *out) {
  const perflibs_int_t m =
      transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? A_rows : A_cols;
  const perflibs_int_t n =
      transB == PERFLIBS_SPARSE_OPERATION_NOTRANS ? B_cols : B_rows;
  const perflibs_int_t k =
      transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? A_cols : A_rows;
  const CBLAS_TRANSPOSE blas_transA =
      transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? CblasNoTrans : CblasTrans;
  const CBLAS_TRANSPOSE blas_transB =
      transB == PERFLIBS_SPARSE_OPERATION_NOTRANS ? CblasNoTrans : CblasTrans;
  if (c_init != NULL) {
    memcpy(out, c_init, (size_t)(m * n) * sizeof(*out));
  }
  cblas_zgemm(CblasRowMajor, blas_transA, blas_transB, (int)m, (int)n, (int)k,
              &alpha, denseA, (int)A_cols, denseB, (int)B_cols, &beta, out,
              (int)n);
}

static void compute_expected_add_complex(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_doublecomplex_t alpha,
    perflibs_doublecomplex_t beta, perflibs_int_t A_rows, perflibs_int_t A_cols,
    const perflibs_doublecomplex_t *denseA, perflibs_int_t B_rows,
    perflibs_int_t B_cols, const perflibs_doublecomplex_t *denseB,
    perflibs_doublecomplex_t *out) {
  perflibs_int_t m =
      transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? A_rows : A_cols;
  perflibs_int_t n =
      transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? A_cols : A_rows;
  for (perflibs_int_t i = 0; i < m; ++i) {
    for (perflibs_int_t j = 0; j < n; ++j) {
      perflibs_doublecomplex_t aval =
          test_access_dense_z(denseA, A_cols, transA, i, j);
      perflibs_doublecomplex_t bval =
          test_access_dense_z(denseB, B_cols, transB, i, j);
      out[i * n + j] = test_cadd(test_cmul(alpha, aval), test_cmul(beta, bval));
    }
  }
}

static int test_spmm_complex_variants() {
  const perflibs_int_t row_ptrA[] = {0, 2, 4, 6, 8};
  const perflibs_int_t col_indxA[] = {0, 2, 0, 1, 1, 2, 2, 3};
  const perflibs_doublecomplex_t valsA[] = {
      TEST_Z(1.0, 1.0),  TEST_Z(-1.0, 0.5), TEST_Z(2.0, 0.0),
      TEST_Z(0.0, 3.0),  TEST_Z(4.0, -1.0), TEST_Z(5.0, 2.0),
      TEST_Z(6.0, -0.5), TEST_Z(7.0, 1.0)};
  const perflibs_doublecomplex_t denseA[] = {
      TEST_Z(1.0, 1.0), TEST_Z(0.0, 0.0),  TEST_Z(-1.0, 0.5), TEST_Z(0.0, 0.0),
      TEST_Z(2.0, 0.0), TEST_Z(0.0, 3.0),  TEST_Z(0.0, 0.0),  TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0), TEST_Z(4.0, -1.0), TEST_Z(5.0, 2.0),  TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0), TEST_Z(0.0, 0.0),  TEST_Z(6.0, -0.5), TEST_Z(7.0, 1.0)};
  const perflibs_doublecomplex_t denseB[] = {
      TEST_Z(1.0, 0.0),  TEST_Z(2.0, 1.0),   TEST_Z(3.0, -1.0),
      TEST_Z(4.0, 0.5),  TEST_Z(5.0, 2.0),   TEST_Z(6.0, 1.0),
      TEST_Z(7.0, 0.0),  TEST_Z(8.0, -1.0),  TEST_Z(9.0, 3.0),
      TEST_Z(10.0, 0.0), TEST_Z(11.0, -2.0), TEST_Z(12.0, 1.5),
      TEST_Z(13.0, 0.0), TEST_Z(14.0, -1.0), TEST_Z(15.0, 2.0),
      TEST_Z(16.0, 0.0)};

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t B = NULL;
  CHECK_STATUS(
      perflibs_spmat_create_csr_z(&A, 4, 4, row_ptrA, col_indxA, valsA, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_z(&B, PERFLIBS_ROW_MAJOR, 4, 4, 4,
                                             denseB, 0));
  perflibs_spmat_t optimize_null = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(optimize_null != NULL, "spmm null output creation failed");
  CHECK_STATUS(perflibs_spmm_optimize(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      PERFLIBS_SPARSE_SCALAR_ANY, A, B, PERFLIBS_SPARSE_SCALAR_ANY,
      optimize_null));
  perflibs_spmat_destroy(optimize_null);

  const enum perflibs_sparse_hint_value trans_modes[] = {
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_TRANS};
  perflibs_doublecomplex_t expected[16];
  for (size_t ia = 0; ia < sizeof(trans_modes) / sizeof(trans_modes[0]); ++ia) {
    for (size_t ib = 0; ib < sizeof(trans_modes) / sizeof(trans_modes[0]);
         ++ib) {
      enum perflibs_sparse_hint_value transA = trans_modes[ia];
      enum perflibs_sparse_hint_value transB = trans_modes[ib];
      perflibs_int_t rows = 4;
      perflibs_int_t cols = 4;
      perflibs_spmat_t C = perflibs_spmat_create_null(rows, cols);
      CHECK_TRUE(C != NULL, "spmm result creation failed");
      perflibs_doublecomplex_t alpha = TEST_Z(1.0, 0.5);
      perflibs_doublecomplex_t beta = TEST_Z(0.0, 0.0);
      CHECK_STATUS(perflibs_spmm_exec_z(transA, transB, alpha, A, B, beta, C));
      perflibs_int_t m = -1, n = -1;
      perflibs_doublecomplex_t *dense = NULL;
      CHECK_STATUS(
          perflibs_spmat_export_dense_z(C, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
      compute_expected_mm_complex(transA, transB, alpha, beta, denseA, 4, 4,
                                  denseB, 4, 4, NULL, expected);
      CHECK_DOUBLECOMPLEX_ARRAY(dense, expected, m * n, 1e-12);
      free(dense);
      perflibs_spmat_destroy(C);
    }
  }

  const perflibs_doublecomplex_t c_init[] = {
      TEST_Z(1.0, 0.0),  TEST_Z(-1.0, 1.0), TEST_Z(0.5, 0.2),
      TEST_Z(2.0, -1.0), TEST_Z(0.0, 0.0),  TEST_Z(0.25, 0.1),
      TEST_Z(-0.5, 0.5), TEST_Z(0.5, -0.5), TEST_Z(1.0, 0.0),
      TEST_Z(1.5, 0.0),  TEST_Z(-1.5, 1.0), TEST_Z(2.5, -0.5),
      TEST_Z(3.0, 0.5),  TEST_Z(-2.0, 0.0), TEST_Z(4.0, 1.0),
      TEST_Z(5.0, -1.0)};
  perflibs_spmat_t c_dense = NULL;
  CHECK_STATUS(perflibs_spmat_create_dense_z(&c_dense, PERFLIBS_ROW_MAJOR, 4, 4,
                                             4, c_init, 0));
  perflibs_doublecomplex_t alpha_beta = TEST_Z(0.7, 0.3);
  perflibs_doublecomplex_t beta = TEST_Z(1.0, 0.5);
  CHECK_STATUS(perflibs_spmm_exec_z(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    alpha_beta, A, B, beta, c_dense));
  perflibs_int_t m = -1, n = -1;
  perflibs_doublecomplex_t *dense_result = NULL;
  CHECK_STATUS(perflibs_spmat_export_dense_z(c_dense, PERFLIBS_ROW_MAJOR, &m,
                                             &n, &dense_result));
  compute_expected_mm_complex(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      alpha_beta, beta, denseA, 4, 4, denseB, 4, 4, c_init, expected);
  CHECK_DOUBLECOMPLEX_ARRAY(dense_result, expected, m * n, 1e-12);
  free(dense_result);
  perflibs_spmat_destroy(c_dense);
  perflibs_spmat_destroy(A);
  perflibs_spmat_destroy(B);
  return EXIT_SUCCESS;
}

static int test_spadd_complex_variants() {
  const perflibs_int_t row_ptrA[] = {1, 3, 5, 7, 9};
  const perflibs_int_t col_indxA[] = {1, 3, 1, 4, 2, 3, 1, 4};
  const perflibs_doublecomplex_t valsA[] = {
      TEST_Z(1.0, 1.0), TEST_Z(-1.0, 2.0), TEST_Z(2.5, -1.0),
      TEST_Z(3.0, 0.5), TEST_Z(-0.5, 0.0), TEST_Z(4.0, -1.5),
      TEST_Z(1.5, 0.5), TEST_Z(-2.0, 1.0)};
  const perflibs_int_t row_ptrB[] = {0, 2, 4, 6, 8};
  const perflibs_int_t col_indxB[] = {1, 3, 0, 2, 1, 3, 0, 2};
  const perflibs_doublecomplex_t valsB[] = {
      TEST_Z(4.0, 0.5), TEST_Z(5.0, -1.0), TEST_Z(-1.5, 1.5),
      TEST_Z(2.0, 0.0), TEST_Z(3.5, -0.5), TEST_Z(-3.0, 0.0),
      TEST_Z(1.0, 1.0), TEST_Z(-2.5, 0.5)};
  perflibs_doublecomplex_t denseA[16];
  perflibs_doublecomplex_t denseB[16];

  test_csr_to_dense_z(4, 4, row_ptrA, col_indxA, valsA, row_ptrA[0], denseA);
  test_csr_to_dense_z(4, 4, row_ptrB, col_indxB, valsB, row_ptrB[0], denseB);
  perflibs_spmat_t A = NULL;
  perflibs_spmat_t B = NULL;
  CHECK_STATUS(
      perflibs_spmat_create_csr_z(&A, 4, 4, row_ptrA, col_indxA, valsA, 0));
  CHECK_STATUS(
      perflibs_spmat_create_csr_z(&B, 4, 4, row_ptrB, col_indxB, valsB, 0));

  perflibs_spmat_t optimize_null = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(optimize_null != NULL, "spadd null output creation failed");
  CHECK_STATUS(perflibs_spadd_optimize(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_TRANS,
      PERFLIBS_SPARSE_SCALAR_ANY, A, PERFLIBS_SPARSE_SCALAR_ANY, B,
      optimize_null));
  perflibs_spmat_destroy(optimize_null);

  perflibs_spmat_t C = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(C != NULL, "spadd result creation failed");
  CHECK_STATUS(perflibs_spadd_exec_z(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_TRANS,
      TEST_Z(1.0, 0.0), A, TEST_Z(1.0, 0.5), B, C));
  perflibs_int_t m = -1, n = -1;
  perflibs_doublecomplex_t *dense_result = NULL;
  CHECK_STATUS(perflibs_spmat_export_dense_z(C, PERFLIBS_ROW_MAJOR, &m, &n,
                                             &dense_result));
  perflibs_doublecomplex_t expected[16];
  compute_expected_add_complex(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_TRANS,
      TEST_Z(1.0, 0.0), TEST_Z(1.0, 0.5), 4, 4, denseA, 4, 4, denseB, expected);
  CHECK_DOUBLECOMPLEX_ARRAY(dense_result, expected, m * n, 1e-12);
  free(dense_result);
  perflibs_spmat_destroy(C);
  perflibs_spmat_destroy(A);
  perflibs_spmat_destroy(B);
  return EXIT_SUCCESS;
}

int main() {
  if (test_spmm_complex_variants() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_spadd_complex_variants() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
