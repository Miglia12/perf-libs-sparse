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
    enum perflibs_sparse_hint_value transB, perflibs_singlecomplex_t alpha,
    perflibs_singlecomplex_t beta, const perflibs_singlecomplex_t *denseA,
    perflibs_int_t A_rows, perflibs_int_t A_cols,
    const perflibs_singlecomplex_t *denseB, perflibs_int_t B_rows,
    perflibs_int_t B_cols, const perflibs_singlecomplex_t *c_init,
    perflibs_singlecomplex_t *out) {
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
  cblas_cgemm(CblasRowMajor, blas_transA, blas_transB, (int)m, (int)n, (int)k,
              &alpha, denseA, (int)A_cols, denseB, (int)B_cols, &beta, out,
              (int)n);
}

static void compute_expected_add_complex(
    enum perflibs_sparse_hint_value transA,
    enum perflibs_sparse_hint_value transB, perflibs_singlecomplex_t alpha,
    perflibs_singlecomplex_t beta, perflibs_int_t A_rows, perflibs_int_t A_cols,
    const perflibs_singlecomplex_t *denseA, perflibs_int_t B_rows,
    perflibs_int_t B_cols, const perflibs_singlecomplex_t *denseB,
    perflibs_singlecomplex_t *out) {
  perflibs_int_t m =
      transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? A_rows : A_cols;
  perflibs_int_t n =
      transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? A_cols : A_rows;
  for (perflibs_int_t i = 0; i < m; ++i) {
    for (perflibs_int_t j = 0; j < n; ++j) {
      perflibs_singlecomplex_t aval =
          test_access_dense_c(denseA, A_cols, transA, i, j);
      perflibs_singlecomplex_t bval =
          test_access_dense_c(denseB, B_cols, transB, i, j);
      out[i * n + j] =
          test_caddf(test_cmulf(alpha, aval), test_cmulf(beta, bval));
    }
  }
}

static int test_spmm_complex_variants() {
  const perflibs_int_t row_ptrA[] = {0, 2, 4, 6, 8};
  const perflibs_int_t col_indxA[] = {0, 2, 0, 1, 1, 2, 2, 3};
  const perflibs_singlecomplex_t valsA[] = {
      TEST_C(1.0f, 1.0f),  TEST_C(-1.0f, 0.5f), TEST_C(2.0f, 0.0f),
      TEST_C(0.0f, 3.0f),  TEST_C(4.0f, -1.0f), TEST_C(5.0f, 2.0f),
      TEST_C(6.0f, -0.5f), TEST_C(7.0f, 1.0f)};
  const perflibs_singlecomplex_t denseA[] = {
      TEST_C(1.0f, 1.0f),  TEST_C(0.0f, 0.0f), TEST_C(-1.0f, 0.5f),
      TEST_C(0.0f, 0.0f),  TEST_C(2.0f, 0.0f), TEST_C(0.0f, 3.0f),
      TEST_C(0.0f, 0.0f),  TEST_C(0.0f, 0.0f), TEST_C(0.0f, 0.0f),
      TEST_C(4.0f, -1.0f), TEST_C(5.0f, 2.0f), TEST_C(0.0f, 0.0f),
      TEST_C(0.0f, 0.0f),  TEST_C(0.0f, 0.0f), TEST_C(6.0f, -0.5f),
      TEST_C(7.0f, 1.0f)};
  const perflibs_singlecomplex_t denseB[] = {
      TEST_C(1.0f, 0.0f),  TEST_C(2.0f, 1.0f),   TEST_C(3.0f, -1.0f),
      TEST_C(4.0f, 0.5f),  TEST_C(5.0f, 2.0f),   TEST_C(6.0f, 1.0f),
      TEST_C(7.0f, 0.0f),  TEST_C(8.0f, -1.0f),  TEST_C(9.0f, 3.0f),
      TEST_C(10.0f, 0.0f), TEST_C(11.0f, -2.0f), TEST_C(12.0f, 1.5f),
      TEST_C(13.0f, 0.0f), TEST_C(14.0f, -1.0f), TEST_C(15.0f, 2.0f),
      TEST_C(16.0f, 0.0f)};

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t B = NULL;
  CHECK_STATUS(
      perflibs_spmat_create_csr_c(&A, 4, 4, row_ptrA, col_indxA, valsA, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_c(&B, PERFLIBS_ROW_MAJOR, 4, 4, 4,
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
  perflibs_singlecomplex_t expected[16];
  for (size_t ia = 0; ia < sizeof(trans_modes) / sizeof(trans_modes[0]); ++ia) {
    for (size_t ib = 0; ib < sizeof(trans_modes) / sizeof(trans_modes[0]);
         ++ib) {
      enum perflibs_sparse_hint_value transA = trans_modes[ia];
      enum perflibs_sparse_hint_value transB = trans_modes[ib];
      perflibs_spmat_t C = perflibs_spmat_create_null(4, 4);
      CHECK_TRUE(C != NULL, "spmm result creation failed");
      perflibs_singlecomplex_t alpha = TEST_C(1.0f, 0.5f);
      perflibs_singlecomplex_t beta = TEST_C(0.0f, 0.0f);
      CHECK_STATUS(perflibs_spmm_exec_c(transA, transB, alpha, A, B, beta, C));
      perflibs_int_t m = -1;
      perflibs_int_t n = -1;
      perflibs_singlecomplex_t *dense = NULL;
      CHECK_STATUS(
          perflibs_spmat_export_dense_c(C, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
      compute_expected_mm_complex(transA, transB, alpha, beta, denseA, 4, 4,
                                  denseB, 4, 4, NULL, expected);
      CHECK_SINGLECOMPLEX_ARRAY(dense, expected, m * n, 2e-5f);
      free(dense);
      perflibs_spmat_destroy(C);
    }
  }

  const perflibs_singlecomplex_t c_init[] = {
      TEST_C(1.0f, 0.0f),  TEST_C(-1.0f, 1.0f), TEST_C(0.5f, 0.2f),
      TEST_C(2.0f, -1.0f), TEST_C(0.0f, 0.0f),  TEST_C(0.25f, 0.1f),
      TEST_C(-0.5f, 0.5f), TEST_C(0.5f, -0.5f), TEST_C(1.0f, 0.0f),
      TEST_C(1.5f, 0.0f),  TEST_C(-1.5f, 1.0f), TEST_C(2.5f, -0.5f),
      TEST_C(3.0f, 0.5f),  TEST_C(-2.0f, 0.0f), TEST_C(4.0f, 1.0f),
      TEST_C(5.0f, -1.0f)};
  perflibs_spmat_t c_dense = NULL;
  CHECK_STATUS(perflibs_spmat_create_dense_c(&c_dense, PERFLIBS_ROW_MAJOR, 4, 4,
                                             4, c_init, 0));
  perflibs_singlecomplex_t alpha_beta = TEST_C(0.7f, 0.3f);
  perflibs_singlecomplex_t beta = TEST_C(1.0f, 0.5f);
  CHECK_STATUS(perflibs_spmm_exec_c(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    alpha_beta, A, B, beta, c_dense));
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  perflibs_singlecomplex_t *dense_result = NULL;
  CHECK_STATUS(perflibs_spmat_export_dense_c(c_dense, PERFLIBS_ROW_MAJOR, &m,
                                             &n, &dense_result));
  compute_expected_mm_complex(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      alpha_beta, beta, denseA, 4, 4, denseB, 4, 4, c_init, expected);
  CHECK_SINGLECOMPLEX_ARRAY(dense_result, expected, m * n, 2e-5f);
  free(dense_result);
  perflibs_spmat_destroy(c_dense);
  perflibs_spmat_destroy(A);
  perflibs_spmat_destroy(B);
  return EXIT_SUCCESS;
}

static int test_spadd_complex_variants() {
  const perflibs_int_t row_ptrA[] = {1, 3, 5, 7, 9};
  const perflibs_int_t col_indxA[] = {1, 3, 1, 4, 2, 3, 1, 4};
  const perflibs_singlecomplex_t valsA[] = {
      TEST_C(1.0f, 1.0f), TEST_C(-1.0f, 2.0f), TEST_C(2.5f, -1.0f),
      TEST_C(3.0f, 0.5f), TEST_C(-0.5f, 0.0f), TEST_C(4.0f, -1.5f),
      TEST_C(1.5f, 0.5f), TEST_C(-2.0f, 1.0f)};
  const perflibs_int_t row_ptrB[] = {0, 2, 4, 6, 8};
  const perflibs_int_t col_indxB[] = {1, 3, 0, 2, 1, 3, 0, 2};
  const perflibs_singlecomplex_t valsB[] = {
      TEST_C(4.0f, 0.5f), TEST_C(5.0f, -1.0f), TEST_C(-1.5f, 1.5f),
      TEST_C(2.0f, 0.0f), TEST_C(3.5f, -0.5f), TEST_C(-3.0f, 0.0f),
      TEST_C(1.0f, 1.0f), TEST_C(-2.5f, 0.5f)};
  perflibs_singlecomplex_t denseA[16];
  perflibs_singlecomplex_t denseB[16];

  test_csr_to_dense_c(4, 4, row_ptrA, col_indxA, valsA, row_ptrA[0], denseA);
  test_csr_to_dense_c(4, 4, row_ptrB, col_indxB, valsB, row_ptrB[0], denseB);

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t B = NULL;
  CHECK_STATUS(
      perflibs_spmat_create_csr_c(&A, 4, 4, row_ptrA, col_indxA, valsA, 0));
  CHECK_STATUS(
      perflibs_spmat_create_csr_c(&B, 4, 4, row_ptrB, col_indxB, valsB, 0));

  perflibs_spmat_t optimize_null = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(optimize_null != NULL, "spadd null output creation failed");
  CHECK_STATUS(perflibs_spadd_optimize(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_TRANS,
      PERFLIBS_SPARSE_SCALAR_ANY, A, PERFLIBS_SPARSE_SCALAR_ANY, B,
      optimize_null));
  perflibs_spmat_destroy(optimize_null);

  perflibs_spmat_t C = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(C != NULL, "spadd result creation failed");
  CHECK_STATUS(perflibs_spadd_exec_c(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_TRANS,
      TEST_C(1.0f, 0.0f), A, TEST_C(1.0f, 0.5f), B, C));
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  perflibs_singlecomplex_t *dense_result = NULL;
  CHECK_STATUS(perflibs_spmat_export_dense_c(C, PERFLIBS_ROW_MAJOR, &m, &n,
                                             &dense_result));
  perflibs_singlecomplex_t expected[16];
  compute_expected_add_complex(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                               PERFLIBS_SPARSE_OPERATION_TRANS,
                               TEST_C(1.0f, 0.0f), TEST_C(1.0f, 0.5f), 4, 4,
                               denseA, 4, 4, denseB, expected);
  CHECK_SINGLECOMPLEX_ARRAY(dense_result, expected, m * n, 2e-5f);
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
