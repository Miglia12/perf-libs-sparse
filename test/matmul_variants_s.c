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

static void compute_expected_mm(enum perflibs_sparse_hint_value transA,
                                enum perflibs_sparse_hint_value transB,
                                float alpha, float beta, const float *denseA,
                                perflibs_int_t A_rows, perflibs_int_t A_cols,
                                const float *denseB, perflibs_int_t B_rows,
                                perflibs_int_t B_cols, const float *c_init,
                                float *out) {
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
  cblas_sgemm(CblasRowMajor, blas_transA, blas_transB, (int)m, (int)n, (int)k,
              alpha, denseA, (int)A_cols, denseB, (int)B_cols, beta, out,
              (int)n);
}

static void compute_expected_add(enum perflibs_sparse_hint_value transA,
                                 enum perflibs_sparse_hint_value transB,
                                 float alpha, float beta, perflibs_int_t A_rows,
                                 perflibs_int_t A_cols, const float *denseA,
                                 perflibs_int_t B_rows, perflibs_int_t B_cols,
                                 const float *denseB, float *out) {
  const perflibs_int_t m =
      transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? A_rows : A_cols;
  const perflibs_int_t n =
      transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? A_cols : A_rows;
  for (perflibs_int_t i = 0; i < m; ++i) {
    for (perflibs_int_t j = 0; j < n; ++j) {
      const float aval = test_access_dense_s(denseA, A_cols, transA, i, j);
      const float bval = test_access_dense_s(denseB, B_cols, transB, i, j);
      out[i * n + j] = alpha * aval + beta * bval;
    }
  }
}

static int test_spmm_transpose_dense_sparse() {
  const perflibs_int_t row_ptrA[] = {0, 2, 4, 6, 8};
  const perflibs_int_t col_indxA[] = {0, 2, 0, 1, 1, 2, 2, 3};
  const float valsA[] = {1.0f, -1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  const float denseA[] = {1.0f, 0.0f, -1.0f, 0.0f, 2.0f, 3.0f, 0.0f, 0.0f,
                          0.0f, 4.0f, 5.0f,  0.0f, 0.0f, 0.0f, 6.0f, 7.0f};
  const float denseB[] = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                          7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                          13.0f, 14.0f, 15.0f, 16.0f};

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t B = NULL;

  CHECK_STATUS(
      perflibs_spmat_create_csr_s(&A, 4, 4, row_ptrA, col_indxA, valsA, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_s(&B, PERFLIBS_ROW_MAJOR, 4, 4, 4,
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
  float expected[16];
  for (size_t ia = 0; ia < sizeof(trans_modes) / sizeof(trans_modes[0]); ++ia) {
    for (size_t ib = 0; ib < sizeof(trans_modes) / sizeof(trans_modes[0]);
         ++ib) {
      const enum perflibs_sparse_hint_value transA = trans_modes[ia];
      const enum perflibs_sparse_hint_value transB = trans_modes[ib];
      perflibs_spmat_t C = perflibs_spmat_create_null(4, 4);
      CHECK_TRUE(C != NULL, "spmm result creation failed");
      const float alpha =
          transA == PERFLIBS_SPARSE_OPERATION_NOTRANS ? 1.0f : 1.5f;
      const float beta = 0.0f;
      CHECK_STATUS(perflibs_spmm_exec_s(transA, transB, alpha, A, B, beta, C));
      perflibs_int_t m = -1;
      perflibs_int_t n = -1;
      float *dense = NULL;
      CHECK_STATUS(
          perflibs_spmat_export_dense_s(C, PERFLIBS_ROW_MAJOR, &m, &n, &dense));
      compute_expected_mm(transA, transB, alpha, beta, denseA, 4, 4, denseB, 4,
                          4, NULL, expected);
      CHECK_FLOAT_ARRAY(dense, expected, m * n, 5e-5f);
      free(dense);
      perflibs_spmat_destroy(C);
    }
  }

  const float c_init[] = {1.0f, -1.0f, 0.5f,  2.0f, 0.0f, 0.25f, -0.5f, 0.5f,
                          1.0f, 1.5f,  -1.5f, 2.5f, 3.0f, -2.0f, 4.0f,  5.0f};
  perflibs_spmat_t c_dense = NULL;
  CHECK_STATUS(perflibs_spmat_create_dense_s(&c_dense, PERFLIBS_ROW_MAJOR, 4, 4,
                                             4, c_init, 0));
  const float alpha_beta = 0.7f;
  const float beta = 1.0f;
  CHECK_STATUS(perflibs_spmm_exec_s(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    alpha_beta, A, B, beta, c_dense));
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  float *dense_result = NULL;
  CHECK_STATUS(perflibs_spmat_export_dense_s(c_dense, PERFLIBS_ROW_MAJOR, &m,
                                             &n, &dense_result));
  compute_expected_mm(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                      PERFLIBS_SPARSE_OPERATION_NOTRANS, alpha_beta, beta,
                      denseA, 4, 4, denseB, 4, 4, c_init, expected);
  CHECK_FLOAT_ARRAY(dense_result, expected, m * n, 5e-5f);

  free(dense_result);
  perflibs_spmat_destroy(c_dense);
  perflibs_spmat_destroy(A);
  perflibs_spmat_destroy(B);
  return EXIT_SUCCESS;
}

static int test_spadd_transpose_index_variants() {
  const perflibs_int_t row_ptrA[] = {1, 3, 5, 7, 9};
  const perflibs_int_t col_indxA[] = {1, 3, 1, 4, 2, 3, 1, 4};
  const float valsA[] = {1.0f, -1.0f, 2.5f, 3.0f, -0.5f, 4.0f, 1.5f, -2.0f};
  const perflibs_int_t row_ptrB[] = {0, 2, 4, 6, 8};
  const perflibs_int_t col_indxB[] = {1, 3, 0, 2, 1, 3, 0, 2};
  const float valsB[] = {4.0f, -2.0f, 5.0f, -1.5f, 2.0f, 3.5f, -3.0f, 1.0f};
  float denseA[16];
  float denseB[16];

  test_csr_to_dense_s(4, 4, row_ptrA, col_indxA, valsA, row_ptrA[0], denseA);
  test_csr_to_dense_s(4, 4, row_ptrB, col_indxB, valsB, row_ptrB[0], denseB);

  perflibs_spmat_t A = NULL;
  perflibs_spmat_t B = NULL;
  CHECK_STATUS(
      perflibs_spmat_create_csr_s(&A, 4, 4, row_ptrA, col_indxA, valsA, 0));
  CHECK_STATUS(
      perflibs_spmat_create_csr_s(&B, 4, 4, row_ptrB, col_indxB, valsB, 0));

  perflibs_spmat_t optimize_null = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(optimize_null != NULL, "spadd null output creation failed");
  CHECK_STATUS(perflibs_spadd_optimize(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_TRANS,
      PERFLIBS_SPARSE_SCALAR_ANY, A, PERFLIBS_SPARSE_SCALAR_ANY, B,
      optimize_null));
  perflibs_spmat_destroy(optimize_null);

  perflibs_spmat_t C = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(C != NULL, "spadd result creation failed");
  CHECK_STATUS(perflibs_spadd_exec_s(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                     PERFLIBS_SPARSE_OPERATION_TRANS, 1.0f, A,
                                     1.0f, B, C));
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  float *dense_result = NULL;
  CHECK_STATUS(perflibs_spmat_export_dense_s(C, PERFLIBS_ROW_MAJOR, &m, &n,
                                             &dense_result));
  float expected_sum[16];
  compute_expected_add(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                       PERFLIBS_SPARSE_OPERATION_TRANS, 1.0f, 1.0f, 4, 4,
                       denseA, 4, 4, denseB, expected_sum);
  CHECK_FLOAT_ARRAY(dense_result, expected_sum, m * n, 5e-5f);

  free(dense_result);
  perflibs_spmat_destroy(C);

  perflibs_spmat_t C_bonly = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(C_bonly != NULL, "spadd result creation failed");
  CHECK_STATUS(perflibs_spadd_exec_s(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                     PERFLIBS_SPARSE_OPERATION_TRANS, 0.0f, A,
                                     1.0f, B, C_bonly));
  dense_result = NULL;
  CHECK_STATUS(perflibs_spmat_export_dense_s(C_bonly, PERFLIBS_ROW_MAJOR, &m,
                                             &n, &dense_result));
  float expected_beta_only[16];
  compute_expected_add(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                       PERFLIBS_SPARSE_OPERATION_TRANS, 0.0f, 1.0f, 4, 4,
                       denseA, 4, 4, denseB, expected_beta_only);
  CHECK_FLOAT_ARRAY(dense_result, expected_beta_only, m * n, 5e-5f);

  free(dense_result);
  perflibs_spmat_destroy(C_bonly);
  perflibs_spmat_destroy(A);
  perflibs_spmat_destroy(B);
  return EXIT_SUCCESS;
}

int main() {
  if (test_spmm_transpose_dense_sparse() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_spadd_transpose_index_variants() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
