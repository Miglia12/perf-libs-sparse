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
#include <string.h>

static void compute_dense_mm_c(const perflibs_singlecomplex_t *A,
                               perflibs_int_t m, perflibs_int_t k,
                               const perflibs_singlecomplex_t *B,
                               perflibs_int_t n, perflibs_singlecomplex_t alpha,
                               perflibs_singlecomplex_t beta,
                               const perflibs_singlecomplex_t *C0,
                               perflibs_singlecomplex_t *out) {
  if (C0 != NULL) {
    memcpy(out, C0, sizeof(*out) * (size_t)m * (size_t)n);
  }
  cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)n, (int)k,
              &alpha, A, (int)k, B, (int)n, &beta, out, (int)n);
}

static void csr_to_dense_c(perflibs_int_t m, perflibs_int_t n,
                           const perflibs_int_t *row_ptr,
                           const perflibs_int_t *col_indx,
                           const perflibs_singlecomplex_t *vals,
                           perflibs_singlecomplex_t *dense) {
  memset(dense, 0, sizeof(*dense) * (size_t)m * (size_t)n);
  for (perflibs_int_t i = 0; i < m; ++i) {
    for (perflibs_int_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
      dense[i * n + col_indx[p]] = vals[p];
    }
  }
}

static int test_dense_dense_gemm_path_c() {
  const perflibs_singlecomplex_t A_vals[] = {
      TEST_C(1.0f, 0.5f),  TEST_C(2.0f, -1.0f), TEST_C(0.0f, 0.0f),
      TEST_C(0.0f, 0.0f),  TEST_C(0.0f, 0.0f),  TEST_C(-1.0f, 0.25f),
      TEST_C(4.0f, -1.0f), TEST_C(0.0f, 0.0f),  TEST_C(3.0f, 2.0f),
      TEST_C(0.0f, 0.0f),  TEST_C(5.0f, -0.5f), TEST_C(6.0f, 1.0f),
      TEST_C(0.0f, 0.0f),  TEST_C(7.0f, 0.5f),  TEST_C(0.0f, 0.0f),
      TEST_C(8.0f, -2.0f)};
  const perflibs_singlecomplex_t B_vals[] = {
      TEST_C(2.0f, 0.0f), TEST_C(1.0f, -0.5f), TEST_C(0.0f, 0.0f),
      TEST_C(3.0f, 0.0f), TEST_C(-1.0f, 1.0f), TEST_C(3.0f, 0.0f),
      TEST_C(0.5f, 2.0f), TEST_C(2.0f, -1.0f), TEST_C(4.0f, 0.0f),
      TEST_C(0.0f, 0.0f), TEST_C(5.0f, 1.0f),  TEST_C(1.0f, 0.5f),
      TEST_C(0.0f, 0.0f), TEST_C(-2.0f, 0.0f), TEST_C(6.0f, -1.0f),
      TEST_C(7.0f, 2.0f)};
  const perflibs_singlecomplex_t C_init[] = {
      TEST_C(1.0f, 0.0f),  TEST_C(2.0f, 1.0f),   TEST_C(3.0f, -1.0f),
      TEST_C(4.0f, 0.0f),  TEST_C(5.0f, 0.5f),   TEST_C(6.0f, 0.0f),
      TEST_C(7.0f, -0.5f), TEST_C(8.0f, 1.0f),   TEST_C(9.0f, -1.0f),
      TEST_C(10.0f, 0.0f), TEST_C(11.0f, 1.0f),  TEST_C(12.0f, -1.0f),
      TEST_C(13.0f, 0.0f), TEST_C(14.0f, -0.5f), TEST_C(15.0f, 0.5f),
      TEST_C(16.0f, 0.0f)};
  const perflibs_singlecomplex_t alpha = TEST_C(1.5f, 0.25f);
  const perflibs_singlecomplex_t beta = TEST_C(0.5f, -0.5f);
  perflibs_singlecomplex_t expected[16];
  perflibs_spmat_t A = NULL;
  perflibs_spmat_t B = NULL;
  perflibs_spmat_t C = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  perflibs_singlecomplex_t *denseC = NULL;

  compute_dense_mm_c(A_vals, 4, 4, B_vals, 4, alpha, beta, C_init, expected);

  CHECK_STATUS(perflibs_spmat_create_dense_c(&A, PERFLIBS_ROW_MAJOR, 4, 4, 4,
                                             A_vals, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_c(&B, PERFLIBS_ROW_MAJOR, 4, 4, 4,
                                             B_vals, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_c(&C, PERFLIBS_ROW_MAJOR, 4, 4, 4,
                                             C_init, 0));
  CHECK_STATUS(perflibs_spmm_exec_c(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    PERFLIBS_SPARSE_OPERATION_NOTRANS, alpha, A,
                                    B, beta, C));
  CHECK_STATUS(
      perflibs_spmat_export_dense_c(C, PERFLIBS_ROW_MAJOR, &m, &n, &denseC));
  CHECK_INT_EQ(m, 4);
  CHECK_INT_EQ(n, 4);
  CHECK_SINGLECOMPLEX_ARRAY(denseC, expected, 16, 2e-5f);

  free(denseC);
  CHECK_STATUS(perflibs_spmat_destroy(A));
  CHECK_STATUS(perflibs_spmat_destroy(B));
  CHECK_STATUS(perflibs_spmat_destroy(C));
  return EXIT_SUCCESS;
}

static int test_dense_sparse_and_identity_paths_c() {
  const perflibs_singlecomplex_t denseA[] = {
      TEST_C(1.0f, 0.0f),  TEST_C(2.0f, 1.0f),  TEST_C(0.0f, 0.0f),
      TEST_C(1.0f, -0.5f), TEST_C(-1.0f, 0.5f), TEST_C(3.0f, 0.0f),
      TEST_C(4.0f, -1.0f), TEST_C(0.0f, 0.0f),  TEST_C(0.5f, 0.5f),
      TEST_C(0.0f, 0.0f),  TEST_C(2.0f, 0.0f),  TEST_C(1.5f, -0.5f),
      TEST_C(0.0f, 0.0f),  TEST_C(-2.0f, 1.0f), TEST_C(1.0f, 0.0f),
      TEST_C(3.0f, -1.0f)};
  const perflibs_int_t row_ptrB[] = {0, 2, 4, 6, 8};
  const perflibs_int_t col_indxB[] = {0, 3, 0, 1, 1, 2, 2, 3};
  const perflibs_singlecomplex_t valsB[] = {
      TEST_C(2.0f, 0.0f),  TEST_C(1.0f, 1.0f), TEST_C(-1.0f, 0.5f),
      TEST_C(3.0f, -1.0f), TEST_C(0.5f, 2.0f), TEST_C(4.0f, 0.0f),
      TEST_C(2.0f, -1.0f), TEST_C(-0.5f, 1.0f)};
  const perflibs_int_t row_ptrCsr[] = {0, 2, 4, 6, 8};
  const perflibs_int_t col_indxCsr[] = {0, 1, 1, 2, 2, 3, 0, 3};
  const perflibs_singlecomplex_t valsCsr[] = {
      TEST_C(1.0f, 1.0f),  TEST_C(2.0f, 0.0f),  TEST_C(4.0f, -1.0f),
      TEST_C(-1.0f, 0.5f), TEST_C(3.0f, 0.25f), TEST_C(5.0f, -0.5f),
      TEST_C(2.5f, 0.0f),  TEST_C(-2.0f, 1.0f)};
  const perflibs_singlecomplex_t alpha_identity_left = TEST_C(2.0f, 0.5f);
  const perflibs_singlecomplex_t alpha_identity_right = TEST_C(-1.5f, 0.25f);
  perflibs_singlecomplex_t denseB[16];
  perflibs_singlecomplex_t expected_dense_sparse[16];
  perflibs_singlecomplex_t expected_identity_left[16];
  perflibs_singlecomplex_t expected_identity_right[16];
  perflibs_spmat_t A_dense = NULL;
  perflibs_spmat_t B_sparse = NULL;
  perflibs_spmat_t C = NULL;
  perflibs_spmat_t identity = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  perflibs_singlecomplex_t *dense_out = NULL;

  csr_to_dense_c(4, 4, row_ptrB, col_indxB, valsB, denseB);
  compute_dense_mm_c(denseA, 4, 4, denseB, 4, TEST_C(1.0f, 0.0f),
                     TEST_C(0.0f, 0.0f), NULL, expected_dense_sparse);

  CHECK_STATUS(perflibs_spmat_create_dense_c(&A_dense, PERFLIBS_ROW_MAJOR, 4, 4,
                                             4, denseA, 0));
  CHECK_STATUS(perflibs_spmat_create_csr_c(&B_sparse, 4, 4, row_ptrB, col_indxB,
                                           valsB, 0));
  C = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(C != NULL, "dense-sparse output creation failed");
  CHECK_STATUS(perflibs_spmm_exec_c(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      TEST_C(1.0f, 0.0f), A_dense, B_sparse, TEST_C(0.0f, 0.0f), C));
  CHECK_STATUS(
      perflibs_spmat_export_dense_c(C, PERFLIBS_ROW_MAJOR, &m, &n, &dense_out));
  CHECK_SINGLECOMPLEX_ARRAY(dense_out, expected_dense_sparse, 16, 2e-5f);
  free(dense_out);
  CHECK_STATUS(perflibs_spmat_destroy(C));

  identity = perflibs_spmat_create_identity(4);
  C = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(identity != NULL && C != NULL, "identity-left creation failed");
  CHECK_STATUS(perflibs_spmm_exec_c(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      alpha_identity_left, identity, B_sparse, TEST_C(0.0f, 0.0f), C));
  CHECK_STATUS(
      perflibs_spmat_export_dense_c(C, PERFLIBS_ROW_MAJOR, &m, &n, &dense_out));
  CHECK_INT_EQ(m, 4);
  CHECK_INT_EQ(n, 4);
  for (perflibs_int_t i = 0; i < 16; ++i) {
    expected_identity_left[i] = test_cmulf(alpha_identity_left, denseB[i]);
  }
  CHECK_SINGLECOMPLEX_ARRAY(dense_out, expected_identity_left, 16, 2e-5f);
  free(dense_out);
  CHECK_STATUS(perflibs_spmat_destroy(C));
  CHECK_STATUS(perflibs_spmat_destroy(identity));

  CHECK_STATUS(perflibs_spmat_create_csr_c(&X, 4, 4, row_ptrCsr, col_indxCsr,
                                           valsCsr, 0));
  identity = perflibs_spmat_create_identity(4);
  C = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(identity != NULL && C != NULL, "identity-right creation failed");
  CHECK_STATUS(perflibs_spmm_exec_c(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      alpha_identity_right, X, identity, TEST_C(0.0f, 0.0f), C));
  CHECK_STATUS(
      perflibs_spmat_export_dense_c(C, PERFLIBS_ROW_MAJOR, &m, &n, &dense_out));
  for (perflibs_int_t i = 0; i < 16; ++i) {
    expected_identity_right[i] = TEST_C(0.0f, 0.0f);
  }
  expected_identity_right[0] =
      test_cmulf(alpha_identity_right, TEST_C(1.0f, 1.0f));
  expected_identity_right[1] =
      test_cmulf(alpha_identity_right, TEST_C(2.0f, 0.0f));
  expected_identity_right[5] =
      test_cmulf(alpha_identity_right, TEST_C(4.0f, -1.0f));
  expected_identity_right[6] =
      test_cmulf(alpha_identity_right, TEST_C(-1.0f, 0.5f));
  expected_identity_right[10] =
      test_cmulf(alpha_identity_right, TEST_C(3.0f, 0.25f));
  expected_identity_right[11] =
      test_cmulf(alpha_identity_right, TEST_C(5.0f, -0.5f));
  expected_identity_right[12] =
      test_cmulf(alpha_identity_right, TEST_C(2.5f, 0.0f));
  expected_identity_right[15] =
      test_cmulf(alpha_identity_right, TEST_C(-2.0f, 1.0f));
  CHECK_SINGLECOMPLEX_ARRAY(dense_out, expected_identity_right, 16, 2e-5f);

  free(dense_out);
  CHECK_STATUS(perflibs_spmat_destroy(A_dense));
  CHECK_STATUS(perflibs_spmat_destroy(B_sparse));
  CHECK_STATUS(perflibs_spmat_destroy(C));
  CHECK_STATUS(perflibs_spmat_destroy(identity));
  CHECK_STATUS(perflibs_spmat_destroy(X));
  return EXIT_SUCCESS;
}

int main() {
  if (test_dense_dense_gemm_path_c() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_dense_sparse_and_identity_paths_c() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
