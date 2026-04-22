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

static void compute_dense_mm_z(const perflibs_doublecomplex_t *A,
                               perflibs_int_t m, perflibs_int_t k,
                               const perflibs_doublecomplex_t *B,
                               perflibs_int_t n, perflibs_doublecomplex_t alpha,
                               perflibs_doublecomplex_t beta,
                               const perflibs_doublecomplex_t *C0,
                               perflibs_doublecomplex_t *out) {
  if (C0 != NULL) {
    memcpy(out, C0, sizeof(*out) * (size_t)m * (size_t)n);
  }
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)n, (int)k,
              &alpha, A, (int)k, B, (int)n, &beta, out, (int)n);
}

static void csr_to_dense_z(perflibs_int_t m, perflibs_int_t n,
                           const perflibs_int_t *row_ptr,
                           const perflibs_int_t *col_indx,
                           const perflibs_doublecomplex_t *vals,
                           perflibs_doublecomplex_t *dense) {
  memset(dense, 0, sizeof(*dense) * (size_t)m * (size_t)n);
  for (perflibs_int_t i = 0; i < m; ++i) {
    for (perflibs_int_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
      dense[i * n + col_indx[p]] = vals[p];
    }
  }
}

static int test_dense_dense_gemm_path_z() {
  const perflibs_doublecomplex_t A_vals[] = {
      TEST_Z(1.0, 0.5),  TEST_Z(2.0, -1.0), TEST_Z(0.0, 0.0),
      TEST_Z(0.0, 0.0),  TEST_Z(0.0, 0.0),  TEST_Z(-1.0, 0.25),
      TEST_Z(4.0, -1.0), TEST_Z(0.0, 0.0),  TEST_Z(3.0, 2.0),
      TEST_Z(0.0, 0.0),  TEST_Z(5.0, -0.5), TEST_Z(6.0, 1.0),
      TEST_Z(0.0, 0.0),  TEST_Z(7.0, 0.5),  TEST_Z(0.0, 0.0),
      TEST_Z(8.0, -2.0)};
  const perflibs_doublecomplex_t B_vals[] = {
      TEST_Z(2.0, 0.0), TEST_Z(1.0, -0.5), TEST_Z(0.0, 0.0),
      TEST_Z(3.0, 0.0), TEST_Z(-1.0, 1.0), TEST_Z(3.0, 0.0),
      TEST_Z(0.5, 2.0), TEST_Z(2.0, -1.0), TEST_Z(4.0, 0.0),
      TEST_Z(0.0, 0.0), TEST_Z(5.0, 1.0),  TEST_Z(1.0, 0.5),
      TEST_Z(0.0, 0.0), TEST_Z(-2.0, 0.0), TEST_Z(6.0, -1.0),
      TEST_Z(7.0, 2.0)};
  const perflibs_doublecomplex_t C_init[] = {
      TEST_Z(1.0, 0.0),  TEST_Z(2.0, 1.0),   TEST_Z(3.0, -1.0),
      TEST_Z(4.0, 0.0),  TEST_Z(5.0, 0.5),   TEST_Z(6.0, 0.0),
      TEST_Z(7.0, -0.5), TEST_Z(8.0, 1.0),   TEST_Z(9.0, -1.0),
      TEST_Z(10.0, 0.0), TEST_Z(11.0, 1.0),  TEST_Z(12.0, -1.0),
      TEST_Z(13.0, 0.0), TEST_Z(14.0, -0.5), TEST_Z(15.0, 0.5),
      TEST_Z(16.0, 0.0)};
  const perflibs_doublecomplex_t alpha = TEST_Z(1.5, 0.25);
  const perflibs_doublecomplex_t beta = TEST_Z(0.5, -0.5);
  perflibs_doublecomplex_t expected[16];
  perflibs_spmat_t A = NULL;
  perflibs_spmat_t B = NULL;
  perflibs_spmat_t C = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  perflibs_doublecomplex_t *denseC = NULL;

  compute_dense_mm_z(A_vals, 4, 4, B_vals, 4, alpha, beta, C_init, expected);

  CHECK_STATUS(perflibs_spmat_create_dense_z(&A, PERFLIBS_ROW_MAJOR, 4, 4, 4,
                                             A_vals, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_z(&B, PERFLIBS_ROW_MAJOR, 4, 4, 4,
                                             B_vals, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_z(&C, PERFLIBS_ROW_MAJOR, 4, 4, 4,
                                             C_init, 0));
  CHECK_STATUS(perflibs_spmm_exec_z(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    PERFLIBS_SPARSE_OPERATION_NOTRANS, alpha, A,
                                    B, beta, C));
  CHECK_STATUS(
      perflibs_spmat_export_dense_z(C, PERFLIBS_ROW_MAJOR, &m, &n, &denseC));
  CHECK_INT_EQ(m, 4);
  CHECK_INT_EQ(n, 4);
  CHECK_DOUBLECOMPLEX_ARRAY(denseC, expected, 16, 1e-12);

  free(denseC);
  CHECK_STATUS(perflibs_spmat_destroy(A));
  CHECK_STATUS(perflibs_spmat_destroy(B));
  CHECK_STATUS(perflibs_spmat_destroy(C));
  return EXIT_SUCCESS;
}

static int test_dense_sparse_and_identity_paths_z() {
  const perflibs_doublecomplex_t denseA[] = {
      TEST_Z(1.0, 0.0),  TEST_Z(2.0, 1.0),  TEST_Z(0.0, 0.0),
      TEST_Z(1.0, -0.5), TEST_Z(-1.0, 0.5), TEST_Z(3.0, 0.0),
      TEST_Z(4.0, -1.0), TEST_Z(0.0, 0.0),  TEST_Z(0.5, 0.5),
      TEST_Z(0.0, 0.0),  TEST_Z(2.0, 0.0),  TEST_Z(1.5, -0.5),
      TEST_Z(0.0, 0.0),  TEST_Z(-2.0, 1.0), TEST_Z(1.0, 0.0),
      TEST_Z(3.0, -1.0)};
  const perflibs_int_t row_ptrB[] = {0, 2, 4, 6, 8};
  const perflibs_int_t col_indxB[] = {0, 3, 0, 1, 1, 2, 2, 3};
  const perflibs_doublecomplex_t valsB[] = {
      TEST_Z(2.0, 0.0), TEST_Z(1.0, 1.0), TEST_Z(-1.0, 0.5), TEST_Z(3.0, -1.0),
      TEST_Z(0.5, 2.0), TEST_Z(4.0, 0.0), TEST_Z(2.0, -1.0), TEST_Z(-0.5, 1.0)};
  const perflibs_int_t row_ptrCsr[] = {0, 2, 4, 6, 8};
  const perflibs_int_t col_indxCsr[] = {0, 1, 1, 2, 2, 3, 0, 3};
  const perflibs_doublecomplex_t valsCsr[] = {
      TEST_Z(1.0, 1.0),  TEST_Z(2.0, 0.0),  TEST_Z(4.0, -1.0),
      TEST_Z(-1.0, 0.5), TEST_Z(3.0, 0.25), TEST_Z(5.0, -0.5),
      TEST_Z(2.5, 0.0),  TEST_Z(-2.0, 1.0)};
  const perflibs_doublecomplex_t alpha_identity_left = TEST_Z(2.0, 0.5);
  const perflibs_doublecomplex_t alpha_identity_right = TEST_Z(-1.5, 0.25);
  perflibs_doublecomplex_t denseB[16];
  perflibs_doublecomplex_t expected_dense_sparse[16];
  perflibs_doublecomplex_t expected_identity_left[16];
  perflibs_doublecomplex_t expected_identity_right[16];
  perflibs_spmat_t A_dense = NULL;
  perflibs_spmat_t B_sparse = NULL;
  perflibs_spmat_t C = NULL;
  perflibs_spmat_t identity = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  perflibs_doublecomplex_t *dense_out = NULL;

  csr_to_dense_z(4, 4, row_ptrB, col_indxB, valsB, denseB);
  compute_dense_mm_z(denseA, 4, 4, denseB, 4, TEST_Z(1.0, 0.0),
                     TEST_Z(0.0, 0.0), NULL, expected_dense_sparse);

  CHECK_STATUS(perflibs_spmat_create_dense_z(&A_dense, PERFLIBS_ROW_MAJOR, 4, 4,
                                             4, denseA, 0));
  CHECK_STATUS(perflibs_spmat_create_csr_z(&B_sparse, 4, 4, row_ptrB, col_indxB,
                                           valsB, 0));
  C = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(C != NULL, "dense-sparse output creation failed");
  CHECK_STATUS(perflibs_spmm_exec_z(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      TEST_Z(1.0, 0.0), A_dense, B_sparse, TEST_Z(0.0, 0.0), C));
  CHECK_STATUS(
      perflibs_spmat_export_dense_z(C, PERFLIBS_ROW_MAJOR, &m, &n, &dense_out));
  CHECK_DOUBLECOMPLEX_ARRAY(dense_out, expected_dense_sparse, 16, 1e-12);
  free(dense_out);
  CHECK_STATUS(perflibs_spmat_destroy(C));

  identity = perflibs_spmat_create_identity(4);
  C = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(identity != NULL && C != NULL, "identity-left creation failed");
  CHECK_STATUS(perflibs_spmm_exec_z(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      alpha_identity_left, identity, B_sparse, TEST_Z(0.0, 0.0), C));
  CHECK_STATUS(
      perflibs_spmat_export_dense_z(C, PERFLIBS_ROW_MAJOR, &m, &n, &dense_out));
  CHECK_INT_EQ(m, 4);
  CHECK_INT_EQ(n, 4);
  for (perflibs_int_t i = 0; i < 16; ++i) {
    expected_identity_left[i] = test_cmul(alpha_identity_left, denseB[i]);
  }
  CHECK_DOUBLECOMPLEX_ARRAY(dense_out, expected_identity_left, 16, 1e-12);
  free(dense_out);
  CHECK_STATUS(perflibs_spmat_destroy(C));
  CHECK_STATUS(perflibs_spmat_destroy(identity));

  CHECK_STATUS(perflibs_spmat_create_csr_z(&X, 4, 4, row_ptrCsr, col_indxCsr,
                                           valsCsr, 0));
  identity = perflibs_spmat_create_identity(4);
  C = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(identity != NULL && C != NULL, "identity-right creation failed");
  CHECK_STATUS(perflibs_spmm_exec_z(
      PERFLIBS_SPARSE_OPERATION_NOTRANS, PERFLIBS_SPARSE_OPERATION_NOTRANS,
      alpha_identity_right, X, identity, TEST_Z(0.0, 0.0), C));
  CHECK_STATUS(
      perflibs_spmat_export_dense_z(C, PERFLIBS_ROW_MAJOR, &m, &n, &dense_out));
  for (perflibs_int_t i = 0; i < 16; ++i) {
    expected_identity_right[i] = TEST_Z(0.0, 0.0);
  }
  expected_identity_right[0] =
      test_cmul(alpha_identity_right, TEST_Z(1.0, 1.0));
  expected_identity_right[1] =
      test_cmul(alpha_identity_right, TEST_Z(2.0, 0.0));
  expected_identity_right[5] =
      test_cmul(alpha_identity_right, TEST_Z(4.0, -1.0));
  expected_identity_right[6] =
      test_cmul(alpha_identity_right, TEST_Z(-1.0, 0.5));
  expected_identity_right[10] =
      test_cmul(alpha_identity_right, TEST_Z(3.0, 0.25));
  expected_identity_right[11] =
      test_cmul(alpha_identity_right, TEST_Z(5.0, -0.5));
  expected_identity_right[12] =
      test_cmul(alpha_identity_right, TEST_Z(2.5, 0.0));
  expected_identity_right[15] =
      test_cmul(alpha_identity_right, TEST_Z(-2.0, 1.0));
  CHECK_DOUBLECOMPLEX_ARRAY(dense_out, expected_identity_right, 16, 1e-12);

  free(dense_out);
  CHECK_STATUS(perflibs_spmat_destroy(A_dense));
  CHECK_STATUS(perflibs_spmat_destroy(B_sparse));
  CHECK_STATUS(perflibs_spmat_destroy(C));
  CHECK_STATUS(perflibs_spmat_destroy(identity));
  CHECK_STATUS(perflibs_spmat_destroy(X));
  return EXIT_SUCCESS;
}

int main() {
  if (test_dense_dense_gemm_path_z() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_dense_sparse_and_identity_paths_z() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
