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

static void compute_dense_mm_s(const float *A, perflibs_int_t m,
                               perflibs_int_t k, const float *B,
                               perflibs_int_t n, float alpha, float beta,
                               const float *C0, float *out) {
  if (C0 != NULL) {
    memcpy(out, C0, sizeof(*out) * (size_t)m * (size_t)n);
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)n, (int)k,
              alpha, A, (int)k, B, (int)n, beta, out, (int)n);
}

static void csr_to_dense_s(perflibs_int_t m, perflibs_int_t n,
                           const perflibs_int_t *row_ptr,
                           const perflibs_int_t *col_indx, const float *vals,
                           float *dense) {
  memset(dense, 0, sizeof(*dense) * (size_t)m * (size_t)n);
  for (perflibs_int_t i = 0; i < m; ++i) {
    for (perflibs_int_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
      dense[i * n + col_indx[p]] = vals[p];
    }
  }
}

static int test_dense_dense_gemm_path() {
  const float A_vals[] = {1.0f, 2.0f, 0.0f, 0.0f, 0.0f, -1.0f, 4.0f, 0.0f,
                          3.0f, 0.0f, 5.0f, 6.0f, 0.0f, 7.0f,  0.0f, 8.0f};
  const float B_vals[] = {2.0f, 1.0f, 0.0f, 3.0f, -1.0f, 3.0f,  0.5f, 2.0f,
                          4.0f, 0.0f, 5.0f, 1.0f, 0.0f,  -2.0f, 6.0f, 7.0f};
  const float C_init[] = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                          7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                          13.0f, 14.0f, 15.0f, 16.0f};
  float expected[16];
  perflibs_spmat_t A = NULL;
  perflibs_spmat_t B = NULL;
  perflibs_spmat_t C = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  float *denseC = NULL;

  compute_dense_mm_s(A_vals, 4, 4, B_vals, 4, 2.0f, 0.5f, C_init, expected);

  CHECK_STATUS(perflibs_spmat_create_dense_s(&A, PERFLIBS_ROW_MAJOR, 4, 4, 4,
                                             A_vals, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_s(&B, PERFLIBS_ROW_MAJOR, 4, 4, 4,
                                             B_vals, 0));
  CHECK_STATUS(perflibs_spmat_create_dense_s(&C, PERFLIBS_ROW_MAJOR, 4, 4, 4,
                                             C_init, 0));
  CHECK_STATUS(perflibs_spmm_exec_s(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    PERFLIBS_SPARSE_OPERATION_NOTRANS, 2.0f, A,
                                    B, 0.5f, C));
  CHECK_STATUS(
      perflibs_spmat_export_dense_s(C, PERFLIBS_ROW_MAJOR, &m, &n, &denseC));
  CHECK_INT_EQ(m, 4);
  CHECK_INT_EQ(n, 4);
  CHECK_FLOAT_ARRAY(denseC, expected, 16, 1e-5f);

  free(denseC);
  CHECK_STATUS(perflibs_spmat_destroy(A));
  CHECK_STATUS(perflibs_spmat_destroy(B));
  CHECK_STATUS(perflibs_spmat_destroy(C));
  return EXIT_SUCCESS;
}

static int test_dense_sparse_and_identity_paths() {
  const float denseA[] = {1.0f, 2.0f, 0.0f, 1.0f, -1.0f, 3.0f,  4.0f, 0.0f,
                          0.5f, 0.0f, 2.0f, 1.5f, 0.0f,  -2.0f, 1.0f, 3.0f};
  const perflibs_int_t row_ptrB[] = {0, 2, 4, 6, 8};
  const perflibs_int_t col_indxB[] = {0, 3, 0, 1, 1, 2, 2, 3};
  const float valsB[] = {2.0f, 1.0f, 1.0f, -1.0f, 3.0f, 0.5f, 4.0f, 2.0f};
  const perflibs_int_t row_ptrCsr[] = {0, 2, 4, 6, 8};
  const perflibs_int_t col_indxCsr[] = {0, 1, 1, 2, 2, 3, 0, 3};
  const float valsCsr[] = {1.0f, 2.0f, 4.0f, -1.0f, 3.0f, 5.0f, 2.5f, -2.0f};
  float denseB[16];
  float expected_dense_sparse[16];
  float expected_identity_left[16];
  float expected_identity_right[16];
  perflibs_spmat_t A_dense = NULL;
  perflibs_spmat_t B_sparse = NULL;
  perflibs_spmat_t C = NULL;
  perflibs_spmat_t identity = NULL;
  perflibs_spmat_t X = NULL;
  perflibs_int_t m = -1;
  perflibs_int_t n = -1;
  float *dense_out = NULL;

  csr_to_dense_s(4, 4, row_ptrB, col_indxB, valsB, denseB);
  compute_dense_mm_s(denseA, 4, 4, denseB, 4, 1.0f, 0.0f, NULL,
                     expected_dense_sparse);

  CHECK_STATUS(perflibs_spmat_create_dense_s(&A_dense, PERFLIBS_ROW_MAJOR, 4, 4,
                                             4, denseA, 0));
  CHECK_STATUS(perflibs_spmat_create_csr_s(&B_sparse, 4, 4, row_ptrB, col_indxB,
                                           valsB, 0));
  C = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(C != NULL, "dense-sparse output creation failed");
  CHECK_STATUS(perflibs_spmm_exec_s(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    PERFLIBS_SPARSE_OPERATION_NOTRANS, 1.0f,
                                    A_dense, B_sparse, 0.0f, C));
  CHECK_STATUS(
      perflibs_spmat_export_dense_s(C, PERFLIBS_ROW_MAJOR, &m, &n, &dense_out));
  CHECK_FLOAT_ARRAY(dense_out, expected_dense_sparse, 16, 1e-5f);
  free(dense_out);
  CHECK_STATUS(perflibs_spmat_destroy(C));

  identity = perflibs_spmat_create_identity(4);
  C = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(identity != NULL && C != NULL, "identity-left creation failed");
  CHECK_STATUS(perflibs_spmm_exec_s(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    PERFLIBS_SPARSE_OPERATION_NOTRANS, 2.0f,
                                    identity, B_sparse, 0.0f, C));
  CHECK_STATUS(
      perflibs_spmat_export_dense_s(C, PERFLIBS_ROW_MAJOR, &m, &n, &dense_out));
  CHECK_INT_EQ(m, 4);
  CHECK_INT_EQ(n, 4);
  for (perflibs_int_t i = 0; i < 16; ++i) {
    expected_identity_left[i] = 2.0f * denseB[i];
  }
  CHECK_FLOAT_ARRAY(dense_out, expected_identity_left, 16, 1e-5f);
  free(dense_out);
  CHECK_STATUS(perflibs_spmat_destroy(C));
  CHECK_STATUS(perflibs_spmat_destroy(identity));

  CHECK_STATUS(perflibs_spmat_create_csr_s(&X, 4, 4, row_ptrCsr, col_indxCsr,
                                           valsCsr, 0));
  identity = perflibs_spmat_create_identity(4);
  C = perflibs_spmat_create_null(4, 4);
  CHECK_TRUE(identity != NULL && C != NULL, "identity-right creation failed");
  CHECK_STATUS(perflibs_spmm_exec_s(PERFLIBS_SPARSE_OPERATION_NOTRANS,
                                    PERFLIBS_SPARSE_OPERATION_NOTRANS, -1.5f, X,
                                    identity, 0.0f, C));
  CHECK_STATUS(
      perflibs_spmat_export_dense_s(C, PERFLIBS_ROW_MAJOR, &m, &n, &dense_out));
  for (perflibs_int_t i = 0; i < 16; ++i) {
    expected_identity_right[i] = 0.0f;
  }
  expected_identity_right[0] = -1.5f;
  expected_identity_right[1] = -3.0f;
  expected_identity_right[5] = -6.0f;
  expected_identity_right[6] = 1.5f;
  expected_identity_right[10] = -4.5f;
  expected_identity_right[11] = -7.5f;
  expected_identity_right[12] = -3.75f;
  expected_identity_right[15] = 3.0f;
  CHECK_FLOAT_ARRAY(dense_out, expected_identity_right, 16, 1e-5f);

  free(dense_out);
  CHECK_STATUS(perflibs_spmat_destroy(A_dense));
  CHECK_STATUS(perflibs_spmat_destroy(B_sparse));
  CHECK_STATUS(perflibs_spmat_destroy(C));
  CHECK_STATUS(perflibs_spmat_destroy(identity));
  CHECK_STATUS(perflibs_spmat_destroy(X));
  return EXIT_SUCCESS;
}

int main() {
  if (test_dense_dense_gemm_path() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_dense_sparse_and_identity_paths() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
