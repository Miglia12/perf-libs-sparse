/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <stdlib.h>

static void build_irregular_csr_structure(perflibs_int_t m, perflibs_int_t n,
                                          perflibs_int_t *row_ptr,
                                          perflibs_int_t *col_indx) {
  row_ptr[0] = 0;
  for (perflibs_int_t i = 0; i < m; ++i) {
    perflibs_int_t len;
    switch (i % 8) {
    case 0:
      len = 7;
      break;
    case 1:
      len = 113;
      break;
    case 2:
      len = 19;
      break;
    case 3:
      len = 181;
      break;
    case 4:
      len = 5;
      break;
    case 5:
      len = 149;
      break;
    case 6:
      len = 37;
      break;
    default:
      len = 223;
      break;
    }
    row_ptr[i + 1] = row_ptr[i] + len;

    if (col_indx != NULL) {
      const perflibs_int_t start = (29 * i + 17) % (n - len);
      for (perflibs_int_t j = 0; j < len; ++j) {
        col_indx[row_ptr[i] + j] = start + j;
      }
    }
  }
}

static int test_float_irregular_spmv_optimized() {
  const perflibs_int_t m = 1536;
  const perflibs_int_t n = 12288;
  const float alpha = 1.75f;
  const float beta = -0.5f;

  perflibs_int_t *row_ptr =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * (m + 1));
  perflibs_int_t *col_indx = NULL;
  float *vals = NULL;
  float *x = NULL;
  float *y = NULL;
  float *y_expected = NULL;
  perflibs_spmat_t A = NULL;

  CHECK_TRUE(row_ptr, "allocation failure for CSR SpMV OpenMP float test");

  build_irregular_csr_structure(m, n, row_ptr, NULL);
  col_indx = (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * row_ptr[m]);
  vals = (float *)malloc(sizeof(float) * row_ptr[m]);
  x = (float *)malloc(sizeof(float) * n);
  y = (float *)malloc(sizeof(float) * m);
  y_expected = (float *)malloc(sizeof(float) * m);
  CHECK_TRUE(col_indx && vals && x && y && y_expected,
             "allocation failure for CSR SpMV OpenMP float buffers");

  build_irregular_csr_structure(m, n, row_ptr, col_indx);

  for (perflibs_int_t i = 0; i < row_ptr[m]; ++i) {
    vals[i] = 0.125f * (float)((i % 13) + 1);
  }
  for (perflibs_int_t i = 0; i < n; ++i) {
    x[i] = ((float)((i % 11) - 5)) * 0.5f;
  }
  for (perflibs_int_t i = 0; i < m; ++i) {
    y[i] = (float)((i % 9) - 4);
    y_expected[i] = y[i];
  }

  for (perflibs_int_t i = 0; i < m; ++i) {
    double sum = 0.0;
    for (perflibs_int_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
      sum += (double)vals[p] * (double)x[col_indx[p]];
    }
    y_expected[i] = beta * y_expected[i] + alpha * (float)sum;
  }

  CHECK_STATUS(
      perflibs_spmat_create_csr_s(&A, m, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_MANY));
  CHECK_STATUS(perflibs_spmv_optimize(A));

  CHECK_STATUS(perflibs_spmv_exec_s(PERFLIBS_SPARSE_OPERATION_NOTRANS, alpha, A,
                                    x, beta, y));
  for (perflibs_int_t i = 0; i < m; ++i) {
    CHECK_DOUBLE_NEAR(y[i], y_expected[i], 2e-4);
  }

  CHECK_STATUS(perflibs_spmat_destroy(A));
  free(row_ptr);
  free(col_indx);
  free(vals);
  free(x);
  free(y);
  free(y_expected);
  return EXIT_SUCCESS;
}

static int test_complex_irregular_spmv_conjnotrans_optimized() {
  const perflibs_int_t m = 1024;
  const perflibs_int_t n = 8192;
  const perflibs_singlecomplex_t alpha = TEST_C(0.75f, -0.5f);
  const perflibs_singlecomplex_t beta = TEST_C(-0.25f, 0.125f);

  perflibs_int_t *row_ptr =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * (m + 1));
  perflibs_int_t *col_indx = NULL;
  perflibs_singlecomplex_t *vals = NULL;
  perflibs_singlecomplex_t *x = NULL;
  perflibs_singlecomplex_t *y = NULL;
  perflibs_singlecomplex_t *y_expected = NULL;
  perflibs_spmat_t A = NULL;

  CHECK_TRUE(row_ptr, "allocation failure for CSR SpMV OpenMP complex test");

  build_irregular_csr_structure(m, n, row_ptr, NULL);
  col_indx = (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * row_ptr[m]);
  vals = (perflibs_singlecomplex_t *)malloc(sizeof(perflibs_singlecomplex_t) *
                                            row_ptr[m]);
  x = (perflibs_singlecomplex_t *)malloc(sizeof(perflibs_singlecomplex_t) * n);
  y = (perflibs_singlecomplex_t *)malloc(sizeof(perflibs_singlecomplex_t) * n);
  y_expected =
      (perflibs_singlecomplex_t *)malloc(sizeof(perflibs_singlecomplex_t) * n);
  CHECK_TRUE(col_indx && vals && x && y && y_expected,
             "allocation failure for CSR SpMV OpenMP complex buffers");

  build_irregular_csr_structure(m, n, row_ptr, col_indx);

  for (perflibs_int_t i = 0; i < row_ptr[m]; ++i) {
    vals[i] = TEST_C(0.1f * (float)((i % 7) + 1), 0.05f * (float)((i % 5) - 2));
  }
  for (perflibs_int_t i = 0; i < n; ++i) {
    x[i] = TEST_C(0.25f * (float)((i % 9) - 4), 0.125f * (float)((i % 7) - 3));
  }
  for (perflibs_int_t i = 0; i < n; ++i) {
    y[i] = TEST_C(0.2f * (float)((i % 6) - 2), -0.1f * (float)((i % 5) - 2));
    y_expected[i] = test_cmulf(beta, y[i]);
  }

  for (perflibs_int_t i = 0; i < m; ++i) {
    for (perflibs_int_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
      perflibs_singlecomplex_t update =
          test_cmulf(alpha, test_cmulf(test_conjf(vals[p]), x[i]));
      y_expected[col_indx[p]] = test_caddf(y_expected[col_indx[p]], update);
    }
  }

  CHECK_STATUS(
      perflibs_spmat_create_csr_c(&A, m, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_CONJTRANS));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPMV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_MANY));
  CHECK_STATUS(perflibs_spmv_optimize(A));

  CHECK_STATUS(perflibs_spmv_exec_c(PERFLIBS_SPARSE_OPERATION_CONJTRANS, alpha,
                                    A, x, beta, y));
  CHECK_SINGLECOMPLEX_ARRAY(y, y_expected, n, 3e-4f);

  CHECK_STATUS(perflibs_spmat_destroy(A));
  free(row_ptr);
  free(col_indx);
  free(vals);
  free(x);
  free(y);
  free(y_expected);
  return EXIT_SUCCESS;
}

int main() {
  if (test_float_irregular_spmv_optimized() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_complex_irregular_spmv_conjnotrans_optimized() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
