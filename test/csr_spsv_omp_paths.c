/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "perflibs_sparse.h"
#include "test_utils.h"

#include <stdlib.h>

static void build_lower_triangular_blocks(perflibs_int_t blocks,
                                          perflibs_int_t block_size,
                                          perflibs_int_t *row_ptr,
                                          perflibs_int_t *col_indx,
                                          double *vals, double *rhs) {
  const perflibs_int_t n = blocks * block_size;
  perflibs_int_t nnz = 0;
  row_ptr[0] = 0;

  for (perflibs_int_t i = 0; i < n; ++i) {
    const perflibs_int_t local = i % block_size;
    const double diag = 2.0 + 0.05 * (double)(local % 7);
    double row_sum = diag;

    col_indx[nnz] = i;
    vals[nnz++] = diag;

    if (local >= 1) {
      col_indx[nnz] = i - 1;
      vals[nnz++] = 0.25;
      row_sum += 0.25;
    }
    if (local >= 2) {
      col_indx[nnz] = i - 2;
      vals[nnz++] = -0.125;
      row_sum += -0.125;
    }

    row_ptr[i + 1] = nnz;
    rhs[i] = row_sum;
  }
}

static void build_upper_triangular_blocks(perflibs_int_t blocks,
                                          perflibs_int_t block_size,
                                          perflibs_int_t *row_ptr,
                                          perflibs_int_t *col_indx,
                                          double *vals, double *rhs) {
  const perflibs_int_t n = blocks * block_size;
  perflibs_int_t nnz = 0;
  row_ptr[0] = 0;

  for (perflibs_int_t i = 0; i < n; ++i) {
    const perflibs_int_t local = i % block_size;
    const double diag = 1.5 + 0.05 * (double)(local % 5);
    double row_sum = diag;

    col_indx[nnz] = i;
    vals[nnz++] = diag;

    if (local + 1 < block_size) {
      col_indx[nnz] = i + 1;
      vals[nnz++] = -0.2;
      row_sum += -0.2;
    }
    if (local + 2 < block_size) {
      col_indx[nnz] = i + 2;
      vals[nnz++] = 0.1;
      row_sum += 0.1;
    }

    row_ptr[i + 1] = nnz;
    rhs[i] = row_sum;
  }
}

static void build_complex_lower_triangular_blocks(
    perflibs_int_t blocks, perflibs_int_t block_size, perflibs_int_t *row_ptr,
    perflibs_int_t *col_indx, perflibs_singlecomplex_t *vals,
    perflibs_singlecomplex_t *rhs) {
  const perflibs_int_t n = blocks * block_size;
  perflibs_int_t nnz = 0;
  row_ptr[0] = 0;

  for (perflibs_int_t i = 0; i < n; ++i) {
    rhs[i] = TEST_C(0.0f, 0.0f);
  }

  for (perflibs_int_t i = 0; i < n; ++i) {
    const perflibs_int_t local = i % block_size;
    const perflibs_singlecomplex_t diag =
        TEST_C(2.0f + 0.05f * (float)(local % 7), 0.1f * (float)(local % 3));

    col_indx[nnz] = i;
    vals[nnz++] = diag;
    rhs[i] = test_caddf(rhs[i], test_conjf(diag));

    if (local >= 1) {
      const perflibs_singlecomplex_t v1 = TEST_C(0.25f, -0.1f);
      col_indx[nnz] = i - 1;
      vals[nnz++] = v1;
      rhs[i - 1] = test_caddf(rhs[i - 1], test_conjf(v1));
    }
    if (local >= 2) {
      const perflibs_singlecomplex_t v2 = TEST_C(-0.125f, 0.05f);
      col_indx[nnz] = i - 2;
      vals[nnz++] = v2;
      rhs[i - 2] = test_caddf(rhs[i - 2], test_conjf(v2));
    }

    row_ptr[i + 1] = nnz;
  }
}

static void build_lower_triangular_chain(perflibs_int_t n,
                                         perflibs_int_t *row_ptr,
                                         perflibs_int_t *col_indx, double *vals,
                                         double *rhs) {
  perflibs_int_t nnz = 0;
  row_ptr[0] = 0;

  for (perflibs_int_t i = 0; i < n; ++i) {
    double row_sum = 0.0;

    if (i > 0) {
      const double offdiag = 0.2 + 0.01 * (double)(i % 5);
      col_indx[nnz] = i - 1;
      vals[nnz++] = offdiag;
      row_sum += offdiag;
    }

    col_indx[nnz] = i;
    vals[nnz++] = 1.75 + 0.02 * (double)(i % 7);
    row_sum += vals[nnz - 1];

    row_ptr[i + 1] = nnz;
    rhs[i] = row_sum;
  }
}

static void build_upper_triangular_chain(perflibs_int_t n,
                                         perflibs_int_t *row_ptr,
                                         perflibs_int_t *col_indx, double *vals,
                                         double *rhs) {
  perflibs_int_t nnz = 0;
  row_ptr[0] = 0;

  for (perflibs_int_t i = 0; i < n; ++i) {
    double row_sum = 0.0;

    col_indx[nnz] = i;
    vals[nnz++] = 1.5 + 0.03 * (double)(i % 5);
    row_sum += vals[nnz - 1];

    if (i + 1 < n) {
      const double offdiag = -0.15 + 0.01 * (double)(i % 3);
      col_indx[nnz] = i + 1;
      vals[nnz++] = offdiag;
      row_sum += offdiag;
    }

    row_ptr[i + 1] = nnz;
    rhs[i] = row_sum;
  }
}

static void build_complex_lower_triangular_chain(
    perflibs_int_t n, perflibs_int_t *row_ptr, perflibs_int_t *col_indx,
    perflibs_singlecomplex_t *vals, perflibs_singlecomplex_t *rhs) {
  perflibs_int_t nnz = 0;
  row_ptr[0] = 0;

  for (perflibs_int_t i = 0; i < n; ++i) {
    rhs[i] = TEST_C(0.0f, 0.0f);
  }

  for (perflibs_int_t i = 0; i < n; ++i) {
    if (i > 0) {
      const perflibs_singlecomplex_t offdiag =
          TEST_C(0.2f + 0.01f * (float)(i % 5), -0.04f);
      col_indx[nnz] = i - 1;
      vals[nnz++] = offdiag;
      rhs[i - 1] = test_caddf(rhs[i - 1], test_conjf(offdiag));
    }

    const perflibs_singlecomplex_t diag =
        TEST_C(1.8f + 0.02f * (float)(i % 7), 0.05f * (float)(i % 3));
    col_indx[nnz] = i;
    vals[nnz++] = diag;
    rhs[i] = test_caddf(rhs[i], test_conjf(diag));

    row_ptr[i + 1] = nnz;
  }
}

static int test_lower_triangular_parallel_path() {
  const perflibs_int_t blocks = 4;
  const perflibs_int_t block_size = 192;
  const perflibs_int_t n = blocks * block_size;
  const perflibs_int_t nnz = blocks * (3 * block_size - 3);

  perflibs_int_t *row_ptr =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * (n + 1));
  perflibs_int_t *col_indx =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * nnz);
  double *vals = (double *)malloc(sizeof(double) * nnz);
  double *rhs = (double *)malloc(sizeof(double) * n);
  double *x = (double *)malloc(sizeof(double) * n);
  perflibs_spmat_t A = NULL;

  CHECK_TRUE(row_ptr && col_indx && vals && rhs && x,
             "allocation failure in lower CSR SpSV OpenMP test");

  build_lower_triangular_blocks(blocks, block_size, row_ptr, col_indx, vals,
                                rhs);

  CHECK_STATUS(
      perflibs_spmat_create_csr_d(&A, n, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPSV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPSV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_MANY));
  CHECK_STATUS(perflibs_spsv_optimize(A));

  for (perflibs_int_t i = 0; i < n; ++i) {
    x[i] = 0.0;
  }
  CHECK_STATUS(
      perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, x, 1.0, rhs));

  for (perflibs_int_t i = 0; i < n; ++i) {
    CHECK_DOUBLE_NEAR(x[i], 1.0, 1e-12);
  }

  CHECK_STATUS(perflibs_spmat_destroy(A));
  free(row_ptr);
  free(col_indx);
  free(vals);
  free(rhs);
  free(x);
  return EXIT_SUCCESS;
}

static int test_upper_triangular_parallel_path() {
  const perflibs_int_t blocks = 4;
  const perflibs_int_t block_size = 192;
  const perflibs_int_t n = blocks * block_size;
  const perflibs_int_t nnz = blocks * (3 * block_size - 3);

  perflibs_int_t *row_ptr =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * (n + 1));
  perflibs_int_t *col_indx =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * nnz);
  double *vals = (double *)malloc(sizeof(double) * nnz);
  double *rhs = (double *)malloc(sizeof(double) * n);
  double *x = (double *)malloc(sizeof(double) * n);
  perflibs_spmat_t A = NULL;

  CHECK_TRUE(row_ptr && col_indx && vals && rhs && x,
             "allocation failure in upper CSR SpSV OpenMP test");

  build_upper_triangular_blocks(blocks, block_size, row_ptr, col_indx, vals,
                                rhs);

  CHECK_STATUS(
      perflibs_spmat_create_csr_d(&A, n, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPSV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPSV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_MANY));
  CHECK_STATUS(perflibs_spsv_optimize(A));

  for (perflibs_int_t i = 0; i < n; ++i) {
    x[i] = 0.0;
  }
  CHECK_STATUS(
      perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, x, 1.0, rhs));

  for (perflibs_int_t i = 0; i < n; ++i) {
    CHECK_DOUBLE_NEAR(x[i], 1.0, 1e-12);
  }

  CHECK_STATUS(perflibs_spmat_destroy(A));
  free(row_ptr);
  free(col_indx);
  free(vals);
  free(rhs);
  free(x);
  return EXIT_SUCCESS;
}

static int test_complex_conjtrans_parallel_path() {
  const perflibs_int_t blocks = 4;
  const perflibs_int_t block_size = 128;
  const perflibs_int_t n = blocks * block_size;
  const perflibs_int_t nnz = blocks * (3 * block_size - 3);
  const perflibs_singlecomplex_t alpha = TEST_C(1.0f, 0.0f);
  const perflibs_singlecomplex_t expected_one = TEST_C(1.0f, 0.0f);

  perflibs_int_t *row_ptr =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * (n + 1));
  perflibs_int_t *col_indx =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * nnz);
  perflibs_singlecomplex_t *vals = (perflibs_singlecomplex_t *)malloc(
      sizeof(perflibs_singlecomplex_t) * nnz);
  perflibs_singlecomplex_t *rhs =
      (perflibs_singlecomplex_t *)malloc(sizeof(perflibs_singlecomplex_t) * n);
  perflibs_singlecomplex_t *x =
      (perflibs_singlecomplex_t *)malloc(sizeof(perflibs_singlecomplex_t) * n);
  perflibs_spmat_t A = NULL;

  CHECK_TRUE(row_ptr && col_indx && vals && rhs && x,
             "allocation failure in complex CSR SpSV OpenMP test");

  build_complex_lower_triangular_blocks(blocks, block_size, row_ptr, col_indx,
                                        vals, rhs);

  CHECK_STATUS(
      perflibs_spmat_create_csr_c(&A, n, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPSV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_CONJTRANS));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPSV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_MANY));
  CHECK_STATUS(perflibs_spsv_optimize(A));

  for (perflibs_int_t i = 0; i < n; ++i) {
    x[i] = TEST_C(0.0f, 0.0f);
  }
  CHECK_STATUS(perflibs_spsv_exec_c(PERFLIBS_SPARSE_OPERATION_CONJTRANS, A, x,
                                    alpha, rhs));

  for (perflibs_int_t i = 0; i < n; ++i) {
    CHECK_SINGLECOMPLEX_NEAR(x[i], expected_one, 2e-5f);
  }

  CHECK_STATUS(perflibs_spmat_destroy(A));
  free(row_ptr);
  free(col_indx);
  free(vals);
  free(rhs);
  free(x);
  return EXIT_SUCCESS;
}

static int test_lower_triangular_chain_parallel_path() {
  const perflibs_int_t n = 1024;
  const perflibs_int_t nnz = 2 * n - 1;

  perflibs_int_t *row_ptr =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * (n + 1));
  perflibs_int_t *col_indx =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * nnz);
  double *vals = (double *)malloc(sizeof(double) * nnz);
  double *rhs = (double *)malloc(sizeof(double) * n);
  double *x = (double *)malloc(sizeof(double) * n);
  perflibs_spmat_t A = NULL;

  CHECK_TRUE(row_ptr && col_indx && vals && rhs && x,
             "allocation failure in lower-chain CSR SpSV OpenMP test");

  build_lower_triangular_chain(n, row_ptr, col_indx, vals, rhs);

  CHECK_STATUS(
      perflibs_spmat_create_csr_d(&A, n, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPSV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPSV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_MANY));
  CHECK_STATUS(perflibs_spsv_optimize(A));

  for (perflibs_int_t i = 0; i < n; ++i) {
    x[i] = 0.0;
  }
  CHECK_STATUS(
      perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, x, 1.0, rhs));

  for (perflibs_int_t i = 0; i < n; ++i) {
    CHECK_DOUBLE_NEAR(x[i], 1.0, 1e-12);
  }

  CHECK_STATUS(perflibs_spmat_destroy(A));
  free(row_ptr);
  free(col_indx);
  free(vals);
  free(rhs);
  free(x);
  return EXIT_SUCCESS;
}

static int test_upper_triangular_chain_parallel_path() {
  const perflibs_int_t n = 1024;
  const perflibs_int_t nnz = 2 * n - 1;

  perflibs_int_t *row_ptr =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * (n + 1));
  perflibs_int_t *col_indx =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * nnz);
  double *vals = (double *)malloc(sizeof(double) * nnz);
  double *rhs = (double *)malloc(sizeof(double) * n);
  double *x = (double *)malloc(sizeof(double) * n);
  perflibs_spmat_t A = NULL;

  CHECK_TRUE(row_ptr && col_indx && vals && rhs && x,
             "allocation failure in upper-chain CSR SpSV OpenMP test");

  build_upper_triangular_chain(n, row_ptr, col_indx, vals, rhs);

  CHECK_STATUS(
      perflibs_spmat_create_csr_d(&A, n, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPSV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_NOTRANS));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPSV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_MANY));
  CHECK_STATUS(perflibs_spsv_optimize(A));

  for (perflibs_int_t i = 0; i < n; ++i) {
    x[i] = 0.0;
  }
  CHECK_STATUS(
      perflibs_spsv_exec_d(PERFLIBS_SPARSE_OPERATION_NOTRANS, A, x, 1.0, rhs));

  for (perflibs_int_t i = 0; i < n; ++i) {
    CHECK_DOUBLE_NEAR(x[i], 1.0, 1e-12);
  }

  CHECK_STATUS(perflibs_spmat_destroy(A));
  free(row_ptr);
  free(col_indx);
  free(vals);
  free(rhs);
  free(x);
  return EXIT_SUCCESS;
}

static int test_complex_conjtrans_chain_parallel_path() {
  const perflibs_int_t n = 768;
  const perflibs_int_t nnz = 2 * n - 1;
  const perflibs_singlecomplex_t alpha = TEST_C(1.0f, 0.0f);
  const perflibs_singlecomplex_t expected_one = TEST_C(1.0f, 0.0f);

  perflibs_int_t *row_ptr =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * (n + 1));
  perflibs_int_t *col_indx =
      (perflibs_int_t *)malloc(sizeof(perflibs_int_t) * nnz);
  perflibs_singlecomplex_t *vals = (perflibs_singlecomplex_t *)malloc(
      sizeof(perflibs_singlecomplex_t) * nnz);
  perflibs_singlecomplex_t *rhs =
      (perflibs_singlecomplex_t *)malloc(sizeof(perflibs_singlecomplex_t) * n);
  perflibs_singlecomplex_t *x =
      (perflibs_singlecomplex_t *)malloc(sizeof(perflibs_singlecomplex_t) * n);
  perflibs_spmat_t A = NULL;

  CHECK_TRUE(row_ptr && col_indx && vals && rhs && x,
             "allocation failure in complex lower-chain CSR SpSV OpenMP test");

  build_complex_lower_triangular_chain(n, row_ptr, col_indx, vals, rhs);

  CHECK_STATUS(
      perflibs_spmat_create_csr_c(&A, n, n, row_ptr, col_indx, vals, 0));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_STRUCTURE,
                                   PERFLIBS_SPARSE_STRUCTURE_TRIANGULAR));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPSV_OPERATION,
                                   PERFLIBS_SPARSE_OPERATION_CONJTRANS));
  CHECK_STATUS(perflibs_spmat_hint(A, PERFLIBS_SPARSE_HINT_SPSV_INVOCATIONS,
                                   PERFLIBS_SPARSE_INVOCATIONS_MANY));
  CHECK_STATUS(perflibs_spsv_optimize(A));

  for (perflibs_int_t i = 0; i < n; ++i) {
    x[i] = TEST_C(0.0f, 0.0f);
  }
  CHECK_STATUS(perflibs_spsv_exec_c(PERFLIBS_SPARSE_OPERATION_CONJTRANS, A, x,
                                    alpha, rhs));

  for (perflibs_int_t i = 0; i < n; ++i) {
    CHECK_SINGLECOMPLEX_NEAR(x[i], expected_one, 2e-5f);
  }

  CHECK_STATUS(perflibs_spmat_destroy(A));
  free(row_ptr);
  free(col_indx);
  free(vals);
  free(rhs);
  free(x);
  return EXIT_SUCCESS;
}

int main() {
  if (test_lower_triangular_parallel_path() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_upper_triangular_parallel_path() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_complex_conjtrans_parallel_path() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_lower_triangular_chain_parallel_path() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_upper_triangular_chain_parallel_path() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (test_complex_conjtrans_chain_parallel_path() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
